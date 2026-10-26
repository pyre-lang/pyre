//! Inputs the build-script codegen cache key hashes besides the LLBC artefacts.
//!
//! PRE-EXISTING-ADAPTATION: RPython's `TranslationDriver` runs
//! `task_annotate`, `task_rtype_lltype`, and `task_backendopt_lltype` in one
//! process over its live translation context (`rpython/translator/driver.py`).
//! Pyre's equivalent prepass runs in a Cargo build script over frozen LLBC and
//! persists its generated outputs, so this cache reconstructs the input closure
//! that the live context owns upstream. The still-real blocker is that the
//! build script and runtime are separate programs; the convergence path is to
//! run translation in the final program image and remove the persistent cache.
//!
//! Derived at runtime from the manifests and sources the prepass actually
//! reads, so a new translator workspace dep, a new `#[path]` include, or a
//! new live `pyre_interpreter` function sample joins the key without a
//! hand-maintained list. Interpreter, object, and jit *bodies* still reach
//! the prepass only through `target/llbc/*.ullbc`; `fail_if_llbc_stale` is
//! the source-vs-artefact gate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Crates whose Charon artefact the prepass consumes. The `.ullbc` and its
/// `.fingerprint` stamp are both named after the crate. Pyre production
/// configures the exact `eval::eval_loop_jit` portal, so unlike generic
/// two-artefact consumers it requires `pyre-jit` too.
///
/// Their sources are not hashed into the codegen cache key: a source-only
/// edit is either refused by `fail_if_llbc_stale` or already rekeys via the
/// artefact bytes. `pyre-interpreter` and `pyre-module` are also build-deps,
/// but only so the script can sample the live fnaddr / static tables
/// (including wrappers that moved into `pyre-module`); those functions are
/// added as single files through [`interpreter_fn_calls`], not as a tree.
///
/// Order is the production hash/load order. Changing it rekeys the cache.
#[cfg(feature = "pyre-module")]
pub const LLBC_CRATES: &[&str] = &[
    "majit-rlib",
    "pyre-object",
    "pyre-interpreter",
    "pyre-module",
    "pyre-jit",
];

/// The core translation: `pyre-module` is not part of it, the way a PyPy
/// build without `objspace.usemodules.<name>` never translates that module.
/// Its builtins stay opaque residual calls.
#[cfg(not(feature = "pyre-module"))]
pub const LLBC_CRATES: &[&str] = &["majit-rlib", "pyre-object", "pyre-interpreter", "pyre-jit"];

/// Directory under `target/` holding the artefacts `LLBC_CRATES` names. The
/// core reads the product set's artefacts for its crates: none of them
/// depends on `pyre-module`.
#[allow(dead_code)] // target/llbc artefact directory
pub const LLBC_SUBDIR: &str = "llbc";

/// Build-dependencies the manifest declares but this translation never links:
/// `pyre-module` is optional and linked only under its own feature, so the core cache
/// key must not hash a tree its prepass never reads.
#[cfg(feature = "pyre-module")]
const UNLINKED_BUILD_DEPS: &[&str] = &[];
#[cfg(not(feature = "pyre-module"))]
const UNLINKED_BUILD_DEPS: &[&str] = &["pyre-module"];

/// Repo-relative trees and files the cache key should hash, plus every
/// manifest read while discovering them (for `cargo::rerun-if-changed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInputs {
    pub src_trees: Vec<String>,
    pub extra_files: Vec<String>,
    pub manifests: Vec<String>,
}

/// Walk the build-script's workspace closure, `#[path]` includes, and
/// live `pyre_interpreter::` samples.
pub fn discover(repo_root: &Path, manifest_dir: &Path) -> CacheInputs {
    let mut manifests = BTreeSet::new();
    let workspace_toml = repo_root.join("Cargo.toml");
    manifests.insert(rel(repo_root, &workspace_toml));
    let members = workspace_member_paths(&read_to_string(&workspace_toml));

    let crate_toml = manifest_dir.join("Cargo.toml");
    manifests.insert(rel(repo_root, &crate_toml));
    let mut trees = BTreeSet::new();
    let mut pending: Vec<String> =
        section_workspace_deps(&read_to_string(&crate_toml), "build-dependencies")
            .into_iter()
            .filter(|name| {
                !LLBC_CRATES.contains(&name.as_str())
                    && !UNLINKED_BUILD_DEPS.contains(&name.as_str())
            })
            .collect();

    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(member) = members.get(&name) else {
            continue;
        };
        let crate_dir = repo_root.join(member);
        let src = crate_dir.join("src");
        if src.is_dir() {
            trees.insert(rel(repo_root, &src));
        }
        let dep_toml = crate_dir.join("Cargo.toml");
        if dep_toml.is_file() {
            manifests.insert(rel(repo_root, &dep_toml));
            for dep in section_workspace_deps(&read_to_string(&dep_toml), "dependencies") {
                if !LLBC_CRATES.contains(&dep.as_str()) {
                    pending.push(dep);
                }
            }
        }
    }

    let mut extra = BTreeSet::new();
    let mut script_sources = Vec::new();
    collect_rs_files(&manifest_dir.join("build"), &mut script_sources);
    let build_rs = manifest_dir.join("build.rs");
    if build_rs.is_file() {
        script_sources.push(build_rs.clone());
    }
    // Live samples are in the build script itself. Path-included modules are
    // data, and this file's tests contain example call spellings that must
    // not pull every `fn foo`.
    let mut sampled_names = BTreeSet::new();
    for path in [&build_rs, &manifest_dir.join("build/prepass.rs")] {
        sampled_names.extend(interpreter_fn_calls(&read_to_string(path)));
    }
    for name in sampled_names {
        for found in find_fn_files(&repo_root.join("pyre/pyre-interpreter/src"), &name) {
            extra.insert(rel(repo_root, &found));
        }
    }
    let mut i = 0;
    while i < script_sources.len() {
        let path = script_sources[i].clone();
        i += 1;
        for include in path_attrs(&read_to_string(&path)) {
            let resolved = canonicalize(&path.parent().unwrap_or(manifest_dir).join(include));
            if resolved.is_file() && !script_sources.iter().any(|have| have == &resolved) {
                script_sources.push(resolved);
            }
        }
    }
    let build_dir = canonicalize(&manifest_dir.join("build"));
    let build_rs = canonicalize(&build_rs);
    for path in &script_sources {
        let path = canonicalize(path);
        if path.starts_with(&build_dir) || path == build_rs {
            continue;
        }
        extra.insert(rel(repo_root, &path));
    }

    CacheInputs {
        src_trees: trees.into_iter().collect(),
        extra_files: extra.into_iter().collect(),
        manifests: manifests.into_iter().collect(),
    }
}

/// Repo-relative spelling of an input path for checkout-independent keys.
///
/// Cargo supplies `CARGO_MANIFEST_DIR` without `..`, while the prepass builds
/// its repo and source paths by appending different `..` suffixes. `Path`'s
/// `strip_prefix` is lexical, so canonicalize both existing paths before the
/// comparison. Inputs outside the repository deliberately retain their full
/// path: two arbitrary external LLBC locations are not interchangeable.
pub fn repo_relative(repo_root: &Path, path: &Path) -> String {
    let canonical_root = std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.into());
    let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.into());
    // Forward slashes: Windows `Path::to_string_lossy` emits `\`, and a
    // backslash key would miss a Unix-built cache entry of the same tree.
    canonical_path
        .strip_prefix(&canonical_root)
        .unwrap_or(&canonical_path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn rel(repo_root: &Path, path: &Path) -> String {
    repo_relative(repo_root, path)
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

fn canonicalize(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// `name → repo-relative crate dir` for every `[workspace.dependencies]`
/// entry that names a path member.
pub fn workspace_member_paths(toml: &str) -> BTreeMap<String, String> {
    let Some(body) = table_body(toml, "workspace.dependencies") else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    for line in body.lines() {
        let Some((name, rhs)) = dep_line(line) else {
            continue;
        };
        if let Some(path) = quoted_value(rhs, "path") {
            out.insert(name.to_string(), path);
        }
    }
    out
}

/// Workspace-member dependency names declared in `[table]` (`dependencies`
/// or `build-dependencies`). Crates.io / git deps are ignored.
pub fn section_workspace_deps(toml: &str, table: &str) -> Vec<String> {
    let Some(body) = table_body(toml, table) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in body.lines() {
        let Some((name, rhs)) = dep_line(line) else {
            continue;
        };
        if rhs.contains("workspace") || quoted_value(rhs, "path").is_some() {
            out.push(name.to_string());
        }
    }
    out
}

/// `#[path = "relative"]` arguments in the order they appear.
pub fn path_attrs(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find("#[path") {
        rest = &rest[start + 6..];
        let after = rest.trim_start();
        let Some(after) = after.strip_prefix('=') else {
            continue;
        };
        let after = after.trim_start();
        let Some(after) = after.strip_prefix('"') else {
            continue;
        };
        let Some(end) = after.find('"') else {
            continue;
        };
        out.push(after[..end].to_string());
    }
    out
}

/// Function names in `pyre_interpreter::name(` call position.
///
/// A type path (`pyre_interpreter::error::PyError`) is not a call: the
/// first segment is followed by `::`, not `(`.
pub fn interpreter_fn_calls(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = source;
    const PREFIX: &str = "pyre_interpreter::";
    while let Some(start) = rest.find(PREFIX) {
        rest = &rest[start + PREFIX.len()..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after = rest[name.len()..].trim_start();
        if after.starts_with('(') {
            names.insert(name);
        }
    }
    names
}

fn find_fn_files(src_root: &Path, name: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![src_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            if file_defines_fn(&read_to_string(&path), name) {
                files.push(path);
            }
        }
    }
    files
}

fn file_defines_fn(source: &str, name: &str) -> bool {
    let mut rest = source;
    while let Some(idx) = rest.find("fn") {
        if idx > 0 {
            let prev = rest.as_bytes()[idx - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                rest = &rest[idx + 2..];
                continue;
            }
        }
        let after = rest[idx + 2..].trim_start();
        if let Some(after_name) = after.strip_prefix(name) {
            let next = after_name.as_bytes().first().copied();
            if !next.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return true;
            }
        }
        rest = &rest[idx + 2..];
    }
    false
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn table_body<'a>(toml: &'a str, header: &str) -> Option<&'a str> {
    let tag = format!("[{header}]");
    let start = toml.find(&tag)? + tag.len();
    let rest = &toml[start..];
    let end = rest.find("\n[").unwrap_or(rest.len());
    Some(&rest[..end])
}

fn dep_line(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let eq = line.find('=')?;
    let name = line[..eq].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }
    Some((name, line[eq + 1..].trim()))
}

fn quoted_value(s: &str, key: &str) -> Option<String> {
    let mut rest = s;
    while let Some(idx) = rest.find(key) {
        let after_key = rest[idx + key.len()..].trim_start();
        if !after_key.starts_with('=') {
            rest = &rest[idx + key.len()..];
            continue;
        }
        let value = after_key[1..].trim_start();
        let value = value.strip_prefix('"')?;
        let end = value.find('"')?;
        return Some(value[..end].to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crate_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn repo_root() -> PathBuf {
        crate_root().join("../..")
    }

    #[test]
    fn workspace_deps_include_translate() {
        let toml = read_to_string(&repo_root().join("Cargo.toml"));
        let deps = workspace_member_paths(&toml);
        assert_eq!(
            deps.get("majit-translate").map(String::as_str),
            Some("majit/majit-translate")
        );
        assert_eq!(
            deps.get("majit-charon-reader").map(String::as_str),
            Some("majit/majit-charon-reader")
        );
    }

    #[test]
    fn discover_tracks_translate_closure_and_live_bindings() {
        let inputs = discover(&repo_root(), &crate_root());
        for tree in [
            "majit/majit-charon-reader/src",
            "majit/majit-ir/src",
            "majit/majit-translate/src",
        ] {
            assert!(
                inputs.src_trees.iter().any(|have| have == tree),
                "missing tree {tree} in {:?}",
                inputs.src_trees
            );
        }
        for file in [
            "pyre/pyre-interpreter/src/jit_fnaddr.rs",
            "pyre/pyre-jit-trace/src/call_spec.rs",
            "pyre/pyre-jit-trace/src/codegen_cache.rs",
            "pyre/pyre-jit-trace/src/pypyjit_driver_layout.rs",
            "pyre/pyre-jit-trace/src/virtualizable_spec.rs",
        ] {
            assert!(
                inputs.extra_files.iter().any(|have| have == file),
                "missing file {file} in {:?}",
                inputs.extra_files
            );
        }
        assert!(
            inputs
                .manifests
                .iter()
                .any(|have| have == "majit/majit-translate/Cargo.toml"),
            "missing translate manifest in {:?}",
            inputs.manifests
        );
    }

    #[test]
    fn discover_excludes_every_llbc_crate_source_tree() {
        let members = workspace_member_paths(&read_to_string(&repo_root().join("Cargo.toml")));
        let trees: BTreeSet<_> = discover(&repo_root(), &crate_root())
            .src_trees
            .into_iter()
            .collect();
        for name in LLBC_CRATES {
            let member = members
                .get(*name)
                .unwrap_or_else(|| panic!("{name} missing from workspace.dependencies"));
            let tree = format!("{member}/src");
            assert!(
                !trees.contains(&tree),
                "{tree} is an LLBC artefact input; hashing its sources re-runs the prepass on an unchanged artefact"
            );
        }
    }

    #[test]
    fn discover_does_not_walk_unrelated_workspace_crates() {
        let trees: BTreeSet<_> = discover(&repo_root(), &crate_root())
            .src_trees
            .into_iter()
            .collect();
        for rel in [
            "majit/majit-backend-cranelift/src",
            "majit/majit-backend-dynasm/src",
            "majit/majit-backend-wasm/src",
            "majit/majit-gc/src",
            "majit/majit-metainterp/src",
            "majit/majit-rlib/src",
            "pyre/pyre-interpreter/src",
            "pyre/pyre-jit/src",
            "pyre/pyre-module/src",
            "pyre/pyre-native/src",
            "pyre/pyre-object/src",
            "pyre/pyre-sandbox/src",
            "pyre/pyre-wasm/src",
        ] {
            assert!(
                !trees.contains(rel),
                "{rel} is not a prepass input; hashing it re-runs the prepass on an unrelated edit"
            );
        }
    }

    #[test]
    fn path_attrs_resolve_quoted_paths() {
        let src = "#[path = \"../src/call_spec.rs\"]\nmod call_spec;\n";
        assert_eq!(path_attrs(src), vec!["../src/call_spec.rs".to_string()]);
    }

    #[test]
    fn interpreter_calls_ignore_type_paths() {
        let src = r#"
            let x = pyre_interpreter::jit_trace_fnaddrs();
            let p = "pyre_interpreter::error::PyError";
        "#;
        assert_eq!(
            interpreter_fn_calls(src),
            BTreeSet::from(["jit_trace_fnaddrs".to_string()])
        );
    }

    #[test]
    fn file_defines_fn_matches_exact_name() {
        assert!(file_defines_fn(
            "pub fn jit_trace_fnaddrs() -> i32 { 0 }",
            "jit_trace_fnaddrs"
        ));
        assert!(!file_defines_fn(
            "fn jit_trace_fnaddrs_contains_root() {}",
            "jit_trace_fnaddrs"
        ));
    }

    #[test]
    fn repo_relative_normalizes_the_prepass_dotdot_spellings() {
        let root = repo_root();
        let source = crate_root().join("../pyre-object/src");
        assert_eq!(repo_relative(&root, &source), "pyre/pyre-object/src");
        assert_eq!(
            repo_relative(&root, &crate_root().join("build.rs")),
            "pyre/pyre-jit-trace/build.rs"
        );
    }

    #[test]
    fn discover_paths_use_forward_slashes() {
        let inputs = discover(&repo_root(), &crate_root());
        for path in inputs
            .src_trees
            .iter()
            .chain(&inputs.extra_files)
            .chain(&inputs.manifests)
        {
            assert!(
                !path.contains('\\'),
                "cache key path must be host-independent: {path}"
            );
        }
    }

    #[test]
    fn prepass_does_not_walk_the_whole_workspace() {
        let src = read_to_string(&crate_root().join("build/prepass.rs"));
        assert!(
            !src.contains(r#"for workspace_dir in ["majit", "pyre"]"#),
            "codegen_cache_key must not walk every workspace crate"
        );
    }
}
