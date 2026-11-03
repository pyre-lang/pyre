//! Crate `build.rs` entry that compiles every ECI this crate declares.

use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Item, Lit, Meta, Token, UseTree};

use crate::ExternalCompilationInfo;
use crate::parse::{EciInput, LlexternalHead};

/// genc always has `rpython/translator/c` (`cdir`) on the include path, so C
/// files write `#include "src/precommondefs.h"`. Writes that header under
/// `$OUT_DIR/eci/cdir` and returns that directory.
fn cdir() -> PathBuf {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let dir = out_dir.join("eci").join("cdir");
    let src = dir.join("src");
    if let Err(err) = std::fs::create_dir_all(&src) {
        panic!("{}: {err}", src.display());
    }
    let header = src.join("precommondefs.h");
    if let Err(err) = std::fs::write(&header, include_str!("precommondefs.h")) {
        panic!("{}: {err}", header.display());
    }
    dir
}

/// ll2ctypes `compile_shared_lib` for one ECI: `convert_sources_to_files` then
/// `cc::Build`. Does not print link directives except what `cc` prints on
/// success. Link directives (`library_dirs`, `libraries`, `link_files`, the C++
/// runtime for `use_cpp_linker`) come from `link_eci` or `compile_crate_ecis`;
/// a direct caller pairs this with `link_eci`.
pub fn compile_eci(eci: &ExternalCompilationInfo, libname: &str) -> Result<(), cc::Error> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cache = out_dir.join("eci").join(libname);
    if let Err(err) = std::fs::remove_dir_all(&cache) {
        if err.kind() != std::io::ErrorKind::NotFound {
            panic!("{}: {err}", cache.display());
        }
    }
    if let Err(err) = std::fs::create_dir_all(&cache) {
        panic!("{}: {err}", cache.display());
    }
    let e = eci.convert_sources_to_files(&cache);
    let mut build = cc::Build::new();
    for file in &e.separate_module_files {
        let path = resolve_against(&manifest, file);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(&path);
    }
    for dir in &e.include_dirs {
        let path = resolve_against(&manifest, dir);
        // genc's Makefile makes each object depend on the headers it includes;
        // Cargo reruns this script when a file under a package include dir changes.
        if path.starts_with(&manifest) && path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        build.include(&path);
    }
    build.include(cdir());
    for flag in &e.compile_extra {
        build.flag(flag);
    }
    build.warnings(false);
    build.try_compile(libname)?;
    Ok(())
}

/// genc link step for an ECI no macro declared: `library_dirs`, `libraries`,
/// `frameworks`, `link_extra`, `link_files`.
pub fn link_eci(eci: &ExternalCompilationInfo) {
    print_eci_link_directives(eci);
    for lib in &eci.libraries {
        println!("cargo:rustc-link-lib={lib}");
    }
    for framework in &eci.frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn print_eci_link_directives(eci: &ExternalCompilationInfo) {
    for dir in &eci.library_dirs {
        println!("cargo:rustc-link-search=native={dir}");
    }
    for arg in &eci.link_extra {
        // Cargo applies `rustc-link-arg` to this package's own link only; a
        // dependent re-emits it from `links` metadata (`_minimal_curses`).
        println!("cargo:rustc-link-arg={arg}");
    }
    for file in &eci.link_files {
        let path = Path::new(file);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let file_name_lower = path
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        // BasePosix._linkfiles hands each file to the linker as is. Cargo links only libraries:
        // archives and shared libraries go in verbatim; anything else is an object, archived
        // first and linked whole.
        let kind = match ext.as_str() {
            "a" | "lib" => "static:+verbatim",
            "so" | "dylib" | "tbd" => "dylib:+verbatim",
            _ if file_name_lower.contains(".so.") => "dylib:+verbatim",
            _ => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("obj");
                let sanitized: String = stem
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let name = format!(
                    "eci_linkfile_{:016x}_{}",
                    fnv1a64(file.as_bytes()),
                    sanitized
                );
                cc::Build::new()
                    .object(path)
                    .cargo_metadata(false)
                    .cargo_warnings(false)
                    .try_compile(&name)
                    .unwrap_or_else(|err| panic!("{file}: {err}"));
                let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
                println!("cargo:rustc-link-search=native={out_dir}");
                // +whole-archive because a direct object joins the link even when
                // nothing references it (a constructor or a registration record),
                // while an ordinary archive member is only pulled on demand.
                println!("cargo:rustc-link-lib=static:+whole-archive={name}");
                continue;
            }
        };
        if let Some(parent) = path.parent() {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
        if let Some(name) = path.file_name() {
            println!("cargo:rustc-link-lib={kind}={}", name.to_string_lossy());
        }
    }
    if eci.use_cpp_linker {
        // genc links with `g++` when `use_cpp_linker` (`posix.py` `CC_LINK`); rustc
        // drives the link here, so add the C++ runtime `g++` links.
        let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if env != "msvc" {
            let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
            let runtime = if matches!(os.as_str(), "macos" | "ios" | "freebsd" | "openbsd") {
                "c++"
            } else {
                "stdc++"
            };
            println!("cargo:rustc-link-lib={runtime}");
        }
    }
}

/// ll2ctypes' C step for every ECI this crate declares: `compile_shared_lib`
/// per ECI (after `convert_sources_to_files`), plus the link directives genc
/// hands the linker. `extra` is merged into every ECI first; it carries what
/// upstream computes from the platform at import time (e.g.
/// `platform.include_dirs_for_libffi()`). The proc macro emits `#[link]` only
/// for what the declaration names; `extra` is computed by the build script.
/// This prints `extra`'s `libraries`/`frameworks`, then per ECI
/// `library_dirs`/`link_extra`/`link_files`.
pub fn compile_crate_ecis(extra: &ExternalCompilationInfo) {
    println!("cargo:rustc-check-cfg=cfg(majit_eci_build)");
    println!("cargo:rustc-cfg=majit_eci_build");
    for lib in &extra.libraries {
        println!("cargo:rustc-link-lib={lib}");
    }
    for framework in &extra.frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let lib = manifest.join("src/lib.rs");
    let mut collected = CrateItems::default();
    collected.modules.push(Vec::new());
    let submodule_dir = submodule_dir_for_file(&lib);
    walk_file(&lib, &submodule_dir, &[], &mut collected);
    merge_macro_wrappers(&mut collected);

    for item in collected.ecis {
        let e = item.eci.merge(&[extra]);
        print_eci_link_directives(&e);
        if e.separate_module_sources.is_empty() && e.separate_module_files.is_empty() {
            continue;
        }
        let libname = eci_libname(&item.module_path, &item.name);
        compile_eci(&e, &libname).unwrap_or_else(|err| panic!("{libname}: {err}"));
    }
}

struct CollectedEci {
    module_path: Vec<String>,
    name: String,
    eci: ExternalCompilationInfo,
}

struct CollectedAlias {
    module_path: Vec<String>,
    name: String,
    ty: syn::Type,
}

struct CollectedLlexternal {
    file_path: PathBuf,
    module_path: Vec<String>,
    head: LlexternalHead,
}

struct CollectedUse {
    module_path: Vec<String>,
    binding: Option<String>,
    target: Vec<String>,
    leading_colon: bool,
}

enum Target {
    Local(Vec<String>, String),
    Extern(Vec<String>),
}

enum ModuleTarget {
    Local(Vec<String>),
    Extern(Vec<String>),
}

#[derive(Default)]
struct CrateItems {
    ecis: Vec<CollectedEci>,
    aliases: Vec<CollectedAlias>,
    llexternals: Vec<CollectedLlexternal>,
    uses: Vec<CollectedUse>,
    modules: Vec<Vec<String>>,
    structs: Vec<(Vec<String>, String)>,
}

/// `rffi.py generate_macro_wrapper`: merge each `macro` llexternal's C
/// wrapper into its ECI's `separate_module_sources` (`eci.merge` with an
/// ECI that holds only that source).
fn merge_macro_wrappers(collected: &mut CrateItems) {
    let llexternals = std::mem::take(&mut collected.llexternals);
    for item in &llexternals {
        let Some(macro_expr) = &item.head.macro_ else {
            continue;
        };
        let macro_name = match macro_c_name(macro_expr, &item.head.c_name) {
            Some(name) => name,
            None => panic!(
                "{}: llexternal `{}`: `macro` is `true` or the C macro's name (`rffi.py llexternal`)",
                item.file_path.display(),
                item.head.name
            ),
        };
        let source = {
            let resolve =
                |path: &syn::Path, module_path: &[String]| -> Option<crate::cdecl::Resolved> {
                    match resolve_path(collected, path, module_path) {
                        Some(Target::Local(m, n)) => {
                            if let Some(alias) = collected
                                .aliases
                                .iter()
                                .find(|a| a.module_path == m && a.name == n)
                            {
                                Some(crate::cdecl::Resolved::Alias(alias.ty.clone(), m))
                            } else if collected
                                .structs
                                .iter()
                                .any(|(tm, tn)| *tm == m && *tn == n)
                            {
                                Some(crate::cdecl::Resolved::LocalStruct(n))
                            } else {
                                None
                            }
                        }
                        Some(Target::Extern(p)) => Some(crate::cdecl::Resolved::Extern(p)),
                        None => None,
                    }
                };
            let args: Vec<String> = item
                .head
                .args
                .iter()
                .map(|ty| {
                    crate::cdecl::functype_arg(ty, &item.module_path, &resolve).unwrap_or_else(
                        |msg| {
                            panic!(
                                "{}: llexternal `{}`: {msg}",
                                item.file_path.display(),
                                item.head.name
                            )
                        },
                    )
                })
                .collect();
            let result =
                crate::cdecl::functype_result(&item.head.result, &item.module_path, &resolve)
                    .unwrap_or_else(|msg| {
                        panic!(
                            "{}: llexternal `{}`: {msg}",
                            item.file_path.display(),
                            item.head.name
                        )
                    });
            crate::generate_macro_wrapper(&item.head.c_name, &macro_name, &args, &result)
        };
        let wrapper = ExternalCompilationInfo {
            separate_module_sources: vec![source],
            ..ExternalCompilationInfo::default()
        };
        let eci_idx = match &item.head.compilation_info {
            Some(expr) => {
                let Some(path) = expr_as_path(expr) else {
                    panic!(
                        "{}: llexternal `{}`: compilation_info is not a path",
                        item.file_path.display(),
                        item.head.name
                    );
                };
                match resolve_path(collected, path, &item.module_path) {
                    Some(Target::Local(m, n)) => {
                        match collected
                            .ecis
                            .iter()
                            .position(|eci| eci.module_path == m && eci.name == n)
                        {
                            Some(idx) => idx,
                            None => {
                                let eci_name = path
                                    .segments
                                    .last()
                                    .map(|seg| seg.ident.to_string())
                                    .unwrap_or_default();
                                panic!(
                                    "{}: llexternal `{}`: compilation_info `{eci_name}` not found",
                                    item.file_path.display(),
                                    item.head.name
                                )
                            }
                        }
                    }
                    Some(Target::Extern(_)) => {
                        panic!(
                            "{}: llexternal `{}`: compilation_info `{}` is outside this crate",
                            item.file_path.display(),
                            item.head.name,
                            path_text(path)
                        )
                    }
                    None => {
                        let eci_name = path
                            .segments
                            .last()
                            .map(|seg| seg.ident.to_string())
                            .unwrap_or_default();
                        panic!(
                            "{}: llexternal `{}`: compilation_info `{eci_name}` not found",
                            item.file_path.display(),
                            item.head.name
                        )
                    }
                }
            }
            None => {
                collected.ecis.push(CollectedEci {
                    module_path: item.module_path.clone(),
                    name: format!("_macro_{}", item.head.c_name),
                    eci: ExternalCompilationInfo::default(),
                });
                collected.ecis.len() - 1
            }
        };
        let merged = collected.ecis[eci_idx].eci.merge(&[&wrapper]);
        collected.ecis[eci_idx].eci = merged;
    }
}

fn macro_c_name(expr: &Expr, c_name: &str) -> Option<String> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Bool(value) if value.value() => Some(c_name.to_string()),
            Lit::Str(value) => Some(value.value()),
            _ => None,
        },
        _ => None,
    }
}

fn expr_as_path(expr: &Expr) -> Option<&syn::Path> {
    match expr {
        Expr::Path(path) => Some(&path.path),
        _ => None,
    }
}

fn path_text(path: &syn::Path) -> String {
    let mut s = String::new();
    if path.leading_colon.is_some() {
        s.push_str("::");
    }
    for (i, seg) in path.segments.iter().enumerate() {
        if i > 0 {
            s.push_str("::");
        }
        s.push_str(&unraw_ident(&seg.ident));
    }
    s
}

fn is_module(collected: &CrateItems, path: &[String]) -> bool {
    collected.modules.iter().any(|m| m.as_slice() == path)
}

fn defines_name(collected: &CrateItems, module: &[String], name: &str) -> bool {
    collected
        .ecis
        .iter()
        .any(|e| e.module_path.as_slice() == module && e.name == name)
        || collected
            .aliases
            .iter()
            .any(|a| a.module_path.as_slice() == module && a.name == name)
        || collected
            .structs
            .iter()
            .any(|(m, n)| m.as_slice() == module && n == name)
}

fn find_binding_use(
    collected: &CrateItems,
    module: &[String],
    name: &str,
    active: &[usize],
) -> Option<usize> {
    collected.uses.iter().enumerate().position(|(i, u)| {
        !active.contains(&i)
            && u.module_path.as_slice() == module
            && u.binding.as_deref() == Some(name)
    })
}

fn with_active_use<R>(
    active: &mut Vec<usize>,
    use_idx: usize,
    f: impl FnOnce(&mut Vec<usize>) -> Option<R>,
) -> Option<R> {
    if active.contains(&use_idx) {
        return None;
    }
    active.push(use_idx);
    let result = f(active);
    active.pop();
    result
}

fn resolve_path(collected: &CrateItems, path: &syn::Path, module: &[String]) -> Option<Target> {
    let segs: Vec<String> = path
        .segments
        .iter()
        .map(|s| unraw_ident(&s.ident))
        .collect();
    let mut active = Vec::new();
    resolve_segs(
        collected,
        &segs,
        path.leading_colon.is_some(),
        module,
        &mut active,
    )
}

fn resolve_use_target(
    collected: &CrateItems,
    use_idx: usize,
    active: &mut Vec<usize>,
) -> Option<Target> {
    with_active_use(active, use_idx, |active| {
        let u = &collected.uses[use_idx];
        let segs = u.target.clone();
        let leading = u.leading_colon;
        let module = u.module_path.clone();
        resolve_segs(collected, &segs, leading, &module, active)
    })
}

fn resolve_segs(
    collected: &CrateItems,
    segs: &[String],
    leading_colon: bool,
    module: &[String],
    active: &mut Vec<usize>,
) -> Option<Target> {
    if segs.is_empty() {
        return None;
    }
    if leading_colon {
        return Some(Target::Extern(segs.to_vec()));
    }
    if segs.len() == 1 && !matches!(segs[0].as_str(), "crate" | "self" | "super") {
        return lookup_name(collected, module, &segs[0], active);
    }
    let last = &segs[segs.len() - 1];
    match resolve_module(
        collected,
        &segs[..segs.len() - 1],
        leading_colon,
        module,
        active,
    ) {
        Some(ModuleTarget::Local(m)) => lookup_name(collected, &m, last, active),
        Some(ModuleTarget::Extern(mut p)) => {
            p.push(last.clone());
            Some(Target::Extern(p))
        }
        None => None,
    }
}

/// `cur + [seg]` through the glob `use`s of `cur`: the first local glob
/// target `g` with `g + [seg]` a module gives `Local(g + [seg])`; failing
/// that, the first extern glob target `p` gives `Extern(p + [seg])`.
/// A local glob target is searched like a qualified path: its child modules,
/// its `use` bindings, then its own globs; `visited` keeps each module to one
/// search.
fn glob_child_module(
    collected: &CrateItems,
    cur: &[String],
    seg: &str,
    active: &mut Vec<usize>,
    visited: &mut Vec<Vec<String>>,
) -> Option<ModuleTarget> {
    // An extern crate's exports are not visible to the scanner, so a local glob
    // that defines the name wins over an extern glob.
    if visited.iter().any(|m| m.as_slice() == cur) {
        return None;
    }
    visited.push(cur.to_vec());
    let mut extern_glob = None;
    for (i, u) in collected.uses.iter().enumerate() {
        if u.binding.is_some() || u.module_path.as_slice() != cur {
            continue;
        }
        match with_active_use(active, i, |active| {
            resolve_module(
                collected,
                &u.target,
                u.leading_colon,
                &u.module_path,
                active,
            )
        }) {
            Some(ModuleTarget::Local(g)) => {
                let mut nested = g.clone();
                nested.push(seg.to_string());
                if is_module(collected, &nested) {
                    return Some(ModuleTarget::Local(nested));
                }
                if let Some(idx) = find_binding_use(collected, &g, seg, active) {
                    match binding_module(collected, &g, idx, active) {
                        Some(ModuleTarget::Local(m)) => {
                            return Some(ModuleTarget::Local(m));
                        }
                        Some(ModuleTarget::Extern(p)) => {
                            if extern_glob.is_none() {
                                extern_glob = Some(p);
                            }
                        }
                        None => {}
                    }
                } else if let Some(found) = glob_child_module(collected, &g, seg, active, visited) {
                    match found {
                        ModuleTarget::Local(m) => return Some(ModuleTarget::Local(m)),
                        ModuleTarget::Extern(p) => {
                            if extern_glob.is_none() {
                                extern_glob = Some(p);
                            }
                        }
                    }
                }
            }
            Some(ModuleTarget::Extern(mut p)) => {
                if extern_glob.is_none() {
                    p.push(seg.to_string());
                    extern_glob = Some(p);
                }
            }
            None => {}
        }
    }
    extern_glob.map(ModuleTarget::Extern)
}

fn binding_module(
    collected: &CrateItems,
    module: &[String],
    idx: usize,
    active: &mut Vec<usize>,
) -> Option<ModuleTarget> {
    with_active_use(active, idx, |active| {
        let u = &collected.uses[idx];
        let target = u.target.clone();
        let leading = u.leading_colon;
        resolve_module(collected, &target, leading, module, active)
    })
}

fn walk_module_seg(
    collected: &CrateItems,
    cur: &[String],
    seg: &str,
    after: &[String],
    active: &mut Vec<usize>,
) -> Option<ModuleTarget> {
    let mut child = cur.to_vec();
    child.push(seg.to_string());
    if is_module(collected, &child) {
        return Some(ModuleTarget::Local(child));
    }
    if let Some(idx) = find_binding_use(collected, cur, seg, active) {
        return match binding_module(collected, cur, idx, active) {
            Some(ModuleTarget::Extern(mut p)) => {
                p.extend(after.iter().cloned());
                Some(ModuleTarget::Extern(p))
            }
            other => other,
        };
    }
    let mut visited = Vec::new();
    match glob_child_module(collected, cur, seg, active, &mut visited) {
        Some(ModuleTarget::Extern(mut p)) => {
            p.extend(after.iter().cloned());
            Some(ModuleTarget::Extern(p))
        }
        other => other,
    }
}

fn resolve_module(
    collected: &CrateItems,
    segs: &[String],
    leading_colon: bool,
    module: &[String],
    active: &mut Vec<usize>,
) -> Option<ModuleTarget> {
    if leading_colon {
        return Some(ModuleTarget::Extern(segs.to_vec()));
    }
    if segs.is_empty() {
        return Some(ModuleTarget::Local(module.to_vec()));
    }
    let (cur, rest): (Vec<String>, &[String]) = match segs[0].as_str() {
        "crate" => (Vec::new(), &segs[1..]),
        "self" => (module.to_vec(), &segs[1..]),
        "super" => {
            let mut cur = module.to_vec();
            let mut i = 0;
            while i < segs.len() && segs[i] == "super" {
                cur.pop();
                i += 1;
            }
            return walk_module_rest(collected, cur, &segs[i..], active);
        }
        _ => {
            let x = &segs[0];
            let mut child = module.to_vec();
            child.push(x.clone());
            if is_module(collected, &child) {
                (child, &segs[1..])
            } else if let Some(idx) = find_binding_use(collected, module, x, active) {
                return with_active_use(active, idx, |active| {
                    let mut spliced = collected.uses[idx].target.clone();
                    spliced.extend(segs[1..].iter().cloned());
                    let leading = collected.uses[idx].leading_colon;
                    resolve_module(collected, &spliced, leading, module, active)
                });
            } else {
                let mut visited = Vec::new();
                if let Some(ModuleTarget::Local(m)) =
                    glob_child_module(collected, module, x, active, &mut visited)
                {
                    (m, &segs[1..])
                } else {
                    // A first segment not found locally names an extern crate; an
                    // extern glob must not capture it (`use libc::*;` must not turn
                    // `std::ptr` into `libc::std::ptr`).
                    return Some(ModuleTarget::Extern(segs.to_vec()));
                }
            }
        }
    };
    walk_module_rest(collected, cur, rest, active)
}

fn walk_module_rest(
    collected: &CrateItems,
    mut cur: Vec<String>,
    rest: &[String],
    active: &mut Vec<usize>,
) -> Option<ModuleTarget> {
    for i in 0..rest.len() {
        let after = &rest[i + 1..];
        match walk_module_seg(collected, &cur, &rest[i], after, active)? {
            ModuleTarget::Local(next) => cur = next,
            ModuleTarget::Extern(p) => return Some(ModuleTarget::Extern(p)),
        }
    }
    Some(ModuleTarget::Local(cur))
}

fn lookup_name(
    collected: &CrateItems,
    module: &[String],
    name: &str,
    active: &mut Vec<usize>,
) -> Option<Target> {
    let mut visited = Vec::new();
    let mut extern_glob = None;
    lookup_name_in(
        collected,
        module,
        name,
        active,
        &mut visited,
        &mut extern_glob,
    )
    .or_else(|| {
        // An extern crate's exports are not visible to the scanner, so a local glob
        // that defines the name wins over an extern glob.
        extern_glob.map(|mut p| {
            p.push(name.to_string());
            Target::Extern(p)
        })
    })
}

/// Definite resolutions only: a definition in `module`, a `use` binding, or a
/// local glob target that resolves the name (recursively). The first extern glob
/// target met anywhere in the walk is recorded in `extern_glob`; it is consulted
/// by `lookup_name` only after the whole local search fails. `visited` holds the
/// modules already searched for this name, so each module is searched once.
fn lookup_name_in(
    collected: &CrateItems,
    module: &[String],
    name: &str,
    active: &mut Vec<usize>,
    visited: &mut Vec<Vec<String>>,
    extern_glob: &mut Option<Vec<String>>,
) -> Option<Target> {
    if visited.iter().any(|m| m.as_slice() == module) {
        return None;
    }
    visited.push(module.to_vec());
    if defines_name(collected, module, name) {
        return Some(Target::Local(module.to_vec(), name.to_string()));
    }
    if let Some(idx) = find_binding_use(collected, module, name, active) {
        return resolve_use_target(collected, idx, active);
    }
    for (i, u) in collected.uses.iter().enumerate() {
        if u.binding.is_some() || u.module_path.as_slice() != module {
            continue;
        }
        if let Some(t) = with_active_use(active, i, |active| {
            match resolve_module(collected, &u.target, u.leading_colon, module, active) {
                Some(ModuleTarget::Local(m)) => {
                    lookup_name_in(collected, &m, name, active, visited, extern_glob)
                }
                Some(ModuleTarget::Extern(p)) => {
                    if extern_glob.is_none() {
                        *extern_glob = Some(p);
                    }
                    None
                }
                None => None,
            }
        }) {
            return Some(t);
        }
    }
    None
}

fn eci_libname(module_path: &[String], const_name: &str) -> String {
    let mut parts: Vec<&str> = module_path.iter().map(String::as_str).collect();
    parts.push(const_name);
    let joined = parts.join("_");
    let mut out = String::from("eci_");
    for c in joined.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out.push_str(&format!(
        "_{:016x}",
        fnv1a64(format!("{}::{}", module_path.join("::"), const_name).as_bytes())
    ));
    out
}

fn resolve_against(manifest: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest.join(path)
    }
}

fn submodule_dir_for_file(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    match name.as_deref() {
        Some("lib.rs") | Some("mod.rs") | Some("main.rs") => path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        _ => path.with_extension(""),
    }
}

fn walk_file(file_path: &Path, submodule_dir: &Path, mod_path: &[String], out: &mut CrateItems) {
    println!("cargo:rerun-if-changed={}", file_path.display());
    let src = std::fs::read_to_string(file_path).unwrap_or_else(|err| {
        panic!("{}: {err}", file_path.display());
    });
    let file = syn::parse_file(&src).unwrap_or_else(|err| {
        panic!("{}: {err}", file_path.display());
    });
    if !attrs_cfgs_hold(&file.attrs, file_path) {
        return;
    }
    walk_items(&file.items, file_path, submodule_dir, mod_path, out);
}

fn walk_items(
    items: &[Item],
    file_path: &Path,
    submodule_dir: &Path,
    mod_path: &[String],
    out: &mut CrateItems,
) {
    for item in items {
        if !item_cfgs_hold(item, file_path) {
            continue;
        }
        match item {
            Item::Type(alias) if alias.generics.params.is_empty() => {
                out.aliases.push(CollectedAlias {
                    module_path: mod_path.to_vec(),
                    name: unraw_ident(&alias.ident),
                    ty: (*alias.ty).clone(),
                });
            }
            Item::Struct(item) => {
                out.structs
                    .push((mod_path.to_vec(), unraw_ident(&item.ident)));
            }
            Item::Use(item) => {
                flatten_use(
                    mod_path,
                    item.leading_colon.is_some(),
                    &item.tree,
                    Vec::new(),
                    &mut out.uses,
                );
            }
            Item::Macro(mac) => {
                let last = mac
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.to_string());
                match last.as_deref() {
                    Some("external_compilation_info") => {
                        let parsed: EciInput =
                            syn::parse2(mac.mac.tokens.clone()).unwrap_or_else(|err| {
                                panic!("{}: {err}", file_path.display());
                            });
                        let eci = parsed.to_eci().unwrap_or_else(|err| {
                            panic!("{}: {err}", file_path.display());
                        });
                        out.ecis.push(CollectedEci {
                            module_path: mod_path.to_vec(),
                            name: unraw_ident(&parsed.name),
                            eci,
                        });
                    }
                    Some("llexternal") => {
                        let parsed: LlexternalHead = syn::parse2(mac.mac.tokens.clone())
                            .unwrap_or_else(|err| {
                                panic!("{}: {err}", file_path.display());
                            });
                        if attrs_cfgs_hold(&parsed.attrs, file_path) && parsed.macro_.is_some() {
                            out.llexternals.push(CollectedLlexternal {
                                file_path: file_path.to_path_buf(),
                                module_path: mod_path.to_vec(),
                                head: parsed,
                            });
                        }
                    }
                    _ => {}
                }
            }
            Item::Mod(module) => {
                let name = unraw_ident(&module.ident);
                let mut child_path = mod_path.to_vec();
                child_path.push(name.clone());
                out.modules.push(child_path.clone());
                match &module.content {
                    Some((_, items)) => {
                        let nested_dir = submodule_dir.join(&name);
                        walk_items(items, file_path, &nested_dir, &child_path, out);
                    }
                    None => {
                        let child_file = resolve_mod_file(
                            file_path,
                            submodule_dir,
                            &name,
                            path_attr(&module.attrs, file_path),
                        );
                        let child_dir = submodule_dir_for_file(&child_file);
                        walk_file(&child_file, &child_dir, &child_path, out);
                    }
                }
            }
            _ => {}
        }
    }
}

/// rustc spells a keyword module `r#struct`; the file is still `struct.rs`.
fn unraw_ident(ident: &syn::Ident) -> String {
    let name = ident.to_string();
    match name.strip_prefix("r#") {
        Some(stem) => stem.to_string(),
        None => name,
    }
}

fn flatten_use(
    module_path: &[String],
    leading_colon: bool,
    tree: &UseTree,
    prefix: Vec<String>,
    out: &mut Vec<CollectedUse>,
) {
    match tree {
        UseTree::Path(p) => {
            let mut prefix = prefix;
            prefix.push(unraw_ident(&p.ident));
            flatten_use(module_path, leading_colon, &p.tree, prefix, out);
        }
        UseTree::Name(n) => {
            if n.ident == "self" {
                if let Some(binding) = prefix.last().cloned() {
                    out.push(CollectedUse {
                        module_path: module_path.to_vec(),
                        binding: Some(binding),
                        target: prefix,
                        leading_colon,
                    });
                }
            } else {
                let ident = unraw_ident(&n.ident);
                let mut target = prefix;
                target.push(ident.clone());
                out.push(CollectedUse {
                    module_path: module_path.to_vec(),
                    binding: Some(ident),
                    target,
                    leading_colon,
                });
            }
        }
        UseTree::Rename(r) => {
            let binding = unraw_ident(&r.rename);
            let mut target = prefix;
            if r.ident != "self" {
                target.push(unraw_ident(&r.ident));
            }
            out.push(CollectedUse {
                module_path: module_path.to_vec(),
                binding: Some(binding),
                target,
                leading_colon,
            });
        }
        UseTree::Glob(_) => {
            out.push(CollectedUse {
                module_path: module_path.to_vec(),
                binding: None,
                target: prefix,
                leading_colon,
            });
        }
        UseTree::Group(g) => {
            for tree in &g.items {
                flatten_use(module_path, leading_colon, tree, prefix.clone(), out);
            }
        }
    }
}

fn resolve_mod_file(
    current_file: &Path,
    submodule_dir: &Path,
    name: &str,
    path_attr: Option<String>,
) -> PathBuf {
    if let Some(p) = path_attr {
        let path = current_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(p);
        if !path.exists() {
            panic!("{}", path.display());
        }
        return path;
    }
    let rs = submodule_dir.join(format!("{name}.rs"));
    if rs.exists() {
        return rs;
    }
    let mod_rs = submodule_dir.join(name).join("mod.rs");
    if mod_rs.exists() {
        return mod_rs;
    }
    panic!("{}", rs.display());
}

fn path_attr(attrs: &[Attribute], file_path: &Path) -> Option<String> {
    for meta in expand_cfg_attr(attrs, file_path) {
        if !meta.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(nv) = meta else {
            continue;
        };
        if let Expr::Lit(expr_lit) = nv.value
            && let Lit::Str(s) = expr_lit.lit
        {
            return Some(s.value());
        }
    }
    None
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

fn item_cfgs_hold(item: &Item, file_path: &Path) -> bool {
    attrs_cfgs_hold(item_attrs(item), file_path)
}

fn expand_cfg_attr(attrs: &[Attribute], file_path: &Path) -> Vec<Meta> {
    let mut out = Vec::new();
    for attr in attrs {
        expand_meta(&attr.meta, file_path, &mut out);
    }
    out
}

fn expand_meta(meta: &Meta, file_path: &Path, out: &mut Vec<Meta>) {
    if meta.path().is_ident("cfg_attr") {
        let Meta::List(list) = meta else {
            return;
        };
        let nested =
            match Punctuated::<Meta, Token![,]>::parse_terminated.parse2(list.tokens.clone()) {
                Ok(p) => p,
                Err(err) => panic!("{}: {err}", file_path.display()),
            };
        let mut iter = nested.into_iter();
        let Some(pred) = iter.next() else {
            return;
        };
        if cfg_holds(&pred) {
            for inner in iter {
                expand_meta(&inner, file_path, out);
            }
        }
    } else {
        out.push(meta.clone());
    }
}

fn attrs_cfgs_hold(attrs: &[Attribute], file_path: &Path) -> bool {
    for meta in expand_cfg_attr(attrs, file_path) {
        if !meta.path().is_ident("cfg") {
            continue;
        }
        let Meta::List(list) = meta else {
            return false;
        };
        let pred = syn::parse2::<Meta>(list.tokens).unwrap_or_else(|err| {
            panic!("{}: {err}", file_path.display());
        });
        if !cfg_holds(&pred) {
            return false;
        }
    }
    true
}

fn cfg_holds(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => {
            let Some(ident) = path.get_ident() else {
                return false;
            };
            let name = ident.to_string();
            if name == "test" || name == "doc" {
                return false;
            }
            std::env::var_os(format!("CARGO_CFG_{}", env_key(&name))).is_some()
        }
        Meta::NameValue(nv) => {
            let Some(ident) = nv.path.get_ident() else {
                return false;
            };
            let key = ident.to_string();
            let Expr::Lit(expr_lit) = &nv.value else {
                return false;
            };
            let Lit::Str(value) = &expr_lit.lit else {
                return false;
            };
            let value = value.value();
            if key == "feature" {
                return std::env::var_os(format!("CARGO_FEATURE_{}", env_key(&value))).is_some();
            }
            match std::env::var(format!("CARGO_CFG_{}", env_key(&key))) {
                Ok(v) => v.split(',').any(|item| item == value),
                Err(_) => false,
            }
        }
        Meta::List(list) => {
            let Some(ident) = list.path.get_ident() else {
                return false;
            };
            let nested =
                match Punctuated::<Meta, Token![,]>::parse_terminated.parse2(list.tokens.clone()) {
                    Ok(p) => p,
                    Err(_) => return false,
                };
            match ident.to_string().as_str() {
                "all" => nested.iter().all(cfg_holds),
                "any" => nested.iter().any(cfg_holds),
                "not" => nested.len() == 1 && !cfg_holds(&nested[0]),
                _ => false,
            }
        }
    }
}

fn env_key(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c == '-' {
                '_'
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect()
}
