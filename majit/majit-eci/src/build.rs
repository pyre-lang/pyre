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
/// success.
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
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "a" || ext == "lib" {
            if let Some(parent) = path.parent() {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
            if let Some(name) = path.file_name() {
                println!(
                    "cargo:rustc-link-lib=static:+verbatim={}",
                    name.to_string_lossy()
                );
            }
        } else {
            // `_linkfiles` hands object files to the linker directly; Cargo links
            // only libraries, so the object is archived first.
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("obj");
            let name = format!(
                "eci_linkfile_{}",
                stem.chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect::<String>()
            );
            cc::Build::new()
                .object(path)
                .cargo_warnings(false)
                .try_compile(&name)
                .unwrap_or_else(|err| panic!("{file}: {err}"));
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
                    match resolve_path(collected, path, module_path, 0) {
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
                    crate::cdecl::gettype(ty, &item.module_path, &resolve).unwrap_or_else(|msg| {
                        panic!(
                            "{}: llexternal `{}`: {msg}",
                            item.file_path.display(),
                            item.head.name
                        )
                    })
                })
                .collect();
            let result = crate::cdecl::gettype(&item.head.result, &item.module_path, &resolve)
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
                match resolve_path(collected, path, &item.module_path, 0) {
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
    skip: Option<usize>,
) -> Option<usize> {
    collected.uses.iter().enumerate().position(|(i, u)| {
        skip != Some(i) && u.module_path.as_slice() == module && u.binding.as_deref() == Some(name)
    })
}

fn resolve_path(
    collected: &CrateItems,
    path: &syn::Path,
    module: &[String],
    depth: usize,
) -> Option<Target> {
    let segs: Vec<String> = path
        .segments
        .iter()
        .map(|s| unraw_ident(&s.ident))
        .collect();
    resolve_segs(
        collected,
        &segs,
        path.leading_colon.is_some(),
        module,
        depth,
        None,
    )
}

fn resolve_use_target(collected: &CrateItems, use_idx: usize, depth: usize) -> Option<Target> {
    let u = &collected.uses[use_idx];
    let segs = u.target.clone();
    let leading = u.leading_colon;
    let module = u.module_path.clone();
    resolve_segs(collected, &segs, leading, &module, depth, Some(use_idx))
}

fn resolve_segs(
    collected: &CrateItems,
    segs: &[String],
    leading_colon: bool,
    module: &[String],
    depth: usize,
    skip_use: Option<usize>,
) -> Option<Target> {
    if depth > 32 {
        return None;
    }
    if segs.is_empty() {
        return None;
    }
    if leading_colon {
        return Some(Target::Extern(segs.to_vec()));
    }
    if segs.len() == 1 && !matches!(segs[0].as_str(), "crate" | "self" | "super") {
        return lookup_name(collected, module, &segs[0], depth, skip_use);
    }
    let last = &segs[segs.len() - 1];
    match resolve_module(
        collected,
        &segs[..segs.len() - 1],
        leading_colon,
        module,
        depth,
        skip_use,
    ) {
        Some(ModuleTarget::Local(m)) => lookup_name(collected, &m, last, depth, skip_use),
        Some(ModuleTarget::Extern(mut p)) => {
            p.push(last.clone());
            Some(Target::Extern(p))
        }
        None => None,
    }
}

fn walk_module_seg(
    collected: &CrateItems,
    cur: &[String],
    seg: &str,
    after: &[String],
    depth: usize,
    skip_use: Option<usize>,
) -> Option<ModuleTarget> {
    let mut child = cur.to_vec();
    child.push(seg.to_string());
    if is_module(collected, &child) {
        return Some(ModuleTarget::Local(child));
    }
    if let Some(idx) = find_binding_use(collected, cur, seg, skip_use) {
        match resolve_use_target(collected, idx, depth + 1) {
            Some(Target::Local(m, n)) => {
                let mut p = m;
                p.push(n);
                if is_module(collected, &p) {
                    Some(ModuleTarget::Local(p))
                } else {
                    None
                }
            }
            Some(Target::Extern(mut p)) => {
                p.extend(after.iter().cloned());
                Some(ModuleTarget::Extern(p))
            }
            None => None,
        }
    } else {
        None
    }
}

fn resolve_module(
    collected: &CrateItems,
    segs: &[String],
    leading_colon: bool,
    module: &[String],
    depth: usize,
    skip_use: Option<usize>,
) -> Option<ModuleTarget> {
    if depth > 32 {
        return None;
    }
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
            return walk_module_rest(collected, cur, &segs[i..], depth, skip_use);
        }
        _ => {
            let x = &segs[0];
            let mut child = module.to_vec();
            child.push(x.clone());
            if is_module(collected, &child) {
                (child, &segs[1..])
            } else if let Some(idx) = find_binding_use(collected, module, x, skip_use) {
                let mut spliced = collected.uses[idx].target.clone();
                spliced.extend(segs[1..].iter().cloned());
                let leading = collected.uses[idx].leading_colon;
                return resolve_module(collected, &spliced, leading, module, depth + 1, Some(idx));
            } else {
                return Some(ModuleTarget::Extern(segs.to_vec()));
            }
        }
    };
    walk_module_rest(collected, cur, rest, depth, skip_use)
}

fn walk_module_rest(
    collected: &CrateItems,
    mut cur: Vec<String>,
    rest: &[String],
    depth: usize,
    skip_use: Option<usize>,
) -> Option<ModuleTarget> {
    for i in 0..rest.len() {
        let after = &rest[i + 1..];
        match walk_module_seg(collected, &cur, &rest[i], after, depth, skip_use)? {
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
    depth: usize,
    skip_use: Option<usize>,
) -> Option<Target> {
    if depth > 32 {
        return None;
    }
    if defines_name(collected, module, name) {
        return Some(Target::Local(module.to_vec(), name.to_string()));
    }
    if let Some(idx) = find_binding_use(collected, module, name, skip_use) {
        return resolve_use_target(collected, idx, depth + 1);
    }
    for (i, u) in collected.uses.iter().enumerate() {
        if u.binding.is_some() || u.module_path.as_slice() != module {
            continue;
        }
        match resolve_module(
            collected,
            &u.target,
            u.leading_colon,
            module,
            depth + 1,
            Some(i),
        ) {
            Some(ModuleTarget::Local(m)) => {
                if let Some(t) = lookup_name(collected, &m, name, depth + 1, None) {
                    return Some(t);
                }
            }
            Some(ModuleTarget::Extern(mut p)) => {
                p.push(name.to_string());
                return Some(Target::Extern(p));
            }
            None => {}
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
