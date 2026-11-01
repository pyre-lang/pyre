//! Crate `build.rs` entry that compiles every ECI this crate declares.

use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Item, Lit, Meta, Token};

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
        build.include(resolve_against(&manifest, dir));
    }
    build.include(cdir());
    for flag in &e.compile_extra {
        build.flag(flag);
    }
    build.warnings(false);
    build.try_compile(libname)?;
    if e.use_cpp_linker {
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
        if let Some(parent) = path.parent() {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
        if let Some(name) = path.file_name() {
            println!(
                "cargo:rustc-link-lib=static:+verbatim={}",
                name.to_string_lossy()
            );
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

#[derive(Default)]
struct CrateItems {
    ecis: Vec<CollectedEci>,
    aliases: Vec<CollectedAlias>,
    llexternals: Vec<CollectedLlexternal>,
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
                |path: &syn::Path, module_path: &[String]| -> Option<(syn::Type, Vec<String>)> {
                    let idx = resolve_index(
                        &collected.aliases,
                        path,
                        module_path,
                        |alias| alias.name.as_str(),
                        |alias| alias.module_path.as_slice(),
                    )?;
                    let alias = &collected.aliases[idx];
                    Some((alias.ty.clone(), alias.module_path.clone()))
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
                match resolve_index(
                    &collected.ecis,
                    path,
                    &item.module_path,
                    |eci| eci.name.as_str(),
                    |eci| eci.module_path.as_slice(),
                ) {
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

fn candidate_modules(path: &syn::Path, module_path: &[String]) -> Vec<Vec<String>> {
    let n = path.segments.len();
    if n == 0 {
        return vec![module_path.to_vec()];
    }
    let prefix: Vec<String> = path
        .segments
        .iter()
        .take(n - 1)
        .map(|seg| unraw_ident(&seg.ident))
        .collect();
    if prefix.is_empty() {
        return vec![module_path.to_vec()];
    }
    match prefix[0].as_str() {
        "crate" => vec![prefix[1..].to_vec()],
        "self" => {
            let mut m = module_path.to_vec();
            m.extend(prefix[1..].iter().cloned());
            vec![m]
        }
        "super" => {
            let mut m = module_path.to_vec();
            let mut rest = 0;
            for (i, seg) in prefix.iter().enumerate() {
                if seg == "super" {
                    let _ = m.pop();
                    rest = i + 1;
                } else {
                    break;
                }
            }
            m.extend(prefix[rest..].iter().cloned());
            vec![m]
        }
        _ => {
            let mut child = module_path.to_vec();
            child.extend(prefix.iter().cloned());
            vec![child, prefix.to_vec()]
        }
    }
}

fn resolve_index<T>(
    items: &[T],
    path: &syn::Path,
    module_path: &[String],
    name_of: impl Fn(&T) -> &str,
    module_of: impl Fn(&T) -> &[String],
) -> Option<usize> {
    let last = path.segments.last()?.ident.to_string();
    for cand in candidate_modules(path, module_path) {
        if let Some(idx) = items
            .iter()
            .position(|item| module_of(item) == cand.as_slice() && name_of(item) == last)
        {
            return Some(idx);
        }
    }
    // A single identifier with no hit in its own module stands in for a `use`
    // import, which the walker does not track.
    if path.segments.len() == 1 {
        let matches: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, item)| name_of(item) == last)
            .map(|(idx, _)| idx)
            .collect();
        if matches.len() == 1 {
            return Some(matches[0]);
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
                    name: alias.ident.to_string(),
                    ty: (*alias.ty).clone(),
                });
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
                            name: parsed.name.to_string(),
                            eci,
                        });
                    }
                    Some("llexternal") => {
                        let parsed: LlexternalHead = syn::parse2(mac.mac.tokens.clone())
                            .unwrap_or_else(|err| {
                                panic!("{}: {err}", file_path.display());
                            });
                        if parsed.macro_.is_some() {
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
                            path_attr(&module.attrs),
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

fn path_attr(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        if let Expr::Lit(expr_lit) = &nv.value
            && let Lit::Str(s) = &expr_lit.lit
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
    for attr in item_attrs(item) {
        if !attr.path().is_ident("cfg") {
            continue;
        }
        let meta = attr.parse_args::<Meta>().unwrap_or_else(|err| {
            panic!("{}: {err}", file_path.display());
        });
        if !cfg_holds(&meta) {
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
