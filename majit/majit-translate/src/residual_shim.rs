//! Generated residual-call shims — the genc analogue for policy-declined graphs.
//!
//! RPython's `getfunctionptr` / `handle_residual_call` give every residual
//! callee a callable address because genc `FunctionCodeGenerator` emits every
//! graph as a C function. pyre has no genc, so a policy-declined graph
//! (`look_inside_graph` reasons such as `loop-without-unroll_safe`) keeps a
//! `symbolic_fnaddr_for_path` hash unless a macro trampoline or hand list
//! bound one. This module collects those residual targets and emits
//! `extern "C"` shims whose ABI matches the calldescr the codewriter already
//! computed.

use std::collections::{BTreeSet, HashMap, HashSet};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{FunDecl, TyRef, TypeDeclKind};
use serde::{Deserialize, Serialize};

use crate::codewriter::call::CallDescriptor;
use crate::front::clause_spec::decl_is_generic;
use crate::parse::CallPath;

/// One residual callee the pipeline may emit a shim for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualShimTarget {
    /// CallPath spelling `symbolic_fnaddr_for_path` recorded.
    pub path: String,
    /// FunDecl `name_path` used as the Rust call.
    pub rust_path: String,
    /// `CallDescr.arg_classes`.
    pub arg_classes: String,
    /// `CallDescr.result_type`.
    pub result_type: char,
    /// Compilable Rust types for each ABI argument, parallel to `arg_classes`.
    pub arg_rust_types: Vec<String>,
    /// Compilable Rust return type (Ok payload when the FunDecl returns `Result`).
    pub ret_rust_type: String,
    /// The FunDecl returns `Result<T, E>`; the shim publishes `E` through
    /// `ResidualError`.
    pub result_is_exc: bool,
    pub is_unsafe: bool,
    pub is_generic: bool,
    pub is_method: bool,
    pub public: bool,
    pub crate_name: String,
    /// True when a parameter or result type cannot be converted.
    pub unsupported_type: bool,
}

/// Counts printed by the consuming build script.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidualShimCensus {
    pub emitted: usize,
    pub unnameable: usize,
    pub generic: usize,
    pub unsupported_type: usize,
}

impl ResidualShimCensus {
    pub fn line(&self) -> String {
        format!(
            "shim-emitted={} shim-unnameable={} shim-generic={} shim-unsupported-type={}",
            self.emitted, self.unnameable, self.generic, self.unsupported_type
        )
    }
}

/// FunDecl snapshot harvested from one loaded LLBC.
#[derive(Debug, Clone)]
pub struct LlbcFnCatalogEntry {
    pub path: String,
    pub crate_name: String,
    pub public: bool,
    pub is_generic: bool,
    pub is_method: bool,
    pub is_unsafe: bool,
    pub arg_rust_types: Option<Vec<String>>,
    pub ret_rust_type: Option<ShimReturn>,
}

/// Return side of a catalogued FunDecl.
#[derive(Debug, Clone)]
pub struct ShimReturn {
    pub rust_type: String,
    pub is_exc: bool,
}

/// Harvest every local FunDecl that could become a residual shim.
pub fn catalog_from_llbc(llbc: &Llbc) -> Vec<LlbcFnCatalogEntry> {
    let crate_name = llbc.crate_name().to_string();
    let mut entries = Vec::new();
    for fd in llbc.iter_fun_decls() {
        if !fd.item_meta.is_local {
            continue;
        }
        if fd.is_global_initializer().is_some() {
            continue;
        }
        let path = fd.item_meta.name_path();
        if path.is_empty() {
            continue;
        }
        let arg_rust_types = fd
            .signature
            .inputs
            .iter()
            .map(|ty| {
                let rust = shim_ty_to_rust(ty, llbc)?;
                rust_ty_convertible(&rust).then_some(rust)
            })
            .collect::<Option<Vec<_>>>();
        let ret_rust_type = shim_return_ty(&fd.signature.output, llbc)
            .filter(|ret| rust_ty_convertible(&ret.rust_type) || ret.rust_type == "()");
        entries.push(LlbcFnCatalogEntry {
            path,
            crate_name: crate_name.clone(),
            public: fd.item_meta.attr_info.public,
            is_generic: decl_is_generic(fd),
            is_method: decl_is_method(fd),
            is_unsafe: fd.signature.is_unsafe,
            arg_rust_types,
            ret_rust_type,
        });
    }
    entries
}

fn decl_is_method(fd: &FunDecl) -> bool {
    if fd.is_adt_constructor() {
        return true;
    }
    if fd
        .src
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|obj| obj.contains_key("TraitImpl"))
    {
        return true;
    }
    let path = fd.item_meta.name_path_str();
    if path.contains("<Impl") {
        return true;
    }
    matches!(fd.first_arg_local_name().as_deref(), Some("self"))
}

fn shim_return_ty(ty: &TyRef, llbc: &Llbc) -> Option<ShimReturn> {
    if let Some((ok, _err)) = tyref_result_args(ty, llbc) {
        let rust_type = shim_ty_to_rust(&ok, llbc)?;
        return Some(ShimReturn {
            rust_type,
            is_exc: true,
        });
    }
    Some(ShimReturn {
        rust_type: shim_ty_to_rust(ty, llbc)?,
        is_exc: false,
    })
}

/// Collect residual shim targets from policy-declined residual calls.
pub fn collect_residual_shim_targets(
    policy_declined: &HashSet<CallPath>,
    residual_call_descrs: &HashMap<CallPath, CallDescriptor>,
    bound_fnaddrs: &HashMap<CallPath, i64>,
    catalog: &[LlbcFnCatalogEntry],
) -> Vec<ResidualShimTarget> {
    let mut by_path: HashMap<&str, &LlbcFnCatalogEntry> = HashMap::new();
    for entry in catalog {
        by_path.entry(entry.path.as_str()).or_insert(entry);
    }
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut declined: Vec<&CallPath> = policy_declined.iter().collect();
    declined.sort_by_key(|path| path.canonical_key());
    for path in declined {
        if bound_fnaddrs.contains_key(path) {
            continue;
        }
        let Some(descriptor) = residual_call_descrs.get(path) else {
            continue;
        };
        let key = path.canonical_key();
        if !seen.insert(key.clone()) {
            continue;
        }
        let entry = lookup_catalog(&by_path, catalog, &key);
        let Some(entry) = entry else {
            continue;
        };
        let unsupported_type = entry.arg_rust_types.is_none()
            || entry.ret_rust_type.is_none()
            || entry.arg_rust_types.as_ref().is_some_and(|args| {
                args.iter().map(|ty| rust_ty_abi_words(ty)).sum::<usize>()
                    != descriptor.arg_classes.chars().filter(|c| *c != 'v').count()
            });
        let (arg_rust_types, ret_rust_type, result_is_exc) =
            match (&entry.arg_rust_types, &entry.ret_rust_type) {
                (Some(args), Some(ret)) if !unsupported_type => {
                    (args.clone(), ret.rust_type.clone(), ret.is_exc)
                }
                _ => (Vec::new(), String::new(), false),
            };
        targets.push(ResidualShimTarget {
            path: key,
            rust_path: entry.path.clone(),
            arg_classes: descriptor.arg_classes.clone(),
            result_type: descriptor.result_type,
            arg_rust_types,
            ret_rust_type,
            result_is_exc,
            is_unsafe: entry.is_unsafe,
            is_generic: entry.is_generic,
            is_method: entry.is_method,
            public: entry.public,
            crate_name: entry.crate_name.clone(),
            unsupported_type,
        });
    }
    targets.sort_by(|a, b| a.path.cmp(&b.path));
    targets
}

fn lookup_catalog<'a>(
    by_path: &HashMap<&str, &'a LlbcFnCatalogEntry>,
    catalog: &'a [LlbcFnCatalogEntry],
    key: &str,
) -> Option<&'a LlbcFnCatalogEntry> {
    if let Some(entry) = by_path.get(key) {
        return Some(*entry);
    }
    let suffix = format!("::{key}");
    let mut matches = catalog
        .iter()
        .filter(|entry| entry.path == key || entry.path.ends_with(&suffix));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// Crate and module paths `pyre-jit-trace` can name, from its Cargo.toml
/// dependencies and the `pub mod` tree of collected sources.
#[derive(Debug, Clone, Default)]
pub struct Nameability {
    pub reachable_crates: BTreeSet<String>,
    pub public_modules: BTreeSet<String>,
}

impl Nameability {
    pub fn from_cargo_and_sources(
        cargo_toml: &str,
        enabled_features: &BTreeSet<String>,
        source_paths: &[String],
    ) -> Self {
        let reachable_crates = rustc_dep_crate_names(cargo_toml, enabled_features);
        let public_modules = public_modules_from_sources(source_paths);
        Self {
            reachable_crates,
            public_modules,
        }
    }

    /// The function's whole module path is reachable and the item is public.
    pub fn can_name(&self, rust_path: &str, fn_public: bool) -> bool {
        if !fn_public {
            return false;
        }
        let mut segments: Vec<&str> = rust_path.split("::").filter(|s| !s.is_empty()).collect();
        if segments.is_empty() {
            return false;
        }
        segments.pop();
        if segments.is_empty() {
            return false;
        }
        let crate_name = segments[0];
        if !self.reachable_crates.contains(crate_name) {
            return false;
        }
        let mut prefix = String::new();
        for (i, segment) in segments.iter().enumerate() {
            if i > 0 {
                prefix.push_str("::");
            }
            prefix.push_str(segment);
            if i == 0 {
                continue;
            }
            if !self.public_modules.contains(&prefix) {
                return false;
            }
        }
        true
    }
}

fn rustc_dep_crate_names(
    cargo_toml: &str,
    enabled_features: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut crates = BTreeSet::new();
    let mut in_deps = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = is_runtime_deps_header(trimmed);
            continue;
        }
        if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(name) = dep_line_crate_name(trimmed) else {
            continue;
        };
        if dep_line_is_optional(trimmed) && !optional_dep_enabled(&name, enabled_features) {
            continue;
        }
        crates.insert(name.replace('-', "_"));
    }
    crates
}

fn is_runtime_deps_header(header: &str) -> bool {
    if header == "[dependencies]" {
        return true;
    }
    header.starts_with("[target.") && header.ends_with(".dependencies]")
}

fn dep_line_crate_name(line: &str) -> Option<String> {
    let (name, _rest) = line.split_once('=')?;
    let name = name.trim();
    if name.is_empty() || name.contains('.') || name.contains('[') {
        return None;
    }
    Some(name.to_string())
}

fn dep_line_is_optional(line: &str) -> bool {
    line.contains("optional") && line.contains("true")
}

fn optional_dep_enabled(dep_name: &str, enabled_features: &BTreeSet<String>) -> bool {
    enabled_features.contains(dep_name) || enabled_features.contains(&dep_name.replace('-', "_"))
}

fn public_modules_from_sources(source_paths: &[String]) -> BTreeSet<String> {
    let mut public = BTreeSet::new();
    for path in source_paths {
        let Some(crate_name) = crate_name_from_source_path(path) else {
            continue;
        };
        public.insert(crate_name.clone());
        let module = crate::module_path::module_path_from_source_file(path);
        let parent = if module.is_empty() {
            crate_name
        } else {
            format!("{crate_name}::{module}")
        };
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&src) else {
            continue;
        };
        for item in file.items {
            let syn::Item::Mod(module_item) = item else {
                continue;
            };
            if !matches!(module_item.vis, syn::Visibility::Public(_)) {
                continue;
            }
            public.insert(format!("{parent}::{}", module_item.ident));
        }
    }
    public
}

fn crate_name_from_source_path(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let idx = normalized.rfind("/src/")?;
    let crate_dir = normalized[..idx].rsplit('/').next()?;
    Some(crate_dir.replace('-', "_"))
}

/// Filter targets and emit shim source plus the census.
pub fn emit_residual_shim_source(
    targets: &[ResidualShimTarget],
    nameability: &Nameability,
) -> (String, ResidualShimCensus) {
    let mut census = ResidualShimCensus::default();
    let mut emitted = Vec::new();
    for target in targets {
        if target.is_generic {
            census.generic += 1;
            continue;
        }
        if target.is_method || target.unsupported_type {
            census.unsupported_type += 1;
            continue;
        }
        if !nameability.can_name(&target.rust_path, target.public)
            || !target_types_nameable(target, nameability)
        {
            census.unnameable += 1;
            continue;
        }
        if shim_conversion_unsupported(target) {
            census.unsupported_type += 1;
            continue;
        }
        census.emitted += 1;
        emitted.push(target);
    }
    (generate_residual_shim_source(&emitted), census)
}

fn target_types_nameable(target: &ResidualShimTarget, nameability: &Nameability) -> bool {
    target
        .arg_rust_types
        .iter()
        .chain(std::iter::once(&target.ret_rust_type))
        .all(|ty| rust_type_nameable(ty, nameability))
}

fn rust_type_nameable(ty: &str, nameability: &Nameability) -> bool {
    named_paths_in_type(ty)
        .into_iter()
        .all(|path| nameability.can_name(&path, true))
}

fn named_paths_in_type(ty: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut current = String::new();
    for ch in ty.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' {
            current.push(ch);
        } else if !current.is_empty() {
            if current.contains("::") {
                paths.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.contains("::") {
        paths.push(current);
    }
    paths
}

fn shim_conversion_unsupported(target: &ResidualShimTarget) -> bool {
    let abi_words: usize = target
        .arg_rust_types
        .iter()
        .map(|ty| rust_ty_abi_words(ty))
        .sum();
    if target.arg_classes.chars().filter(|c| *c != 'v').count() != abi_words {
        return true;
    }
    if target.result_type != 'v' && target.ret_rust_type.is_empty() {
        return true;
    }
    if target
        .arg_rust_types
        .iter()
        .any(|ty| !rust_ty_convertible(ty))
    {
        return true;
    }
    if target.result_type != 'v'
        && target.ret_rust_type != "()"
        && !rust_ty_convertible(&target.ret_rust_type)
    {
        return true;
    }
    false
}

/// Empty table used when LLBC extraction writes placeholders.
pub fn empty_residual_shim_source() -> String {
    generate_residual_shim_source(&[])
}

pub fn generate_residual_shim_source(targets: &[&ResidualShimTarget]) -> String {
    let mut out = String::from(
        "// Generated residual-call shims. genc FunctionCodeGenerator analogue:\n\
         // every policy-declined residual callee gets a callable lltype ABI entry.\n\
         #[derive(Clone, Copy)]\n\
         pub struct ResidualShimAddr(pub *const ());\n\
         unsafe impl Sync for ResidualShimAddr {}\n\
         unsafe impl Send for ResidualShimAddr {}\n\n",
    );
    for (index, target) in targets.iter().enumerate() {
        out.push_str(&emit_one_shim(index, target));
        out.push('\n');
    }
    if targets.is_empty() {
        out.push_str("pub static GENERATED_RESIDUAL_SHIMS: &[(&str, ResidualShimAddr)] = &[];\n");
        return out;
    }
    out.push_str("pub static GENERATED_RESIDUAL_SHIMS: &[(&str, ResidualShimAddr)] = &[\n");
    for (index, target) in targets.iter().enumerate() {
        let ident = shim_ident(index, &target.path);
        out.push_str(&format!(
            "    (\"{}\", ResidualShimAddr({ident} as *const ())),\n",
            escape_path(&target.path)
        ));
        if target.path != target.rust_path {
            out.push_str(&format!(
                "    (\"{}\", ResidualShimAddr({ident} as *const ())),\n",
                escape_path(&target.rust_path)
            ));
        }
    }
    out.push_str("];\n");
    out
}

fn escape_path(path: &str) -> String {
    path.replace('\\', "\\\\").replace('"', "\\\"")
}

fn shim_ident(index: usize, path: &str) -> String {
    let mut ident = format!("__residual_shim_{index}_");
    for ch in path.chars() {
        if ch.is_ascii_alphanumeric() {
            ident.push(ch);
        } else {
            ident.push('_');
        }
    }
    ident
}

fn emit_one_shim(index: usize, target: &ResidualShimTarget) -> String {
    let ident = shim_ident(index, &target.path);
    let params = shim_params(&target.arg_classes);
    let ret_abi = shim_ret_abi(target.result_type);
    let body = shim_body(target);
    let ret_arrow = if ret_abi.is_empty() {
        String::new()
    } else {
        format!(" -> {ret_abi}")
    };
    format!(
        "#[allow(non_snake_case, unused_unsafe, clippy::unused_unit)]\n\
         unsafe extern \"C\" fn {ident}({params}){ret_arrow} {{\n{body}}}\n"
    )
}

fn shim_params(arg_classes: &str) -> String {
    arg_classes
        .chars()
        .filter(|c| *c != 'v')
        .enumerate()
        .map(|(i, class)| {
            let ty = match class {
                'f' | 'L' => "f64",
                _ => "i64",
            };
            format!("__a{i}: {ty}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn shim_ret_abi(result_type: char) -> &'static str {
    match result_type {
        'v' => "",
        'f' | 'L' => "f64",
        _ => "i64",
    }
}

fn shim_body(target: &ResidualShimTarget) -> String {
    let mut lines = String::new();
    let mut args = Vec::new();
    let classes: Vec<char> = target.arg_classes.chars().filter(|c| *c != 'v').collect();
    let mut word_i = 0usize;
    for (i, rust_ty) in target.arg_rust_types.iter().enumerate() {
        if rust_ty_abi_words(rust_ty) == 2 {
            let ptr_word = format!("__a{word_i}");
            word_i += 1;
            let len_word = format!("__a{word_i}");
            word_i += 1;
            let conv = pair_slice_from_words(rust_ty, &ptr_word, &len_word);
            lines.push_str(&format!("    let __p{i}: {rust_ty} = {conv};\n"));
        } else if let Some(item) = object_gcarray_item(rust_ty) {
            let word = format!("__a{word_i}");
            word_i += 1;
            let conv = format!(
                "unsafe {{ ::majit_ir::helper_fnaddr::object_gcarray_slice::<{item}>({word}) }}"
            );
            lines.push_str(&format!("    let __p{i}: {rust_ty} = {conv};\n"));
        } else {
            let class = classes.get(word_i).copied().unwrap_or('i');
            let word = format!("__a{word_i}");
            word_i += 1;
            let conv = from_word_expr(rust_ty, class, &word);
            lines.push_str(&format!("    let __p{i}: {rust_ty} = {conv};\n"));
        }
        args.push(format!("__p{i}"));
    }
    let call_args = args.join(", ");
    let invoke = if target.is_unsafe {
        format!("unsafe {{ {}({call_args}) }}", target.rust_path)
    } else {
        format!("{}({call_args})", target.rust_path)
    };
    if target.result_type == 'v' && !target.result_is_exc {
        lines.push_str(&format!("    {invoke};\n"));
        return lines;
    }
    if target.result_is_exc {
        match target.result_type {
            'v' => lines.push_str(&format!(
                "    ::majit_ir::helper_fnaddr::residual_result_void({invoke});\n"
            )),
            'f' | 'L' => lines.push_str(&format!(
                "    ::majit_ir::helper_fnaddr::residual_result_f64({invoke})\n"
            )),
            _ => lines.push_str(&format!(
                "    ::majit_ir::helper_fnaddr::residual_result_i64({invoke})\n"
            )),
        }
        return lines;
    }
    lines.push_str(&format!("    let __r = {invoke};\n"));
    lines.push_str(&format!(
        "    {}\n",
        into_word_expr(&target.ret_rust_type, target.result_type, "__r")
    ));
    lines
}

fn pair_slice_from_words(rust_ty: &str, ptr: &str, len: &str) -> String {
    let is_mut = rust_ty.starts_with("&mut [");
    let item = rust_ty
        .trim_start_matches("&mut [")
        .trim_start_matches("&[")
        .trim_end_matches(']');
    if is_mut {
        format!(
            "if {len} == 0 {{ &mut [] }} else {{ unsafe {{ ::core::slice::from_raw_parts_mut({ptr} as usize as *mut {item}, {len} as usize) }} }}"
        )
    } else {
        format!(
            "if {len} == 0 {{ &[] }} else {{ unsafe {{ ::core::slice::from_raw_parts({ptr} as usize as *const {item}, {len} as usize) }} }}"
        )
    }
}

fn from_word_expr(rust_ty: &str, class: char, word: &str) -> String {
    match class {
        'f' | 'L' => format!(
            "<{rust_ty} as ::majit_ir::helper_fnaddr::ResidualFromF64>::from_residual_f64({word})"
        ),
        _ => format!(
            "unsafe {{ <{rust_ty} as ::majit_ir::helper_fnaddr::ResidualFromI64>::from_residual_i64({word}) }}"
        ),
    }
}

fn into_word_expr(rust_ty: &str, class: char, value: &str) -> String {
    match class {
        'f' | 'L' => format!(
            "<{rust_ty} as ::majit_ir::helper_fnaddr::ResidualIntoF64>::into_residual_f64({value})"
        ),
        _ => format!(
            "<{rust_ty} as ::majit_ir::helper_fnaddr::ResidualIntoI64>::into_residual_i64({value})"
        ),
    }
}

fn shim_ty_to_rust(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    shim_ty(ty, llbc, 0)
}

fn shim_ty(ty: &TyRef, llbc: &Llbc, depth: usize) -> Option<String> {
    if depth > 24 {
        return None;
    }
    let body = tyref_body(ty, llbc)?;
    shim_ty_value(body, llbc, depth)
}

fn tyref_body<'a>(ty: &'a TyRef, llbc: &'a Llbc) -> Option<&'a serde_json::Value> {
    match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

fn shim_ty_value(v: &serde_json::Value, llbc: &Llbc, depth: usize) -> Option<String> {
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return shim_ty_value(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return shim_ty_value(&arr[1], llbc, depth + 1);
    }
    if let Some(lit) = obj.get("Literal").or_else(|| obj.get("Scalar")) {
        let rendered = crate::front::mir::charon_literal_to_ast_string(lit);
        if rendered.starts_with("??") {
            return None;
        }
        return Some(rendered);
    }
    if let Some(r) = obj.get("Ref") {
        let arr = r.as_array()?;
        let inner = shim_ty_value(arr.get(1)?, llbc, depth + 1)?;
        let mutable = arr
            .get(2)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|k| k.eq_ignore_ascii_case("Mut"));
        // `&[T]` / `&mut [T]`: the Slice arm returns `[T]`. Object-pointer
        // items on a shared slice are one `Ref` word (`object_gcarray_param`);
        // other one-word items are the two-word `rii` shape `pair_slice_param`
        // already emits for macros. `&mut [PyObjectRef]` stays unsupported.
        if let Some(item) = inner
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            if is_object_pointer_item(item) {
                if mutable {
                    return None;
                }
                return Some(format!("&{inner}"));
            }
            if is_pair_slice_item(item) {
                return Some(if mutable {
                    format!("&mut {inner}")
                } else {
                    format!("&{inner}")
                });
            }
            return None;
        }
        if is_wide_pointee(&inner) {
            return None;
        }
        return Some(if mutable {
            format!("&mut {inner}")
        } else {
            format!("&{inner}")
        });
    }
    if let Some(elem) = obj
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    {
        let inner = shim_ty_value(elem, llbc, depth + 1)?;
        if !is_pair_slice_item(&inner) && !is_object_pointer_item(&inner) {
            return None;
        }
        return Some(format!("[{inner}]"));
    }
    if let Some(rp) = obj.get("RawPtr") {
        let arr = rp.as_array()?;
        if arr.len() != 2 {
            return None;
        }
        let inner = shim_ty_value(&arr[0], llbc, depth + 1)?;
        if is_wide_pointee(&inner) {
            return None;
        }
        let mutable = arr[1]
            .as_str()
            .is_some_and(|k| k.eq_ignore_ascii_case("Mut"));
        return Some(if mutable {
            format!("*mut {inner}")
        } else {
            format!("*const {inner}")
        });
    }
    if let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) {
        return shim_adt(adt, llbc, depth);
    }
    None
}

fn shim_adt(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> Option<String> {
    if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Tuple") {
        let args = adt_type_args(adt, llbc, depth)?;
        return if args.is_empty() {
            Some("()".to_string())
        } else {
            None
        };
    }
    if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Slice") {
        let args = adt_type_args(adt, llbc, depth)?;
        let inner = args.first()?;
        if !is_pair_slice_item(inner) && !is_object_pointer_item(inner) {
            return None;
        }
        return Some(format!("[{inner}]"));
    }
    if adt.get("builtin").is_some_and(|b| !b.is_null()) {
        return None;
    }
    let def_id = adt.get("id").and_then(serde_json::Value::as_u64)?;
    let td = llbc.type_by_id(def_id)?;
    if let TypeDeclKind::Alias(aliased) = &td.kind
        && let Ok(aliased_ty) = serde_json::from_value::<TyRef>(aliased.clone())
    {
        return shim_ty(&aliased_ty, llbc, depth + 1);
    }
    let name = td.item_meta.name_path();
    let args = adt_type_args(adt, llbc, depth).unwrap_or_default();
    if name.ends_with("::Result") && args.len() == 2 {
        return Some(format!("Result<{}, {}>", args[0], args[1]));
    }
    if name.ends_with("::Option") && args.len() == 1 {
        let inner = &args[0];
        if inner.starts_with("*mut ")
            || inner.starts_with("*const ")
            || is_object_pointer_item(inner)
        {
            return Some(format!("Option<{inner}>"));
        }
        return None;
    }
    if !args.is_empty() {
        return None;
    }
    if name.contains('<') {
        return None;
    }
    if !td.item_meta.attr_info.public {
        return None;
    }
    Some(name)
}

fn adt_type_args(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> Option<Vec<String>> {
    let types = adt
        .get("generics")
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)?;
    types
        .iter()
        .map(|t| {
            let ty: TyRef = serde_json::from_value(t.clone()).ok()?;
            shim_ty(&ty, llbc, depth + 1)
        })
        .collect()
}

fn rust_ty_convertible(ty: &str) -> bool {
    let ty = ty.trim();
    if ty == "()" {
        return true;
    }
    if matches!(
        ty,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "f64"
            | "f32"
    ) {
        return true;
    }
    if ty.starts_with("*mut ") || ty.starts_with("*const ") {
        return !is_wide_pointee(&ty[ty.find(' ').map(|i| i + 1).unwrap_or(0)..]);
    }
    if let Some(inner) = ty.strip_prefix("&mut ") {
        if inner.starts_with('[') && inner.ends_with(']') {
            return is_pair_slice_item(&inner[1..inner.len() - 1]);
        }
        return !is_wide_pointee(inner);
    }
    if let Some(inner) = ty.strip_prefix('&') {
        if inner.starts_with('[') && inner.ends_with(']') {
            let item = &inner[1..inner.len() - 1];
            return is_object_pointer_item(item) || is_pair_slice_item(item);
        }
        return !is_wide_pointee(inner);
    }
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        return inner.starts_with("*mut ")
            || inner.starts_with("*const ")
            || is_object_pointer_item(inner);
    }
    let last = ty.rsplit("::").next();
    last == Some("GcRef") || last == Some("PyObjectRef")
}

fn is_object_pointer_item(ty: &str) -> bool {
    let ty = ty.trim();
    if ty.rsplit("::").next() == Some("PyObjectRef") {
        return true;
    }
    let inner = if let Some(rest) = ty.strip_prefix("*mut ") {
        rest
    } else if let Some(rest) = ty.strip_prefix("*const ") {
        rest
    } else {
        return false;
    };
    inner.rsplit("::").next() == Some("PyObject")
}

fn object_gcarray_item(ty: &str) -> Option<&str> {
    let ty = ty.trim();
    let inner = ty.strip_prefix('&')?;
    if inner.starts_with("mut ") {
        return None;
    }
    let item = inner.strip_prefix('[')?.strip_suffix(']')?;
    is_object_pointer_item(item).then_some(item)
}

fn is_object_gcarray_slice(ty: &str) -> bool {
    object_gcarray_item(ty).is_some()
}

fn is_pair_slice_item(ty: &str) -> bool {
    let ty = ty.trim();
    if is_object_pointer_item(ty) {
        return false;
    }
    ty.starts_with("*mut ")
        || ty.starts_with("*const ")
        || matches!(ty, "i64" | "u64" | "isize" | "usize")
        || ty.rsplit("::").next() == Some("GcRef")
}

fn rust_ty_abi_words(ty: &str) -> usize {
    let ty = ty.trim();
    if is_object_gcarray_slice(ty) {
        1
    } else if ty.starts_with("&[") || ty.starts_with("&mut [") {
        2
    } else {
        1
    }
}

fn is_wide_pointee(ty: &str) -> bool {
    ty == "str"
        || ty.starts_with('[')
        || ty.contains("dyn ")
        || ty.rsplit("::").next() == Some("Wtf8")
        || ty.rsplit("::").next() == Some("Path")
}

fn tyref_result_args(ty: &TyRef, llbc: &Llbc) -> Option<(TyRef, TyRef)> {
    let body = tyref_body(ty, llbc)?;
    let adt = body.get("Adt")?.as_object()?;
    let def_id = adt.get("id").and_then(serde_json::Value::as_u64)?;
    let td = llbc.type_by_id(def_id)?;
    if !td.item_meta.name_path().ends_with("::Result") {
        return None;
    }
    let types = adt.get("generics")?.get("types")?.as_array()?;
    if types.len() != 2 {
        return None;
    }
    let ok = serde_json::from_value(types[0].clone()).ok()?;
    let err = serde_json::from_value(types[1].clone()).ok()?;
    Some((ok, err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::effectinfo::EffectInfo;
    use majit_ir::value::Type;

    fn sample_target() -> ResidualShimTarget {
        ResidualShimTarget {
            path: "pyre_object::function::w_method_new".into(),
            rust_path: "pyre_object::function::w_method_new".into(),
            arg_classes: "rrr".into(),
            result_type: 'r',
            arg_rust_types: vec![
                "*mut pyre_object::pyobject::PyObject".into(),
                "*mut pyre_object::pyobject::PyObject".into(),
                "*mut pyre_object::pyobject::PyObject".into(),
            ],
            ret_rust_type: "*mut pyre_object::pyobject::PyObject".into(),
            result_is_exc: false,
            is_unsafe: false,
            is_generic: false,
            is_method: false,
            public: true,
            crate_name: "pyre_object".into(),
            unsupported_type: false,
        }
    }

    #[test]
    fn collect_keeps_policy_declined_free_functions_without_fnaddr() {
        let path = CallPath::from_segments(["pyre_object", "function", "w_method_new"]);
        let mut declined = HashSet::new();
        declined.insert(path.clone());
        let mut descrs = HashMap::new();
        descrs.insert(
            path.clone(),
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Ref, Type::Ref],
                Type::Ref,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_object::function::w_method_new".into(),
            crate_name: "pyre_object".into(),
            public: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec![
                "pyre_object::pyobject::PyObjectRef".into(),
                "pyre_object::pyobject::PyObjectRef".into(),
                "pyre_object::pyobject::PyObjectRef".into(),
            ]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "pyre_object::pyobject::PyObjectRef".into(),
                is_exc: false,
            }),
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].path, "pyre_object::function::w_method_new");
        assert_eq!(targets[0].arg_classes, "rrr");
        assert_eq!(targets[0].result_type, 'r');
    }

    #[test]
    fn collect_skips_a_bound_fnaddr() {
        let path = CallPath::from_segments(["pyre_object", "function", "w_method_new"]);
        let mut declined = HashSet::new();
        declined.insert(path.clone());
        let mut descrs = HashMap::new();
        descrs.insert(
            path.clone(),
            CallDescriptor::from_signature(&[Type::Ref], Type::Ref, EffectInfo::default()),
        );
        let mut fnaddrs = HashMap::new();
        fnaddrs.insert(path, 1);
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_object::function::w_method_new".into(),
            crate_name: "pyre_object".into(),
            public: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec!["pyre_object::pyobject::PyObjectRef".into()]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "pyre_object::pyobject::PyObjectRef".into(),
                is_exc: false,
            }),
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &fnaddrs, &catalog);
        assert!(targets.is_empty());
    }

    #[test]
    fn generate_emits_extern_c_shim_and_table() {
        let target = sample_target();
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("extern \"C\""));
        assert!(src.contains("pyre_object::function::w_method_new"));
        assert!(src.contains("GENERATED_RESIDUAL_SHIMS"));
        assert!(src.contains("ResidualFromI64"));
        assert!(src.contains("ResidualIntoI64"));
        assert!(src.contains("__a0: i64, __a1: i64, __a2: i64"));
        assert!(src.contains("-> i64"));
    }

    #[test]
    fn collect_object_gcarray_slice_matches_rr_calldescr() {
        let path =
            CallPath::from_segments(["pyre_interpreter", "call", "builtin_code_call_positional"]);
        let mut declined = HashSet::new();
        declined.insert(path.clone());
        let mut descrs = HashMap::new();
        descrs.insert(
            path,
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Ref],
                Type::Ref,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_interpreter::call::builtin_code_call_positional".into(),
            crate_name: "pyre_interpreter".into(),
            public: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec![
                "*mut pyre_object::pyobject::PyObject".into(),
                "&[*mut pyre_object::pyobject::PyObject]".into(),
            ]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "*mut pyre_object::pyobject::PyObject".into(),
                is_exc: true,
            }),
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].unsupported_type);
        assert_eq!(targets[0].arg_classes, "rr");
        assert_eq!(targets[0].arg_rust_types.len(), 2);
    }

    #[test]
    fn generate_rebuilds_an_object_gcarray_slice_from_one_ref_word() {
        let target = ResidualShimTarget {
            path: "pyre_interpreter::call::builtin_code_call_positional".into(),
            rust_path: "pyre_interpreter::call::builtin_code_call_positional".into(),
            arg_classes: "rr".into(),
            result_type: 'r',
            arg_rust_types: vec![
                "*mut pyre_object::pyobject::PyObject".into(),
                "&[*mut pyre_object::pyobject::PyObject]".into(),
            ],
            ret_rust_type: "*mut pyre_object::pyobject::PyObject".into(),
            result_is_exc: true,
            is_unsafe: false,
            is_generic: false,
            is_method: false,
            public: true,
            crate_name: "pyre_interpreter".into(),
            unsupported_type: false,
        };
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("object_gcarray_slice"));
        assert!(src.contains("__a0: i64, __a1: i64"));
        assert!(!src.contains("__a2"));
        assert!(src.contains("residual_result_i64"));
    }

    #[test]
    fn generate_rebuilds_a_pair_slice_from_two_abi_words() {
        let target = ResidualShimTarget {
            path: "example::take_bytes".into(),
            rust_path: "example::take_bytes".into(),
            arg_classes: "rii".into(),
            result_type: 'r',
            arg_rust_types: vec![
                "*mut pyre_object::pyobject::PyObject".into(),
                "&[*mut u8]".into(),
            ],
            ret_rust_type: "*mut pyre_object::pyobject::PyObject".into(),
            result_is_exc: true,
            is_unsafe: false,
            is_generic: false,
            is_method: false,
            public: true,
            crate_name: "example".into(),
            unsupported_type: false,
        };
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("from_raw_parts"));
        assert!(src.contains("__a0: i64, __a1: i64, __a2: i64"));
        assert!(src.contains("residual_result_i64"));
    }

    #[test]
    fn generate_wraps_result_through_residual_error() {
        let mut target = sample_target();
        target.result_is_exc = true;
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("residual_result_i64"));
    }

    #[test]
    fn emit_counts_generic_method_unnameable_and_unsupported() {
        let mut generic = sample_target();
        generic.is_generic = true;
        generic.path = "a::generic".into();
        let mut method = sample_target();
        method.is_method = true;
        method.path = "a::method".into();
        method.rust_path = "a::method".into();
        let mut unnameable = sample_target();
        unnameable.public = false;
        unnameable.path = "a::hidden".into();
        unnameable.rust_path = "a::hidden".into();
        let mut unsupported = sample_target();
        unsupported.unsupported_type = true;
        unsupported.path = "a::bad".into();
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_object".into()]),
            public_modules: BTreeSet::from([
                "pyre_object::function".into(),
                "pyre_object::pyobject".into(),
            ]),
        };
        let (_src, census) = emit_residual_shim_source(
            &[generic, method, unnameable, unsupported, sample_target()],
            &nameability,
        );
        assert_eq!(census.generic, 1);
        assert_eq!(census.unsupported_type, 2);
        assert_eq!(census.unnameable, 1);
        assert_eq!(census.emitted, 1);
    }

    #[test]
    fn nameability_reads_pub_mod_from_source_and_cargo_deps() {
        let dir = tempfile_dir();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\n[dependencies]\npyre-object = { workspace = true }\n",
        )
        .unwrap();
        let src_dir = dir.join("pyre-object").join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("lib.rs"), "pub mod function;\n").unwrap();
        std::fs::write(src_dir.join("function.rs"), "pub fn w_method_new() {}\n").unwrap();
        let cargo = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        let sources = vec![
            src_dir.join("lib.rs").to_string_lossy().into_owned(),
            src_dir.join("function.rs").to_string_lossy().into_owned(),
        ];
        let nameability = Nameability::from_cargo_and_sources(&cargo, &BTreeSet::new(), &sources);
        assert!(nameability.reachable_crates.contains("pyre_object"));
        assert!(nameability.public_modules.contains("pyre_object::function"));
        assert!(nameability.can_name("pyre_object::function::w_method_new", true));
        assert!(!nameability.can_name("pyre_object::function::w_method_new", false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pyre-residual-shim-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn empty_source_is_an_empty_table() {
        let src = empty_residual_shim_source();
        assert!(src.contains("GENERATED_RESIDUAL_SHIMS"));
        assert!(src.contains("&[]"));
    }

    #[test]
    fn rustc_dep_crate_names_skips_build_dependencies() {
        let toml = "[dependencies]\npyre-object = { workspace = true }\n\
                    [build-dependencies]\npyre-module = { workspace = true }\n";
        let names = rustc_dep_crate_names(toml, &BTreeSet::new());
        assert!(names.contains("pyre_object"));
        assert!(!names.contains("pyre_module"));
    }

    #[test]
    fn catalog_converts_object_gcarray_slice_of_object_refs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/llbc/pyre-interpreter.ullbc");
        if !path.exists() {
            return;
        }
        let llbc = majit_charon_reader::Llbc::load(&path).expect("load pyre-interpreter.ullbc");
        let entries = catalog_from_llbc(&llbc);
        let entry = entries
            .iter()
            .find(|entry| {
                entry.path == "pyre_interpreter::call::builtin_code_call_positional"
                    && !entry.is_method
            })
            .expect("builtin_code_call_positional FunDecl");
        assert!(!entry.is_generic);
        assert!(entry.public);
        let args = entry
            .arg_rust_types
            .as_ref()
            .expect("object-gcarray args convert");
        assert_eq!(args.len(), 2);
        assert_eq!(rust_ty_abi_words(&args[0]), 1);
        assert_eq!(rust_ty_abi_words(&args[1]), 1);
        assert!(args[1].starts_with("&["));
        assert!(is_object_gcarray_slice(&args[1]));
        assert!(entry.ret_rust_type.as_ref().is_some_and(|ret| ret.is_exc));
    }
}
