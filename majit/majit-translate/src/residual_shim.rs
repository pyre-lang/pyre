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
    /// Call-site generic type arguments spelled as Rust paths. Empty when
    /// the residual names a monomorphic item. The shim instantiates
    /// `rust_path` with these args (`pop_n::<H>`).
    pub generic_rust_args: Vec<String>,
    /// The FunDecl lives in an impl block and `rust_path` is a nameable
    /// UFCS / inherent path. False rejects an Impl item the catalog could
    /// not rewrite — the LLBC `Impl` path element, not a rendered `"<Impl>"`.
    pub impl_callable: bool,
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
    /// `CallPath` the residual records (`registered_path_for_fun_decl`).
    /// Trait-impl methods register as `Owner::method`, not `<Impl>::method`.
    pub lookup_path: String,
    /// Rust item the generated shim calls: UFCS for a trait-impl method,
    /// `Owner::method` for an inherent impl, `name_path` for a free function.
    pub callable_path: String,
    pub crate_name: String,
    pub public: bool,
    pub is_generic: bool,
    pub is_method: bool,
    pub is_unsafe: bool,
    pub arg_rust_types: Option<Vec<String>>,
    pub ret_rust_type: Option<ShimReturn>,
    /// The FunDecl's `item_meta.name` contains an `Impl` path element and
    /// `callable_path` was rewritten to UFCS or `Owner::method`. False when
    /// the Impl item could not be named in Rust.
    pub impl_callable: bool,
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
        let lookup_path = crate::front::mir::registered_path_for_fun_decl(llbc, fd).canonical_key();
        let (callable_path, impl_callable) = callable_rust_path(fd, llbc);
        let arg_rust_types = fd
            .signature
            .inputs
            .iter()
            .map(|ty| {
                let rust = shim_ty_to_rust(ty, llbc)?;
                rust_ty_catalog_ok(&rust).then_some(rust)
            })
            .collect::<Option<Vec<_>>>();
        let ret_rust_type = shim_return_ty(&fd.signature.output, llbc)
            .filter(|ret| rust_ty_catalog_ok(&ret.rust_type) || ret.rust_type == "()");
        entries.push(LlbcFnCatalogEntry {
            path,
            lookup_path,
            callable_path,
            crate_name: crate_name.clone(),
            public: fd.item_meta.attr_info.public,
            is_generic: decl_is_generic(fd),
            is_method: decl_is_method(fd),
            is_unsafe: fd.signature.is_unsafe,
            arg_rust_types,
            ret_rust_type,
            impl_callable,
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
    if fd.item_meta.has_impl_path_elem() {
        return true;
    }
    matches!(fd.first_arg_local_name().as_deref(), Some("self"))
}

fn shim_return_ty(ty: &TyRef, llbc: &Llbc) -> Option<ShimReturn> {
    if let Some((ok, err)) = tyref_result_args(ty, llbc) {
        let rust_type = shim_ty_to_rust(&ok, llbc)?;
        // `residual_result_*` is bounded on `E: ResidualError`. Emit that
        // path only when this artefact's `trait_impls` table has that impl
        // for `E` — the same fact the generated shim must satisfy to
        // compile, independent of how `E` is spelled.
        if !tyref_implements_residual_error(&err, llbc) {
            return None;
        }
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

/// Whether `ty` is an ADT whose LLBC `trait_impls` include
/// `ResidualError` — the bound [`residual_result_i64`] /
/// [`residual_result_f64`] / [`residual_result_void`] require.
///
/// [`residual_result_i64`]: majit_ir::helper_fnaddr::residual_result_i64
/// [`residual_result_f64`]: majit_ir::helper_fnaddr::residual_result_f64
/// [`residual_result_void`]: majit_ir::helper_fnaddr::residual_result_void
fn tyref_implements_residual_error(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(body) = tyref_body(ty, llbc) else {
        return false;
    };
    let Some(err_id) = tyexpr_nominal_adt_id(body, llbc) else {
        return false;
    };
    llbc.trait_impls_raw()
        .iter()
        .any(|row| residual_error_impl_self_adt(row, llbc) == Some(err_id))
}

fn residual_error_impl_self_adt(row: &serde_json::Value, llbc: &Llbc) -> Option<u64> {
    let impl_trait = row.get("impl_trait")?;
    let trait_id = impl_trait.get("id").and_then(serde_json::Value::as_u64)?;
    let decl = llbc.trait_by_id(trait_id)?;
    // Identity of the trait `residual_result_*` is generic over, the same
    // way `has_explicit_drop_impl` identifies `core::ops::drop::Drop`.
    if decl.item_meta.name_path() != "majit_ir::helper_fnaddr::ResidualError" {
        return None;
    }
    let self_ty = impl_trait
        .get("generics")?
        .get("types")?
        .as_array()?
        .first()?;
    tyexpr_nominal_adt_id(self_ty, llbc)
}

fn tyexpr_nominal_adt_id(v: &serde_json::Value, llbc: &Llbc) -> Option<u64> {
    let mut v = v;
    for _ in 0..24 {
        if let Some(id) = v.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            if let Some(adt) = llbc.dedup_to_adt_def_id(id) {
                return peel_alias_adt_id(adt, llbc);
            }
            v = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = v.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        break;
    }
    let adt = v.get("Adt")?.as_object()?;
    let id = crate::front::mir::type_decl_ref_adt_id(adt)?;
    peel_alias_adt_id(id, llbc)
}

fn peel_alias_adt_id(id: u64, llbc: &Llbc) -> Option<u64> {
    let td = llbc.type_by_id(id)?;
    if let TypeDeclKind::Alias(aliased) = &td.kind {
        return tyexpr_nominal_adt_id(alias_inner(aliased), llbc);
    }
    Some(id)
}

/// Charon stores a type alias as a `TyRef` or as `{"aliased_ty": <TyRef>}`
/// (`TypeDeclKind::Alias`, mir.rs `spill_payloads_of_adt`). `TyRef`'s
/// untagged `Other` arm accepts the wrapper object, so the `aliased_ty`
/// key has to be read before the body is parsed as a `TyRef`.
fn alias_tyref(body: &serde_json::Value) -> Option<TyRef> {
    serde_json::from_value(alias_inner(body).clone()).ok()
}

fn alias_inner(body: &serde_json::Value) -> &serde_json::Value {
    body.get("aliased_ty").unwrap_or(body)
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
        if entry.lookup_path != entry.path {
            by_path.entry(entry.lookup_path.as_str()).or_insert(entry);
        }
        if entry.callable_path != entry.path && entry.callable_path != entry.lookup_path {
            by_path.entry(entry.callable_path.as_str()).or_insert(entry);
        }
    }
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut residual_paths: Vec<&CallPath> = residual_call_descrs.keys().collect();
    residual_paths.extend(
        policy_declined
            .iter()
            .filter(|path| !residual_call_descrs.contains_key(*path)),
    );
    residual_paths.sort_by_key(|path| path.canonical_key());
    residual_paths.dedup_by_key(|path| path.canonical_key());
    for path in residual_paths {
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
        let (template_key, generic_rust_args) = split_instance_path(&key);
        let entry = lookup_catalog(&by_path, catalog, &key)
            .or_else(|| lookup_catalog(&by_path, catalog, &template_key));
        let Some(entry) = entry else {
            continue;
        };
        let rust_path = instantiated_callable_path(&entry.callable_path, &generic_rust_args);
        let subst = |ty: &str| substitute_type_vars(ty, &generic_rust_args);
        let arg_rust_types_subst = entry
            .arg_rust_types
            .as_ref()
            .map(|args| args.iter().map(|ty| subst(ty)).collect::<Vec<_>>());
        let ret_rust_type_subst = entry.ret_rust_type.as_ref().map(|ret| ShimReturn {
            rust_type: subst(&ret.rust_type),
            is_exc: ret.is_exc,
        });
        let unsupported_type = arg_rust_types_subst.is_none()
            || ret_rust_type_subst.is_none()
            || arg_rust_types_subst.as_ref().is_some_and(|args| {
                args.iter()
                    .any(|ty| type_var_placeholder_in(ty) || !rust_ty_convertible(ty))
                    || args.iter().map(|ty| rust_ty_abi_words(ty)).sum::<usize>()
                        != descriptor.arg_classes.chars().filter(|c| *c != 'v').count()
            })
            || ret_rust_type_subst.as_ref().is_some_and(|ret| {
                ret.rust_type != "()"
                    && (type_var_placeholder_in(&ret.rust_type)
                        || !rust_ty_convertible(&ret.rust_type))
            });
        let (arg_rust_types, ret_rust_type, result_is_exc) =
            match (&arg_rust_types_subst, &ret_rust_type_subst) {
                (Some(args), Some(ret)) if !unsupported_type => {
                    (args.clone(), ret.rust_type.clone(), ret.is_exc)
                }
                _ => (Vec::new(), String::new(), false),
            };
        targets.push(ResidualShimTarget {
            path: key,
            rust_path,
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
            generic_rust_args,
            impl_callable: entry.impl_callable,
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
    let mut matches = catalog.iter().filter(|entry| {
        entry.path == key
            || entry.lookup_path == key
            || entry.callable_path == key
            || entry.path.ends_with(&suffix)
            || entry.lookup_path.ends_with(&suffix)
    });
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

/// UFCS `<Self as Trait>::method` for a trait-impl FunDecl, `Owner::method`
/// for an inherent impl, otherwise `name_path`. The bool is whether that
/// spelling is a Rust path the shim crate can name: an `Impl` path element
/// that was not rewritten is not.
fn callable_rust_path(fd: &FunDecl, llbc: &Llbc) -> (String, bool) {
    if let Some(ufcs) = ufcs_callable_path(fd, llbc) {
        return (ufcs, true);
    }
    if let Some(inherent) = crate::front::mir::inherent_callable_rust_path(llbc, fd) {
        return (inherent, true);
    }
    (fd.item_meta.name_path(), !fd.item_meta.has_impl_path_elem())
}

fn ufcs_callable_path(fd: &FunDecl, llbc: &Llbc) -> Option<String> {
    let impl_id = fd.item_meta.trait_impl_id()?;
    let row = llbc.trait_impl_by_id(impl_id)?;
    let impl_trait = row.get("impl_trait")?;
    let trait_id = impl_trait.get("id")?.as_u64()?;
    let trait_path = llbc.trait_by_id(trait_id)?.item_meta.name_path();
    if trait_path.is_empty() {
        return None;
    }
    let self_ty = impl_trait
        .get("generics")?
        .get("types")?
        .as_array()?
        .first()?;
    let self_path = crate::front::clause_spec::spec_type_name(self_ty, llbc, 0, false);
    if self_path.is_empty() || self_path.starts_with('?') {
        return None;
    }
    let name_path = fd.item_meta.name_path();
    let leaf = name_path.rsplit("::").next()?;
    Some(format!("<{self_path} as {trait_path}>::{leaf}"))
}

/// Split a residual path into the FunDecl template and call-site type args.
/// `shared_opcode::pop_n::<pyre_interpreter::pyframe::PyFrame>` →
/// (`shared_opcode::pop_n`, [`pyre_interpreter::pyframe::PyFrame`]).
///
/// `::<` at depth 0 is the callee's instantiation. Nested `::<` inside a
/// type argument is depth > 0 because it sits in `<...>`.
fn split_instance_path(path: &str) -> (String, Vec<String>) {
    let bytes = path.as_bytes();
    let mut depth = 0usize;
    let mut turbofish_at = None;
    let mut i = 0usize;
    while i + 2 < bytes.len() {
        match bytes[i] {
            b'<' => {
                depth += 1;
                i += 1;
            }
            b'>' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b':' if depth == 0 && bytes[i + 1] == b':' && bytes[i + 2] == b'<' => {
                turbofish_at = Some(i);
                i += 3;
            }
            _ => i += 1,
        }
    }
    let Some(at) = turbofish_at else {
        return (path.to_string(), Vec::new());
    };
    let template = path[..at].to_string();
    let rest = &path[at + 3..];
    let args = rest.strip_suffix('>').unwrap_or(rest);
    (template, split_top_level_generic_args(args))
}

fn split_top_level_generic_args(args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (i, ch) in args.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let piece = args[start..i].trim();
                if !piece.is_empty() {
                    out.push(piece.to_string());
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let piece = args[start..].trim();
    if !piece.is_empty() {
        out.push(piece.to_string());
    }
    out
}

/// Residual lookup path that names the monomorphic instance. Regular
/// `graphs_from` keeps the template/spec segments; only the residual
/// funcptr / shim table uses this spelling.
pub fn residual_instance_callpath(path: CallPath, generic_rust_args: &[String]) -> CallPath {
    if generic_rust_args.is_empty() {
        return path;
    }
    let mut segs = path.segments;
    if let Some(last) = segs.last_mut() {
        let leaf = residual_template_leaf(last);
        *last = format!("{}::<{}>", leaf, generic_rust_args.join(", "));
    }
    CallPath { segments: segs }
}

fn residual_template_leaf(last: &str) -> String {
    let (base, _) = split_instance_path(last);
    match base.split_once("__spec_") {
        Some((name, _)) => name.to_string(),
        None => base,
    }
}

fn instantiated_callable_path(callable: &str, generic_rust_args: &[String]) -> String {
    if generic_rust_args.is_empty() {
        return callable.to_string();
    }
    let template = split_instance_path(callable).0;
    format!("{template}::<{}>", generic_rust_args.join(", "))
}

/// Crate and module paths `pyre-jit-trace` can name, from its Cargo.toml
/// dependencies and the `pub mod` tree of collected sources.
#[derive(Debug, Clone, Default)]
pub struct Nameability {
    pub reachable_crates: BTreeSet<String>,
    pub public_modules: BTreeSet<String>,
    /// `pub struct` / `enum` / `type` / `trait` paths from the same sources.
    /// Empty in unit-test fixtures that only populate modules.
    pub public_types: BTreeSet<String>,
}

impl Nameability {
    pub fn from_cargo_and_sources(
        cargo_toml: &str,
        enabled_features: &BTreeSet<String>,
        source_paths: &[String],
    ) -> Self {
        let reachable_crates = rustc_dep_crate_names(cargo_toml, enabled_features);
        let (public_modules, public_types) = public_items_from_sources(source_paths);
        Self {
            reachable_crates,
            public_modules,
            public_types,
        }
    }

    /// The function's whole module path is reachable and the item is public.
    pub fn can_name(&self, rust_path: &str, fn_public: bool) -> bool {
        if !fn_public {
            return false;
        }
        let template = split_instance_path(rust_path).0;
        if let Some((self_ty, trait_ty)) = parse_ufcs_item(&template) {
            return self.can_name_item_path(self_ty) && self.can_name_item_path(trait_ty);
        }
        self.can_name_item_path(&template)
    }

    fn can_name_item_path(&self, rust_path: &str) -> bool {
        let mut segments: Vec<String> = crate::model::split_qualified_path(rust_path)
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        if segments.is_empty() {
            return false;
        }
        segments.pop();
        if segments.is_empty() {
            return false;
        }
        // `module::Type::method`: the type is not a `pub mod`. A snake_case
        // last segment is a module and must itself be public.
        if segments.len() >= 2 {
            let module_prefix = segments.join("::");
            if !self.public_modules.contains(&module_prefix)
                && segment_is_type_name(segments.last().map(String::as_str).unwrap_or(""))
            {
                segments.pop();
            }
        }
        if segments.is_empty() {
            return false;
        }
        let crate_name = segments[0].as_str();
        if matches!(crate_name, "core" | "std" | "alloc") {
            return true;
        }
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

fn segment_is_type_name(seg: &str) -> bool {
    seg.chars()
        .find(|c| c.is_ascii_alphabetic())
        .is_some_and(|c| c.is_ascii_uppercase())
}

/// `<Self as Trait>::method` → (`Self`, `Trait`).
fn parse_ufcs_item(rust_path: &str) -> Option<(&str, &str)> {
    let rest = rust_path.strip_prefix('<')?;
    let (self_ty, after_as) = rest.split_once(" as ")?;
    let (trait_ty, _method) = after_as.split_once(">::")?;
    if self_ty.is_empty() || trait_ty.is_empty() {
        return None;
    }
    Some((self_ty, trait_ty))
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

fn public_items_from_sources(source_paths: &[String]) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut public_modules = BTreeSet::new();
    let mut public_types = BTreeSet::new();
    for path in source_paths {
        let Some(crate_name) = crate_name_from_source_path(path) else {
            continue;
        };
        public_modules.insert(crate_name.clone());
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
            match item {
                syn::Item::Mod(module_item)
                    if matches!(module_item.vis, syn::Visibility::Public(_)) =>
                {
                    public_modules.insert(format!("{parent}::{}", module_item.ident));
                }
                syn::Item::Struct(item) if matches!(item.vis, syn::Visibility::Public(_)) => {
                    public_types.insert(format!("{parent}::{}", item.ident));
                }
                syn::Item::Enum(item) if matches!(item.vis, syn::Visibility::Public(_)) => {
                    public_types.insert(format!("{parent}::{}", item.ident));
                }
                syn::Item::Union(item) if matches!(item.vis, syn::Visibility::Public(_)) => {
                    public_types.insert(format!("{parent}::{}", item.ident));
                }
                syn::Item::Type(item) if matches!(item.vis, syn::Visibility::Public(_)) => {
                    public_types.insert(format!("{parent}::{}", item.ident));
                }
                syn::Item::Trait(item) if matches!(item.vis, syn::Visibility::Public(_)) => {
                    public_types.insert(format!("{parent}::{}", item.ident));
                }
                _ => {}
            }
        }
    }
    (public_modules, public_types)
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
        if target.is_generic && target.generic_rust_args.is_empty() {
            census.generic += 1;
            continue;
        }
        // Call-site args must be Rust types the shim can turbofish.
        // A Charon `TraitType` JSON blob is not one (`spec_type_name`
        // spells an unresolved projection as `?TraitType`).
        if target
            .generic_rust_args
            .iter()
            .any(|ty| !rust_ty_convertible(ty))
        {
            census.generic += 1;
            continue;
        }
        if target.unsupported_type {
            census.unsupported_type += 1;
            continue;
        }
        if !target.impl_callable {
            census.unnameable += 1;
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
    // `residual_word_to_value` / `residual_value_to_word` take `T` from the
    // callee signature (`getfunctionptr` of the function itself). A Ref
    // argument whose TypeDecl lives in another crate (`CodeObject`) is
    // unnameable here and must not refuse the shim. Pair-slice rebuilds
    // and object-GcArray helpers still write the item type.
    target.arg_rust_types.iter().all(|ty| {
        if rust_ty_written_in_shim(ty) {
            rust_type_nameable(ty, nameability)
        } else {
            true
        }
    }) && target
        .generic_rust_args
        .iter()
        .all(|ty| rust_type_nameable(ty, nameability))
}

fn rust_ty_written_in_shim(ty: &str) -> bool {
    rust_ty_abi_words(ty) == 2 || is_object_gcarray_slice(ty)
}

fn rust_type_nameable(ty: &str, nameability: &Nameability) -> bool {
    named_paths_in_type(ty).into_iter().all(|path| {
        if !nameability.can_name(&path, true) {
            return false;
        }
        let crate_name = path.split("::").next().unwrap_or("");
        if matches!(crate_name, "core" | "std" | "alloc") {
            return std_type_path_writable(&path);
        }
        if !nameability.public_types.is_empty()
            && nameability.public_modules.contains(crate_name)
            && !nameability.public_types.contains(&path)
        {
            return false;
        }
        true
    })
}

/// Charon spells `std::hash::RandomState` as `std::hash::random::RandomState`
/// (the defining module is private). A turbofish has to name the public path.
fn std_type_path_writable(path: &str) -> bool {
    let mut segs: Vec<&str> = path.split("::").collect();
    if segs.len() < 2 {
        return false;
    }
    segs.pop();
    let prefix = segs.join("::");
    matches!(
        prefix.as_str(),
        "std"
            | "std::hash"
            | "std::collections"
            | "std::collections::hash_map"
            | "std::vec"
            | "std::boxed"
            | "std::rc"
            | "std::sync"
            | "std::sync::atomic"
            | "std::cell"
            | "std::mem"
            | "std::ptr"
            | "std::ops"
            | "std::cmp"
            | "std::fmt"
            | "std::io"
            | "std::iter"
            | "std::slice"
            | "std::str"
            | "std::marker"
            | "std::any"
            | "std::borrow"
            | "std::alloc"
            | "std::ffi"
            | "std::path"
            | "std::convert"
            | "std::default"
            | "std::clone"
            | "std::option"
            | "std::result"
            | "core"
            | "core::hash"
            | "core::cell"
            | "core::mem"
            | "core::ptr"
            | "core::ops"
            | "core::cmp"
            | "core::fmt"
            | "core::iter"
            | "core::slice"
            | "core::str"
            | "core::marker"
            | "core::any"
            | "core::borrow"
            | "core::alloc"
            | "core::option"
            | "core::result"
            | "core::convert"
            | "core::default"
            | "core::clone"
            | "core::num"
            | "core::sync"
            | "core::sync::atomic"
            | "alloc"
            | "alloc::vec"
            | "alloc::boxed"
            | "alloc::rc"
            | "alloc::sync"
            | "alloc::collections"
    )
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
            lines.push_str(&format!("    let __p{i} = {conv};\n"));
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
        _ => format!("unsafe {{ ::majit_ir::helper_fnaddr::residual_word_to_value({word}) }}"),
    }
}

fn into_word_expr(rust_ty: &str, class: char, value: &str) -> String {
    match class {
        'f' | 'L' => format!(
            "<{rust_ty} as ::majit_ir::helper_fnaddr::ResidualIntoF64>::into_residual_f64({value})"
        ),
        _ => format!("::majit_ir::helper_fnaddr::residual_value_to_word({value})"),
    }
}

fn shim_ty_to_rust(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    shim_ty(ty, llbc, 0, true)
}

fn shim_ty(ty: &TyRef, llbc: &Llbc, depth: usize, as_value: bool) -> Option<String> {
    if depth > 24 {
        return None;
    }
    let body = tyref_body(ty, llbc)?;
    shim_ty_value(body, llbc, depth, as_value)
}

fn tyref_body<'a>(ty: &'a TyRef, llbc: &'a Llbc) -> Option<&'a serde_json::Value> {
    match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

fn shim_ty_value(
    v: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
    as_value: bool,
) -> Option<String> {
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return shim_ty_value(llbc.dedup_body(id)?, llbc, depth + 1, as_value);
    }
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return shim_ty_value(&arr[1], llbc, depth + 1, as_value);
    }
    if let Some(index) = crate::front::clause_spec::type_var_index(v) {
        return Some(format!("__T{index}"));
    }
    if let Some(arr) = obj.get("TraitType").and_then(serde_json::Value::as_array)
        && arr.len() == 3
        && let Some(rust) = shim_trait_type(arr, llbc, depth, as_value)
    {
        return Some(rust);
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
        let inner = shim_ty_value(arr.get(1)?, llbc, depth + 1, false)?;
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
        let inner = shim_ty_value(elem, llbc, depth + 1, true)?;
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
        let inner = shim_ty_value(&arr[0], llbc, depth + 1, false)?;
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
        return shim_adt(adt, llbc, depth, as_value);
    }
    None
}

fn shim_adt(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
    as_value: bool,
) -> Option<String> {
    if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Tuple") {
        let args = adt_type_args(adt, llbc, depth)?;
        return match args.as_slice() {
            [] => Some("()".to_string()),
            [one] => Some(format!("({one},)")),
            _ => Some(format!("({})", args.join(", "))),
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
        && let Some(aliased_ty) = alias_tyref(aliased)
    {
        return shim_ty(&aliased_ty, llbc, depth + 1, as_value);
    }
    let name = td.item_meta.name_path();
    let args = adt_type_args(adt, llbc, depth).unwrap_or_default();
    if name.ends_with("::Result") {
        if args.len() == 2 {
            return Some(format!("Result<{}, {}>", args[0], args[1]));
        }
        return None;
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
    if name.ends_with("::Vec") {
        if args.len() == 1 && vec_item_shim_ok(&args[0]) {
            return Some(format!("Vec<{}>", args[0]));
        }
        return None;
    }
    if name.contains('<') {
        return None;
    }
    if !td.item_meta.attr_info.public {
        return None;
    }
    // `CodeObject<C>` is one ABI word behind a Ref. Spelling the
    // instantiation lets the catalog keep the argument; the shim
    // infers `T` from the callee and does not write this path.
    if !args.is_empty() {
        return Some(format!("{name}<{}>", args.join(", ")));
    }
    // A residual-call argument is one ABI word. A value that fits in a
    // word is copied; a larger public ADT is the address of its storage
    // (RPython: a GcStruct argument is a pointer, a non-GC Struct is
    // `Ptr`). Catalog the VALUE type so `residual_word_to_value` can
    // `ptr::read` it from that address. `rust_ty_abi_words` stays 1.
    Some(name)
}

/// One residual-call ABI word: Charon's layout size/align both fit in
/// `i64`. Missing layout is a refusal, not a name-based guess.
#[cfg(test)]
fn type_decl_fits_abi_word(td: &majit_charon_reader::ullbc::TypeDecl, llbc: &Llbc) -> bool {
    let target = std::env::var("TARGET").unwrap_or_default();
    let Some((size, align)) = td.size_align_for_target(llbc, &target) else {
        return false;
    };
    size.unwrap_or(u64::MAX) <= 8 && align.unwrap_or(u64::MAX) <= 8
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
            shim_ty(&ty, llbc, depth + 1, true)
        })
        .collect()
}

fn rust_ty_catalog_ok(ty: &str) -> bool {
    rust_ty_convertible(ty) || type_var_placeholder_in(ty)
}

fn type_var_placeholder_in(ty: &str) -> bool {
    ty.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .any(is_type_var_placeholder)
}

fn is_type_var_placeholder(ty: &str) -> bool {
    ty.strip_prefix("__T")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

fn substitute_type_vars(ty: &str, args: &[String]) -> String {
    let mut out = ty.to_string();
    for (i, arg) in args.iter().enumerate() {
        out = out.replace(&format!("__T{i}"), arg);
    }
    out
}

fn vec_item_shim_ok(inner: &str) -> bool {
    is_object_pointer_item(inner) || is_type_var_placeholder(inner)
}

fn shim_trait_type(
    arr: &[serde_json::Value],
    llbc: &Llbc,
    depth: usize,
    as_value: bool,
) -> Option<String> {
    let trait_id = traitref_decl_id(&arr[0], llbc)?;
    if let Some(bound) = llbc.unique_trait_assoc_value(trait_id, &arr[1]) {
        let ty: TyRef = serde_json::from_value(bound.clone()).ok()?;
        return shim_ty(&ty, llbc, depth + 1, as_value);
    }
    None
}

fn traitref_decl_id(v: &serde_json::Value, llbc: &Llbc) -> Option<u64> {
    let mut v = v;
    for _ in 0..8 {
        if let Some(id) = v.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = llbc.dedup_trait_body(id)?;
            continue;
        }
        if let Some(arr) = v.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        break;
    }
    v.get("trait_decl_ref")?
        .get("skip_binder")?
        .get("id")?
        .as_u64()
}

fn rust_ty_convertible(ty: &str) -> bool {
    let ty = ty.trim();
    if ty == "()" {
        return true;
    }
    if let Some(inner) = ty.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let inner = inner.trim().trim_end_matches(',');
        if inner.is_empty() {
            return true;
        }
        return split_top_level_generic_args(inner)
            .iter()
            .all(|elem| rust_ty_convertible(elem));
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
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
        return is_object_pointer_item(inner) || rust_ty_convertible(inner);
    }
    let last = ty.rsplit("::").next();
    if last == Some("GcRef") || last == Some("PyObjectRef") {
        return true;
    }
    // Named ADTs reach here after [`shim_adt`]: one ABI word, either the
    // value itself or the address of a larger by-value argument.
    !ty.is_empty() && !ty.contains('<') && !ty.contains('[') && !ty.contains('(')
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
        || matches!(ty, "i64" | "u64" | "isize" | "usize" | "GCREF")
        || ty.rsplit("::").next() == Some("GCREF")
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
            generic_rust_args: Vec::new(),
            impl_callable: true,
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
            lookup_path: "pyre_object::function::w_method_new".into(),
            callable_path: "pyre_object::function::w_method_new".into(),
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
            impl_callable: true,
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
            lookup_path: "pyre_object::function::w_method_new".into(),
            callable_path: "pyre_object::function::w_method_new".into(),
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
            impl_callable: true,
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
        assert!(src.contains("residual_word_to_value"));
        assert!(src.contains("residual_value_to_word"));
        assert!(src.contains("__a0: i64, __a1: i64, __a2: i64"));
        assert!(src.contains("-> i64"));
    }

    #[test]
    fn collect_keeps_a_residual_descr_without_a_policy_decline() {
        let path = CallPath::from_segments(["pyre_object", "function", "w_method_new"]);
        let mut descrs = HashMap::new();
        descrs.insert(
            path,
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Ref, Type::Ref],
                Type::Ref,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_object::function::w_method_new".into(),
            lookup_path: "pyre_object::function::w_method_new".into(),
            callable_path: "pyre_object::function::w_method_new".into(),
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
            impl_callable: true,
        }];
        let targets =
            collect_residual_shim_targets(&HashSet::new(), &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
    }

    #[test]
    fn collect_pair_slice_matches_rii_calldescr() {
        let path = CallPath::from_segments(["majit_rlib", "lltypesystem", "rvec", "ll_slice_copy"]);
        let mut declined = HashSet::new();
        declined.insert(path.clone());
        let mut descrs = HashMap::new();
        descrs.insert(
            path,
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Int, Type::Int],
                Type::Void,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "majit_rlib::lltypesystem::rvec::ll_slice_copy".into(),
            lookup_path: "majit_rlib::lltypesystem::rvec::ll_slice_copy".into(),
            callable_path: "majit_rlib::lltypesystem::rvec::ll_slice_copy".into(),
            crate_name: "majit_rlib".into(),
            public: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec!["*mut u8".into(), "&[i64]".into()]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "()".into(),
                is_exc: false,
            }),
            impl_callable: true,
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].unsupported_type);
        assert_eq!(targets[0].arg_classes, "rii");
        assert_eq!(targets[0].arg_rust_types.len(), 2);
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
            lookup_path: "pyre_interpreter::call::builtin_code_call_positional".into(),
            callable_path: "pyre_interpreter::call::builtin_code_call_positional".into(),
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
            impl_callable: true,
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].unsupported_type);
        assert_eq!(targets[0].arg_classes, "rr");
        assert_eq!(targets[0].arg_rust_types.len(), 2);
        assert_eq!(rust_ty_abi_words(&targets[0].arg_rust_types[1]), 1);
    }

    #[test]
    fn generate_rebuilds_a_pair_slice_from_two_abi_words() {
        let target = ResidualShimTarget {
            path: "majit_rlib::lltypesystem::rvec::ll_slice_copy".into(),
            rust_path: "majit_rlib::lltypesystem::rvec::ll_slice_copy".into(),
            arg_classes: "rii".into(),
            result_type: 'v',
            arg_rust_types: vec!["*mut u8".into(), "&[i64]".into()],
            ret_rust_type: "()".into(),
            result_is_exc: false,
            is_unsafe: false,
            is_generic: false,
            is_method: false,
            public: true,
            crate_name: "majit_rlib".into(),
            unsupported_type: false,
            generic_rust_args: Vec::new(),
            impl_callable: true,
        };
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("from_raw_parts"));
        assert!(src.contains("__a0: i64, __a1: i64, __a2: i64"));
    }

    #[test]
    fn a_two_word_tuple_return_is_convertible() {
        assert!(rust_ty_convertible("(usize, u64)"));
        assert!(rust_ty_convertible("(usize,)"));
        assert!(rust_ty_convertible("()"));
    }

    /// `getfunctionptr` of a translated graph: a residual whose Rust result
    /// is a two-word tuple still gets a shim that calls the function itself.
    /// `residual_value_to_word` packs that tuple as the `'r'` GcStruct word.
    #[test]
    fn collect_keeps_a_tuple_returning_translated_function() {
        let path = CallPath::from_segments([
            "pyre_interpreter",
            "pyopcode",
            "decode_instruction_forward_dispatch",
        ]);
        let mut declined = HashSet::new();
        declined.insert(path.clone());
        let mut descrs = HashMap::new();
        descrs.insert(
            path.clone(),
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Int],
                Type::Ref,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_interpreter::pyopcode::decode_instruction_forward_dispatch".into(),
            lookup_path: "pyre_interpreter::pyopcode::decode_instruction_forward_dispatch".into(),
            callable_path: "pyre_interpreter::pyopcode::decode_instruction_forward_dispatch".into(),
            crate_name: "pyre_interpreter".into(),
            public: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec![
                "&pyre_interpreter::pycode::CodeObject".into(),
                "usize".into(),
            ]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "(usize, u64)".into(),
                is_exc: false,
            }),
            impl_callable: true,
        }];
        let targets = collect_residual_shim_targets(&declined, &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert!(!targets[0].unsupported_type, "{:?}", targets[0]);
        assert_eq!(targets[0].ret_rust_type, "(usize, u64)");
        let src = generate_residual_shim_source(&[&targets[0]]);
        assert!(
            src.contains("decode_instruction_forward_dispatch"),
            "shim calls the function itself: {src}"
        );
        assert!(
            src.contains("residual_value_to_word"),
            "tuple result is packed as one residual word: {src}"
        );
        assert!(
            !src.contains("residual_value_to_word::<"),
            "tuple T is inferred from the callee: {src}"
        );
    }

    /// The shim crate names the function; rustc infers `&CodeObject` from
    /// that signature. The TypeDecl lives in `rustpython_compiler_core`,
    /// which `pyre-jit-trace` does not name.
    #[test]
    fn emit_infers_unnameable_arg_types_from_the_callee() {
        let target = ResidualShimTarget {
            path: "pyre_interpreter::pyopcode::decode_instruction_forward_dispatch".into(),
            rust_path: "pyre_interpreter::pyopcode::decode_instruction_forward_dispatch".into(),
            arg_classes: "ri".into(),
            result_type: 'r',
            arg_rust_types: vec![
                "&rustpython_compiler_core::bytecode::CodeObject".into(),
                "usize".into(),
            ],
            ret_rust_type: "(usize, u64)".into(),
            result_is_exc: false,
            is_unsafe: false,
            is_generic: false,
            is_method: false,
            public: true,
            crate_name: "pyre_interpreter".into(),
            unsupported_type: false,
            generic_rust_args: Vec::new(),
            impl_callable: true,
        };
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_interpreter".into()]),
            public_modules: BTreeSet::from(["pyre_interpreter::pyopcode".into()]),
            public_types: BTreeSet::new(),
        };
        let (src, census) = emit_residual_shim_source(&[target], &nameability);
        assert_eq!(census.emitted, 1, "{census:?} {src}");
        assert_eq!(census.unnameable, 0);
        assert!(src.contains("decode_instruction_forward_dispatch"), "{src}");
        assert!(
            !src.contains("rustpython_compiler_core"),
            "arg T is inferred, not written: {src}"
        );
        assert!(src.contains("residual_value_to_word"), "{src}");
    }

    #[test]
    fn catalog_harvests_decode_instruction_forward_dispatch_from_llbc() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-interpreter.ullbc"
        );
        let llbc = majit_charon_reader::Llbc::load(path)
            .unwrap_or_else(|err| panic!("load {path}: {err}"));
        let catalog = catalog_from_llbc(&llbc);
        let entry = catalog
            .iter()
            .find(|e| {
                e.path.ends_with("::decode_instruction_forward_dispatch")
                    || e.path == "decode_instruction_forward_dispatch"
            })
            .unwrap_or_else(|| panic!("catalog missing decode_instruction_forward_dispatch"));
        assert!(entry.public, "{entry:?}");
        assert!(entry.impl_callable, "{entry:?}");
        let ret = entry
            .ret_rust_type
            .as_ref()
            .unwrap_or_else(|| panic!("ret missing: {entry:?}"));
        assert_eq!(ret.rust_type, "(usize, u64)", "{entry:?}");
        assert!(!ret.is_exc);
        let args = entry
            .arg_rust_types
            .as_ref()
            .unwrap_or_else(|| panic!("args missing: {entry:?}"));
        assert_eq!(args.len(), 2, "{args:?}");
        assert!(
            args[0].contains("CodeObject") && rust_ty_convertible(&args[0]),
            "arg0 {} is not a convertible CodeObject ref",
            args[0]
        );
        assert_eq!(args[1], "usize", "{args:?}");
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
            generic_rust_args: Vec::new(),
            impl_callable: true,
        };
        let src = generate_residual_shim_source(&[&target]);
        assert!(src.contains("object_gcarray_slice"));
        assert!(src.contains("__a0: i64, __a1: i64"));
        assert!(!src.contains("__a2"));
        assert!(src.contains("residual_result_i64"));
        assert!(!src.contains("from_raw_parts"));
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
            public_types: BTreeSet::new(),
        };
        let (_src, census) = emit_residual_shim_source(
            &[generic, method, unnameable, unsupported, sample_target()],
            &nameability,
        );
        assert_eq!(census.generic, 1);
        assert_eq!(census.unsupported_type, 1);
        assert_eq!(census.unnameable, 2);
        assert_eq!(census.emitted, 1);
    }

    #[test]
    fn emit_skips_an_impl_item_the_catalog_could_not_rewrite() {
        let mut target = sample_target();
        target.impl_callable = false;
        target.path = "eval::<Impl>::hidden".into();
        target.rust_path = "pyre_interpreter::eval::<Impl>::hidden".into();
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_interpreter".into()]),
            public_modules: BTreeSet::from(["pyre_interpreter::eval".into()]),
            public_types: BTreeSet::new(),
        };
        let (_src, census) = emit_residual_shim_source(&[target], &nameability);
        assert_eq!(census.unnameable, 1);
        assert_eq!(census.emitted, 0);
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
        assert!(nameability.can_name(
            "<pyre_object::function::Foo as pyre_object::function::Trait>::method",
            true
        ));
        assert!(nameability.can_name(
            "pyre_object::function::pop_n::<pyre_object::pyobject::PyObject>",
            true
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_instantiates_a_generic_callee_from_call_site_args() {
        let mut generic = sample_target();
        generic.is_generic = true;
        generic.generic_rust_args = vec!["pyre_object::pyobject::PyObject".into()];
        generic.rust_path =
            "pyre_object::function::w_method_new::<pyre_object::pyobject::PyObject>".into();
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_object".into()]),
            public_modules: BTreeSet::from([
                "pyre_object::function".into(),
                "pyre_object::pyobject".into(),
            ]),
            public_types: BTreeSet::new(),
        };
        let (src, census) = emit_residual_shim_source(&[generic], &nameability);
        assert_eq!(census.generic, 0);
        assert_eq!(census.emitted, 1);
        assert!(src.contains("w_method_new::<pyre_object::pyobject::PyObject>"));
    }

    /// A Charon `TraitType` blob is not a turbofish argument. The shim
    /// table skips it (`census.generic`) rather than emitting invalid Rust.
    #[test]
    fn emit_skips_a_charon_traittype_generic_arg() {
        let mut generic = sample_target();
        generic.is_generic = true;
        generic.path = "rordereddict_entries::entries_item_ptr".into();
        generic.rust_path = "pyre_object::rordereddict_entries::entries_item_ptr".into();
        generic.generic_rust_args = vec![
            r#"{"TraitType":[{"kind":{"Clause":{"Bound":[0,0]}}},0,{}]}"#.into(),
            "()".into(),
        ];
        generic.arg_classes = "i".into();
        generic.arg_rust_types = vec!["i64".into()];
        generic.result_type = 'i';
        generic.ret_rust_type = "i64".into();
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_object".into()]),
            public_modules: BTreeSet::from(["pyre_object::rordereddict_entries".into()]),
            public_types: BTreeSet::new(),
        };
        let (src, census) = emit_residual_shim_source(&[generic], &nameability);
        assert_eq!(census.generic, 1);
        assert_eq!(census.emitted, 0);
        assert!(!src.contains("TraitType"));
        assert!(!src.contains("entries_item_ptr::<"));
    }

    #[test]
    fn emit_skips_a_private_std_module_and_a_private_struct_in_turbofish() {
        let mut random_state = sample_target();
        random_state.path = "pyre_object::rordereddict::ll_dict_del_trampoline".into();
        random_state.rust_path = "pyre_object::rordereddict::ll_dict_del_trampoline::<pyre_object::identitydict::IdentityKey, *mut pyre_object::pyobject::PyObject, std::hash::random::RandomState>".into();
        random_state.generic_rust_args = vec![
            "pyre_object::identitydict::IdentityKey".into(),
            "*mut pyre_object::pyobject::PyObject".into(),
            "std::hash::random::RandomState".into(),
        ];
        random_state.is_generic = true;
        let mut private_struct = sample_target();
        private_struct.path = "pyre_object::rordereddict::ll_dict_lookup_trampoline".into();
        private_struct.rust_path = "pyre_object::rordereddict::ll_dict_lookup_trampoline::<pyre_object::dictmultiobject::StrLookupKey>".into();
        private_struct.generic_rust_args =
            vec!["pyre_object::dictmultiobject::StrLookupKey".into()];
        private_struct.is_generic = true;
        let nameability = Nameability {
            reachable_crates: BTreeSet::from(["pyre_object".into()]),
            public_modules: BTreeSet::from([
                "pyre_object".into(),
                "pyre_object::rordereddict".into(),
                "pyre_object::identitydict".into(),
                "pyre_object::pyobject".into(),
                "pyre_object::dictmultiobject".into(),
            ]),
            public_types: BTreeSet::from([
                "pyre_object::identitydict::IdentityKey".into(),
                "pyre_object::pyobject::PyObject".into(),
            ]),
        };
        let (_src, census) =
            emit_residual_shim_source(&[random_state, private_struct], &nameability);
        assert_eq!(census.emitted, 0, "{census:?}");
        assert_eq!(census.unnameable, 2);
        assert!(!std_type_path_writable("std::hash::random::RandomState"));
        assert!(std_type_path_writable("std::hash::RandomState"));
        assert!(std_type_path_writable("core::hash::BuildHasherDefault"));
    }

    #[test]
    fn collect_matches_a_trait_impl_method_by_registered_path() {
        let path = CallPath::for_impl_method("pyframe::PyFrame", "call_kw");
        let mut descrs = HashMap::new();
        descrs.insert(
            path.clone(),
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Int],
                Type::Void,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_interpreter::eval::<Impl>::call_kw".into(),
            lookup_path: path.canonical_key(),
            callable_path:
                "<pyre_interpreter::pyframe::PyFrame as pyre_interpreter::pyopcode::OpcodeStepExecutor>::call_kw"
                    .into(),
            crate_name: "pyre_interpreter".into(),
            public: true,
            is_generic: false,
            is_method: true,
            is_unsafe: false,
            arg_rust_types: Some(vec![
                "&mut pyre_interpreter::pyframe::PyFrame".into(),
                "usize".into(),
            ]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "()".into(),
                is_exc: true,
            }),
            impl_callable: true,
        }];
        let targets =
            collect_residual_shim_targets(&HashSet::new(), &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].path, "pyframe::PyFrame::call_kw");
        assert!(targets[0].rust_path.contains(" as "));
        assert!(targets[0].rust_path.ends_with("::call_kw"));
        assert_eq!(targets[0].arg_classes, "ri");
        assert!(!targets[0].unsupported_type);
    }

    #[test]
    fn collect_names_a_generic_instance_from_turbofish() {
        let path = CallPath::from_segments([
            "shared_opcode",
            "pop_n::<pyre_interpreter::pyframe::PyFrame>",
        ]);
        let mut descrs = HashMap::new();
        descrs.insert(
            path.clone(),
            CallDescriptor::from_signature(
                &[Type::Ref, Type::Int],
                Type::Ref,
                EffectInfo::default(),
            ),
        );
        let catalog = vec![LlbcFnCatalogEntry {
            path: "pyre_interpreter::shared_opcode::pop_n".into(),
            lookup_path: "shared_opcode::pop_n".into(),
            callable_path: "pyre_interpreter::shared_opcode::pop_n".into(),
            crate_name: "pyre_interpreter".into(),
            public: true,
            is_generic: true,
            is_method: false,
            is_unsafe: false,
            arg_rust_types: Some(vec![
                "&mut pyre_interpreter::pyframe::PyFrame".into(),
                "usize".into(),
            ]),
            ret_rust_type: Some(ShimReturn {
                rust_type: "Vec<*mut pyre_object::pyobject::PyObject>".into(),
                is_exc: true,
            }),
            impl_callable: true,
        }];
        let targets =
            collect_residual_shim_targets(&HashSet::new(), &descrs, &HashMap::new(), &catalog);
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].generic_rust_args,
            vec!["pyre_interpreter::pyframe::PyFrame".to_string()]
        );
        assert_eq!(
            targets[0].rust_path,
            "pyre_interpreter::shared_opcode::pop_n::<pyre_interpreter::pyframe::PyFrame>"
        );
        assert!(targets[0].is_generic);
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
        // `Result<_, E>` is `is_exc` only when this artefact implements
        // `ResidualError` for `E` (`runtime_ops.rs` for the interpreter
        // carrier). A `Result` whose `E` has no such impl is refused.
        assert!(entry.ret_rust_type.as_ref().is_some_and(|ret| ret.is_exc));
        assert!(
            llbc.trait_impls_raw()
                .iter()
                .any(|row| { residual_error_impl_self_adt(row, &llbc).is_some() }),
            "pyre-interpreter LLBC must carry ResidualError for the carrier"
        );
        let pop_n = entries
            .iter()
            .find(|entry| entry.path.ends_with("::pop_n") && !entry.is_method);
        if let Some(entry) = pop_n {
            assert!(entry.is_generic);
            assert!(
                entry.arg_rust_types.is_some(),
                "TypeVar args catalog as placeholders"
            );
        }
        let make_pyerror = entries
            .iter()
            .find(|entry| entry.path.ends_with("::make_pyerror") && !entry.is_method);
        if let Some(entry) = make_pyerror {
            let args = entry
                .arg_rust_types
                .as_ref()
                .expect("DisplayMessage is catalogued as a by-value type");
            let message = args.iter().find(|ty| ty.ends_with("::DisplayMessage"));
            let message = message.expect("make_pyerror takes DisplayMessage");
            assert_eq!(
                rust_ty_abi_words(message),
                1,
                "by-value DisplayMessage is one address word"
            );
            let td = llbc
                .iter_type_decls()
                .find(|td| td.item_meta.name_path().ends_with("::DisplayMessage"))
                .expect("DisplayMessage TypeDecl");
            assert!(
                !type_decl_fits_abi_word(td, &llbc),
                "DisplayMessage is larger than one ABI word; the shim ptr::reads it"
            );
        }
    }
}
