//! Harvest JIT-hint markers from the ullbc surrogate consts the
//! `majit_macros` proc-macros emit (`_elidable_function_<NAME>`,
//! `_jit_look_inside_<NAME>`, `_jit_loop_invariant_<NAME>`,
//! `_jit_unroll_safe_<NAME>`, `_not_rpython_<NAME>`, `oopspec_<NAME>`,
//! `_gctransformer_hint_close_stack_<NAME>`,
//! `_call_aroundstate_target_<NAME>`,
//! `_random_effects_on_gcobjs_<NAME>`,
//! `_callable_elidable_function_<NAME>`,
//! `_funcptr_canraise_false_<NAME>`,
//! `_jit_llop_<NAME>`,
//! `_MAJIT_MAY_FORCE_<NAME>`,
//! `_annspecialcase_ll_<NAME>`).
//!
//! The source attribute (`#[elidable]` / `#[dont_look_inside]` / …) is
//! consumed by the proc-macro at expansion time and does NOT survive in
//! Charon's `attr_info`.  The macro instead leaves these `#[doc(hidden)]`
//! marker consts next to the user function, and Charon extracts them
//! into `global_decls`.  Reading them back is the analog of RPython's
//! translator reading `func._elidable_function_` off the function
//! object.
//!
//! The harvested map is keyed and ordered so that a function's header
//! (`front::mir` `SemanticFunctionHeader::new`) can apply the hints to it
//! order-exact.

use majit_charon_reader::{
    Llbc,
    ullbc::{FunDecl, GlobalDecl, Operand, PlaceKind, Rvalue, StmtKind, TypeDeclKind},
};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{HOST_ENV, HostObject};

/// Marker-const name prefix → the JIT hint strings it implies.  The
/// user function's leaf name is the const leaf with the prefix stripped.
///
/// This is the inverse of `majit_macros::rpython_attribute_const_for`.
/// `_jit_look_inside_` is handled separately because the same marker
/// prefix carries a bool value (`true` = `jit_look_inside`, `false` =
/// `dont_look_inside`).
const CONST_PREFIX_HINTS: &[(&str, &[&str])] = &[
    ("_elidable_function_", &["elidable"]),
    // Keep this before `_always_inline_`: the best-effort marker retains the
    // common prefix and would otherwise be attributed to `try_<function>`.
    ("_always_inline_try_", &["always_inline_try"]),
    ("_always_inline_", &["always_inline"]),
    ("_jit_elidable_cannot_raise_", &["elidable_cannot_raise"]),
    ("_jit_elidable_or_memerror_", &["elidable_or_memerror"]),
    ("_jit_cannot_raise_", &["cannot_raise"]),
    ("_jit_loop_invariant_", &["loopinvariant"]),
    // `#[jit_may_force]` — harvested like `_jit_cannot_raise_`. RPython
    // has no may-force attribute; `graphanalyze.py` `analyze` returns
    // `top_result()` for an `indirect_call` whose graph set is None, and
    // the walker residualizes the surface.
    ("_MAJIT_MAY_FORCE_", &["may_force"]),
    ("_jit_unroll_safe_", &["unroll_safe"]),
    // objectmodel.py `not_rpython` sets `_not_rpython_ = True`;
    // flowspace/objspace.py:21 rejects the function before graph building.
    ("_not_rpython_", &["not_rpython"]),
    // objectmodel.py specialize.memo's function attribute, consumed by
    // Bookkeeper.newfuncdesc (not an elidable/residual JIT-policy flag).
    ("_annspecialcase_memo_", &["specialize:memo"]),
    // objectmodel.py specialize.call_location's function attribute.
    (
        "_annspecialcase_call_location_",
        &["specialize:call_location"],
    ),
    // objectmodel.py specialize.ll (`rgc.py` `ll_arraycopy`).
    ("_annspecialcase_ll_", &["specialize:ll"]),
    // rffi.py `call_external_function._gctransformer_hint_close_stack_ = True`
    // (`call.py` `guess_call_kind` / `get_jitcode` residual gate).
    ("_gctransformer_hint_close_stack_", &["close_stack"]),
];

/// Build a `{crate_stripped_fn_path → sorted-deduped hints}` map from
/// the marker consts present in `llbcs`.
pub fn harvest_hints_from_llbcs<L: Borrow<Llbc>>(llbcs: &[L]) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for llbc in llbcs {
        let llbc = llbc.borrow();
        let function_paths: HashSet<String> = llbc
            .iter_local_fns()
            .map(|fd| strip_crate_prefix(&fd.item_meta.name_path()))
            .collect();
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            // Charon emits a crate-less duplicate Ident for some
            // body-local markers (`_elidable_function_invert` next to
            // `rbigint::RBigInt::invert::_elidable_function_invert`).
            // `call.py` reads `_elidable_function_` off that funcobj; a
            // single-segment name is not a funcobj path. Harvesting it
            // as the bare leaf lets `apply_harvested_funcobj_hints`
            // stamp `DecoratorAttr::Elidable` onto every free function
            // registered under that leaf (`descroperation::invert` via
            // `free_function_alias_paths`), so `policy.py
            // _reject_function` / `look_inside_graph` residualizes it
            // and `graphs_from` never enters `invert_inner` /
            // `_int_invert`. Crate-root markers keep the crate Ident
            // (`majit_rlib::_elidable_function_foo`) and still harvest.
            if !path.contains("::") {
                continue;
            }
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            if let Some(fn_leaf) = leaf.strip_prefix("_jit_look_inside_") {
                if should_skip_generated_elidable_helper(fn_leaf) {
                    continue;
                }
                // The marker const exists (its name routed control here),
                // so its bool initializer must decode.  Mirror policy.py:56
                // honouring the exact `_jit_look_inside_` value rather than
                // silently coercing an undecodable marker to
                // `dont_look_inside`: a `None` here means the marker is
                // present but its literal bool could not be read, which is a
                // decoder/encoding fault, not a `False`.
                let look_inside = global_marker_bool(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_jit_look_inside_ marker `{path}` has an undecodable bool \
                         initializer; init={:?} value={}",
                        marker_init_fun_id(gd),
                        gd.const_expr().map(|v| v.to_string()).unwrap_or_default()
                    )
                });
                let hint = if look_inside {
                    "jit_look_inside"
                } else {
                    "dont_look_inside"
                };
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "_jit_look_inside_", &function_paths),
                    hint,
                );
                continue;
            }
            // `#[jit_may_force]` body-local `const _MAJIT_MAY_FORCE` (no
            // function-name suffix). The suffixed sibling
            // `_MAJIT_MAY_FORCE_<fn>` is harvested via `CONST_PREFIX_HINTS`.
            if leaf == "_MAJIT_MAY_FORCE" {
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "_MAJIT_MAY_FORCE", &function_paths),
                    "may_force",
                );
                continue;
            }
            // `#[oopspec("spec")]` emits `oopspec_<NAME>: &'static str =
            // "spec"` (majit_macros::oopspec, rlib/jit.py `func.oopspec
            // = spec`).  Unlike the fixed-hint markers, the payload is the
            // const's *value*, so decode the string literal and emit a
            // companion `oopspec:<spec>` hint that `lib.rs` consumes via
            // `CallControl::mark_oopspec` — which `guess_call_kind` then
            // classifies as `CallKind::Builtin` (call.py).
            if leaf.starts_with("oopspec_") {
                // Fail fast like the `_jit_look_inside_` arm above: the macro
                // emits a literal string, so an undecodable initializer is a
                // marker encoding/decoder drift, not an absent spec. Silently
                // dropping the hint would quietly disable oopspec lowering for
                // the function path.
                let spec = global_marker_str(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "oopspec marker `{path}` has an undecodable string \
                         initializer; this signals a marker encoding/decoder drift"
                    )
                });
                // Only the spec string is harvested, not a companion
                // `oopspec_argnames`.  The support layer's `parse_oopspec`
                // splits the spec's argument list positionally, which is exact
                // for the current specs (every arg is a bare parameter name in
                // declaration order).  A spec that referenced its parameters
                // out of order or by expression would need the argnames map to
                // resolve names→positions; emit it here when such a spec lands.
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "oopspec_", &function_paths),
                    &format!("oopspec:{spec}"),
                );
                continue;
            }
            // `#[jit_llop("opname")]` emits `_jit_llop_<NAME>: &'static str =
            // "opname"`. The declaration stands for the upstream callee
            // identity (`ExtRegistryEntry._about_`); the opname names the
            // llop its calls specialize to.
            if leaf.starts_with("_jit_llop_") {
                let opname = global_marker_str(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_jit_llop_ marker `{path}` has an undecodable string \
                         initializer; this signals a marker encoding/decoder drift"
                    )
                });
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "_jit_llop_", &function_paths),
                    &format!("llop:{opname}"),
                );
                continue;
            }
            // `#[call_aroundstate_target(funcptr = .., save_err = N)]` emits
            // `_call_aroundstate_target_<fn> = (funcptr, save_err)`
            // (`rffi.py` `call_external_function._call_aroundstate_target_`).
            // The payload is the const's value: the extern funcptr's path
            // and/or `link_name`, plus `save_err`. `call.py` `getcalldescr`
            // reads that pair into `EffectInfo.call_release_gil_target`.
            if leaf.starts_with("_call_aroundstate_target_") {
                let (identity, save_err) =
                    decode_aroundstate_marker(llbc, gd).unwrap_or_else(|| {
                        panic!(
                            "_call_aroundstate_target_ marker `{path}` has an \
                         undecodable initializer; the macro emits a \
                         (funcptr, save_err) tuple, so this signals a Charon \
                         encoding change"
                        )
                    });
                let fn_path =
                    marker_path_to_fn_path(&path, "_call_aroundstate_target_", &function_paths);
                push_hint(&mut out, fn_path.clone(), "aroundstate");
                push_hint(
                    &mut out,
                    fn_path,
                    &format!("aroundstate_target:{save_err}:{identity}"),
                );
                continue;
            }
            // rffi.py llexternal stores `random_effects_on_gcobjs` on the
            // graphless C functionptr (`effectinfo.py RandomEffectsAnalyzer`).
            // The marker initializer is that funcptr (`*const ()`); harvest
            // the FunDecl the same way `_call_aroundstate_target_` does.
            if leaf.starts_with("_random_effects_on_gcobjs_") {
                let fd = decode_funcptr_marker_fun_decl(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_random_effects_on_gcobjs_ marker `{path}` has an \
                         undecodable initializer; the macro emits a funcptr, \
                         so this signals a Charon encoding change"
                    )
                });
                push_hint(&mut out, funcptr_marker_call_path_key(fd), "gc_effects");
                continue;
            }
            // rffi.py llexternal sets `_callable._elidable_function_` on the graphless C
            // functionptr; `call.py getcalldescr` reads it off the funcobj.
            if leaf.starts_with("_callable_elidable_function_") {
                let fd = decode_funcptr_marker_fun_decl(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_callable_elidable_function_ marker `{path}` has an \
                         undecodable initializer; the macro emits a funcptr, \
                         so this signals a Charon encoding change"
                    )
                });
                push_hint(&mut out, funcptr_marker_call_path_key(fd), "elidable");
                continue;
            }
            // rffi.py llexternal creates the C function with `canraise=False`;
            // `canraise.py analyze_external_call` reads it off the funcobj.
            if leaf.starts_with("_funcptr_canraise_false_") {
                let fd = decode_funcptr_marker_fun_decl(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_funcptr_canraise_false_ marker `{path}` has an \
                         undecodable initializer; the macro emits a funcptr, \
                         so this signals a Charon encoding change"
                    )
                });
                push_hint(
                    &mut out,
                    funcptr_marker_call_path_key(fd),
                    "funcptr_canraise_false",
                );
                continue;
            }
            // `#[specialize_argtype(N, …)]` emits the body-local
            // `_annspecialcase_argtype_<fn> = "specialize:argtype(0,1)"`
            // (`objectmodel.py` `_Specialize._wrap`). The const value is
            // the whole hint (`specialize.py` `specialize_argtype`); do
            // not prefix it again. Multi-index form is already the
            // comma-joined payload `parse_specializer_directive` splits.
            if leaf.starts_with("_annspecialcase_argtype_") {
                let hint = global_marker_str(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "specialize:argtype marker `{path}` has an undecodable \
                         string initializer; this signals a marker encoding/decoder drift"
                    )
                });
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "_annspecialcase_argtype_", &function_paths),
                    &hint,
                );
                continue;
            }
            for (prefix, hints) in CONST_PREFIX_HINTS {
                if leaf.starts_with(prefix) {
                    // `elidable_promote` renames the decorated body to
                    // `_orig_<name>_unlikely_name` and that body carries
                    // `_elidable_function_`: `jit.py elidable_promote` calls
                    // `elidable(func)` on the original, and the promoting
                    // wrapper `f` carries no marker.  The body is harvested
                    // like any other elidable function, so its call sites
                    // stay residual `CALL_PURE`s.
                    let key = marker_path_to_fn_path(&path, prefix, &function_paths);
                    for hint in *hints {
                        push_hint(&mut out, key.clone(), hint);
                    }
                    break;
                }
            }
        }
        // Residualize the generated `type_object()` accessors.  The body is a
        // `OnceLock<usize>` type-object cache — unliftable host plumbing whose
        // lift fails at the `CELL` read and poisons every caller.  Stamping
        // `dont_look_inside` prefills a signature-only stub and lets the
        // codewriter emit a residual call to the accessor's registered C ABI
        // address.  Gated on the exact accessor signature — leaf `type_object`,
        // no inputs, `*mut PyObject` return — never a stray same-named
        // function.  Every generator of such an accessor registers its address
        // for the consumer's `fnaddr` table, on every target, so the stamped
        // set equals the published set and a stamped accessor always resolves
        // to a real address.
        for fd in llbc.iter_local_fns() {
            let path = strip_crate_prefix(&fd.item_meta.name_path());
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            if leaf == "type_object"
                && fd.signature.inputs.is_empty()
                && crate::front::mir::output_type_is_objectptr(&fd.signature.output, llbc)
            {
                push_hint(&mut out, path, "dont_look_inside");
            }
        }
    }
    for v in out.values_mut() {
        v.sort();
        v.dedup();
    }
    out
}

/// Marker-const prefix `#[jit_immutable_fields]` leaves next to the
/// annotated struct.  The const leaf with the prefix stripped is the
/// struct name; the value is the comma-joined entry list.
const IMMUTABLE_FIELDS_PREFIX: &str = "_immutable_fields_";

/// Build a `{struct_name → [(field_name, rank)]}` map from the
/// `_immutable_fields_<Struct>` marker consts present in `llbcs`.
///
/// RPython reads `cls._immutable_fields_` off the class object
/// (`rclass.py _parse_field_list`); the marker const is the
/// ullbc-visible carrier for the same declaration, since the
/// proc-macro attribute is consumed at expansion time and never
/// reaches Charon's `attr_info`.
///
/// Both the bare struct name and the crate-stripped qualified path are
/// inserted, mirroring `SemanticProgram.struct_fields`: the codewriter
/// looks the declaration up by the analyzer's `owner_root`, which is
/// the bare use-site identifier today and the qualified path once the
/// use-import resolver lands.
pub fn harvest_immutable_fields_from_llbcs(
    llbcs: &[Llbc],
) -> HashMap<String, Vec<(String, crate::model::ImmutableRank)>> {
    let mut out: HashMap<String, Vec<(String, crate::model::ImmutableRank)>> = HashMap::new();
    for llbc in llbcs {
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            let Some(struct_name) = leaf.strip_prefix(IMMUTABLE_FIELDS_PREFIX) else {
                continue;
            };
            // Fail fast like the `oopspec_` arm: the macro emits a literal
            // string, so an undecodable initializer is a marker
            // encoding/decoder drift, not an absent declaration.  Silently
            // dropping it would quietly demote every listed field back to
            // mutable.
            let joined = global_marker_str(llbc, gd).unwrap_or_else(|| {
                panic!(
                    "_immutable_fields_ marker `{path}` has an undecodable string \
                     initializer; this signals a marker encoding/decoder drift"
                )
            });
            let fields = parse_immutable_entry_list(&joined);
            if fields.is_empty() {
                continue;
            }
            let qualified = {
                let stripped = strip_crate_prefix(&path);
                match stripped.rsplit_once("::") {
                    Some((module, _)) => format!("{module}::{struct_name}"),
                    None => struct_name.to_string(),
                }
            };
            for key in [struct_name.to_string(), qualified] {
                out.entry(key).or_insert_with(|| fields.clone());
            }
        }
    }
    out
}

/// Marker-const prefix for `_alloc_flavor_ = 'raw'` (`rclass.py
/// getgcflavor` / `classdesc.py ClassDesc.get_param`).
const ALLOC_FLAVOR_PREFIX: &str = "_alloc_flavor_";

/// Build the set of structs whose `_alloc_flavor_<Struct>` marker is
/// `"raw"`.
///
/// Each marker is resolved to the struct `TypeDecl` in the same `Llbc`
/// whose name-path leaf (generics stripped) equals the marker's struct
/// name. One candidate, or the unique candidate in the marker's own
/// module, is the identity; several distinct candidates panic as a
/// marker encoding drift. The stored string is the class intern key
/// `canonical_struct_name` of the crate-stripped TypeDecl path, the
/// same key `Bookkeeper::stamp_raw_alloc_flavor` receives.
/// `rclass.py getgcflavor` / `classdesc.py ClassDesc.get_param` read
/// `_alloc_flavor_` off that classdesc.
pub fn harvest_alloc_flavor_raw_from_llbcs(llbcs: &[Llbc]) -> HashSet<String> {
    let mut out = HashSet::new();
    for llbc in llbcs {
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            let Some(struct_name) = leaf.strip_prefix(ALLOC_FLAVOR_PREFIX) else {
                continue;
            };
            let flavor = global_marker_str(llbc, gd).unwrap_or_else(|| {
                panic!(
                    "_alloc_flavor_ marker `{path}` has an undecodable string \
                     initializer; this signals a marker encoding/decoder drift"
                )
            });
            if flavor != "raw" {
                continue;
            }
            out.insert(resolve_alloc_flavor_raw_identity(llbc, &path, struct_name));
        }
    }
    out
}

/// Intern key for a TypeDecl path: `canonical_struct_name` of the
/// crate-stripped name, matching `Bookkeeper::intern_class_by_qualname`
/// / `intern_class_by_qualname_with_bases`.
pub(crate) fn alloc_flavor_class_intern_key(type_name_path: &str) -> String {
    majit_ir::descr::canonical_struct_name(&strip_crate_prefix(type_name_path))
}

fn path_leaf_stripped(path: &str) -> &str {
    let leaf = path.rsplit("::").next().unwrap_or(path);
    leaf.split('<').next().unwrap_or(leaf)
}

fn path_module(stripped: &str) -> Option<&str> {
    stripped.rsplit_once("::").map(|(module, _)| module)
}

/// Resolve `_alloc_flavor_<Struct>` to one intern-key identity in `llbc`.
fn resolve_alloc_flavor_raw_identity(llbc: &Llbc, marker_path: &str, struct_name: &str) -> String {
    let marker_stripped = strip_crate_prefix(marker_path);
    let marker_module = path_module(&marker_stripped);
    let mut candidates: Vec<(String, String)> = Vec::new();
    for td in llbc.iter_type_decls() {
        // A Rust enum is the class hierarchy `rclass.py` flavors as one:
        // its variants inherit `_alloc_flavor_` from the base
        // (`ClassDesc.get_param` reads the attribute through the MRO).
        if !matches!(&td.kind, TypeDeclKind::Struct(_) | TypeDeclKind::Enum(_)) {
            continue;
        }
        let td_path = td.item_meta.name_path();
        if path_leaf_stripped(&td_path) != struct_name {
            continue;
        }
        let identity = alloc_flavor_class_intern_key(&td_path);
        if !candidates.iter().any(|(id, _)| id == &identity) {
            candidates.push((identity, td_path));
        }
    }
    if candidates.len() == 1 {
        return candidates.pop().expect("len == 1").0;
    }
    let mut in_module: Vec<(String, String)> = Vec::new();
    for (identity, td_path) in &candidates {
        let stripped = strip_crate_prefix(td_path);
        if path_module(&stripped) == marker_module
            && !in_module.iter().any(|(id, _)| id == identity)
        {
            in_module.push((identity.clone(), td_path.clone()));
        }
    }
    if in_module.len() == 1 {
        return in_module.pop().expect("len == 1").0;
    }
    let named: Vec<&str> = candidates.iter().map(|(_, path)| path.as_str()).collect();
    panic!(
        "_alloc_flavor_ marker `{marker_path}` does not resolve to a unique \
         struct or enum TypeDecl (candidates: {named:?}); this signals a marker \
         encoding/decoder drift"
    );
}

/// Marker-const prefix for an RPython exception class hierarchy.
///
/// `argument.py` `class ArgErr(Exception)` and its subclasses are one
/// Rust enum in the interpreter. The const `_exception_class_<Enum>`
/// declares that enum. The base class is the declaration intern key
/// (`exception_class_key`, subclass of the host `Exception`) and each
/// variant is `{enum_path}::{variant}`. The const value is a
/// whitespace-separated list of nesting overrides
/// `Variant(ParentVariant)`; a variant that is absent derives from the
/// base. An empty string is a flat hierarchy.
const EXCEPTION_CLASS_PREFIX: &str = "_exception_class_";

/// One `_exception_class_<Enum>` declaration.
///
/// `enum_path` is the class intern key `harvest_alloc_flavor_raw_from_llbcs`
/// stores (`canonical_struct_name` of the crate-stripped TypeDecl path).
/// `overrides` is the raw marker string. `variants` is the `TypeDecl`
/// variant list in declaration order; the discriminant map is not that
/// order. `classes` holds the hierarchy's class objects, created once
/// when the marker is harvested. [`Clone`] shares those [`HostObject`]
/// arcs (`PartialEq` is `Arc` identity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionClassHint {
    pub enum_path: String,
    pub overrides: String,
    pub variants: Vec<String>,
    pub classes: ExceptionClassObjects,
}

/// Class objects of one harvested hierarchy.
///
/// `base` subclasses the host `Exception`. `variants` follows
/// [`parse_exception_class_overrides`]: a parent precedes its child.
/// Each variant host's bases are `[parent]` — the base host when the
/// variant has no override. These are the objects the flow graph and
/// the bookkeeper share. Nothing else calls [`HostObject::new_class`]
/// for a minted class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionClassObjects {
    pub base: HostObject,
    /// `(variant, host)` in [`parse_exception_class_overrides`] order.
    pub variants: Vec<(String, HostObject)>,
}

impl ExceptionClassHint {
    /// Build one hierarchy, including its class objects.
    ///
    /// Harvest and tests both go through here so a hint cannot exist
    /// without the objects [`build_exception_class_objects`] creates.
    pub fn new(
        enum_path: impl Into<String>,
        overrides: impl Into<String>,
        variants: Vec<String>,
    ) -> Self {
        let enum_path = enum_path.into();
        let overrides = overrides.into();
        let classes = build_exception_class_objects(&enum_path, &overrides, &variants);
        Self {
            enum_path,
            overrides,
            variants,
            classes,
        }
    }
}

/// Class objects for one marker. The only `HostObject::new_class` site
/// for a minted hierarchy.
///
/// The base's bases are `[HOST_ENV Exception]`. A variant's bases are
/// `[parent host]`. `parse_exception_class_overrides` emits a parent
/// before its child, so the parent host already exists.
fn build_exception_class_objects(
    enum_path: &str,
    overrides: &str,
    variants: &[String],
) -> ExceptionClassObjects {
    let enum_name = exception_class_leaf(enum_path);
    let order = parse_exception_class_overrides(enum_name, overrides, variants);
    let exception = HOST_ENV
        .lookup_exception_class("Exception")
        .expect("HOST_ENV missing builtin Exception");
    let base_name = exception_class_key(enum_path, None);
    if base_name.is_empty() {
        panic!("_exception_class_ {enum_name}: empty class name");
    }
    let base = HostObject::new_class(base_name, vec![exception]);
    let mut hosts: Vec<(String, HostObject)> = Vec::with_capacity(order.len());
    for (variant, parent) in order {
        let parent_host = match parent.as_deref() {
            None => base.clone(),
            Some(parent_name) => hosts
                .iter()
                .find(|(name, _)| name == parent_name)
                .map(|(_, host)| host.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "_exception_class_ {enum_name}: parent `{parent_name}` was not built \
                         before `{variant}`"
                    )
                }),
        };
        let class_name = exception_class_key(enum_path, Some(&variant));
        if class_name.is_empty() {
            panic!("_exception_class_ {enum_name}: empty class name");
        }
        hosts.push((
            variant,
            HostObject::new_class(class_name, vec![parent_host]),
        ));
    }
    ExceptionClassObjects {
        base,
        variants: hosts,
    }
}

fn exception_class_objects_match(
    left: &ExceptionClassObjects,
    right: &ExceptionClassObjects,
) -> bool {
    left.base == right.base
        && left.variants.len() == right.variants.len()
        && left.variants.iter().zip(&right.variants).all(
            |((left_name, left_host), (right_name, right_host))| {
                left_name == right_name && left_host == right_host
            },
        )
}

/// Leaf of an exception-class enum path (`argument::ArgErr` → `ArgErr`).
pub(crate) fn exception_class_leaf(enum_path: &str) -> &str {
    path_leaf_stripped(enum_path)
}

/// Intern key of one class in the hierarchy.
///
/// `None` is the declaration key (`hint.enum_path`), the same string
/// `intern_class_by_qualname` uses for that TypeDecl. `Some(variant)` is
/// `{enum_path}::{variant}`, the spelling a ctor stamps (`synth::E::B`)
/// and an unmarked enum uses for its variant subclass. A leaf is not a
/// key: `exception_class_leaf` is diagnostic text only.
pub(crate) fn exception_class_key(enum_path: &str, variant: Option<&str>) -> String {
    match variant {
        Some(variant) => format!("{enum_path}::{variant}"),
        None => enum_path.to_string(),
    }
}

/// Whether `query` names the harvested enum identity `stored`.
///
/// Same comparisons as [`alloc_flavor_raw_identity_matches`]: the stored
/// intern key, `canonical_struct_name` of the query (a bare leaf whose
/// origin is registered), or the crate-stripped query.
pub(crate) fn exception_class_enum_path_matches(stored: &str, query: &str) -> bool {
    if stored == query {
        return true;
    }
    let identity = majit_ir::descr::canonical_struct_name(query);
    if stored == identity {
        return true;
    }
    let stripped = strip_crate_prefix(query);
    stripped != query && stored == stripped
}

/// Whether `name` is the base or a variant class of one hierarchy.
///
/// Accepted spellings are the declaration key, `{enum_path}::{variant}`,
/// an enum-path spelling [`exception_class_enum_path_matches`] accepts
/// (crate-stripped path, or a bare leaf whose origin is registered), and
/// `owner::variant` when `variant` is declared and `owner` matches the
/// enum path the same way. A bare leaf with no origin does not match.
/// The ctor stamps `synth::E::B`.
pub(crate) fn exception_class_variants_belong(
    enum_path: &str,
    variants: &[&str],
    name: &str,
) -> bool {
    if name.is_empty() {
        return false;
    }
    let base = exception_class_key(enum_path, None);
    if name == base || exception_class_enum_path_matches(enum_path, name) {
        return true;
    }
    if variants
        .iter()
        .any(|variant| name == exception_class_key(enum_path, Some(variant)))
    {
        return true;
    }
    let Some((owner, variant)) = name.rsplit_once("::") else {
        return false;
    };
    variants.contains(&variant)
        && (exception_class_enum_path_matches(enum_path, owner) || owner == base)
}

/// Whether `name` belongs to any harvested `_exception_class_` hierarchy.
///
/// Empty `hints` matches nothing: an unmarked error stays on the
/// carrier conversion.
pub fn exception_class_name_belongs(hints: &[ExceptionClassHint], name: &str) -> bool {
    hints.iter().any(|hint| {
        let variants: Vec<&str> = hint.variants.iter().map(String::as_str).collect();
        exception_class_variants_belong(&hint.enum_path, &variants, name)
    })
}

/// Nesting edges in mint order.
///
/// Each pair is `(variant, parent variant)`. `None` means the base class.
/// Declaration order is kept, except a child is emitted after its parent
/// so `assign_inheritance_ids` sees the parent `ClassDef` first.
/// Unknown variants, a cycle, a self-parent, a malformed token, and a
/// duplicate override panic with `enum_name`.
pub fn parse_exception_class_overrides(
    enum_name: &str,
    marker: &str,
    variants_in_declaration_order: &[String],
) -> Vec<(String, Option<String>)> {
    let mut parent_of: Vec<Option<String>> = vec![None; variants_in_declaration_order.len()];
    let mut overridden = vec![false; variants_in_declaration_order.len()];
    for token in marker.split_whitespace() {
        let (child, parent) = split_exception_override(enum_name, token);
        if child == parent {
            panic!("_exception_class_ {enum_name}: `{child}` cannot parent itself");
        }
        let child_index = variant_index(enum_name, variants_in_declaration_order, &child);
        if !variants_in_declaration_order
            .iter()
            .any(|name| name == &parent)
        {
            panic!("_exception_class_ {enum_name}: unknown variant `{parent}`");
        }
        if overridden[child_index] {
            panic!("_exception_class_ {enum_name}: duplicate override for `{child}`");
        }
        overridden[child_index] = true;
        parent_of[child_index] = Some(parent);
    }

    let mut emitted = vec![false; variants_in_declaration_order.len()];
    let mut order = Vec::with_capacity(variants_in_declaration_order.len());
    while order.len() < variants_in_declaration_order.len() {
        let next = variants_in_declaration_order
            .iter()
            .enumerate()
            .find(|(index, _)| {
                if emitted[*index] {
                    return false;
                }
                match &parent_of[*index] {
                    None => true,
                    Some(parent) => variants_in_declaration_order
                        .iter()
                        .enumerate()
                        .any(|(parent_index, name)| name == parent && emitted[parent_index]),
                }
            });
        let Some((index, name)) = next else {
            let pending: Vec<&str> = variants_in_declaration_order
                .iter()
                .enumerate()
                .filter(|(index, _)| !emitted[*index])
                .map(|(_, name)| name.as_str())
                .collect();
            panic!("_exception_class_ {enum_name}: cycle among {pending:?}");
        };
        order.push((name.clone(), parent_of[index].clone()));
        emitted[index] = true;
    }
    order
}

/// Farthest declaring ancestor, same walk as registration.
///
/// Walk `links` from `variant` to the root. A parent that does not
/// declare the field stays in the chain. The result is the farthest
/// class for which `declares` is true, or `variant` when none declares.
/// `ClassDef.get_owner` (`classdesc.py`) returns the class in
/// `getmro()` that holds the attr; `_generalize_attr` leaves that attr
/// on the topmost declarer. `register_exception_class_variant_fields`
/// and `exception_class_attr_owner` both call this walk. A cycle in
/// `links` panics with the chain.
pub(crate) fn exception_class_field_owner<'a>(
    links: &'a [(String, Option<String>)],
    variant: &'a str,
    declares: impl Fn(&str) -> bool,
) -> &'a str {
    let mut owner: Option<&'a str> = None;
    let mut current = variant;
    let mut seen = Vec::new();
    loop {
        if seen.contains(&current) {
            panic!("exception class field owner: cycle in override links {seen:?} -> {current}");
        }
        seen.push(current);
        if declares(current) {
            owner = Some(current);
        }
        let Some(parent) = links
            .iter()
            .find(|(name, _)| name == current)
            .and_then(|(_, parent)| parent.as_deref())
        else {
            break;
        };
        current = parent;
    }
    owner.unwrap_or(variant)
}

fn variant_index(enum_name: &str, variants: &[String], name: &str) -> usize {
    variants
        .iter()
        .position(|variant| variant == name)
        .unwrap_or_else(|| panic!("_exception_class_ {enum_name}: unknown variant `{name}`"))
}

fn split_exception_override(enum_name: &str, token: &str) -> (String, String) {
    let malformed =
        || -> ! { panic!("_exception_class_ {enum_name}: malformed override `{token}`") };
    let Some(open) = token.find('(') else {
        malformed();
    };
    if !token.ends_with(')') || token.matches('(').count() != 1 || token.matches(')').count() != 1 {
        malformed();
    }
    let child = &token[..open];
    let parent = &token[open + 1..token.len() - 1];
    if !is_exception_variant_ident(child) || !is_exception_variant_ident(parent) {
        malformed();
    }
    (child.to_string(), parent.to_string())
}

fn is_exception_variant_ident(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

/// Append one artefact's harvest onto the program-wide hint list.
///
/// Callers walk artefacts in dependency order and append each harvest
/// once. A repeated `enum_path` panics with the same text as
/// [`crate::front::mir::absorb_semantic_program`]: the class was
/// declared twice. A repeated path whose class objects are a different
/// `Arc` panics with the different-object message. Within one program
/// every hint for a path is a clone of one harvest.
pub fn extend_program_exception_class_hints(
    acc: &mut Vec<ExceptionClassHint>,
    own: impl IntoIterator<Item = ExceptionClassHint>,
) {
    for hint in own {
        push_program_exception_class_hint(acc, hint);
    }
}

/// Record `hint` on the program-wide list.
///
/// A repeated `enum_path` panics. When the arriving hosts are a
/// different `Arc`, the message says so. A clone of the recorded hint
/// panics with the duplicate-declaration message. Both builds take the
/// same branch.
pub(crate) fn push_program_exception_class_hint(
    acc: &mut Vec<ExceptionClassHint>,
    hint: ExceptionClassHint,
) {
    if let Some(existing) = acc
        .iter()
        .find(|existing| existing.enum_path == hint.enum_path)
    {
        let enum_name = exception_class_leaf(&hint.enum_path);
        let message = if exception_class_objects_match(&existing.classes, &hint.classes) {
            format!(
                "_exception_class_ {enum_name}: duplicate hierarchy `{}`",
                hint.enum_path
            )
        } else {
            format!(
                "_exception_class_ {enum_name}: duplicate hierarchy `{}` arrived with a different class object",
                hint.enum_path
            )
        };
        panic!("{message}");
    }
    acc.push(hint);
}

/// One raw-owned attribute the minted class's light destructor frees.
///
/// `offset` is the pointer slot in the instance (the host object
/// header, then each ancestor's declared fields). `type_path` is the TypeDecl
/// `name_path` the slot boxes. A subclass repeats an inherited attr at
/// the embedded-super offset: the field row stays on the declaring
/// class, and the subclass instance has its own tid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintedRawOwnedDrop {
    pub offset: usize,
    pub type_path: String,
}

/// One minted class, parent before child, for the runtime static table.
///
/// `parent` is an index into the same list. `None` is a hierarchy base:
/// its RPython parent is host `Exception`, which is not a row.
/// `size` and `gc_ptr_offsets` stay zero and empty until
/// `attach_minted_exception_raw_owned_drops` fills them from
/// `minted_exception_instance_layout`. `raw_owned` is filled by that
/// same call. A class with no raw-owned attr, including one that only
/// inherits none, stays empty and gets no destructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintedExceptionClassEmit {
    pub name: String,
    pub parent: Option<usize>,
    pub size: usize,
    pub gc_ptr_offsets: Vec<usize>,
    pub raw_owned: Vec<MintedRawOwnedDrop>,
}

/// Flatten harvested hierarchies into parent-before-child rows.
///
/// The base row's parent is `None`. A variant's parent is the earlier
/// row whose host is `class_bases()[0]` (`HostObject` equality, not the
/// override spelling). A repeated intern key panics. A variant whose
/// parent host was not emitted earlier panics.
pub fn minted_exception_class_rows(hints: &[ExceptionClassHint]) -> Vec<MintedExceptionClassEmit> {
    let mut rows = Vec::new();
    let mut hosts: Vec<HostObject> = Vec::new();
    for hint in hints {
        push_minted_exception_row(&mut rows, &mut hosts, &hint.classes.base, None);
        for (_, host) in &hint.classes.variants {
            let parent_host = host
                .class_bases()
                .and_then(|bases| bases.first())
                .unwrap_or_else(|| {
                    panic!(
                        "minted exception class `{}` has no parent host",
                        host.qualname()
                    )
                });
            let parent = hosts
                .iter()
                .position(|earlier| earlier == parent_host)
                .unwrap_or_else(|| {
                    panic!(
                        "minted exception class `{}` parent was not emitted earlier",
                        host.qualname()
                    )
                });
            push_minted_exception_row(&mut rows, &mut hosts, host, Some(parent));
        }
    }
    rows
}

fn push_minted_exception_row(
    rows: &mut Vec<MintedExceptionClassEmit>,
    hosts: &mut Vec<HostObject>,
    host: &HostObject,
    parent: Option<usize>,
) {
    let name = host.qualname().to_string();
    if rows.iter().any(|row| row.name == name) {
        panic!("duplicate minted exception class `{name}`");
    }
    rows.push(MintedExceptionClassEmit {
        name,
        parent,
        size: 0,
        gc_ptr_offsets: Vec::new(),
        raw_owned: Vec::new(),
    });
    hosts.push(host.clone());
}

/// `NameManager` `global_prefix` (`gensupp.py` `NameManager.__init__`).
/// A first claim of an intern key is this prefix plus the key after
/// `C_IDENTIFIER`.
const MINTED_EXC_PREFIX: &str = "MINTED_EXC_";

/// `NameManager.uniquename` `lenmax` (`gensupp.py`). Upstream's default
/// is 50. The render-test keys (`synth::E::B`, `Arg::TooManyMethod`)
/// and a `_destructor` companion are shorter than that. A longer key is
/// cut by `basename[:lenmax]` before `C_IDENTIFIER`.
/// `NameManager._ensure_unique` still separates two keys that share the
/// truncated basename.
const MINTED_EXC_NAME_LENMAX: usize = 50;

/// Rust source for the minted exception class table.
///
/// Included in the module that defines `MintedExceptionClass`, so the
/// table uses that short name. An empty `rows` is an empty table.
/// One `NameManager` (`gensupp.py`) claims each static, each
/// `_destructor` companion, and each `_alloc_with_del` /
/// `_alloc_fnaddr` companion, walking `rows` in order. `global_prefix`
/// is `MINTED_EXC_` and `number_sep` is `_`. `C_IDENTIFIER` folds every
/// non-identifier byte to `_` inside `NameManager.uniquename`.
/// Each claim omits `with_number`, as `SourceGenerator.uniquecname`
/// and `_LocalScope.uniquename` do. `NameManager.uniquename` then
/// sets `with_number = basename in ('v', 'w_')`. When that holds and
/// the last character is not a digit, `fmt = '%s%d'`, so basename
/// `v` is claimed as `v0`. Any other basename already in `seennames`
/// — a sanitized collision or an earlier companion — comes back with
/// the `_N` suffix from `NameManager._ensure_unique`. Claim order is the
/// bare ident, then `_destructor`, then `_alloc_with_del`, then
/// `_alloc_fnaddr`, so a later row whose basename is an earlier
/// `_destructor` still takes that companion's `_1` suffix.
///
/// Every emitted `static` and `fn` name is `MINTED_EXC_` plus a
/// `NameManager.uniquename` claim. The including module defines no
/// `MINTED_EXC_`-prefixed item, so the table and the module's names
/// sit outside the claimed namespace.
///
/// A non-empty `raw_owned` list also emits `support.py`
/// `build_ll_0_alloc_with_del`: a zero-argument `extern "C"` wrapper
/// closed over the row index and the row `PyType`, registered on
/// `HELPER_FNADDRS` under `build_ll_0_alloc_with_del::{intern key}`.
pub fn render_minted_exception_classes_rs(rows: &[MintedExceptionClassEmit]) -> String {
    if rows.is_empty() {
        return "pub static MINTED_EXCEPTION_CLASSES: &[MintedExceptionClass] = &[];\n".to_string();
    }
    let mut names = crate::translator::gensupp::NameManager::new(MINTED_EXC_PREFIX, "_");
    let mut defs = String::new();
    let mut table =
        String::from("pub static MINTED_EXCEPTION_CLASSES: &[MintedExceptionClass] = &[\n");
    for (index, row) in rows.iter().enumerate() {
        // `NameManager.uniquename` (`gensupp.py`) with the default
        // `with_number=None` (`basename in ('v', 'w_')`). Companions are
        // claimed for every row, emitted or not, so a later ident
        // cannot take `{bare}_destructor` or `{bare}_alloc_with_del`.
        let (bare, ident) = names.uniquename_bare(&row.name, None, MINTED_EXC_NAME_LENMAX);
        let destructor_name =
            names.uniquename(&format!("{bare}_destructor"), None, MINTED_EXC_NAME_LENMAX);
        let alloc_name = names.uniquename(
            &format!("{bare}_alloc_with_del"),
            None,
            MINTED_EXC_NAME_LENMAX,
        );
        let alloc_fnaddr_name = names.uniquename(
            &format!("{bare}_alloc_fnaddr"),
            None,
            MINTED_EXC_NAME_LENMAX,
        );
        defs.push_str(&format!(
            "static {ident}: pyre_object::PyType = pyre_object::pyobject::new_pytype(\"{}\");\n",
            escape_rust_string(&row.name)
        ));
        let emit_destructor = !row.raw_owned.is_empty();
        if emit_destructor {
            defs.push_str(&format!(
                "unsafe fn {destructor_name}(obj_addr: usize) {{\n    unsafe {{\n"
            ));
            for raw in &row.raw_owned {
                if raw.type_path.is_empty() {
                    panic!(
                        "minted exception class `{}` raw-owned attr has an empty type path",
                        row.name
                    );
                }
                defs.push_str(&format!(
                    "        drop_raw_owned_attr::<{}>(obj_addr, {});\n",
                    raw.type_path, raw.offset
                ));
            }
            defs.push_str("    }\n}\n");
            // `support.py` `build_ll_0_alloc_with_del(RESULT, vtable)`:
            // zero arguments; RESULT and the vtable are closed over.
            // `HELPER_FNADDRS` publishes the path `handle_residual_call`
            // records on `ConstFnAddr`. wasm32 has no `linkme` slice
            // (`helper_fnaddr.rs` `register`).
            defs.push_str(&format!(
                "unsafe extern \"C\" fn {alloc_name}() -> i64 {{\n    alloc_with_del_for_index({index}, &{ident} as *const _ as usize)\n}}\n"
            ));
            let fnaddr_path = format!(
                "build_ll_0_alloc_with_del::{}",
                escape_rust_string(&row.name)
            );
            defs.push_str(&format!(
                "#[cfg(not(target_arch = \"wasm32\"))]\n#[::linkme::distributed_slice(::majit_ir::helper_fnaddr::HELPER_FNADDRS)]\n#[linkme(crate = ::majit_ir::linkme)]\nstatic {alloc_fnaddr_name}: ::majit_ir::helper_fnaddr::HelperFnAddr =\n    ::majit_ir::helper_fnaddr::HelperFnAddr::new(\"{fnaddr_path}\", {alloc_name} as *const (), 0, \">w\");\n#[cfg(target_arch = \"wasm32\")]\n#[::ctor::ctor(unsafe)]\nfn {alloc_fnaddr_name}() {{\n    ::majit_ir::helper_fnaddr::register(\"{fnaddr_path}\", {alloc_name} as *const (), 0, \">w\");\n}}\n"
            ));
        }
        let destructor_field = if emit_destructor {
            format!("Some({destructor_name})")
        } else {
            "None".to_string()
        };
        let alloc_field = if emit_destructor {
            format!("Some({alloc_name})")
        } else {
            "None".to_string()
        };
        let parent = match row.parent {
            None => "None".to_string(),
            Some(parent) => format!("Some({parent})"),
        };
        let offsets = render_gc_ptr_offsets(&row.gc_ptr_offsets);
        table.push_str(&format!(
            "    MintedExceptionClass {{ name: \"{}\", pytype: &{ident}, parent: {parent}, size: {size}, gc_ptr_offsets: {offsets}, destructor: {destructor_field}, alloc_with_del: {alloc_field} }},\n",
            escape_rust_string(&row.name),
            size = row.size,
        ));
    }
    defs.push_str(&table);
    defs.push_str("];\n");
    defs
}

/// `&[]` or `&[16, 32]`. The slice is a static promotion inside the
/// emitted `static` item.
fn render_gc_ptr_offsets(offsets: &[usize]) -> String {
    if offsets.is_empty() {
        return "&[]".to_string();
    }
    let body = offsets
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{body}]")
}

fn escape_rust_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

struct ResolvedExceptionClass {
    enum_path: String,
    overrides: String,
    variants: Vec<String>,
}

/// Marker resolutions in `llbcs`, without class objects.
///
/// Same resolution as [`harvest_exception_class_hints_from_llbcs`]. The
/// override list is parsed so a bad marker panics here. Class objects
/// are built only by the harvest.
fn resolved_exception_class_markers(llbcs: &[Llbc]) -> Vec<ResolvedExceptionClass> {
    let mut out: Vec<ResolvedExceptionClass> = Vec::new();
    for llbc in llbcs {
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            let Some(enum_name) = leaf.strip_prefix(EXCEPTION_CLASS_PREFIX) else {
                continue;
            };
            if enum_name.is_empty() {
                panic!("_exception_class_ marker `{path}` names an empty enum");
            }
            let overrides = global_marker_str(llbc, gd).unwrap_or_else(|| {
                panic!(
                    "_exception_class_ {enum_name}: marker `{path}` has an undecodable \
                     string initializer; this signals a marker encoding/decoder drift"
                )
            });
            let (enum_path, variants) = resolve_exception_class_enum(llbc, &path, enum_name);
            let _ = parse_exception_class_overrides(enum_name, &overrides, &variants);
            if out.iter().any(|hint| hint.enum_path == enum_path) {
                panic!("_exception_class_ {enum_name}: duplicate hierarchy `{enum_path}`");
            }
            out.push(ResolvedExceptionClass {
                enum_path,
                overrides,
                variants,
            });
        }
    }
    out
}

/// `_exception_class_<Enum>` markers in `llbcs`.
///
/// Resolved like [`harvest_alloc_flavor_raw_from_llbcs`]: one enum
/// `TypeDecl` whose leaf matches the marker, or the unique one in the
/// marker's module. The stored path is that decl's intern key. Variant
/// names stay in `TypeDecl` order. A marker that does not resolve, is
/// not an enum, or whose override list does not parse panics with the
/// enum name. Each hint's class objects are created here, once.
pub fn harvest_exception_class_hints_from_llbcs(llbcs: &[Llbc]) -> Vec<ExceptionClassHint> {
    resolved_exception_class_markers(llbcs)
        .into_iter()
        .map(|resolved| {
            ExceptionClassHint::new(resolved.enum_path, resolved.overrides, resolved.variants)
        })
        .collect()
}

/// This artefact's own hints, as clones of `seeded`.
///
/// `seeded` is [`crate::front::mir::llbc_exception_class_hints`]: the
/// program-wide prefix, or — when the artefact was not seeded — the
/// harvest that call cached. Paths come from a second resolution that
/// does not build hosts. An own path missing from `seeded` panics: the
/// production prefix includes the artefact's own harvest, and an
/// unseeded read is that harvest.
pub(crate) fn own_exception_class_hints(
    llbc: &Llbc,
    seeded: &[ExceptionClassHint],
) -> Vec<ExceptionClassHint> {
    let mut out = Vec::new();
    for resolved in resolved_exception_class_markers(std::slice::from_ref(llbc)) {
        let Some(hint) = seeded
            .iter()
            .find(|hint| hint.enum_path == resolved.enum_path)
        else {
            let enum_name = exception_class_leaf(&resolved.enum_path);
            panic!(
                "_exception_class_ {enum_name}: own hierarchy `{}` is absent from the seeded hint set",
                resolved.enum_path
            );
        };
        out.push(hint.clone());
    }
    out
}

struct ExceptionEnumCandidate {
    identity: String,
    path: String,
    variants: Vec<String>,
}

/// Resolve `_exception_class_<Enum>` to one enum intern key and its
/// declaration-order variant names.
fn resolve_exception_class_enum(
    llbc: &Llbc,
    marker_path: &str,
    enum_name: &str,
) -> (String, Vec<String>) {
    let marker_stripped = strip_crate_prefix(marker_path);
    let marker_module = path_module(&marker_stripped);
    let mut candidates: Vec<ExceptionEnumCandidate> = Vec::new();
    for td in llbc.iter_type_decls() {
        let TypeDeclKind::Enum(variants) = &td.kind else {
            continue;
        };
        let td_path = td.item_meta.name_path();
        if path_leaf_stripped(&td_path) != enum_name {
            continue;
        }
        let names: Vec<String> = variants
            .iter()
            .map(|variant| variant.name.clone())
            .collect();
        if names.iter().any(|name| name.is_empty()) {
            panic!("_exception_class_ {enum_name}: empty variant name on `{td_path}`");
        }
        if names
            .iter()
            .enumerate()
            .any(|(index, name)| names[..index].iter().any(|earlier| earlier == name))
        {
            panic!("_exception_class_ {enum_name}: duplicate variant `{td_path}`");
        }
        let identity = alloc_flavor_class_intern_key(&td_path);
        if let Some(existing) = candidates
            .iter()
            .find(|candidate| candidate.identity == identity)
        {
            if existing.variants != names {
                panic!(
                    "_exception_class_ {enum_name}: enum `{identity}` has conflicting \
                     variant lists"
                );
            }
            continue;
        }
        candidates.push(ExceptionEnumCandidate {
            identity,
            path: td_path,
            variants: names,
        });
    }
    let chosen = if candidates.len() == 1 {
        candidates.pop().expect("len == 1")
    } else {
        let all_named: Vec<String> = candidates
            .iter()
            .map(|candidate| candidate.path.clone())
            .collect();
        let mut in_module: Vec<ExceptionEnumCandidate> = Vec::new();
        for candidate in candidates.drain(..) {
            let stripped = strip_crate_prefix(&candidate.path);
            if path_module(&stripped) == marker_module
                && !in_module
                    .iter()
                    .any(|existing| existing.identity == candidate.identity)
            {
                in_module.push(candidate);
            }
        }
        if in_module.len() == 1 {
            in_module.pop().expect("len == 1")
        } else {
            let named: Vec<&str> = if in_module.is_empty() {
                all_named.iter().map(String::as_str).collect()
            } else {
                in_module
                    .iter()
                    .map(|candidate| candidate.path.as_str())
                    .collect()
            };
            panic!(
                "_exception_class_ {enum_name}: marker `{marker_path}` does not resolve \
                 to a unique enum TypeDecl (candidates: {named:?})"
            );
        }
    };
    (chosen.identity, chosen.variants)
}

/// Peel a one-word pointer spelling to its pointee.
///
/// `*const T` / `*mut T` / `&mut T` / `&T` / `Box<T>` / `Arc<T>` /
/// `Rc<T>` / `NonNull<T>` / `AtomicPtr<T>` (qualified paths included).
/// A fat pointer (`Box<dyn Trait>`, `&[T]`, `&str`, `Arc<dyn Trait>`)
/// is left whole: it is two words, not a raw address.
/// `history.getkind` / `descr.py get_type_flag` classify `Ptr` from
/// `TO._gckind`, so the flavor matcher has to name `T`.
fn peel_pointer_spelling(path: &str) -> &str {
    let path = path.trim();
    if crate::fat_ptr_layout::spelling_is_fat_ptr(path) {
        return path;
    }
    crate::fat_ptr_layout::pointer_pointee(path).unwrap_or(path)
}

/// Whether `spelling` is one pointer word whose pointee is a harvested
/// `_alloc_flavor_ = raw` identity. Fat pointers stay off this arm.
pub(crate) fn spelling_is_one_word_flavor_raw_ptr(raw: &HashSet<String>, spelling: &str) -> bool {
    if crate::fat_ptr_layout::spelling_is_fat_ptr(spelling) {
        return false;
    }
    let peeled = peel_pointer_spelling(spelling);
    if peeled == spelling.trim() {
        return false;
    }
    alloc_flavor_raw_contains(raw, spelling)
}

/// Whether `path` is a harvested `_alloc_flavor_ = raw` identity.
///
/// `path` is the intern key `Bookkeeper::stamp_raw_alloc_flavor` receives
/// (`canonical_struct_name` of the struct), a crate-prefixed TypeDecl
/// path the front paints on `ValueType::Ref(Some(..))`, or a `*const` /
/// `*mut` / `&` spelling of either. The harvested set stores the
/// crate-stripped intern key; a qualified path in another module does
/// not match. A bare leaf matches when [`majit_ir::descr::canonical_struct_name`]
/// qualifies it through `STRUCT_ORIGIN_REGISTRY` (unique among all
/// known struct keys). A tombstoned or unregistered leaf does not
/// guess from the raw subset. `rclass.py getgcflavor` /
/// `classdesc.py ClassDesc.get_param` read `_alloc_flavor_` off that
/// classdesc. An enum variant owner `X::Variant` is the same class as
/// `X` (`ClassDesc.get_param` walks the MRO).
pub(crate) fn alloc_flavor_raw_contains(raw: &HashSet<String>, path: &str) -> bool {
    if raw.is_empty() {
        return false;
    }
    let path = peel_pointer_spelling(path);
    // `_alloc_flavor_` is a class attribute: every instantiation of the
    // ADT (`CodeObject<ConstantData>`) shares its flavor.
    let path = path.split('<').next().unwrap_or(path);
    if alloc_flavor_raw_identity_matches(raw, path) {
        return true;
    }
    // `MapNode::Terminator` inherits `_alloc_flavor_MapNode`.
    if let Some((enum_owner, _)) = path.rsplit_once("::")
        && !enum_owner.is_empty()
        && alloc_flavor_raw_identity_matches(raw, enum_owner)
    {
        return true;
    }
    false
}

fn alloc_flavor_raw_identity_matches(raw: &HashSet<String>, path: &str) -> bool {
    let identity = majit_ir::descr::canonical_struct_name(path);
    if raw.contains(path) || raw.contains(&identity) {
        return true;
    }
    let stripped = strip_crate_prefix(path);
    stripped != path && raw.contains(&stripped)
}

/// Split the marker's comma-joined payload into `(field_name, rank)`
/// pairs — `rclass.py _parse_field_list` over the declared
/// `_immutable_fields_` list.  The suffix grammar lives entirely in
/// [`crate::model::ImmutableRank::parse`].
fn parse_immutable_entry_list(joined: &str) -> Vec<(String, crate::model::ImmutableRank)> {
    joined
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(crate::model::ImmutableRank::parse)
        .collect()
}

fn should_skip_generated_elidable_helper(fn_name: &str) -> bool {
    fn_name.starts_with("_orig_") && fn_name.ends_with("_unlikely_name")
}

/// Union harvested JIT-hint tokens onto `graph.hints` and project the
/// policy tokens onto `graph.func`.
///
/// `harvest_hints_from_llbcs` keys the marker consts. `look_inside_graph`
/// reads `_elidable_function_`, `_jit_look_inside_` and `_jit_unroll_safe_`
/// off `graph.func`. A replace would drop tokens an earlier alias already
/// carried, so this is monotonic. A token already in the vec is still
/// projected: a direct assignment of `graph.hints` has not set `func`.
pub fn merge_hints_into_graph(graph: &mut crate::model::FunctionGraph, hints: &[String]) {
    for hint in hints {
        graph.push_hint(hint.clone());
        // rlib/jit.py `@oopspec` stores the spec on `func.oopspec`. The
        // harvester spells that attribute as an `oopspec:` hint token;
        // `call.py` `hasattr(func, 'oopspec')` reads the attribute, not
        // the token list.
        if graph.func.oopspec.is_none()
            && let Some(spec) = hint.strip_prefix("oopspec:")
            && !spec.is_empty()
        {
            graph.func.oopspec = Some(spec.to_string());
        }
        if graph.func.llop.is_none()
            && let Some(opname) = hint.strip_prefix("llop:")
            && !opname.is_empty()
        {
            graph.func.llop = Some(opname.to_string());
        }
    }
}

fn push_hint(out: &mut HashMap<String, Vec<String>>, key: String, hint: &str) {
    out.entry(key).or_default().push(hint.to_string());
}

fn marker_path_to_fn_path(
    marker_path: &str,
    prefix: &str,
    function_paths: &HashSet<String>,
) -> String {
    let stripped = strip_crate_prefix(marker_path);
    match stripped.rsplit_once("::") {
        Some((module, leaf)) => {
            let fn_leaf = leaf.strip_prefix(prefix).unwrap_or(leaf);
            // `#[jit_elidable]` on an impl method cannot emit a module-level
            // sibling const (trait impls reject foreign associated items).
            // The macro therefore emits a body-local marker, which Charon
            // promotes under `<method>::_elidable_function_<method>`.
            // In that spelling the parent already is the function path; do
            // not append the leaf a second time.
            // A module and its free function may legitimately have the same
            // leaf (`stack_check::stack_check`).  The old leaf-equality
            // heuristic collapsed that path to `stack_check`, silently
            // dropping its `dont_look_inside` hint.  Resolve against Charon's
            // actual function declarations: a sibling marker names
            // `module::fn_leaf`, while an impl/body-local marker names its
            // already-complete parent function path.
            let sibling_path = format!("{module}::{fn_leaf}");
            if function_paths.contains(&sibling_path) {
                sibling_path
            } else if function_paths.contains(module) {
                module.to_string()
            } else {
                sibling_path
            }
        }
        None => stripped
            .strip_prefix(prefix)
            .unwrap_or(&stripped)
            .to_string(),
    }
}

fn strip_crate_prefix(path: &str) -> String {
    let mut parts = path.split("::");
    match (parts.next(), parts.next()) {
        (Some(_crate), Some(second)) => {
            let mut out = String::from(second);
            for part in parts {
                out.push_str("::");
                out.push_str(part);
            }
            out
        }
        _ => path.to_string(),
    }
}

fn global_marker_bool(llbc: &Llbc, gd: &GlobalDecl) -> Option<bool> {
    if let Some(value) = gd.const_expr()
        && let Some(b) = decode_bool_const(llbc, value)
    {
        return Some(b);
    }
    let init_id = marker_init_fun_id(gd)?;
    bool_assigned_to_return(llbc, init_id)
}

pub(crate) fn marker_init_fun_id(gd: &GlobalDecl) -> Option<u64> {
    gd.init_fun_id()
}

fn bool_assigned_to_return(llbc: &Llbc, init_id: u64) -> Option<bool> {
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    let forward_blocks = crate::front::mir::forward_reachable_mask(llbc, &body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(place, Rvalue::Use(Operand::Const(value), _))) =
                stmt.stmt_kind()
            else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(0)) {
                continue;
            }
            if let Some(b) = decode_bool_const(llbc, &value) {
                return Some(b);
            }
        }
    }
    None
}

/// Read the `&'static str` value of an oopspec marker const. The init
/// body assigns `Local(0) = Const(ConstantExpr)` whose kind is the
/// string literal (`[{"Str": spec}, ty]`), mirroring [`global_marker_bool`]'s bool path.
pub(crate) fn global_marker_str(llbc: &Llbc, gd: &GlobalDecl) -> Option<String> {
    crate::front::mir::cached_global_marker_str(llbc, gd.def_id, || {
        global_marker_str_uncached(llbc, gd)
    })
}

fn global_marker_str_uncached(llbc: &Llbc, gd: &GlobalDecl) -> Option<String> {
    if let Some(value) = gd.const_expr()
        && let Some(spec) = decode_str_const(llbc, value)
    {
        return Some(spec);
    }
    // Named consts point at their initializer by a `Call` inside `value`,
    // not by an `init` field.
    let init_id = marker_init_fun_id(gd)?;
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    let forward_blocks = crate::front::mir::forward_reachable_mask(llbc, &body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &block.statements {
            let StmtKind::Assign(place, Rvalue::Use(Operand::Const(value), _)) =
                stmt.stmt_kind().ok()?
            else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(0)) {
                continue;
            }
            if let Some(s) = decode_str_const(llbc, &value) {
                return Some(s);
            }
        }
    }
    None
}

fn decode_str_const(llbc: &Llbc, value: &serde_json::Value) -> Option<String> {
    llbc.const_expr_literal(value)?
        .get("Str")?
        .as_str()
        .map(str::to_string)
}

/// Walk the marker const's initializer body, the same `Call` `value`
/// `marker_init_fun_id` names. `_call_aroundstate_target_` and the
/// funcptr markers (`_random_effects_on_gcobjs_`,
/// `_callable_elidable_function_`, `_funcptr_canraise_false_`) store a
/// funcptr there.
fn visit_marker_init_rvalues(llbc: &Llbc, gd: &GlobalDecl, mut visit: impl FnMut(&Rvalue)) -> bool {
    let Some(init_id) = marker_init_fun_id(gd) else {
        return false;
    };
    let Some(body) = llbc.fn_by_id(init_id).and_then(|fd| fd.unstructured()) else {
        return false;
    };
    let forward_blocks = crate::front::mir::forward_reachable_mask(llbc, &body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(_, rvalue)) = stmt.stmt_kind() else {
                continue;
            };
            visit(&rvalue);
        }
    }
    true
}

/// FunDecl the marker initializer names. Reuses the aroundstate `FnDef`
/// walk (`collect_aroundstate_rvalue` / `const_fn_def_regular_id`).
/// The funcptr markers (`_random_effects_on_gcobjs_`,
/// `_callable_elidable_function_`, `_funcptr_canraise_false_`) decode
/// through this.
fn decode_funcptr_marker_fun_decl<'a>(llbc: &'a Llbc, gd: &GlobalDecl) -> Option<&'a FunDecl> {
    let mut fn_id = None;
    let mut save_err = None;
    if !visit_marker_init_rvalues(llbc, gd, |rvalue| {
        collect_aroundstate_rvalue(llbc, rvalue, &mut fn_id, &mut save_err);
    }) {
        return None;
    }
    llbc.fn_by_id(fn_id?)
}

/// CallPath key the funcptr markers harvest under.
/// `analyze_random_effects` / `external_funcobj` /
/// `mark_external_gc_effects` look a direct call up by it; harvested
/// `"elidable"` uses the same key for `_callable_elidable_function_`,
/// and `"funcptr_canraise_false"` for `_funcptr_canraise_false_`.
///
/// `unspecialized_fun_segments` splits `name_path()` on `::`; `target_to_path`
/// keeps those segments. A local majit_rlib FunDecl is stored crate-stripped
/// so `apply_harvested_funcobj_hints` / `free_function_alias_paths` register
/// the module-relative key; a foreign `libc::…` FunDecl keeps the crate.
fn funcptr_marker_call_path_key(fd: &FunDecl) -> String {
    let path = fd.item_meta.name_path();
    if fd.item_meta.is_local {
        strip_crate_prefix(&path)
    } else {
        path
    }
}

/// `_call_aroundstate_target_<fn>` initializer: `(funcptr, save_err: i64)`.
///
/// Charon lowers the tuple as an `Aggregate` (or a single const) whose
/// operands are an `FnDef` of the extern funcptr and a scalar. The funcptr
/// identity is its crate-stripped path, and `ItemMeta.attr_info.rename`
/// when that is the `link_name`. The two are joined with a tab so a later
/// `function_fnaddrs` lookup (`register_macro_helper_trace_fnaddr`) can try
/// either spelling. `None` means the initializer is present but not that
/// shape — the caller panics.
fn decode_aroundstate_marker(llbc: &Llbc, gd: &GlobalDecl) -> Option<(String, i64)> {
    let mut fn_id: Option<u64> = None;
    let mut save_err: Option<i64> = None;
    if !visit_marker_init_rvalues(llbc, gd, |rvalue| {
        collect_aroundstate_rvalue(llbc, rvalue, &mut fn_id, &mut save_err);
    }) {
        return None;
    }
    let save_err = save_err?;
    let fd = llbc.fn_by_id(fn_id?)?;
    let path = strip_crate_prefix(&fd.item_meta.name_path());
    let link = fd
        .item_meta
        .attr_info
        .rename
        .as_deref()
        .filter(|name| !name.is_empty());
    let identity = match link {
        Some(link) if !path.is_empty() => format!("{path}\t{link}"),
        Some(link) => link.to_string(),
        None => path,
    };
    if identity.is_empty() {
        return None;
    }
    Some((identity, save_err))
}

fn collect_aroundstate_rvalue(
    llbc: &Llbc,
    rvalue: &Rvalue,
    fn_id: &mut Option<u64>,
    save_err: &mut Option<i64>,
) {
    match rvalue {
        Rvalue::Use(operand, _) => collect_aroundstate_operand(llbc, operand, fn_id, save_err),
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                collect_aroundstate_operand(llbc, operand, fn_id, save_err);
            }
        }
        // `fn_ptr as unsafe extern "C" fn(...)` is `UnaryOp(Cast(FnPtr), FnDef)`,
        // not a bare `Cast` rvalue. The tuple aggregate then moves that local.
        Rvalue::UnaryOp(_, operand) | Rvalue::Cast(_, operand, _) => {
            collect_aroundstate_operand(llbc, operand, fn_id, save_err);
        }
        _ => {}
    }
}

fn collect_aroundstate_operand(
    llbc: &Llbc,
    operand: &Operand,
    fn_id: &mut Option<u64>,
    save_err: &mut Option<i64>,
) {
    let Operand::Const(value) = operand else {
        return;
    };
    if fn_id.is_none() {
        *fn_id = majit_charon_reader::ullbc::const_fn_def_regular_id(llbc, value);
    }
    if save_err.is_none() {
        *save_err = llbc
            .const_expr_literal(value)
            .and_then(|lit| scalar_pair_i64(lit.get("Scalar")?));
    }
}

/// `{"Signed": [ty, "n"]}` / `{"Unsigned": [ty, "n"]}`, the `save_err`
/// half of `_call_aroundstate_target_`.
fn scalar_pair_i64(scalar: &serde_json::Value) -> Option<i64> {
    let obj = scalar.as_object()?;
    ["Signed", "Unsigned"].iter().find_map(|key| {
        obj.get(*key)?
            .as_array()?
            .get(1)?
            .as_str()?
            .parse::<i64>()
            .ok()
    })
}

fn decode_bool_const(llbc: &Llbc, value: &serde_json::Value) -> Option<bool> {
    llbc.const_expr_literal(value)?.get("Bool")?.as_bool()
}

#[cfg(test)]
mod tests {
    use super::{
        decode_str_const, exception_class_enum_path_matches, exception_class_field_owner,
        exception_class_key, extend_program_exception_class_hints,
        harvest_exception_class_hints_from_llbcs, marker_path_to_fn_path,
        parse_exception_class_overrides, parse_immutable_entry_list,
    };
    use crate::model::ImmutableRank;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn memo_marker_recovers_its_function_attribute_and_owner() {
        let (prefix, hints) = super::CONST_PREFIX_HINTS
            .iter()
            .find(|(prefix, _)| *prefix == "_annspecialcase_memo_")
            .unwrap();
        assert_eq!(*hints, &["specialize:memo"]);
        let functions = HashSet::from(["owner::Cache::getorbuild".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "pyre_interpreter::owner::Cache::getorbuild::_annspecialcase_memo_getorbuild",
                prefix,
                &functions,
            ),
            "owner::Cache::getorbuild"
        );
    }

    #[test]
    fn argtype_body_local_marker_binds_trait_impl_static_method() {
        let functions = HashSet::from(["owner::<Impl>::convert".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "crate::owner::<Impl>::convert::_annspecialcase_argtype_convert",
                "_annspecialcase_argtype_",
                &functions,
            ),
            "owner::<Impl>::convert"
        );
    }

    #[test]
    fn argtype_body_local_marker_binds_free_function() {
        let functions = HashSet::from(["rstring::startswith".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "majit_rlib::rstring::startswith::_annspecialcase_argtype_startswith",
                "_annspecialcase_argtype_",
                &functions,
            ),
            "rstring::startswith"
        );
    }

    #[test]
    fn call_location_marker_recovers_its_function_attribute() {
        let (prefix, hints) = super::CONST_PREFIX_HINTS
            .iter()
            .find(|(prefix, _)| *prefix == "_annspecialcase_call_location_")
            .unwrap();
        assert_eq!(*hints, &["specialize:call_location"]);
        let functions = HashSet::from(["jit::isconstant".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "majit_rlib::jit::isconstant::_annspecialcase_call_location_isconstant",
                prefix,
                &functions,
            ),
            "jit::isconstant"
        );
    }

    #[test]
    fn ll_marker_recovers_its_function_attribute() {
        let (prefix, hints) = super::CONST_PREFIX_HINTS
            .iter()
            .find(|(prefix, _)| *prefix == "_annspecialcase_ll_")
            .unwrap();
        assert_eq!(*hints, &["specialize:ll"]);
        let functions = HashSet::from(["object_array::jit_ll_arraycopy".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "pyre_object::object_array::jit_ll_arraycopy::_annspecialcase_ll_jit_ll_arraycopy",
                prefix,
                &functions,
            ),
            "object_array::jit_ll_arraycopy"
        );
    }

    #[test]
    fn marker_path_keeps_same_named_module_function() {
        let functions = HashSet::from(["stack_check::stack_check".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "pyre_interpreter::stack_check::_jit_look_inside_stack_check",
                "_jit_look_inside_",
                &functions,
            ),
            "stack_check::stack_check"
        );
    }

    #[test]
    fn marker_path_keeps_impl_body_local_parent() {
        let functions = HashSet::from(["rbigint::RBigInt::mul".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "majit_rlib::rbigint::RBigInt::mul::_elidable_function_mul",
                "_elidable_function_",
                &functions,
            ),
            "rbigint::RBigInt::mul"
        );
    }

    #[test]
    fn marker_path_keeps_oopspec_impl_body_local_parent() {
        let functions = HashSet::from(["rordereddict::<Impl>::lookup".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "pyre_object::rordereddict::<Impl>::lookup::oopspec_lookup",
                "oopspec_",
                &functions,
            ),
            "rordereddict::<Impl>::lookup"
        );
    }

    #[test]
    fn marker_path_keeps_jit_llop_impl_body_local_parent() {
        let functions = HashSet::from(["owner::Type::jit_force_quasi_immutable".to_string()]);
        assert_eq!(
            marker_path_to_fn_path(
                "mod::owner::Type::jit_force_quasi_immutable::_jit_llop_jit_force_quasi_immutable",
                "_jit_llop_",
                &functions,
            ),
            "owner::Type::jit_force_quasi_immutable"
        );
    }

    #[test]
    fn module_less_elidable_marker_is_not_harvested_as_bare_leaf() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/llbc/majit-rlib.ullbc"
        );
        let llbc = crate::test_support::load_ullbc(path);
        let harvested = super::harvest_hints_from_llbcs(std::slice::from_ref(llbc));
        let invert_keys: Vec<_> = harvested
            .iter()
            .filter(|(k, hints)| k.contains("invert") && hints.iter().any(|h| h == "elidable"))
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(
            !harvested.contains_key("invert"),
            "module-less Charon duplicate must not harvest as bare invert: invert_keys={invert_keys:?}"
        );
        assert!(
            !harvested.contains_key("neg"),
            "module-less Charon duplicate must not harvest as bare neg"
        );
        assert!(
            invert_keys
                .iter()
                .any(|k| k.ends_with("invert") && k.contains("rbigint")),
            "body-local RBigInt::invert marker: invert_keys={invert_keys:?}"
        );
        let neg_keys: Vec<_> = harvested
            .iter()
            .filter(|(k, hints)| {
                k.rsplit("::").next() == Some("neg") && hints.iter().any(|h| h == "elidable")
            })
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(
            neg_keys.iter().any(|k| k.contains("rbigint")),
            "body-local RBigInt::neg marker: neg_keys={neg_keys:?}"
        );
    }

    #[test]
    fn random_effects_marker_keys_the_graphless_leaf() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/llbc/majit-rlib.ullbc"
        );
        let llbc = crate::test_support::load_ullbc(path);
        let harvested = super::harvest_hints_from_llbcs(std::slice::from_ref(llbc));
        let keys: Vec<&str> = harvested
            .iter()
            .filter(|(_, hints)| hints.iter().any(|h| h == "gc_effects"))
            .map(|(k, _)| k.as_str())
            .collect();
        // `clibffi.c_ffi_call` releases the GIL; its local leaf is keyed
        // without the crate, as a call to it is analysed.
        assert!(
            keys.contains(&"clibffi::__rffi_fp_c_ffi_call"),
            "gc_effects keys: {keys:?}"
        );
        // `rposix_stat.c_stat` is `macro = true`: its leaf is the
        // `pypy_macro_wrapper_stat` funcptr. `rposix_stat` is `#[cfg(unix)]`.
        if cfg!(unix) {
            assert!(
                keys.contains(&"rposix_stat::__rffi_fp_c_stat"),
                "gc_effects keys: {keys:?}"
            );
        }
        assert!(
            !keys.iter().any(|k| k.starts_with("majit_rlib::")),
            "gc_effects keys: {keys:?}"
        );
    }

    #[test]
    fn immutable_entry_list_parses_every_rank_suffix() {
        // The payload shape `#[jit_immutable_fields]` writes into
        // `_immutable_fields_<Struct>`: the declared entries joined by
        // commas, each still carrying its `rclass.py:649-661` suffix.
        assert_eq!(
            parse_immutable_entry_list("_digits[*],_size,version?,defs_w?[*]"),
            vec![
                ("_digits".to_string(), ImmutableRank::ImmutableArray),
                ("_size".to_string(), ImmutableRank::Immutable),
                ("version".to_string(), ImmutableRank::QuasiImmutable),
                ("defs_w".to_string(), ImmutableRank::QuasiImmutableArray),
            ]
        );
    }

    #[test]
    fn immutable_entry_list_drops_blanks() {
        assert_eq!(
            parse_immutable_entry_list(" name , , instantiate "),
            vec![
                ("name".to_string(), ImmutableRank::Immutable),
                ("instantiate".to_string(), ImmutableRank::Immutable),
            ]
        );
        assert!(parse_immutable_entry_list("").is_empty());
    }

    fn empty_llbc() -> majit_charon_reader::Llbc {
        majit_charon_reader::Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn decode_str_const_reads_constant_expr_literal() {
        let llbc = empty_llbc();
        // `const X: &str = "spec"` is a hash-consed constant expression
        // `{"Value":[id, [{"Str": spec}, ty]]}`.
        let value = serde_json::json!({
            "Value": [1, [{"Str": "list.int_capacity(l)"}, {"Deduplicated": 7911}]]
        });
        assert_eq!(
            decode_str_const(&llbc, &value).as_deref(),
            Some("list.int_capacity(l)")
        );
        let inline = serde_json::json!([{"Str": "list.int_capacity(l)"}, {"Deduplicated": 7911}]);
        assert_eq!(
            decode_str_const(&llbc, &inline).as_deref(),
            Some("list.int_capacity(l)")
        );
    }

    #[test]
    fn decode_str_const_rejects_non_string() {
        let llbc = empty_llbc();
        assert_eq!(decode_str_const(&llbc, &serde_json::json!(42)), None);
        assert_eq!(
            decode_str_const(
                &llbc,
                &serde_json::json!([{"Bool": true}, {"Deduplicated": 1}])
            ),
            None
        );
    }

    #[test]
    fn aroundstate_operands_decode_the_funcptr_and_save_err() {
        // `(funcptr, save_err)`: an `FnDef` const and an `Integer` const,
        // each in the `[kind, ty]` `ConstantExpr` shape.
        let llbc = majit_charon_reader::Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[]}}"#,
        )
        .unwrap();
        let ty = serde_json::json!({"Deduplicated": 1});
        let funcptr = super::Operand::Const(serde_json::json!([
            {"FnDef": {"kind": {"Fun": 7}}},
            ty
        ]));
        let save_err = super::Operand::Const(serde_json::json!([
            {"Integer": {"Signed": ["I64", "5"]}},
            ty
        ]));
        let (mut fn_id, mut save) = (None, None);
        super::collect_aroundstate_operand(&llbc, &funcptr, &mut fn_id, &mut save);
        super::collect_aroundstate_operand(&llbc, &save_err, &mut fn_id, &mut save);
        assert_eq!(fn_id, Some(7));
        assert_eq!(save, Some(5));
    }

    #[test]
    fn scalar_pair_i64_reads_signed_and_unsigned() {
        let signed = serde_json::json!({"Signed": ["I64", "-3"]});
        let unsigned = serde_json::json!({"Unsigned": ["U32", "4"]});
        assert_eq!(super::scalar_pair_i64(&signed), Some(-3));
        assert_eq!(super::scalar_pair_i64(&unsigned), Some(4));
        assert_eq!(
            super::scalar_pair_i64(&serde_json::json!({"Bool": true})),
            None
        );
    }

    #[test]
    fn merge_hints_into_graph_unions_unroll_safe_onto_existing_tokens() {
        let mut g = crate::model::FunctionGraph::new("f");
        g.hints = vec!["elidable".into()];
        super::merge_hints_into_graph(&mut g, &["unroll_safe".to_string(), "elidable".to_string()]);
        assert_eq!(
            g.hints,
            vec!["elidable".to_string(), "unroll_safe".to_string()]
        );
        assert!(g.func.elidable);
        assert!(g.func.unroll_safe);
    }

    #[test]
    fn alloc_flavor_raw_identity_does_not_match_another_module_leaf() {
        // Hold the origin-registry lock from the start so leftover
        // CodeObject origins from sibling tests cannot qualify a bare
        // leaf. Tombstone until the unique-leaf case below
        // (`canonical_struct_name` of an empty module path stays the leaf).
        let _origins = crate::test_support::register_struct_origins_serialized(HashMap::from([(
            "CodeObject".to_string(),
            String::new(),
        )]));
        let raw = HashSet::from(["bytecode::CodeObject".to_string()]);
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "bytecode::CodeObject"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "rustpython_compiler_core::bytecode::CodeObject"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "*const rustpython_compiler_core::bytecode::CodeObject"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "&bytecode::CodeObject"
        ));
        for spelling in [
            "&mut bytecode::CodeObject",
            "Box<bytecode::CodeObject>",
            "NonNull<bytecode::CodeObject>",
            "Arc<bytecode::CodeObject>",
            "Rc<bytecode::CodeObject>",
            "AtomicPtr<bytecode::CodeObject>",
            "alloc::boxed::Box<bytecode::CodeObject>",
            "alloc::sync::Arc<bytecode::CodeObject>",
            "core::ptr::non_null::NonNull<bytecode::CodeObject>",
            "std::sync::atomic::AtomicPtr<bytecode::CodeObject>",
        ] {
            assert!(
                super::alloc_flavor_raw_contains(&raw, spelling),
                "{spelling}"
            );
            assert!(
                super::spelling_is_one_word_flavor_raw_ptr(&raw, spelling),
                "{spelling}"
            );
        }
        assert!(!super::spelling_is_one_word_flavor_raw_ptr(
            &raw,
            "bytecode::CodeObject"
        ));
        assert!(!super::alloc_flavor_raw_contains(
            &raw,
            "Box<dyn bytecode::CodeObject>"
        ));
        assert!(!super::spelling_is_one_word_flavor_raw_ptr(
            &raw,
            "Box<dyn Trait>"
        ));
        assert!(!super::spelling_is_one_word_flavor_raw_ptr(&raw, "&[u8]"));
        assert!(!super::spelling_is_one_word_flavor_raw_ptr(&raw, "&str"));
        assert!(!super::alloc_flavor_raw_contains(&raw, "other::CodeObject"));
        // A bare leaf does not guess from the raw-key subset.
        assert!(!super::alloc_flavor_raw_contains(&raw, "CodeObject"));
        majit_ir::descr::register_struct_origins(HashMap::from([(
            "CodeObject".to_string(),
            "bytecode".to_string(),
        )]));
        assert!(super::alloc_flavor_raw_contains(&raw, "CodeObject"));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "CodeObject<ConstantData>"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "&CodeObject<ConstantData>"
        ));
        majit_ir::descr::register_struct_origins(HashMap::from([(
            "Node".to_string(),
            String::new(),
        )]));
        let raw_node = HashSet::from(["a::Node".to_string()]);
        assert!(super::alloc_flavor_raw_contains(&raw_node, "a::Node"));
        assert!(!super::alloc_flavor_raw_contains(&raw_node, "Node"));
    }

    #[test]
    fn alloc_flavor_raw_enum_variant_inherits_enum_identity() {
        let raw = HashSet::from(["mapdict::MapNode".to_string()]);
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "mapdict::MapNode::Terminator"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "pyre_object::mapdict::MapNode::Terminator"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "*const pyre_object::mapdict::MapNode::Terminator"
        ));
        assert!(super::alloc_flavor_raw_contains(
            &raw,
            "&mapdict::MapNode::Terminator"
        ));
        assert!(super::alloc_flavor_raw_contains(&raw, "mapdict::MapNode"));
        assert!(!super::alloc_flavor_raw_contains(
            &raw,
            "mapdict::Other::Terminator"
        ));
        assert!(!super::alloc_flavor_raw_contains(&raw, "Terminator"));
    }

    fn names(variants: &[&str]) -> Vec<String> {
        variants.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn exception_class_override_empty_marker_keeps_declaration_order() {
        let variants = names(&["A", "B", "C"]);
        let flat = vec![
            ("A".to_string(), None),
            ("B".to_string(), None),
            ("C".to_string(), None),
        ];
        assert_eq!(parse_exception_class_overrides("E", "", &variants), flat);
        assert_eq!(
            parse_exception_class_overrides("E", " \n\t", &variants),
            flat
        );
    }

    #[test]
    fn exception_class_override_emits_parent_before_child() {
        let declared = names(&["TooManyMethod", "TooMany", "Missing"]);
        assert_eq!(
            parse_exception_class_overrides("ArgErr", "TooManyMethod(TooMany)", &declared),
            vec![
                ("TooMany".to_string(), None),
                ("TooManyMethod".to_string(), Some("TooMany".to_string())),
                ("Missing".to_string(), None),
            ]
        );
        let chain = names(&["C", "B", "A"]);
        assert_eq!(
            parse_exception_class_overrides("E", "C(B) B(A)", &chain),
            vec![
                ("A".to_string(), None),
                ("B".to_string(), Some("A".to_string())),
                ("C".to_string(), Some("B".to_string())),
            ]
        );
    }

    fn field_owner<'a>(
        links: &'a [(String, Option<String>)],
        variant: &'a str,
        declares: &[&str],
    ) -> &'a str {
        exception_class_field_owner(links, variant, |name| declares.contains(&name))
    }

    #[test]
    fn exception_class_field_owner_reaches_past_a_non_declaring_parent() {
        let variants = names(&["C", "B", "A"]);
        let links = parse_exception_class_overrides("E", "C(B) B(A)", &variants);
        // B does not declare. A does, so B's farthest declarer is A.
        // `ClassDef.get_owner` walks the whole `getmro()`.
        assert_eq!(field_owner(&links, "C", &["A", "C"]), "A");
        assert_eq!(field_owner(&links, "B", &["A", "C"]), "A");
        assert_eq!(field_owner(&links, "A", &["A", "C"]), "A");
        assert_eq!(field_owner(&links, "C", &["B", "C"]), "B");
        assert_eq!(field_owner(&links, "C", &[]), "C");
    }

    #[test]
    #[should_panic(expected = "cycle in override links")]
    fn exception_class_field_owner_panics_on_a_cycle() {
        let links = vec![
            ("A".to_string(), Some("B".to_string())),
            ("B".to_string(), Some("A".to_string())),
        ];
        let _ = exception_class_field_owner(&links, "A", |_| true);
    }

    #[test]
    fn exception_class_key_is_the_declaration() {
        assert_eq!(
            exception_class_key("argument::ArgErr", None),
            "argument::ArgErr"
        );
        assert_eq!(
            exception_class_key("argument::ArgErr", Some("Missing")),
            "argument::ArgErr::Missing"
        );
        assert_eq!(exception_class_key("synth::E", Some("C")), "synth::E::C");
        assert_ne!(
            exception_class_key("pkg::E", Some("AB")),
            exception_class_key("pkg::EA", Some("B"))
        );
    }

    #[test]
    fn exception_class_name_belongs_accepts_base_variant_and_ctor_path() {
        let hints = vec![super::ExceptionClassHint::new(
            "synth::E",
            "C(B)",
            vec!["A".into(), "B".into(), "C".into()],
        )];
        let belongs = |name: &str| super::exception_class_name_belongs(&hints, name);
        assert!(belongs("synth::E"));
        assert!(belongs("fixture::synth::E"));
        assert!(belongs("synth::E::A"));
        assert!(belongs("synth::E::B"));
        assert!(belongs("synth::E::C"));
        assert!(belongs("fixture::synth::E::B"));
        assert!(!belongs("E"));
        assert!(!belongs("EA"));
        assert!(!belongs("EB"));
        assert!(!belongs("EC"));
        assert!(!belongs("E::B"));
        assert!(!belongs("synth::T::Y"));
        assert!(!belongs("synth::E::Y"));
        assert!(!belongs(""));
        assert!(!super::exception_class_name_belongs(&[], "E"));
    }

    #[test]
    fn exception_class_enum_path_matches_crate_prefix_not_a_shorter_stored_key() {
        assert!(exception_class_enum_path_matches("E", "E"));
        assert!(exception_class_enum_path_matches("E", "fixture::E"));
        assert!(exception_class_enum_path_matches(
            "synth::E",
            "fixture::synth::E"
        ));
        // `strip_crate_prefix` drops the first segment, so a stored bare
        // key matches `synth::E`. The stored qualified key does not match
        // the bare leaf: the compare is stored-then-query only.
        assert!(exception_class_enum_path_matches("E", "synth::E"));
        assert!(!exception_class_enum_path_matches("synth::E", "E"));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn exception_class_override_unknown_variant_panics() {
        let _ = parse_exception_class_overrides("E", "Nope(A)", &names(&["A", "B"]));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn exception_class_override_unknown_parent_panics() {
        let _ = parse_exception_class_overrides("E", "A(Nope)", &names(&["A", "B"]));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn exception_class_override_self_parent_panics() {
        let _ = parse_exception_class_overrides("E", "A(A)", &names(&["A", "B"]));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn exception_class_override_cycle_panics() {
        let _ = parse_exception_class_overrides("E", "A(B) B(A)", &names(&["A", "B"]));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn exception_class_override_duplicate_panics() {
        let _ = parse_exception_class_overrides("E", "A(B) A(C)", &names(&["A", "B", "C"]));
    }

    #[test]
    fn exception_class_override_malformed_tokens_panic() {
        let variants = names(&["A", "B"]);
        for marker in ["A", "A(", "(B)", "A()", "A(B)C", "1A(B)", "A(B C)"] {
            let caught = std::panic::catch_unwind(|| {
                let _ = parse_exception_class_overrides("E", marker, &variants);
            });
            let err = caught.unwrap_err();
            let msg = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("_exception_class_ E:"),
                "{marker} panicked with {msg}"
            );
        }
    }

    fn item_meta(path: &[&str]) -> serde_json::Value {
        let span = serde_json::json!({"data": {"file_id": 0,
            "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
        serde_json::json!({
            "name": path.iter().map(|segment| serde_json::json!({"Ident": [segment, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    }

    fn enum_type(id: u64, path: &[&str], variants: &[&str]) -> serde_json::Value {
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(path),
            "kind": {"Enum": variants.iter().enumerate().map(|(index, name)| serde_json::json!({
                "name": name,
                "fields": [],
                "discriminant": {"Scalar": {"Unsigned": ["U8", index.to_string()]}}
            })).collect::<Vec<_>>()}
        })
    }

    fn struct_type(id: u64, path: &[&str]) -> serde_json::Value {
        let attr = serde_json::json!({
            "attributes": [], "inline": null, "rename": null, "public": true
        });
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(path),
            "kind": {"Struct": [{
                "name": "x",
                "ty": {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "attr_info": attr
            }]}
        })
    }

    fn marker_global(id: u64, path: &[&str], value: &str) -> serde_json::Value {
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(path),
            "global_kind": "NamedConst",
            "value": {"Value": [1, [{"Str": value}, {"Deduplicated": 1}]]}
        })
    }

    fn hint_llbc(
        types: Vec<serde_json::Value>,
        globals: Vec<serde_json::Value>,
    ) -> majit_charon_reader::Llbc {
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": types,
                "fun_decls": [],
                "global_decls": globals,
                "trait_decls": [],
                "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("hint fixture parses")
    }

    #[test]
    fn harvest_exception_class_keeps_declaration_order_and_intern_key() {
        let llbc = hint_llbc(
            vec![enum_type(1, &["fixture", "synth", "E"], &["C", "B", "A"])],
            vec![marker_global(
                1,
                &["fixture", "synth", "_exception_class_E"],
                "C(B)",
            )],
        );
        let hints = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
        assert_eq!(hints.len(), 1);
        let hint = &hints[0];
        assert_eq!(hint.enum_path, "synth::E");
        assert_eq!(hint.overrides, "C(B)");
        assert_eq!(hint.variants, names(&["C", "B", "A"]));
        let exception = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        assert_eq!(hint.classes.base.qualname(), "synth::E");
        assert_eq!(
            hint.classes.base.class_bases(),
            Some(std::slice::from_ref(&exception))
        );
        // `C(B)`: parent B before child C, then A. Declaration order stays C, B, A.
        let names_in_mint_order: Vec<&str> = hint
            .classes
            .variants
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names_in_mint_order, ["B", "C", "A"]);
        let b = &hint.classes.variants[0].1;
        let c = &hint.classes.variants[1].1;
        let a = &hint.classes.variants[2].1;
        assert_eq!(b.qualname(), "synth::E::B");
        assert_eq!(c.qualname(), "synth::E::C");
        assert_eq!(a.qualname(), "synth::E::A");
        assert_eq!(
            b.class_bases(),
            Some(std::slice::from_ref(&hint.classes.base))
        );
        assert_eq!(c.class_bases(), Some(std::slice::from_ref(b)));
        assert_eq!(
            a.class_bases(),
            Some(std::slice::from_ref(&hint.classes.base))
        );
        let again = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
        assert_ne!(again[0].classes.base, hint.classes.base);
        assert_eq!(hint.clone().classes.base, hint.classes.base);
    }

    #[test]
    fn harvest_exception_class_ignores_same_leaf_struct() {
        let llbc = hint_llbc(
            vec![
                struct_type(1, &["fixture", "other", "E"]),
                enum_type(2, &["fixture", "synth", "E"], &["A"]),
            ],
            vec![marker_global(
                1,
                &["fixture", "synth", "_exception_class_E"],
                "",
            )],
        );
        let hints = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].enum_path, "synth::E");
        assert_eq!(hints[0].variants, names(&["A"]));
    }

    #[test]
    fn harvest_exception_class_picks_the_marker_module() {
        let llbc = hint_llbc(
            vec![
                enum_type(1, &["fixture", "a", "E"], &["OnlyA"]),
                enum_type(2, &["fixture", "b", "E"], &["OnlyB"]),
            ],
            vec![marker_global(
                1,
                &["fixture", "a", "_exception_class_E"],
                "",
            )],
        );
        let hints = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
        assert_eq!(hints[0].enum_path, "a::E");
        assert_eq!(hints[0].variants, names(&["OnlyA"]));
    }

    #[test]
    fn crate_lowering_finish_copies_exception_class_hint() {
        let llbc = hint_llbc(
            vec![enum_type(1, &["fixture", "synth", "E"], &["C", "B", "A"])],
            vec![marker_global(
                1,
                &["fixture", "synth", "_exception_class_E"],
                "C(B)",
            )],
        );
        let gc = std::collections::HashSet::<majit_ir::descr::StructId>::new();
        let raw = std::collections::HashSet::<String>::new();
        let state = crate::front::mir::CrateLoweringState::new(
            &llbc,
            &std::collections::HashSet::new(),
            std::collections::HashMap::new(),
            Default::default(),
            &gc,
            &raw,
            None,
            None,
        );
        let program = state.finish(Vec::new());
        assert_eq!(program.exception_class_hints.len(), 1);
        assert_eq!(program.exception_class_hints[0].enum_path, "synth::E");
        assert_eq!(
            program.exception_class_hints[0].variants,
            names(&["C", "B", "A"])
        );
        assert_eq!(program.exception_class_hints[0].overrides, "C(B)");
        let cached = crate::front::mir::llbc_exception_class_hints(&llbc);
        assert_eq!(
            program.exception_class_hints[0].classes.base,
            cached[0].classes.base
        );
        assert_eq!(
            program.exception_class_hints[0].classes.variants[0].1,
            cached[0].classes.variants[0].1
        );
    }

    #[test]
    #[should_panic(expected = "_exception_class_ Gone:")]
    fn harvest_exception_class_missing_enum_panics() {
        let llbc = hint_llbc(
            vec![],
            vec![marker_global(1, &["fixture", "_exception_class_Gone"], "")],
        );
        let _ = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn harvest_exception_class_bad_override_panics() {
        let llbc = hint_llbc(
            vec![enum_type(1, &["fixture", "E"], &["A"])],
            vec![marker_global(
                1,
                &["fixture", "_exception_class_E"],
                "Nope(A)",
            )],
        );
        let _ = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E:")]
    fn harvest_exception_class_duplicate_hierarchy_panics() {
        let llbc = hint_llbc(
            vec![enum_type(1, &["fixture", "E"], &["A"])],
            vec![
                marker_global(1, &["fixture", "a", "_exception_class_E"], ""),
                marker_global(2, &["fixture", "b", "_exception_class_E"], ""),
            ],
        );
        let _ = harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&llbc));
    }

    #[test]
    #[should_panic(expected = "exception class hints queried before the program-wide seed")]
    fn seed_program_exception_class_hints_after_a_query_panics() {
        let llbc = hint_llbc(
            vec![enum_type(1, &["fixture", "synth", "E"], &["A"])],
            vec![marker_global(
                1,
                &["fixture", "synth", "_exception_class_E"],
                "",
            )],
        );
        let _ = crate::front::mir::llbc_exception_class_hints(&llbc);
        crate::front::mir::seed_program_exception_class_hints(&llbc, Vec::new());
    }

    #[test]
    #[should_panic(expected = "_exception_class_ E: duplicate hierarchy `synth::E`")]
    fn extend_program_exception_class_hints_rejects_a_repeated_enum_path() {
        let artefact = |marker_id| {
            hint_llbc(
                vec![enum_type(1, &["fixture", "synth", "E"], &["A"])],
                vec![marker_global(
                    marker_id,
                    &["fixture", "synth", "_exception_class_E"],
                    "",
                )],
            )
        };
        let first = artefact(1);
        let second = artefact(2);
        let mut acc = Vec::new();
        extend_program_exception_class_hints(
            &mut acc,
            harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&first)),
        );
        extend_program_exception_class_hints(
            &mut acc,
            harvest_exception_class_hints_from_llbcs(std::slice::from_ref(&second)),
        );
    }

    fn minted_emit(name: &str, raw_owned: bool) -> super::MintedExceptionClassEmit {
        super::MintedExceptionClassEmit {
            name: name.to_string(),
            parent: None,
            size: 0,
            gc_ptr_offsets: Vec::new(),
            raw_owned: if raw_owned {
                vec![super::MintedRawOwnedDrop {
                    offset: 8,
                    type_path: "fixture::Signature".into(),
                }]
            } else {
                Vec::new()
            },
        }
    }

    /// `pkg::E::A` and `pkg::E__A` both sanitize to `pkg__E__A`.
    /// `NameManager.uniquename` keeps the first claim and gives the
    /// second the `_1` suffix from `NameManager._ensure_unique`.
    #[test]
    fn minted_exception_sanitized_collision_uses_uniquename_suffix() {
        let rows = vec![
            minted_emit("pkg::E::A", true),
            minted_emit("pkg::E__A", true),
        ];
        let src = super::render_minted_exception_classes_rs(&rows);
        assert_eq!(src, super::render_minted_exception_classes_rs(&rows));
        let first = src
            .find("static MINTED_EXC_pkg__E__A:")
            .expect("first ident");
        let second = src
            .find("static MINTED_EXC_pkg__E__A_1:")
            .expect("second ident");
        assert!(first < second, "{src}");
        assert!(src.contains("pytype: &MINTED_EXC_pkg__E__A,"), "{src}");
        assert!(src.contains("pytype: &MINTED_EXC_pkg__E__A_1,"), "{src}");
        assert!(
            src.contains("unsafe fn MINTED_EXC_pkg__E__A_destructor("),
            "{src}"
        );
        assert!(
            src.contains("unsafe fn MINTED_EXC_pkg__E__A_1_destructor("),
            "{src}"
        );
        assert!(
            src.contains("destructor: Some(MINTED_EXC_pkg__E__A_destructor)"),
            "{src}"
        );
        assert!(
            src.contains("destructor: Some(MINTED_EXC_pkg__E__A_1_destructor)"),
            "{src}"
        );
    }

    /// A later intern key that sanitizes to an earlier row's
    /// `_destructor` companion gets its own `_1` suffix.
    #[test]
    fn minted_exception_destructor_claim_disambiguates_later_ident() {
        let rows = vec![
            minted_emit("pkg::E::A", true),
            minted_emit("pkg::E::A_destructor", false),
        ];
        let src = super::render_minted_exception_classes_rs(&rows);
        assert!(
            src.contains("unsafe fn MINTED_EXC_pkg__E__A_destructor("),
            "{src}"
        );
        assert!(
            src.contains("static MINTED_EXC_pkg__E__A_destructor_1:"),
            "{src}"
        );
        assert!(src.contains("pytype: &MINTED_EXC_pkg__E__A,"), "{src}");
        assert!(
            src.contains("pytype: &MINTED_EXC_pkg__E__A_destructor_1,"),
            "{src}"
        );
        assert!(
            !src.contains("static MINTED_EXC_pkg__E__A_destructor:"),
            "{src}"
        );
    }

    /// `NameManager.uniquename` with omitted `with_number` (`gensupp.py`)
    /// sets `with_number = basename in ('v', 'w_')` after `C_IDENTIFIER`.
    /// A key that sanitizes to `v` is claimed as `v0` (`fmt = '%s%d'`).
    /// `NameManager.global_prefix` then prepends `MINTED_EXC_`.
    #[test]
    fn minted_exception_v_key_takes_default_with_number() {
        let rows = vec![minted_emit("v", false)];
        let src = super::render_minted_exception_classes_rs(&rows);
        assert!(src.contains("static MINTED_EXC_v0:"), "{src}");
        assert!(src.contains("pytype: &MINTED_EXC_v0,"), "{src}");
        assert!(!src.contains("static MINTED_EXC_v:"), "{src}");
        assert!(!src.contains("MINTED_EXC_v_0"), "{src}");
    }

    /// Item name on a rendered `static` or `fn` line.
    fn rendered_item_ident(line: &str) -> Option<&str> {
        let rest = [
            "pub unsafe extern \"C\" fn ",
            "unsafe extern \"C\" fn ",
            "pub unsafe fn ",
            "pub static ",
            "pub fn ",
            "unsafe fn ",
            "static ",
            "fn ",
        ]
        .into_iter()
        .find_map(|prefix| line.strip_prefix(prefix))?;
        let end = rest.find(|ch: char| ch == ':' || ch == '(')?;
        Some(&rest[..end])
    }

    /// `MINTED_EXCEPTION_CLASSES`, `MintedExceptionClass`, and
    /// `EPTION_CLASSES` still emit only `MINTED_EXC_` `static` and `fn`
    /// idents. `MINTED_EXCEPTION_CLASSES` is defined once, as the table.
    #[test]
    fn minted_exception_row_idents_keep_the_prefix() {
        let rows = vec![
            minted_emit("MINTED_EXCEPTION_CLASSES", true),
            minted_emit("MintedExceptionClass", true),
            minted_emit("EPTION_CLASSES", true),
        ];
        let src = super::render_minted_exception_classes_rs(&rows);
        let mut statics = 0usize;
        let mut fns = 0usize;
        for line in src.lines() {
            let Some(ident) = rendered_item_ident(line) else {
                continue;
            };
            if line.starts_with("static ") {
                statics += 1;
            } else if line.starts_with("unsafe extern \"C\" fn ")
                || line.starts_with("unsafe fn ")
                || line.starts_with("fn ")
            {
                fns += 1;
            } else {
                continue;
            }
            assert!(ident.starts_with("MINTED_EXC_"), "{ident}\n{src}");
        }
        // Each raw-owned row emits a pytype static and a `HELPER_FNADDRS`
        // static, plus a destructor, an `alloc_with_del` extern, and a
        // wasm32 ctor.
        assert_eq!(statics, rows.len() * 2, "{src}");
        assert_eq!(fns, rows.len() * 3, "{src}");
        let definitions: Vec<&str> = src
            .lines()
            .filter(|line| rendered_item_ident(line) == Some("MINTED_EXCEPTION_CLASSES"))
            .collect();
        assert_eq!(
            definitions,
            ["pub static MINTED_EXCEPTION_CLASSES: &[MintedExceptionClass] = &["],
            "{src}"
        );
    }
}
