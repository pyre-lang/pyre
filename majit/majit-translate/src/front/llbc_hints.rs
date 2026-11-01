//! Harvest JIT-hint markers from the ullbc surrogate consts the
//! `majit_macros` proc-macros emit (`_elidable_function_<NAME>`,
//! `_jit_look_inside_<NAME>`, `_jit_loop_invariant_<NAME>`,
//! `_jit_unroll_safe_<NAME>`, `_not_rpython_<NAME>`, `oopspec_<NAME>`,
//! `_gctransformer_hint_close_stack_<NAME>`,
//! `_call_aroundstate_target_<NAME>`,
//! `_random_effects_on_gcobjs_<NAME>`,
//! `_callable_elidable_function_<NAME>`,
//! `_funcptr_canraise_false_<NAME>`,
//! `_jit_llop_<NAME>`).
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
                        gd.rest
                            .get("value")
                            .map(|v| v.to_string())
                            .unwrap_or_default()
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
fn alloc_flavor_class_intern_key(type_name_path: &str) -> String {
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
    if let Some(value) = gd.rest.get("value")
        && let Some(b) = decode_bool_const(llbc, value)
    {
        return Some(b);
    }
    let init_id = marker_init_fun_id(gd)?;
    bool_assigned_to_return(llbc, init_id)
}

pub(crate) fn marker_init_fun_id(gd: &GlobalDecl) -> Option<u64> {
    let value = gd.rest.get("value")?;
    let body = value
        .get("Value")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.get(1))
        .unwrap_or(value);
    let lit = body.as_array()?.first()?;
    lit.pointer("/Call/0/kind/Fun")
        .and_then(serde_json::Value::as_u64)
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
fn global_marker_str(llbc: &Llbc, gd: &GlobalDecl) -> Option<String> {
    if let Some(value) = gd.rest.get("value")
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
    use super::{decode_str_const, marker_path_to_fn_path, parse_immutable_entry_list};
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
        let llbc = majit_charon_reader::Llbc::load(path).expect("load majit-rlib.ullbc");
        let harvested = super::harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
        let invert_keys: Vec<_> = harvested
            .iter()
            .filter(|(k, hints)| k.contains("invert") && hints.iter().any(|h| h == "elidable"))
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(
            harvested.get("invert").is_none(),
            "module-less Charon duplicate must not harvest as bare invert: invert_keys={invert_keys:?}"
        );
        assert!(
            harvested.get("neg").is_none(),
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
        let llbc = majit_charon_reader::Llbc::load(path).expect("load majit-rlib.ullbc");
        let harvested = super::harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
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
        // `rposix_stat.c_stat` is `macro = libc::stat`; the foreign leaf
        // keeps its crate. `rposix_stat` is `#[cfg(unix)]`.
        if cfg!(unix) {
            assert!(
                keys.iter()
                    .any(|k| k.starts_with("libc::") && k.ends_with("::stat")),
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
}
