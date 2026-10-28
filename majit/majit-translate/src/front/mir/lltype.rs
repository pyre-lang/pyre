//! Charon type → lltype / rtyper type mapping, and trait/impl resolution.

use super::*;

/// Free-function version of [`Lowering::impl_method_owner`] for callers
/// that only have the `Llbc` + `FunDecl` and do not want to instantiate
/// a full `Lowering` context just to ask the question.  Used by
/// `build_semantic_program_from_llbc` to populate
/// `SemanticFunction.self_ty_root` on the canonical SemanticProgram
/// produced by the MIR driver.
///
/// Mirrors the instance method line-for-line; any change here must be
/// kept in sync with the `&self` version.
/// `func.__module__` for a lowered function: the crate-qualified leading
/// module segments of its Charon name, spelled as `module_path!()` spells
/// them (`pyre_module::module::unicodedata::interp_ucd`).
///
/// A method's name is `[crate, mod.., <Impl>, method]` and ends its module
/// at the impl segment; a free function's is `[crate, mod.., fn]` and ends
/// it at the leaf.  Charon's `Ident` segments do not say whether they name
/// a module, so an item nested under another `Ident` — a trait default
/// method, a closure, a fn inside a fn — keeps that enclosing segment.
/// `JitPolicy.look_inside_function` subclasses test the path by prefix, and
/// a trailing segment does not move a prefix test.
pub(in crate::front::mir) fn fundecl_module(fd: &FunDecl) -> Option<String> {
    let name: Vec<&NameSeg> = fd.item_meta.template_name().collect();
    let mut segs: Vec<&str> = name
        .iter()
        .map_while(|seg| match seg {
            NameSeg::Ident { ident: (s, _) } => Some(s.as_str()),
            NameSeg::Other(_) => None,
        })
        .collect();
    if segs.len() == name.len() {
        segs.pop();
    }
    (!segs.is_empty()).then(|| segs.join("::"))
}

/// Class a `repr(transparent)` GC handle's methods are registered on.
///
/// The handle intern's as a subclass of `Deref::Target` and keeps its own
/// fields, so methods stay on the wrapper (`error::PyError`). Painting the
/// target made getattr of `.0` / a `MutRef<Handle>.value` resolve on the
/// payload.
pub(in crate::front::mir) fn gc_handle_method_class_root(
    _llbc: &Llbc,
    _fd: &FunDecl,
    _adt_def_id: u64,
    _tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    None
}

pub(in crate::front::mir) fn impl_method_owner_for_fundecl(
    llbc: &Llbc,
    fd: &FunDecl,
) -> Option<(String, String)> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    let leaf = match &segs[last_idx] {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let owner_qualified = match resolve_impl_owner_adt_def_id_free(llbc, impl_payload) {
        Some(adt_def_id) => {
            let td = llbc.type_by_id(adt_def_id)?;
            // Owner-qualification convention: bare ident qualified by the
            // type's defining module path (e.g. `gc_roots::RootScope`).  Strip
            // the crate name from the TypeDecl's full name_path so the
            // `self_ty_root` keys land on a `[module::Owner, method]` CallPath.
            // Without this qualification the canonical registration loop at
            // `lib.rs` cannot find the graph keyed by
            // `[qualified_owner, method]`.
            let owner = strip_crate_prefix(&td.item_meta.name_path());
            match impl_atomic_alias_leaf(llbc, impl_payload) {
                Some(alias) => match owner.rsplit_once("::") {
                    Some((module, _)) => format!("{module}::{alias}"),
                    None => alias,
                },
                None => gc_handle_method_class_root(llbc, fd, adt_def_id, no_tombstoned_leaves())
                    .unwrap_or(owner),
            }
        }
        // Non-ADT `Self` has no TypeDecl and no `CallTarget::Method`
        // owner.  `is_null` is not registered here: the front lowers
        // `<*const T>::is_null` / `<*mut T>::is_null` to `is_`
        // (`is_raw_ptr_is_null`) instead of a method key. A primitive
        // trait impl (`PartialOrd<char> for char`) is distinguished by
        // `spec_segments` via the impl's trait-ref identity, not by minting a
        // `self_ty_root` that would rewrite `bool::then` / `f64::from`.
        None => return None,
    };
    if owner_qualified.is_empty() {
        return None;
    }
    Some((owner_qualified, leaf))
}

/// The call path of a provided trait method (`src: {"TraitDefault": ..}`)
/// called at a concrete `Self`: `[owner(Self), leaf]`, qualified like an
/// impl owner.  The default body is generic over `Self`, so the call site's
/// first type argument names the type whose impl inherits the method —
/// `PartialOrd::gt` at `RBigInt` is `rbigint::RBigInt::gt`.  `None` for any
/// other callee, or when `Self` is not a nominal ADT.
pub(in crate::front::mir) fn trait_default_call_segments(
    llbc: &Llbc,
    fd: &FunDecl,
    generics: &serde_json::Value,
) -> Option<Vec<String>> {
    fd.src.as_ref()?.get("TraitDefault")?;
    let self_ty = generics.get("types")?.as_array()?.first()?;
    let td = llbc.type_by_id(resolve_tyexpr_to_adt_def_id_free(llbc, self_ty)?)?;
    let owner = strip_crate_prefix(&td.item_meta.name_path());
    if owner.is_empty() {
        return None;
    }
    let leaf = match fd.item_meta.template_name().last()? {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    let mut segments: Vec<String> = owner.split("::").map(str::to_string).collect();
    segments.push(leaf);
    Some(segments)
}

pub(in crate::front::mir) fn tyref_as_json(ty: &TyRef) -> serde_json::Value {
    match ty {
        TyRef::Dedup { id } => serde_json::json!({ "Deduplicated": id }),
        TyRef::Inline { value: (id, body) } => serde_json::json!({ "Value": [id, body] }),
        TyRef::Other(body) => body.clone(),
    }
}

pub(in crate::front::mir) fn depth0_typevar_index(node: &serde_json::Value) -> Option<usize> {
    let var = node.get("TypeVar")?;
    if let Some(bound) = var.get("Bound").and_then(serde_json::Value::as_array) {
        if bound.first()?.as_u64()? != 0 {
            return None;
        }
        return Some(bound.get(1)?.as_u64()? as usize);
    }
    var.get("Free")
        .and_then(serde_json::Value::as_u64)
        .map(|index| index as usize)
}

pub(in crate::front::mir) fn depth0_constvar_index(node: &serde_json::Value) -> Option<usize> {
    if let Some(n) = node.get("Var").and_then(serde_json::Value::as_u64) {
        return Some(n as usize);
    }
    for key in ["Var", "ConstVar", "TypeVar"] {
        let Some(var) = node.get(key) else {
            continue;
        };
        if let Some(bound) = var.get("Bound").and_then(serde_json::Value::as_array) {
            if bound.first()?.as_u64()? != 0 {
                return None;
            }
            return Some(bound.get(1)?.as_u64()? as usize);
        }
        if let Some(index) = var.get("Free").and_then(serde_json::Value::as_u64) {
            return Some(index as usize);
        }
        if let Some(index) = var.as_u64() {
            return Some(index as usize);
        }
    }
    None
}

pub(in crate::front::mir) fn ref_inner_and_kind(
    r: &serde_json::Value,
) -> (Option<&serde_json::Value>, Option<&str>) {
    if let Some(arr) = r.as_array() {
        (arr.get(1), arr.get(2).and_then(serde_json::Value::as_str))
    } else if let Some(obj) = r.as_object() {
        (
            obj.get("ty"),
            obj.get("kind").and_then(serde_json::Value::as_str),
        )
    } else {
        (None, None)
    }
}

pub(in crate::front::mir) fn json_ref_kind_is_mut(kind: Option<&str>) -> bool {
    kind.is_some_and(|k| k.eq_ignore_ascii_case("Mut"))
}

/// Unify `pattern` (impl `Self`, or a nested type argument) against the
/// call receiver. Shared vs Mut vs owned wrappers stay distinct:
/// `strip_ty_wrappers` would over-peel `&&T`. `ClassDesc.lookup`.
pub(in crate::front::mir) fn unify_receiver_ty(
    pattern: &serde_json::Value,
    actual: &serde_json::Value,
    subst: &mut Vec<Option<serde_json::Value>>,
    const_subst: &mut Vec<Option<serde_json::Value>>,
    llbc: &Llbc,
) -> bool {
    let Some(pattern) = strip_ty_indirections(pattern, llbc) else {
        return false;
    };
    let Some(actual) = strip_ty_indirections(actual, llbc) else {
        return false;
    };
    if let Some(index) = depth0_typevar_index(pattern) {
        if index >= subst.len() {
            subst.resize(index + 1, None);
        }
        match &subst[index] {
            Some(bound) => {
                let bound = bound.clone();
                let Some(bound) = strip_ty_indirections(&bound, llbc) else {
                    return false;
                };
                bound == actual
            }
            None => {
                subst[index] = Some(actual.clone());
                true
            }
        }
    } else if let (Some(pref), Some(aref)) = (pattern.get("Ref"), actual.get("Ref")) {
        let (pty, pkind) = ref_inner_and_kind(pref);
        let (aty, akind) = ref_inner_and_kind(aref);
        if json_ref_kind_is_mut(pkind) != json_ref_kind_is_mut(akind) {
            return false;
        }
        match (pty, aty) {
            (Some(pty), Some(aty)) => unify_receiver_ty(pty, aty, subst, const_subst, llbc),
            _ => false,
        }
    } else if pattern.get("Ref").is_some() || actual.get("Ref").is_some() {
        false
    } else if let (Some(ps), Some(as_)) = (pattern.get("Scalar"), actual.get("Scalar")) {
        charon_literal_to_ast_string(ps) == charon_literal_to_ast_string(as_)
    } else if pattern.get("Scalar").is_some() || actual.get("Scalar").is_some() {
        false
    } else if let (Some(pa), Some(aa)) = (
        pattern.get("Adt").and_then(serde_json::Value::as_object),
        actual.get("Adt").and_then(serde_json::Value::as_object),
    ) {
        match (type_decl_ref_adt_id(pa), type_decl_ref_adt_id(aa)) {
            (Some(pid), Some(aid)) => {
                if pid != aid && !same_nominal_adt(llbc, pid, aid) {
                    return false;
                }
            }
            (None, None) => {
                if pa.get("builtin") != aa.get("builtin") {
                    return false;
                }
            }
            _ => return false,
        }
        if !unify_const_generic_lists(
            pa.get("generics")
                .and_then(|g| g.get("const_generics"))
                .and_then(serde_json::Value::as_array),
            aa.get("generics")
                .and_then(|g| g.get("const_generics"))
                .and_then(serde_json::Value::as_array),
            const_subst,
        ) {
            return false;
        }
        unify_generic_type_lists(
            pa.get("generics")
                .and_then(|g| g.get("types"))
                .and_then(serde_json::Value::as_array),
            aa.get("generics")
                .and_then(|g| g.get("types"))
                .and_then(serde_json::Value::as_array),
            subst,
            const_subst,
            llbc,
        )
    } else {
        pattern == actual
    }
}

pub(in crate::front::mir) fn unify_const_generic_lists(
    pattern: Option<&Vec<serde_json::Value>>,
    actual: Option<&Vec<serde_json::Value>>,
    const_subst: &mut Vec<Option<serde_json::Value>>,
) -> bool {
    let empty: Vec<serde_json::Value> = Vec::new();
    let pattern = pattern.unwrap_or(&empty);
    let actual = actual.unwrap_or(&empty);
    if pattern.len() != actual.len() {
        return false;
    }
    pattern.iter().zip(actual.iter()).all(|(pattern, actual)| {
        if let Some(index) = depth0_constvar_index(pattern) {
            if index >= const_subst.len() {
                const_subst.resize(index + 1, None);
            }
            match &const_subst[index] {
                Some(bound) => bound == actual,
                None => {
                    const_subst[index] = Some(actual.clone());
                    true
                }
            }
        } else {
            pattern == actual
        }
    })
}

pub(in crate::front::mir) fn unify_generic_type_lists(
    pattern: Option<&Vec<serde_json::Value>>,
    actual: Option<&Vec<serde_json::Value>>,
    subst: &mut Vec<Option<serde_json::Value>>,
    const_subst: &mut Vec<Option<serde_json::Value>>,
    llbc: &Llbc,
) -> bool {
    let empty: Vec<serde_json::Value> = Vec::new();
    let pattern = pattern.unwrap_or(&empty);
    let actual = actual.unwrap_or(&empty);
    if pattern.len() != actual.len() {
        return false;
    }
    pattern
        .iter()
        .zip(actual.iter())
        .all(|(p, a)| unify_receiver_ty(p, a, subst, const_subst, llbc))
}

pub(in crate::front::mir) fn types_equal_after_subst(
    pattern: &serde_json::Value,
    actual: &serde_json::Value,
    subst: &[Option<serde_json::Value>],
    llbc: &Llbc,
) -> bool {
    let mut pattern = pattern.clone();
    let types: Vec<serde_json::Value> = subst
        .iter()
        .map(|bound| bound.clone().unwrap_or(serde_json::Value::Null))
        .collect();
    crate::front::clause_spec::substitute_type_vars(&mut pattern, llbc, &types, &[]);
    unify_receiver_ty(&pattern, actual, &mut Vec::new(), &mut Vec::new(), llbc)
}

pub(in crate::front::mir) fn trait_generic_args_match(
    impl_trait: &serde_json::Value,
    call_types: Option<&Vec<serde_json::Value>>,
    subst: &[Option<serde_json::Value>],
    llbc: &Llbc,
) -> bool {
    let Some(impl_types) = impl_trait
        .pointer("/generics/types")
        .and_then(serde_json::Value::as_array)
    else {
        return call_types.is_none_or(|types| types.is_empty());
    };
    let Some(call_types) = call_types else {
        return true;
    };
    if call_types.is_empty() {
        return true;
    }
    let impl_slice: &[serde_json::Value] = if call_types.len() + 1 == impl_types.len() {
        &impl_types[1..]
    } else if call_types.len() == impl_types.len() {
        impl_types
    } else {
        return false;
    };
    impl_slice
        .iter()
        .zip(call_types.iter())
        .all(|(pattern, actual)| types_equal_after_subst(pattern, actual, subst, llbc))
}

pub(in crate::front::mir) fn reconstructed_impl_generics(
    subst: &[Option<serde_json::Value>],
    const_subst: &[Option<serde_json::Value>],
    row: &serde_json::Value,
) -> Option<serde_json::Value> {
    let type_n = row
        .pointer("/generics/types")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .unwrap_or(subst.len());
    let const_n = row
        .pointer("/generics/const_generics")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .unwrap_or(const_subst.len());
    let types = filled_subst(subst, type_n)?;
    let const_generics = filled_subst(const_subst, const_n)?;
    let trait_refs = row
        .pointer("/generics/trait_refs")
        .cloned()
        .unwrap_or_else(|| serde_json::json!([]));
    Some(serde_json::json!({
        "types": types,
        "const_generics": const_generics,
        "trait_refs": trait_refs,
    }))
}

pub(in crate::front::mir) fn filled_subst(
    subst: &[Option<serde_json::Value>],
    n: usize,
) -> Option<Vec<serde_json::Value>> {
    if subst.len() < n {
        return None;
    }
    subst[..n]
        .iter()
        .cloned()
        .collect::<Option<Vec<serde_json::Value>>>()
}

pub(in crate::front::mir) fn merge_impl_and_method_generics(
    impl_generics: &serde_json::Value,
    method_generics: &serde_json::Value,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for key in ["types", "const_generics", "trait_refs"] {
        let mut items = impl_generics
            .get(key)
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if let Some(more) = method_generics
            .get(key)
            .and_then(serde_json::Value::as_array)
        {
            items.extend(more.iter().cloned());
        }
        out.insert(key.to_string(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(out)
}

/// Unique impl of `trait_id` whose method first input unifies with the
/// call receiver, including every reference layer and trait generic args.
/// Const-generic bindings come from that unification
/// (`FunctionDesc.cachedgraph`). Zero or several matches stay unresolved
/// (`ClassDesc.lookup`).
pub(in crate::front::mir) fn trait_impl_match_for_receiver(
    first_arg_ty: &TyRef,
    payload: &serde_json::Value,
    llbc: &Llbc,
) -> Option<(u64, u64, serde_json::Value)> {
    let recv = tyref_as_json(first_arg_ty);
    let arr = payload.as_array()?;
    let trait_id = trait_ref_decl_id(arr.first()?, llbc)?;
    let method_idx = arr.get(1)?.as_u64()?;
    let call_types = traitref_unwrap(arr.first()?, llbc, 0)
        .and_then(crate::front::clause_spec::trait_decl_pred)
        .and_then(|pred| pred.pointer("/generics/types"))
        .and_then(serde_json::Value::as_array);
    let mut found = None;
    for (i, row) in llbc.trait_impl_rows_of(trait_id) {
        let Some(impl_trait) = row.get("impl_trait") else {
            continue;
        };
        let impl_id = i as u64;
        let Some(fn_id) = crate::front::clause_spec::impl_method_fn_id(llbc, impl_id, method_idx)
            .or_else(|| trait_default_method_id(llbc, trait_id, method_idx))
        else {
            continue;
        };
        let Some(fd) = llbc.fn_by_id(fn_id) else {
            continue;
        };
        let Some(first_input) = fd.signature.inputs.first() else {
            continue;
        };
        let pattern = tyref_as_json(first_input);
        let mut subst = Vec::new();
        let mut const_subst = Vec::new();
        if !unify_receiver_ty(&pattern, &recv, &mut subst, &mut const_subst, llbc) {
            continue;
        }
        if !trait_generic_args_match(impl_trait, call_types, &subst, llbc) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let Some(impl_generics) = reconstructed_impl_generics(&subst, &const_subst, row) else {
            return None;
        };
        found = Some((impl_id, fn_id, impl_generics));
    }
    found
}

/// Impl (or trait-default) FunDecl a `CallKind::Trait` names when the
/// receiver unifies with exactly one impl method. `description.py`
/// `ClassDesc.lookup`: the method is looked up on that class; a missing
/// override inherits the default. Returns the FunDecl id, the path
/// `spec_segments` would register, and the impl generics reconstructed
/// from that unification (`FunctionDesc.cachedgraph` / `specialize`).
/// `None` when no unique impl matches, or the method has no body.
pub(in crate::front::mir) fn trait_method_on_known_receiver(
    first_arg_ty: Option<&TyRef>,
    payload: &serde_json::Value,
    llbc: &Llbc,
) -> Option<(u64, Vec<String>, serde_json::Value)> {
    let ty = first_arg_ty?;
    let (_, fn_id, impl_generics) = trait_impl_match_for_receiver(ty, payload, llbc)?;
    let fd = llbc.fn_by_id(fn_id)?;
    if !crate::front::clause_spec::decl_has_unstructured_body(fd) {
        return None;
    }
    let path = fd.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or("fn");
    Some((fn_id, spec_segments(llbc, fd, leaf), impl_generics))
}

/// The std alias an inherent `impl Atomic<T>` block's methods are spelled
/// under: `AtomicU8` for `Atomic<u8>`, `AtomicPtr` for `Atomic<*mut T>`.
/// `core::sync::atomic` declares each width as a type alias of `Atomic<T>`
/// (`pub type AtomicU8 = Atomic<u8>`) with its own inherent impl, so the
/// type argument, not the generic `Atomic` decl, names the owner.  `None`
/// for any other impl.
pub(in crate::front::mir) fn impl_atomic_alias_leaf(
    llbc: &Llbc,
    impl_payload: &serde_json::Value,
) -> Option<String> {
    let self_ty = impl_payload.get("Ty")?.get("skip_binder")?;
    let self_ty: TyRef = serde_json::from_value(self_ty.clone()).ok()?;
    if tyref_atomic_is_ptr(&self_ty, llbc) {
        return Some("AtomicPtr".to_string());
    }
    let scalar = tyref_atomic_inner_scalar_str(&self_ty, llbc)?;
    let mut chars = scalar.chars();
    let first = chars.next()?.to_ascii_uppercase();
    Some(format!("Atomic{first}{}", chars.as_str()))
}

/// Path segments used to *call* a function named by a `FnDef` constant or
/// type — the same spelling `decode_constant` stores in a function-item
/// value and `call_target_segments` emits for a direct call.  The leaf is
/// [`instance_leaf`] when Charon instantiated the declaration, matching
/// [`registered_path_for_fun_decl`] (`FunctionDesc.cachedgraph`).
pub(in crate::front::mir) fn fundecl_fn_item_segments(llbc: &Llbc, fd: &FunDecl) -> Vec<String> {
    registered_path_for_fun_decl(llbc, fd).segments
}

/// For a `Deref` / `DerefMut` trait-impl method, resolve the leaf
/// identifier of the implementing `Self` ADT (`Box`, `Rc`, `Arc`, …)
/// directly from the impl's `Self` type, bypassing the
/// registry-keyed `impl_method_owner_for_fundecl` (which resolves only
/// self-receiver methods it can key into `CallRegistry`).  Used to
/// subtract the `cast_pointer` thin-pointer rewrite for the
/// header-offset handles whose word is not the pointee address.
/// Returns `None` when the owner cannot be resolved, in which case the
/// caller keeps the ordinary thin-pointer treatment.
pub(in crate::front::mir) fn deref_impl_owner_leaf(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let path = impl_owner_path(llbc, fd)?;
    Some(path.rsplit("::").next().unwrap_or(&path).to_string())
}

/// The full `name_path` of the ADT whose inherent or trait impl owns `fd`.
pub(in crate::front::mir) fn impl_owner_path(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let adt_def_id = resolve_impl_owner_adt_def_id_free(llbc, impl_payload)?;
    Some(llbc.type_by_id(adt_def_id)?.item_meta.name_path())
}

/// Collect, from the lowered MIR,
/// `(path-segments, Signature, FUNC.RESULT token)` for every local `unsafe
/// fn` / unsafe impl-method whose return type projects to a token
/// [`crate::translator::rtyper::cutover::residual_return_shell`] can model.
/// Being `unsafe` does not hold a body back from lowering — no gate reads
/// `signature.is_unsafe` outside this collector.  What an unsafe callee is
/// missing is a registry key under the spelling its
/// `OpKind::Call::FunctionPath` sites emit; without one the dual gate Skips
/// with "not registered in CallRegistry".
///
/// A stub supplies that key and nothing else: the entry's graph is a
/// one-link return shell carrying the declared result, so what registration
/// changes is whether the *caller* can annotate past the call.  That is why the
/// result token is projected through the same
/// [`dont_look_inside_return_token`] the `@jit.dont_look_inside` residual
/// path uses: both describe one residual boundary's declared result, and
/// keeping one projection keeps the two in step.
///
/// The single registration key must equal the call-site lookup. A free
/// fn keys as the crate-included `name_path()` split on every `::`
/// (`["pyre_interpreter", "objspace", "std", "mapdict", "fn"]`) — the
/// exact segment vector the Call terminator and `FnDef`-constant
/// call-sites emit (`call_target_segments` / `decode_constant` both
/// `name_path().split("::")` without stripping the crate).
/// `register_unsafe_fn_stubs` registers a single verbatim key with no
/// alias fan-out (unlike `free_function_alias_paths`), and three-plus-
/// segment paths are excluded from the `lookup_with_leaf_match`
/// fallback, so a crate-stripped or module-collapsed key would miss the
/// nested call site.  Free functions and impl-owned functions are both
/// collected and keyed on `name_path()`: an impl method usually lowers
/// to `CallTarget::Method` (resolved through the receiver classdef), but
/// a receiver-less associated function and any impl method reached
/// through an `FnDef` constant fall back to `CallTarget::FunctionPath {
/// name_path }`, whose lookup is served only by this registry.  Argument
/// names come from the Charon body locals, falling back to `arg{N}`.
/// A return type `residual_return_shell` declines surfaces no entry,
/// preserving the original "not registered" Skip for that fn.
///
/// `pub` so `test_cel_census`'s registration classifier can ask this
/// collector which paths the stub channel holds instead of mirroring the
/// accept filters below — a mirror that drifts labels a declined fn
/// registered, which is the one direction that hides residue.
pub fn collect_unsafe_fn_stubs_from_llbc(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    error_carrier: crate::ErrorCarrierSpec<'_>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    Option<String>,
)> {
    collect_fn_stubs_from_llbc_if(
        llbc,
        gc_struct_ids,
        error_carrier,
        |fd| fd.signature.is_unsafe,
        false,
        raw_alloc_flavor_structs,
    )
}

/// Collect signature-only residual stubs for every local function whose
/// markers make it opaque to the JIT policy ([`hints_reject_body`]:
/// `#[dont_look_inside]` and `#[elidable]` alike), including functions which
/// the JIT policy correctly omitted from `CallControl::function_graphs`.
///
/// RPython creates the `FunctionDesc` from the callable independently of
/// `find_all_graphs`, then `JitPolicy.look_inside_graph` merely keeps its body
/// out of the JitCode closure.  Feeding these declarations through the same
/// annotator-only stub carrier restores that ordering: a caller can annotate
/// the declared residual call even though there is intentionally no body to
/// compile.
///
/// Both hint spellings need the same carrier for the same reason, and a
/// declaration reached only through a caller's lift needs it most: the
/// `function_graphs` residualize arm in
/// `cutover::populate_call_registry_from_call_graphs` can only stub a callee
/// the MIR loop already turned into a `SemanticFunction`, so an opaque callee
/// whose body never lowered (`build_semantic_program`'s
/// `declaration-has-no-unstructured-body` and unrecognised-shape drops) held
/// no registry key at all and every caller that named it failed its own lift
/// with "not registered in CallRegistry".  Keying off the marker rather than
/// off the lowered body is what `_reject_function` already implies: the body
/// was never going to be compiled, so whether it could be lowered must not
/// decide whether the call site can be annotated.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn collect_policy_opaque_fn_stubs_from_llbc(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    error_carrier: crate::ErrorCarrierSpec<'_>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    Option<String>,
)> {
    let marked = policy_opaque_fn_set_of(llbc);
    collect_fn_stubs_from_llbc_if(
        llbc,
        gc_struct_ids,
        error_carrier,
        |fd| marked.contains(&strip_crate_prefix(&fd.item_meta.name_path())),
        true,
        raw_alloc_flavor_structs,
    )
}

/// Signature row of a declaration the front end declares as an
/// `rffi.llexternal` ([`CrateLowering::declare_llexternal`]); its body is
/// never walked.
pub(in crate::front::mir) fn declined_atomic_load_fun_decl(
    llbc: &Llbc,
    fd: &FunDecl,
    decline_reason: String,
) -> crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl {
    use crate::translator::rtyper::lltypesystem::module::ll_extaccessor::DeclinedFunDecl;
    let segments: Vec<String> = fd
        .item_meta
        .name_path()
        .split("::")
        .map(String::from)
        .collect();
    DeclinedFunDecl {
        segments,
        arg_lltypes: fd
            .signature
            .inputs
            .iter()
            .map(|ty| tyref_to_external_lltype(ty, llbc))
            .collect(),
        result_lltype: tyref_to_external_lltype(&fd.signature.output, llbc),
        has_translatable_body: false,
        decline_reason,
    }
}

pub(in crate::front::mir) fn tyref_to_external_lltype(
    ty: &TyRef,
    llbc: &Llbc,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    if is_unit_type(ty, llbc) {
        return LowLevelType::Void;
    }
    // `tyref_to_value_type` peels `&T` and types an atomic wrapper as
    // its inner scalar, so a pointer argument would look like a word.
    // The residual-call ABI refuses pointer args; keep them Address.
    if output_type_is_ref(ty, llbc) || tyref_is_raw_pointer(ty, llbc) {
        return LowLevelType::Address;
    }
    match tyref_to_value_type(ty, llbc) {
        ValueType::Int => LowLevelType::Signed,
        ValueType::Unsigned => LowLevelType::Unsigned,
        ValueType::Int128 => LowLevelType::SignedLongLongLong,
        ValueType::UInt128 => LowLevelType::UnsignedLongLongLong,
        ValueType::Bool => LowLevelType::Bool,
        ValueType::Float => LowLevelType::Float,
        ValueType::SingleFloat => LowLevelType::SingleFloat,
        ValueType::Void => LowLevelType::Void,
        _ => LowLevelType::Address,
    }
}

pub(in crate::front::mir) fn tyref_is_raw_pointer(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .and_then(serde_json::Value::as_object)
        .is_some_and(|obj| obj.contains_key("RawPtr"))
}

/// `ty` is one `&` / `&mut` / `*const` / `*mut` whose pointee is a
/// primitive (`i64`, `usize`, `bool`, `f64`, …).
///
/// `lltype.Ptr` cannot point at a primitive. `Rvalue::Ref` of a primitive
/// aliases the scalar, so a reference's flow value is that word and
/// `history.py` `getkind` banks it as `int` or `float`. A raw pointer is
/// the address: `getkind` banks every raw `Ptr` as `int`, including
/// `*mut f64`. A byte pointer stays `None` (the caller keeps `Ref`):
/// pyre erases a GC reference to `*mut u8` (`gc_hook::try_gc_write_barrier`,
/// frame and dict storage). Returns `None` for a pointer to an ADT
/// (`*mut PyObject`), a pointer to a pointer, and every non-pointer type.
/// 128-bit primitives stay `None`: `getkind` rejects them.
pub(in crate::front::mir) fn tyref_scalar_pointee_value_type(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let obj = node.as_object()?;
    let (pointee, is_raw) = match obj.get("Ref").and_then(serde_json::Value::as_array) {
        Some(arr) => (arr.get(1)?, false),
        None => (
            obj.get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())?,
            true,
        ),
    };
    let prim = tyref_primitive_node(pointee, llbc)?;
    let scalar = prim.as_object()?.get("Scalar")?;
    let pointee_vt = scalar_value_type(scalar)?;
    let supported = matches!(
        pointee_vt,
        ValueType::Int
            | ValueType::Unsigned
            | ValueType::Bool
            | ValueType::Float
            | ValueType::SingleFloat
    );
    if !supported {
        return None;
    }
    if is_raw {
        let tag = scalar
            .as_object()
            .and_then(|obj| obj.get("Integer"))
            .and_then(serde_json::Value::as_object)
            .and_then(|int| int.get("Signed").or_else(|| int.get("Unsigned")))
            .and_then(serde_json::Value::as_str);
        if matches!(tag, Some("I8" | "U8")) {
            return None;
        }
        return Some(ValueType::Int);
    }
    Some(pointee_vt)
}

/// Nested `Ref` of a primitive (`&&i64`, `&&&f64`).
///
/// [`tyref_scalar_pointee_value_type`] peels one pointer and returns
/// `None` for a pointer to a pointer, which is correct for call-result
/// banking (`iterator_payload_element` of `&[&i64]`). A declared
/// parameter of `PartialOrd<&B> for &A` is `&&T`; the caller already
/// aliases `&T` to the scalar (`history.py` `getkind`). Used only by
/// [`input_param_value_type`].
pub(in crate::front::mir) fn tyref_nested_ref_to_scalar_value_type(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    let mut node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let mut refs = 0u8;
    loop {
        let obj = node.as_object()?;
        let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) else {
            return None;
        };
        node = strip_ty_indirections(arr.get(1)?, llbc)?;
        refs = refs.saturating_add(1);
        if refs > 24 {
            return None;
        }
        let Some(prim) = tyref_primitive_node(node, llbc) else {
            continue;
        };
        if refs < 2 {
            return None;
        }
        let scalar = prim.as_object()?.get("Scalar")?;
        let pointee_vt = scalar_value_type(scalar)?;
        return matches!(
            pointee_vt,
            ValueType::Int
                | ValueType::Unsigned
                | ValueType::Bool
                | ValueType::Float
                | ValueType::SingleFloat
        )
        .then_some(pointee_vt);
    }
}

/// `char` / `&char` / `&&char` stay `ValueType::Int` (`getkind` of
/// UniChar is `'int'`), while the annotator cell is
/// `SomeUnicodeCodePoint`. Carry `"char"` so
/// `derive_subject_inputcells` seeds that cell.
pub(in crate::front::mir) fn tyref_char_class_root(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let mut node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            node = strip_ty_indirections(arr.get(1)?, llbc)?;
            continue;
        }
        let scalar = obj.get("Scalar")?;
        let is_char = match scalar {
            serde_json::Value::String(s) => s == "Char",
            serde_json::Value::Object(map) => map.contains_key("Char"),
            _ => false,
        };
        return is_char.then(|| "char".to_string());
    }
    None
}

/// `&[T]` / `&mut [T]` whose element is a GC pointer, not a raw address.
pub(in crate::front::mir) fn borrow_pointee_is_ptr_slice(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let Some(obj) = node.as_object() else {
        return false;
    };
    if obj.get("RawPtr").is_some() {
        return false;
    }
    let Some(borrow) = obj.get("Ref").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let Some(pointee) = borrow
        .get(1)
        .and_then(|pointee| strip_ty_indirections(pointee, llbc))
    else {
        return false;
    };
    let Some(elem) = pointee
        .as_object()
        .and_then(|obj| obj.get("Slice"))
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|elem| strip_ty_indirections(elem, llbc))
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    pointer_elem_pointee_is_gc(elem, llbc, gc_struct_ids)
}

/// The slice element is a `Ref` or `RawPtr` to a declared GC struct.
/// A pointer to a scalar or a raw allocation stays an address.
pub(in crate::front::mir) fn pointer_elem_pointee_is_gc(
    elem: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let (key, pointee_idx) = if elem.contains_key("Ref") {
        ("Ref", 1)
    } else if elem.contains_key("RawPtr") {
        ("RawPtr", 0)
    } else {
        return false;
    };
    let Some(pointee) = elem
        .get(key)
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.get(pointee_idx))
        .and_then(|pointee| strip_ty_indirections(pointee, llbc))
    else {
        return false;
    };
    if let Some(def_id) = adt_node_def_id(pointee)
        && adt_path_is_gc_container(llbc, def_id)
    {
        return true;
    }
    let Some(td) = adt_node_def_id(pointee).and_then(|id| llbc.type_by_id(id)) else {
        return false;
    };
    let Some(sid) = declared_adt_struct_id(pointee, llbc) else {
        return false;
    };
    struct_id_is_declared_gc(
        gc_struct_ids,
        sid,
        &strip_crate_prefix(&td.item_meta.name_path()),
    )
}

/// `getkind` of a raw `Ptr` whose pointee is a non-byte primitive.
///
/// References return `None`. Their flow value stays the address;
/// [`tyref_scalar_pointee_value_type`] names the pointee for a parameter
/// or a `Rvalue::Ref`, not for a call result.
pub(in crate::front::mir) fn raw_scalar_address_value_type(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    if tyref_is_raw_pointer(ty, llbc) {
        tyref_scalar_pointee_value_type(ty, llbc)
    } else {
        None
    }
}

/// `ty` is one `&` / `&mut` of a primitive. A raw pointer is not a borrow.
pub(in crate::front::mir) fn tyref_is_ref_of_primitive(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_raw_pointer(ty, llbc) {
        return false;
    }
    tyref_peel_one_ref_node(ty, llbc).is_some_and(|pointee| json_ty_is_copy_scalar(pointee, llbc))
}

/// `ty` is one `*const` / `*mut` of a primitive. A reference is not a raw pointer.
pub(in crate::front::mir) fn tyref_is_raw_ptr_of_primitive(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_peel_one_raw_ptr_node(ty, llbc)
        .is_some_and(|pointee| json_ty_is_copy_scalar(pointee, llbc))
}

/// Kind `'r'`: `lltype.GCREF` / `Ptr(GcStruct)` (`history.py` `getkind`).
/// `GcRef` is detected by the type decl's own name, without peeling
/// `repr(transparent)` to `usize`. A raw pointer to a primitive or an
/// `&T` can hold a `rffi.scoped_alloc` / `lltype.malloc(flavor='raw')`
/// address and is not this kind.
pub(in crate::front::mir) fn tyref_is_gc_reference(ty: &TyRef, llbc: &Llbc) -> bool {
    if output_type_is_ref(ty, llbc) {
        return false;
    }
    if tyref_is_raw_ptr_of_primitive(ty, llbc) {
        return false;
    }
    if tyref_decl_names_gcref(ty, llbc) {
        return true;
    }
    if !tyref_is_raw_pointer(ty, llbc) {
        return false;
    }
    if raw_scalar_address_value_type(ty, llbc).is_some() {
        return false;
    }
    matches!(
        tyref_to_value_type(ty, llbc),
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder
    )
}

pub(in crate::front::mir) fn tyref_decl_names_gcref(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let Some(def_id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    let path = td.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
    crate::annotator::bookkeeper::is_gcref_type_spelling(leaf)
        || crate::annotator::bookkeeper::is_gcref_type_spelling(&path)
        || leaf.eq_ignore_ascii_case("GcRef")
}

pub(in crate::front::mir) fn ref_kind_is_mut(ty: &TyRef, llbc: &Llbc) -> bool {
    pointer_kind_is_mut(ty, llbc, "Ref", 2)
}

pub(in crate::front::mir) fn raw_ptr_kind_is_mut(ty: &TyRef, llbc: &Llbc) -> bool {
    pointer_kind_is_mut(ty, llbc, "RawPtr", 1)
}

pub(in crate::front::mir) fn pointer_kind_is_mut(
    ty: &TyRef,
    llbc: &Llbc,
    key: &str,
    kind_index: usize,
) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .and_then(|node| node.as_object()?.get(key)?.as_array()?.get(kind_index))
        .is_some_and(borrow_kind_is_exclusive)
}

pub(in crate::front::mir) fn collect_fn_stubs_from_llbc_if(
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    error_carrier: crate::ErrorCarrierSpec<'_>,
    include: impl Fn(&FunDecl) -> bool,
    admit_option_raw_adt_ptr: bool,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    Option<String>,
)> {
    use crate::flowspace::argument::Signature;
    let mut out = Vec::new();
    for fd in llbc.iter_local_fns() {
        if !include(fd) {
            continue;
        }
        // Global initializers are synthetic, not user-callable fns; the
        // MIR-driver lowering loop skips them too.
        if fd.is_global_initializer().is_some() {
            continue;
        }
        // `&bool` / `&()` stay out. A reference to a declared-Raw ADT is
        // the same address as `*mut` (`lltype.Ptr` of a raw struct), so
        // it takes the `raw:<owner>` token instead of being skipped.
        if output_type_is_ref(&fd.signature.output, llbc)
            && raw_address_owner_root(
                &fd.signature.output,
                llbc,
                no_tombstoned_leaves(),
                gc_struct_ids,
                raw_alloc_flavor_structs,
            )
            .is_none()
        {
            continue;
        }
        // `None` here means unit, which `residual_return_shell` maps to
        // `Void`; a return type it cannot model still reaches this arm as
        // a token it declines, and the fn then keeps its original "not
        // registered" Skip.
        let token = dont_look_inside_return_token(
            &fd.signature.output,
            llbc,
            error_carrier,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        if crate::translator::rtyper::cutover::residual_return_shell(token.as_deref()).is_none() {
            continue;
        }
        // The `ref` token collapses every non-`*mut PyObject` ADT onto one
        // GC-reference word, so on its own it would also admit a by-value
        // aggregate — `w_tuple_items_copy_as_vec` returns
        // `Vec<PyObjectRef>`, three words returned through `sret`. The
        // residual-call ABI has only a one-word return and no `sret` path,
        // so a multiword return cannot be published at all. Admit `ref` only
        // for returns that genuinely are one word.
        //
        // Policy-opaque `Option<*mut NominalADT>` is one word at the
        // residual boundary: `#[dont_look_inside]`'s `extern "C"` trampoline
        // (`helper_return_to_i64` `nullable_pointer_option`) converts
        // `Some(p)` to `p` and `None` to null, matching `Optional[W_Root]`.
        // The original Rust return is two words; unsafe-fn stubs call that
        // original and keep the decline (`admit_option_raw_adt_ptr = false`).
        let single_word_ref =
            ref_return_is_single_word(&tyref_to_ast_string(&fd.signature.output, llbc))
                || option_shared_ref_return_is_one_word(&fd.signature.output, llbc, error_carrier)
                || (admit_option_raw_adt_ptr
                    && option_raw_adt_ptr_return_is_one_word(
                        &fd.signature.output,
                        llbc,
                        error_carrier,
                    ));
        if matches!(
            token.as_deref(),
            Some("ref" | crate::translator::rtyper::cutover::STR_RETURN_TYPE)
        ) && !single_word_ref
        {
            continue;
        }
        // Both free functions and impl-owned functions are collected,
        // keyed on `name_path()` — the segment vector
        // `call_target_segments` emits for a `CallKind::Fun(Regular)`
        // in this file.  An impl method usually lowers to
        // `CallTarget::Method` (resolved via the receiver classdef), but a
        // receiver-less associated function (`Owner::new() -> ()/bool`) and
        // any impl method reached through an `FnDef` constant fall back to
        // `CallTarget::FunctionPath { name_path }`,
        // whose lookup is served only by this registry — skipping impl
        // owners would leave those call sites "not registered".
        let segments: Vec<String> = fd
            .item_meta
            .name_path()
            .split("::")
            .map(String::from)
            .collect();
        // Prefer the Charon body's declared parameter names
        // (`locals[1..=argc]`, the same source the regular lowering reads
        // at `local.name`); fall back to positional `arg{N}` when the body
        // is opaque or a local is unnamed.
        let body = fd.unstructured();
        let argnames: Vec<String> = (0..fd.signature.inputs.len())
            .map(|i| {
                body.as_ref()
                    .and_then(|u| u.locals.locals.get(i + 1))
                    .and_then(|l| l.name.clone())
                    .unwrap_or_else(|| format!("arg{i}"))
            })
            .collect();
        out.push((segments, Signature::new(argnames, None, None), token));
    }
    out
}

/// Collect, from the lowered MIR, `(path-segments, Signature, result
/// ValueType)` for every method whose impl-owner is a **foreign opaque**
/// ADT and whose result type can be modeled faithfully.
///
/// An opaque owner has no extracted body, so the annotator never mints a
/// `ClassDef` for it; the receiver lands as a classdef-less
/// `SomeInstance` and a `CallTarget::Method` getattr panics.  These
/// methods are the foreign helpers the interpreter's cold
/// `@jit.dont_look_inside` overflow→long arms operate on
/// (`<BigInt as Add>::add`, `…::clone`, …), so the faithful treatment is
/// the `register_external` analog: residualize the call rather than trace
/// into the foreign body.  [`Lowering::impl_method_owner`] already
/// declines the `CallTarget::Method` hint for an opaque owner (routing the
/// call through `CallTarget::FunctionPath`); this collection feeds the
/// matching registry entries so the `FunctionPath` lookup resolves instead
/// of raising "not registered in CallRegistry".
///
/// The registration key is derived through the SAME
/// [`impl_method_owner_for_fundecl`] the declined-Method
/// `call_target_segments` arm uses, so the key equals the call-site
/// lookup (`[strip_crate(owner.name_path).split("::"), leaf]` —
/// e.g. `["bigint", "BigInt", "add"]`).
///
/// **Faithful result shell — no blanket Ref.**  The result `ValueType` is
/// read from the method's LLBC output signature:
///
/// - output is a scalar literal (`i64` / `f64` / `bool`) → that
///   `ValueType` (the residual really produces an integer / float / bool);
/// - output is itself a foreign **opaque** ADT (`BigInt` → `BigInt`,
///   the `Add`/`Sub`/`Mul`/`clone` cluster) → `Ref(None)`, the
///   classdef-less `SomeInstance` shell `bigint_from` produces;
/// - output is a payload-carrying enum (`GcAllocOutcome`) →
///   `Ref(Some(root))`, the intern key a caller `match` on
///   `Allocated`/`Failed`/`NoRoute` getattr's `__discriminant` through;
/// - anything else — an `Option<i64>` (`to_i64`), a fieldless enum
///   (`sign`), a tuple, a reference, a non-opaque struct — is
///   **declined** (no entry), leaving the method at the original "not
///   registered" Skip.  Modeling an `Option<i64>` return as a bare
///   integer or as `Ref(None)` would mis-type the value and only migrate
///   the failure to a deeper wall, so those methods stay residual until
///   their result type can be modeled.
///
/// A second arm harvests **bodyless foreign associated functions**
/// (`!is_local`, no unstructured body, no `self` receiver).  `majit_gc`
/// is not an extracted crate, so Charon records
/// `GcAllocOutcome::from_hook` as Foreign/Opaque while the TypeDecl is
/// the full payload Enum — not `TypeDeclKind::Opaque`, and the first
/// argument is the hook `Option<*mut u8>`, not `self`.  The
/// registration key is still [`impl_method_owner_for_fundecl`] +
/// [`strip_crate_prefix`], matching the call-site
/// `["GcAllocOutcome", "from_hook"]`.
///
/// `Range::contains` on the extracted `Range` Struct is not harvested:
/// its Charon-included body is the graph the FunctionPath resolves to,
/// and a bodyless one leaves the unregistered FunctionPath (census
/// Skip).  A
/// result type that cannot be modeled (`allocated_or_abort`'s RawPtr) is
/// still declined by [`foreign_opaque_method_result_valuetype`].  A
/// zero-arg method on an opaque owner (`Default::default`) has no
/// `self`; it is harvested with the opaque-owner methods because
/// `opaque_nullary_impl_target` emits that path.
///
/// A bodyless trait-impl method on a Struct owner is harvested too
/// (`Iterator::next` after CHARON includes `RangeInclusive`). Including
/// the TypeDecl makes the owner a Struct, so the Opaque-owner arm no
/// longer fires, while `impl_method_owner` still declines the Method
/// hint (`!has_unstructured_body && Struct`) and the call stays a
/// FunctionPath (`Bookkeeper.getdesc`). Inherent bodyless methods on a
/// Struct (`Range::contains`) stay unharvested.
///
/// A GlobalInitializer FunDecl is not a call target (`admit_decl`).
/// `Place::Global` folds a compile-time constant from the Charon
/// initializer (`flowcontext.py` `find_global` `const(value)`); otherwise
/// the read stays the unregistered residual (`extfunc.py`
/// `register_external` requires a real function). An Opaque import of a
/// local-crate free function is the same residual when this artefact has
/// no body; `register_external` is non-overwriting, so a graph the
/// defining crate lowered still wins.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn collect_foreign_opaque_method_externals(
    llbc: &Llbc,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    ValueType,
)> {
    use crate::flowspace::argument::Signature;
    let mut out = Vec::new();
    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer().is_some() {
            continue;
        }
        if let Some((owner_qualified, leaf)) = impl_method_owner_for_fundecl(llbc, fd) {
            let Some(owner_def_id) = impl_owner_adt_def_id_for_fundecl(llbc, fd) else {
                continue;
            };
            let Some(owner_td) = llbc.type_by_id(owner_def_id) else {
                continue;
            };
            let self_receiver = first_input_is_adt_free(llbc, fd, owner_def_id);
            let opaque_owner =
                matches!(owner_td.kind, TypeDeclKind::Opaque) || owner_td.published_struct_copy;
            let opaque_method = opaque_owner && (self_receiver || fd.signature.inputs.is_empty());
            let foreign_assoc_fn =
                !fd.item_meta.is_local && !fd.has_unstructured_body() && !self_receiver;
            // Bodyless trait-impl method whose owner TypeDecl was
            // included as a Struct (`RangeInclusive::next`). Inherent
            // `Range::contains` has no trait-impl id and stays out.
            let bodyless_trait_impl_method = !fd.item_meta.is_local
                && !fd.has_unstructured_body()
                && self_receiver
                && trait_impl_id_for_fundecl(fd).is_some();
            if !opaque_method && !foreign_assoc_fn && !bodyless_trait_impl_method {
                continue;
            }
            let Some(result_ty) =
                foreign_opaque_method_result_valuetype(&fd.signature.output, llbc)
            else {
                continue;
            };
            let signature = Signature::new(fundecl_stub_argnames(fd), None, None);
            let mut segments: Vec<String> =
                owner_qualified.split("::").map(str::to_string).collect();
            segments.push(leaf.clone());
            // A callsite names the method `owner::leaf`
            // (`CallControl.guess_call_kind`); `opaque_nullary_impl_target`
            // names it by its identity-qualified path. Both spellings
            // register the one callable (`extfunc.py` `register_external`),
            // as `free_function_alias_paths` does for a free function.
            let identity = spec_segments(llbc, fd, &leaf);
            if identity != segments {
                out.push((identity, signature.clone(), result_ty.clone()));
            }
            out.push((segments, signature, result_ty));
            continue;
        }
        if let Some(row) = local_crate_opaque_import_external(llbc, fd) {
            out.push(row);
        }
    }
    out
}

#[cfg(any(test, feature = "mir-frontend"))]
pub(in crate::front::mir) fn fundecl_stub_argnames(fd: &FunDecl) -> Vec<String> {
    let body = fd.unstructured();
    (0..fd.signature.inputs.len())
        .map(|i| {
            body.as_ref()
                .and_then(|u| u.locals.locals.get(i + 1))
                .and_then(|l| l.name.clone())
                .unwrap_or_else(|| format!("arg{i}"))
        })
        .collect()
}

/// Opaque import of a free function defined in another extracted crate.
/// The callsite keeps the crate-qualified spelling
/// (`CallControl.guess_call_kind`); a graph the defining crate lowered
/// still wins at register time (`extfunc.py` `register_external`).
#[cfg(any(test, feature = "mir-frontend"))]
pub(in crate::front::mir) fn local_crate_opaque_import_external(
    llbc: &Llbc,
    fd: &FunDecl,
) -> Option<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    ValueType,
)> {
    use crate::flowspace::argument::Signature;
    if fd.has_unstructured_body() || fd.item_meta.is_local {
        return None;
    }
    let Some(NameSeg::Ident { ident: (root, _) }) = fd.item_meta.name.first() else {
        return None;
    };
    if !crate::local_crates::is_local_crate_root(root) {
        return None;
    }
    let result_ty = foreign_opaque_method_result_valuetype(&fd.signature.output, llbc)?;
    let segments: Vec<String> = fd
        .item_meta
        .name_path()
        .split("::")
        .map(str::to_string)
        .collect();
    Some((
        segments,
        Signature::new(fundecl_stub_argnames(fd), None, None),
        result_ty,
    ))
}

/// Resolve the impl-owner ADT `def_id` of an impl-block method `fd`
/// directly from its `<Impl>` NameSeg, the free-function twin of
/// [`Lowering::resolve_impl_owner_adt_def_id`] over the `<Impl>` segment.
pub(in crate::front::mir) fn impl_owner_adt_def_id_for_fundecl(
    llbc: &Llbc,
    fd: &FunDecl,
) -> Option<u64> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    resolve_impl_owner_adt_def_id_free(llbc, impl_payload)
}

/// Free-function twin of [`Lowering::first_input_is_adt`]: true when the
/// method's first input (`self`, possibly behind `&`/`&mut`/`*`) is the
/// owner ADT.
#[cfg(any(test, feature = "mir-frontend"))]
pub(in crate::front::mir) fn first_input_is_adt_free(
    llbc: &Llbc,
    fd: &FunDecl,
    adt_def_id: u64,
) -> bool {
    fd.signature
        .inputs
        .first()
        .and_then(|t| tyref_node(t, llbc))
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .is_some_and(|id| same_nominal_adt(llbc, id, adt_def_id))
}

/// Faithful result `ValueType` for a residualized foreign-opaque method,
/// or `None` to decline (see [`collect_foreign_opaque_method_externals`]).
/// A scalar literal output keeps its `ValueType`; `()` is `Void`
/// (`history.py` `getkind`, `Vec::set_len` / `Vec::resize`); an
/// opaque-ADT output projects to `Ref(None)`; a payload-carrying enum
/// keeps `Ref(Some(root))` when `tyref_to_value_type` painted a class
/// root (`unaryop.py` `SomeInstance.getattr` needs `self.classdef`).
/// A payload enum with no class root is declined. A fieldless enum,
/// tuple, reference, or non-opaque struct is declined.
pub(in crate::front::mir) fn foreign_opaque_method_result_valuetype(
    output: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    // A reference return (`&T`) is not the owned residual result the
    // stub models; decline.
    if output_type_is_ref(output, llbc) {
        return None;
    }
    match tyref_to_value_type(output, llbc) {
        // Scalar literal results are produced directly by the residual.
        vt @ (ValueType::Int | ValueType::Unsigned | ValueType::Float | ValueType::Bool) => {
            Some(vt)
        }
        // `()` is `lltype.Void`. FOREIGN_STDLIB already residualizes
        // `Vec::set_len` this way; an Opaque-owner `Vec::resize` is the
        // same `extfunc.py` `register_external(..., None)` shell.
        ValueType::Void => Some(ValueType::Void),
        // A `Ref` projection covers every non-scalar ADT shape (`BigInt`,
        // `Option<i64>`, tuples, payload enums, …).  Accept it when:
        //
        // - the result ADT is itself a foreign opaque type (the
        //   `BigInt`-returning arithmetic cluster) → classdef-less
        //   `SomeInstance`, the shell `bigint_from` produces;
        // - the result is a payload-carrying enum (`GcAllocOutcome`)
        //   whose class root `tyref_to_value_type` already painted →
        //   keep `Ref(Some(root))` so registration intern's the
        //   enum-base ClassDef a caller `match` getattr's.
        //
        // A non-opaque struct / `Option` / fieldless enum would be
        // mis-typed as an opaque GcRef, so decline it.
        ValueType::Ref(root) => {
            let def_id = output_adt_def_id_free(output, llbc)?;
            let td = llbc.type_by_id(def_id)?;
            match &td.kind {
                TypeDeclKind::Opaque => Some(ValueType::Ref(None)),
                TypeDeclKind::Enum(_) if !type_decl_is_fieldless_enum(td, llbc) => {
                    root.map(|r| ValueType::Ref(Some(r)))
                }
                _ if td.published_struct_copy => Some(ValueType::Ref(None)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Free-function twin of [`Lowering::tyref_adt_def_id`]: resolve a
/// `TyRef`'s ADT `def_id`, following the dedup index for a `Dedup` shape.
pub(in crate::front::mir) fn output_adt_def_id_free(ty: &TyRef, llbc: &Llbc) -> Option<u64> {
    match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
        TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
    }
}

/// Free-function version of [`Lowering::resolve_impl_owner_adt_def_id`].
pub(in crate::front::mir) fn impl_payload_self_ty<'a>(
    llbc: &'a Llbc,
    impl_payload: &'a serde_json::Value,
) -> Option<&'a serde_json::Value> {
    let obj = impl_payload.as_object()?;
    if let Some(ty) = obj.get("Ty") {
        return ty.as_object()?.get("skip_binder");
    }
    if let Some(trait_impl_id) = obj.get("Trait").and_then(serde_json::Value::as_u64) {
        let ti = llbc.trait_impls_raw().get(trait_impl_id as usize)?;
        return ti
            .as_object()?
            .get("impl_trait")?
            .as_object()?
            .get("generics")?
            .as_object()?
            .get("types")?
            .as_array()?
            .first();
    }
    None
}

pub(in crate::front::mir) fn resolve_impl_owner_adt_def_id_free(
    llbc: &Llbc,
    impl_payload: &serde_json::Value,
) -> Option<u64> {
    resolve_tyexpr_to_adt_def_id_free(llbc, impl_payload_self_ty(llbc, impl_payload)?)
}

/// When `fd` is a trait-impl method (i.e. its NameSeg's penultimate
/// segment is an `Impl` with a `{"Trait": <trait_impl_id>}` payload),
/// return the implemented trait's leaf identifier.  Returns `None`
/// for free functions, inherent impl methods, and trait default
/// bodies (those carry the trait name directly in `name_path()`'s
/// penultimate segment, so the caller can read it through
/// [`trait_method_owner`] without a `trait_impls` indirection).
///
/// Used by `build_semantic_program_from_llbc` to populate
/// `SemanticFunction.trait_root` (leaf) and `trait_qualified` (this
/// fn's return value, the full `name_path()`) so the canonical
/// registration loop can call `CallControl::register_trait_method`
/// instead of routing through `extract_trait_impls`, and so the
/// unique-impl map can key on trait identity rather than a bare leaf
/// (two distinct traits may share a final segment).
pub(in crate::front::mir) fn trait_impl_trait_path_for_fundecl(
    llbc: &Llbc,
    fd: &FunDecl,
) -> Option<String> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let trait_impl_id = impl_payload
        .as_object()?
        .get("Trait")
        .and_then(serde_json::Value::as_u64)?;
    let trait_impls = llbc.trait_impls_raw();
    let ti = trait_impls.get(trait_impl_id as usize)?;
    // `impl_trait` is a TraitDeclRef `{"id": <trait_decl_id>,
    // "generics": {...}}` — same shape `resolve_impl_owner_adt_def_id`
    // reads `generics.types[0]` from.
    let trait_id = ti
        .as_object()?
        .get("impl_trait")?
        .as_object()?
        .get("id")?
        .as_u64()?;
    let td = llbc.trait_by_id(trait_id)?;
    // Only source-local traits participate: std/core trait impls
    // (`FnOnce` closure shims, `Destruct`/`Drop` glue, `PartialEq`, …)
    // are host machinery, not translated-program classes, and keep
    // their inherent classification.
    if !td.item_meta.is_local {
        return None;
    }
    let trait_path = td.item_meta.name_path();
    if trait_path.is_empty() {
        return None;
    }
    Some(trait_path)
}

/// Return the concrete Charon trait-impl id carried by an impl-owned
/// function's penultimate name segment. Inherent impls and free functions do
/// not carry this identity.
pub(in crate::front::mir) fn trait_impl_id_for_fundecl(fd: &FunDecl) -> Option<u64> {
    let last_idx = fd
        .item_meta
        .name
        .iter()
        .rposition(|segment| matches!(segment, NameSeg::Ident { .. }))?;
    let impl_payload = match fd.item_meta.name.get(last_idx.checked_sub(1)?)? {
        NameSeg::Other(value) => value.as_object()?.get("Impl")?,
        _ => return None,
    };
    impl_payload.as_object()?.get("Trait")?.as_u64()
}

/// A local trait-impl method whose first parameter is a ZST `&self`.
///
/// rustc still passes that `&self` in a pointer register (`ResidualSlot`
/// for `&T`). RPython method `indirect_call` always has self. Keep the
/// word as `Ref` on graphs reached as `indirectcalltargets` so the
/// recorded/executed call matches rustc's `(&self, ...)` arity and the
/// family's FUNC type. rustc mixes ZST unit-struct impls
/// (`KwargsDictStrategy`) with GC `ModuleDictStrategy`; upstream types
/// every `DictStrategy` `self` as a GC instance
/// (`class DictStrategy(object)`), so `history.getkind` of that Ptr is
/// `'ref'`. One `indirect_call` has one FUNC type
/// (`CallControl.getcalldescr`). A GC strategy stays a GC root
/// (`walk_bh_regs` / `drag_out_root`). By-value ZST (`From::from`) stays
/// `Void`. Core/std impls stay peeled
/// (`trait_impl_trait_path_for_fundecl` is `None` for non-local traits).
pub(in crate::front::mir) fn fundecl_keeps_zst_self(
    llbc: &Llbc,
    fd: &FunDecl,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    if trait_impl_trait_path_for_fundecl(llbc, fd).is_none() {
        return false;
    }
    let Some(locals) = fd.unstructured_locals() else {
        return false;
    };
    let Some(local) = locals.locals.get(1) else {
        return false;
    };
    let ty = tyref_to_value_type_with(
        &local.ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    );
    matches!(ty, ValueType::Ref(None)) && tyref_is_void_zst(&local.ty, llbc)
}

/// Declared-parameter ValueType, matching [`pygraph_initial_block`] /
/// [`Lowering::new`]. A ZST `&self` on an `indirectcalltarget` trait impl
/// stays `Ref`; a named empty-ADT ZST stays an instance of its empty class;
/// every other void-ZST parameter is `Void`. A nested `Ref` of a primitive
/// (`&&i64`) is the scalar, matching one-level `&i64` (`history.py` `getkind`).
pub(in crate::front::mir) fn input_param_value_type(
    ty: ValueType,
    local_ty: &TyRef,
    llbc: &Llbc,
    keep_zst_self: bool,
    is_first_param: bool,
    tombstoned_leaves: &std::collections::HashSet<String>,
) -> ValueType {
    if matches!(ty, ValueType::Ref(None))
        && tyref_is_void_zst(local_ty, llbc)
        && !tyref_keeps_empty_class_instance(local_ty, llbc, tombstoned_leaves)
    {
        if keep_zst_self && is_first_param {
            ty
        } else {
            ValueType::Void
        }
    } else if matches!(ty, ValueType::Ref(None))
        && let Some(scalar) = tyref_scalar_pointee_value_type(local_ty, llbc)
            .or_else(|| tyref_nested_ref_to_scalar_value_type(local_ty, llbc))
    {
        scalar
    } else {
        ty
    }
}

/// The `From::from` body selected for `error_ty` into `carrier_path`.
///
/// `impl From<T> for U` stores `[U, T]` on `impl_trait.generics.types`,
/// the same order `blanket_into_devirt` reads off a `U: From<T>`
/// obligation. The call path is the one `trait_impl_identity_path`
/// registers, so `from` stays a `FunctionPath` (it is an associated
/// function; a `CallTarget::Method` would bind the argument as the
/// receiver). Zero matches and several matches both decline: raising
/// the foreign payload would skip `From::from`.
pub(in crate::front::mir) fn from_impl_conversion(
    llbc: &Llbc,
    carrier_path: &str,
    error_ty: &str,
) -> Result<crate::front::result_exc::FromResidualConversion, String> {
    use crate::front::result_exc::same_type_spelling;
    let from_id = llbc
        .iter_trait_decls()
        .find(|decl| decl.item_meta.name_path() == "core::convert::From")
        .map(|decl| decl.def_id);
    let Some(from_id) = from_id else {
        return Err(format!(
            "core::convert::From is absent, so {error_ty} cannot convert into {carrier_path}"
        ));
    };
    let mut found = Vec::new();
    for row in llbc.trait_impls_raw() {
        let Some(impl_trait) = row.get("impl_trait") else {
            continue;
        };
        let Some(trait_id) = impl_trait.get("id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        if trait_id != from_id {
            continue;
        }
        let Some(types) = impl_trait
            .get("generics")
            .and_then(|generics| generics.get("types"))
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        let (Some(self_ty), Some(arg_ty)) = (types.first(), types.get(1)) else {
            continue;
        };
        let self_rendered = charon_type_value_to_ast_string(self_ty, llbc, 0);
        let arg_rendered = charon_type_value_to_ast_string(arg_ty, llbc, 0);
        if !same_type_spelling(&self_rendered, carrier_path)
            || !same_type_spelling(&arg_rendered, error_ty)
        {
            continue;
        }
        let Some(methods) = row.get("methods").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for method in methods {
            let Some(fn_id) = method
                .get("skip_binder")
                .and_then(|binder| binder.get("id"))
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let Some(fd) = llbc.fn_by_id(fn_id) else {
                continue;
            };
            let leaf = fd.item_meta.name_path();
            if leaf.rsplit("::").next() != Some("from") {
                continue;
            }
            let Some((owner, method_name)) = impl_method_owner_for_fundecl(llbc, fd) else {
                continue;
            };
            let Some(impl_id) = trait_impl_id_for_fundecl(fd) else {
                continue;
            };
            let Some(identity) = crate::front::clause_spec::render_trait_impl(llbc, impl_id) else {
                continue;
            };
            let path = crate::parse::CallPath::from_segments(trait_impl_identity_path(
                &owner,
                &identity,
                &method_name,
            ));
            let pass_payload = fd.signature.inputs.first().is_some_and(|ty| {
                !tyref_is_void_zst(ty, llbc)
                    || tyref_keeps_empty_class_instance(ty, llbc, no_tombstoned_leaves())
            });
            found.push(crate::front::result_exc::FromResidualConversion {
                segments: path.segments,
                pass_payload,
            });
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(format!(
            "no From impl converts {error_ty} into {carrier_path}"
        )),
        _ => Err(format!(
            "several From impls convert {error_ty} into {carrier_path}"
        )),
    }
}

/// Detect a trait-default body — a function whose penultimate NameSeg
/// is a bare `Ident` matching a known trait leaf (no `Impl` segment).
/// Charon emits trait default impls inline in the trait's namespace,
/// so they look like `pyre_interpreter::pyopcode::LocalOpcodeHandler::
/// load_local_checked_value` with the trait leaf as the parent ident.
///
/// Returns the trait leaf so `build_semantic_program_from_llbc` can
/// populate `SemanticFunction.trait_root` and the canonical
/// registration loop (`lib.rs`) can find the body without
/// going through `extract_trait_impls`'s `<default methods of <T>>`
/// pseudo-impl-type detour.
pub(in crate::front::mir) fn trait_default_owner_for_fundecl(
    fd: &FunDecl,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    let (parent, _leaf) = trait_method_owner(fd)?;
    if known_trait_names.contains(&parent) {
        Some(parent)
    } else {
        None
    }
}

/// Free-function version of [`Lowering::resolve_tyexpr_to_adt_def_id`].
pub(in crate::front::mir) fn resolve_tyexpr_to_adt_def_id_free(
    llbc: &Llbc,
    ty: &serde_json::Value,
) -> Option<u64> {
    let obj = ty.as_object()?;
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && let Some(body) = arr.get(1)
    {
        return adt_node_def_id(body);
    }
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_to_adt_def_id(id);
    }
    None
}

/// Empty set for paint calls outside a program lowering. Duplicate-leaf
/// tombstones belong to the lowering that computed `struct_origins`.
pub(in crate::front::mir) fn no_tombstoned_leaves() -> &'static std::collections::HashSet<String> {
    static EMPTY: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    EMPTY.get_or_init(std::collections::HashSet::new)
}

/// Empty `_immutable_fields_` table for paint calls outside a crate
/// lowering. Production dest overlays and `adt_def_is_raw_storage` read
/// the table [`CrateLoweringState`] harvested once.
pub(in crate::front::mir) fn no_immutable_fields() -> &'static ImmutableFieldsTable {
    static EMPTY: std::sync::OnceLock<ImmutableFieldsTable> = std::sync::OnceLock::new();
    EMPTY.get_or_init(ImmutableFieldsTable::new)
}

/// `_immutable_fields_` rows for `td`, keyed the same way the GC-struct
/// schema path looks them up (`harvest_immutable_fields_from_llbcs`:
/// crate-stripped path, then leaf, then the full path).
pub(in crate::front::mir) fn type_decl_immutable_fields<'a>(
    td: &TypeDecl,
    table: &'a ImmutableFieldsTable,
) -> Option<&'a Vec<(String, crate::model::ImmutableRank)>> {
    let path = td.item_meta.name_path();
    let stripped = strip_crate_prefix(&path);
    let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
    table
        .get(&stripped)
        .or_else(|| table.get(leaf))
        .or_else(|| table.get(&path))
}

/// `STRUCT._immutable_field(name)` (`jtransform.py` `rewrite_op_getfield`):
/// any rank, including quasi, makes a raw GC-field read `_pure`.
pub(in crate::front::mir) fn field_declared_immutable(
    td: &TypeDecl,
    field_name: &str,
    table: &ImmutableFieldsTable,
) -> bool {
    type_decl_immutable_fields(td, table)
        .is_some_and(|fields| fields.iter().any(|(name, _)| name == field_name))
}

/// Declared name of an Opaque scalar-word field, from the defining
/// crate's harvest. `__pos_N` only when that owner has no row.
pub(in crate::front::mir) fn opaque_struct_field_name(
    owner: &str,
    field_idx: usize,
    registry: &crate::front::semantic::StructFieldRegistry,
) -> String {
    registry
        .field_name_at(owner, field_idx)
        .map(str::to_string)
        .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(field_idx))
}

pub(in crate::front::mir) fn scalar_value_type(scalar: &serde_json::Value) -> Option<ValueType> {
    if let Some(atom) = scalar.as_str() {
        return match atom {
            "Bool" => Some(ValueType::Bool),
            "Char" => Some(ValueType::Int),
            _ => Some(ValueType::Ref(None)),
        };
    }
    let obj = scalar.as_object()?;
    if let Some(int) = obj.get("Integer").and_then(serde_json::Value::as_object) {
        if int.get("Signed").and_then(serde_json::Value::as_str) == Some("I128") {
            return Some(ValueType::Int128);
        }
        if int.get("Unsigned").and_then(serde_json::Value::as_str) == Some("U128") {
            return Some(ValueType::UInt128);
        }
        if int.contains_key("Unsigned") {
            return Some(ValueType::Unsigned);
        }
        if int.contains_key("Signed") {
            return Some(ValueType::Int);
        }
    }
    if let Some(f) = obj.get("Float").and_then(serde_json::Value::as_str) {
        return Some(if f == "F32" {
            ValueType::SingleFloat
        } else {
            ValueType::Float
        });
    }
    None
}

/// Pointee of exactly one pointer word: `Ref`, `RawPtr`, a thin `Box`, or
/// `core::ptr::non_null::NonNull`. A further pointer stays inside the pointee.
pub(in crate::front::mir) fn one_level_pointer_pointee<'a>(
    node: &'a serde_json::Value,
    llbc: &'a Llbc,
) -> Option<&'a serde_json::Value> {
    // `type T = *mut U` is its own `TypeDecl` (`kind: Alias`). Charon
    // names the alias at use sites (`PyObjectRef`), so a transparent
    // handle field is an `Adt` of the alias until this peel. Then the
    // pointee's `gc_struct_ids` entry decides `history.getkind`.
    let node = type_node_peel_aliases(node, llbc)?;
    let obj = node.as_object()?;
    if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
        return arr.get(1);
    }
    if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
        return arr.first();
    }
    if type_node_is_thin_box(node, llbc) {
        return type_node_box_pointee(node, llbc);
    }
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    if td.item_meta.name_path() == "core::ptr::non_null::NonNull" {
        let adt = obj.get("Adt")?.as_object()?;
        return type_decl_ref_generics(adt, llbc)?
            .get("types")?
            .as_array()?
            .first();
    }
    None
}

/// Whether `node` names a harvested `_alloc_flavor_ = raw` ADT
/// (`rclass.py getgcflavor`).
pub(in crate::front::mir) fn flavor_raw_adt_node(
    node: &serde_json::Value,
    llbc: &Llbc,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    if raw_alloc_flavor_structs.is_empty() {
        return false;
    }
    let node = strip_ty_indirections(node, llbc).unwrap_or(node);
    let Some(def_id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    crate::front::llbc_hints::alloc_flavor_raw_contains(
        raw_alloc_flavor_structs,
        &td.item_meta.name_path(),
    )
}

/// Pointee of one address word for class identity / layout / field access:
/// `Ref` / `RawPtr` / thin `Box` / `NonNull` ([`one_level_pointer_pointee`]),
/// or `AtomicPtr<T>` (layout-identical `*mut T`). A thin `Arc` / `Rc` word
/// points at the refcount header (`ArcInner` / `RcBox`), not at `T`
/// (`deref_cast_root`); [`one_word_pointer_bank_pointee`] peels those for
/// the int/pointer bank only. A fat pointer (`&dyn`, `&[T]`, `&str`,
/// `Box<dyn>`) is not one word.
pub(in crate::front::mir) fn one_word_pointer_pointee<'l>(
    node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    if let Some(pointee) = one_level_pointer_pointee(node, llbc) {
        let peeled = type_node_peel_aliases(node, llbc).unwrap_or(node);
        let is_borrow = peeled
            .as_object()
            .is_some_and(|obj| obj.contains_key("Ref"));
        if is_borrow && !json_ty_is_statically_sized(pointee, llbc) {
            return None;
        }
        return Some(pointee);
    }
    let node = type_node_peel_aliases(node, llbc)?;
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    let leaf = match td.item_meta.template_name().next_back()? {
        NameSeg::Ident { ident: (s, _) } => s.as_str(),
        _ => return None,
    };
    match leaf {
        "Atomic" | "AtomicPtr" => {
            if !td
                .item_meta
                .template_name()
                .any(|s| matches!(s, NameSeg::Ident { ident: (id, _) } if id == "atomic"))
            {
                return None;
            }
            let arg = type_decl_ref_generics(node.as_object()?.get("Adt")?.as_object()?, llbc)?
                .get("types")?
                .as_array()?
                .first()?;
            let arg = strip_ty_indirections(arg, llbc).unwrap_or(arg);
            if leaf == "AtomicPtr" {
                return json_ty_is_statically_sized(arg, llbc).then_some(arg);
            }
            arg.as_object()?.get("RawPtr")?.as_array()?.first()
        }
        _ => None,
    }
}

/// Pointee of one address word for the int/pointer bank (`history.getkind`
/// / `get_type_flag`). Same as [`one_word_pointer_pointee`], plus a thin
/// `Arc` / `Rc` whose payload names the bank's flavor owner. The handle
/// word is still one raw (non-GC) word.
pub(in crate::front::mir) fn one_word_pointer_bank_pointee<'l>(
    node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    if let Some(pointee) = one_word_pointer_pointee(node, llbc) {
        return Some(pointee);
    }
    let node = type_node_peel_aliases(node, llbc)?;
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    let leaf = match td.item_meta.template_name().next_back()? {
        NameSeg::Ident { ident: (s, _) } => s.as_str(),
        _ => return None,
    };
    match leaf {
        "Arc" | "Rc" => {
            let path = td.item_meta.name_path();
            if !(path.contains("::sync::") || path.contains("::rc::")) {
                return None;
            }
            let inner = type_decl_ref_generics(node.as_object()?.get("Adt")?.as_object()?, llbc)?
                .get("types")?
                .as_array()?
                .first()?;
            let inner = strip_ty_indirections(inner, llbc).unwrap_or(inner);
            json_ty_is_statically_sized(inner, llbc).then_some(inner)
        }
        _ => None,
    }
}

/// Declared gckind of the struct or enum `ty` names, after one pointer peel.
///
/// `None` when `ty` is not a struct or enum. A `GcType` seed is Gc. An owner
/// the raw-address classifier rejects (exception carrier, a GC-pointer field)
/// stays a GC base: the pointer bank already kept it as `Ref`. Only an owner
/// [`declared_raw_adt_node`] accepts is Raw.
pub(in crate::front::mir) fn container_declared_gc(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
) -> Option<bool> {
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    let pointee = one_level_pointer_pointee(node, llbc)
        .and_then(|pointee| strip_ty_indirections(pointee, llbc))
        .unwrap_or(node);
    declared_adt_struct_id(pointee, llbc)?;
    // The same raw-storage test as the dest overlay: a GC pointer field
    // in `_immutable_fields_` is a pure raw word (`rewrite_op_getfield`
    // refuses only a non-pure `getfield_raw_r`).
    if declared_raw_adt_node(pointee, llbc, gc_struct_ids, immutable_fields).is_some() {
        return Some(false);
    }
    Some(true)
}

/// `StructId` of a non-builtin struct or enum node, after dedup/`Value` wrappers.
pub(in crate::front::mir) fn declared_adt_struct_id(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<majit_ir::descr::StructId> {
    let node = strip_ty_indirections(node, llbc)?;
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    if type_decl_is_builtin_adt(td) {
        return None;
    }
    match &td.kind {
        TypeDeclKind::Struct(_) | TypeDeclKind::Enum(_) => Some(struct_id_for_type_decl(td)),
        TypeDeclKind::Alias(_) if type_decl_struct_fields(td, llbc).is_some() => {
            Some(struct_id_for_type_decl(td))
        }
        _ => None,
    }
}

/// The node names the published exception carrier ADT.
pub(in crate::front::mir) fn exception_carrier_node(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let carrier = llbc.exception_carrier();
    if carrier.is_empty() {
        return false;
    }
    let Some(def_id) = adt_node_def_id(node) else {
        return false;
    };
    llbc.type_by_id(def_id)
        .is_some_and(|td| td.item_meta.name_path() == carrier)
}

/// A non-builtin struct/enum node whose declared kind is Raw.
///
/// Strings, builtins, scalars, `str` / slice / `dyn`, and zero-sized pointees
/// are `None` so the classifier that already owns them still runs. A pointee
/// present in `gc_struct_ids` is Gc.
pub(in crate::front::mir) fn declared_raw_adt_node<'a>(
    node: &'a serde_json::Value,
    llbc: &'a Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
) -> Option<&'a serde_json::Value> {
    let node = strip_ty_indirections(node, llbc)?;
    if node_is_string_value(node, llbc) {
        return None;
    }
    let sid = declared_adt_struct_id(node, llbc)?;
    // The exception carrier is the object `raise` stores. RPython's
    // exception instance is a GC pointer (`getkind` ref, `flatten.py`
    // `make_return`), not a raw address.
    if exception_carrier_node(node, llbc) {
        return None;
    }
    let name = adt_node_def_id(node)
        .and_then(|id| llbc.type_by_id(id))
        .map(|td| strip_crate_prefix(&td.item_meta.name_path()))
        .unwrap_or_default();
    if struct_id_is_declared_gc(gc_struct_ids, sid, &name) {
        return None;
    }
    if tyref_is_zero_sized(&TyRef::Other(node.clone()), llbc) {
        return None;
    }
    // A raw struct's fields are addresses and scalars. A GC pointer field
    // would be `getfield_raw_r`; `rewrite_op_getfield` raises only for a
    // non-pure `getfield_raw_r`, so an `_immutable_fields_` GC field is
    // still raw storage (`MapNodeTerminatorView.w_cls`).
    let def_id = adt_node_def_id(node)?;
    let mut stack = Vec::new();
    if !adt_def_is_raw_storage(def_id, llbc, gc_struct_ids, immutable_fields, &mut stack) {
        return None;
    }
    Some(node)
}

/// The ADT's fields are raw words: scalars, nested raw structs, or one
/// address pointing at those. A `Vec` / `String` / other GC field is not.
pub(in crate::front::mir) fn adt_def_is_raw_storage(
    def_id: u64,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    stack: &mut Vec<u64>,
) -> bool {
    if stack.contains(&def_id) {
        // A pointer cycle. By-value nesting cannot repeat a def without a
        // pointer, so this is not a depth cutoff and it is not evidence
        // that the owner is GC storage.
        return true;
    }
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    if struct_id_is_declared_gc(
        gc_struct_ids,
        struct_id_for_type_decl(td),
        &strip_crate_prefix(&td.item_meta.name_path()),
    ) {
        return false;
    }
    let carrier = llbc.exception_carrier();
    if !carrier.is_empty() && td.item_meta.name_path() == carrier {
        return false;
    }
    stack.push(def_id);
    let ok = match &td.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            adt_fields_are_raw_storage(td, fields, llbc, gc_struct_ids, immutable_fields, stack)
        }
        TypeDeclKind::Enum(variants) => variants.iter().all(|variant| {
            adt_fields_are_raw_storage(
                td,
                &variant.fields,
                llbc,
                gc_struct_ids,
                immutable_fields,
                stack,
            )
        }),
        TypeDeclKind::Alias(body) => type_node_peel_aliases(body, llbc)
            .and_then(adt_node_def_id)
            .is_some_and(|id| {
                adt_def_is_raw_storage(id, llbc, gc_struct_ids, immutable_fields, stack)
            }),
        _ => false,
    };
    stack.pop();
    ok
}

/// Fields of `owner` are raw words, or a GC pointer listed in
/// `_immutable_fields_` (`rewrite_op_getfield`: only a non-pure
/// `getfield_raw_r` is refused).
pub(in crate::front::mir) fn adt_fields_are_raw_storage(
    owner: &TypeDecl,
    fields: &[FieldDecl],
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    stack: &mut Vec<u64>,
) -> bool {
    fields.iter().enumerate().all(|(i, field)| {
        if field_ty_is_raw_word(&field.ty, llbc, gc_struct_ids, immutable_fields, stack) {
            return true;
        }
        field_ty_is_gc_pointer(&field.ty, llbc, gc_struct_ids)
            && field_declared_immutable(owner, &field_decl_name(i, field), immutable_fields)
    })
}

pub(in crate::front::mir) fn tyref_is_raw_value(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if one_level_pointer_pointee(node, llbc).is_some() {
        return false;
    }
    field_ty_is_raw_word(
        ty,
        llbc,
        gc_struct_ids,
        no_immutable_fields(),
        &mut Vec::new(),
    )
}

pub(in crate::front::mir) fn field_ty_is_raw_word(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    immutable_fields: &ImmutableFieldsTable,
    stack: &mut Vec<u64>,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if node.get("Scalar").is_some() {
        return true;
    }
    if let Some(pointee) = one_level_pointer_pointee(node, llbc) {
        return pointee_is_raw_word(pointee, llbc, gc_struct_ids);
    }
    if let Some(def_id) = adt_node_def_id(node) {
        if adt_path_is_gc_container(llbc, def_id) {
            return false;
        }
        return adt_def_is_raw_storage(def_id, llbc, gc_struct_ids, immutable_fields, stack);
    }
    false
}

/// Whether a field type is a GC pointer (`TO._gckind == 'gc'`).
pub(in crate::front::mir) fn field_ty_is_gc_pointer(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let Some(pointee) = one_level_pointer_pointee(node, llbc) else {
        return false;
    };
    !pointee_is_raw_word(pointee, llbc, gc_struct_ids)
}

/// `Ptr(TO)` is a raw address when `TO._gckind` is raw: a scalar, or an ADT
/// that is not a GC container, the exception carrier, or a `GcType` seed.
/// Recursing into the pointee's fields would treat a raw pointer to a
/// non-GC struct that itself holds a GC word as a GC pointer.
pub(in crate::front::mir) fn pointee_is_raw_word(
    node: &serde_json::Value,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(node) = strip_ty_indirections(node, llbc) else {
        return false;
    };
    if node.get("Scalar").is_some() {
        return true;
    }
    if let Some(def_id) = adt_node_def_id(node) {
        if adt_path_is_gc_container(llbc, def_id) {
            return false;
        }
        if exception_carrier_node(node, llbc) {
            return false;
        }
        let Some(td) = llbc.type_by_id(def_id) else {
            return false;
        };
        if struct_id_is_declared_gc(
            gc_struct_ids,
            struct_id_for_type_decl(td),
            &strip_crate_prefix(&td.item_meta.name_path()),
        ) {
            return false;
        }
        return match &td.kind {
            TypeDeclKind::Struct(_) | TypeDeclKind::Enum(_) | TypeDeclKind::Union(_) => true,
            TypeDeclKind::Alias(body) => type_node_peel_aliases(body, llbc)
                .is_some_and(|n| pointee_is_raw_word(n, llbc, gc_struct_ids)),
            _ => false,
        };
    }
    false
}

/// `Vec` / `String` / `Rc` and the object header. A pointer cycle through
/// `PyObject` is not raw storage: the header holds a class pointer.
pub(in crate::front::mir) fn adt_path_is_gc_container(llbc: &Llbc, def_id: u64) -> bool {
    let Some(path) = llbc.type_by_id(def_id).map(|td| td.item_meta.name_path()) else {
        return true;
    };
    matches!(
        path.as_str(),
        "alloc::vec::Vec"
            | "alloc::string::String"
            | "std::vec::Vec"
            | "std::string::String"
            | "alloc::rc::Rc"
            | "alloc::sync::Arc"
            | "core::cell::RefCell"
    ) || path.contains("::collections::")
        || path.ends_with("pyobject::PyObject")
}

/// Whether an operand's type is one pointer word to a Raw struct or enum.
pub(in crate::front::mir) fn operand_is_declared_raw(
    operand: &Operand,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            ty_is_raw_address(&place.ty, llbc, gc_struct_ids, raw_alloc_flavor_structs)
        }
        Operand::Const(_) => false,
    }
}

/// Whether `ty` is one pointer word to a storage-raw struct or enum.
///
/// `history.getkind` classifies `Ptr(TO)` from `TO._gckind`
/// (`rewrite_op_getfield`). Storage-raw (`declared_raw_adt_node`) answers
/// true for a one-word pointer. A flavor-raw pointee (`rclass.py
/// getgcflavor`) stays a front `Ref`: `crosses_raw_ref` must not emit
/// `cast_ptr_to_int` for it. A borrow of a storage-raw struct that is
/// not a raw enum stays `Ref` (`RootScope::get`).
pub(in crate::front::mir) fn ty_is_raw_address(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| type_node_peel_aliases(node, llbc))
    else {
        return false;
    };
    if raw_ptr_node_is_declared_raw(node, llbc, gc_struct_ids) {
        return true;
    }
    if let Some(pointee) = one_word_pointer_bank_pointee(node, llbc) {
        let is_borrow = node.as_object().is_some_and(|obj| obj.contains_key("Ref"));
        // Non-borrow one-word pointer to a storage-raw ADT (`Box` /
        // `NonNull` / `Arc` of a C struct). A `Ref` of a raw struct
        // stays `Ref` (`RootScope::get`).
        if !is_borrow
            && declared_raw_adt_node(pointee, llbc, gc_struct_ids, no_immutable_fields()).is_some()
        {
            return true;
        }
        // `&mut Enum` / `&Enum` of raw storage is one address word. A raw
        // struct borrow stays `Ref`. Classifying it as an address makes
        // `RootScope::get` a `[Ref, Int]` signature called with `[Int, Int]`.
        // A fieldless enum is the tag integer (`type_decl_is_fieldless_enum`);
        // `Rvalue::Ref` is the identity on that word, so the borrow is not
        // an address (`rclass.py` has no `cast_pointer` of an int).
        if is_borrow && borrowed_pointee_is_raw_enum(pointee, llbc, gc_struct_ids) {
            return true;
        }
        // Arc/Rc of a flavor-raw ADT: the handle word is one raw (non-GC)
        // word (`history.getkind` int) and does not name T
        // (`deref_cast_root`). Identity pointee stays None; bank peels
        // to T. Box / NonNull / `&T` keep identity and stay off this arm.
        if !is_borrow && one_word_pointer_pointee(node, llbc).is_none() {
            let pointee = strip_ty_indirections(pointee, llbc).unwrap_or(pointee);
            if flavor_raw_adt_node(pointee, llbc, raw_alloc_flavor_structs) {
                return true;
            }
        }
    }
    // Niche `Option<NonNull<Raw>>` / `Option<Box<Raw>>` aliases `Some(p)`
    // to the payload word. Classify the Option with that payload's bank.
    // `Option<*mut T>` has no niche (`*mut T` may be null), so it stays the
    // tagged Option instance: `flatten.py` `make_return` / `history.getkind`
    // of that CFG return is `ref`, and `FUNC.RESULT` must agree.
    if let Some(payload) = crate::front::result_exc::tyref_option_payload(ty, llbc)
        && !tyref_is_raw_pointer(&payload, llbc)
    {
        return ty_is_raw_address(&payload, llbc, gc_struct_ids, raw_alloc_flavor_structs);
    }
    false
}

pub(in crate::front::mir) fn borrowed_pointee_is_raw_enum(
    pointee: &serde_json::Value,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(pointee) = strip_ty_indirections(pointee, llbc) else {
        return false;
    };
    let Some(def_id) = adt_node_def_id(pointee) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    matches!(td.kind, TypeDeclKind::Enum(_))
        && !type_decl_is_fieldless_enum(td, llbc)
        && declared_raw_adt_node(pointee, llbc, gc_struct_ids, no_immutable_fields()).is_some()
}

/// Class-root leaf of the Raw pointee behind one pointer word.
///
/// [`adt_node_class_root_with`] declines `core` / `std` / `alloc` heads that
/// carry type arguments, and this returns `None` with it.
pub(in crate::front::mir) fn raw_address_owner_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    let pointee = one_word_pointer_pointee(node, llbc)?;
    let pointee = strip_ty_indirections(pointee, llbc).unwrap_or(pointee);
    let pointee = if declared_raw_adt_node(pointee, llbc, gc_struct_ids, no_immutable_fields())
        .is_some()
        || flavor_raw_adt_node(pointee, llbc, raw_alloc_flavor_structs)
    {
        pointee
    } else {
        return None;
    };
    let root = adt_node_class_root_with(pointee, llbc, tombstoned)?;
    // A flavor-raw class intern's through `getuniqueclassdef` of the
    // same leaf `tyref_class_root_with` paints (`rclass.py getgcflavor`).
    // Named-ADT arguments (`CodeObject<ConstantData>`) are the omitted
    // default of that one class (`struct_instantiation_suffix`); appending
    // them here minted a second ClassDef and `mergeinputargs` could not
    // unify the two.
    if flavor_raw_adt_node(pointee, llbc, raw_alloc_flavor_structs) {
        return Some(root);
    }
    // The class root collapses `Raw<f64>` to `Raw`. The seed still needs
    // the concrete arguments so a field spelled `T` becomes `f64`.
    if root.contains('<') {
        return Some(root);
    }
    let args = pointee
        .as_object()
        .and_then(|obj| obj.get("Adt"))
        .and_then(serde_json::Value::as_object)
        .map(|adt| render_adt_type_args(adt, llbc, 0))
        .unwrap_or_default();
    if args.is_empty() {
        return Some(root);
    }
    Some(format!("{root}<{}>", args.join(",")))
}

/// Class root of a one-word pointer to a harvested `_alloc_flavor_ = raw`
/// ADT (`rclass.py getgcflavor`). The front stays `Ref(Some(root))`; the
/// int bank is `history.getkind` of the raw `InstanceRepr`'s `Ptr`.
pub(in crate::front::mir) fn flavor_raw_pointer_class_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    // Identity pointee: `&T` / `*T` / `Box` / `NonNull` / `AtomicPtr`.
    // Arc / Rc stay off this root (`deref_cast_root`); their bank is
    // [`ty_is_raw_address`] / [`one_word_pointer_bank_pointee`].
    let pointee = one_word_pointer_pointee(node, llbc)?;
    let pointee = strip_ty_indirections(pointee, llbc).unwrap_or(pointee);
    if !flavor_raw_adt_node(pointee, llbc, raw_alloc_flavor_structs) {
        return None;
    }
    adt_node_class_root_with(pointee, llbc, tombstoned)
}

/// Whether a wrapper-stripped node is a `RawPtr` to a Raw struct or enum.
pub(in crate::front::mir) fn raw_ptr_node_is_declared_raw(
    node: &serde_json::Value,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> bool {
    let Some(pointee) = node
        .as_object()
        .and_then(|obj| obj.get("RawPtr"))
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return false;
    };
    declared_raw_adt_node(pointee, llbc, gc_struct_ids, no_immutable_fields()).is_some()
}

pub(in crate::front::mir) fn tyref_to_value_type_with(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    // The Value arm carries the body inline; primitives
    // typically land here.  The Deduplicated arm carries only an
    // ID; consult the dedup-body index to recover the inline shape
    // when it was recorded.  Ids never seen inline (or scanned out
    // of order by the reader) fall back to `Ref` — the same
    // projection downstream uses for any non-primitive shape.
    //
    // Nested `{"Deduplicated": id}` inside `TyRef::Other` is the same
    // wrapper: a Ref pointee of `&i64` is that node
    // (`tyref_ref_pointee_value_type`), not `TyRef::Dedup`. Follow it
    // before matching Scalar so the pointee stays Signed and
    // `retype_word` emits `intmask` (`cast_uint_to_int`) after
    // `ll_slice_getitem_fast_i`.
    let Some(resolved) = tyref_node(ty, llbc) else {
        return ValueType::Ref(None);
    };
    let value = strip_ty_indirections(resolved, llbc).unwrap_or(resolved);
    // `()` is `lltype.Void` (`history.py` `getkind`). Charon serializes
    // it as an empty Tuple ADT; painting that `Ref` interned `SomeTuple`
    // and `map_err` of `Result<(), E>` joined Tuple ∪ OsError
    // (`native_result`, `exceptiontransform.py` `unit_ok`).
    if is_unit_type(ty, llbc) {
        return ValueType::Void;
    }
    // Primitive shapes Charon emits inline.  The literal-type schema
    // splits across two forms:
    //
    //   - atom: `{"Scalar": "Bool"}`, `{"Scalar": "Char"}`.
    //   - object: `{"Scalar": {"Integer": {"Signed": "Isize"}}}`,
    //     `{"Scalar": {"Integer": {"Unsigned": "Usize"}}}`,
    //     `{"Scalar": {"Float": "F64"}}`.
    if let Some(obj) = value.as_object()
        && let Some(scalar) = obj.get("Scalar")
        && let Some(kind) = scalar_value_type(scalar)
    {
        return kind;
    }
    // A trait associated-type projection (`Self::Truth`) has no
    // representation of its own — it is whatever the selected impl binds
    // it to.  The decl-level type cannot name that, but with exactly ONE
    // impl of the trait in this LLBC the binding is recoverable, which is
    // the same walk `charon_type_value_to_ast_string` already performs for
    // the rendered form (`resolve_trait_assoc_type_value`).
    //
    // Resolving it HERE is what keeps one projection in one register bank
    // across the whole program.  A *required* trait method is lowered from
    // the impl, where `Self` is concrete and the projection is already the
    // bound type; a *provided* default body with no override is lowered
    // from the trait, where it is not.  Without this arm the two spellings
    // of the same value disagree on its bank — a caller reaches
    // `concrete_truth_as_bool(_, truth)` wanting `i` and
    // `record_branch_guard(_, truth, ..)` wanting `r`, and no kind for the
    // caller's variable satisfies both.
    if let Some(resolved) = trait_assoc_projection_target(value, llbc) {
        return tyref_to_value_type_with(
            &resolved,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
    }
    // A borrow hides the projection: `&Self::Anchor` is a `Ref` whose
    // pointee is the associated type.  `trait_assoc_projection_target`
    // only sees a top-level `TraitType`, and the transparent peel below
    // only sees an ADT, so the borrow would stay a GC reference while the
    // by-value projection is the impl's word.  Peel first, then resolve,
    // and answer only when the binding is a transparent wrapper: that is
    // the one borrow the transparent peel already types as its field.
    if let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc))
        && let Some(resolved) = trait_assoc_projection_target(node, llbc)
        && let Some(inner) = tyref_transparent_inner_value_type(
            &resolved,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        )
    {
        return inner;
    }
    // RPython `history.getkind` classifies `Ptr(FuncType)` through the raw
    // pointer arm, hence as `int`.  Charon's equivalent is a top-level
    // `FnPtr`: it is the machine address that `FunctionReprBase.call` feeds
    // to `indirect_call`, not a GC reference.  Keeping it in the Ref bank
    // makes `jtransform.handle_regular_indirect_call` spell the otherwise
    // impossible `int_guard_value/r` and gives every following residual call
    // a Ref funcptr operand (`residual_call_*/r...`).
    if type_node_is_fn_ptr(value, llbc) {
        return ValueType::Int;
    }
    // A `Vec` of one-word items is the address of its raw `{ptr, len, cap}`
    // header (`RustVecRepr`, kind `int`), not a GC reference.
    if tyref_rust_vec_item_kind(ty, llbc).is_some() {
        return ValueType::Int;
    }
    // A densely numbered fieldless enum gives `Option<E>` a scalar niche
    // representation.  The whole Option therefore lives in the Int bank;
    // it is not an aggregate reference.
    if tyref_option_fieldless_niche(ty, llbc).is_some() {
        return ValueType::Int;
    }
    // `Option<NonZero*>` stores `None` as the zero the integer forbids.
    // The word is that integer's own bank: unsigned `NonZeroU*` stays
    // `Unsigned`, so `None` is `ConstUInt(0)` and does not join a signed int.
    if let Some(kind) = option_nonzero_word_kind(ty, llbc) {
        return kind;
    }
    // `Option<fn(..)>` is the nullable `Ptr(FuncType)`: one raw machine
    // address, `None` being the null function pointer. `history.getkind`
    // puts `Ptr(FuncType)` in the int bank (the bare `FnPtr` arm above).
    // Leaving the Option in the Ref bank emits `is_(opt:Ref, null:Int)`.
    if tyref_option_payload_is_fn_ptr(ty, llbc) {
        return ValueType::Int;
    }
    // Non-`Literal` (ADT / pointer / tuple) shapes only. Atomic wrappers
    // type as their inner value, including signedness. Checked after the
    // cheap `Literal` fast-path so primitive operands never pay the lookup.
    if let Some(inner) =
        tyref_atomic_inner_value_type(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs)
    {
        return inner;
    }
    // `rstring.py` `StringBuilder` is `SomeStringBuilder`, not the word its
    // one-field Rust struct wraps; checked before the transparent peel.
    if tyref_is_string_builder(ty, llbc) {
        return ValueType::StringBuilder;
    }
    // A `repr(transparent)` GC handle keeps the wrapper's class. The
    // handle intern's as a subclass of `Deref::Target` (`error::PyError`
    // under `error::PyErrorObject`); painting the target made getattr of
    // the handle's `.0` / a `MutRef<Handle>.value` resolve on the payload.
    if transparent_deref_target_class(
        ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )
    .is_some()
        && let Some(root) = tyref_node(ty, llbc)
            .and_then(|node| strip_ty_wrappers(node, llbc))
            .and_then(|node| adt_node_class_root_identity(node, llbc, tombstoned))
    {
        return ValueType::Ref(Some(root));
    }
    // A transparent one-field struct has the same low-level value shape as
    // its field. Charon records the representation in `TypeDecl.layout`, so
    // preserve the field's register bank instead of treating the wrapper as
    // a GC reference. A non-transparent by-value integer newtype
    // (`VersionTag(u64)`) is the same word: `history.getkind` / `call.py`
    // `getcalldescr` read that lltype off the value, not a Ref wrapper.
    if let Some(inner) = tyref_transparent_inner_value_type(
        ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return inner;
    }
    // `OpArg` and compiler-core's `newtype_oparg!` wrappers are transparent
    // `u32` bytecode operands.  Their dependency declarations are opaque in
    // interpreter LLBC, so model the exact upstream family as unsigned
    // integers (`tyref_is_oparg`).  This is not merely a register-bank
    // choice: `SomeInteger(unsigned=True)` carries the non-negative fact
    // through annotation, which is required when an oparg-derived count is
    // passed to a `u32`/`usize` helper.
    if tyref_is_oparg(ty, llbc) {
        return ValueType::Unsigned;
    }
    // A fieldless (C-like) enum is represented by-value as its
    // discriminant integer — RPython has no enum type; a fieldless enum
    // is a named `int` (`SomeInteger`).  Model it as `Int` so the value
    // lives in the int register bank at every site (param, field read,
    // construction, `Discriminant` read): `Rvalue::Discriminant` then
    // aliases the int directly instead of reading a `__discriminant`
    // field off an aggregate `Ref` base whose enum has no rtype clsfield.
    // Rust's default discriminant is `isize` (Signed); Charon's physical
    // tag width may spell `u8` for layout, but the annotation stays
    // Signed so a getattr of the field and a parameter of the same enum
    // compare as one integer kind.
    if tyref_is_fieldless_enum_free(ty, llbc) {
        return ValueType::Int;
    }
    // "At every site" above includes a BORROW of such an enum: `Rvalue::Ref`
    // is the identity on a value that is already its own tag, so `&E` carries
    // no representation of its own and a `fn f(&self)` method on `E` receives
    // the int.  Without this arm the enum is `Int` by value and `Ref(None)`
    // through a borrow, and the `Rvalue::Discriminant` fold above — which
    // tests the place TYPE (the pointee, a fieldless enum) and then aliases
    // the place VALUE (which `resolve_place` collapses a `Deref` out of) —
    // hands a ref-kinded operand to a `SwitchInt`, which
    // `codewriter/flatten.rs` rejects outright (`switch exitswitch must be
    // int`).  `match self { .. }` on a C-like enum is the shape that hits it.
    if tyref_is_borrowed_fieldless_enum_free(ty, llbc) {
        return ValueType::Int;
    }
    // `is_none` / `is_some` take `&self`. The borrow is identity on the
    // Option word (null pointer or tagged enum), so `&Option<P>` uses
    // the same paint as `Option<P>`. `Option<NonZero<_>>` is an integer
    // niche: an ordinary `&Option<NonZero<_>>` parameter stays a
    // pointer (`option_nonzero_word_kind_deref_receiver` peels only
    // the Discriminant place).
    if let Some(option_ty) = tyref_shared_borrow_option_pointee(ty, llbc)
        && option_nonzero_word_kind(&option_ty, llbc).is_none()
    {
        return tyref_to_value_type_with(
            &option_ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
    }
    if ty_is_raw_address(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs) {
        return ValueType::Int;
    }
    // A one-word pointer to a flavor-raw ADT (`rclass.py getgcflavor`)
    // stays `Ref(Some(root))` so the annotator seeds `SomeInstance`
    // and `valuetype_to_concrete_for_legacy` can still map it to
    // Signed. Storage-raw already returned Int above.
    if let Some(root) =
        flavor_raw_pointer_class_root(ty, llbc, tombstoned, raw_alloc_flavor_structs)
    {
        return ValueType::Ref(Some(root));
    }
    // A `str`/`Wtf8`/`Wtf8Buf` value is the single immutable rpy_string in
    // the value model (`tyref_is_string_value`). The exactly-one-raw-pointer
    // form is the same translated value for pyre's owning storage spelling:
    // both `malloc_raw(Wtf8Buf)` and `gc_alloc_storage_box(Wtf8Buf)` return
    // `*mut Wtf8Buf`. Preserve `SomeString` on the return/phi as well as on
    // the struct field; otherwise the GC-box success arm injects a nominal
    // `SomeInstance(Wtf8Buf)` and generalizes `_utf8` away from its
    // `StringRepr`.
    //
    // A Rust-heap `String` is a `{cap, ptr, len}` header, not rstr `STR`
    // (`{hash, chars}`). Painting it `ValueType::Str` makes `ll_str` identity
    // and `pair(StringRepr, StringRepr).rtype_add` → `ll_strconcat` consume
    // that header as a GC string.
    if tyref_is_rust_heap_string(ty, llbc) {
        return ValueType::Ref(tyref_class_root_with(ty, llbc, tombstoned));
    }
    if tyref_is_u8_window(ty, llbc) {
        return ValueType::Int;
    }
    if tyref_is_string_value(ty, llbc)
        || tyref_raw_ptr_pointee_is_string_value(ty, llbc)
        || tyref_is_option_of_string_value(ty, llbc)
    {
        return ValueType::Str;
    }
    if let Some(paint) = tyref_niche_option_payload_paint(ty, llbc, tombstoned) {
        return paint;
    }
    // A payload-carrying enum (any variant has a non-ZST field) is a
    // `SomeInstance` of its interned enum-base classdef.  Fieldless
    // (C-like) enums are already `Int` above; a mixed enum would
    // otherwise fall through to `Ref(None)` and
    // `valuetype_to_someshell(Ref)` would seed `SomeInstance(classdef=None)`,
    // which `union(Instance, Instance)` absorbs over every typed variant.
    //
    // `ValueType::Str` is the same kind of precise paint (a string field
    // must not seed the classdef-less `Ref` shell).  Enums intern through
    // `ClassDef` like structs, so the existing `Ref(Some(root))` carrier
    // is the paint: the intern key is the same class root
    // `Input.class_root` and `getuniqueclassdef_for_struct_root` already
    // consume (`tyref_class_root` → `adt_node_class_root`, the ADT leaf
    // plus a per-instantiation suffix when the head splits).
    // `InstanceRepr.rtype_getattr` (`rclass.rs`) then reads
    // `__discriminant` off that interned base; a classdef-less root
    // falls to `getclsfield`, which is the upstream path — do not
    // synthesize a tag there.
    if let Some(root) = tyref_payload_enum_class_root_with(ty, llbc, tombstoned) {
        return ValueType::Ref(Some(root));
    }
    ValueType::Ref(None)
}

/// Per-artefact inputs of [`tyref_to_value_type`]: the declared GC struct ids and the
/// `_alloc_flavor_ = raw` structs. Both depend only on the artefact.
pub(in crate::front::mir) struct ValueTypeInputs {
    pub(in crate::front::mir) gc_struct_ids: std::collections::HashSet<majit_ir::descr::StructId>,
    pub(in crate::front::mir) raw_alloc_flavor_structs: std::collections::HashSet<String>,
}

/// Values derived from one artefact and reused across queries; stored in `Llbc::derived`,
/// which `Llbc::type_decls_mut` drops.
#[derive(Default)]
pub(in crate::front::mir) struct LlbcDerived {
    pub(in crate::front::mir) value_type_inputs: std::sync::OnceLock<ValueTypeInputs>,
    pub(in crate::front::mir) struct_decls_by_owner: std::sync::OnceLock<StructDeclsByOwner>,
}

pub(in crate::front::mir) fn llbc_derived(llbc: &Llbc) -> &LlbcDerived {
    llbc.derived(LlbcDerived::default)
}

pub(in crate::front::mir) fn value_type_inputs(llbc: &Llbc) -> &ValueTypeInputs {
    llbc_derived(llbc)
        .value_type_inputs
        .get_or_init(|| ValueTypeInputs {
            gc_struct_ids: harvest_declared_gc_facts(llbc).gc_struct_ids(),
            raw_alloc_flavor_structs: crate::front::llbc_hints::harvest_alloc_flavor_raw_from_llbcs(
                std::slice::from_ref(llbc),
            ),
        })
}

pub(in crate::front::mir) fn tyref_to_value_type(ty: &TyRef, llbc: &Llbc) -> ValueType {
    let inputs = value_type_inputs(llbc);
    tyref_to_value_type_with(
        ty,
        llbc,
        no_tombstoned_leaves(),
        &inputs.gc_struct_ids,
        &inputs.raw_alloc_flavor_structs,
    )
}

/// Register-bank kind of the value behind a Charon reference destination.
///
/// `Index::index` and `IndexMut::index_mut` expose `&T` / `&mut T`, but their
/// devirtualized flow operation is RPython's `getarrayitem`, whose result is
/// `T`. Preserve that distinction instead of banking the reference wrapper
/// itself as a GC ref.
pub(in crate::front::mir) fn tyref_deref_value_type_with(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc)) else {
        return ValueType::Ref(None);
    };
    tyref_to_value_type_with(
        &TyRef::Other(node.clone()),
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )
}

/// The register bank of a raw address field. A raw `*u8` is CCHARP in
/// any owner (`rffi.py` `CCHARP`); a raw owner's `*mut i8` is `'int'`
/// (`history.py` `getkind` of a raw `Ptr`). Shared by the field read, the
/// field write and a cast of the read value, so all three agree.
pub(in crate::front::mir) fn raw_field_address_value_type(
    place_ty: &TyRef,
    field_ty: &TyRef,
    owner_declared_gc: Option<bool>,
    llbc: &Llbc,
) -> Option<ValueType> {
    // `rffi.py` `CCHARP`: a raw `*u8` field is CCHARP in any owner,
    // Int-banked so overlay Signed keeps the stamp.
    if let Some(ccharp) =
        raw_u8_ccharp_bank(place_ty, llbc).or_else(|| raw_u8_ccharp_bank(field_ty, llbc))
    {
        return Some(ccharp);
    }
    // `history.py` `getkind` of a raw `Ptr` to a non-byte primitive is
    // `'int'`. Kept out of [`tyref_to_value_type_with`]: an items-base
    // adapter spells the same `*mut i64` and must stay the GC header
    // (`typed_array_base_call_stays_ref`).
    if let Some(raw) = raw_scalar_address_value_type(place_ty, llbc)
        .or_else(|| raw_scalar_address_value_type(field_ty, llbc))
    {
        return Some(raw);
    }
    // Same raw-owner gate as [`fat_part_value_type`] / `field_spelling_lltype`.
    // `tyref_scalar_pointee_value_type` leaves a raw `*mut i8` as the
    // erased GCREF spelling; only a raw owner banks that word as `'int'`.
    if owner_declared_gc == Some(false)
        && (tyref_is_raw_byte_ptr(place_ty, llbc) || tyref_is_raw_byte_ptr(field_ty, llbc))
    {
        return Some(ValueType::Int);
    }
    None
}

/// Register-bank kind of an ADT field read.
///
/// Prefers the place's post-projection type (generic substitution is
/// already applied there).  A shell variant peels a shared borrow of a
/// primitive the way [`tyref_enum_payload_value_type`] does.  A closure-env
/// field whose declared type is that same borrow is the integer too: the
/// body reads the capture through the borrow as a scalar (`Rvalue::Ref`
/// aliases the referent, `resolve_place` collapses `Deref`, and
/// `Ptr(Signed)` does not exist), so a captured `&usize` is the `usize`.
/// Without the peel the field stays `Ref` and a later `int_add` assembles
/// as `int_add/ri>i`.  An ordinary struct's `&P` field is a pointer the
/// program stores and compares, so it keeps the Ref bank.
///
/// A raw `*u8` field is CCHARP in any owner; a raw owner's
/// (`owner_declared_gc == Some(false)`) `*mut i8` field is an address:
/// `history.py` `getkind` of a raw `Ptr` is `'int'`. [`fat_part_value_type`]
/// (`FatData` of a raw owner) and `field_spelling_lltype`
/// (`parent_gckind == Raw` → `Address`) apply the same gate.
pub(in crate::front::mir) fn adt_field_read_value_type(
    place_ty: &TyRef,
    field_ty: &TyRef,
    container_is_enum: bool,
    owner_is_closure_env: bool,
    owner_declared_gc: Option<bool>,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    if let Some(raw) = raw_field_address_value_type(place_ty, field_ty, owner_declared_gc, llbc) {
        return raw;
    }
    let declared = if container_is_enum {
        tyref_enum_payload_value_type(
            place_ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        )
    } else {
        tyref_to_value_type_with(
            place_ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        )
    };
    match declared {
        ValueType::Ref(None) => {
            if owner_is_closure_env
                && let Some(peeled) = tyref_shared_borrow_primitive_value(field_ty, llbc)
                    .or_else(|| tyref_shared_borrow_primitive_value(place_ty, llbc))
            {
                return peeled;
            }
            // A by-value raw struct such as `Position` is not a GC
            // pointer. `getkind` of that word is int, so the load is
            // `getfield_raw_i`. A mutable `getfield_raw_r` is refused.
            if tyref_is_raw_value(place_ty, llbc, gc_struct_ids)
                || tyref_is_raw_value(field_ty, llbc, gc_struct_ids)
            {
                return ValueType::Int;
            }
            tyref_to_value_type_with(
                field_ty,
                llbc,
                tombstoned,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            )
        }
        resolved => resolved,
    }
}

/// Register-bank kind of a tuple `__pos_N` read.
///
/// `place_ty` is the post-projection element. A raw pointer to a non-byte
/// primitive is the address (`history.py` `getkind` of a raw `Ptr`,
/// [`raw_scalar_address_value_type`]); `Ref(None)` would make `raw_load`
/// of `slots().0` (`*mut usize`) see `SomeInstance`.
pub(in crate::front::mir) fn tuple_element_field_value_type(
    place_ty: &TyRef,
    element_ty: Option<TyRef>,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    if let Some(ccharp) = raw_u8_ccharp_bank(place_ty, llbc).or_else(|| {
        element_ty
            .as_ref()
            .and_then(|ty| raw_u8_ccharp_bank(ty, llbc))
    }) {
        return ccharp;
    }
    if let Some(raw) = raw_scalar_address_value_type(place_ty, llbc).or_else(|| {
        element_ty
            .as_ref()
            .and_then(|ty| raw_scalar_address_value_type(ty, llbc))
    }) {
        return raw;
    }
    match tyref_to_value_type_with(
        place_ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        ValueType::Ref(None) => element_ty
            .map(|ty| {
                tyref_enum_payload_value_type(
                    &ty,
                    llbc,
                    tombstoned,
                    gc_struct_ids,
                    raw_alloc_flavor_structs,
                )
            })
            .unwrap_or(ValueType::Ref(None)),
        ty => ty,
    }
}

/// The owner a closure env's `call_once` graph is registered under: the
/// env type's name path without its crate, the same key
/// `impl_method_owner_for_fundecl` stamps on the impl's `self_ty_root`.
pub(in crate::front::mir) fn closure_env_call_once_owner(td: &TypeDecl) -> String {
    strip_crate_prefix(&td.item_meta.name_path())
}

/// Byte compared in a predicate body of the form `item == c`.
pub(in crate::front::mir) fn const_eq_u8_in_unstructured(
    llbc: &Llbc,
    body: &Unstructured,
) -> Option<u8> {
    let mut found = None;
    for bb in &body.body {
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(_, Rvalue::BinaryOp(op, lhs, rhs))) = stmt.stmt_kind_ref()
            else {
                continue;
            };
            if binop_label(&op).ok()?.as_str() != "eq" {
                continue;
            }
            let Some(byte) = operand_const_u8(llbc, &lhs).or_else(|| operand_const_u8(llbc, &rhs))
            else {
                continue;
            };
            match found {
                None => found = Some(byte),
                Some(prev) if prev == byte => {}
                Some(_) => return None,
            }
        }
    }
    found
}

pub(in crate::front::mir) fn operand_const_u8(llbc: &Llbc, op: &Operand) -> Option<u8> {
    let Operand::Const(value) = op else {
        return None;
    };
    match decode_constant(llbc, value).ok()? {
        DecodedConst::Int(n) => u8::try_from(n).ok(),
        DecodedConst::UInt(n) => u8::try_from(n).ok(),
        DecodedConst::UniChar(c) => u8::try_from(c).ok(),
        _ => None,
    }
}

/// Whether `td` is a compiler-generated closure environment.
///
/// Charon stamps those with `src: { Closure: … }`; ordinary ADTs are
/// `"Normal"`.  Identified from that origin, never from the `closure` /
/// `closure#N` name leaf.
pub(in crate::front::mir) fn type_decl_is_closure_env(td: &TypeDecl) -> bool {
    td.src
        .as_ref()
        .is_some_and(|src| src.get("Closure").is_some())
}

/// A compiler-synthesised tuple, `str`, or `Box` declaration. Its `TypeDecl`
/// exists so type refs can name it by id; it is not a user struct.
pub(in crate::front::mir) fn type_decl_is_builtin_adt(td: &TypeDecl) -> bool {
    td.src
        .as_ref()
        .is_some_and(|src| src.get("Builtin").is_some())
}

/// Whether `ty` resolves to a closure-env ADT, after peeling reference
/// wrappers so a `&self` receiver answers the same as a by-value one.
pub(in crate::front::mir) fn tyref_is_closure_env(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_wrappers(node, llbc))
        .and_then(adt_node_def_id)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .is_some_and(type_decl_is_closure_env)
}

pub(in crate::front::mir) fn tyref_shared_borrow_primitive_value(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    let pointee = tyref_shared_borrow_primitive_pointee(ty, llbc)?;
    Some(tyref_to_value_type(&TyRef::Other(pointee), llbc))
}

/// Register-bank kind of an enum variant's payload — `Option<T>`,
/// `Result<T, E>`, `ControlFlow<T, _>`, `Bound<T>`.
///
/// A slice accessor that hands the element back BY REFERENCE puts a `&P`
/// there for a primitive `P`: `<[u8]>::get` returns `Option<&u8>`, so
/// `let tag = *code.get(pc)?; match tag { .. }` reaches the switch through
/// the variant payload rather than through `Index::index` (which
/// [`tyref_deref_value_type_with`] already covers).  That borrow carries no
/// representation of its own — `lltype.py` `class Ptr.__new__` raises
/// `TypeError("can only point to a Container type")`, so `Ptr(Signed)` /
/// `Ptr(Char)` do not exist and the byte is the only representable model —
/// and `resolve_place` already collapses its `Deref` to the base.  Banking
/// it as a GC ref while collapsing the deref hands `codewriter/flatten.rs`
/// a ref-kinded `SwitchInt` operand (`switch exitswitch must be int`).
///
/// Peels EXACTLY ONE level and only here.  [`tyref_to_value_type`] must
/// stay non-peeling for its other consumers: a caller that already
/// unwrapped one reference — [`iterator_payload_element`], which takes
/// `&[&i64]`'s `Option<&&i64>` payload down to the `&i64` element — holds
/// a pointer that IS the value, and a second peel would put it in the
/// integer bank.  A `&&i64` reaching here peels to nothing for the same
/// reason [`tyref_shared_borrow_primitive_pointee`] declines it: its
/// pointee is another reference, not a primitive.
pub(in crate::front::mir) fn tyref_enum_payload_value_type(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    // `rffi.py` `CCHARP`. Int-banked so overlay Signed keeps the stamp.
    if let Some(ccharp) = raw_u8_ccharp_bank(ty, llbc) {
        return ccharp;
    }
    if let Some(pointee) = tyref_shared_borrow_primitive_pointee(ty, llbc) {
        return tyref_to_value_type_with(
            &TyRef::Other(pointee),
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
    }
    // A payload slot read straight out of `generics.types` may still carry
    // the `Value` / `Deduplicated` wrappers. [`tyref_to_value_type_with`]
    // follows those itself; stripping here keeps the node the peel above
    // already resolved.
    match tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) {
        Some(node) => tyref_to_value_type_with(
            &TyRef::Other(node.clone()),
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        ),
        None => ValueType::Ref(None),
    }
}

/// `ty` resolves to an enum, whose variant payloads
/// [`tyref_enum_payload_value_type`] projects through.
///
/// Gates that projection at a field read, where the peel and the ordinary
/// answer are the SAME type node: a `&u8` variant payload and a struct's own
/// `&u8` field both serialize as `{"Ref": [_, {"Scalar": ..}, "Shared"]}`, so
/// nothing in the field's type tells them apart.  The container does — a
/// variant payload is reached by matching on it, so the borrow belongs to the
/// match, and sibling arms that supply the same value by value force one bank
/// across all of them.  A struct's `&P` field is a reference the program wrote
/// down and stores.
///
/// Keyed on enum-ness rather than on a list of named shells: `<[T]>::get`
/// hands a `&P` back through `Option` and `RangeBounds::start_bound` through
/// `Bound`, and only the first is a `?` desugaring.  A named list types the
/// unnamed one's payload as a pointer, which then merges with an integer
/// sibling arm and reaches the assembler as `Move(dst=%i2, src=%r2)`.
///
/// Peels `Ref` on the way in ([`strip_ty_wrappers`]) so a `match *slot { .. }`
/// whose base is a borrow answers the same as a by-value one: the borrow adds
/// no variant and no payload.
pub(in crate::front::mir) fn tyref_is_enum_free(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_wrappers(node, llbc))
        .and_then(adt_node_def_id)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .is_some_and(|td| matches!(td.kind, TypeDeclKind::Enum(_)))
}

/// Whether `ty` resolves to a type Charon's resolved layout reports as
/// zero-sized on every target it emitted.  A zero-sized field occupies no
/// bytes of its owner, so an enum whose every variant field is zero-sized
/// has no payload at all and is the same by-value tag integer a
/// syntactically fieldless enum is.
/// A value with no runtime representation: unit `()`, a layout-size-0
/// struct, or one borrow of such a struct (`&self` on a ZST).
///
/// A closure environment and a function item stay values. Each names one
/// callable, and the annotator has to see which one (`SomePBC`). `getkind`
/// (`history.py`) returns `"void"` only for `lltype.Void`, and that low-level
/// repr belongs to `FunctionsPBCRepr` / `SingleFrozenPBCRepr`, not to this
/// flow value. Erasing a closure env to `Void` makes a `getattr` of
/// `call_once` read a `Constant(None, Void)`. Identified from the type decl's
/// `src: Closure` origin, or from a `FnDef` type node — never from the
/// `closure` name leaf.
///
/// A fieldless enum is a discriminant integer, including through a borrow,
/// so it is not void. `strip_ty_wrappers` peels the borrow; the pointee's
/// layout is what `tyref_is_zero_sized` reads.
pub(in crate::front::mir) fn tyref_is_void_zst(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_keeps_callable_identity(ty, llbc) {
        return false;
    }
    if tyref_is_fieldless_enum_free(ty, llbc) || tyref_is_borrowed_fieldless_enum_free(ty, llbc) {
        return false;
    }
    if is_unit_type(ty, llbc) || tyref_is_zero_sized(ty, llbc) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc)) else {
        return false;
    };
    let peeled = TyRef::Other(node.clone());
    if tyref_keeps_callable_identity(&peeled, llbc) {
        return false;
    }
    if tyref_is_fieldless_enum_free(&peeled, llbc) {
        return false;
    }
    is_unit_type(&peeled, llbc) || tyref_is_zero_sized(&peeled, llbc)
}

/// A closure ADT or a function item (`FnDef`). Layout size 0 does not erase
/// either one: the flow value is the callable the annotator reads.
pub(in crate::front::mir) fn tyref_keeps_callable_identity(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_closure_env(ty, llbc) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    if type_node_fn_def_fun_id(node, llbc).is_some() {
        return true;
    }
    strip_ty_wrappers(node, llbc)
        .is_some_and(|peeled| type_node_fn_def_fun_id(peeled, llbc).is_some())
}

pub(in crate::front::mir) fn tyref_adt_def_id_of(ty: &TyRef, llbc: &Llbc) -> Option<u64> {
    match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
        TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
    }
}

pub(in crate::front::mir) fn tyref_is_zero_sized(ty: &TyRef, llbc: &Llbc) -> bool {
    if adt_path_of_tyref(ty, llbc)
        .is_some_and(|path| crate::codewriter::call::type_str_leaf_is_void_zst(&path))
    {
        return true;
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    tyref_adt_def_id_of(ty, llbc)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .and_then(|td| td.layout_for_target(llbc, &target))
        .is_some_and(|layout| layout.size == Some(0))
}

/// Free-function form of [`Lowering::tyref_is_fieldless_enum`] for the
/// standalone [`tyref_to_value_type`] helper (which holds no `Lowering`):
/// `true` when `ty` resolves to an enum with at least one variant and
/// every variant carrying zero payload fields.
pub(in crate::front::mir) fn tyref_is_fieldless_enum_free(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_fieldless_enum_def(ty, llbc).is_some()
}

/// `ty` is a **borrow** of a fieldless (C-like) enum — `&E` / `&mut E` for an
/// `E` that [`tyref_is_fieldless_enum_free`] accepts.
///
/// A fieldless enum is modelled by-value as its discriminant integer, and
/// `Rvalue::Ref` is the identity on it, so the borrow carries no
/// representation of its own (the same statement
/// [`Lowerer::tyref_is_borrowed_fieldless_enum`] documents).  A `&self`
/// method on such an enum therefore receives the tag, not a pointer to it.
///
/// Peels `Ref` ONLY, deliberately — unlike
/// [`Lowerer::tyref_ref_adt_def_id`], which also peels `RawPtr`.  A
/// `*const E` / `*mut E` is a genuine pointer value that some other code is
/// free to compare, offset or null-check; folding it to the tag would be a
/// wrong answer rather than a missed optimization.  A borrow is not.
pub(in crate::front::mir) fn tyref_is_borrowed_fieldless_enum_free(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    tyref_borrowed_fieldless_enum_def(ty, llbc).is_some()
}

/// Pointee of one shared borrow of `Option<T>` (`&Option<T>`).
///
/// `is_none` / `is_some` take `&self`. The Option representation — null
/// pointer or tagged enum — lives on the by-value word; the borrow adds
/// none, the same way [`tyref_is_borrowed_fieldless_enum_free`] treats
/// `&E`. A raw pointer or a second borrow is a real address and stays
/// out. Mut is out: a write through `&mut Option<T>` is a store to that
/// address (`option_nonzero_word_kind` keeps ordinary NonZero params
/// as pointers for the same reason).
pub(in crate::front::mir) fn tyref_shared_borrow_option_pointee(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<TyRef> {
    let mut v: &serde_json::Value = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let mut peeled_ref = false;
    loop {
        let obj = v.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        if obj.get("RawPtr").is_some() {
            return None;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            if peeled_ref || arr.get(2).and_then(serde_json::Value::as_str) != Some("Shared") {
                return None;
            }
            peeled_ref = true;
            v = arr.get(1)?;
            continue;
        }
        break;
    }
    if !peeled_ref {
        return None;
    }
    let option_ty = TyRef::Other(v.clone());
    crate::front::result_exc::tyref_is_option(&option_ty, llbc).then_some(option_ty)
}

pub(in crate::front::mir) fn tyref_borrowed_fieldless_enum_def<'l>(
    ty: &TyRef,
    llbc: &'l Llbc,
) -> Option<&'l TypeDecl> {
    let mut v: &serde_json::Value = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let mut peeled_a_ref = false;
    loop {
        let obj = v.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            v = arr.get(1)?;
            peeled_a_ref = true;
            continue;
        }
        // Reached a non-indirection node. Only a genuine borrow qualifies;
        // the by-value shape is the caller's other arm.
        if !peeled_a_ref {
            return None;
        }
        let def_id = inline_adt_def_id(v)?;
        return llbc
            .type_by_id(def_id)
            .filter(|td| type_decl_is_fieldless_enum(td, llbc));
    }
}

/// The pointee of `ty` when `ty` is a **shared** borrow of a primitive —
/// `&u8`, `&i64`, `&bool`, `&char`, `&f64` — and `None` otherwise.
///
/// Charon writes a reference as `{"Ref": [region, pointee, kind]}` with `kind`
/// one of `"Shared"` / `"Mut"`, and a primitive as `{"Scalar": ...}`.  Both
/// are required.
///
/// SHARED ONLY, and not by conservatism.  `&mut P` must stay a reference
/// because a write through it is a real operation on a real pointer
/// (`__deref_write` takes the reference as an argument); handing that an
/// integer would corrupt the store.  `*const P` / `*mut P` stay references
/// for the reason [`tyref_is_borrowed_fieldless_enum_free`] already gives —
/// a raw pointer is a value other code may compare, offset or null-check.  A
/// shared borrow of a primitive is none of those: it cannot be written
/// through, and a primitive has no interior mutability to hide behind one.
///
/// Peels one level and the pointee must itself be the primitive, so `&&u8`
/// is DECLINED rather than peeled twice.  That is what makes the answer
/// independent of how many levels the caller already unwrapped.
///
/// Deliberately NOT written in terms of [`strip_ty_wrappers`], which also
/// peels owning wrappers (`Box`, `Rc`, …).  Those are real heap references
/// with representations of their own; only a borrow is the identity.
pub(in crate::front::mir) fn tyref_shared_borrow_primitive_pointee(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<serde_json::Value> {
    let mut v: &serde_json::Value = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    loop {
        let obj = v.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        // `[region, pointee, kind]`.  A `Mut` borrow stops here with `None`.
        let arr = obj.get("Ref").and_then(serde_json::Value::as_array)?;
        if arr.get(2).and_then(serde_json::Value::as_str) != Some("Shared") {
            return None;
        }
        let pointee = arr.get(1)?;
        return tyref_primitive_node(pointee, llbc).cloned();
    }
}

/// `node` resolved through indirections when it is a Charon primitive
/// (`{"Scalar": ...}`); `None` for every aggregate, reference or ADT shape.
///
/// Split out from [`tyref_shared_borrow_primitive_pointee`] because the
/// pointee is reached after an arbitrary number of `Deduplicated` /
/// `Value` hops, exactly as the borrow itself was.
pub(in crate::front::mir) fn tyref_primitive_node<'l>(
    node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    let mut v = node;
    loop {
        let obj = v.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
        return obj.contains_key("Scalar").then_some(v);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) struct FieldlessOptionNiche {
    pub(in crate::front::mir) none_tag: i64,
    pub(in crate::front::mir) storage_ty: &'static str,
}

/// Physical scalar representation of `Option<E>` when `E` is a densely
/// numbered fieldless enum.
///
/// rustc uses E's first invalid scalar as the `None` niche, while every valid
/// E value is `Some(E)`.  Charon does not serialize a separate layout for a
/// generic `core::option::Option<T>` declaration, but it does serialize E's
/// exact tag width and discriminants.  Restricting this projection to the
/// dense `0..N` shape makes the inferred niche unique and rejects enums with
/// explicit gaps, negative tags, or a full scalar range.
pub(in crate::front::mir) fn tyref_option_fieldless_niche(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<FieldlessOptionNiche> {
    if !crate::front::result_exc::tyref_is_option(ty, llbc) {
        return None;
    }
    let option = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let payload = type_decl_ref_generics(option.get("Adt")?.as_object()?, llbc)?
        .get("types")?
        .as_array()?
        .first()?;
    let payload = strip_ty_indirections(payload, llbc)?;
    let td = llbc.type_by_id(adt_node_def_id(payload)?)?;
    let TypeDeclKind::Enum(variants) = &td.kind else {
        return None;
    };
    if variants.is_empty() || variants.iter().any(|variant| !variant.fields.is_empty()) {
        return None;
    }
    for (index, variant) in variants.iter().enumerate() {
        if variant.discriminant_i64()? != index as i64 {
            return None;
        }
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    let storage_ty = td
        .layout_for_target(llbc, &target)?
        .discriminant_int_type()?;
    let none_tag = fieldless_option_none_tag(variants.len(), storage_ty)?;
    Some(FieldlessOptionNiche {
        none_tag,
        storage_ty,
    })
}

/// The value the compiler reserved for `None` in an `Option<E>` over a
/// fieldless enum with `variants` densely numbered variants stored in
/// `storage_ty` — `None` when this layer cannot know it, which is every
/// case today.
///
/// MEASURED, and the measurement refutes `variants` itself as the tag. On
/// this toolchain a dense fieldless enum in a `u8` puts `None` at `2` with
/// two variants but at `255` — the storage maximum, not the variant count —
/// with three, four, five and eight, a 300-variant enum in a `u16` puts it
/// at `65535`, and a single-variant enum is inverted outright (`None` is
/// `0`, `Some` is `1`). So a derived `variants` tag names a value the
/// compiler never writes, the `ne` against it misclassifies every `Some`,
/// and a constructed `None` is indistinguishable from a live variant.
///
/// The tag that would be right is the niche the compiler actually reserved,
/// and this artefact does not carry it: Charon spells a discriminant only as
/// `Branch { offset, int_ty }` or `Known`, with no niche encoding, and runs
/// with `monomorphize: false`, so there is no `Option<E>` layout to read one
/// off either. Rust does not specify the placement, so it is not derivable
/// by rule. Declining here leaves every `Option<fieldless enum>` on the
/// ordinary aggregate lowering.
pub(in crate::front::mir) fn fieldless_option_none_tag(
    _variants: usize,
    _storage_ty: &str,
) -> Option<i64> {
    None
}

/// `Option<NonZeroU*>` / `Option<NonZeroI*>` is one integer. `None` is 0,
/// the value the payload type excludes.
pub(in crate::front::mir) fn tyref_option_nonzero_scalar(ty: &TyRef, llbc: &Llbc) -> bool {
    option_nonzero_word_kind(ty, llbc).is_some()
}

/// Register bank of an `Option<NonZero*>` word. `None` for a 128-bit niche
/// or any option that is not that scalar.
pub(in crate::front::mir) fn option_nonzero_word_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    option_nonzero_word_kind_inner(ty, llbc, false)
}

/// `map_or` reborrows the option, so its discriminant place is
/// `&Option<NonZero<_>>`. That receiver is dereferenced on purpose.
/// An ordinary `&Option<NonZero<_>>` parameter stays a pointer.
pub(in crate::front::mir) fn option_nonzero_word_kind_deref_receiver(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    option_nonzero_word_kind_inner(ty, llbc, true)
}

pub(in crate::front::mir) fn option_nonzero_word_kind_inner(
    ty: &TyRef,
    llbc: &Llbc,
    peel_shared: bool,
) -> Option<ValueType> {
    if !crate::front::result_exc::tyref_is_option(ty, llbc)
        && !(peel_shared && crate::front::result_exc::tyref_is_option_ref(ty, llbc))
    {
        return None;
    }
    let Some(mut option) = tyref_node(ty, llbc) else {
        return None;
    };
    // One shared borrow models the pointee (`&Option<NonZero<_>>` in
    // `map_or`). A raw pointer is the address, and a second borrow is not
    // that receiver.
    let mut peeled_ref = false;
    for _ in 0..24 {
        let Some(obj) = option.as_object() else {
            return None;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            option = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            option = &arr[1];
            continue;
        }
        if obj.get("RawPtr").is_some() {
            return None;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            if !peel_shared
                || peeled_ref
                || arr.get(2).and_then(serde_json::Value::as_str) != Some("Shared")
            {
                return None;
            }
            peeled_ref = true;
            option = arr.get(1)?;
            continue;
        }
        break;
    }
    let Some(payload) = option
        .get("Adt")
        .and_then(|adt| adt.as_object())
        .and_then(|adt| type_decl_ref_generics(adt, llbc))
        .and_then(|generics| generics.get("types"))
        .and_then(|types| types.as_array())
        .and_then(|types| types.first())
        .and_then(|payload| strip_ty_indirections(payload, llbc))
    else {
        return None;
    };
    let Some(path) = adt_node_def_id(payload)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .map(|td| td.item_meta.name_path())
    else {
        return None;
    };
    let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
    if !is_std_nonzero_path(&path) {
        return None;
    }
    nonzero_leaf_word_kind(leaf, payload, llbc)
}

/// `core::num::NonZero*` / `core::num::nonzero::NonZero`. A user ADT whose
/// leaf happens to be `NonZeroU8` is an ordinary aggregate.
pub(in crate::front::mir) fn is_std_nonzero_path(path: &str) -> bool {
    let Some((owner, _)) = path.rsplit_once("::") else {
        return false;
    };
    matches!(
        owner,
        "core::num" | "core::num::nonzero" | "std::num" | "std::num::nonzero"
    )
}

pub(in crate::front::mir) fn nonzero_leaf_word_kind(
    leaf: &str,
    payload: &serde_json::Value,
    llbc: &Llbc,
) -> Option<ValueType> {
    // A niche wider than the machine word stays an aggregate. 128-bit
    // values never fit. On a 32-bit target, `NonZeroU64` / `NonZeroI64`
    // are eight bytes, the same split `scalar_lltype` uses for `i64`.
    if matches!(leaf, "NonZeroU128" | "NonZeroI128")
        || (crate::layout::target_word_size() == 4 && matches!(leaf, "NonZeroU64" | "NonZeroI64"))
    {
        return None;
    }
    if leaf == "NonZero" && adt_type_arg_wider_than_word(payload, llbc) {
        return None;
    }
    match leaf {
        "NonZeroU8" | "NonZeroU16" | "NonZeroU32" | "NonZeroU64" | "NonZeroUsize" => {
            Some(ValueType::Unsigned)
        }
        "NonZeroI8" | "NonZeroI16" | "NonZeroI32" | "NonZeroI64" | "NonZeroIsize" => {
            Some(ValueType::Int)
        }
        "NonZero" => nonzero_type_arg_word_kind(payload, llbc),
        _ => None,
    }
}

pub(in crate::front::mir) fn nonzero_type_arg_word_kind(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<ValueType> {
    let inner = node
        .get("Adt")
        .and_then(|adt| adt.as_object())
        .and_then(|adt| type_decl_ref_generics(adt, llbc))
        .and_then(|generics| generics.get("types"))
        .and_then(|types| types.as_array())
        .and_then(|types| types.first())
        .and_then(|ty| strip_ty_indirections(ty, llbc))?;
    let scalar = inner.get("Scalar")?;
    match scalar_value_type(scalar) {
        Some(ValueType::Unsigned) => Some(ValueType::Unsigned),
        Some(ValueType::Int) => Some(ValueType::Int),
        _ => None,
    }
}

/// `None` of an `Option<NonZero*>` whose class-root spelling names an
/// unsigned word. Signed niches stay `ConstInt(0)`.
pub(crate) fn nonzero_option_zero(owner: &str) -> OpKind {
    if option_owner_is_unsigned_nonzero(owner) {
        OpKind::ConstUInt(0)
    } else {
        OpKind::ConstInt(0)
    }
}

/// `builtin_unichr` (`IntegerRepr.rtype_unichr`): INT code point → UniChar.
/// A Rust `char` stays in the Int register bank (`getkind` of UniChar)
/// while the annotator cell is `SomeUnicodeCodePoint`.
pub(crate) fn unichr_call(arg: Variable) -> OpKind {
    OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: vec![crate::runtime_names::shims::UNICHR.to_string()],
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        },
        args: crate::model::call_args(vec![arg]),
        result_ty: ValueType::Int,
    }
}

/// Emit [`unichr_call`] on `bb` and return the UniChar result.
pub(crate) fn push_unichr(graph: &mut FunctionGraph, bb: BlockId, arg: Variable) -> Variable {
    graph
        .push_op_var(bb, unichr_call(arg), true)
        .expect("unichr produces a value")
}

pub(in crate::front::mir) fn option_owner_is_unsigned_nonzero(owner: &str) -> bool {
    if owner.contains("NonZeroU128")
        || owner.contains("NonZero<u128")
        || owner.contains("NonZeroI128")
        || owner.contains("NonZero<i128")
    {
        return false;
    }
    owner.contains("NonZeroU") || owner.contains("NonZero<u")
}

pub(in crate::front::mir) fn adt_type_arg_wider_than_word(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(inner) = node
        .get("Adt")
        .and_then(|adt| adt.as_object())
        .and_then(|adt| type_decl_ref_generics(adt, llbc))
        .and_then(|generics| generics.get("types"))
        .and_then(|types| types.as_array())
        .and_then(|types| types.first())
        .and_then(|ty| strip_ty_indirections(ty, llbc))
    else {
        return false;
    };
    let Some(kind) = inner
        .get("Scalar")
        .and_then(serde_json::Value::as_object)
        .and_then(|scalar| scalar.get("Integer"))
        .and_then(serde_json::Value::as_object)
        .and_then(|int| int.get("Signed").or_else(|| int.get("Unsigned")))
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };
    matches!(kind, "I128" | "U128")
        || (crate::layout::target_word_size() == 4 && matches!(kind, "I64" | "U64"))
}

/// The `TypeDecl` behind `ty` when it resolves to a fieldless (C-like)
/// enum (at least one variant, every variant carrying zero payload
/// fields); `None` otherwise.  Shared by [`tyref_is_fieldless_enum_free`]
/// and [`tyref_fieldless_enum_class_root`].
pub(in crate::front::mir) fn tyref_fieldless_enum_def<'l>(
    ty: &TyRef,
    llbc: &'l Llbc,
) -> Option<&'l TypeDecl> {
    let def_id = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
        TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
    }?;
    llbc.type_by_id(def_id)
        .filter(|td| type_decl_is_fieldless_enum(td, llbc))
}

/// Discriminant-scalar spelling of a fieldless enum field, including a
/// borrow of one.  `known_struct_names` still contains the enum
/// declaration for switch/debug metadata; treating an inline field of
/// that declaration as a nested Struct makes `all_fielddescrs` drop it
/// (there are no payload fields) and the fallback mints an eight-byte
/// Ref descriptor.  Charon's branching discriminator is the physical
/// low-level type RPython would put in this field (`tyref_to_value_type`
/// colors the same enum `Int`).
pub(in crate::front::mir) fn fieldless_enum_field_layout_spelling(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let decl = tyref_fieldless_enum_def(ty, llbc)
        .or_else(|| tyref_borrowed_fieldless_enum_def(ty, llbc))?;
    let target = std::env::var("TARGET").unwrap_or_default();
    decl.layout_for_target(llbc, &target)
        .and_then(|layout| layout.discriminant_int_type())
        .map(str::to_string)
}

/// The integer-comparison `BinOp` opname a derived `PartialEq` method on
/// a fieldless enum folds to, or `None` for any other method leaf.  Only
/// the two equality halves the `#[derive(PartialEq)]` expansion provides
/// are recognized; ordering (`PartialOrd`) is a separate derive that no
/// fieldless enum in the corpus compares by.
pub(in crate::front::mir) fn fieldless_enum_cmp_binop(method: &str) -> Option<&'static str> {
    match method {
        "eq" => Some("eq"),
        "ne" => Some("ne"),
        _ => None,
    }
}

/// Whether a `TypeDecl` is a fieldless (C-like) enum: at least one
/// variant, every variant carrying zero payload bytes (no fields, or only
/// zero-sized ones).  A zero-sized field occupies no bytes in the enum, so
/// it is not a payload.  Such an enum is modelled by-value as its
/// discriminant integer, so it has no `SomeInstance` at all.
pub(in crate::front::mir) fn type_decl_is_fieldless_enum(td: &TypeDecl, llbc: &Llbc) -> bool {
    match &td.kind {
        TypeDeclKind::Enum(variants) => {
            !variants.is_empty()
                && variants
                    .iter()
                    .all(|v| v.fields.iter().all(|f| tyref_is_zero_sized(&f.ty, llbc)))
        }
        TypeDeclKind::Opaque => opaque_fieldless_enum_tags(llbc, td).is_some(),
        _ => false,
    }
}

/// The variant tags of an `Opaque` declaration whose layout is a tagged
/// enum without fields: another crate's fieldless enum, which this
/// artefact knows only by its layout.  Variant `i` is the value `tags[i]`,
/// as the defining crate's `discriminant_i64` spells it.
pub(in crate::front::mir) fn opaque_fieldless_enum_tags(
    llbc: &Llbc,
    td: &TypeDecl,
) -> Option<Vec<i64>> {
    td.layout_for_target(llbc, &std::env::var("TARGET").unwrap_or_default())?
        .fieldless_enum_tags()
}

/// `ty` itself, or a `Ref`/`RawPtr` of it, names a fieldless (C-like) enum.
pub(in crate::front::mir) fn tyref_fieldless_enum_pointee<'l>(
    ty: &TyRef,
    llbc: &'l Llbc,
) -> Option<&'l TypeDecl> {
    let mut node = tyref_node(ty, llbc)?;
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            node = arr.get(1)?;
            continue;
        }
        if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
            node = arr.first()?;
            continue;
        }
        let def_id = inline_adt_def_id(node)?;
        return llbc
            .type_by_id(def_id)
            .filter(|td| type_decl_is_fieldless_enum(td, llbc));
    }
    None
}

/// Integer bank, byte width, and signedness of a fieldless enum's tag.
pub(in crate::front::mir) fn fieldless_enum_raw_store_descr(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(ValueType, usize, bool)> {
    let td = tyref_fieldless_enum_pointee(ty, llbc)?;
    let layout = td.layout_for_target(llbc, &std::env::var("TARGET").unwrap_or_default())?;
    if let Some(int_ty) = layout.discriminant_int_type() {
        return discr_int_ty_raw_store(int_ty);
    }
    let size = usize::try_from(layout.size?).ok()?;
    matches!(size, 1 | 2 | 4 | 8).then_some((ValueType::Int, size, true))
}

pub(in crate::front::mir) fn discr_int_ty_raw_store(
    int_ty: &str,
) -> Option<(ValueType, usize, bool)> {
    Some(match int_ty {
        "i8" => (ValueType::Int, 1, true),
        "i16" => (ValueType::Int, 2, true),
        "i32" => (ValueType::Int, 4, true),
        "i64" => (ValueType::Int, 8, true),
        "u8" => (ValueType::Unsigned, 1, false),
        "u16" => (ValueType::Unsigned, 2, false),
        "u32" => (ValueType::Unsigned, 4, false),
        "u64" => (ValueType::Unsigned, 8, false),
        _ => return None,
    })
}

/// The crate-stripped `module::Enum` spelling of a fieldless enum `ty`,
/// or `None` when `ty` is not a fieldless enum.  Matches the spelling
/// [`debug_enum_variants_by_discr`] compares against
/// (`strip_crate_prefix(name_path())`) so the `Debug`-fmt collapse can
/// recover the enum's variant table from the value's origin `Input` op.
pub(in crate::front::mir) fn tyref_fieldless_enum_class_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let td = tyref_fieldless_enum_def(ty, llbc)?;
    Some(strip_crate_prefix(&td.item_meta.name_path()))
}

/// Interned class-root spelling of a payload-carrying enum `ty`.
///
/// Fieldless (C-like) enums, including a borrow of one, are the tag
/// integer and have no instance classdef.  A mixed enum — any variant
/// carries a non-ZST field — is a `SomeInstance` of the enum-base class
/// `Bookkeeper::getuniqueclassdef_for_struct_root` intern's.  The
/// spelling matches `OpKind::Input.class_root` ([`tyref_class_root`]):
/// the ADT leaf, plus a per-instantiation suffix when the head splits.
/// `None` when `ty` is not such an enum, or when the root resolver
/// declines it (the core/std/alloc container family, which has dedicated
/// annotator models rather than a classdef).
pub(in crate::front::mir) fn tyref_payload_enum_class_root_with(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    if tyref_is_fieldless_enum_free(ty, llbc) || tyref_is_borrowed_fieldless_enum_free(ty, llbc) {
        return None;
    }
    if !tyref_is_enum_free(ty, llbc) {
        return None;
    }
    tyref_class_root_with(ty, llbc, tombstoned)
}

/// Machine-scalar FUNC.RESULT spellings.  `"ref"` is not one: it collapses
/// every aggregate onto a GC reference, which the call-signature validator
/// must not start enforcing for ordinary struct returns. `"()"` is Void,
/// the declared result of a `fn(...) -> ()` (and of `Result<(), PyError>`
/// after the `Ok` projection).
pub(in crate::front::mir) fn scalar_result_token(token: &str) -> bool {
    matches!(token, "bool" | "i64" | "u64" | "f32" | "f64" | "()")
}

/// FUNC.RESULT token for a non-unit tuple return: `Tuple<t0,t1,…>` of
/// each element's residual token. `model.py SomeTuple` / `rtuple.py`.
/// `None` when `ty` is not a non-unit tuple.
pub(in crate::front::mir) fn tuple_residual_return_token(
    ty: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let stripped = TyRef::Other(node.clone());
    if tyref_tuple_suffix(&stripped, llbc).is_empty() {
        return None;
    }
    let mut items = Vec::new();
    let mut idx = 0usize;
    while let Some(elem) = tyref_tuple_element(&stripped, idx, llbc) {
        items.push(
            dont_look_inside_return_token(
                &elem,
                llbc,
                spec,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            )
            .unwrap_or_else(|| "()".to_string()),
        );
        idx += 1;
    }
    (!items.is_empty()).then(|| format!("Tuple<{}>", items.join(",")))
}

/// Encode the FUNC.RESULT of a `dont_look_inside` callee into the
/// canonical `FunctionGraph.return_type` token the rtyper stub
/// classifier (`cutover.rs` `dont_look_inside` arm) decodes, so the
/// residual-call result shells to the SAME register kind the legacy
/// walker derives from the Call op's `result_ty`
/// (`legacy_resolve.rs infer_concrete_from_op` →
/// `valuetype_to_concrete`).  The classification mirrors `lower_call`'s
/// `result_ty` derivation (unit is void; a non-byte raw scalar pointer
/// is the `i64` token; otherwise `tyref_to_value_type`) so the real path
/// and the legacy walker converge by construction.
///
/// The tokens are the Rust primitive spellings the codewriter's
/// `return_type_string_to_value_type` already recognizes (`bool`/`i64`/
/// `u64`/`f64` → int/float), plus `"()"` for unit (`FUNC.RESULT = Void`)
/// and the `ref` sentinel for any pointer / reference / opaque-ADT return
/// (codewriter wildcards it to `Type::Ref`, matching
/// `valuetype_to_concrete(Ref) = GcRef`).  `None` is only for `Unknown`
/// results (whose legacy concrete is also Unknown — the dead-var backfill
/// family, not a genuine kind mismatch).
pub(in crate::front::mir) fn dont_look_inside_return_token(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    // A `Result<T, PyError>` callee is reached through the residual-call
    // ABI, which hands back the `Ok` payload and routes the error through
    // `BH_LAST_EXC_VALUE`, so `FUNC.RESULT` is `T`.  Reading the kind off
    // the `Result` ADT instead reports `ref` whatever `T` is, and the
    // caller — whose call result is the unwrapped payload — then disagrees
    // with the token for every non-reference `T` (`getcalldescr`'s
    // `RESULT == FUNC.RESULT` check).  `Result<(), PyError>` reaches the
    // unit arm the same way its callee reaches `widen_unit_return_to_void`.
    //
    // Only a `PyError` carrier is lowered that way.  Any other `Result`
    // stays a real ADT the callee returns whole and the caller matches on
    // (`rbigint::_AsDouble -> Result<f64, RBigIntError>`), so projecting it
    // would name the payload's bank for a value that arrives as a
    // reference — the same disagreement in the other direction.
    //
    // The `OBJECTPTR` carve-out below deliberately keeps reading the
    // declared `output`: it decides the pointer's lowering, not its bank,
    // and a `Result`-typed output has never taken it.
    let payload = if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec) {
        crate::front::result_exc::tyref_result_ok(output, llbc)
    } else {
        None
    };
    let kind_src = payload.as_ref().unwrap_or(output);
    if is_unit_type(kind_src, llbc) {
        return Some("()".to_string());
    }
    // `history.py` `getkind` of a raw `Ptr` is `int`, including `*mut f64`.
    // `return_type_string_to_value_type` reads the `i64` token as that int.
    // A byte pointer stays on the `ref` arm below.
    if raw_scalar_address_value_type(kind_src, llbc).is_some() {
        return Some("i64".to_string());
    }
    // A pair slice returns its item pointer word; the length goes through
    // the caller's slot.
    if tyref_pair_slice_item_kind(kind_src, llbc)
        .is_some_and(|kind| PAIR_SLICE_ITEM_KINDS.contains(&kind))
    {
        return Some("u64".to_string());
    }
    // A function-return tuple is `model.py SomeTuple` (`rtuple.py`).
    // Stamp each element's residual token *before* `tyref_to_value_type`
    // collapses the ADT to `Ref` / `"ref"` — a two-word
    // `(PyObjectRef, PyObjectRef)` is not one GC word, and the collector
    // would skip that `"ref"` as multiword.
    if let Some(token) = tuple_residual_return_token(
        kind_src,
        llbc,
        spec,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return Some(token);
    }
    let token = match tyref_to_value_type_with(
        kind_src,
        llbc,
        no_tombstoned_leaves(),
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        ValueType::Bool => "bool",
        ValueType::Int => {
            // A storage-raw pointer is `ValueType::Int` (`ty_is_raw_address`)
            // and still names that owner. The token stays in the int bank
            // (`return_type_string_to_value_type`) while the stub shell is
            // `SomePtr` of the raw struct.
            if let Some(root) = raw_address_owner_root(
                kind_src,
                llbc,
                no_tombstoned_leaves(),
                gc_struct_ids,
                raw_alloc_flavor_structs,
            ) {
                return Some(format!("raw:{root}"));
            }
            "i64"
        }
        ValueType::Unsigned => "u64",
        ValueType::Int128 => "i128",
        ValueType::UInt128 => "u128",
        ValueType::Float => "f64",
        ValueType::SingleFloat => "f32",
        // A `*mut PyObject` result keeps its typed `OBJECTPTR` lowering, so a
        // caller reading the returned object's fields rtypes against the real
        // struct; the generic `GCREF` `ref` token would erase that and stub
        // the opaque callee with the wrong low-level pointer.
        ValueType::Ref(_) if output_type_is_objectptr(output, llbc) => {
            return Some(crate::translator::rtyper::cutover::OBJECTPTR_RETURN_TYPE.to_string());
        }
        // An rstr `STR` result is the `SomeString` its body produces, not
        // the classdef-less instance behind `ref`.
        ValueType::Str => crate::translator::rtyper::cutover::STR_RETURN_TYPE,
        ValueType::Ref(_) => {
            // Flavor-raw pointer: front stays `Ref(Some(flavor))` while
            // `history.getkind` of the raw `Ptr` is int. Stamp `raw:<owner>`
            // so residual FUNC.RESULT is Int and `raw_ptr_token_annotation`
            // seeds `SomeInstance`.
            if let Some(root) = flavor_raw_pointer_class_root(
                kind_src,
                llbc,
                no_tombstoned_leaves(),
                raw_alloc_flavor_structs,
            ) {
                return Some(format!("raw:{root}"));
            }
            "ref"
        }
        ValueType::StringBuilder => "ref",
        ValueType::Void | ValueType::State | ValueType::Unknown => return None,
    };
    Some(token.to_string())
}

/// Class key of a callee's by-value ADT result — the value
/// `FunctionGraph::return_class_root` carries so a `dont_look_inside` stub
/// returns `SomeInstance(classdef)` instead of the classdef-less `ref`
/// shell.  Reads the same `Result<T, PyError>` payload
/// [`dont_look_inside_return_token`] reads.
///
/// An enum instantiation keys as its template path plus the `<…>` suffix,
/// the spelling a `Some(..)` / `Ok(..)` construction of that instantiation
/// mints.  Only instantiations whose type arguments are all by-value ADTs
/// qualify: a reference, raw-pointer or `Box` argument can make the enum a
/// null-niche pointer, which has no enum class.  A named struct keys as its
/// class root.  `None` otherwise.
pub(in crate::front::mir) fn dont_look_inside_return_class_root(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Option<String> {
    let payload = if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec) {
        crate::front::result_exc::tyref_result_ok(output, llbc)
    } else {
        None
    };
    let ty = payload.as_ref().unwrap_or(output);
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let adt = node.as_object()?.get("Adt")?.as_object()?;
    let td = llbc.type_by_id(adt_node_def_id(node)?)?;
    if !matches!(td.kind, TypeDeclKind::Enum(_)) {
        return adt_node_class_root_with(node, llbc, no_tombstoned_leaves());
    }
    let type_args = adt
        .get("generics")
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)?;
    if type_args.is_empty() {
        return adt_node_class_root_with(node, llbc, no_tombstoned_leaves());
    }
    let by_value_args = type_args.iter().all(|arg| {
        strip_ty_indirections(arg, llbc).is_some_and(|arg| {
            adt_node_def_id(arg).is_some() && type_node_box_pointee(arg, llbc).is_none()
        })
    });
    if !by_value_args {
        return None;
    }
    let suffix = adt_head_instantiation_suffix(adt, llbc)?;
    Some(format!("{}{suffix}", td.item_meta.name_path()))
}

/// Field-layout spelling of a `*mut Vec<T>` / `*mut VecDeque<T>` / `*mut [T]`
/// result, or `None`.
///
/// A field of that type projects to `SomeList` (`raw_ptr_pointee_container_root`);
/// the `ref` token alone shells the call as a classdef-less instance, and
/// `List ∪ Instance` is a phi `pairtype` never builds. The spelling is the
/// same string `published_struct_field_layout` stores, so the stub and the
/// field project one list.
///
/// A one-word `Vec<T>` is the raw `{ptr, len, cap}` header (`RustVecRepr`).
/// [`tyref_to_field_layout_string`] spells that word `isize`, which this
/// filter drops, so the `dont_look_inside` trampoline of `look_inside_iff`
/// (`rlib/jit.py`) would annotate as `SomeInteger` while `func(...)`
/// annotates as `SomeRustVec`. [`annotation_struct_field_type`] is the
/// spelling `Bookkeeper::project_rust_vec` reads. The FUNC.RESULT token
/// stays `i64`: `history.getkind` of the header is `int`.
pub(in crate::front::mir) fn dont_look_inside_return_container_root(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Option<String> {
    // Same Result<T, PyError> payload [`dont_look_inside_return_token`]
    // reads. `encode_ascii` returns `Result<Vec<u8>, PyError>`; looking
    // at the Result ADT yields no Vec/slice spelling, so the stub was a
    // classdef-less `ref` that `List ∪ Instance` cannot merge.
    let payload = if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec) {
        crate::front::result_exc::tyref_result_ok(output, llbc)
    } else {
        None
    };
    let ty = payload.as_ref().unwrap_or(output);
    // One-word-item `Vec` is `*mut Vec<…>` (`annotation_struct_field_type`);
    // `tyref_to_field_layout_string` writes `isize` and the stub would be
    // `SomeInteger`. The `look_inside_iff` wrapper then merges that with
    // the inlined `_orig_` body's `SomeRustVec` (`ll_newlist` after
    // `l.length = n`).
    let spelling = annotation_struct_field_type(
        ty,
        llbc,
        gc_struct_ids,
        &std::collections::HashSet::<String>::new(),
    );
    if majit_ir::descr::is_shared_u8_slice_spelling(&spelling)
        || majit_ir::descr::is_option_of_u8_window_spelling(&spelling)
    {
        return Some(spelling);
    }
    slice_or_vec_container_root_spelling(&spelling).map(str::to_string)
}

/// Spellings whose residual stub must be `SomeList`, not a classdef-less
/// `ref`. `*mut Vec<T>` / `*mut [T]` already qualify; a shared `&[u8]`
/// is a fat window and is excluded below. `getanyitem` on the
/// classdef-less shell panics.
pub(in crate::front::mir) fn slice_or_vec_container_root_spelling(spelling: &str) -> Option<&str> {
    if majit_ir::descr::is_shared_u8_slice_spelling(spelling) {
        return None;
    }
    let pointee = spelling
        .strip_prefix("*const ")
        .or_else(|| spelling.strip_prefix("*mut "))
        .map(str::trim)
        .unwrap_or_else(|| peel_ref_prefix(spelling));
    // `[T]` is a GcArray / SomeList. `[T; N]` is a fixed Array: the
    // outermost brackets carry `; N` (`slice_element_spelling`).
    let is_unsized_slice = slice_element_spelling(pointee).is_some();
    (pointee.starts_with("Vec<") || pointee.starts_with("VecDeque<") || is_unsized_slice)
        .then_some(spelling)
}

/// Whether a callee's result is a pointer to the low-level `STR` storage
/// (`*mut Utf8Str` / rstr `rpy_string`) — the value
/// `FunctionGraph::return_is_str` carries so a `dont_look_inside` stub
/// returns `SomeString` instead of the classdef-less `ref` shell.  Reads the
/// same `Result<T, PyError>` payload [`dont_look_inside_return_token`]
/// reads.  A by-value Rust `String` does not qualify: the residual-call ABI
/// cannot return its three words. A shared `&[u8]` is a window, not STR.
pub(in crate::front::mir) fn dont_look_inside_return_is_str(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> bool {
    let payload = if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec) {
        crate::front::result_exc::tyref_result_ok(output, llbc)
    } else {
        None
    };
    let ty = payload.as_ref().unwrap_or(output);
    tyref_is_string_value(ty, llbc)
        || tyref_raw_ptr_pointee_is_string_value(ty, llbc)
        || tyref_is_option_of_string_value(ty, llbc)
}

/// True when `ty` (after stripping `&`/`&mut`/`*` wrappers) resolves to the
/// exact `majit_rlib::rbigint::RBigInt` ADT. Guards the
/// RBigInt binary-operator retarget (`front::bigint_binop`) so a same-named
/// operator (`BitAnd`/`Sub`/`Mul`/…) on any other type is never redirected to
/// a `jit_bigint_*` residual that would misread its non-RBigInt pointer.
///
/// In `pyre-object.ullbc` the declaration is the local struct whose bodies
/// stay available to source translation. In dependent-crate LLBCs Charon
/// records that same nominal type as an external/opaque declaration. The
/// qualified identity is therefore the invariant; requiring either storage
/// kind would make the rewrite depend on which crate owns the caller.
pub(in crate::front::mir) fn tyref_is_rbigint(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            let path = td.item_meta.name_path();
            path == "rbigint::RBigInt" || path.ends_with("::rbigint::RBigInt")
        })
}

/// True when `ty` (after stripping `&`/`&mut`/`*` wrappers) resolves to
/// `adt_path` whose sole element type is a word-sized signed integer
/// (`I64` / `Isize`). Exclusive `Range` is a struct ADT; inclusive
/// `RangeInclusive` is a struct after CHARON include.
pub(in crate::front::mir) fn tyref_is_word_signed_range_adt(
    ty: &TyRef,
    llbc: &Llbc,
    adt_path: &str,
    require_opaque: bool,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(adt) = node
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|v| v.as_object())
    else {
        return false;
    };
    let is_range = adt_node_def_id(node)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            (!require_opaque || matches!(td.kind, TypeDeclKind::Opaque))
                && td.item_meta.name_path() == adt_path
        });
    if !is_range {
        return false;
    }
    let elem = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .and_then(|t| t.first())
        .and_then(|n| strip_ty_wrappers(n, llbc));
    matches!(
        elem.and_then(|e| e.as_object())
            .and_then(|m| m.get("Scalar"))
            .and_then(|l| l.get("Integer"))
            .and_then(|l| l.get("Signed"))
            .and_then(serde_json::Value::as_str),
        Some("I64" | "Isize")
    )
}

/// True when `ty` is `core::ops::range::RangeInclusive` over `I64` /
/// `Isize`. Gates the AnonConst un-promote of a promoted `(a..=b)` so a
/// same-named static of any other type, and a character range, stays a
/// Global read. `bookkeeper.py` `immutablevalue`.
pub(in crate::front::mir) fn tyref_is_int_range_inclusive(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_word_signed_range_adt(ty, llbc, "core::ops::range::RangeInclusive", false)
}

pub(in crate::front::mir) fn tyref_to_attr_value_type_with(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    let Some(resolved) = tyref_node(ty, llbc) else {
        return ValueType::Ref(None);
    };
    let value = strip_ty_indirections(resolved, llbc).unwrap_or(resolved);
    if let Some(obj) = value.as_object()
        && let Some(scalar) = obj.get("Scalar")
        && let Some(kind) = scalar_value_type(scalar)
    {
        return kind;
    }
    // Non-`Literal` (ADT / pointer / tuple) shapes only.  Atomic wrappers
    // type as their inner value; unlike [`tyref_to_value_type`] the
    // signed/unsigned split is kept here (`AtomicUsize` → `Unsigned`) so
    // the per-field someshell matches the inner scalar.  Checked after
    // the cheap `Literal` fast-path so primitive fields never pay it.
    if let Some(inner) =
        tyref_atomic_inner_value_type(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs)
    {
        return inner;
    }
    // Matching [`tyref_to_value_type`]: a `Vec` of one-word items is the
    // address of its raw `{ptr, len, cap}` header (`RustVecRepr`, kind
    // `int`). Layout and the register bank keep that Int. FORCE does not
    // seed the field (`register_struct_fields` skips a Vec layout row) so
    // `project_struct_rows` installs `SomeRustVec` from the `*mut Vec<…>`
    // annotation spelling. An Integer FORCE made `len` of the field
    // `'int' has no length`.
    if tyref_rust_vec_item_kind(ty, llbc).is_some() {
        return ValueType::Int;
    }
    // `rutf8.py` `Utf8StringBuilder._s` is a `StringBuilder` instance
    // attribute: seed it `SomeStringBuilder`, matching the value site.
    if tyref_is_string_builder(ty, llbc) {
        return ValueType::StringBuilder;
    }
    if let Some(inner) = tyref_transparent_inner_value_type(
        ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return inner;
    }
    if ty_is_raw_address(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs) {
        return ValueType::Int;
    }
    // Flavor-raw pointer field: front `Ref(Some(root))` so FORCE is a
    // classdef-less Instance that Pass-2 replaces with
    // `project_struct_field_type`'s `SomeInstance(pointee)`, and so
    // `valuetype_to_concrete_for_legacy` sees `Ref(Some(flavor))` → Signed.
    if let Some(root) =
        flavor_raw_pointer_class_root(ty, llbc, tombstoned, raw_alloc_flavor_structs)
    {
        return ValueType::Ref(Some(root));
    }
    // `history.py` `getkind` of a raw `Ptr` to a non-byte primitive is
    // `'int'`. Kept out of [`tyref_to_value_type_with`]: an items-base
    // adapter spells the same `*mut i64` and must stay the GC header
    // (`typed_array_base_call_stays_ref`). FORCE of a named `*mut usize`
    // field (and harvest of a Charon tuple TypeDecl) must match
    // [`tuple_field_value_type`] so `_init_classdef` does not seed
    // `Ref(None)` over the positional Int attr.
    if let Some(raw) = raw_scalar_address_value_type(ty, llbc) {
        return raw;
    }
    // A fieldless (C-like) enum field is represented by-value as its
    // discriminant integer, matching `tyref_to_value_type` which colors
    // the same enum `Int` at every value site (construction, field read,
    // `Discriminant`).  Seed the attr as `Int` so the forced class
    // attribute agrees with the discriminant write; otherwise the field
    // seeds a classdef-less `SomeInstance(None)` shell and the first typed
    // write (`err.kind = PyErrorKind::…`) unions `Integer ∪ Instance` with
    // no `pair(SomeInteger, SomeInstance).union()` handler and walls.
    if tyref_is_fieldless_enum_free(ty, llbc) {
        return ValueType::Int;
    }
    // Matching [`tyref_to_value_type`]: a borrow of a fieldless enum is
    // the same tag word (`tyref_is_borrowed_fieldless_enum_free`). FORCE
    // must seed Integer so a closure-env capture of `&E` setattr's the
    // int (`rclass.py` `InstanceRepr.setfield`), not `Integer ∪ Instance`.
    if tyref_is_borrowed_fieldless_enum_free(ty, llbc) {
        return ValueType::Int;
    }
    // Matching [`tyref_to_value_type`]: a shared borrow of `Option<P>`
    // is the Option word. `Option<NonZero<_>>` stays a pointer param.
    if let Some(option_ty) = tyref_shared_borrow_option_pointee(ty, llbc)
        && option_nonzero_word_kind(&option_ty, llbc).is_none()
    {
        return tyref_to_attr_value_type_with(
            &option_ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
    }
    // A `str`/`Wtf8` field seeds a `SomeString` attr shell (via
    // valuetype_to_someshell) instead of the classdef-less `Ref(None)`
    // SomeInstance the fallback yields; the latter walls when the first
    // typed string write unions `String ∪ Instance(classdef-less)` with no
    // pair(SomeString, SomeInstance).union() handler.  The raw-pointer form
    // is the same translated value when a Rust object field owns the buffer
    // behind `*mut Wtf8Buf` (`W_UnicodeObject.value`).
    // A Rust-heap `String` is not rstr `STR`; seed it as `SomeInstance(String)`.
    if tyref_is_rust_heap_string(ty, llbc) {
        return ValueType::Ref(tyref_class_root_with(ty, llbc, tombstoned));
    }
    if tyref_is_u8_window(ty, llbc) {
        return ValueType::Int;
    }
    if tyref_is_string_value(ty, llbc)
        || tyref_raw_ptr_pointee_is_string_value(ty, llbc)
        || tyref_is_option_of_string_value(ty, llbc)
    {
        return ValueType::Str;
    }
    if let Some(paint) = tyref_niche_option_payload_paint(ty, llbc, tombstoned) {
        return paint;
    }
    // `history.getkind` puts `Ptr(FuncType)` in the int bank. A struct
    // field or enum payload (`Option<fn>`'s `Some` word, a stored `fn`)
    // must use the same classifier as a value site. Leaving it
    // `Ref(Some(root))` below colors the field Ref while the place is
    // Int, and the join copies Ref into an Int inputarg.
    if type_node_is_fn_ptr(value, llbc) || tyref_option_payload_is_fn_ptr(ty, llbc) {
        return tyref_to_value_type_with(
            ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
    }
    // `Option<NonZero*>` and a fieldless-enum niche are integers at every
    // value site. Seeding the field as `Ref(Some(root))` below makes the
    // first `Some`/`None` write union an integer with an instance.
    if let Some(kind) = option_nonzero_word_kind(ty, llbc) {
        return kind;
    }
    if tyref_option_fieldless_niche(ty, llbc).is_some() {
        return ValueType::Int;
    }
    // Matching [`tyref_to_value_type`]: a payload-carrying enum field
    // seeds `Ref(Some(root))` so the FORCE-attr / call-result narrow
    // intern the enum base instead of the classdef-less `Ref(None)` shell.
    if let Some(root) = tyref_payload_enum_class_root_with(ty, llbc, tombstoned) {
        return ValueType::Ref(Some(root));
    }
    // A raw `*u8` / `CCHARP` field stays `Ref(None)` here so FORCE is
    // the untyped shell `project_struct_field_type` replaces with
    // `lltype_to_annotation(CCHARP)`. Unsigned would seed `SomeInteger`.
    ValueType::Ref(None)
}

/// Whether a declared field type is one of the decl's own type parameters
/// (`core::ops::range::Range<Idx>::start`, `Option<T>::Some.0`).
///
/// Such a field has no register class the decl can name: `Range<usize>`
/// stores an int there, `Range<f64>` a float.  Its FORCE attr is `Unknown`,
/// which `valuetype_to_someshell` does not shell, so `register_struct_fields`
/// forces nothing for it and the attribute takes the annotation of the
/// values written to it.  Forcing the classdef-less `Ref(None)` fallback
/// instead made every `start = <int>` write union `Integer ∪ Instance` and
/// fail.
pub(in crate::front::mir) fn tyref_is_type_parameter(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .and_then(typevar_bound_index)
        .is_some()
}

/// Register-class of a struct field, matching [`tyref_to_attr_value_type`]
/// except for a closure-env capture whose declared type is a shared borrow
/// of a primitive: seed that attr as the scalar so FORCE does not install
/// a Ref class field against an Int-banked getfield.  Ordinary struct
/// fields of reference type stay `Ref`; a type-parameter field is
/// `Unknown` ([`tyref_is_type_parameter`]).
pub(in crate::front::mir) fn tyref_to_attr_value_type_for_struct_field(
    ty: &TyRef,
    owner: &TypeDecl,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> ValueType {
    if tyref_is_type_parameter(ty, llbc) {
        return ValueType::Unknown;
    }
    if type_decl_is_closure_env(owner)
        && let Some(peeled) = tyref_shared_borrow_primitive_value(ty, llbc)
    {
        return peeled;
    }
    tyref_to_attr_value_type_with(
        ty,
        llbc,
        no_tombstoned_leaves(),
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )
}

/// The bare leaf name of `ty`'s named-ADT root, after stripping
/// reference wrappers (`&T` / `&mut T` → `T`, the same contract as
/// [`tyref_to_ast_string`]).  This is the value `OpKind::Input.class_root`
/// carries so `derive_subject_inputcells`
/// (`flowspace_adapter.rs`) can seed a `Ref` parameter with
/// its cached struct-root `ClassDef` instead of the classdef-less
/// `SomeInstance` shell.
///
/// Raw pointers (`*const T` / `*mut T`) resolve to their pointee's
/// ADT root: a `*mut W_Foo` parameter is the same instance-lattice
/// value as a `&W_Foo` one (upstream `SomePtr(PTRTYPE)` carries the
/// pointee type either way).  The pointer-method answer (`is_null`)
/// stays intact — `SomeInstance.getattr` resolves it as a bound
/// method BEFORE projecting the classdef (`unaryop.rs`), so a
/// seeded classdef no longer bypasses it.
///
/// Returns `None` for:
///   - primitives / tuples / builtin containers (no class root);
///   - generic ADT instantiations (`Arg<u32>`) — the registry rows for
///     a generic decl carry unresolved type-variable field strings, so
///     a seeded classdef would project bogus attr shells.
#[allow(dead_code)] // rtyper classdef root of a pointer ADT
pub(in crate::front::mir) fn tyref_class_root(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    tyref_class_root_with(ty, llbc, no_tombstoned_leaves())
}

pub(in crate::front::mir) fn tyref_class_root_with(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    adt_node_class_root_with(node, llbc, tombstoned)
        .or_else(|| raw_ptr_pointee_class_root_with(node, llbc, tombstoned))
}

/// The underlying JSON type node of a `TyRef`, resolving the `Dedup`
/// indirection through the LLBC dedup-body index.
pub(in crate::front::mir) fn tyref_node<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

/// Whether a `TyRef` resolves (behind the usual wrappers) to the
/// `alloc::string::String` ADT.
pub(in crate::front::mir) fn tyref_is_wtf8buf(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path().ends_with("Wtf8Buf"))
}

pub(in crate::front::mir) fn tyref_is_string_adt(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path() == "alloc::string::String")
}

/// A by-value or exactly-one-raw-pointer `alloc::string::String`.
///
/// That ADT is a Rust-heap `{cap, ptr, len}` header. It is not rstr `STR`
/// (`rpy_string` / `Utf8Str`: `{hash, chars}`), so it must not paint as
/// `ValueType::Str`.
pub(in crate::front::mir) fn tyref_is_rust_heap_string(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_string_adt(ty, llbc) {
        return true;
    }
    let Some(pointee) = tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| n.as_object()?.get("RawPtr")?.as_array()?.first())
        .and_then(|n| strip_ty_wrappers(n, llbc))
    else {
        return false;
    };
    adt_node_def_id(pointee)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path() == "alloc::string::String")
}

/// Class root of the Rust-heap `alloc::string::String` (crate-stripped
/// identity `string::String`; the bare `String` spelling is still
/// produced by synthesized roots).
pub(in crate::front::mir) fn is_rust_heap_string_root(root: &str) -> bool {
    matches!(root, "String" | "string::String")
}

/// Class identity of `pyre_object::pyobject::PyObject`, the `W_Root` port
/// (`bookkeeper.py Bookkeeper.getdesc`).
pub(crate) const W_ROOT_CLASS_ROOT: &str = "pyobject::PyObject";

/// Class identity of `pyre_object::dictmultiobject::ObjectKey`.
pub(crate) const OBJECT_KEY_CLASS_ROOT: &str = "dictmultiobject::ObjectKey";

/// Exact class-root spellings of the rstr `STR` / WTF-8 string family.
///
/// `class_identity_spelling` of the declarations:
/// - `pyre_object::unicodeobject::Utf8Str` → `unicodeobject::Utf8Str`
/// - `pyre_object::unicodeobject::UnicodeValueStorage` (`type` alias;
///   `resolve_adt_field` keeps the alias owner) →
///   `unicodeobject::UnicodeValueStorage`
/// - `pyre_object::bytesobject::BytesBlock` (`type BytesBlock = Utf8Str`,
///   unpeeled owner) → `bytesobject::BytesBlock`
/// - `pyre_object::dictmultiobject::BytesKey` → `dictmultiobject::BytesKey`
/// - `pyre_object::celldict::StrKey` → `celldict::StrKey`
/// - `rustpython_wtf8::Wtf8` / `Wtf8Buf` (crate-root) → `Wtf8` / `Wtf8Buf`
///
/// Bare leaves stay because [`charon_adt_to_ast_string`] still emits the
/// TypeDecl leaf for field-type AST strings
/// (`Bookkeeper.project_struct_field_type`).
pub(crate) const STR_FAMILY_CLASS_ROOTS: &[&str] = &[
    "unicodeobject::Utf8Str",
    "unicodeobject::UnicodeValueStorage",
    "bytesobject::BytesBlock",
    "dictmultiobject::BytesKey",
    "celldict::StrKey",
    "Wtf8",
    "Wtf8Buf",
    // [`charon_adt_to_ast_string`] field-type leaves:
    "Utf8Str",
    "UnicodeValueStorage",
    "BytesBlock",
    "BytesKey",
    "StrKey",
];

/// FieldRead `owner_root` of an rstr `STR` length word
/// (`LLHelpers.ll_strlen`). Produced by `resolve_adt_field` /
/// `class_identity_spelling` (aliases unpeeled).
pub(crate) const STR_LENGTH_FIELD_CLASS_ROOTS: &[&str] = &[
    "unicodeobject::Utf8Str",
    "unicodeobject::UnicodeValueStorage",
    "bytesobject::BytesBlock",
    "lowlevel_string::LowlevelStringHeader",
];

/// InstanceRepr `GcStruct._name` of STR oopspec receivers
/// (`jtransform.py Transformer._handle_stroruni_call`, `SoU.TO == rstr.STR`).
/// `rpy_string` is the synthetic rstr `STR` name (`rstr.py` `GcStruct`) and
/// is matched at the call site, not here.
pub(crate) const STR_OOPSPEC_CLASS_ROOTS: &[&str] =
    &["unicodeobject::Utf8Str", "bytesobject::BytesBlock"];

pub(crate) fn is_str_family_class_root(root: &str) -> bool {
    STR_FAMILY_CLASS_ROOTS.contains(&root) || is_rust_heap_string_root(root) || root == "str"
}

pub(crate) fn is_str_length_field_class_root(root: &str) -> bool {
    STR_LENGTH_FIELD_CLASS_ROOTS.contains(&root)
}

pub(crate) fn is_str_oopspec_class_root(root: &str) -> bool {
    STR_OOPSPEC_CLASS_ROOTS.contains(&root)
}

/// Crate-root `class_identity_spelling` of `rustpython_wtf8::Wtf8`, plus
/// the crate-prefixed FunDecl `name_path` segments
/// (`unspecialized_fun_segments`).
fn is_declared_wtf8_owner(owner: &str) -> bool {
    matches!(owner, "Wtf8" | "rustpython_wtf8::Wtf8")
}

/// Crate-root `class_identity_spelling` of `rustpython_wtf8::Wtf8Buf`, plus
/// the crate-prefixed FunDecl `name_path` segments.
fn is_declared_wtf8buf_owner(owner: &str) -> bool {
    matches!(owner, "Wtf8Buf" | "rustpython_wtf8::Wtf8Buf")
}

/// `Wtf8::new` / `Wtf8Buf::from_string` / `From<&str> for Wtf8Buf` — the
/// wrap of a string-family value into the WTF-8 buffer, an identity at
/// the annotation level (`rstr.py` has one string type).
pub(in crate::front::mir) fn wtf8_string_identity_segments(segments: &[String]) -> bool {
    if segments.len() < 2 {
        return false;
    }
    let last = segments[segments.len() - 1].as_str();
    let owner = |end: usize| segments[..end].join("::");
    if last == "from_string" && is_declared_wtf8buf_owner(&owner(segments.len() - 1)) {
        return true;
    }
    if last == "new" && is_declared_wtf8_owner(&owner(segments.len() - 1)) {
        return true;
    }
    if segments.len() >= 3
        && last == "from"
        && is_trait_impl_identity_segment(&segments[segments.len() - 2])
    {
        let impl_seg = segments[segments.len() - 2].as_str();
        return impl_seg.contains("From<")
            && impl_seg.contains("&str")
            && is_declared_wtf8buf_owner(&owner(segments.len() - 2));
    }
    false
}

/// Identity-view family: rstr `STR` wrappers or a rust-heap `String`.
///
/// `as_str` / `as_ref` / `as_bytes` / `to_string` / `deref` are identity
/// on either spelling (`rstr.py` has one string type). A rust-heap
/// `String` still paints `ValueType::Ref` so `ll_strconcat` does not
/// consume a `{cap, ptr, len}` header.
pub(in crate::front::mir) fn tyref_is_string_family(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_string_value(ty, llbc) || tyref_is_rust_heap_string(ty, llbc)
}

/// Whether a `TyRef` resolves (behind the usual wrappers) to `rstring.py`'s
/// `StringBuilder`, spelled `pyre_object::rstring::StringBuilder`: a
/// `SomeStringBuilder` value (`STRINGBUILDERPTR`), whatever word the Rust
/// struct holds. A borrow of it is the same builder.
pub(in crate::front::mir) fn tyref_is_string_builder(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path() == STRING_BUILDER_PATH)
}

/// The `name_path` of `rstring.py`'s `StringBuilder` in pyre-object.
pub(in crate::front::mir) const STRING_BUILDER_PATH: &str = "pyre_object::rstring::StringBuilder";

/// Whether a `TyRef` resolves (behind `Ref`/dedup/hash-cons wrappers) to
/// the `str` builtin — the unsized string slice (`{"Builtin": "Str"}`).
/// Distinct from `[u8]` (the `Slice` builtin) and from `[T]`/`Box`/named
/// ADTs, so it pins the string-family result of a `String -> &str` view
/// apart from `String`'s sibling `AsRef<[u8]>` / `AsRef<OsStr>` impls.
pub(in crate::front::mir) fn tyref_strips_to_str(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| n.get("Adt"))
        .and_then(type_decl_ref_builtin)
        == Some("Str")
}

/// `<String as AsRef<[u8]>>::as_ref` / `<str as AsRef<[u8]>>::as_ref`: std
/// defines it as `self.as_bytes()`, so on a string-family receiver it is the
/// same byte view as `as_bytes`.
pub(in crate::front::mir) fn is_as_ref_byte_view(
    leaf: Option<&str>,
    dest_ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    leaf == Some("as_ref")
        && (tyref_is_shared_u8_slice(dest_ty, llbc) || tyref_is_adt_shared_u8_slice(dest_ty, llbc))
}

/// Whether a `TyRef` resolves (behind `Ref` / dedup wrappers) to a
/// string-family value whose storage is rstr `STR`:
/// the `rustpython_wtf8` `Wtf8` / `Wtf8Buf` wrappers, pyre's `BytesBlock`
/// owner, `Utf8Str`, the `{Builtin: "Str"}` node. A shared `&[u8]` /
/// `*const [u8]` is a fat window, not this family. A Rust-heap
/// `alloc::string::String` is Vec-shaped (`{cap, ptr, len}`) and is not
/// this family.
pub(in crate::front::mir) fn tyref_is_string_value(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    node_is_string_value(node, llbc)
}

/// One-word STR view: `&[u8]` / `&str` / the string ADT itself.
/// `&&[u8]` / `&&str` is a reborrow of that word, not a second `Ptr(STR)`.
pub(in crate::front::mir) fn tyref_is_one_word_str_view(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_u8_window(ty, llbc) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc)) else {
        return false;
    };
    if let Some(arr) = node.get("Ref").and_then(serde_json::Value::as_array) {
        let Some(pointee) = arr.get(1).and_then(|p| strip_ty_indirections(p, llbc)) else {
            return false;
        };
        if pointee.get("Ref").is_some() {
            return false;
        }
        return node_is_string_value(pointee, llbc);
    }
    node_is_string_value(node, llbc)
}

/// A field type that is itself an inlined `lltype.Struct`, not a pointer
/// word and not a type the rtyper already represents as one value.
///
/// `heaptracker.py all_fielddescrs` recurses on `isinstance(FIELD,
/// lltype.Struct)` and emits one descr per leaf; a `Ptr`, `Box`, or
/// `Vec` stays a single stored word.
pub(in crate::front::mir) fn tyref_by_value_struct_decl<'l>(
    ty: &TyRef,
    llbc: &'l Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Option<&'l TypeDecl> {
    let node = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc))?;
    let obj = node.as_object()?;
    if obj.contains_key("Ref") || obj.contains_key("RawPtr") {
        return None;
    }
    if type_node_is_thin_box(node, llbc) {
        return None;
    }
    let id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(id)?;
    if type_decl_is_builtin_adt(td) || type_decl_is_closure_env(td) {
        return None;
    }
    if td.item_meta.name_path() == "alloc::boxed::Box" {
        return None;
    }
    if !matches!(td.kind, TypeDeclKind::Struct(_)) {
        return None;
    }
    if field_ty_is_inline_vec(ty, llbc) || tyref_is_string_value(ty, llbc) {
        return None;
    }
    // One integer word is not an inlined STRUCT. `getcalldescr` reads
    // that lltype off the value (`call.py`); leaf-copy would GETFIELD
    // `__pos_0` with the Int as the base (`VersionTag(u64)`).
    if let TypeDeclKind::Struct(fields) = &td.kind
        && integer_newtype_inner_field(td, fields, llbc, gc_struct_ids).is_some()
    {
        return None;
    }
    Some(td)
}

/// `Struct` declarations of one artefact by owner spelling, for [`type_decl_by_owner_root`].
pub(in crate::front::mir) struct StructDeclsByOwner {
    /// `(spelling, table position)` for each `Struct` decl's `name_path()` and its
    /// `strip_crate_prefix` form, sorted; the smallest position per spelling is the first
    /// decl in table order.
    pub(in crate::front::mir) exact: Vec<(String, usize)>,
    /// `(last path segment, table position)` for each `Struct` decl, sorted.
    pub(in crate::front::mir) leaf: Vec<(String, usize)>,
}

pub(in crate::front::mir) fn struct_decls_by_owner(llbc: &Llbc) -> &StructDeclsByOwner {
    llbc_derived(llbc).struct_decls_by_owner.get_or_init(|| {
        let mut exact = Vec::new();
        let mut leaf = Vec::new();
        for (i, td) in llbc.file.translated.type_decls.iter().enumerate() {
            let Some(td) = td else {
                continue;
            };
            if !matches!(td.kind, TypeDeclKind::Struct(_)) {
                continue;
            }
            let path = td.item_meta.name_path();
            exact.push((path.clone(), i));
            exact.push((strip_crate_prefix(&path), i));
            leaf.push((path.rsplit("::").next().unwrap_or(&path).to_string(), i));
        }
        exact.sort_unstable();
        leaf.sort_unstable();
        StructDeclsByOwner { exact, leaf }
    })
}

pub(in crate::front::mir) fn type_decl_by_owner_root<'l>(
    llbc: &'l Llbc,
    owner: &str,
) -> Option<&'l TypeDecl> {
    let index = struct_decls_by_owner(llbc);
    let start = index.exact.partition_point(|(k, _)| k.as_str() < owner);
    if let Some((key, pos)) = index.exact.get(start) {
        if key == owner {
            return llbc.type_by_id(*pos as u64);
        }
    }
    let owner_leaf = owner.rsplit("::").next().unwrap_or(owner);
    let start = index.leaf.partition_point(|(k, _)| k.as_str() < owner_leaf);
    let end = index
        .leaf
        .partition_point(|(k, _)| k.as_str() <= owner_leaf);
    match &index.leaf[start..end] {
        [(_, pos)] => llbc.type_by_id(*pos as u64),
        _ => None,
    }
}

pub(in crate::front::mir) fn struct_copy_owner_keys(
    td: &TypeDecl,
    tombstoned: &std::collections::HashSet<String>,
) -> (String, majit_ir::descr::StructId) {
    let name_path = td.item_meta.name_path();
    let owner_leaf = name_path.rsplit("::").next().unwrap_or("").to_string();
    // Same closure exception as [`Lowering::resolve_adt_field`]: a crate-root
    // or cross-crate-stripped closure would get the full path from
    // [`tombstoned_owner_spelling`], while field reads keep the crate-stripped
    // path.
    let owner_root = if majit_charon_reader::ullbc::is_closure_leaf(&owner_leaf) {
        strip_crate_prefix(&name_path)
    } else {
        class_identity_spelling(&name_path, tombstoned)
    };
    let owner_id =
        majit_ir::descr::StructId::from_canonical(&decl_path_for_tombstone(&name_path, tombstoned));
    (owner_root, owner_id)
}

pub(in crate::front::mir) fn collect_by_value_struct_leaves(
    td: &TypeDecl,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    depth: usize,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<(FieldDescriptor, ValueType)> {
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return Vec::new();
    };
    if depth == 0 {
        return Vec::new();
    }
    let (owner_root, owner_id) = struct_copy_owner_keys(td, tombstoned);
    let mut leaves = Vec::new();
    for (i, f) in fields.iter().enumerate() {
        let name = f
            .name
            .clone()
            .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(i));
        if tyref_is_void_zst(&f.ty, llbc)
            && !tyref_keeps_empty_class_instance(&f.ty, llbc, tombstoned)
        {
            continue;
        }
        if tyref_pair_field_kind(&f.ty, llbc).is_some() {
            let len = pair_len_field_name(&name);
            for fname in [name, len] {
                leaves.push((
                    FieldDescriptor::new(fname, Some(owner_root.clone()))
                        .with_owner_id(Some(owner_id))
                        .with_base_is_deref(false),
                    ValueType::Unsigned,
                ));
            }
            continue;
        }
        if let Some(nested) = tyref_by_value_struct_decl(&f.ty, llbc, gc_struct_ids) {
            leaves.extend(collect_by_value_struct_leaves(
                nested,
                llbc,
                tombstoned,
                gc_struct_ids,
                depth - 1,
                raw_alloc_flavor_structs,
            ));
            continue;
        }
        leaves.push((
            FieldDescriptor::new(name, Some(owner_root.clone()))
                .with_owner_id(Some(owner_id))
                .with_base_is_deref(false)
                .with_inline_vec(field_ty_is_inline_vec(&f.ty, llbc)),
            tyref_to_attr_value_type_for_struct_field(
                &f.ty,
                td,
                llbc,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            ),
        ));
    }
    leaves
}

/// Leaves of `owner.field` when that field is an inlined by-value struct.
///
/// `None` keeps the existing one-word copy.  `Some` is declaration order
/// through the inner struct's own descriptors (`owner_root` = the inner
/// struct), matching a nested `FieldRead` of `frame.ob_header.w_class`
/// whose base is the outer object and whose descriptor belongs to
/// `PyObject`.  `typeptr`-like header words are ordinary leaves: this is
/// a copy of an existing value, not an allocation.
pub(in crate::front::mir) fn inline_struct_field_leaves(
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    owner: &str,
    field_name: &str,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<Vec<(FieldDescriptor, ValueType)>> {
    let td = type_decl_by_owner_root(llbc, owner)?;
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return None;
    };
    let field = fields
        .iter()
        .find(|f| f.name.as_deref() == Some(field_name))?;
    let nested = tyref_by_value_struct_decl(&field.ty, llbc, gc_struct_ids)?;
    Some(collect_by_value_struct_leaves(
        nested,
        llbc,
        tombstoned,
        gc_struct_ids,
        16,
        raw_alloc_flavor_structs,
    ))
}

/// Whether the pointee of exactly one raw-pointer layer is a string-family
/// value.  This is deliberately narrower than teaching
/// [`strip_ty_wrappers`] to erase `RawPtr`: ordinary typed raw pointers retain
/// their pointee identity; only field-annotation projection asks whether the
/// pointer is the storage spelling of RPython's string value.
pub(in crate::front::mir) fn tyref_raw_ptr_pointee_is_string_value(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    let Some(pointee) = tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| n.as_object()?.get("RawPtr")?.as_array()?.first())
        .and_then(|n| strip_ty_wrappers(n, llbc))
    else {
        return false;
    };
    node_is_string_value(pointee, llbc)
}

pub(in crate::front::mir) fn node_is_string_value(node: &serde_json::Value, llbc: &Llbc) -> bool {
    // `builtin: "Str"` — the bare `str` value.
    if node.get("Adt").and_then(type_decl_ref_builtin) == Some("Str") {
        return true;
    }
    // Named string ADTs whose layout is rstr `STR` (`[hash, length, chars]`):
    // the WTF-8 wrappers, `Utf8Str` / `UnicodeValueStorage`,
    // `BytesBlock` (`type BytesBlock = Utf8Str`, `W_BytesObject._value`),
    // and `LowlevelStringHeader` (the `{hash, len}` view
    // `bh_lowlevel_string_len` loads; `rstr.py` `STR`). Dest overlay must
    // not recast that header as a raw `SomePtr` — `ann_cast_int_to_ptr`
    // of `Ptr(STR)` is the string itself, and `ll_strlen` is `len(s.chars)`.
    // `alloc::string::String` is a Rust-heap `{cap, ptr, len}` header and
    // is not this family (`tyref_is_rust_heap_string`).
    adt_node_def_id(node)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            matches!(
                td.item_meta.name_path().rsplit("::").next(),
                Some(
                    "Wtf8"
                        | "Wtf8Buf"
                        | "Utf8Str"
                        | "UnicodeValueStorage"
                        | "BytesBlock"
                        | "LowlevelStringHeader"
                )
            )
        })
}

/// Whether `reg` resolves to a `Deref::deref` / `DerefMut::deref_mut`
/// leaf, by the callee's `name_path` suffix.  Unlike `deref_cast_root`
/// this makes no claim about the dereferenced type — the caller pairs it
/// with a `tyref_is_string_value` receiver / dest gate.
pub(in crate::front::mir) fn is_deref_call(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        let np = fd.item_meta.name_path();
        np.ends_with("::deref") || np.ends_with("::deref_mut")
    })
}

/// The qualified declaration path of a `TyRef`'s base ADT, after
/// stripping `Ref` / hash-cons wrappers.  `None` for non-ADT types.
pub(in crate::front::mir) fn adt_path_of_tyref(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let node = tyref_node(ty, llbc)?;
    let node = strip_ty_wrappers(node, llbc)?;
    let id = adt_node_def_id(node)?;
    Some(llbc.type_by_id(id)?.item_meta.name_path())
}

/// Whether `td` declares a field named `name` — a struct field, or any
/// enum variant's field (Model A flattens variant fields into the same
/// class attr namespace).
pub(in crate::front::mir) fn adt_declares_field(td: &TypeDecl, name: &str) -> bool {
    match &td.kind {
        TypeDeclKind::Struct(fields) => fields.iter().any(|f| f.name.as_deref() == Some(name)),
        TypeDeclKind::Enum(variants) => variants
            .iter()
            .any(|v| v.fields.iter().any(|f| f.name.as_deref() == Some(name))),
        _ => false,
    }
}

/// The struct-root `class_root` for a `Ref`-typed input param: the bare
/// ADT leaf (`tyref_class_root`), except for a closure env, whose shared
/// bare leaf `closure` `harden_duplicate_leaf_metadata` drops from the
/// field registry — carry its full `::` `name_path` (a surviving
/// registry key) so `derive_subject_inputcells` projects the captured
/// fields rather than seeding a classdef-less `SomeInstance(None)` shell
/// its `self.<capture>` reads would wall on.  RPython forbids closures.
pub(in crate::front::mir) fn tyref_input_class_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    // Matching [`tyref_to_value_type`]: `&Option<P>` is the Option word.
    if let Some(option_ty) = tyref_shared_borrow_option_pointee(ty, llbc)
        && option_nonzero_word_kind(&option_ty, llbc).is_none()
    {
        return tyref_input_class_root(&option_ty, llbc, tombstoned)
            .or_else(|| tyref_split_core_enum_param_root(&option_ty, llbc, tombstoned));
    }
    if let Some(ValueType::Ref(root)) = tyref_niche_option_payload_paint(ty, llbc, tombstoned) {
        return root;
    }
    let leaf = tyref_class_root_with(ty, llbc, tombstoned);
    if leaf
        .as_deref()
        .is_some_and(majit_charon_reader::ullbc::is_closure_leaf)
    {
        adt_path_of_tyref(ty, llbc).or(leaf)
    } else {
        leaf
    }
}

/// Instantiation class of a split-eligible core enum used as a `Ref`
/// parameter (`bookkeeper.py` `getuniqueclassdef(cls)`).
///
/// [`adt_node_class_root_identity`] paints split-eligible `Option` and
/// `Result` as values so a call result recasts to that class. Binary
/// `ControlFlow` stays classdef-less as a *value*: painting it recasts
/// `Try::branch` onto the exception link (`simplify.py` "raising
/// operation result cannot flow into exception link"). A parameter has
/// no such recast — the Input `class_root` is the annotator seed
/// (`derive_subject_inputcells` → `getuniqueclassdef_for_struct_root`)
/// — and a constructor of the same instantiation already interned that
/// classdef with `__discriminant` (`rclass.py` `getclsfield`). Without
/// this seed the formal is `SomeInstance { classdef: None }` and
/// `getattr __discriminant` walls.
///
/// A pointer-niche Option ([`tyref_is_niche_option`]) is the null word
/// on both the value and the parameter path. Extra Instantiated args
/// beside Option's payload are not the niche test. Vec/Box stay on
/// their annotator models (`adt_head_instantiation_suffix` is enum-only).
pub(in crate::front::mir) fn tyref_split_core_enum_param_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc))?;
    let adt = node.as_object()?.get("Adt")?.as_object()?;
    let def_id = adt_node_def_id(node)?;
    let name = llbc.type_by_id(def_id)?.item_meta.name_path();
    if !matches!(
        name.split("::").next().unwrap_or(&name),
        "core" | "std" | "alloc"
    ) {
        return None;
    }
    let types = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())?;
    // A modeled pointer-niche Option is the null word, not an enum class.
    // A tagged Option whose payload is a named transparent handle still
    // intern's (`getattr __discriminant`). Bare pointer/ref *payloads*
    // the niche recognizer leaves as aggregates stay classdef-less.
    // Extra Instantiated args are not the payload (`tyref_is_niche_option`
    // reads `types[0]`). `ty` here is already wrapper-stripped
    // (`strip_ty_wrappers` above).
    if tyref_is_niche_option(&TyRef::Other(node.clone()), llbc)
        || (name.ends_with("::option::Option") && option_bare_pointer_payload(types, llbc))
    {
        return None;
    }
    let suffix = adt_head_instantiation_suffix(adt, llbc)?;
    let identity = class_identity_spelling(&name, tombstoned);
    Some(majit_ir::descr::with_instantiation_suffix(
        &identity, &suffix,
    ))
}

/// `OpKind::Input.class_root` for a `Ref`-typed parameter.
///
/// A list-typed param (`Vec<T>`, `&[T]`, Charon builtin `Slice<T>`) is
/// the annotator's list model. Prefer the monomorphic spelling over an
/// ADT leaf (`Slice`) so `derive_subject_inputcells` seeds `SomeList`
/// instead of a classdef-less `SomeInstance` whose `len()` walls at
/// `getattr("__len__")`.
pub(in crate::front::mir) fn tyref_ref_param_class_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    generics: Option<&serde_json::Value>,
) -> Option<String> {
    let spelling = tyref_to_ast_string(ty, llbc);
    if majit_ir::descr::is_list_container_spelling(&spelling) {
        return Some(spelling);
    }
    // A Charon builtin Tuple TypeDecl leaf is `"Tuple"`; that token is
    // not a shaped name (`is_shaped_tuple_name` excludes it) and would
    // intern a classdef-less `SomeInstance`. Prefer the per-shape
    // `Tuple<A,B>` root so `derive_subject_inputcells` seeds `SomeTuple`
    // (`rtuple.py` `TupleRepr`, `binaryop.py` `getitem((tup1, int2))`).
    // A closure `call_once` `tupled_args` is this param (`FnOnce::Args`).
    if let Some(root) = tyref_shaped_tuple_root(ty, llbc) {
        return Some(root);
    }
    tyref_input_class_root(ty, llbc, tombstoned)
        .or_else(|| tyref_strips_to_str(ty, llbc).then(|| "str".to_string()))
        .or_else(|| tyref_generic_trait_bound_root(ty, llbc, generics))
        .or_else(|| tyref_split_core_enum_param_root(ty, llbc, tombstoned))
}

/// The inner [`ValueType`] of a `core::sync::atomic::Atomic<T>`
/// (`Atomic<i64>` → `Int`, `Atomic<usize>` → `Unsigned`, `Atomic<bool>` →
/// `Bool`, `Atomic<*mut T>` → `Int` when `T` is storage-raw,
/// `Ref(Some(root))` when `T` is flavor-raw, else `Ref(None)`), or `None`
/// when `ty` is not a std atomic.  The atomic wrappers are
/// layout-transparent over their inner scalar/pointer (asserted at
/// pyobject.rs for the `PyType` vtable `subclassrange_*` / `instantiate`
/// fields), and the upstream typeptr ranges are plain int fields, so a
/// field of one types as that inner value and its `load`/`store` fold to
/// it (`is_atomic_load`).
pub(in crate::front::mir) fn tyref_atomic_inner_value_type(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<ValueType> {
    let arg = tyref_atomic_arg(ty, llbc)?.as_object()?;
    if arg.contains_key("RawPtr") {
        // `Atomic<*mut T>` is layout-identical to `*mut T`. Storage-raw
        // is an int word (`ty_is_raw_address`). Flavor-raw stays
        // `Ref(Some(root))` (`rclass.py getgcflavor`); a GC pointee
        // stays `Ref(None)`.
        if ty_is_raw_address(ty, llbc, gc_struct_ids, raw_alloc_flavor_structs) {
            return Some(ValueType::Int);
        }
        if let Some(root) = flavor_raw_pointer_class_root(
            ty,
            llbc,
            no_tombstoned_leaves(),
            raw_alloc_flavor_structs,
        ) {
            return Some(ValueType::Ref(Some(root)));
        }
        return Some(ValueType::Ref(None));
    }
    match arg.get("Scalar")? {
        serde_json::Value::String(s) if s == "Bool" => Some(ValueType::Bool),
        serde_json::Value::Object(scalar) => {
            let int = scalar.get("Integer")?.as_object()?;
            if int.contains_key("Signed") {
                Some(ValueType::Int)
            } else if int.contains_key("Unsigned") {
                Some(ValueType::Unsigned)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The type argument `T` of a `core::sync::atomic::Atomic<T>`, resolved
/// through the dedup table, or `None` when `ty` is not a std atomic.  The
/// `AtomicU32` / `AtomicPtr<T>` / … names are type aliases of `Atomic<T>`,
/// so the argument is what tells them apart.
pub(in crate::front::mir) fn tyref_atomic_arg<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let id = adt_node_def_id(node)?;
    let meta = &llbc.type_by_id(id)?.item_meta;
    // Cheap leaf check first — no path-string allocation (unlike
    // `name_path` / `adt_path_of_tyref`).  This runs on the hot
    // `tyref_to_value_type` fallback path, so it must stay
    // allocation-free: bail before the module scan unless the type's
    // last segment is `Atomic`.
    match meta.template_name().next_back()? {
        NameSeg::Ident { ident: (s, _) } if s == "Atomic" => {}
        _ => return None,
    }
    // Confirm std's `core::sync::atomic` module so a user type
    // coincidentally named `Atomic` does not match.
    let in_atomic_mod = meta
        .template_name()
        .any(|s| matches!(s, NameSeg::Ident { ident: (id, _) } if id == "atomic"));
    if !in_atomic_mod {
        return None;
    }
    let arg = type_decl_ref_generics(node.get("Adt")?.as_object()?, llbc)?
        .get("types")?
        .as_array()?
        .first()?;
    if let Some(dedup) = arg.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_body(dedup);
    }
    if let Some(body) = arg
        .get("Value")
        .and_then(serde_json::Value::as_array)
        .and_then(|pair| pair.get(1))
    {
        return Some(body);
    }
    Some(arg)
}

/// The inner scalar of a `core::sync::atomic` integer/bool wrapper, spelled
/// as the Rust primitive the wrapper is layout-transparent over
/// (`Atomic<usize>` → `usize`), or `None` for anything else.
///
/// `Atomic<*mut T>` is deliberately absent: its inner value is a pointer,
/// which is already what an unrecognised name answers.
pub(in crate::front::mir) fn tyref_atomic_inner_scalar_str(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let scalar = tyref_atomic_arg(ty, llbc)?.get("Scalar")?;
    Some(charon_literal_to_ast_string(scalar))
}

/// Whether `ty` is `core::sync::atomic::Atomic<*mut T>` (`AtomicPtr<T>`).
pub(in crate::front::mir) fn tyref_atomic_is_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_atomic_arg(ty, llbc)
        .and_then(serde_json::Value::as_object)
        .is_some_and(|arg| arg.contains_key("RawPtr"))
}

/// Index and name of the one sized field of a `#[repr(transparent)]` struct.
///
/// Zero-sized fields occupy no bytes and are ignored, so a wrapper whose
/// only sized field is a scalar (plus a marker) is that scalar.
/// [`strip_ty_wrappers`] peels `&T` / `&mut T` and leaves `RawPtr` alone, so
/// a borrow of the wrapper is the same word a fieldless-enum borrow is
/// ([`tyref_is_borrowed_fieldless_enum`]). A raw pointer remains an address.
pub(in crate::front::mir) fn transparent_nonzst_field(
    decl: &TypeDecl,
    llbc: &Llbc,
) -> Option<(usize, String)> {
    if !decl.is_repr_transparent() || decl.published_struct_copy {
        return None;
    }
    let TypeDeclKind::Struct(fields) = &decl.kind else {
        return None;
    };
    let mut found = None;
    for (index, field) in fields.iter().enumerate() {
        if field_is_zst(&field.ty, llbc) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let name = field
            .name
            .clone()
            .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(index));
        found = Some((index, name));
    }
    found
}

pub(in crate::front::mir) fn tyref_transparent_nonzst_field(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(usize, String)> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let decl = llbc.type_by_id(adt_node_def_id(node)?)?;
    transparent_nonzst_field(decl, llbc)
}

/// Inner struct of a by-value `#[repr(transparent)]` newtype.
///
/// `celldict.py` `erase({})` stores the dicttable itself; `ModuleDictStorage`
/// is a no-op alias of that `RDict` (`heaptracker.py all_fielddescrs` /
/// `inline_struct_field_leaves`). A pointer-shaped payload is a GC handle
/// (`PyError.0` is `*mut PyErrorObject`) and keeps the wrapper's own field.
pub(in crate::front::mir) fn transparent_by_value_inner_fields<'a>(
    td: &'a TypeDecl,
    llbc: &'a Llbc,
) -> Option<(&'a TypeDecl, &'a [FieldDecl])> {
    let (index, _) = transparent_nonzst_field(td, llbc)?;
    let TypeDeclKind::Struct(fields) = &td.kind else {
        return None;
    };
    let field = fields.get(index)?;
    let field_node = strip_ty_wrappers(tyref_node(&field.ty, llbc)?, llbc)?;
    if field_node
        .as_object()
        .is_some_and(|obj| obj.contains_key("RawPtr"))
        || type_node_is_mut_ref(field_node, llbc)
        || type_node_shared_ref_pointee(field_node, llbc).is_some()
        || type_node_is_fn_ptr(field_node, llbc)
        || type_node_is_thin_box(field_node, llbc)
    {
        return None;
    }
    let inner_id = adt_node_def_id(field_node)?;
    if inner_id == td.def_id {
        return None;
    }
    let inner = llbc.type_by_id(inner_id)?;
    let owner = type_decl_struct_owner(inner, llbc)?;
    let inner_fields = match &owner.kind {
        TypeDeclKind::Struct(fields) => fields.as_slice(),
        _ => return None,
    };
    (!inner_fields.is_empty()).then_some((owner, inner_fields))
}

pub(in crate::front::mir) fn field_is_zst(ty: &TyRef, llbc: &Llbc) -> bool {
    if is_unit_type(ty, llbc) || tyref_is_zero_sized(ty, llbc) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if json_ty_byte_size(node, llbc) == Some(0) {
        return true;
    }
    let Some(adt) = node
        .as_object()
        .and_then(|obj| obj.get("Adt"))
        .and_then(|adt| adt.as_object())
    else {
        return false;
    };
    let is_tuple = adt.get("builtin").and_then(|id| id.as_str()) == Some("Tuple");
    let empty = type_decl_ref_generics(adt, llbc)
        .and_then(|generics| generics.get("types"))
        .and_then(|types| types.as_array())
        .is_some_and(|types| types.is_empty());
    is_tuple && empty
}

/// `&T` / `&mut T` where `T` is a `repr(transparent)` wrapper of a scalar.
/// A residual callee expects a real address for that borrow; the word is not
/// one. A raw pointer is left alone.
pub(in crate::front::mir) fn tyref_is_borrowed_transparent_scalar(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    if !output_type_is_ref(ty, llbc) {
        return false;
    }
    matches!(
        tyref_transparent_inner_value_type(
            ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs
        ),
        Some(
            ValueType::Int
                | ValueType::Unsigned
                | ValueType::Bool
                | ValueType::Float
                | ValueType::SingleFloat
                | ValueType::Int128
                | ValueType::UInt128
        )
    )
}

/// ADT id of a type expression: inline `Value`, `Deduplicated`, or a bare `Adt`.
pub(in crate::front::mir) fn tyexpr_to_adt_def_id(
    llbc: &Llbc,
    ty: &serde_json::Value,
) -> Option<u64> {
    resolve_tyexpr_to_adt_def_id_free(llbc, ty).or_else(|| adt_node_def_id(ty))
}

pub(in crate::front::mir) fn deref_trait_decl_id(llbc: &Llbc) -> Option<u64> {
    let mut found = None;
    for decl in llbc.iter_trait_decls() {
        if decl.item_meta.name_path() != "core::ops::deref::Deref" {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(decl.def_id);
    }
    found
}

pub(in crate::front::mir) enum DerefTargetBind<'a> {
    Absent,
    Value(&'a serde_json::Value),
    Ambiguous,
}

/// The one `TraitType` of `trait_id` on this impl row.
///
/// Two bindings for the same trait are not a guess: the caller peels.
pub(in crate::front::mir) fn deref_impl_target_bind(
    row: &serde_json::Value,
    trait_id: u64,
) -> DerefTargetBind<'_> {
    let Some(entries) = row.get("types").and_then(|types| types.as_array()) else {
        return DerefTargetBind::Absent;
    };
    let mut count = 0usize;
    let mut value = None;
    for entry in entries {
        let Some(kind) = entry
            .get("kind")
            .and_then(|kind| kind.get("TraitType"))
            .and_then(|kind| kind.as_array())
        else {
            continue;
        };
        if kind.first().and_then(|id| id.as_u64()) != Some(trait_id) {
            continue;
        }
        count += 1;
        if count > 1 {
            return DerefTargetBind::Ambiguous;
        }
        value = entry
            .get("skip_binder")
            .and_then(|binder| binder.get("value"));
    }
    match value {
        Some(value) if count == 1 => DerefTargetBind::Value(value),
        _ => DerefTargetBind::Absent,
    }
}

/// Class root of a type expression that does not re-enter
/// [`transparent_deref_target_class`]. A `Deref::Target` is already a
/// struct; its class is the nominal leaf, tombstone spelling included.
pub(in crate::front::mir) fn nominal_adt_class_root(
    ty: &serde_json::Value,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let tyref = TyRef::Other(ty.clone());
    let node = strip_ty_wrappers(tyref_node(&tyref, llbc)?, llbc)?.clone();
    if let Some(root) = adt_node_class_root_identity(&node, llbc, tombstoned) {
        return Some(root);
    }
    let raw_pointee = node
        .as_object()?
        .get("RawPtr")?
        .as_array()?
        .first()?
        .clone();
    let pointee = strip_ty_wrappers(&raw_pointee, llbc)?.clone();
    adt_node_class_root_identity(&pointee, llbc, tombstoned)
}

/// Whether `decl` is a GC handle whose `Deref::Target` class applies.
///
/// A `Struct` must have one sized field that is already a `Ref`. The
/// field being this wrapper would re-enter the rule; `*mut T` is not an
/// ADT, so a pointer field does not trip that check. An `Opaque`
/// dependency view has no field list: one pointer word that is not a
/// linked scalar is the same handle.
pub(in crate::front::mir) fn transparent_handle_has_deref_class(
    decl: &TypeDecl,
    wrapper_id: u64,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    match &decl.kind {
        TypeDeclKind::Struct(fields) => {
            let Some((index, _)) = transparent_nonzst_field(decl, llbc) else {
                return false;
            };
            let Some(field) = fields.get(index) else {
                return false;
            };
            if let Some(field_node) =
                tyref_node(&field.ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc))
                && adt_node_def_id(field_node) == Some(wrapper_id)
            {
                return false;
            }
            matches!(
                tyref_to_value_type_with(
                    &field.ty,
                    llbc,
                    tombstoned,
                    gc_struct_ids,
                    raw_alloc_flavor_structs
                ),
                ValueType::Ref(_)
            )
        }
        TypeDeclKind::Opaque => opaque_pointer_handle(decl, llbc),
        _ => false,
    }
}

/// `Opaque` `repr(transparent)` declaration whose layout is one pointer
/// and which [`discover_transparent_scalar_kinds`] did not link as a
/// scalar. The `Deref` impl, not the missing field type, names the class.
pub(in crate::front::mir) fn opaque_pointer_handle(decl: &TypeDecl, llbc: &Llbc) -> bool {
    if !decl.is_repr_transparent() {
        return false;
    }
    if llbc
        .transparent_scalar_kind(&decl.item_meta.name_path())
        .is_some()
    {
        return false;
    }
    let Some(width) = llbc.target_pointer_size() else {
        return false;
    };
    decl.layout_for_target(llbc, "")
        .and_then(|layout| layout.size)
        == Some(u64::from(width))
}

/// Class of a `repr(transparent)` GC handle's `Deref::Target`.
///
/// The defining crate records one sized field, and that field must
/// already be a `Ref` (including `Ref(None)`). A dependency artefact
/// keeps the `repr(transparent)` layout and the `Deref` impl but emits
/// the declaration as `Opaque`, with no field type
/// (`link_transparent_scalar_types` is the same opacity for scalars).
/// That view is this handle when its layout is one pointer word and it
/// is not a scalar whose bank was linked from the defining crate.
/// `Target` must be one struct, or the same struct seen as `Opaque` in
/// a dependency artefact. Its class is [`nominal_adt_class_root`], so a
/// tombstoned leaf stays disambiguated and the target lookup does not
/// re-enter this rule. No impl, an enum or alias `Target`, or two
/// targets with different classes keep the transparent peel. A target
/// with no class root does not apply. Two impls that name the same
/// class do.
pub(in crate::front::mir) fn transparent_deref_target_class(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let wrapper_id = adt_node_def_id(node)?;
    let decl = llbc.type_by_id(wrapper_id)?;
    if !transparent_handle_has_deref_class(
        decl,
        wrapper_id,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return None;
    }
    let trait_id = deref_trait_decl_id(llbc)?;
    let mut class = None;
    for row in llbc.trait_impls_of(trait_id) {
        let Some(self_ty) = row.pointer("/impl_trait/generics/types/0") else {
            continue;
        };
        let Some(self_id) = tyexpr_to_adt_def_id(llbc, self_ty) else {
            continue;
        };
        if !same_nominal_adt(llbc, self_id, wrapper_id) {
            continue;
        }
        let target_ty = match deref_impl_target_bind(row, trait_id) {
            DerefTargetBind::Ambiguous => return None,
            DerefTargetBind::Absent => continue,
            DerefTargetBind::Value(target_ty) => target_ty,
        };
        let Some(target_id) = tyexpr_to_adt_def_id(llbc, target_ty) else {
            continue;
        };
        let Some(target_decl) = llbc.type_by_id(target_id) else {
            continue;
        };
        // A dependency may see the target struct as `Opaque`. An enum,
        // union, or alias is not that class.
        if !matches!(
            target_decl.kind,
            TypeDeclKind::Struct(_) | TypeDeclKind::Opaque
        ) {
            continue;
        }
        // A target whose class root is `None` (core/std/alloc with type
        // arguments, via the nominal leaf) does not apply. The nominal
        // root keeps a handle `Target` from re-entering this function
        // through [`adt_node_class_root_with`].
        let Some(root) = nominal_adt_class_root(target_ty, llbc, tombstoned) else {
            return None;
        };
        match &class {
            None => class = Some(root),
            Some(previous) if previous != &root => return None,
            Some(_) => {}
        }
    }
    class
}

/// `TyRef` of a named ADT declaration, for Deref-target / handle-field
/// lookups that already have the `TypeDecl` in hand.
pub(in crate::front::mir) fn type_decl_adt_tyref(def_id: u64) -> TyRef {
    TyRef::Other(serde_json::json!({"Adt": {"id": def_id, "generics": {"types": []}}}))
}

/// Field-row spelling of a transparent GC handle, or of a raw pointer to one.
///
/// `project_struct_field_type` turns the row into the payload class, and
/// `field_metadata` sizes it. The cast writes the `Deref::Target` class
/// (`OperationError` carries `w_type` and `_w_value`). A bare class name is
/// a by-value nested struct of that object's full size; `*mut {class}`
/// projects to the same class and stays one pointer word.
pub(in crate::front::mir) fn transparent_handle_field_spelling(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    if let Some(root) = transparent_deref_target_class(
        ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return Some(format!("*mut {root}"));
    }
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let pointee = node.as_object()?.get("RawPtr")?.as_array()?.first()?;
    let pointee_ty = TyRef::Other(pointee.clone());
    let root = transparent_deref_target_class(
        &pointee_ty,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    )?;
    Some(format!("*mut {root}"))
}

pub(in crate::front::mir) fn tyref_transparent_inner_value_type(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<ValueType> {
    // Peel the reference the way the atomic sibling does. A borrow of the
    // wrapper has to answer with the same bank as the wrapper itself:
    // `Rvalue::Ref` aliases the referent's Variable rather than introducing a
    // new one, so a wrapper that types as its inner scalar by value and as a
    // GC reference through a borrow puts one machine word in two register
    // banks — the caller passes it in the int list while the callee's
    // parameter is declared `r`. `strip_ty_wrappers` leaves `RawPtr` alone,
    // so a raw pointer to the wrapper keeps pointer semantics.
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let id = adt_node_def_id(node)?;
    let decl = llbc.type_by_id(id)?;
    // A foreign `Opaque` decl keeps `repr.transparent: false`: the defining
    // crate is outside the extraction set, so the layout does not record the
    // wrapper. The linked kind is that missing field.
    // `apply_published_structs` turns that import into a `Struct` and does
    // not copy `repr`, so the same false flag remains. Reading the copied
    // field would retype the word; the kind stays the scalar table.
    if (matches!(&decl.kind, TypeDeclKind::Opaque) || decl.published_struct_copy)
        && let Some(kind) = llbc.transparent_scalar_kind(&decl.item_meta.name_path())
    {
        return Some(transparent_scalar_value_type(kind));
    }
    match &decl.kind {
        TypeDeclKind::Struct(fields) if !decl.published_struct_copy => {
            if decl.is_repr_transparent() {
                let field = if let Some((index, _)) = transparent_nonzst_field(decl, llbc) {
                    fields.get(index)?
                } else if let [only] = fields.as_slice() {
                    only
                } else {
                    return None;
                };
                Some(tyref_to_value_type_with(
                    &field.ty,
                    llbc,
                    tombstoned,
                    gc_struct_ids,
                    raw_alloc_flavor_structs,
                ))
            } else if output_type_is_ref(ty, llbc) {
                // `&Plain` / `&VersionTag` is an address. Peeling the
                // referent into Int here makes
                // `refuse_borrowed_transparent_scalar_residual` treat a
                // non-transparent borrow as a scalar word.
                None
            } else {
                // A by-value Copy newtype around one primitive integer is
                // that integer. RPython's rtyper gives one lltype per
                // value (`history.getkind`) and `getcalldescr` reads
                // `[v.concretetype for v in op.args[1:]]` (`call.py`),
                // so the helper Input / FUNC.ARGS / calldescr kind cannot
                // disagree with the live word. `VersionTag(u64)` is the
                // case: it is not `repr(transparent)`, construction and
                // SROA alias the inner serial as Int, and a Ref paint
                // here aborted CALL_PURE of
                // `_getdictvalue_no_unwrapping_pure` with
                // `GotoIfNotValueNotConcrete` on that Int.
                integer_newtype_inner_value_type(decl, fields, llbc, gc_struct_ids)
            }
        }
        _ => None,
    }
}

/// Unique positional field of a non-GC tuple-struct newtype that is a
/// one-word integer scalar (`iN` / `uN` / `isize` / `usize`, not
/// `i128`/`u128`). Named one-field wrappers (`RangeFrom.start`, a
/// rust-heap `String`'s header word) keep their own type.
/// Multi-field structs, ZST markers beside a payload, GC instances,
/// pointers, and floats stay `None` so they keep their own bank.
fn integer_newtype_inner_field<'a>(
    decl: &TypeDecl,
    fields: &'a [FieldDecl],
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Option<&'a TyRef> {
    if gc_struct_ids.contains(&struct_id_for_type_decl(decl)) {
        return None;
    }
    integer_newtype_inner_field_structural(decl, fields, llbc)
}

/// Shape of a by-value one-integer tuple struct, ignoring GC taint.
///
/// `history.getkind` / `descr.py get_field_descr` of that FIELD type is
/// the inner integer. Field-0 GC closure must not follow this shape as
/// an inlined parent (`struct_field0_type_id`).
pub(in crate::front::mir) fn type_decl_is_by_value_integer_newtype(
    decl: &TypeDecl,
    llbc: &Llbc,
) -> bool {
    let TypeDeclKind::Struct(fields) = &decl.kind else {
        return false;
    };
    integer_newtype_inner_field_structural(decl, fields, llbc).is_some()
}

fn integer_newtype_inner_field_structural<'a>(
    decl: &TypeDecl,
    fields: &'a [FieldDecl],
    llbc: &Llbc,
) -> Option<&'a TyRef> {
    let path = decl.item_meta.name_path();
    if path == STRING_BUILDER_PATH || path == "alloc::string::String" {
        return None;
    }
    let carrier = llbc.exception_carrier();
    if !carrier.is_empty() && path == carrier {
        return None;
    }
    let [field] = fields else {
        return None;
    };
    // Charon drops a positional `_0` to `name: None`
    // (`ullbc.rs FieldDecl`). A named payload is a real field, not a
    // newtype wrap of the integer (`RangeFrom.start`).
    if field.name.is_some() {
        return None;
    }
    if field_is_zst(&field.ty, llbc) {
        return None;
    }
    one_word_integer_scalar_value_type(&field.ty, llbc)?;
    Some(&field.ty)
}

fn integer_newtype_inner_value_type(
    decl: &TypeDecl,
    fields: &[FieldDecl],
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Option<ValueType> {
    one_word_integer_scalar_value_type(
        integer_newtype_inner_field(decl, fields, llbc, gc_struct_ids)?,
        llbc,
    )
}

/// Field type is a Charon `Scalar` integer that occupies one JIT int
/// word. Pointers, fieldless enums, and other shapes that
/// [`tyref_to_value_type_with`] paints `Int` are not this: peeling
/// those would collapse an address or a tag wrapper into the payload.
fn one_word_integer_scalar_value_type(ty: &TyRef, llbc: &Llbc) -> Option<ValueType> {
    let resolved = tyref_node(ty, llbc)?;
    let value = strip_ty_indirections(resolved, llbc).unwrap_or(resolved);
    let scalar = value.as_object()?.get("Scalar")?;
    let int = scalar.as_object()?.get("Integer")?.as_object()?;
    if int.get("Signed").and_then(serde_json::Value::as_str) == Some("I128")
        || int.get("Unsigned").and_then(serde_json::Value::as_str) == Some("U128")
    {
        return None;
    }
    if int.contains_key("Unsigned") {
        Some(ValueType::Unsigned)
    } else if int.contains_key("Signed") {
        Some(ValueType::Int)
    } else {
        None
    }
}

pub(in crate::front::mir) fn transparent_scalar_value_type(
    kind: majit_charon_reader::TransparentScalarKind,
) -> ValueType {
    match kind {
        majit_charon_reader::TransparentScalarKind::Signed => ValueType::Int,
        majit_charon_reader::TransparentScalarKind::Unsigned => ValueType::Unsigned,
        majit_charon_reader::TransparentScalarKind::Bool => ValueType::Bool,
        majit_charon_reader::TransparentScalarKind::Float => ValueType::Float,
    }
}

/// `core::cell::UnsafeCell<T>`, the `#[repr(transparent)]` cell word.
pub(in crate::front::mir) fn tyref_is_unsafecell(ty: &TyRef, llbc: &Llbc) -> bool {
    adt_path_of_tyref(ty, llbc)
        .is_some_and(|p| p == "core::cell::UnsafeCell" || p.ends_with("::UnsafeCell"))
}

/// `Arg<T>` from `rustpython_compiler_core::bytecode::instruction` —
/// the zero-sized oparg marker (`pub struct Arg<T: OpArgType>
/// (PhantomData<T>)`, instruction.rs).  The external decl is
/// Opaque in the LLBC, so a payload row spelled through it would
/// project to an attr the annotator cannot type; the lifted model
/// carries the marker as a plain integer instead.  Its consumer
/// `Arg::get` keeps its ordinary (residual) call lowering — the ZST
/// marker is never dereferenced.
pub(in crate::front::mir) fn tyref_is_bytecode_arg_marker(ty: &TyRef, llbc: &Llbc) -> bool {
    adt_path_of_tyref(ty, llbc)
        .is_some_and(|p| p == "rustpython_compiler_core::bytecode::instruction::Arg")
}

/// An instance of a class with no fields is still an instance.
/// `Bookkeeper.immutablevalue` of an instance is `SomeInstance(classdef)`
/// (`bookkeeper.py`); `InstanceRepr.convert_const` materialises one
/// prebuilt instance per class (`rclass.py convert_const`).
/// `FunctionDesc.pycall` unions that constant into a callee `Ref` input
/// (`description.py`).
/// `Arg<T>` stays Void (`tyref_is_bytecode_arg_marker`). `()` and
/// `PhantomData` have no class_root.
pub(in crate::front::mir) fn tyref_keeps_empty_class_instance(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> bool {
    tyref_is_void_zst(ty, llbc)
        && !tyref_is_bytecode_arg_marker(ty, llbc)
        && tyref_input_class_root(ty, llbc, tombstoned).is_some()
}

/// Whether a `TyRef` resolves (behind reference wrappers) to an exact
/// RustPython compiler-core transparent bytecode-oparg integer wrapper.
///
/// `OpArg` and every type emitted by upstream's `newtype_oparg!` macro are
/// `#[repr(transparent)] struct Name(u32)`.  They are opaque dependency
/// declarations in `pyre-interpreter.ullbc`, so their source shape cannot be
/// rediscovered structurally there; keep the literal upstream macro product
/// list instead.  Multi-field oparg records and fieldless oparg enums are not
/// included (the latter take the separate discriminant-int path).
pub(in crate::front::mir) fn tyref_is_oparg(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(id) else {
        return false;
    };
    let path = td.item_meta.name_path();
    let Some(leaf) = path.strip_prefix("rustpython_compiler_core::bytecode::oparg::") else {
        return false;
    };
    matches!(
        leaf,
        "OpArg"
            | "ConstIdx"
            | "VarNum"
            | "VarNums"
            | "LoadAttr"
            | "LoadSuperAttr"
            | "Label"
            | "ResumeContext"
    )
}

/// The ADT def_id of an (already wrapper-stripped) type node, or
/// `None` for non-ADT nodes.
pub(in crate::front::mir) fn adt_node_def_id(node: &serde_json::Value) -> Option<u64> {
    type_decl_ref_adt_id(node.as_object()?.get("Adt")?.as_object()?)
}

/// The def id of a nominal ADT `TypeDeclRef` (`{"id": N, "generics": ..,
/// "builtin": null}`). Tuples and `str` are type decls too, tagged by a
/// non-null `builtin`; they have no nominal owner and answer `None`. `Box`
/// is tagged as well but its decl is the nominal `alloc::boxed::Box`, the
/// owner of `impl<T> Box<T>` and `impl Clone for Box<T>`, so it answers
/// its id.
pub(crate) fn type_decl_ref_adt_id(
    tref: &serde_json::Map<String, serde_json::Value>,
) -> Option<u64> {
    match tref.get("builtin").and_then(serde_json::Value::as_str) {
        None | Some("Box") => tref.get("id")?.as_u64(),
        Some(_) => None,
    }
}

/// Whether two ADT decl ids name one nominal type.  Charon's
/// `--monomorphize` makes each instance of a generic ADT a decl of its own;
/// the instances share the template path they were instantiated from.
pub(crate) fn same_nominal_adt(llbc: &Llbc, a: u64, b: u64) -> bool {
    if a == b {
        return true;
    }
    let (Some(x), Some(y)) = (llbc.type_by_id(a), llbc.type_by_id(b)) else {
        return false;
    };
    x.item_meta.instantiation().is_some()
        && y.item_meta.instantiation().is_some()
        && x.item_meta.name_path() == y.item_meta.name_path()
}

/// The generic arguments of the type a `TypeDeclRef` names.  A
/// Charon-monomorphized type, tuples included, is a decl of its own whose
/// references carry no arguments; they are the arguments its name was
/// instantiated at.
pub(crate) fn type_decl_ref_generics<'a>(
    tref: &'a serde_json::Map<String, serde_json::Value>,
    llbc: &'a Llbc,
) -> Option<&'a serde_json::Value> {
    let own = tref.get("generics");
    let own_has_args = own.is_some_and(|generics| generic_args_are_nonempty(generics));
    if own_has_args {
        return own;
    }
    tref.get("id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| llbc.type_by_id(id))
        .and_then(|decl| decl.item_meta.instantiation())
        .and_then(|inst| instantiation_generic_args(inst, llbc))
        .or(own)
}

pub(in crate::front::mir) fn generic_args_are_nonempty(generics: &serde_json::Value) -> bool {
    ["types", "const_generics"].iter().any(|key| {
        generics
            .get(key)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|args| !args.is_empty())
    })
}

/// Instantiated `skip_binder` as GenericArgs, including a Deduplicated /
/// Value wrapper of those args or of the instantiated Adt whose own
/// generics are the arguments. A still-wrapped skip_binder with no
/// `types` array would intern a monomorphized Option as the template
/// (`classdef: None`, `getattr __discriminant`).
pub(in crate::front::mir) fn instantiation_generic_args<'a>(
    inst: &'a serde_json::Value,
    llbc: &'a Llbc,
) -> Option<&'a serde_json::Value> {
    let mut cur = inst;
    for _ in 0..8 {
        if generic_args_are_nonempty(cur) {
            return Some(cur);
        }
        if let Some(id) = cur.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            cur = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = cur.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            cur = &arr[1];
            continue;
        }
        if let Some(adt) = cur.get("Adt").and_then(serde_json::Value::as_object) {
            return type_decl_ref_generics(adt, llbc);
        }
        return None;
    }
    None
}

/// The `builtin` tag of a `TypeDeclRef`: `"Tuple"`, `"Str"` or `"Box"`.
pub(crate) fn type_decl_ref_builtin(tref: &serde_json::Value) -> Option<&str> {
    tref.get("builtin")?.as_str()
}

/// The monomorphic-ADT class root of an (already wrapper-stripped)
/// type node, or `None` for non-ADTs and generic instantiations.
///
/// A `repr(transparent)` GC handle keeps the wrapper leaf. The handle
/// intern's as a subclass of `Deref::Target`, so getattr of the handle's
/// own fields (`error::PyError`.0) resolves on the wrapper.
///
/// A by-value transparent newtype intern's as the inner struct
/// (`celldict.py` `erase({})` / `ModuleDictStorage` → `RDict`). A
/// `type T = Struct<…>` alias intern's as that struct
/// (`ModuleDictEntries` → `RDict`). `tyref_to_value_type` already
/// peels the same wrappers; unique-graph `mergeinputargs` needs the
/// intern key to match (`annrpython.py` `mergeinputargs`).
pub(in crate::front::mir) fn adt_node_class_root_with(
    node: &serde_json::Value,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    // Pointer-shaped payloads stay on the wrapper (`PyError`).
    if transparent_by_value_inner_fields(td, llbc).is_some()
        && let Some((index, _)) = transparent_nonzst_field(td, llbc)
        && let TypeDeclKind::Struct(fields) = &td.kind
        && let Some(field) = fields.get(index)
        && let Some(field_node) =
            tyref_node(&field.ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc))
        && adt_node_def_id(field_node) != Some(def_id)
    {
        return adt_node_class_root_with(field_node, llbc, tombstoned);
    }
    if let TypeDeclKind::Alias(body) = &td.kind {
        let alias_node = body.get("aliased_ty").unwrap_or(body);
        if let Some(peeled) = type_node_peel_aliases(alias_node, llbc)
            && adt_node_def_id(peeled) != Some(def_id)
        {
            return adt_node_class_root_with(peeled, llbc, tombstoned);
        }
    }
    adt_node_class_root_identity(node, llbc, tombstoned)
}

/// `true` when a generic argument is a pointer-shaped payload: a
/// shared/mut ref, raw pointer, thin `Box`, function pointer, or a
/// transparent wrapper of one (`NonNull<T>`). `Option` of those is the
/// null-pointer niche (`tyref_is_niche_option_ptr`), not an enum class.
pub(in crate::front::mir) fn adt_type_arg_is_pointer_like(
    arg: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    adt_type_arg_is_pointer_like_depth(arg, llbc, 0)
}

/// Option's payload is `types[0]`. A leaked Instantiated argument that
/// is pointer-shaped (a hasher's `fn() -> H` phantom) is not the payload
/// and does not make the Option a null-word niche.
pub(in crate::front::mir) fn option_bare_pointer_payload(
    types: &[serde_json::Value],
    llbc: &Llbc,
) -> bool {
    types.first().is_some_and(|t| {
        adt_type_arg_is_pointer_like(t, llbc)
            && strip_ty_indirections(t, llbc)
                .and_then(adt_node_def_id)
                .is_none()
    })
}

pub(in crate::front::mir) fn adt_type_arg_is_pointer_like_depth(
    arg: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    let Some(node) = strip_ty_indirections(arg, llbc) else {
        return false;
    };
    if type_node_is_mut_ref(node, llbc)
        || type_node_is_fn_ptr(node, llbc)
        || type_node_is_thin_box(node, llbc)
        || type_node_shared_ref_pointee(node, llbc).is_some()
        || type_node_raw_ptr_pointee(node, llbc).is_some()
        || owner_root_guard::type_node_is_guard(node, llbc)
    {
        return true;
    }
    let Some(def_id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    if td.item_meta.name_path() == "core::ptr::non_null::NonNull" {
        return true;
    }
    if let Some((index, _)) = transparent_nonzst_field(td, llbc)
        && let TypeDeclKind::Struct(fields) = &td.kind
        && let Some(field) = fields.get(index)
        && let Some(inner) = tyref_node(&field.ty, llbc)
    {
        return adt_type_arg_is_pointer_like_depth(inner, llbc, depth + 1);
    }
    false
}

/// Class identity of an ADT node: the crate-stripped `module::Type`
/// (`bookkeeper.py Bookkeeper.getdesc`), plus an instantiation suffix
/// when the head splits. Does not apply the `Deref::Target` rule.
pub(in crate::front::mir) fn adt_node_class_root_identity(
    node: &serde_json::Value,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let adt = node.as_object()?.get("Adt")?.as_object()?;
    let def_id = adt_node_def_id(node)?;
    let has_type_args = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    let name = llbc.type_by_id(def_id)?.item_meta.name_path();
    if has_type_args {
        // A parameterised workspace ADT (e.g. `CodeObject<C>` used at
        // its one `ConstantData` instantiation) registers in
        // `struct_fields` under its ungeneric name like any other
        // decl, so it resolves to that flat classdef — the same
        // generics collapse `derive_program_metadata` applies.  The
        // core/std/alloc container family (`Vec<T>`, `Box<T>`, …)
        // stays excluded: those map to dedicated annotator models
        // (lists, wrappers), never to a classdef.  Same crate-root
        // convention as the trait-bound resolver above.
        //
        // Split-eligible core enums (`Option<i64>`, `Result<i32, E>`,
        // `ControlFlow<i32, E>`) take the instantiation suffix below so a
        // returned or parameter value annotates as that class
        // (`bookkeeper.py` `getuniqueclassdef(cls)`).
        // `InstanceRepr.rtype_getattr` then reads `__discriminant` off the
        // interned tagged struct (rclass.py); `ok` / `unwrap_or_default` are
        // a match on that tag returning Option or the Ok payload. A
        // pointer-niche Option is the null word, not an enum class, and
        // stays excluded. Result and ControlFlow keep their tag even when a
        // type argument is pointer-like, so the niche exclusion does not
        // apply to them. Leaving ControlFlow classdef-less collapsed every
        // `Continue` onto one template whose `__pos_0` unions `int` with
        // `()` (`classdesc.py` `generalize_attr` / `unaryop.py` `setattr`);
        // `consider_op` then never binds the `?` success payload
        // (`annrpython.py`). The `?` rewrite peels the recast this paint
        // inserts on a call-returned Option / ControlFlow
        // (`result_exc::peel_recast_chain_from`). Result-of-carrier call
        // results skip the classdef recast (`result_narrow_root`).
        let crate_root = name.split("::").next().unwrap_or(&name);
        if matches!(crate_root, "core" | "std" | "alloc") {
            let types = type_decl_ref_generics(adt, llbc)
                .and_then(|g| g.get("types"))
                .and_then(|t| t.as_array());
            let unary = types.is_some_and(|t| t.len() == 1);
            let tagged_result = name.ends_with("::result::Result");
            let tagged_control_flow = name.ends_with("::control_flow::ControlFlow");
            let tagged_option = name.ends_with("::option::Option");
            let split = adt_head_instantiation_suffix(adt, llbc).is_some();
            // A pointer-niche Option is the null word, not an enum class.
            // Gate on the modeled niche (`tyref_is_niche_option`). A
            // transparent pointer handle over a nullable raw pointer still
            // has a Branch tag, so `getattr __discriminant` needs the
            // interned class. Bare pointer/ref *payloads* the niche
            // recognizer leaves as aggregates (scalar `*mut u8`) stay
            // classdef-less. Extra Instantiated args beside that payload
            // (a hasher phantom `fn() -> H`) are not the niche test.
            // Result keeps its tag even when a type argument is pointer-like.
            let modeled_niche = tyref_is_niche_option(&TyRef::Other(node.clone()), llbc);
            let bare_pointer_arg = types.is_some_and(|types| {
                if tagged_option {
                    option_bare_pointer_payload(types, llbc)
                } else {
                    types.iter().any(|t| {
                        adt_type_arg_is_pointer_like(t, llbc)
                            && strip_ty_indirections(t, llbc)
                                .and_then(adt_node_def_id)
                                .is_none()
                    })
                }
            });
            // A generic struct in a harvested foreign-body crate is one
            // class per instantiation (`bookkeeper.py` `getuniqueclassdef`).
            // Local-crate structs stay unsuffixed (closures). Exclusive
            // `Range` / `RangeInclusive` are named during type harvest so
            // the class does not wait on FunDecl crate-root notes. The
            // container family above stays excluded; Opaque TypeDecls
            // are not `Struct` and do not match.
            let classed_struct = range_struct_instantiation_suffix(&name, adt, llbc).is_some()
                || range_inclusive_struct_instantiation_suffix(&name, adt, llbc).is_some();
            let foreign_struct =
                foreign_body_generic_struct_instantiation_suffix(&name, adt, llbc).is_some();
            let paint_core_enum =
                ((unary || tagged_option) && split && !modeled_niche && !bare_pointer_arg)
                    || (tagged_result && split)
                    || (tagged_control_flow && split);
            if !classed_struct && !foreign_struct && !paint_core_enum {
                return None;
            }
        }
    } else if matches!(
        name.split("::").next().unwrap_or(&name),
        "core" | "std" | "alloc"
    ) && matches!(
        llbc.type_by_id(def_id).map(|td| &td.kind),
        Some(TypeDeclKind::Enum(_))
    ) {
        // A core enum with no type arguments is the template, not an
        // instantiation. Painting it recasts a call result onto the
        // shared variant ClassDef whose `__pos_0` is the typevar
        // (`bookkeeper.py` `getuniqueclassdef(cls)`).
        return None;
    }
    let leaf = class_identity_spelling(&name, tombstoned);
    // A reference-payload workspace enum instantiation projects to a
    // per-instantiation base class (`result::Result<Tuple>`) so its variant
    // payloads do not union across instantiations.  The discriminant
    // narrowing reads this base class name back to mint the matching
    // `result::Result<Tuple>::Ok` variant subclass, so the suffix here and the
    // constructor / field-read sites must agree — they share
    // `adt_head_instantiation_suffix`.  Non-enum and primitive-payload
    // heads return `None` and keep the class identity.
    if let Some(suffix) = adt_head_instantiation_suffix(adt, llbc) {
        return Some(majit_ir::descr::with_instantiation_suffix(&leaf, &suffix));
    }
    if let Some(suffix) = struct_instantiation_suffix(&name, adt, llbc) {
        return Some(majit_ir::descr::with_instantiation_suffix(&leaf, &suffix));
    }
    Some(leaf)
}

/// `<K,V>` suffix for `rordereddict_entries::Entry`. One extracted body
/// serves every monomorphization; the bare `Entry` class would union `key`.
pub(in crate::front::mir) fn entry_struct_instantiation_suffix(
    name_path: &str,
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    if !name_path.ends_with("::rordereddict_entries::Entry") {
        return None;
    }
    split_struct_instantiation_suffix(adt, llbc)
}

/// `<Idx>` suffix for exclusive `core::ops::range::Range`. Named-ADT
/// arguments split (`bookkeeper.py` `getuniqueclassdef`); an unsuffixed
/// `Range` unions `start` across instantiations.
pub(in crate::front::mir) fn range_struct_instantiation_suffix(
    name_path: &str,
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    if name_path != "core::ops::range::Range" && !name_path.ends_with("::ops::range::Range") {
        return None;
    }
    split_struct_instantiation_suffix(adt, llbc)
}

/// `<Idx>` suffix for `core::ops::range::RangeInclusive`. Scalar type
/// args (`RangeInclusive<u8>` vs `<i64>`) change field layout and split
/// (`bookkeeper.py` `getuniqueclassdef`). Named during type harvest so
/// the class does not wait on FunDecl crate-root notes.
pub(in crate::front::mir) fn range_inclusive_struct_instantiation_suffix(
    name_path: &str,
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    if name_path != "core::ops::range::RangeInclusive"
        && !name_path.ends_with("::ops::range::RangeInclusive")
    {
        return None;
    }
    split_struct_instantiation_suffix(adt, llbc)
}

/// True when this artefact harvested a FunDecl body under `crate_root`.
/// Same admission as [`note_body_crate_root_into`]: not `crate`, not a
/// local alias root, and some FunDecl with an unstructured body starts
/// at that crate (`bookkeeper.py` `Bookkeeper.getdesc`).
pub(in crate::front::mir) fn llbc_harvested_foreign_body_crate_root(
    llbc: &Llbc,
    crate_root: &str,
) -> bool {
    if crate_root == "crate" || crate::local_crates::is_local_crate_root(crate_root) {
        return false;
    }
    llbc.has_body_under_crate_root(crate_root)
}

/// `<T>` suffix for a generic struct in a harvested foreign-body crate.
/// RPython has no generic classes: each monomorphization is its own
/// class (`bookkeeper.py` `getuniqueclassdef`). Local-crate structs
/// are not split (prepass4: seeding every suffixed root added closure
/// failures).
///
/// Scalar type args (`Slot<i64>` vs `<f64>`) change field layout and
/// split. Named-ADT args (`CodeObject<ConstantData>`) are the
/// omitted-default spelling of one class and stay unsuffixed.
/// Harvested crate roots used to live in TLS (`is_foreign_body_crate_root`);
/// they now live on `StructFieldRegistry`, and this site reconstructs
/// the same FunDecl-body set from the artefact (`note_body_crate_root_into`).
pub(in crate::front::mir) fn foreign_body_generic_struct_instantiation_suffix(
    name_path: &str,
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    let crate_root = name_path.split("::").next().unwrap_or(name_path);
    if !llbc_harvested_foreign_body_crate_root(llbc, crate_root) {
        return None;
    }
    let def_id = type_decl_ref_adt_id(adt)?;
    let td = llbc.type_by_id(def_id)?;
    if !matches!(td.kind, TypeDeclKind::Struct(_)) {
        return None;
    }
    let type_args = render_adt_type_args(adt, llbc, 0);
    if type_args.is_empty()
        || !type_args
            .iter()
            .any(|a| super::super::semantic::type_arg_is_scalar_layout(a))
    {
        return None;
    }
    split_struct_instantiation_suffix(adt, llbc)
}

pub(in crate::front::mir) fn struct_instantiation_suffix(
    name_path: &str,
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    entry_struct_instantiation_suffix(name_path, adt, llbc)
        .or_else(|| range_struct_instantiation_suffix(name_path, adt, llbc))
        .or_else(|| range_inclusive_struct_instantiation_suffix(name_path, adt, llbc))
        .or_else(|| foreign_body_generic_struct_instantiation_suffix(name_path, adt, llbc))
}

pub(in crate::front::mir) fn split_struct_instantiation_suffix(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    let type_args = render_adt_type_args(adt, llbc, 0);
    if type_args.is_empty()
        || !type_args
            .iter()
            .all(|a| type_arg_splits_per_instantiation(a))
    {
        return None;
    }
    Some(format!("<{}>", type_args.join(",")))
}

/// The pointee's monomorphic-ADT class root of an (already
/// wrapper-stripped) `RawPtr` type node, or `None` when the node is
/// not a raw pointer onto a plain ADT.
pub(in crate::front::mir) fn raw_ptr_pointee_class_root(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<String> {
    raw_ptr_pointee_class_root_with(node, llbc, no_tombstoned_leaves())
}

pub(in crate::front::mir) fn raw_ptr_pointee_class_root_with(
    node: &serde_json::Value,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let raw = node.as_object()?.get("RawPtr")?.as_array()?;
    let pointee = strip_ty_wrappers(raw.first()?, llbc)?;
    adt_node_class_root_with(pointee, llbc, tombstoned)
}

/// The projected list or tuple root of a raw pointer onto a Rust container.
///
/// RPython keeps pointers to `GcArray` values typed, including null pointers
/// (`lltype.nullptr(ARRAY)`).  Rust's corresponding owner can be a
/// `*mut Vec<T>` field, which otherwise enters the annotator as a
/// classdef-less pointer and cannot union with the `SomeList` produced when
/// the array is allocated.  A `*const (Vec<T>, Vec<T>)` is the kwargs
/// unerase dest (`KwargsDictStrategy.copy` `unerase` /
/// `lltype.cast_opaque_ptr`): Charon spells the builtin tuple as `(A,B)`,
/// while FieldRead owners and `is_shaped_tuple_name` use `Tuple<A,B>`.
/// Return the same rendered container spelling
/// `Bookkeeper::project_struct_field_type` / `project_shaped_tuple`
/// consume at the cast boundary.
pub(in crate::front::mir) fn raw_ptr_pointee_container_root(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<String> {
    let pointee = node
        .as_object()?
        .get("RawPtr")?
        .as_array()?
        .first()
        .and_then(|node| strip_ty_wrappers(node, llbc))?;
    if let Some(root) = tyref_shaped_tuple_root(&TyRef::Other(pointee.clone()), llbc) {
        return Some(root);
    }
    let rendered = charon_type_value_to_ast_string(pointee, llbc, 0);
    (rendered.starts_with("Vec<") || rendered.starts_with("VecDeque<") || rendered.starts_with('['))
        .then_some(rendered)
}

/// Whether `ty` is a raw pointer onto a byte-sized integer literal —
/// `*mut u8` / `*const u8` / the `i8` spellings.  This is the
/// type-erased "opaque memory address" pointer: `llmemory.Address`
/// upstream, the parameter type every heterogeneous-heap-block hook in
/// `pyre_object::gc_hook` declares.  A pointee-typed raw pointer
/// (`*mut PyFrame`, `*mut *mut PyObject`) and a wider primitive array
/// head (`*mut i64`) are all excluded — only the byte pointee carries
/// the "no pointee type at all" contract.
/// `*const u8` / `*mut u8` only. `i8` is a distinct pointee: a byte
/// string's `chars` pointer is unsigned (`rstr.py` `STR`).
pub(in crate::front::mir) fn tyref_is_raw_u8_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_peel_one_raw_ptr_node(ty, llbc)
        .and_then(|pointee| json_ty_scalar_element_spelling(pointee, llbc))
        .as_deref()
        == Some("u8")
}

/// The return is a `u8` list, or a `Result` whose `Ok` payload is one.
/// `rffi.py` `charpsize2str` returns the str and raises `MemoryError`;
/// `Result<Vec<u8>, MemoryError>` spells that raise as the `Err` arm, so
/// the value the function produces is still the `Ok` payload.
pub(in crate::front::mir) fn tyref_returns_u8_list(ret_ty: &TyRef, llbc: &Llbc) -> bool {
    let ok = crate::front::result_exc::tyref_result_ok(ret_ty, llbc);
    tyref_is_u8_list(ok.as_ref().unwrap_or(ret_ty), llbc)
}

/// Every raw `*u8` parameter is `CCHARP` (`rffi.py` `CCHARP`).
/// `getkind` of that raw Ptr is int.
pub(in crate::front::mir) fn tyref_raw_u8_ptr_param_is_ccharp(
    param_ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    tyref_is_raw_u8_ptr(param_ty, llbc)
}

/// Int-bank of a raw `*u8` value (`rffi.py` `CCHARP`). Overlay Signed
/// preserves `getkind(CCHARP)` when the producer is also stamped.
pub(in crate::front::mir) fn raw_u8_ccharp_bank(ty: &TyRef, llbc: &Llbc) -> Option<ValueType> {
    tyref_is_raw_u8_ptr(ty, llbc).then_some(ValueType::Unsigned)
}

/// Stamp a raw `*u8` cell `CCHARP`. The producing op must be Int-banked.
pub(in crate::front::mir) fn stamp_raw_u8_ccharp(var: &Variable, ty: &TyRef, llbc: &Llbc) {
    if tyref_is_raw_u8_ptr(ty, llbc) {
        stamp_as_ccharp(var);
    }
}

/// Clone a Charon `Ref` / `RawPtr` node with `pointee` in the inner slot.
fn retarget_pointer_wrapper(
    wrapper: &serde_json::Value,
    pointee: serde_json::Value,
) -> serde_json::Value {
    let mut obj = wrapper.clone();
    let Some(map) = obj.as_object_mut() else {
        return obj;
    };
    if let Some(r) = map.get_mut("Ref") {
        if let Some(arr) = r.as_array_mut() {
            if arr.len() > 1 {
                arr[1] = pointee;
            }
        } else if let Some(inner) = r.as_object_mut() {
            inner.insert("ty".to_string(), pointee);
        }
    } else if let Some(arr) = map.get_mut("RawPtr").and_then(|v| v.as_array_mut())
        && !arr.is_empty()
    {
        arr[0] = pointee;
    }
    obj
}

/// Bound TypeVar at `ty` replaced by `container`'s `Adt.generics.types`
/// (`substitute_field_type`). A `Result<T,E>::Ok.0` decl is `T`; the
/// use-site container carries `*mut u8`. `PtrRepr.rtype_getattr` keeps
/// the field's declared pointer wrappers; only the type variable is
/// replaced (`&T` with `T = i64` stays `&i64`).
pub(in crate::front::mir) fn tyref_subst_bound_typevar(
    ty: &TyRef,
    container: &TyRef,
    llbc: &Llbc,
) -> TyRef {
    let Some(mut cur) = tyref_node(ty, llbc) else {
        return clone_tyref(ty);
    };
    let mut wrappers: Vec<serde_json::Value> = Vec::new();
    for _ in 0..24 {
        let Some(peeled) = strip_ty_indirections(cur, llbc) else {
            return clone_tyref(ty);
        };
        cur = peeled;
        let Some(obj) = cur.as_object() else {
            return clone_tyref(ty);
        };
        if let Some(r) = obj.get("Ref") {
            wrappers.push(cur.clone());
            let (inner, _) = ref_inner_and_kind(r);
            let Some(inner) = inner else {
                return clone_tyref(ty);
            };
            cur = inner;
            continue;
        }
        if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
            wrappers.push(cur.clone());
            let Some(inner) = arr.first() else {
                return clone_tyref(ty);
            };
            cur = inner;
            continue;
        }
        break;
    }
    let Some(idx) = typevar_bound_index(cur) else {
        return clone_tyref(ty);
    };
    let Some(arg) = tyref_adt_generics_type_arg(container, idx as usize, llbc) else {
        return clone_tyref(ty);
    };
    if wrappers.is_empty() {
        return arg;
    }
    let Some(mut node) = tyref_node(&arg, llbc).cloned() else {
        return clone_tyref(ty);
    };
    for wrapper in wrappers.iter().rev() {
        node = retarget_pointer_wrapper(wrapper, node);
    }
    TyRef::Other(node)
}

/// One `Adt.generics.types[index]` from `container`, peeling `Ref`.
pub(in crate::front::mir) fn tyref_adt_generics_type_arg(
    container: &TyRef,
    index: usize,
    llbc: &Llbc,
) -> Option<TyRef> {
    let adt = tyref_node(container, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))?
        .as_object()?
        .get("Adt")?
        .as_object()?;
    let node = type_decl_ref_generics(adt, llbc)?
        .as_object()?
        .get("types")?
        .as_array()?
        .get(index)?;
    serde_json::from_value(node.clone()).ok()
}

/// Shared immutable `&[u8]` / `*const [u8]` (Charon `Ref`/`RawPtr` of
/// `Slice<u8>`). This TyRef is a fat window, not `rstr.py` `STR`.
/// `&mut [u8]` / `*mut [u8]` stay a list of unsigned.
pub(in crate::front::mir) fn tyref_is_shared_u8_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let (pointee, kind) = match (node.get("Ref"), node.get("RawPtr")) {
        (Some(reference), _) => {
            let Some(arr) = reference.as_array() else {
                return false;
            };
            let Some(pointee) = arr.get(1) else {
                return false;
            };
            let Some(kind) = arr.get(2).and_then(serde_json::Value::as_str) else {
                return false;
            };
            (pointee, kind)
        }
        (None, Some(raw)) => {
            let Some(arr) = raw.as_array() else {
                return false;
            };
            let Some(pointee) = arr.first() else {
                return false;
            };
            let Some(kind) = arr.get(1).and_then(serde_json::Value::as_str) else {
                return false;
            };
            (pointee, kind)
        }
        (None, None) => return false,
    };
    if kind == "Mut" {
        return false;
    }
    let Some(pointee) = strip_ty_indirections(pointee, llbc) else {
        return false;
    };
    let Some(item) = pointee
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return false;
    };
    json_ty_scalar_element_spelling(item, llbc).as_deref() == Some("u8")
}

/// Bare unsized `[u8]` / Charon `Slice<u8>` (no `Ref`/`RawPtr` wrapper).
/// Charon spells `Option<&[u8]>` as `Option<[u8]>`; the payload is the
/// fat shared slice's referent, not already-STR. A window is the
/// `Ref`/`RawPtr` wrapper (`tyref_is_shared_u8_slice`).
pub(in crate::front::mir) fn tyref_is_bare_u8_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if node.get("Ref").is_some() || node.get("RawPtr").is_some() {
        return false;
    }
    let Some(item) = node
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return false;
    };
    json_ty_scalar_element_spelling(item, llbc).as_deref() == Some("u8")
}

/// Charon's `Option<[PyObject]>` spelling of `Option<&[PyObjectRef]>`:
/// a bare slice of object pointers, one GcArray word.
pub(in crate::front::mir) fn tyref_is_bare_object_pointer_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if node.get("Ref").is_some() || node.get("RawPtr").is_some() {
        return false;
    }
    let Some(item) = node
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|item| strip_ty_indirections(item, llbc))
    else {
        return false;
    };
    json_ty_is_objectptr(item, llbc)
        || object_pointer_item_spelling(&charon_type_value_to_ast_string(item, llbc, 0))
}

/// `Option<&[u8]>` / Charon `Option<[u8]>`: a nullable fat window.
pub(in crate::front::mir) fn tyref_is_option_of_shared_u8_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    crate::front::result_exc::tyref_option_payload(ty, llbc).is_some_and(|payload| {
        tyref_is_shared_u8_slice(&payload, llbc) || tyref_is_bare_u8_slice(&payload, llbc)
    })
}

/// `Option<&[PyObjectRef]>` / Charon `Option<[PyObject]>`: one nullable
/// object GcArray (`rlist.py` `FixedSizeListRepr` / `ll_fixed_items`).
pub(in crate::front::mir) fn tyref_is_option_of_object_pointer_slice(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    crate::front::result_exc::tyref_option_payload(ty, llbc).is_some_and(|payload| {
        tyref_is_object_pointer_gcarray_slice(&payload, llbc)
            || tyref_is_bare_object_pointer_slice(&payload, llbc)
    })
}

/// `Option` of an rstr `STR` value: a nullable `SomeString`.
///
/// `P` is a value the classifier paints `ValueType::Str`
/// ([`tyref_is_string_value`]): the builtin `str`, `Utf8Str` /
/// `BytesBlock`, a shared `&str` / `&Wtf8` / `&Wtf8Buf`, or a
/// by-value `Wtf8Buf` / `Utf8Str`. `&[u8]` is the
/// [`tyref_is_u8_window`] window, painted `Int`. A rust-heap
/// `String` (`{cap, ptr, len}`) is not this family, so
/// `Option<String>` / `Option<&String>` / `Option<*mut String>`
/// stay out. `Option<Vec<u8>>` is a list.
/// The `Some` word of the admitted set is the string itself
/// (`tyref_is_niche_option`); painting the Option as the payload-enum
/// class makes a `Some` arm that aliases the base (`__pos_0`) join
/// `Instance(Option) ∪ String`.
pub(in crate::front::mir) fn tyref_is_option_of_string_value(ty: &TyRef, llbc: &Llbc) -> bool {
    crate::front::result_exc::tyref_option_payload(ty, llbc)
        .is_some_and(|payload| tyref_is_string_value(&payload, llbc))
}

/// Paint of a niche `Option<P>` whose payload is a rust-heap `String`.
///
/// The niche is one pointer word: `Some` is the payload, `None` is null.
/// `annotator/model.py` `SomeInstance(can_be_None=True)` / `SomePtr`
/// nullable uses the payload's paint, so this Option intern's as that
/// `Ref` made nullable rather than as the `Option<String>` enum class.
pub(in crate::front::mir) fn tyref_niche_option_payload_paint(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<ValueType> {
    if !tyref_is_niche_option(ty, llbc) {
        return None;
    }
    let payload = crate::front::result_exc::tyref_option_payload(ty, llbc)?;
    if tyref_is_rust_heap_string(&payload, llbc) {
        return Some(ValueType::Ref(tyref_class_root_with(
            &payload, llbc, tombstoned,
        )));
    }
    // Shared object-pointer slice: the Option *is* the nullable list.
    if let Some(root) = object_pointer_gcarray_list_root(&payload, llbc) {
        return Some(ValueType::Ref(Some(root)));
    }
    // `Option<Vec<T>>` / `Option<VecDeque<T>>` of multi-word items is the
    // nullable list (`rlist.py`). `None` is `convert_const(None)` on that
    // `ListRepr` (`option_niche_null_cast`). A `Vec` of one-word items
    // is not this niche (`tyref_is_niche_option` keeps the tag).
    let rendered = tyref_to_ast_string(&payload, llbc);
    if rendered.starts_with("Vec<") || rendered.starts_with("VecDeque<") {
        return Some(ValueType::Ref(Some(rendered)));
    }
    None
}

/// `true` when Charon's resolved layout for `ty` is a two-variant
/// enum stored in one pointer word with no `Branch` tag.
///
/// That is the physical encoding of a pointer niche: `None` is the
/// null word and `Some(p)` is `p`.  A `Branch` discriminator is a
/// tagged enum and stays an aggregate `__discriminant` read.  Generic
/// declarations (including `core::option::Option`) emit no layout;
/// those return `false` here and the payload-shape walk decides.
pub(in crate::front::mir) fn tyref_enum_layout_is_pointer_niche(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(def_id) = tyref_adt_def_id_of(ty, llbc) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(def_id) else {
        return false;
    };
    let TypeDeclKind::Enum(variants) = &td.kind else {
        return false;
    };
    if variants.len() != 2 {
        return false;
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    let Some(layout) = td.layout_for_target(llbc, &target) else {
        return false;
    };
    !layout.has_branch_discriminant() && matches!(layout.size, Some(4 | 8))
}

/// Whether `td` is a sized pointee of a shared reference: its last field
/// is sized (an empty struct or union is sized; every enum variant's last
/// field is sized). Recurses through named ADTs so a DST newtype whose
/// last field is a Slice or `str` declines. A TypeVar last field is sized
/// (the default `T: Sized` bound). Depth greater than 8 declines.
pub(in crate::front::mir) fn type_decl_is_sized_pointee(
    td: &TypeDecl,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    match &td.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            last_field_is_sized_pointee(fields, llbc, depth)
        }
        TypeDeclKind::Enum(variants) => variants
            .iter()
            .all(|v| last_field_is_sized_pointee(&v.fields, llbc, depth)),
        TypeDeclKind::Alias(body) => {
            let aliased = body.get("aliased_ty").unwrap_or(body);
            type_node_is_sized_pointee(aliased, llbc, depth)
        }
        TypeDeclKind::Opaque | TypeDeclKind::Unknown => false,
    }
}

pub(in crate::front::mir) fn last_field_is_sized_pointee(
    fields: &[FieldDecl],
    llbc: &Llbc,
    depth: usize,
) -> bool {
    match fields.last() {
        None => true,
        Some(f) => tyref_is_sized_pointee(&f.ty, llbc, depth),
    }
}

pub(in crate::front::mir) fn tyref_is_sized_pointee(ty: &TyRef, llbc: &Llbc, depth: usize) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    type_node_is_sized_pointee(node, llbc, depth)
}

pub(in crate::front::mir) fn type_node_is_sized_pointee(
    node: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 8 {
        return false;
    }
    let Some(obj) = strip_ty_indirections(node, llbc).and_then(serde_json::Value::as_object) else {
        return false;
    };
    if obj.contains_key("TypeVar") {
        return true;
    }
    if obj.contains_key("Scalar")
        || obj.contains_key("Array")
        || obj.contains_key("Ref")
        || obj.contains_key("RawPtr")
        || obj.contains_key("FnPtr")
    {
        return true;
    }
    if obj.contains_key("Slice") {
        return false;
    }
    let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    match adt.get("builtin").and_then(serde_json::Value::as_str) {
        Some("Str") => false,
        Some("Box") => true,
        Some("Tuple") => {
            let last = adt
                .get("generics")
                .and_then(|g| g.get("types"))
                .and_then(|t| t.as_array())
                .and_then(|t| t.last());
            match last {
                None => true,
                Some(last) => type_node_is_sized_pointee(last, llbc, depth + 1),
            }
        }
        None => {
            let Some(id) = adt.get("id").and_then(serde_json::Value::as_u64) else {
                return false;
            };
            let Some(td) = llbc.type_by_id(id) else {
                return false;
            };
            type_decl_is_sized_pointee(td, llbc, depth + 1)
        }
        Some(_) => false,
    }
}

/// See [`Lowering::tyref_is_niche_option_ptr`].
pub(in crate::front::mir) fn tyref_is_niche_option(ty: &TyRef, llbc: &Llbc) -> bool {
    if !crate::front::result_exc::tyref_is_option(ty, llbc) {
        return false;
    }
    // `Option<&[u8]>` is a two-word nullable window, not a one-word
    // STR niche. Pair locals carry it as `{data, len}` with a null
    // data word for `None`.
    if tyref_is_u8_window(ty, llbc) {
        return false;
    }
    // `Option` of an rstr `STR` value is the nullable string: `None` is
    // the null `STR` pointer (`StringRepr.convert_const(None)`), `Some`
    // is the string itself. `Option<&Wtf8>` and `Option<Wtf8Buf>` are the same
    // set. A rust-heap `String` is not this set; it is niched by the
    // [`tyref_is_string_family`] arm below.
    if tyref_is_option_of_string_value(ty, llbc) {
        return true;
    }
    // Shared `&[PyObjectRef]` / Charon's `Option<[PyObject]>` spelling
    // is the length-prefixed object array: one GcArray word, `None` the
    // null list (`rlist.py` `FixedSizeListRepr`). Mut object-pointer
    // slices stay the `{ptr, len}` pair.
    if tyref_is_option_of_object_pointer_slice(ty, llbc) {
        return true;
    }
    // A `Vec` of one-word items is the header address, never null
    // (`RustVecRepr`). Charon's `Option<Vec<T>>` layout may still be
    // a pointer niche (the source `Unique` word); the translated
    // Option keeps its tag so the payload stays `Int` for
    // `ll_vec_length` / `gcarray_from_pyobject_vec`.
    if crate::front::result_exc::tyref_option_payload(ty, llbc)
        .is_some_and(|payload| tyref_rust_vec_item_kind(&payload, llbc).is_some())
    {
        return false;
    }
    // Instantiated `Option<T>` whose Charon layout occupies one
    // pointer word with no `Branch` tag is already a null niche.
    // Generic `Option` declarations carry no layout; those fall
    // through to the payload-shape walk below.
    if tyref_enum_layout_is_pointer_niche(ty, llbc) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    let Some(payload) = node
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
        .and_then(|a| type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .and_then(|t| t.first())
    else {
        return false;
    };
    // `opcode_for_iter`'s match is `Option<H::Value>`. The only
    // `IterOpcodeHandler` impl binds `Value = *mut PyObject`, which
    // Rust represents as a null niche. The generic body still spells
    // the payload as a trait projection, so peel that unique binding
    // before the raw-pointer test. A multi-impl projection stays
    // unresolved and keeps the tagged aggregate.
    let resolved_assoc = trait_payload_node(payload, llbc)
        .and_then(|node| trait_assoc_projection_target(node, llbc));
    let resolved_body = resolved_assoc.as_ref().and_then(|ty| tyref_node(ty, llbc));
    let payload = resolved_body.unwrap_or(payload);
    let Some(payload) = type_node_peel_aliases(payload, llbc) else {
        return false;
    };
    // `OwnerRootGuard` lowers as the one reference it roots, so
    // `Option<OwnerRootGuard>` is the same nullable word as
    // `Option<*mut T>`.
    if owner_root_guard::type_node_is_guard(payload, llbc) {
        return true;
    }
    if type_node_is_mut_ref(payload, llbc) {
        return true;
    }
    // `Box<T>` is a non-null owning pointer, so `Option<Box<T>>` uses the
    // same null niche as `Option<&T>`.  Keep the same concrete-layout gate
    // as the reference/raw-pointer arms: `Box<dyn Trait>` and `Box<[T]>`
    // are fat pointers, while a boxed nominal ADT such as the interpreter's
    // error object is one word.  RPython's owning GC pointer has this same
    // nullable `SomeInstance` shape; there is no separate Option object.
    if type_node_is_thin_box(payload, llbc) {
        return true;
    }
    // A function pointer is one word, and `Option<fn(..)>` represents
    // `None` as the null pointer.  The payload therefore aliases the
    // base pointer directly and carries no metadata word.
    if type_node_is_fn_ptr(payload, llbc) {
        return true;
    }
    // A shared reference `&T` is equally a one-word null-pointer niche
    // (`Option<&T>` = nullable pointer, `Some(&T)` = the non-null pointer),
    // but ONLY when the pointee is thin (Sized). A fat reference — `&str` /
    // `&[T]` / `&dyn Trait`, and crucially an unsized nominal newtype like
    // `rustpython_wtf8::Wtf8 { bytes: [u8] }` (`&Wtf8` = data + length word) —
    // is two words, so aliasing the payload to the base pointer would read
    // the metadata word. String-value fat refs (`&str`, `&Wtf8`)
    // returned above; `&[u8]` is the window. A def-id alone does NOT
    // imply thin: a DST newtype is a nominal ADT with a def-id.
    // Concrete layout size is the fast path (`FrameDebugData`). A
    // generic struct still records a layout whose size expression is
    // unresolved (`None`); that is not a DST signature. A missing or
    // unconcrete size is still a thin pointee when the last field is
    // sized (`type_decl_is_sized_pointee`). A last-field Slice or `str`
    // declines, so a DST newtype stays fat.
    if let Some(shared_pointee) = type_node_shared_ref_pointee(payload, llbc) {
        let Some(stripped) = strip_ty_wrappers(shared_pointee, llbc) else {
            return false;
        };
        // Shared `&[u8]` is a two-word window. `Option<&[u8]>` is
        // handled above (`tyref_is_u8_window`); a bare `[u8]`
        // referent is not a one-word STR niche.
        if tyref_is_bare_u8_slice(&TyRef::Other(stripped.clone()), llbc) {
            return false;
        }
        // `&*mut T` is a thin shared reference even though its pointee is
        // itself a raw pointer rather than a nominal ADT.  This is the
        // shape of `<[*mut PyObject]>::get`: `Option<&PyObjectRef>` is one
        // nullable reference word, and its successful value is the list
        // item pointer.  RPython has no Option aggregate here; it is the
        // same `SomeInstance(T, can_be_None=True)` pointer lattice as the
        // direct `&T` nominal-ADT arm below.
        if type_node_raw_ptr_pointee(stripped, llbc).is_some() {
            return true;
        }
        if let Some(def_id) = adt_node_def_id(stripped)
            && llbc.type_by_id(def_id).is_some_and(|td| {
                let target = std::env::var("TARGET").unwrap_or_default();
                match td.layout_for_target(llbc, &target).and_then(|l| l.size) {
                    Some(_) => true,
                    None => type_decl_is_sized_pointee(td, llbc, 0),
                }
            })
        {
            return true;
        }
    }
    // A direct raw pointer to a nominal ADT is the one-word address of
    // that aggregate in these graphs. Keep the concrete-layout gate even
    // though such raw-pointer DSTs are not expected: Charon represents an
    // unsized nominal pointee as a fat pointer, whose metadata cannot
    // alias the single base word. Primitive/scalar pointees have no ADT
    // def-id and deliberately remain excluded because `Some(null)` may be
    // observably distinct from `None` for such payloads.
    //
    // `Option<*mut PyObjectRef>` expands to `Option<*mut *mut PyObject>`:
    // the pointee is itself a raw pointer, not an ADT, so the layout
    // gate below misses. It is still one nullable word.
    if let Some(raw_pointee) = type_node_raw_ptr_pointee(payload, llbc)
        && let Some(stripped) = strip_ty_wrappers(raw_pointee, llbc)
        && type_node_raw_ptr_pointee(stripped, llbc).is_some()
    {
        return true;
    }
    if let Some(raw_pointee) = type_node_raw_ptr_pointee(payload, llbc)
        && let Some(stripped) = strip_ty_wrappers(raw_pointee, llbc)
        && let Some(def_id) = adt_node_def_id(stripped)
        && llbc.type_by_id(def_id).is_some_and(|td| {
            // A recorded layout must name a concrete size (a DST is not
            // one word). A cross-crate nominal pointee often has no
            // layout in this ullbc at all; `RawPtr` is already the thin
            // spelling, so a missing layout is still the nullable
            // pointer, not a two-word shell. Reading `__discriminant`
            // off that pointer loads the pointee header and the match
            // falls into `unreachable`.
            match td.layout_for_target(llbc, &std::env::var("TARGET").unwrap_or_default()) {
                Some(layout) => layout.size.is_some(),
                None => td.layout.is_none(),
            }
        })
    {
        return true;
    }
    let Some(payload) = strip_ty_wrappers(payload, llbc) else {
        return false;
    };
    // The same projection as `Option<Vec<T>>` below applies to the
    // string family: rstr `STR` wrappers and rust-heap
    // `alloc::string::String` ([`tyref_is_string_family`], `SomeString`).
    // `node_is_string_value` excludes the rust-heap header; the family
    // predicate is what `option_niche_null_cast` already uses.
    // `Option<String>` is the nullable form (`s is None` /
    // `s is not None`; `pairtype(SomeObject, SomeObject).is_`,
    // `rtype_is_` / `ptr_iszero`), and `Some` is the string itself —
    // there is no `__discriminant` / `__pos_0` field.
    if tyref_is_string_family(&TyRef::Other(payload.clone()), llbc) {
        return true;
    }
    let Some(def_id) = adt_node_def_id(payload) else {
        return false;
    };
    let payload_path = llbc.type_by_id(def_id).map(|td| td.item_meta.name_path());
    // PyPy TextIOWrapper.pending_bytes is `None` or an ordinary RPython
    // list (`interp_textio.py:_writeflush`). Rust's `Vec<T>` is a
    // three-word source spelling, but the front translates it to the same
    // one-pointer `SomeList`; retaining Rust's enum tag after that
    // projection manufactures an attribute no RPython object has.
    // A `Vec` of one-word items is instead the address of its raw
    // `{ptr, len, cap}` header, which is never null, so its `Option`
    // keeps the tagged aggregate.
    match payload_path.as_deref() {
        Some("core::ptr::non_null::NonNull") => true,
        Some("alloc::vec::Vec") => {
            tyref_rust_vec_item_kind(&TyRef::Other(payload.clone()), llbc).is_none()
        }
        _ => false,
    }
}

/// Shared `&[u8]` / `*const [u8]` / `Option<&[u8]>`: the W3 window.
pub(in crate::front::mir) fn tyref_is_u8_window(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_shared_u8_slice(ty, llbc) || tyref_is_option_of_shared_u8_slice(ty, llbc)
}

/// `Vec<u8>` / `&mut [u8]` / `*mut [u8]` / `Box<[u8]>`: W2 list.
/// Bare unsized `[u8]` is the window referent (`tyref_is_bare_u8_slice`),
/// not a list: a `Deref` of `&[u8]` keeps the window / STR view.
pub(in crate::front::mir) fn tyref_is_u8_list(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_u8_window(ty, llbc) || tyref_is_bare_u8_slice(ty, llbc) {
        return false;
    }
    if tyref_is_box_u8_slice(ty, llbc) {
        return true;
    }
    let spelling = tyref_to_ast_string(ty, llbc);
    majit_ir::descr::is_list_container_spelling(&spelling)
        && (spelling.contains("[u8]")
            || spelling.contains("Vec<u8>")
            || spelling.contains("Slice<u8>"))
}

/// `Box<[u8]>` (and a `Ref` of one): a Box ADT whose type argument is a
/// `[u8]` slice. The Box identity is the LLBC `builtin: "Box"` tag, or
/// the `alloc::boxed::Box` decl the tag names; the argument is the
/// `Slice` of `u8`.
pub(in crate::front::mir) fn tyref_is_box_u8_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(adt) = node.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    let is_box = adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Box")
        || type_decl_ref_adt_id(adt)
            .and_then(|id| llbc.type_by_id(id))
            .is_some_and(|decl| decl.item_meta.name_path() == "alloc::boxed::Box");
    if !is_box {
        return false;
    }
    let Some(first) = type_decl_ref_generics(adt, llbc)
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        .and_then(|types| types.first())
        .and_then(|ty| strip_ty_indirections(ty, llbc))
    else {
        return false;
    };
    first
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .is_some_and(|item| json_ty_scalar_element_spelling(item, llbc).as_deref() == Some("u8"))
}

/// Fixed-size u8 list owner: `Box<[u8]>` or `[u8; N]`.
/// `rgc._check_resizable_list_of_chars` forces `listdef.resize()`, so
/// a fixed list has no `nonmoving_raw_ptr_for_resizable_list`.
pub(in crate::front::mir) fn tyref_is_fixed_u8_list(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_box_u8_slice(ty, llbc) {
        return true;
    }
    let array_ty = ref_pointee_ty(ty, llbc).unwrap_or_else(|| clone_tyref(ty));
    tyref_raw_array_parts(&array_ty, llbc).is_some_and(|(item, _)| {
        json_ty_scalar_element_spelling(item, llbc).as_deref() == Some("u8")
    })
}

pub(in crate::front::mir) fn tyref_is_raw_byte_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(pointee) = tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| n.as_object()?.get("RawPtr")?.as_array()?.first())
        .and_then(|n| strip_ty_wrappers(n, llbc))
    else {
        return false;
    };
    let Some(lit) = pointee.as_object().and_then(|o| o.get("Scalar")) else {
        return false;
    };
    // `{"Integer": {"Unsigned": "U8"}}` / `{"Integer": {"Signed": "I8"}}`.
    let Some(lit_obj) = lit.as_object() else {
        return false;
    };
    lit_obj
        .get("Integer")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|int| {
            matches!(
                int.get("Unsigned")
                    .or_else(|| int.get("Signed"))
                    .and_then(serde_json::Value::as_str),
                Some("U8" | "I8")
            )
        })
}

/// Whether `ty` is a pointer to a MANAGED REFERENCE — `RawPtr` onto a `RawPtr`
/// onto a named class root — the pointer-to-pointer signature of the object
/// element an items block lays out after its length header, and the receiver
/// type of brick 3's `*base.add(idx)`
/// ([`Lowering::is_list_items_elem_ptr_add`]).
///
/// Two tests, and what each one excludes is different. The inner `RawPtr`
/// rejects a typed primitive array, whose pointee is a scalar (`*mut u8` /
/// `*mut i64`) — that is what keeps a `TypedItemsBlock` out of the `Ref`-typed
/// array ops. [`raw_ptr_pointee_class_root`] answering `Some` rejects a
/// pointer-to-pointer whose pointee is not a class at all.
///
/// The root is derived rather than fixed. Comparing it against the literal
/// `"PyObject"` would make the route unreachable from any other host:
/// `adt_node_class_root` answers with the pointee's own leaf name, so a class
/// family rooted at another header answers its own name. This matches
/// `Lowering::resolve_place`'s class-singleton narrow already carries: a
/// prebuilt is annotated with the class of the object itself
/// (`rpython/annotator/bookkeeper.py` `immutablevalue`), so the root comes off the type
/// rather than out of the front end. Naming one host's root here bought no
/// safety the accessor gate does not already provide — what makes the base an
/// items pointer is [`base_traces_to_items_block_accessor`], not the spelling
/// of its element.
pub(in crate::front::mir) fn is_object_ref_items_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(outer) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(inner) = outer
        .as_object()
        .and_then(|m| m.get("RawPtr"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|n| strip_ty_wrappers(n, llbc))
    else {
        return false;
    };
    raw_ptr_pointee_class_root(inner, llbc).is_some()
}

/// List spelling of an object-items pointer (`*mut *mut Class`).
///
/// `items_block_items_base` returns that pointer, null when the block is
/// null, and the null arm is a classdef-less instance. Brick 3 lowers
/// `*base.add(idx)` to `ArrayRead`, which flowspace spells `getitem`. On a
/// classdef-less receiver that rewrites to `getattr("__getitem__")`
/// (`getitem_SomeInstance`). [`OBJECT_REF_GCARRAY_TYPE_ID`] is the descr
/// key, not a list type, so the base is narrowed through the `[Class]`
/// spelling `Bookkeeper::project_struct_field_type` already models as
/// `SomeList` — the same boundary the `[str]` and `[i64]` arms use. The
/// element class is the pointee's own leaf (`bookkeeper.py`
/// `getuniqueclassdef`).
pub(in crate::front::mir) fn object_ref_items_list_root(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let outer = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc))?;
    let inner = outer
        .as_object()?
        .get("RawPtr")?
        .as_array()?
        .first()
        .and_then(|n| strip_ty_wrappers(n, llbc))?;
    let root = raw_ptr_pointee_class_root(inner, llbc)?;
    Some(format!("[{root}]"))
}

/// Element class root of a stored object pointer (`*mut Class`).
///
/// `items_block_set_ref` writes that pointer into the object-strategy
/// items list. Call sites may name a `type T = *mut Class` alias;
/// peel it so the `RawPtr` is visible. The class is the pointee's
/// own leaf (`bookkeeper.py` `getuniqueclassdef`).
pub(in crate::front::mir) fn object_stored_ptr_class_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let node = tyref_node(ty, llbc)?;
    let node = type_node_peel_aliases(node, llbc)?;
    raw_ptr_pointee_class_root(node, llbc)
}

/// List spelling of a stored object pointer (`*mut Class`).
///
/// `items_block_set_ref` writes that pointer into the object-strategy
/// items list. The list root is `[Class]`, the same spelling
/// [`object_ref_items_list_root`] builds from an items pointer
/// (`*mut *mut Class`).
pub(in crate::front::mir) fn object_stored_ptr_items_list_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let root = object_stored_ptr_class_root(ty, llbc)?;
    Some(format!("[{root}]"))
}

/// List spelling of an object-pointer slice (`&[T]` / `*const [T]`).
///
/// `emit_object_vec_gcarray` copies a `Vec<T>` into the length-prefixed
/// object GcArray (`ll_fixed_items`). A list that is never resized is
/// `FixedSizeListRepr`; the slice is that list, annotated through the
/// position ListDef (`Bookkeeper.getlistdef`) of `[Class]`. The element
/// class is the pointee leaf, the same root `object_ref_items_list_root`
/// names for an items pointer.
pub(in crate::front::mir) fn object_pointer_slice_list_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    tyref_object_pointer_slice_mutability(ty, llbc)?;
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    let pointee = match (node.get("Ref"), node.get("RawPtr")) {
        (Some(reference), _) => reference.as_array()?.get(1)?,
        (None, Some(raw)) => raw.as_array()?.first()?,
        _ => return None,
    };
    let pointee = strip_ty_indirections(pointee, llbc)?;
    let item = pointee.get("Slice")?.as_array()?.first()?;
    let item = strip_ty_indirections(item, llbc)?;
    let root = raw_ptr_pointee_class_root(item, llbc)
        .or_else(|| adt_node_class_root_identity(item, llbc, no_tombstoned_leaves()))?;
    Some(format!("[{root}]"))
}

/// List spelling of a shared / const object-pointer slice, including
/// Charon's bare `[PyObject]` Option payload. Mut slices are the
/// `{ptr, len}` pair and stay out.
pub(in crate::front::mir) fn object_pointer_gcarray_list_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    if tyref_is_object_pointer_gcarray_slice(ty, llbc) {
        return object_pointer_slice_list_root(ty, llbc);
    }
    if !tyref_is_bare_object_pointer_slice(ty, llbc) {
        return None;
    }
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    let item = node.get("Slice")?.as_array()?.first()?;
    let item = strip_ty_indirections(item, llbc)?;
    let root = raw_ptr_pointee_class_root(item, llbc)
        .or_else(|| adt_node_class_root_identity(item, llbc, no_tombstoned_leaves()))?;
    Some(format!("[{root}]"))
}

/// Item spelling inside a `[Class]` list root. A `[T; N]` tail is not one.
pub(in crate::front::mir) fn list_spelling_item(list_root: &str) -> Option<&str> {
    let item = list_root.strip_prefix('[')?.strip_suffix(']')?;
    (!item.is_empty() && !item.contains(';')).then_some(item)
}

/// RPython item bank and ARRAY identity carried by a raw pointer into a
/// `TypedItemsBlock`. That block is exactly `GcArray(Signed)` or
/// `GcArray(Float)`, so admit only the two source spellings its runtime layout
/// documents; the ARRAY name supplies the 8-byte stride and header offset.
pub(in crate::front::mir) fn raw_ptr_typed_items_element(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(ValueType, String)> {
    let pointee = type_node_raw_ptr_pointee(tyref_node(ty, llbc)?, llbc)?;
    let spelling = json_ty_scalar_element_spelling(pointee, llbc)?;
    if !matches!(spelling.as_str(), "i64" | "f64") {
        return None;
    }
    let pointee = strip_ty_indirections(pointee, llbc)?;
    let item_ty = tyref_to_value_type(&TyRef::Other(pointee.clone()), llbc);
    Some((item_ty, format!("[{spelling}]")))
}

pub(in crate::front::mir) fn raw_ptr_u8_slice_element(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(ValueType, String)> {
    let pointee = type_node_raw_ptr_pointee(tyref_node(ty, llbc)?, llbc)?;
    let spelling = json_ty_scalar_element_spelling(pointee, llbc)?;
    if spelling != "u8" {
        return None;
    }
    Some((ValueType::Unsigned, "[u8]".to_string()))
}

/// `*const u8` (shared window `as_ptr`), not `*mut u8` (list items).
pub(in crate::front::mir) fn raw_ptr_u8_is_const(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let Some(arr) = node.get("RawPtr").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let Some(kind) = arr.get(1).and_then(serde_json::Value::as_str) else {
        return false;
    };
    kind != "Mut" && raw_ptr_u8_slice_element(ty, llbc).is_some()
}

/// The `__cast_pointer` marker call — front::mir's carrier for
/// the upstream `cast_pointer(PTRTYPE, ptr)` op (lltype.py).  The
/// target class is a trailing `ByteStr` Constant; the flowspace
/// adapter interns it and rebuilds the 2-arg upstream shape.
/// jtransform re-aliases the call to its operand
/// (`rewrite_op_cast_pointer` → `same_as`).
pub(in crate::front::mir) fn cast_pointer_marker_op(root: String, arg: Variable) -> OpKind {
    crate::model::cast_pointer_call(root, arg)
}

/// Whether a Charon type node's top-level constructor is a MUTABLE reference,
/// after following only serialization indirections. Unlike
/// [`strip_ty_wrappers`], this deliberately does not peel the reference
/// itself: callers use the result to distinguish Rust's guaranteed non-null
/// `&mut T` representation from nullable raw pointers.
///
/// Shared `&T` references carry the same null niche and are now recognised too,
/// but separately — by [`type_node_shared_ref_pointee`] gated to a thin (Sized)
/// ADT pointee inside `tyref_is_niche_option_ptr`, not by this helper (which
/// stays `&mut`-only). See that recognizer's doc for the `front::iter_next`
/// interaction the shared arm introduces.
pub(in crate::front::mir) fn type_node_is_mut_ref<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> bool {
    for _ in 0..24 {
        let Some(obj) = node.as_object() else {
            return false;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            let Some(body) = llbc.dedup_body(id) else {
                return false;
            };
            node = body;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        // `{"Ref": [region, ty, kind]}` — `kind` is `"Shared" | "Mut" | …`.
        return obj
            .get("Ref")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| arr.get(2))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind.to_ascii_lowercase().contains("mut"));
    }
    false
}

/// Int-banked pointer whose `as usize` / `as i64` is `rbuiltin.py`
/// `gen_cast` of a Ptr origin: a function pointer (`Ptr(FuncType)`), or a
/// raw pointer / reference whose pointee is `alloc::vec::Vec` (`RustVecRepr`,
/// `Ptr(Struct(raw) "RustVec")`). A raw buffer address (`*mut u8`) is the
/// type-erased byte pointer, not this origin.
pub(in crate::front::mir) fn tyref_is_ptr_to_int_cast_origin(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_option_payload_is_fn_ptr(ty, llbc)
        || tyref_node(ty, llbc).is_some_and(|node| type_node_is_fn_ptr(node, llbc))
        || tyref_ptr_pointee_is_vec(ty, llbc)
}

/// `Option<fn(..)>` — the nullable raw function pointer, one machine address.
pub(in crate::front::mir) fn tyref_option_payload_is_fn_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    if !crate::front::result_exc::tyref_is_option(ty, llbc) {
        return false;
    }
    crate::front::result_exc::tyref_option_payload(ty, llbc)
        .as_ref()
        .and_then(|payload| tyref_node(payload, llbc))
        .is_some_and(|node| type_node_is_fn_ptr(node, llbc))
}

/// `class_root` of an `Int`-banked function-pointer parameter. `history.getkind`
/// of `Ptr(FuncType)` is `int`; `derive_subject_inputcells` reads this
/// spelling to seed `SomePtr` (`fn_ptr_somevalue_for_spelling`) instead of
/// `SomeInteger`. A `fn` and `Option<fn(...)>` both carry the signature
/// spelling (`charon_type_value_to_ast_string`) so every producer of one
/// Rust `fn` type annotates one `FuncType`.
pub(in crate::front::mir) fn tyref_fn_ptr_input_class_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let owned = tyref_shared_borrow_option_pointee(ty, llbc);
    let ty = owned.as_ref().unwrap_or(ty);
    if tyref_option_payload_is_fn_ptr(ty, llbc)
        || tyref_node(ty, llbc).is_some_and(|node| type_node_is_fn_ptr(node, llbc))
    {
        return Some(tyref_to_ast_string(ty, llbc));
    }
    None
}

/// Inner `fn(inputs) -> output` spelling of a function pointer or of
/// `Option<fn(...)>`. Stamps `null_fn` with the payload's `FuncType`
/// (`lltype.nullptr`).
pub(in crate::front::mir) fn tyref_fn_ptr_spelling(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    if let Some(payload) = crate::front::result_exc::tyref_option_payload(ty, llbc)
        && tyref_node(&payload, llbc).is_some_and(|node| type_node_is_fn_ptr(node, llbc))
    {
        return Some(tyref_to_ast_string(&payload, llbc));
    }
    tyref_node(ty, llbc)
        .is_some_and(|node| type_node_is_fn_ptr(node, llbc))
        .then(|| tyref_to_ast_string(ty, llbc))
}

/// Whether a Charon type node's top-level constructor is a function pointer,
/// after following serialization indirections and `type T = fn(…)` aliases
/// (`TypeDeclKind::Alias`). A type alias of a function pointer is the
/// pointer (`lltype.FuncType` / `SomePtr`), not a nominal ADT.
pub(in crate::front::mir) fn type_node_is_fn_ptr(node: &serde_json::Value, llbc: &Llbc) -> bool {
    type_node_peel_aliases(node, llbc)
        .and_then(serde_json::Value::as_object)
        .is_some_and(|obj| obj.get("FnPtr").is_some())
}

/// The `FnDef` payload of a function-item type, after following
/// serialization indirections.  `None` for a function pointer (`FnPtr`), a
/// closure ADT, or any other shape.  Does not peel `Ref` / `RawPtr`: a
/// function item is passed by value.
pub(in crate::front::mir) fn type_node_fn_def<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        // The item's `FnPtr` sits under a `RegionBinder`
        // (`{"FnDef": {"regions": …, "skip_binder": {"kind": …}}}`).
        return obj.get("FnDef");
    }
    None
}

/// FunDecl id of a function-item (`FnDef`) type, after following
/// serialization indirections.  `None` for a function pointer (`FnPtr`), a
/// closure ADT, or any other shape.  Does not peel `Ref` / `RawPtr`: a
/// function item is passed by value.
pub(in crate::front::mir) fn type_node_fn_def_fun_id<'l>(
    node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<u64> {
    let fn_def = type_node_fn_def(node, llbc)?;
    fn_def
        .get("skip_binder")
        .unwrap_or(fn_def)
        .get("kind")?
        .get("Fun")?
        .as_u64()
}

/// Direct-call `FunctionPath` segments of a `FnDef` type, or `None` when
/// `ty` is not a named function item.
pub(in crate::front::mir) fn tyref_fn_def_call_segments(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<Vec<String>> {
    let fun_id = type_node_fn_def_fun_id(tyref_node(ty, llbc)?, llbc)?;
    let fd = llbc.fn_by_id(fun_id)?;
    let segments = fundecl_fn_item_segments(llbc, fd);
    (!segments.is_empty()).then_some(segments)
}

/// Callable named by a combinator's second argument when that argument is a
/// function item: a `FnDef` constant, or a Copy/Move of a `FnDef`-typed
/// local.  `None` for a closure ADT (handled via `call_once`) and for any
/// unproven shape.
pub(in crate::front::mir) fn operand_fn_item_segments(
    llbc: &Llbc,
    op: Option<&Operand>,
) -> Option<Vec<String>> {
    match op? {
        Operand::Const(value) => match decode_constant(llbc, value) {
            Ok(DecodedConst::FnPath(segments)) if !segments.is_empty() => Some(segments),
            _ => None,
        },
        Operand::Copy(p) | Operand::Move(p) => tyref_fn_def_call_segments(&p.ty, llbc),
    }
}

/// The `FunDecl` a combinator's function-item argument names — the operand
/// shapes [`operand_fn_item_segments`] accepts.
pub(in crate::front::mir) fn operand_fn_item_decl<'l>(
    llbc: &'l Llbc,
    op: Option<&Operand>,
) -> Option<&'l FunDecl> {
    let fun_id = match op? {
        Operand::Const(value) => {
            let kind = llbc.const_expr_kind(value)?;
            let fn_def = kind.get("FnDef")?;
            fn_def
                .get("skip_binder")
                .unwrap_or(fn_def)
                .get("kind")?
                .get("Fun")?
                .as_u64()?
        }
        Operand::Copy(p) | Operand::Move(p) => {
            type_node_fn_def_fun_id(tyref_node(&p.ty, llbc)?, llbc)?
        }
    };
    llbc.fn_by_id(fun_id)
}

/// The function item a combinator's argument names, spelled as the
/// [`RegularCall`] a direct call to it would carry — Fun id and the
/// instantiation `generics` on the `FnDef`.  Same operand shapes as
/// [`operand_fn_item_decl`]: a `FnDef` constant, or a Copy/Move of a
/// `FnDef`-typed local.  `None` when that node cannot be read as a
/// `RegularCall`.
pub(in crate::front::mir) fn operand_fn_item_call(
    llbc: &Llbc,
    op: Option<&Operand>,
) -> Option<RegularCall> {
    match op? {
        Operand::Const(value) => fn_def_of_const(llbc, value),
        Operand::Copy(p) | Operand::Move(p) => {
            regular_call_from_fn_def(type_node_fn_def(tyref_node(&p.ty, llbc)?, llbc)?)
        }
    }
}

/// The pointee type node of a shared reference `&T` (`{"Ref": [region, ty,
/// "Shared"]}`), after unwrapping `Deduplicated` / `Value`
/// indirection — or `None` when `node` is not a shared reference. The
/// companion of [`type_node_is_mut_ref`]: that answers "is this `&mut`?",
/// this returns the `&T` pointee so the caller can gate on its shape.
pub(in crate::front::mir) fn type_node_shared_ref_pointee<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        // `{"Ref": [region, ty, kind]}` — `kind` is `"Shared" | "Mut" | …`.
        let arr = obj.get("Ref").and_then(serde_json::Value::as_array)?;
        let kind = arr.get(2).and_then(serde_json::Value::as_str)?;
        if kind.eq_ignore_ascii_case("shared") {
            return arr.get(1);
        }
        return None;
    }
    None
}

/// The pointee type node of a raw pointer (`{"RawPtr": [ty, kind]}`), after
/// unwrapping `Deduplicated` / `Value` indirection — or `None` when
/// `node` is not a direct raw pointer. Deliberately does not peel a leading
/// reference: `&RawPtr` is a shared-reference payload, not a raw-pointer
/// payload.
pub(in crate::front::mir) fn type_node_raw_ptr_pointee<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        return obj
            .get("RawPtr")
            .and_then(serde_json::Value::as_array)?
            .first();
    }
    None
}

/// The pointee of Charon's builtin `Box<T>` node, after peeling serialized
/// dedup/hash-cons wrappers.  References are deliberately not peeled: a
/// `&Box<T>` is a reference niche in its own right and belongs to the shared
/// reference arm, not to the owning-box arm.
pub(in crate::front::mir) fn type_node_box_pointee<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(parts) = obj.get("Value").and_then(serde_json::Value::as_array)
            && parts.len() == 2
        {
            node = &parts[1];
            continue;
        }
        let adt = obj.get("Adt")?.as_object()?;
        let is_box = adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Box")
            || adt
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .and_then(|def_id| llbc.type_by_id(def_id))
                .is_some_and(|td| {
                    td.item_meta.name_path() == "alloc::boxed::Box"
                        || td.src.as_ref().is_some_and(|src| {
                            src.get("Builtin").and_then(serde_json::Value::as_str) == Some("Box")
                        })
                });
        if !is_box {
            return None;
        }
        return type_decl_ref_generics(adt, llbc)?
            .get("types")?
            .as_array()?
            .first();
    }
    None
}

/// Whether a Charon type node is an owning `Box<T>` whose pointee has a
/// concrete sized layout, and is therefore one pointer word rather than a fat
/// pointer.  Kept separate from the Option recognizer so real-artefact tests
/// can pin Charon's Box spelling directly.
pub(in crate::front::mir) fn type_node_is_thin_box(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(pointee) = type_node_box_pointee(node, llbc) else {
        return false;
    };
    let Some(peeled) = strip_ty_wrappers(pointee, llbc) else {
        return false;
    };
    // `Box<i64>`'s pointee is a literal, not an ADT. A sized literal is one
    // pointer word, the same as `Box<Struct>` whose layout size is known.
    if json_ty_literal_byte_size(peeled).is_some() {
        return true;
    }
    adt_node_def_id(peeled)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .and_then(|td| td.layout_for_target(llbc, &std::env::var("TARGET").unwrap_or_default()))
        .and_then(|layout| layout.size)
        .is_some()
}

/// `&dyn Trait` / `*const dyn Trait`: the pointee is unsized, so the
/// pointer is two words and the second is the vtable.
pub(in crate::front::mir) fn tyref_is_dyn_pointer(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    let Some(inner) = strip_ty_wrappers(node, llbc) else {
        return false;
    };
    inner
        .as_object()
        .is_some_and(|obj| obj.contains_key("DynTrait"))
}

/// `Box<[T]>`, `Box<str>`, and `Box<dyn Trait>` are fat pointers. A thin
/// `Box<Sized>` is one word and stays a normal field read.
pub(in crate::front::mir) fn tyref_is_inline_fat_box(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    type_node_box_pointee(node, llbc).is_some() && !type_node_is_thin_box(node, llbc)
}

/// An inline fat pointer: a fat `Box` or a fat shared/mut reference
/// (`&dyn Trait`, `&[T]`, `&str`). The metadata word is the vtable
/// (`FatMeta`) or the length (`FatLen`); `ptr_metadata` picks by
/// whether the pointee is `dyn`.
pub(in crate::front::mir) fn tyref_is_inline_fat_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_inline_fat_box(ty, llbc) || tyref_is_fat_ref(ty, llbc)
}

pub(in crate::front::mir) fn tyref_is_fat_ref(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    let Some(obj) = strip_ty_indirections(node, llbc).and_then(serde_json::Value::as_object) else {
        return false;
    };
    let Some(pointee) = obj
        .get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|a| a.get(1))
    else {
        return false;
    };
    !json_ty_is_statically_sized(pointee, llbc)
}

/// A fat `Box<dyn Trait>`, or exactly one `&dyn Trait` / `&mut dyn Trait`.
///
/// `Box<[T]>` and `Box<str>` are fat pointers and are not dyn.
/// `&&dyn Trait` is a thin pointer to a fat pointer: peeling every `Ref`
/// would accept it, so this stops after one.
pub(crate) fn tyref_is_fat_dyn(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    if let Some(pointee) = type_node_box_pointee(node, llbc) {
        if type_node_is_thin_box(node, llbc) {
            return false;
        }
        return node_is_dyn_trait(pointee, llbc);
    }
    let Some(inner) = strip_ty_indirections(node, llbc)
        .and_then(|node| node.as_object())
        .and_then(|obj| obj.get("Ref"))
        .and_then(serde_json::Value::as_array)
        .and_then(|reference| reference.get(1))
    else {
        return false;
    };
    node_is_dyn_trait(inner, llbc)
}

pub(in crate::front::mir) fn node_is_dyn_trait(node: &serde_json::Value, llbc: &Llbc) -> bool {
    strip_ty_indirections(node, llbc)
        .and_then(|node| node.as_object())
        .is_some_and(|obj| obj.contains_key("DynTrait"))
}

/// Strip the indirection wrappers a Charon type node can carry —
/// `{"Deduplicated": id}` / `{"Value": [id, ty]}` /
/// `{"Ref": [region, ty, kind]}` — and return the underlying type node
/// (`Adt`, `TypeVar`, `Literal`, …).
pub(in crate::front::mir) fn strip_ty_wrappers<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            node = arr.get(1)?;
            continue;
        }
        return Some(node);
    }
    None
}

/// Strip only the indirection wrappers — `{"Deduplicated": id}` and
/// `{"Value": [id, ty]}` — leaving any `Ref` in place.
///
/// [`strip_ty_wrappers`] peels `Ref` too, and peels it repeatedly.  A caller
/// that must account for exactly one reference level cannot use it: over
/// `&&i64` it answers `i64`, which is the item type of neither.
pub(in crate::front::mir) fn strip_ty_indirections<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        return Some(node);
    }
    None
}

/// Follow `type T = U` declarations to the aliased type node.
///
/// Charon keeps a type alias as its own `TypeDecl` (`kind: Alias`) and
/// call sites name the alias, not `U`.  A payload spelled
/// `Option<PyObjectRef>` is therefore an `Adt` of the alias rather than
/// the `RawPtr` `PyObjectRef` stands for; niche recognition has to peel
/// that layer before it can see a pointer word.
pub(in crate::front::mir) fn type_node_peel_aliases<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let stripped = strip_ty_indirections(node, llbc)?;
        let Some(def_id) = adt_node_def_id(stripped) else {
            return Some(stripped);
        };
        let Some(td) = llbc.type_by_id(def_id) else {
            return Some(stripped);
        };
        match &td.kind {
            TypeDeclKind::Alias(aliased) => {
                node = aliased;
            }
            _ => return Some(stripped),
        }
    }
    None
}

/// Struct field list of `td`, peeling a `type T = Struct<…>` alias.
///
/// Charon keeps a type alias as `TypeDeclKind::Alias` whose body is the
/// generic struct. Field projections still name those fields.
pub(in crate::front::mir) fn type_decl_struct_fields<'a>(
    td: &'a TypeDecl,
    llbc: &'a Llbc,
) -> Option<&'a [FieldDecl]> {
    match &type_decl_struct_owner(td, llbc)?.kind {
        TypeDeclKind::Struct(fields) => Some(fields.as_slice()),
        _ => None,
    }
}

/// The struct `TypeDecl` that owns `td`'s fields, peeling a
/// `type T = Struct<…>` alias (`ModuleDictEntries` → `RDict`).
pub(in crate::front::mir) fn type_decl_struct_owner<'a>(
    td: &'a TypeDecl,
    llbc: &'a Llbc,
) -> Option<&'a TypeDecl> {
    match &td.kind {
        TypeDeclKind::Struct(_) => Some(td),
        TypeDeclKind::Alias(body) => {
            let node = body.get("aliased_ty").unwrap_or(body);
            let peeled = type_node_peel_aliases(node, llbc)?;
            let inner = llbc.type_by_id(adt_node_def_id(peeled)?)?;
            matches!(&inner.kind, TypeDeclKind::Struct(_)).then_some(inner)
        }
        _ => None,
    }
}

/// The array element type behind an `Index::index` / `IndexMut::index_mut`
/// destination — exactly one reference level off its `&T` / `&mut T`.
///
/// [`strip_ty_wrappers`] cannot do this: it peels `Ref` repeatedly, so over a
/// `Vec<&i64>`'s `&&i64` it answers `i64`, the element type of neither the
/// vector nor the borrow.
pub(in crate::front::mir) fn tyref_index_element_node<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?
        .as_object()?
        .get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|a| a.get(1))
}

/// The Rust spelling of an array element whose exact width `get_type_flag`
/// computes, to be carried as the array identity.
///
/// `get_array_descr` keys on the ARRAY lltype and takes its item size from
/// `get_array_token`; there is no identity-less array upstream. Naming the
/// element restores that: `arraydescrof_concrete` reads the real width
/// instead of assuming one target word, so a `u8` element strides 1 rather
/// than 8. [`ValueType`] cannot supply it — `Unsigned` is one width-less
/// variant spanning `u8` through `usize`.
///
/// A spelling the flag table does not carry is not named, and the caller
/// leaves that site residual rather than describe it with a width nothing
/// computed. `char` is carried: it is RPython's `UniChar`, a 4-byte
/// unsigned item.
pub(in crate::front::mir) fn json_ty_scalar_element_spelling(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<String> {
    let obj = strip_ty_indirections(node, llbc)?.as_object()?;
    let lit = obj.get("Scalar")?;
    match lit.as_str() {
        Some("Bool") => return Some("bool".to_string()),
        Some("Char") => return Some("char".to_string()),
        _ => {}
    }
    let lit = lit.as_object()?;
    if lit.contains_key("Bool") {
        return Some("bool".to_string());
    }
    if lit.contains_key("Char") {
        return Some("char".to_string());
    }
    if let Some(int) = lit.get("Integer").and_then(serde_json::Value::as_object) {
        let kind = int
            .get("Signed")
            .or_else(|| int.get("Unsigned"))
            .and_then(serde_json::Value::as_str)?;
        let rust = match kind {
            "I8" => "i8",
            "I16" => "i16",
            "I32" => "i32",
            "I64" => "i64",
            "Isize" => "isize",
            "U8" => "u8",
            "U16" => "u16",
            "U32" => "u32",
            "U64" => "u64",
            "Usize" => "usize",
            _ => return None,
        };
        return Some(rust.to_string());
    }
    match lit.get("Float")?.as_str()? {
        "F32" => Some("f32".to_string()),
        "F64" => Some("f64".to_string()),
        _ => None,
    }
}

/// `true` when an array element of this type is a pointer that occupies one
/// target word — the width the identity-less descr assumes, and the one shape
/// that needs no spelling because `get_type_flag`'s reference fallback and
/// that assumption already agree.
///
/// A pointer is one word only while it is thin: a pointer to an unsized
/// pointee carries a length or a vtable beside the address.
/// `Option<Box<T>>` / `Option<&T>` / `Option<*mut T>` laid out as one
/// nullable pointer word. `rmodel.py` represents that pointer with
/// `can_be_none`; `None` is the null word and there is no tag field.
pub(in crate::front::mir) fn json_ty_is_niche_option_word(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(node) = strip_ty_wrappers(node, llbc) else {
        return false;
    };
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    if llbc
        .type_by_id(id)
        .is_none_or(|td| td.item_meta.name_path() != "core::option::Option")
    {
        return false;
    }
    let Some(payload) = node
        .get("Adt")
        .and_then(|a| a.get("generics"))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .and_then(|t| t.first())
    else {
        return false;
    };
    type_node_is_thin_box(payload, llbc) || json_ty_is_thin_pointer_element(payload, llbc)
}

pub(in crate::front::mir) fn json_ty_is_thin_pointer_element(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    // `Box<T>` of a sized `T` is one pointer word, the same width as `&T`.
    if type_node_is_thin_box(node, llbc) {
        return true;
    }
    let Some(obj) = strip_ty_indirections(node, llbc).and_then(serde_json::Value::as_object) else {
        return false;
    };
    let Some(pointee) = obj
        .get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|a| a.get(1))
        .or_else(|| {
            obj.get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|a| a.first())
        })
    else {
        return false;
    };
    json_ty_is_statically_sized(pointee, llbc)
}

/// `true` when `node` names a type of compile-time-known size, which is what
/// makes a pointer to it thin.
///
/// Recognition is positive-only, and deliberately so: `[T]` (`{"Slice": ..}`),
/// `str` (`{"Adt": {.., "builtin": "Str"}}`) and `dyn Trait` are spelled
/// apart, so a list of the unsized shapes would answer "sized" for whichever
/// one it forgot. A whitelist answers `false` for all of them, and for an
/// unresolved trait projection too.
pub(in crate::front::mir) fn json_ty_is_statically_sized(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(obj) = strip_ty_indirections(node, llbc).and_then(serde_json::Value::as_object) else {
        return false;
    };
    // A scalar, a fixed-length array, and any pointer (thin or fat) are all
    // sized values.
    if obj.contains_key("Scalar")
        || obj.contains_key("Array")
        || obj.contains_key("Ref")
        || obj.contains_key("RawPtr")
    {
        return true;
    }
    let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) else {
        return false;
    };
    // `str` is unsized. Tuple and Box are sized, and a named declaration
    // (builtin absent) is sized at this gate; `[T]` stays a top-level Slice.
    match adt.get("builtin").and_then(serde_json::Value::as_str) {
        Some("Str") => false,
        Some("Tuple" | "Box") | None => adt.get("id").and_then(serde_json::Value::as_u64).is_some(),
        Some(_) => false,
    }
}

/// A Copy scalar: a Charon `Literal` integer, float, `bool`, or `char`.
/// Arrays, tuples, and named ADTs are not scalars even when they are Copy.
pub(in crate::front::mir) fn json_ty_is_copy_scalar(node: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(obj) = strip_ty_indirections(node, llbc).and_then(serde_json::Value::as_object) else {
        return false;
    };
    let Some(scalar) = obj.get("Scalar") else {
        return false;
    };
    if matches!(scalar.as_str(), Some("Bool" | "Char")) {
        return true;
    }
    let Some(scalar) = scalar.as_object() else {
        return false;
    };
    scalar.contains_key("Integer") || scalar.contains_key("Float")
}

pub(in crate::front::mir) fn json_ty_is_copy_scalar_or_thin_ptr(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    json_ty_is_copy_scalar(node, llbc) || json_ty_is_thin_pointer_element(node, llbc)
}

/// Copy-scalar / thin-pointer clone identity, plus a zero-sized Copy
/// type (`BuildHasherDefault<_>`, `PhantomData<_>`). `ll_dict_copy`
/// (`rordereddict.py`) has no hasher object, so that clone is the value.
pub(in crate::front::mir) fn json_ty_is_copy_or_zst_clone_identity(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    json_ty_is_copy_scalar_or_thin_ptr(node, llbc)
        || tyref_is_void_zst(&TyRef::Other(node.clone()), llbc)
}

pub(in crate::front::mir) fn tyref_peel_one_ref_node<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let pointee = node
        .as_object()?
        .get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.get(1))?;
    strip_ty_indirections(pointee, llbc)
}

pub(in crate::front::mir) fn tyref_peel_one_raw_ptr_node<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let pointee = node
        .as_object()?
        .get("RawPtr")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())?;
    strip_ty_indirections(pointee, llbc)
}

/// `MutRef<T>` is the cell a GC reference occupies for an `&mut` borrow.
/// The callee writes the field, so the post-call read is not the pre-call
/// SSA value. `optimizeopt/virtualize.py` removes the cell once the callee
/// is inlined; this pass must leave the constructor.
pub(in crate::front::mir) fn is_mut_ref_cell_name(name: &str) -> bool {
    name.starts_with("MutRef<")
}

/// A GC reference behind `&mut` lives in a one-field GC cell for the
/// duration of the borrow. Copy-in/copy-out is exact because the borrow
/// is exclusive and does not outlive the call. Raw memory cannot hold a
/// GC reference.
///
/// The pointee has to lower to `ValueType::Ref`, and it has to be a thin
/// pointer (raw, borrowed, or a transparent newtype of one) whose target
/// also lowers to `Ref`.
pub(in crate::front::mir) fn mut_ref_cell_root_of(
    pointee: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    if !matches!(
        tyref_to_value_type_with(
            pointee,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs
        ),
        ValueType::Ref(_)
    ) {
        return None;
    }
    if !gc_ref_pointer(
        pointee,
        llbc,
        tombstoned,
        gc_struct_ids,
        0,
        raw_alloc_flavor_structs,
    ) {
        return None;
    }
    Some(format!("MutRef<{}>", tyref_to_ast_string(pointee, llbc)))
}

pub(in crate::front::mir) fn gc_ref_pointer(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    depth: usize,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    if depth > 4 {
        return false;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    if let Some(target) = node.as_object().and_then(pointer_target_json).cloned() {
        let Ok(target_ty) = serde_json::from_value::<TyRef>(target) else {
            return false;
        };
        return matches!(
            tyref_to_value_type_with(
                &target_ty,
                llbc,
                tombstoned,
                gc_struct_ids,
                raw_alloc_flavor_structs
            ),
            ValueType::Ref(_)
        );
    }
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(decl) = llbc.type_by_id(id) else {
        return false;
    };
    if !decl.is_repr_transparent() || decl.published_struct_copy {
        return false;
    }
    let TypeDeclKind::Struct(fields) = &decl.kind else {
        return false;
    };
    let field = if let Some((index, _)) = transparent_nonzst_field(decl, llbc) {
        fields.get(index)
    } else if fields.len() == 1 {
        fields.first()
    } else {
        None
    };
    field.is_some_and(|field| {
        gc_ref_pointer(
            &field.ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            depth + 1,
            raw_alloc_flavor_structs,
        )
    })
}

pub(in crate::front::mir) fn pointer_target_json(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<&serde_json::Value> {
    if let Some(target) = obj
        .get("RawPtr")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    {
        return Some(target);
    }
    obj.get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.get(1))
}

pub(in crate::front::mir) fn tyref_mut_ref_pointee(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let reference = node.as_object()?.get("Ref")?.as_array()?;
    if reference.get(2)?.as_str() != Some("Mut") {
        return None;
    }
    serde_json::from_value(reference.get(1)?.clone()).ok()
}

/// `&mut T` occupies a MutRef cell when `T` is a GC pointer.
/// Shared by callee input seeding (`gc_mut_ref_param_locals`), caller
/// wrapping (`gc_mut_ref_signature`), and Method routing
/// (`impl_method_owner`).
pub(in crate::front::mir) fn tyref_gc_mut_ref_cell_root(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    tyref_mut_ref_pointee(ty, llbc).and_then(|pointee| {
        mut_ref_cell_root_of(
            &pointee,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        )
    })
}

pub(in crate::front::mir) fn tyref_is_type_var(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc).is_some_and(|node| node.get("TypeVar").is_some())
}

/// `dont_look_inside` is keyed by `strip_crate_prefix(name_path())`.
pub(in crate::front::mir) fn fun_decl_is_dont_look_inside(
    fd: &FunDecl,
    dont_look_inside: &std::collections::HashSet<String>,
) -> bool {
    dont_look_inside.contains(&strip_crate_prefix(&fd.item_meta.name_path()))
}

/// A callee reached through its real Rust ABI. It has no unstructured
/// body to enter, or its path is in `dont_look_inside`.
pub(in crate::front::mir) fn fun_decl_is_residual(
    fd: &FunDecl,
    dont_look_inside: &std::collections::HashSet<String>,
) -> bool {
    !fd.has_unstructured_body() || fun_decl_is_dont_look_inside(fd, dont_look_inside)
}

pub(in crate::front::mir) fn gc_mut_ref_param_locals(
    locals: &majit_charon_reader::ullbc::Locals,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    self_dont_look_inside: bool,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> std::collections::HashSet<usize> {
    // A `dont_look_inside` body is called through the same raw-word ABI
    // its residual callers use, so its parameters are not cells.
    if self_dont_look_inside {
        return std::collections::HashSet::new();
    }
    let mut params = std::collections::HashSet::new();
    let arg_count = locals.arg_count as usize;
    for index in 1..=arg_count {
        let Some(local) = locals.locals.get(index) else {
            continue;
        };
        if tyref_gc_mut_ref_cell_root(
            &local.ty,
            llbc,
            tombstoned,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        )
        .is_some()
        {
            params.insert(index);
        }
    }
    params
}

pub(in crate::front::mir) fn borrow_kind_is_exclusive(kind: &serde_json::Value) -> bool {
    match kind.as_str() {
        Some("Mut" | "TwoPhaseMut") => true,
        Some(_) => false,
        None => kind
            .as_object()
            .is_some_and(|obj| obj.contains_key("Mut") || obj.contains_key("TwoPhaseMut")),
    }
}

pub(in crate::front::mir) fn gc_mut_ref_not_whole_local() -> LowerError {
    LowerError::Unsupported(
        "mutable borrow of a GC reference that is not a whole local".to_string(),
    )
}

pub(in crate::front::mir) enum GcMutRefArg {
    WholeLocal(usize),
    CellParam(usize),
    /// Field or array element: copy-in is `getfield` / `getarrayitem`,
    /// copy-out is `setfield` / `setarrayitem`.
    Place(Place),
}

pub(in crate::front::mir) enum GcMutRefCopy {
    Local {
        cell: Variable,
        root: String,
        local: usize,
    },
    Place {
        cell: Variable,
        root: String,
        place: Place,
    },
}

pub(in crate::front::mir) fn path_names_decl(decl_path: &str, segments: &[String]) -> bool {
    if segments.is_empty() {
        return false;
    }
    let joined = segments.join("::");
    if joined == decl_path {
        return true;
    }
    // A bare leaf (`new`, `get`) collides across decls. A module segment
    // keeps the name on this decl.
    joined.contains("::")
        && (decl_path.ends_with(&format!("::{joined}"))
            || joined.ends_with(&format!("::{decl_path}")))
}

/// The pointee of `&self` for `Clone::clone`, or the type itself when the
/// argument is already a by-value Copy scalar / thin pointer.
pub(in crate::front::mir) fn tyref_clone_pointee_node<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    tyref_peel_one_ref_node(ty, llbc).or_else(|| strip_ty_indirections(tyref_node(ty, llbc)?, llbc))
}

pub(in crate::front::mir) fn tyref_is_copy_scalar_or_thin_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .is_some_and(|node| json_ty_is_copy_scalar_or_thin_ptr(node, llbc))
}

pub(in crate::front::mir) fn tyref_is_char(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .is_some_and(|node| {
            let Some(lit) = node.as_object().and_then(|o| o.get("Scalar")) else {
                return false;
            };
            lit.as_str() == Some("Char") || lit.as_object().is_some_and(|o| o.contains_key("Char"))
        })
}

/// Needle of `str::contains<P>` after Pattern monomorphisation.
pub(in crate::front::mir) enum StrContainsNeedleKind {
    /// `char`, including a utf8iter code-point Variable still typed as `char`.
    Char,
    /// `&str` / `&String` / string-family.
    Str,
}

/// `str::contains<P>`'s monomorphised Pattern type `P`.
///
/// Charon `--monomorphize` copies the instance arguments onto the call
/// (`RegularCall::take_instance_arguments`); the FunDecl Instantiated
/// segment is the fallback when the call's `types` list is empty.
pub(in crate::front::mir) fn str_contains_pattern_generic(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<TyRef> {
    let from_call = reg
        .generics
        .get("types")
        .and_then(serde_json::Value::as_array)
        .and_then(|types| types.first())
        .cloned();
    let from_decl = match &reg.kind {
        CallKind::Fun(FunId::Regular { id }) => llbc
            .fn_by_id(*id)
            .and_then(|fd| fd.item_meta.instantiation())
            .and_then(|inst| inst.get("types"))
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.first())
            .cloned(),
        _ => None,
    };
    let raw = from_call.or(from_decl)?;
    Some(serde_json::from_value(raw.clone()).unwrap_or_else(|_| TyRef::Other(raw)))
}

/// `{"Scalar": {"Integer": …}}` after stripping `&T` / `*T`.
/// utf8iter yields a nonneg SomeInteger code point for `str::chars`.
pub(in crate::front::mir) fn tyref_is_integer_scalar(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .and_then(|node| node.as_object()?.get("Scalar"))
        .and_then(serde_json::Value::as_object)
        .is_some_and(|scalar| scalar.contains_key("Integer"))
}

pub(in crate::front::mir) fn json_ty_literal_byte_size(node: &serde_json::Value) -> Option<i64> {
    let lit = node.as_object()?.get("Scalar")?;
    if lit.as_str() == Some("Bool") {
        return Some(1);
    }
    if lit.as_str() == Some("Char") {
        return Some(4);
    }
    let lit = lit.as_object()?;
    if lit.contains_key("Bool") {
        return Some(1);
    }
    if lit.contains_key("Char") {
        return Some(4);
    }
    let word = crate::layout::target_word_size() as i64;
    if let Some(int) = lit.get("Integer").and_then(serde_json::Value::as_object) {
        let atom = int
            .get("Signed")
            .or_else(|| int.get("Unsigned"))
            .and_then(serde_json::Value::as_str);
        if let Some(atom) = atom {
            return Some(match atom {
                "U8" | "I8" => 1,
                "U16" | "I16" => 2,
                "U32" | "I32" => 4,
                "U64" | "I64" => 8,
                "U128" | "I128" => 16,
                "Usize" | "Isize" => word,
                _ => return None,
            });
        }
    }
    match lit.get("Float").and_then(serde_json::Value::as_str) {
        Some("F32") => Some(4),
        Some("F64") => Some(8),
        _ => None,
    }
}

pub(in crate::front::mir) fn json_ty_byte_size(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<i64> {
    let node = strip_ty_indirections(node, llbc)?;
    if json_ty_is_thin_pointer_element(node, llbc) {
        return Some(crate::layout::target_word_size() as i64);
    }
    if let Some(size) = json_ty_literal_byte_size(node) {
        return Some(size);
    }
    let adt = inline_adt_def_id(node)?;
    let target = std::env::var("TARGET").unwrap_or_default();
    let decl = llbc.type_by_id(adt)?;
    // `layout_for_target` drops the whole layout when a field offset is
    // still a deduplicated expression. `<*const T>::add` only needs the
    // byte size (`size_align_for_target`).
    if let Some(size) = decl
        .layout_for_target(llbc, &target)
        .and_then(|layout| layout.size)
    {
        return Some(size as i64);
    }
    decl.size_align_for_target(llbc, &target)?
        .0
        .map(|size| size as i64)
}

pub(in crate::front::mir) fn json_ty_raw_store_descr(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> Option<(ValueType, usize, bool)> {
    let node = strip_ty_indirections(node, llbc)?;
    if json_ty_is_thin_pointer_element(node, llbc) {
        return Some((ValueType::Int, crate::layout::target_word_size(), false));
    }
    let size = json_ty_literal_byte_size(node)? as usize;
    let lit = node.as_object()?.get("Scalar")?;
    if matches!(lit.as_str(), Some("Bool" | "Char")) {
        return Some((ValueType::Int, size, false));
    }
    let lit = lit.as_object()?;
    if lit.contains_key("Bool") || lit.contains_key("Char") {
        return Some((ValueType::Int, size, false));
    }
    if let Some(int) = lit.get("Integer").and_then(serde_json::Value::as_object) {
        if int.get("Unsigned").and_then(serde_json::Value::as_str) == Some("U128") {
            return Some((ValueType::UInt128, size, false));
        }
        if int.get("Signed").and_then(serde_json::Value::as_str) == Some("I128") {
            return Some((ValueType::Int128, size, true));
        }
        if int.contains_key("Unsigned") {
            return Some((ValueType::Unsigned, size, false));
        }
        if int.contains_key("Signed") {
            return Some((ValueType::Int, size, true));
        }
    }
    if lit.get("Float").and_then(serde_json::Value::as_str) == Some("F32") {
        return Some((ValueType::SingleFloat, size, false));
    }
    if lit.contains_key("Float") {
        return Some((ValueType::Float, size, false));
    }
    None
}

/// Does the iterator ADT named by `path` hand back a reference *it* added,
/// rather than the element itself?
///
/// `core::slice::iter::Iter<'a, T>` yields `Option<&'a T>`: that `&` is the
/// iterator's, and peeling it is what leaves `&[i64]`'s element an `i64`.
/// The by-value iterators [`Lowering::is_concrete_iter_constructor`] admits
/// alongside it — `alloc::vec::into_iter::IntoIter<T>` for `Vec<T>` and
/// `core::array::iter::IntoIter<T, N>` for `[T; N]` — yield `Option<T>`
/// instead, so their payload is already the element and the same peel would
/// strip a reference the element owns.
///
/// Recognition is positive-only: an unlisted or unreadable receiver does not
/// peel, which leaves a `&T` payload typed `Ref` — the answer the fold
/// assumed unconditionally before any element type was carried.
/// `Map<I, F>::collect` records the closure argument. `slice::Iter<T>`'s
/// type argument is `T`, but `Iterator::Item` is `&T` and the closure is
/// lowered against that borrow. `tyref_to_value_type` keeps `&i64` in the
/// Ref bank and `i64` in the int bank, so recording `T` writes `__pos_0`
/// as `Int` while the body reads `Ref`.
#[allow(dead_code)] // Iterator::Item payload for Map::collect
pub(in crate::front::mir) fn map_collect_payload_value_type(
    item_ty: &TyRef,
    llbc: &Llbc,
    adds_reference: bool,
) -> ValueType {
    if !adds_reference {
        return tyref_to_value_type(item_ty, llbc);
    }
    let node = match item_ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v.clone(),
        TyRef::Dedup { id } => llbc
            .dedup_body(*id)
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    };
    let borrowed = serde_json::json!({"Ref": ["Erased", node, "Shared"]});
    tyref_to_value_type(&TyRef::Other(borrowed), llbc)
}

pub(in crate::front::mir) fn iterator_adds_a_reference(path: &str) -> bool {
    matches!(
        path,
        "core::slice::iter::Iter" | "core::slice::iter::IterMut"
    )
}

/// The element type behind an iterator's `next()` payload.
///
/// When the iterator added a reference ([`iterator_adds_a_reference`]),
/// exactly one reference level belongs to it and the rest belongs to the
/// element: over `&[i64]` the payload is `&i64` and the element is `i64`,
/// but over `&[&i64]` it is `&&i64` and the element is `&i64` — a pointer,
/// which belongs in the Ref bank.  Peeling every `Ref` would put that
/// pointer in the integer bank.
///
/// When it did not, the payload is the element already, and peeling would
/// commit the mirror-image error: `Vec<&i64>::into_iter()` yields
/// `Option<&i64>`, whose `&i64` is the element itself.
pub(in crate::front::mir) fn iterator_payload_element<'l>(
    payload: &'l serde_json::Value,
    llbc: &'l Llbc,
    iterator_adds_a_reference: bool,
) -> Option<&'l serde_json::Value> {
    let node = strip_ty_indirections(payload, llbc)?;
    if !iterator_adds_a_reference {
        return Some(node);
    }
    let Some(arr) = node.get("Ref").and_then(serde_json::Value::as_array) else {
        return Some(node);
    };
    strip_ty_indirections(arr.get(1)?, llbc)
}

/// De Bruijn *index* of a `{"TypeVar": {"Bound": [depth, index]}}` node.
/// The binder depth differs between a parameter-type position and a
/// trait-clause subject position (the clause subject sits one binder
/// deeper), so only the index participates in matching the two.
pub(in crate::front::mir) fn typevar_bound_index(node: &serde_json::Value) -> Option<u64> {
    node.get("TypeVar")?
        .get("Bound")?
        .as_array()?
        .get(1)?
        .as_u64()
}

/// True when fn-level type variable `var_index` carries an
/// `Into<String>` / `Into<Wtf8Buf>` trait clause.  The clause's trait generics spell
/// `[<subject>, <target>]`; the subject must be our variable and the
/// target must resolve to the translated string-value family, with the
/// trait decl's leaf name `Into` (a `From<String>` bound has the same
/// generics shape but means the *opposite* conversion).
///
/// The target test is [`node_is_string_value`], whose family is `str`,
/// `Wtf8`, `Wtf8Buf`, `Utf8Str`, `BytesBlock`.
pub(in crate::front::mir) fn typevar_bounded_by_into_string_value(
    var_index: u64,
    fn_generics: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(clauses) = fn_generics
        .as_object()
        .and_then(|g| g.get("trait_clauses"))
        .and_then(|c| c.as_array())
    else {
        return false;
    };
    fn strip<'a>(llbc: &'a Llbc, mut v: &'a serde_json::Value) -> Option<&'a serde_json::Value> {
        loop {
            let obj = v.as_object()?;
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                v = llbc.dedup_body(id)?;
                continue;
            }
            if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            return Some(v);
        }
    }
    for clause in clauses {
        let Some(sb) = clause
            .as_object()
            .and_then(|c| c.get("trait_"))
            .and_then(|t| t.as_object())
            .and_then(|t| t.get("skip_binder"))
            .and_then(|s| s.as_object())
        else {
            continue;
        };
        let is_into = sb
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|id| llbc.trait_by_id(id))
            .map(|td| td.item_meta.name_path())
            .is_some_and(|n| n.rsplit("::").next() == Some("Into"));
        if !is_into {
            continue;
        }
        let Some(types) = sb
            .get("generics")
            .and_then(|g| g.get("types"))
            .and_then(|t| t.as_array())
        else {
            continue;
        };
        let subject_is_var = types
            .first()
            .and_then(|s| strip(llbc, s))
            .and_then(typevar_bound_index)
            == Some(var_index);
        if !subject_is_var {
            continue;
        }
        let target_is_string = types
            .get(1)
            .and_then(|target| strip(llbc, target))
            .is_some_and(|target| node_is_string_value(target, llbc));
        if target_is_string {
            return true;
        }
    }
    false
}

/// Resolve a generic parameter type (`&T` where `T: Trait`, including a
/// trait default body's `&Self`) to the bound trait's qualified
/// `name_path()`.
///
/// [`tyref_class_root`] answers `None` for such a parameter — a
/// `TypeVar` has no ADT decl — so `OpKind::Input.class_root` stayed
/// empty and the subject graph annotated the receiver as the
/// classdef-less `SomeInstance(None)` shell, which fails on the first
/// `getattr`.  The bound trait names the receiver's only possible shape
/// when the analyzed world has exactly one concrete impl;
/// `derive_subject_inputcells` resolves the returned trait path through
/// `Bookkeeper::trait_unique_impls` and only seeds a classdef on a
/// unique hit, so carrying a multi-impl (or foreign) trait path here is
/// inert.  The qualified path (not the leaf) is the map key so two
/// distinct traits sharing a final segment cannot seed each other's
/// impl type.
///
/// Bounds declared in `core`/`std`/`alloc` (`MetaSized`, `Sized`, …)
/// are skipped: marker/stdlib traits never name a project struct, and
/// they precede the user bound in `trait_clauses` order.
pub(in crate::front::mir) fn tyref_generic_trait_bound_root(
    ty: &TyRef,
    llbc: &Llbc,
    generics: Option<&serde_json::Value>,
) -> Option<String> {
    let generics = generics?;
    let node = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let param_index = typevar_bound_index(strip_ty_wrappers(node, llbc)?)?;
    // `T: Into<String>` / `T: Into<Wtf8Buf>` — the conventional
    // message-parameter bound (`PyError::new(message: impl Into<Wtf8Buf>)`).
    // Such a variable is a string at the annotation level (the model maps
    // `String`, `str`, `Wtf8`, and `Wtf8Buf` to one string type), so it
    // resolves to the `"String"` root
    // and the input-cell derivation seeds `s_unicode0` instead of a
    // classdef-less instance shell whose field writes would poison
    // classdef attr cells.
    if typevar_bounded_by_into_string_value(param_index, generics, llbc) {
        return Some("String".to_string());
    }
    for clause in generics.get("trait_clauses")?.as_array()? {
        let Some(pred) = clause.get("trait_").and_then(|t| t.get("skip_binder")) else {
            continue;
        };
        let subject_index = pred
            .get("generics")
            .and_then(|g| g.get("types"))
            .and_then(serde_json::Value::as_array)
            .and_then(|t| t.first())
            .and_then(|s| strip_ty_wrappers(s, llbc))
            .and_then(typevar_bound_index);
        if subject_index != Some(param_index) {
            continue;
        }
        let Some(trait_id) = pred.get("id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let Some(td) = llbc.trait_by_id(trait_id) else {
            continue;
        };
        let name = td.item_meta.name_path();
        let crate_root = name.split("::").next().unwrap_or(&name);
        if matches!(crate_root, "core" | "std" | "alloc") {
            continue;
        }
        return Some(name);
    }
    None
}

/// True when `ty` is a non-unit tuple `(A, B, ...)` — Charon's
/// synthetic `Tuple` Adt with a non-empty type-argument list.  A local
/// of this type that is not a scalar `*Checked` `BinOp` result is a
/// genuine Ref tuple whose `.N` reads extract element N via a typed
/// `FieldRead`.  The inverse-emptiness check of [`is_unit_type`]: a
/// `()` (empty `types`) is the void unit, never a field-projectable
/// tuple.
pub(in crate::front::mir) fn tyref_is_tuple(ty: &TyRef, llbc: &Llbc) -> bool {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    let Some(adt) = value
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
    else {
        return false;
    };
    let is_tuple = adt.get("builtin").and_then(|i| i.as_str()) == Some("Tuple");
    let non_empty = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    is_tuple && non_empty
}

/// The numeric `.0` type of Charon's checked-arithmetic `(numeric, bool)`
/// destination. The MIR front-end represents that tuple local as the scalar
/// arithmetic result, so its [`ValueType`] must come from the tuple's first
/// Rust type argument rather than from the aggregate tuple itself.
pub(in crate::front::mir) fn tyref_checked_binop_value_type(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<ValueType> {
    let adt = tyref_node(ty, llbc)?.as_object()?.get("Adt")?.as_object()?;
    if adt.get("builtin").and_then(serde_json::Value::as_str) != Some("Tuple") {
        return None;
    }
    let types = type_decl_ref_generics(adt, llbc)?
        .as_object()?
        .get("types")?
        .as_array()?;
    if types.len() != 2 {
        return None;
    }
    let overflow_ty = strip_ty_wrappers(&types[1], llbc)?;
    if overflow_ty
        .as_object()?
        .get("Scalar")
        .and_then(serde_json::Value::as_str)
        != Some("Bool")
    {
        return None;
    }
    let value_ty = strip_ty_wrappers(&types[0], llbc)?;
    Some(tyref_to_value_type(&TyRef::Other(value_ty.clone()), llbc))
}

/// Default of `ty` is the unit value without running a `Default` body:
/// `()` (`is_unit_type`) or `core::marker::PhantomData`. Any other ZST,
/// a user type named `PhantomData` included, may have a custom
/// `Default` (`tyref_is_zero_sized` is not this).
pub(in crate::front::mir) fn tyref_default_is_unit(ty: &TyRef, llbc: &Llbc) -> bool {
    is_unit_type(ty, llbc)
        || adt_path_of_tyref(ty, llbc).as_deref() == Some("core::marker::PhantomData")
}

/// True when `ty` is Charon's unit type `()`.
///
/// Unit serializes as an `Adt` carrying the synthetic `"Tuple"`
/// type-id with zero type arguments:
/// `{"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": [], …}}}`.  A
/// non-empty `types` array is a real tuple (`(A, B)`) — a genuine
/// aggregate that is NOT void — so the emptiness check matters.
///
/// Used by the `Return` terminator to route `-> ()` bodies through
/// the void return path ([`FunctionGraph::set_return`] with `None`),
/// which drops a `Const(None, VOID)` return link.  Without it the
/// implicit `_0 = ()` unit aggregate lowers to a Ref-typed
/// transparent ctor and colors the result kind 'r', contradicting the
/// declared void kind and tripping the codewriter cross-check
/// (`codewriter.rs`).
pub(in crate::front::mir) fn is_unit_type(ty: &TyRef, llbc: &Llbc) -> bool {
    if let Some(value) = tyref_node(ty, llbc)
        && type_value_is_empty_tuple(value, llbc)
    {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc)) else {
        return false;
    };
    type_value_is_empty_tuple(node, llbc) || charon_type_value_to_ast_string(node, llbc, 0) == "()"
}

pub(in crate::front::mir) fn type_value_is_empty_tuple(
    value: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(adt) = value
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
    else {
        return false;
    };
    let is_tuple = adt.get("builtin").and_then(|i| i.as_str()) == Some("Tuple");
    let empty_types = type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| t.is_empty());
    is_tuple && empty_types
}

/// True when `ty`'s top-level constructor — after the dedup /
/// hash-cons indirections [`charon_type_value_to_ast_string`] itself
/// follows — is a reference (`&T` / `&mut T`).
///
/// `tyref_to_ast_string` strips references to their referent, so a
/// `-> &bool` return would otherwise classify as a plain `bool` stub.
/// `simple_return_type_to_lltype` rejects `syn::Type::Reference` (only
/// a bare `bool` / unit projects), so the unsafe-stub collector skips
/// reference returns to keep the stub set parity-exact.
pub(in crate::front::mir) fn output_type_is_ref(ty: &TyRef, llbc: &Llbc) -> bool {
    let mut node = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    for _ in 0..24 {
        let Some(obj) = node.as_object() else {
            return false;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            match llbc.dedup_body(id) {
                Some(body) => {
                    node = body;
                    continue;
                }
                None => return false,
            }
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        return obj.contains_key("Ref");
    }
    false
}

/// True when `ty`'s top-level constructor — after the dedup /
/// hash-cons / reference indirections [`strip_ty_wrappers`] follows —
/// is a raw pointer (`*mut` / `*const`) onto `PyObject` (the
/// `PyObjectRef` alias, `pyobject.rs`).
///
/// This is the return shape of the `W_ListObject` / `W_TupleObject`
/// constructors (`listobject.rs w_list_new`, `tupleobject.rs
/// w_tuple_new`).  `output_type_is_ref` only catches `&T`/`&mut T`
/// references — a raw pointer carries the `RawPtr` constructor, not
/// `Ref` — so the object-pointer return needs its own probe.  Feeds the
/// `dont_look_inside` residual prefill (`cutover.rs`): without a
/// `return_type` marker an object-pointer-returning opaque callee maps
/// `None`→`Void`, residualizing to a void result the caller cannot use.
/// True when a `ref`-tokened return is genuinely one machine word, so the
/// token's one-word `SomeInstance` shell matches the callee's return ABI.
///
/// Reads the `tyref_to_ast_string` spelling because the `ref` token itself
/// has already erased the distinction: `tyref_to_value_type` maps every
/// non-primitive ADT to `ValueType::Ref`, a `Vec<T>` (three words, `sret`)
/// exactly like a `*mut T`.
///
/// Admitted:
/// - a raw pointer (`*mut T` / `*const T`), one word by construction;
/// - `Result<T, PyError>` whose `Ok` payload is itself one word.  That is
///   the shape the exception bridge already gives a one-word ABI — the `Ok`
///   payload is returned in the result register and the `Err` rides
///   `BH_LAST_EXC_VALUE` + `jit_publish_exception` — and it is what
///   `PyResult` spells.
///
/// Everything else — `Vec<T>`, `String`, tuples, `Option<T>`, a named ADT by
/// value — is declined, and its callers keep their "not registered" Skip.
/// A raw pointer has no niche, so the *original* `Option<*mut T>` /
/// `Option<PyObjectRef>` return is two words. Unsafe-fn stubs call that
/// original and stay declined. Policy-opaque residuals call the
/// `#[dont_look_inside]` trampoline, which converts to one nullable word
/// ([`option_raw_adt_ptr_return_is_one_word`]).
/// `Option<&Sized>` is one nullable pointer. `tyref_to_ast_string`
/// strips the reference, so the string form cannot tell it from a
/// by-value `Option<Struct>`.
pub(in crate::front::mir) fn option_shared_ref_return_is_one_word(
    ty: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> bool {
    let mut ty = ty;
    let owned;
    if crate::front::result_exc::tyref_is_result_of_carrier(ty, llbc, spec)
        && let Some(ok) = crate::front::result_exc::tyref_result_ok(ty, llbc)
    {
        owned = ok;
        ty = &owned;
    }
    if !crate::front::result_exc::tyref_is_option(ty, llbc) {
        return false;
    }
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    let Some(payload) = node
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
        .and_then(|a| type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .and_then(|t| t.first())
    else {
        return false;
    };
    let Some(pointee) = type_node_shared_ref_pointee(payload, llbc) else {
        return false;
    };
    json_ty_byte_size(pointee, llbc).is_some()
}

/// Policy-opaque residual of `Option<*mut T>` / `Option<*const T>` whose
/// pointee is a thin nominal ADT (or a pointer to one), including a
/// `Result<Option<*mut T>, PyError>` of that shape.
///
/// Rust's original return is two words (`*mut T` has no niche). The
/// residual target is the `#[dont_look_inside]` `extern "C"` trampoline
/// (`helper_return_to_i64` `nullable_pointer_option`): `Some(p)` is `p`,
/// `None` is null — `Optional[W_Root]` / `can_be_none`. A scalar pointee
/// (`Option<*mut u8>`) stays declined: `Some(null)` is observably distinct
/// from `None` there.
pub(in crate::front::mir) fn option_raw_adt_ptr_return_is_one_word(
    ty: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> bool {
    let mut ty = ty;
    let owned;
    if crate::front::result_exc::tyref_is_result_of_carrier(ty, llbc, spec)
        && let Some(ok) = crate::front::result_exc::tyref_result_ok(ty, llbc)
    {
        owned = ok;
        ty = &owned;
    }
    if !crate::front::result_exc::tyref_is_option(ty, llbc) {
        return false;
    }
    let Some(payload) = crate::front::result_exc::tyref_option_payload(ty, llbc) else {
        return false;
    };
    let Some(node) = tyref_node(&payload, llbc) else {
        return false;
    };
    let Some(node) = type_node_peel_aliases(node, llbc) else {
        return false;
    };
    let Some(pointee) = type_node_raw_ptr_pointee(node, llbc) else {
        return false;
    };
    let Some(stripped) = strip_ty_wrappers(pointee, llbc) else {
        return false;
    };
    // `Option<*mut PyObjectRef>` expands to `Option<*mut *mut PyObject>`:
    // the pointee is itself a raw pointer, still one address word.
    if type_node_raw_ptr_pointee(stripped, llbc).is_some() {
        return true;
    }
    let Some(def_id) = adt_node_def_id(stripped) else {
        return false;
    };
    llbc.type_by_id(def_id).is_some_and(|td| {
        match td.layout_for_target(llbc, &std::env::var("TARGET").unwrap_or_default()) {
            Some(layout) => layout.size.is_some(),
            None => td.layout.is_none(),
        }
    })
}

pub(in crate::front::mir) fn ref_return_is_single_word(ast: &str) -> bool {
    let ast = ast.trim();
    if ast.starts_with("*mut ") || ast.starts_with("*const ") {
        return true;
    }
    // `Result<T,PyError>` — `tyref_to_ast_string` comma-joins type args
    // without spaces, so split the payload off the trailing `,PyError>`.
    if let Some(args) = ast
        .strip_prefix("Result<")
        .and_then(|s| s.strip_suffix(">"))
        && let Some(ok) = args.strip_suffix(",PyError")
    {
        return ok == "()" || ref_return_is_single_word(ok);
    }
    false
}

pub(crate) fn output_type_is_objectptr(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc).is_some_and(|n| json_ty_is_objectptr(n, llbc))
}

pub(in crate::front::mir) fn json_ty_is_objectptr(node: &serde_json::Value, llbc: &Llbc) -> bool {
    strip_ty_wrappers(node, llbc)
        .and_then(|n| raw_ptr_pointee_class_root(n, llbc))
        .is_some_and(|root| root == W_ROOT_CLASS_ROOT)
}

/// Resolve a Charon [`TyRef`] to the Rust type STRING the
/// `struct_fields` registry consumers expect, so
/// `derive_program_metadata` can fill `struct_fields` with real type
/// strings instead of `TyRef::label()` placeholders.
///
/// Format contract:
///   - references are STRIPPED (`&T` / `&mut T` -> `T`);
///   - raw pointers keep `*mut ` / `*const ` prefixes;
///   - integer / float / bool / char primitives use their Rust spelling;
///   - `Vec<T>` / `Option<T>` / `HashMap<K,V>` etc. are angle-bracketed
///     with comma-joined args (no spaces);
///   - slices `[T]`, arrays `[T;N]`, tuples `(A,B)` / `()`;
///   - named structs/enums use their leaf name (the registry publishes
///     both the qualified path and the bare leaf, and every consumer
///     keys on a leaf-ish form after stripping wrappers).
///
/// Shapes the resolver does not yet recognise produce a `??<key>:<json>`
/// marker that surfaces them for a follow-up widening rather than silently
/// mislabelling a field.
pub(in crate::front::mir) fn tyref_to_ast_string(ty: &TyRef, llbc: &Llbc) -> String {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    };
    match body {
        Some(v) => charon_type_value_to_ast_string(v, llbc, 0),
        None => match ty {
            TyRef::Dedup { id } => format!("??unresolved_dedup#{id}"),
            _ => "??no_body".to_string(),
        },
    }
}

pub(in crate::front::mir) fn tyref_to_field_layout_string(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> String {
    // A borrow of a fieldless enum is the tag word
    // (`tyref_to_value_type` / `tyref_is_borrowed_fieldless_enum_free`).
    // Spell it as the discriminant scalar before the reference arm, which
    // would otherwise keep `&E` as a pointer row against an Int FORCE.
    if let Some(spelling) = fieldless_enum_field_layout_spelling(ty, llbc) {
        return spelling;
    }
    // A reference field is one pointer repr, not an inline copy of its
    // referent.  `tyref_to_ast_string` intentionally erases `&T` for nominal
    // annotation lookup, but feeding that erased spelling to the physical
    // layout registry makes `&Engine` recursively enumerate every `Engine`
    // field inside the containing struct.  RPython's lltype row here is
    // `Ptr(T)`, so retain the reference constructor at the layout boundary.
    //
    // Checked before handle spelling: `transparent_handle_field_spelling`
    // peels `&` via `strip_ty_wrappers` and would paint a captured
    // `&Handle` as `*mut {Deref::Target}`. The capture's declared type is
    // the handle, whose class the live `self` already carries
    // (`ClassDef.generalize_attr` with the value's `s_value`).
    if let Some(reference) = tyref_reference_layout_string(ty, llbc) {
        return reference;
    }
    // A transparent GC handle is one pointer to its `Deref::Target`.
    // Spell that pointer before the transparent-scalar arms: the bare
    // wrapper leaf would project as the wrapper's class, and the target's
    // own name would be laid out as the whole object.
    if let Some(spelling) = transparent_handle_field_spelling(
        ty,
        llbc,
        no_tombstoned_leaves(),
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ) {
        return spelling;
    }
    // An atomic wrapper is layout-transparent over its inner scalar, and the
    // typed side already models a field of one as that inner value
    // (`tyref_atomic_inner_value_type`).  Spell the field with the scalar so
    // the string-keyed layout tables — `get_type_flag`, `type_flag_from_str`,
    // the annotator's field projection — reach the same answer instead of
    // falling to their unknown-name GC-pointer wildcard, which would mint a
    // `Ref` field descr against the runtime's `Int` publish.
    if let Some(scalar) = tyref_atomic_inner_scalar_str(ty, llbc) {
        return scalar;
    }
    if tyref_atomic_is_ptr(ty, llbc) {
        // Pointer-width, not a nested struct. `tyref_atomic_inner_scalar_str`
        // leaves `Atomic<*mut T>` out because the inner value is a pointer;
        // the field walk must still see a pointer spelling, or
        // `is_known_struct` treats the atomic as a by-value struct and drops
        // `instantiate`.  Spell the actual pointer type: an `AtomicPtr` over a
        // raw (non-GC) struct, and an erased `*mut PyObject` would make the
        // annotator's field projection answer `Instance(PyObject)` for it.
        return tyref_atomic_arg(ty, llbc)
            .and_then(|arg| serde_json::from_value::<TyRef>(arg.clone()).ok())
            .map(|arg| tyref_to_ast_string(&arg, llbc))
            .unwrap_or_else(|| "*mut PyObject".to_string());
    }
    // A `Vec` of one-word items is the header address (`RustVecRepr`,
    // kind `int`): one pointer-sized word, not the 24-byte inline Vec.
    // Spelling the Vec ADT here registers a `FLAG_STRUCT` row whose
    // size disagrees with the Int publish.
    if tyref_rust_vec_item_kind(ty, llbc).is_some() {
        return "isize".to_string();
    }
    // `Option<E>` over a densely numbered fieldless E uses E's scalar tag
    // plus one reserved value for `None`.  Preserve that physical width in
    // the containing struct's descriptor instead of treating it as a Ref.
    if let Some(niche) = tyref_option_fieldless_niche(ty, llbc) {
        return niche.storage_ty.to_string();
    }
    let Some(node) = tyref_node(ty, llbc) else {
        return tyref_to_ast_string(ty, llbc);
    };
    let Some(id) = adt_node_def_id(node) else {
        return tyref_to_ast_string(ty, llbc);
    };
    let Some(decl) = llbc.type_by_id(id) else {
        return tyref_to_ast_string(ty, llbc);
    };
    // A by-value integer newtype is the inner scalar in the containing
    // struct's descriptor (`history.getkind`). Spelling the wrapper leaf
    // would fall to the unknown-name GC-pointer wildcard and mint a Ref
    // field descr against the Int publish (`VersionTag(u64)`).
    if let TypeDeclKind::Struct(fields) = &decl.kind
        && !decl.published_struct_copy
        && let Some(inner) = integer_newtype_inner_field(decl, fields, llbc, gc_struct_ids)
    {
        return tyref_to_ast_string(inner, llbc);
    }
    if !decl.is_repr_transparent() {
        return tyref_to_ast_string(ty, llbc);
    }
    if let TypeDeclKind::Struct(fields) = &decl.kind
        && !decl.published_struct_copy
        && let Some((index, _)) = transparent_nonzst_field(decl, llbc)
        && let Some(field) = fields.get(index)
        && !matches!(
            tyref_to_value_type(&field.ty, llbc),
            ValueType::Ref(_)
                | ValueType::Str
                | ValueType::StringBuilder
                | ValueType::Void
                | ValueType::State
                | ValueType::Unknown
        )
    {
        return tyref_to_ast_string(&field.ty, llbc);
    }
    let Some(kind) = llbc.transparent_scalar_kind(&decl.item_meta.name_path()) else {
        return tyref_to_ast_string(ty, llbc);
    };
    let target = std::env::var("TARGET").unwrap_or_default();
    let size = decl
        .layout_for_target(llbc, &target)
        .and_then(|layout| layout.size)
        .unwrap_or(8);
    match (kind, size) {
        (majit_charon_reader::TransparentScalarKind::Signed, 1) => "i8".to_string(),
        (majit_charon_reader::TransparentScalarKind::Signed, 2) => "i16".to_string(),
        (majit_charon_reader::TransparentScalarKind::Signed, 4) => "i32".to_string(),
        (majit_charon_reader::TransparentScalarKind::Signed, _) => "i64".to_string(),
        (majit_charon_reader::TransparentScalarKind::Unsigned, 1) => "u8".to_string(),
        (majit_charon_reader::TransparentScalarKind::Unsigned, 2) => "u16".to_string(),
        (majit_charon_reader::TransparentScalarKind::Unsigned, 4) => "u32".to_string(),
        (majit_charon_reader::TransparentScalarKind::Unsigned, _) => "u64".to_string(),
        (majit_charon_reader::TransparentScalarKind::Bool, _) => "bool".to_string(),
        (majit_charon_reader::TransparentScalarKind::Float, 4) => "f32".to_string(),
        (majit_charon_reader::TransparentScalarKind::Float, _) => "f64".to_string(),
    }
}

/// The physical type spelling of a top-level Rust reference field.
pub(in crate::front::mir) fn tyref_reference_layout_string(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    fn visit(value: &serde_json::Value, llbc: &Llbc, depth: usize) -> Option<String> {
        if depth > 24 {
            return None;
        }
        let obj = value.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            return visit(llbc.dedup_body(id)?, llbc, depth + 1);
        }
        if let Some(pair) = obj.get("Value").and_then(serde_json::Value::as_array)
            && pair.len() == 2
        {
            return visit(&pair[1], llbc, depth + 1);
        }
        let reference = obj.get("Ref")?.as_array()?;
        let inner = charon_type_value_to_ast_string(reference.get(1)?, llbc, depth + 1);
        let mutable = reference
            .get(2)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind.eq_ignore_ascii_case("Mut"));
        Some(if mutable {
            format!("&mut {inner}")
        } else {
            format!("&{inner}")
        })
    }

    let value = match ty {
        TyRef::Inline { value: (_, value) } | TyRef::Other(value) => value,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    visit(value, llbc, 0)
}

pub(in crate::front::mir) fn peel_ref_prefix(spelling: &str) -> &str {
    let mut normalized = spelling.trim();
    loop {
        let stripped = normalized
            .strip_prefix("&mut ")
            .or_else(|| normalized.strip_prefix('&'))
            .or_else(|| normalized.strip_prefix("mut "))
            .map(str::trim_start);
        match stripped {
            Some(rest) if rest != normalized => normalized = rest,
            _ => return normalized,
        }
    }
}

/// `[T; N]` at bracket depth 0. A slice of arrays (`[[u8; 4]]`) is not one:
/// its `;` sits inside the element.
pub(in crate::front::mir) fn fixed_array_spelling(spelling: &str) -> bool {
    let Some(inner) = spelling
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
    else {
        return false;
    };
    crate::front::typestr::depth0_sep(inner, ';').is_some()
}

pub(in crate::front::mir) fn slice_element_spelling(spelling: &str) -> Option<&str> {
    let inner = spelling
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))?;
    if crate::front::typestr::depth0_sep(inner, ';').is_some() {
        return None;
    }
    let inner = inner.trim();
    (!inner.is_empty()).then_some(inner)
}

pub(in crate::front::mir) fn object_pointer_spelling(element: &str) -> bool {
    let mut pointee = element.trim();
    let mut pointer = false;
    loop {
        let stripped = pointee
            .strip_prefix("*const ")
            .or_else(|| pointee.strip_prefix("*mut "))
            .or_else(|| pointee.strip_prefix("&mut "))
            .or_else(|| pointee.strip_prefix('&'))
            .map(str::trim_start);
        match stripped {
            Some(rest) if rest != pointee => {
                pointer = true;
                pointee = rest;
            }
            _ => break,
        }
    }
    pointer && (pointee == "PyObject" || pointee.ends_with("::PyObject"))
}

/// Spellings `json_ty_scalar_element_spelling` hands to a slice reader.
pub(in crate::front::mir) fn reader_scalar_spelling(element: &str) -> bool {
    matches!(
        element,
        "bool"
            | "char"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "f32"
            | "f64"
    )
}

/// ARRAY identity a `&[T]` / `Vec<T>` reader already names for `T`.
///
/// `Index::index`, `ProjectionElem::Index`, and `<[T]>::get` all call
/// this. An object pointer is [`OBJECT_REF_GCARRAY_TYPE_ID`]. A scalar
/// spelling is `[{spelling}]` — `[u8]` / `[i64]` / `[f64]` are the
/// length-prefixed identities `nolength_from_array_type_id` already
/// treats as a GcArray, and every other scalar (`[u32]`, `[i32]`,
/// `[bool]`, `[usize]`, `[f32]`, …) stays a headerless item run. A
/// `[T; N]` place is not a slice and returns `None`. Any other slice
/// place keeps its own spelling, which is what the reader used before
/// a fixed array was allocated to match it.
pub fn slice_array_type_id(spelling: &str) -> Option<String> {
    let normalized = peel_ref_prefix(spelling);
    if normalized.starts_with("??") || normalized.is_empty() {
        return None;
    }
    // Fixed-array path. A `[T; N]` place is not a slice reader. It uses
    // the `&[T]` reader identity. `[i64]` / `[f64]` and `GcArray<i64>` /
    // `GcArray<f64>` become one descr in `canonical_array_type_id`.
    if fixed_array_spelling(normalized) {
        let (item, _) = shaped_array_parts(normalized)?;
        return slice_array_type_id(&format!("&[{item}]"));
    }
    let element = slice_element_spelling(normalized).unwrap_or(normalized);
    if element.is_empty() {
        return None;
    }
    if object_pointer_spelling(element) {
        return Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string());
    }
    if reader_scalar_spelling(element) {
        return Some(format!("[{element}]"));
    }
    slice_element_spelling(normalized).map(|_| normalized.to_string())
}

/// `local[i]` versus `(*p)[i]`. A deref's type is the pointee, so the
/// borrow spelling is the parent's (`&[T; N]`, `*const [T; N]`).
pub(in crate::front::mir) fn index_identity_inputs(place: &Place, llbc: &Llbc) -> (bool, String) {
    let (value_local, ty) = match &place.kind {
        PlaceKind::Local(_) => (true, &place.ty),
        PlaceKind::Projection(parent, ProjectionElem::Atom(name)) if name == "Deref" => {
            (false, &parent.ty)
        }
        _ => (false, &place.ty),
    };
    (value_local, tyref_to_ast_string(ty, llbc))
}

/// Pointee spelling and whether a borrow or pointer prefix was peeled.
pub(in crate::front::mir) fn pointee_spelling(spelling: &str) -> (&str, bool) {
    let trimmed = spelling.trim();
    let mut s = peel_ref_prefix(trimmed);
    let mut borrowed = s.len() != trimmed.len();
    loop {
        let next = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .map(str::trim_start);
        match next {
            Some(rest) if rest != s => {
                borrowed = true;
                s = rest;
            }
            _ => return (s, borrowed),
        }
    }
}

/// A borrowed `[T; N]` whose `&[T]` reader is a length-prefixed GcArray,
/// except `u8`. Every indexed `&[u8; N]` / `&mut [u8; N]` in the
/// interpreter is a host or inline byte array (codec state, hex tables,
/// md5 chunks, hash secrets, `smallbuf`, `Utf8LocElem::ofs`), so item 0
/// is the first byte. No `&[i64; N]`, `&[f64; N]`, `&[str; N]`, or
/// object-pointer fixed-array index exists; a borrow of those is the
/// ctor value `do_fixed_newlist` allocated.
pub(in crate::front::mir) fn borrowed_fixed_list_is_gcarray(fixed: &str) -> bool {
    let Some((item, _)) = shaped_array_parts(fixed) else {
        return false;
    };
    if item == "u8" {
        return false;
    }
    let Some(id) = slice_array_type_id(&format!("&[{item}]")) else {
        return false;
    };
    !crate::front::typestr::nolength_from_array_type_id(Some(id.as_str()))
}

/// Identity of an index of `type_spelling`.
///
/// `value_local` is `local[i]` of the `[T; N]` value itself (a copy or a
/// phi included). A field projection stays the headerless spelling: item
/// 0 is at the pointer, the positional layout of an inline
/// `FixedSizeArray`. A borrow (`&[T; N]`, `(*p)[i]`, `*const [T; N]`)
/// follows the pointee: length-prefixed when [`borrowed_fixed_list_is_gcarray`]
/// says the corpus's pointees are ctor GcArrays, headerless for `u8`.
pub(crate) fn fixed_array_index_identity(
    value_local: bool,
    type_spelling: &str,
) -> (Option<String>, bool) {
    let (normalized, borrowed) = pointee_spelling(type_spelling);
    if normalized.starts_with("??") || normalized.is_empty() {
        return (None, false);
    }
    let id = if fixed_array_spelling(normalized) {
        if (value_local && !borrowed) || (borrowed && borrowed_fixed_list_is_gcarray(normalized)) {
            slice_array_type_id(normalized).unwrap_or_else(|| normalized.to_string())
        } else {
            normalized.to_string()
        }
    } else {
        slice_array_type_id(type_spelling).unwrap_or_else(|| normalized.to_string())
    };
    let nolength = crate::front::typestr::nolength_from_array_type_id(Some(id.as_str()));
    (Some(id), nolength)
}

/// ARRAY identity for a scalar element read whose element is an integer
/// narrower than the 8 bytes the identity-less descr arm assumes.
///
/// `ValueType` banks `u8` / `u16` / `u32` / `bool` together with the
/// word-wide integers as one width-less `Unsigned`, so an `ArrayRead` that
/// carries no identity reaches `arraydescrof_concrete`'s third arm and
/// strides every int-banked element by 8.  `get_type_flag` already returns
/// the real width, but it is keyed on the element spelling and that arm has
/// none — recovering the spelling from the receiver is what routes the read
/// onto it.  Same identity role as [`fixed_array_index_identity`], reached
/// from a `Index::index` callsite instead of a `ProjectionElem::Index`.
///
/// Declines unless the spelling round-trips:
///
/// * `nolength_from_array_type_id` must read `false` on it, agreeing with
///   the `nolength: false` the callers stamp.  `get_array_descr` answers a
///   cache hit with the descr the first mint built and drops the later
///   `base_size`/`nolength`, so an identity shared with a header-less mint
///   would hand one of the two the other's base.
/// * `get_type_flag` must bank the named element as an integer narrower
///   than 8 bytes.  Everything else — pointers, floats, structs, `String`,
///   and the unknown-name fallback — keeps `None`.
///
/// A `None` here is not on its own a decline: the callsite also admits an
/// element `element_is_addressable` proves the identity-less third arm already
/// describes — a scalar spelling or a thin pointer.  An element
/// neither answers for — a multi-word aggregate, `String`, a fat pointer —
/// reaches no `ArrayRead` at all and stays a residual call.
pub(in crate::front::mir) fn narrow_item_array_type_id(
    receiver: &TyRef,
    llbc: &Llbc,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<String> {
    let identity = tyref_to_ast_string(receiver, llbc);
    if identity.starts_with("??")
        || crate::front::typestr::nolength_from_array_type_id(Some(identity.as_str()))
    {
        return None;
    }
    let elem = crate::codewriter::call::extract_element_type_from_str(&identity)?;
    let (_, item_type, item_size) =
        crate::codewriter::call::get_type_flag(&elem, raw_alloc_flavor_structs);
    (item_type == majit_ir::value::Type::Int && item_size < 8).then_some(identity)
}

/// Recursive worker for [`tyref_to_ast_string`] operating on a raw
/// Charon type-expression `Value` (a TyRef body or a nested
/// generic-argument type).  `depth` guards against pathological cycles.
pub(crate) fn charon_type_value_to_ast_string(
    v: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> String {
    if depth > 24 {
        return "??deep".to_string();
    }
    let Some(obj) = v.as_object() else {
        return "??scalar".to_string();
    };
    // Indirections — follow the dedup table / inline hash-cons one hop.
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return match llbc.dedup_body(id) {
            Some(body) => charon_type_value_to_ast_string(body, llbc, depth + 1),
            None => format!("??unresolved_dedup#{id}"),
        };
    }
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return charon_type_value_to_ast_string(&arr[1], llbc, depth + 1);
    }
    // Primitive literals.
    if let Some(lit) = obj.get("Scalar") {
        return charon_literal_to_ast_string(lit);
    }
    // References are stripped to their referent (`&T` / `&mut T` -> `T`).
    if let Some(r) = obj.get("Ref") {
        if let Some(arr) = r.as_array() {
            // `{"Ref": [region, ty, kind]}`.
            if let Some(inner) = arr.get(1) {
                return charon_type_value_to_ast_string(inner, llbc, depth + 1);
            }
        }
        return "??ref_shape".to_string();
    }
    // Raw pointers keep the mutability prefix.
    if let Some(rp) = obj.get("RawPtr") {
        if let Some(arr) = rp.as_array()
            && arr.len() == 2
        {
            let inner = charon_type_value_to_ast_string(&arr[0], llbc, depth + 1);
            let mutbl = arr[1].as_str().unwrap_or("");
            let prefix = if mutbl.eq_ignore_ascii_case("Mut") {
                "*mut "
            } else {
                "*const "
            };
            return format!("{prefix}{inner}");
        }
        return "??rawptr_shape".to_string();
    }
    // A function-pointer type, including `type T = fn(...)` aliases.
    // `lltype.FuncType` / `getfunctionptr` derive one signature; the
    // rendered spelling is that signature so every producer of one Rust
    // `fn` type annotates one `Ptr(FuncType)`.
    if type_node_is_fn_ptr(v, llbc) {
        let peeled = type_node_peel_aliases(v, llbc).unwrap_or(v);
        return charon_fn_ptr_to_ast_string(peeled, llbc, depth);
    }
    // ADTs: tuples, builtins (Box/Slice/Str/Array), and named types.
    if let Some(adt) = obj.get("Adt").and_then(|a| a.as_object()) {
        return charon_adt_to_ast_string(adt, llbc, depth);
    }
    // Top-level array `{"Array": [elem, len, trait_ref]}` -> `[elem;len]`.
    if let Some(arr) = obj.get("Array").and_then(serde_json::Value::as_array)
        && arr.len() == 3
    {
        let elem = charon_type_value_to_ast_string(&arr[0], llbc, depth + 1);
        let len = charon_array_len_to_string(&arr[1], llbc);
        return format!("[{elem};{len}]");
    }
    // Top-level slice `{"Slice": [elem, trait_ref]}` -> `[elem]`.
    if let Some(elem) = obj
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    {
        return format!(
            "[{}]",
            charon_type_value_to_ast_string(elem, llbc, depth + 1)
        );
    }
    // Trait associated-type projections (`C::Name`).  The decl-level
    // type cannot spell a concrete type, but the program-level
    // resolution is recoverable when the LLBC carries exactly ONE impl
    // of the trait: the impl's `types[]` binds each associated type to
    // its concrete value (e.g. `impl Constant for ConstantData { type
    // Name = String }` → `CodeObject<C>`'s `varnames: Box<[C::Name]>`
    // row renders `Box<[String]>`).  Ambiguous (multi-impl) or
    // unresolvable projections keep the `??TraitType` fallback below,
    // so a lookup miss stays conservative.
    if let Some(arr) = obj.get("TraitType").and_then(serde_json::Value::as_array)
        && arr.len() == 3
        && let Some(rendered) = resolve_trait_assoc_type(&arr[0], &arr[1], llbc, depth)
    {
        return rendered;
    }
    // `dyn Trait` -> `dyn <trait-root>`; recover the trait's leaf name
    // from the first trait-ref's resolved decl when present.
    if obj.contains_key("DynTrait") {
        return charon_dyn_trait_to_ast_string(&obj["DynTrait"], llbc);
    }
    // A declaration field of type `T` has no concrete spelling. The index
    // is the declaration parameter, so `RawPair<f64, usize>` can substitute
    // `??TypeVar#0` and `??TypeVar#1` separately.
    if let Some(index) = typevar_bound_index(v) {
        return format!("??TypeVar#{index}");
    }
    let key = obj.keys().next().cloned().unwrap_or_else(|| "?".into());
    format!("??{key}")
}

/// `fn(inputs) -> output` / `unsafe fn(inputs) -> output` for a Charon
/// `FnPtr` node. Nested types use the same renderer as the rest of a
/// field / `class_root` spelling (`charon_type_value_to_ast_string`).
pub(in crate::front::mir) fn charon_fn_ptr_to_ast_string(
    node: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> String {
    let fnptr = node
        .as_object()
        .and_then(|obj| obj.get("FnPtr"))
        .unwrap_or(node);
    let Some(sig) = fnptr.get("skip_binder").unwrap_or(fnptr).as_object() else {
        return "fn".to_string();
    };
    let inputs = sig
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|ty| charon_type_value_to_ast_string(ty, llbc, depth + 1))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let output = sig
        .get("output")
        .map(|ty| charon_type_value_to_ast_string(ty, llbc, depth + 1))
        .unwrap_or_else(|| "()".to_string());
    if sig.get("is_unsafe").and_then(serde_json::Value::as_bool) == Some(true) {
        format!("unsafe fn({inputs}) -> {output}")
    } else {
        format!("fn({inputs}) -> {output}")
    }
}

/// Resolve a `TraitType [traitref, assoc, generics]` projection through the
/// trait's unique impl, rendering the bound concrete type.
///
/// `traitref` names the trait via `trait_decl_ref.id`
/// (possibly behind `Value`/`Deduplicated` / `skip_binder` indirections);
/// `assoc` selects the associated item (an index in current Charon
/// output).  Returns `None` — keeping the caller's `??TraitType`
/// fallback — when the trait id cannot be recovered, when zero or
/// more than one impl of the trait exists in this LLBC (a multi-impl
/// projection is genuinely instantiation-dependent), or when the
/// unique impl carries no binding for the selected item.
pub(in crate::front::mir) fn resolve_trait_assoc_type(
    traitref: &serde_json::Value,
    assoc: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> Option<String> {
    let value = resolve_trait_assoc_type_value(traitref, assoc, llbc)?;
    Some(charon_type_value_to_ast_string(value, llbc, depth + 1))
}

/// Peel a trait associated-type projection to the type the trait's unique
/// impl binds it to, following a chain of projections but stopping at a
/// small bound so a self-referential binding cannot spin.  `None` when the
/// value is not a projection, or when the walk cannot resolve one.
pub(in crate::front::mir) fn trait_assoc_projection_target<'a>(
    value: &'a serde_json::Value,
    llbc: &'a Llbc,
) -> Option<TyRef> {
    let mut cur: &serde_json::Value = value;
    let mut resolved: Option<&'a serde_json::Value> = None;
    for _ in 0..4 {
        let Some(arr) = trait_type_projection_args(cur) else {
            break;
        };
        let next = resolve_trait_assoc_type_value(&arr[0], &arr[1], llbc)?;
        resolved = Some(next);
        cur = next;
    }
    // Hand back only a value that is no longer a projection.  The caller
    // resolves the answer by recursing into `tyref_to_value_type`, which
    // enters this walk again with a fresh hop budget, so returning a
    // still-unresolved projection would make the four-hop cap local rather
    // than total: an `A::X = B::Y`, `B::Y = A::X` binding would then recurse
    // without bound.  Declining here caps the whole resolution at four hops
    // and leaves such a chain typed `Ref`, which is what it was before this
    // walk existed.
    if trait_type_projection_args(cur).is_some() {
        return None;
    }
    // The binding is an ordinary type reference: it may be inline, hash
    // consed or a dedup id, and `TyRef`'s own deserializer is what knows
    // the three shapes apart.
    serde_json::from_value::<TyRef>(resolved?.clone()).ok()
}

/// Peel `Deduplicated` / `Value` so a trait projection in an
/// ADT generic argument is visible to [`trait_assoc_projection_target`].
pub(in crate::front::mir) fn trait_payload_node<'a>(
    mut node: &'a serde_json::Value,
    llbc: &'a Llbc,
) -> Option<&'a serde_json::Value> {
    for _ in 0..8 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        break;
    }
    Some(node)
}

/// The `[traitref, assoc, generics]` triple of an unresolved `Self::Assoc`
/// projection, or `None` when the value is any other type shape.
pub(in crate::front::mir) fn trait_type_projection_args(
    value: &serde_json::Value,
) -> Option<&[serde_json::Value]> {
    value
        .as_object()
        .and_then(|o| o.get("TraitType"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .filter(|arr| arr.len() == 3)
}

/// The type value the trait's unique impl binds to `assoc` — the shared
/// half of [`resolve_trait_assoc_type`], kept separate so a caller that
/// needs the register bank rather than the rendered name
/// ([`tyref_to_value_type`]) reads the same binding through the same
/// uniqueness rule instead of re-deriving it.
pub(in crate::front::mir) fn resolve_trait_assoc_type_value<'a>(
    traitref: &serde_json::Value,
    assoc: &serde_json::Value,
    llbc: &'a Llbc,
) -> Option<&'a serde_json::Value> {
    // A spec copy's clause substitution leaves a concrete `TraitImpl`
    // ref. That impl's `types[]` binding wins; `unique_trait_assoc_value`
    // is only the fallback when the ref is not an impl or the impl has
    // no binding for `assoc` (two impls then answer `None`).
    if let Some(impl_id) = traitref_impl_id(traitref, llbc, 0)
        && let Some(value) = trait_impl_assoc_value(llbc, impl_id, assoc)
    {
        return Some(value);
    }
    let trait_id = traitref_decl_id(traitref, llbc, 0)?;
    // One index per LLBC: the same projection is rendered once per
    // instantiation, and scanning every impl each time is quadratic in
    // the crate.
    llbc.unique_trait_assoc_value(trait_id, assoc)
}

/// `types[entry].skip_binder.value` of `impl_id` whose `TraitType` kind
/// selects `assoc`. Same row shape [`Llbc::unique_trait_assoc_value`]
/// indexes, read off this impl instead of the unique one.
pub(in crate::front::mir) fn trait_impl_assoc_value<'a>(
    llbc: &'a Llbc,
    impl_id: u64,
    assoc: &serde_json::Value,
) -> Option<&'a serde_json::Value> {
    let entries = llbc
        .trait_impls_raw()
        .get(impl_id as usize)?
        .get("types")?
        .as_array()?;
    for entry in entries {
        let Some(kind) = entry
            .get("kind")
            .and_then(|kind| kind.get("TraitType"))
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        if kind.len() != 2 || &kind[1] != assoc {
            continue;
        }
        return entry.get("skip_binder")?.get("value");
    }
    None
}

/// Recover the trait decl id a `TraitRef` names —
/// `trait_decl_ref.id`, behind the usual
/// `Value` / `Deduplicated` / `skip_binder` indirections.
pub(in crate::front::mir) fn traitref_decl_id(
    v: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> Option<u64> {
    crate::front::clause_spec::trait_decl_pred(traitref_unwrap(v, llbc, depth)?)?
        .get("id")?
        .as_u64()
}

/// Recover the trait *impl* id a resolved `TraitRef` selected —
/// `kind.TraitImpl.id`, behind `Value` / `Deduplicated` and a
/// `PolyTraitRef` `skip_binder` (`Distinguish higher-ranked trait proofs`).
/// `None` when the ref is still a clause/builtin obligation rather than
/// a selected impl.
pub(in crate::front::mir) fn traitref_impl_id(
    v: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> Option<u64> {
    traitref_unwrap(v, llbc, depth)?
        .get("kind")?
        .get("TraitImpl")?
        .get("id")?
        .as_u64()
}

/// Unwrap a `TraitRef`'s `Value` / `Deduplicated` indirections
/// to the underlying trait-ref object.
pub(in crate::front::mir) fn traitref_unwrap<'a>(
    v: &'a serde_json::Value,
    llbc: &'a Llbc,
    depth: usize,
) -> Option<&'a serde_json::Value> {
    if depth > 8 {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        let body = llbc.dedup_trait_body(id)?;
        return traitref_unwrap(body, llbc, depth + 1);
    }
    if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return traitref_unwrap(&arr[1], llbc, depth + 1);
    }
    if let Some(inner) = obj.get("skip_binder") {
        return traitref_unwrap(inner, llbc, depth + 1);
    }
    Some(v)
}

/// Hash-cons identity of a type expression: the `Deduplicated` id or
/// the inline `Value: [id, …]` id.  Two type refs with the
/// same key denote the same monomorphized type.
pub(in crate::front::mir) fn ty_dedup_key(v: &serde_json::Value) -> Option<u64> {
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return Some(id);
    }
    obj.get("Value")?.as_array()?.first()?.as_u64()
}

/// Outcome of devirtualizing a blanket `core::convert::<Impl>::into`
/// callsite — see `Lowering::blanket_into_devirt`.
pub(in crate::front::mir) enum IntoDevirt {
    /// The reflexive `impl<T> From<T> for T` was selected — the call
    /// is a `T -> T` identity conversion.
    Identity,
    /// A concrete `impl From<T> for U` was selected; the segments are
    /// its `from` function's path.
    Target(Vec<String>),
}

/// Render a Charon `DynTrait` body to `dyn <trait-leaf>`.  Falls back to
/// `dyn` when the predicate shape does not expose a resolvable trait id.
pub(in crate::front::mir) fn charon_dyn_trait_to_ast_string(
    dynt: &serde_json::Value,
    llbc: &Llbc,
) -> String {
    // Charon nests the principal trait id a few ways across revisions;
    // scan for the first `{"trait_decl_id": <id>}` (or bare `id`) and
    // resolve it to the trait's leaf name.
    fn find_trait_id(v: &serde_json::Value) -> Option<u64> {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(id) = m.get("trait_decl_id").and_then(serde_json::Value::as_u64) {
                    return Some(id);
                }
                m.values().find_map(find_trait_id)
            }
            serde_json::Value::Array(a) => a.iter().find_map(find_trait_id),
            _ => None,
        }
    }
    match find_trait_id(dynt).and_then(|id| llbc.trait_by_id(id)) {
        Some(td) => {
            let name = td.item_meta.name_path();
            let leaf = name.rsplit("::").next().unwrap_or(&name);
            format!("dyn {leaf}")
        }
        None => "dyn".to_string(),
    }
}

/// Map a Charon `Literal` type body to its Rust spelling.
pub(crate) fn charon_literal_to_ast_string(lit: &serde_json::Value) -> String {
    if let Some(atom) = lit.as_str() {
        return match atom {
            "Bool" => "bool",
            "Char" => "char",
            other => return format!("??lit_atom_{other}"),
        }
        .to_string();
    }
    if let Some(obj) = lit.as_object() {
        if let Some(int) = obj.get("Integer").and_then(serde_json::Value::as_object) {
            if let Some(kind) = int.get("Signed").and_then(serde_json::Value::as_str) {
                return charon_int_kind_to_rust(kind, true);
            }
            if let Some(kind) = int.get("Unsigned").and_then(serde_json::Value::as_str) {
                return charon_int_kind_to_rust(kind, false);
            }
        }
        if let Some(float) = obj.get("Float").and_then(serde_json::Value::as_str) {
            return match float {
                "F16" => "f16",
                "F32" => "f32",
                "F64" => "f64",
                "F128" => "f128",
                other => return format!("??float_{other}"),
            }
            .to_string();
        }
    }
    "??lit".to_string()
}

/// Translate a Charon integer-kind tag (`"I64"`, `"Usize"`, `"U8"`, …)
/// to its Rust spelling.  `signed` disambiguates the `Isize`/`Usize`
/// spelling when the kind tag itself omits the sign.
pub(in crate::front::mir) fn charon_int_kind_to_rust(kind: &str, signed: bool) -> String {
    let lowered = kind.to_ascii_lowercase();
    // Kind tags already carry the leading `i`/`u` for most widths
    // (`I64` -> `i64`, `U8` -> `u8`, `Usize` -> `usize`).  The single
    // `Integer` form may hand back a bare width — fall back to `signed`.
    if lowered.starts_with('i') || lowered.starts_with('u') {
        return lowered;
    }
    if signed {
        format!("i{lowered}")
    } else {
        format!("u{lowered}")
    }
}

/// Render an ADT head's generic type-arguments (`generics.types[]`) to
/// their AST strings, dropping the default allocator / hasher type-args
/// Charon makes explicit (`Vec<T, Global>`, `HashMap<K, V, RandomState,
/// Global>`).  Empty when the head carries no type-arguments.  Shared by
/// [`charon_adt_to_ast_string`] (full type printer) and
/// [`adt_head_instantiation_suffix`] (variant-class projection) so a
/// rendered type and its `<…>` class suffix spell one instantiation
/// identically.
pub(in crate::front::mir) fn render_adt_type_args(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> Vec<String> {
    type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .map(|t| charon_type_value_to_ast_string(t, llbc, depth + 1))
                .filter(|s| s != "Global" && s != "RandomState")
                .collect()
        })
        .unwrap_or_default()
}

/// Payload-row spellings of an ADT head's type arguments.
///
/// Same order and `Global` / `RandomState` filter as
/// [`render_adt_type_args`] (that one still spells the `<…>` class
/// suffix). A transparent GC handle is [`transparent_handle_field_spelling`]
/// so a substituted `Err` payload projects to the cast's class.
pub(in crate::front::mir) fn render_adt_payload_type_args(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<String> {
    type_decl_ref_generics(adt, llbc)
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let rendered = charon_type_value_to_ast_string(t, llbc, depth + 1);
                    if rendered == "Global" || rendered == "RandomState" {
                        return None;
                    }
                    let ty = TyRef::Other(t.clone());
                    Some(
                        transparent_handle_field_spelling(
                            &ty,
                            llbc,
                            no_tombstoned_leaves(),
                            gc_struct_ids,
                            raw_alloc_flavor_structs,
                        )
                        .unwrap_or(rendered),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A non-empty synthetic tuple aggregate's classdef carries a `<T0,T1,…>`
/// suffix rendered from its element types, so `(BigInt,BigInt)` and `(f64,f64)`
/// get distinct classdefs instead of colliding on the shared `__pos_N`
/// attributes of one global `Tuple` class — the `generalize_attr` UnionError
/// that walls every graph mixing tuple element categories (`bigint_truediv`,
/// …) off the two-phase rtyper.

/// Route inline-Field `dyn Trait` virtual calls through the faithful
/// `CallTarget::Indirect` vtable pipeline instead of the synthetic
/// `__dyn_call` residual (see the `(CallClass::Dynamic, ..)` arm).
///
/// `__dyn_call` is not a lowering, it is a placeholder: an unregistered
/// synthetic path that stops whatever graph reaches it.  Routing through
/// `Indirect` is what `jtransform.py`'s `indirect_call` does, so this is the
/// faithful side.  It was gated while the widening it forces — the annotator
/// now annotates every family member's body — outweighed what it bought;
/// census at the flip point, both states, same corpus:
///
/// | gate | records | biggest wall |
/// |---|---:|---|
/// | off | 243 | `__dyn_call` 48 |
/// | on  | **224** | `Wtf8::as_str` 37 |
///
/// Lower a call through a plain function pointer whose PBC family is not
/// recoverable from the LLBC — a host-registered callback table, a settable
/// hook — as `OpKind::IndirectCall` with `graphs: None` instead of the
/// synthetic `__dyn_call` residual.
///
/// `graphs is None` is upstream's own spelling for "cannot follow the
/// indirect call": `call.py` `graphs_from` skips the candidate filter, `call.py`
/// `guess_call_kind` answers `residual`, and `jtransform.py rewrite_op_indirect_call` emits
/// `handle_residual_call`.  `__dyn_call` has no such continuation — it is an
/// unregistered synthetic path that stops the graph reaching it. Before this
/// arm existed only the
/// `BuiltinCodeFn` signature reached `IndirectCall`, so every other
/// fn-pointer call site was a wall; census at the flip point, both states,
/// same artifacts:
///
/// | artifact | `__dyn_call` off→on | `IndirectCall(graphs=None)` off→on |
/// |---|---:|---:|
/// | `cel.ullbc` | 6 → **0** | 0 → 6 |
/// | `pyre-object.ullbc` | 33 → 17 | 0 → 16 |
/// | `pyre-jit.ullbc` | 1 → **0** | 0 → 1 |
/// | `pyre-interpreter.ullbc` | 38 → 12 | 0 → 26 |
///
/// The residue there was exactly the `unsafe fn` pointers (pyre's GC and
/// slot hooks), which the classifier declined outright when that census was
/// taken.  [`operand_fn_ptr_family`] now admits them as
/// `FnPtrFamily::Unknown` — an `unsafe fn` pointer is the same value with
/// the same ABI as a safe one, so `graphs: None` is an honest residual for
/// it; only `BuiltinWrapper` stays closed to it, because that family is
/// matched on signature *shape* and an `unsafe fn` of that shape is not one
/// of its members.  **The `on` column has not been re-measured since**; the
/// `off` column is unaffected by construction, since this arm emits nothing
/// with the switch off.  Nothing else moved in the original census: the
/// vtable-slot calls already on `CallTarget::Indirect` held at 47 / 24, and
/// `pyre-interpreter`'s 203 `BuiltinCodeFn` sites kept the `Some([])`
/// marker in both states.
///
/// OFF by default, and the reason is a pyre-side consequence of being
/// right rather than a defect in the lowering.  `__dyn_call` is an
/// unregistered path, so every effect analyzer classified it as an
/// *external* call and took its bottom result; a `graphs: None`
/// `IndirectCall` is the top result instead
/// (`analyze_random_effects` → `BoolGraphAnalyzer.top_result()`,
/// `graphanalyze.py`).  Turning the arm on therefore propagates
/// `EF_RANDOM_EFFECTS` up through every caller that transitively reaches
/// one of pyre's hook pointers, and `getcalldescr`'s elidable
/// post-condition (`call.py`) then rejects the first
/// `#[elidable_cannot_raise]` helper that does:
///
/// ```text
/// getcalldescr: pyre_object::longobject::jit_bigint_is_zero is marked
/// elidable but got extraeffect=RandomEffects
/// ```
///
/// `cargo build -p pyre-jit-trace --no-default-features --features
/// cranelift` reproduces it with the switch on and is green with it off.
/// Clearing that needs pyre's elidable helpers to stop reaching an
/// unanalyzable callee — or those hooks to carry a real family — which is
/// its own change; until then this stays opt-in with
/// `MAJIT_FNPTR_INDIRECT=1`.  The `BuiltinCodeFn` family is unaffected by
/// the switch: it reached `IndirectCall` before this arm and still does.
pub(crate) fn fnptr_indirect_enabled() -> bool {
    matches!(
        std::env::var("MAJIT_FNPTR_INDIRECT").as_deref(),
        Ok("1") | Ok("true")
    )
}
/// Narrow a `dont_look_inside` residual call whose destination is
/// `Option<*mut PyObject>` from the classdef-less `ref` GCREF token to the
/// per-instantiation `Option` enum root, so the caller's `if let Some(x) =
/// ..` (`__discriminant` read + Some-arm `__pos_0` payload) resolves against
/// the Option classdef instead of panicking with `Ptr{Opaque(GCREF)}
/// instance has no field "__discriminant"`.

/// The per-shape `<…>` suffix for a synthetic tuple whose Charon ADT
/// descriptor is `adt` (`{"id": N, "builtin": "Tuple", "generics": {"types": [..]}}`),
/// or `""` when `adt` is not a tuple, or the tuple is the unit
/// `()` (empty type list — kept bare `"Tuple"` so the unit-value special-cases
/// in `jtransform` / `unit_variant_fold` still match).  Rendered via the same
/// [`render_adt_type_args`] both the construction and the projection derive
/// from the tuple's `place.ty` node (NOT the element-type-less `AggregateKind`
/// head), so the `__pos_N` read and write owners agree.
pub(in crate::front::mir) fn tuple_shape_suffix(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> String {
    if adt.get("builtin").and_then(serde_json::Value::as_str) != Some("Tuple") {
        return String::new();
    }
    let args = render_adt_type_args(adt, llbc, 0);
    if args.is_empty() || args.iter().any(|a| a.contains("??")) {
        return String::new();
    }
    format!("<{}>", args.join(","))
}

/// The per-shape tuple suffix for a place/operand [`TyRef`] — used by BOTH the
/// construction side (the aggregate's destination `place.ty`) and the
/// projection side (the read place's `.ty`), where the tuple type is `ty`'s
/// `{"Adt": {.., "builtin": "Tuple"}}` node.  Keying both off the same `place.ty`
/// source (rather than the element-type-less `AggregateKind` head) makes the
/// `__pos_N` write and read owners agree.  `""` for non-tuple / gate-off /
/// unit; see [`tuple_shape_suffix`].
pub(in crate::front::mir) fn tyref_tuple_suffix(ty: &TyRef, llbc: &Llbc) -> String {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return String::new(),
        },
    };
    value
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(serde_json::Value::as_object)
        .map(|adt| tuple_shape_suffix(adt, llbc))
        .unwrap_or_default()
}

/// Number of type arguments on a Charon `Tuple` Adt, including `()` as 0.
/// A non-tuple is 0, matching [`tyref_tuple_element`] returning `None` at
/// index 0.
pub(in crate::front::mir) fn tyref_tuple_arity(ty: &TyRef, llbc: &Llbc) -> usize {
    let mut n = 0;
    while tyref_tuple_element(ty, n, llbc).is_some() {
        n += 1;
    }
    n
}

/// Element `idx` of a tuple type `ty`, read off its `generics.types`.
pub(in crate::front::mir) fn tyref_tuple_element(
    ty: &TyRef,
    idx: usize,
    llbc: &Llbc,
) -> Option<TyRef> {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let adt = value.as_object()?.get("Adt")?;
    if type_decl_ref_builtin(adt) != Some("Tuple") {
        return None;
    }
    let mut element = adt.get("generics")?.get("types")?.as_array()?.get(idx)?;
    // Follow the dedup table / inline hash-cons indirections to the body.
    loop {
        let obj = element.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            element = llbc.dedup_body(id)?;
        } else if let Some([_, body]) = obj
            .get("HashConsedValue")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
        {
            element = body;
        } else {
            break;
        }
    }
    Some(TyRef::Other(element.clone()))
}

/// The `Tuple<A,B>` shape of a non-unit tuple behind `ty`'s `Ref` layers —
/// the owner a `(*p).N` read of it names (the read place is the deref'd
/// tuple) — or `None` for any other type.
pub(in crate::front::mir) fn tyref_shaped_tuple_root(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let suffix = tyref_tuple_suffix(&TyRef::Other(node.clone()), llbc);
    (!suffix.is_empty()).then(|| format!("Tuple{suffix}"))
}

/// The per-shape suffix for a fixed-size array destination
/// (`{"Array":[item,len]}`), including both the rendered item type and the
/// concrete length. Charon's `AggregateKind::Array` head carries neither, so
/// construction and positional reads derive this spelling from the same
/// destination/place [`TyRef`]. An unresolved const-generic length is kept
/// bare and therefore outside the shaped-array allocation rewrite.
pub(in crate::front::mir) fn tyref_array_suffix(ty: &TyRef, llbc: &Llbc) -> String {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return String::new(),
        },
    };
    let Some(array) = value
        .as_object()
        .and_then(|body| body.get("Array"))
        .and_then(serde_json::Value::as_array)
        .filter(|array| array.len() == 3)
    else {
        return String::new();
    };
    let item = charon_type_value_to_ast_string(&array[0], llbc, 0);
    let len = charon_array_len_to_string(&array[1], llbc);
    if item.is_empty() || len.parse::<usize>().is_err() {
        return String::new();
    }
    format!("<{item};{len}>")
}

/// Per-shape suffix shared by the synthetic positional aggregate constructor
/// and its constant-index `__pos_N` reads.
pub(in crate::front::mir) fn tyref_positional_aggregate_suffix(ty: &TyRef, llbc: &Llbc) -> String {
    let tuple = tyref_tuple_suffix(ty, llbc);
    if tuple.is_empty() {
        tyref_array_suffix(ty, llbc)
    } else {
        tuple
    }
}

/// Exact synthetic owner of a non-empty tuple or fixed-size array type.
///
/// RPython carries a function-return tuple as `SomeTuple` and its rtyper
/// chooses the matching per-shape `TupleRepr`.  Charon's call destination
/// retains that complete element shape even though [`tyref_to_value_type`]
/// only reports the common reference bank, so call-result narrowing uses this
/// owner to preserve the same positional representation across the boundary.
pub(in crate::front::mir) fn tyref_positional_aggregate_root(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let node = tyref_node(ty, llbc)?;
    let suffix = tyref_positional_aggregate_suffix(ty, llbc);
    if suffix.is_empty() {
        return None;
    }
    if node
        .as_object()
        .and_then(|body| body.get("Adt"))
        .and_then(serde_json::Value::as_object)
        .and_then(|adt| adt.get("id"))
        .and_then(serde_json::Value::as_str)
        == Some("Tuple")
        || node
            .as_object()
            .and_then(|body| body.get("Adt"))
            .and_then(serde_json::Value::as_object)
            .and_then(|adt| adt.get("builtin"))
            .and_then(serde_json::Value::as_str)
            == Some("Tuple")
    {
        return Some(format!("Tuple{suffix}"));
    }
    node.as_object()
        .is_some_and(|body| body.contains_key("Array"))
        .then(|| format!("Array{suffix}"))
}

/// The `TypeDeclRef` (`{"id", "generics", …}`) of a TyRef's ADT, after
/// peeling `Deduplicated` / `Value` / `Ref` wrappers.  An aggregate head
/// is often a bare type_id; the destination place still names the
/// instantiated ADT, including when Charon hash-consed it behind a
/// `Value` wrapper.  `None` when `ty` does not resolve to a named ADT.
pub(in crate::front::mir) fn tyref_adt_map<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Map<String, serde_json::Value>> {
    let node = tyref_node(ty, llbc)?;
    strip_ty_wrappers(node, llbc)?
        .as_object()?
        .get("Adt")?
        .as_object()
}

/// The `<X>` enum-instantiation suffix for a destination `Option<X>` /
/// `Result<X, E>` local `ty`.  A runtime-discriminant decomposition
/// (`checked_neg`, `usize::try_from`) constructs the enum ROOT — no static
/// variant annotates the destination — so it cannot route through
/// `resolve_aggregate_adt`'s enum arm; without a suffix it mints the bare
/// template (`core.option.Option`), whose `__pos_0` then unions every
/// caller's payload (`int ∪ FrameDebugData ∪ …`) onto one classdef.
/// Suffixing the root with the `<X>` the `place.ty` carries mints one root
/// per instantiation instead, so each `__pos_0` sees a single concrete
/// payload — the spelling a static `Some(..)`/`Ok(..)` construction already
/// mints.  Extracts `ty`'s `{"Adt": …}` node (as [`tyref_tuple_suffix`]
/// does) and defers to [`adt_head_instantiation_suffix`], already RC2-gated
/// (an `f32`/`()`/`""` argument yields `None`).  Fail-closed: a
/// missing `Adt` node, a non-enum type, or a deferred argument yields `""` —
/// the bare owner, unchanged.
pub(in crate::front::mir) fn tyref_enum_instantiation_suffix(ty: &TyRef, llbc: &Llbc) -> String {
    tyref_adt_map(ty, llbc)
        .and_then(|adt| adt_head_instantiation_suffix(adt, llbc))
        .unwrap_or_default()
}

/// The `<X>` suffix for the single-element closure-`Args` tuple `(payload,)` a
/// unary `map_or`/`and_then`/`map` closure receives, where `payload` is the
/// receiver `Option<X>`'s Some payload.  The extracted `call_once` body reads
/// this tuple's `.0` via a real Charon `Tuple<X>` container (owner derived at
/// [`Lowering::resolve_place`] through [`tyref_tuple_suffix`]); the
/// synthesized WRITE must derive the identical leaf so the `__pos_0` write and
/// read key under one classdef.  Both terminate in `render_adt_type_args` →
/// `charon_type_value_to_ast_string` (which strips `&T → T`) on the SAME
/// payload node, so the strings are equal by construction.  Sourced from
/// `recv_ty` (`Option<X>`, un-erased) — NOT `site.payload_ty`, which is
/// `Ref(None)` erased.  Fail-closed: a non-Option recv, a missing payload node,
/// or a deferred arg yields "" (bare `Tuple`, unchanged).
pub(in crate::front::mir) fn option_payload_tuple_suffix(recv_ty: &TyRef, llbc: &Llbc) -> String {
    // 1. pull recv_ty.Adt.generics.types[0] exactly as
    //    tyref_option_payload_value_type does.
    let body = match recv_ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return String::new(),
        },
    };
    let Some(node) = body
        .get("Adt")
        .and_then(|a| a.as_object())
        .and_then(|a| type_decl_ref_generics(a, llbc))
        .and_then(|g| g.get("types"))
        .and_then(|t| t.get(0))
    else {
        return String::new();
    };
    // 2. synthesize wrapper {"Adt":{"builtin":"Tuple","generics":{"types":[node]}}}.
    let wrapper = serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": { "types": [node.clone()] }
        }
    });
    // 3. return tyref_tuple_suffix of the wrapper — the same renderer the read
    //    side routes through, so the `&`-stripped leaf matches by construction.
    tyref_tuple_suffix(&TyRef::Other(wrapper), llbc)
}

/// The closure-argument tuple spelling for an already projected enum payload.
/// Both the synthesized writer and Charon's closure `call_once` reader use
/// `tyref_tuple_suffix`, so reference erasure and generic rendering stay
/// identical on the two sides.
pub(in crate::front::mir) fn payload_tuple_suffix(payload: &TyRef, llbc: &Llbc) -> String {
    let node = match payload {
        TyRef::Inline { value: (_, value) } | TyRef::Other(value) => value.clone(),
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(value) => value.clone(),
            None => return String::new(),
        },
    };
    let wrapper = serde_json::json!({
        "Adt": {
            "id": 0, "builtin": "Tuple",
            "generics": { "types": [node] }
        }
    });
    tyref_tuple_suffix(&TyRef::Other(wrapper), llbc)
}

/// Preserve a reference payload's concrete class across a synthesized enum
/// and closure-tuple field write.  RPython derives the field repr from that
/// class annotation (`rclass.py:InstanceRepr._setup_repr`); the register-bank
/// [`ValueType`] alone cannot distinguish two reference classes.
pub(in crate::front::mir) fn enum_payload_instance_class_root(
    payload: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Option<String> {
    let payload = strip_ty_indirections(tyref_node(payload, llbc)?, llbc)?;
    if let Some(root) = raw_ptr_pointee_class_root_with(payload, llbc, tombstoned) {
        return Some(root);
    }
    if let Some(root) = adt_node_class_root_with(payload, llbc, tombstoned) {
        return Some(root);
    }
    let pointee = payload.as_object()?.get("Ref")?.as_array()?.get(1)?;
    let pointee = strip_ty_wrappers(pointee, llbc)?;
    raw_ptr_pointee_class_root_with(pointee, llbc, tombstoned)
        .or_else(|| adt_node_class_root_with(pointee, llbc, tombstoned))
}

/// A type-argument drives a per-instantiation variant-class split unless
/// it is in the deferred-or-degenerate set below.  A split mints a
/// distinct `<…>`-suffixed variant classdef (`Result<bool>::Ok` vs
/// `Result<i64>::Ok` vs `Result<Tuple>::Ok`) whose payload field sees only
/// that one concrete type, so the field annotates to that type's own repr
/// (`BoolRepr`, `IntegerRepr`, the GC-pointer `InstanceRepr`).  Without the
/// split every instantiation collapses onto one bare variant class whose
/// `__pos_0` unions the whole program — a cross-category `int ∪ float ∪
/// char ∪ ()` merge that generalises to a generic `InstanceRepr` and walls
/// the rtyper on the materialised store.
///
/// Reference payloads (a heap class `Tuple`/`W_Object`/`str`, a nested
/// generic enum `Option<…>`, a tuple of references) are one word-sized GC
/// pointer; the scalar integer/bool/char primitives are unboxed but each
/// carries a well-defined int-banked repr, so all of these split.
/// An `f64` payload splits too: the per-instantiation row records `f64`, and
/// the codewriter's `get_type_flag`/`type_flag_from_str` selects the float
/// bank, producing `setfield_gc_f`/`getfield_gc_f` rather than the generic
/// GC-word fallback.  Excluded:
///   - `f32` — RPython's `SingleFloat` is int-banked (`getkind(...) == "int"`),
///     while the textual field-descriptor path still classifies float atoms
///     as float-banked. Defer the split until both paths agree.
///   - `()` — a unit-only generic (`Option<()>`, `Result<(), ()>`) has
///     no materialised Ok payload (`widen_unit_return_to_void`). A mixed
///     `Result<(), E>` still suffixes when `E` splits, so the Err
///     payload does not share a ClassDef with a sibling `E`
///     (`native_result` `map_err`, `exceptiontransform.py` unit_ok).
///   - `""` — a degenerate/absent type-arg that would render a malformed
///     `<>` suffix.
pub(in crate::front::mir) fn type_arg_splits_per_instantiation(arg: &str) -> bool {
    const DEFERRED: &[&str] = &["f32", "()", ""];
    if DEFERRED.contains(&arg) {
        return false;
    }
    // Unsubstituted parameters are not an instantiation.
    // `RangeInclusive::new` keeps `types: [TypeVar]` on the Aggregate
    // head; splitting on that interned one class for every spec.
    if arg.starts_with("TypeVar") || arg.starts_with("??") {
        return false;
    }
    true
}

/// True when `arg` names a fully substituted type: no `TypeVar` / `??`
/// anywhere, nested arguments included. Distinct from
/// [`type_arg_splits_per_instantiation`]: `()` and `f32` are concrete
/// (`Bookkeeper.getuniqueclassdef`) and still do not split.
pub(in crate::front::mir) fn type_arg_is_concrete(arg: &str) -> bool {
    !arg.contains("TypeVar") && !arg.contains("??")
}

pub(in crate::front::mir) fn adt_type_args_are_concrete(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> bool {
    render_adt_type_args(adt, llbc, 0)
        .iter()
        .all(|arg| type_arg_is_concrete(arg))
}

/// Prefer a destination TypeDeclRef whose arguments are fully
/// substituted (`Result<(), E>` is concrete even though `()` does not
/// split), then a concrete head, then the head. A still-open `TypeVar`
/// is not concrete (`RangeInclusive::new` Aggregate generics).
/// `Bookkeeper.getuniqueclassdef`.
pub(in crate::front::mir) fn concrete_instantiation_adt<'a>(
    head_adt: Option<&'a serde_json::Map<String, serde_json::Value>>,
    dest_adt: Option<&'a serde_json::Map<String, serde_json::Value>>,
    llbc: &Llbc,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    let concrete = |adt: Option<&'a serde_json::Map<String, serde_json::Value>>| {
        adt.filter(|h| adt_type_args_are_concrete(h, llbc))
    };
    concrete(dest_adt)
        .or_else(|| concrete(head_adt))
        .or(head_adt)
}

/// `<…>` spelling for an enum instantiation, or `None` when no type
/// argument justifies a split. Deferred atoms (`()`, `f32`) stay in the
/// spelling so `Result<(), PyError>` and `Result<(), (i32, String)>`
/// do not share `Result::Err.__pos_0` (`native_result` Tuple ∪ OsError).
pub(in crate::front::mir) fn enum_instantiation_suffix_from_args(
    type_args: &[String],
) -> Option<String> {
    if type_args.is_empty()
        || !type_args
            .iter()
            .any(|a| type_arg_splits_per_instantiation(a))
    {
        return None;
    }
    Some(format!("<{}>", type_args.join(",")))
}

/// The `<…>` generic-argument suffix of an `AggregateKind::Adt` /
/// type-node head (`<Tuple>` for a `Result<Tuple, PyError>`
/// instantiation), or `None` when the head is not a reference-payload
/// `enum` instantiation.  Returned `Some` only when the head names an
/// `enum` (struct generics keep collapsing to their flat classdef) and
/// [`enum_instantiation_suffix_from_args`] accepts the rendered
/// type-arguments.  Every projection site (receiver type,
/// variant constructor, variant field read, numbering pre-scan) routes
/// through this one predicate so they all agree on which instantiations
/// split and how they spell the suffix.
///
/// `core`/`std`/`alloc` is NOT excluded here: the variant classes that
/// collide are `Result::Ok` / `Option::Some`, minted by the constructor
/// path, so the split must reach `core::result::Result` /
/// `core::option::Option`.  The receiver-type projection
/// [`adt_node_class_root_with`] keeps `Vec<T>` / `Box<T>` on their
/// annotator models, and now paints split-eligible core enums with this
/// same suffix so a returned `Option<usize>` annotates as that class.
pub(crate) fn adt_head_instantiation_suffix(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    let def_id = type_decl_ref_adt_id(adt)?;
    let td = llbc.type_by_id(def_id)?;
    if !matches!(td.kind, TypeDeclKind::Enum(_)) {
        return None;
    }
    enum_instantiation_suffix_from_args(&render_adt_type_args(adt, llbc, 0))
}

/// Format a Charon `Adt` type body (`{"id": …, "generics": {"types": […]}}`).
pub(in crate::front::mir) fn charon_adt_to_ast_string(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> String {
    let id = adt.get("id");
    let type_args: Vec<String> = render_adt_type_args(adt, llbc, depth);
    // `id` is a `TypeDecl` id. `builtin` names a compiler ADT (`Tuple`,
    // `Box`, `Str`) when the declaration is not a user type.
    if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Tuple") {
        return if type_args.is_empty() {
            "()".to_string()
        } else {
            format!("({})", type_args.join(","))
        };
    }
    if let Some(builtin) = adt.get("builtin").filter(|b| !b.is_null()) {
        return charon_builtin_adt_to_ast_string(builtin, &type_args);
    }
    if let Some(def_id) = id.and_then(serde_json::Value::as_u64) {
        let name = llbc
            .type_by_id(def_id)
            .map(|td| td.item_meta.name_path())
            .unwrap_or_else(|| format!("??adt#{def_id}"));
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        if type_args.is_empty() {
            return leaf;
        }
        // Keep supplied type arguments (`bookkeeper.py`
        // `getuniqueclassdef`). An omitted-default spelling is only
        // valid when Charon carries the declared default and the
        // argument equals it.
        return format!("{leaf}<{}>", type_args.join(","));
    }
    // A field type copied from another crate names the ADT by path. The
    // leaf is the same spelling the defining crate's numeric id rendered.
    if let Some(path) = id.and_then(|i| i.get("Path")).and_then(|p| p.as_str()) {
        let leaf = path.rsplit("::").next().unwrap_or(path);
        if type_args.is_empty() {
            return leaf.to_string();
        }
        return format!("{leaf}<{}>", type_args.join(","));
    }
    let key = id
        .and_then(|i| i.as_object())
        .and_then(|m| m.keys().next().cloned())
        .or_else(|| id.and_then(|i| i.as_str()).map(str::to_string))
        .unwrap_or_else(|| "?".into());
    format!("??adt_id_{key}")
}

/// Format a Charon builtin ADT id (`Box`/`Slice`/`Str`/`Array`).
pub(in crate::front::mir) fn charon_builtin_adt_to_ast_string(
    builtin: &serde_json::Value,
    type_args: &[String],
) -> String {
    let name = builtin
        .as_str()
        .or_else(|| {
            builtin
                .as_object()
                .and_then(|m| m.keys().next().map(String::as_str))
        })
        .unwrap_or("?");
    match name {
        // Charon's `Box` builtin maps to the `Box<T>` spelling.
        "Box" => match type_args.first() {
            Some(inner) => format!("Box<{inner}>"),
            None => "Box".to_string(),
        },
        "Str" => "str".to_string(),
        "Slice" => match type_args.first() {
            Some(inner) => format!("[{inner}]"),
            None => "[]".to_string(),
        },
        other => format!("??builtin_{other}"),
    }
}

/// Best-effort render of an array length.
/// `TyKind::Array`'s length: a `ConstantExpr` `[literal, ty]`, inline or
/// hash-consed in the constant table. Renders the decimal `N` of `[T; N]`.
pub(crate) fn charon_array_len_to_string(len: &serde_json::Value, llbc: &Llbc) -> String {
    let body = if let Some(id) = len.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        match llbc.dedup_const_body(id) {
            Some(body) => body,
            None => return format!("??unresolved_len#{id}"),
        }
    } else if let Some(body) = len
        .get("Value")
        .and_then(serde_json::Value::as_array)
        .and_then(|pair| pair.get(1))
    {
        body
    } else {
        len
    };
    body.as_array()
        .and_then(|pair| pair.first())
        .and_then(|lit| lit.get("Integer"))
        .and_then(serde_json::Value::as_object)
        .and_then(|int| int.values().next())
        .and_then(serde_json::Value::as_array)
        .and_then(|parts| parts.get(1))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("??len:{body}"))
}

/// The by-value nested struct leaves [`Lowering::flatten_list_storage_field`]
/// names `nested.leaf` on their GC owner: `W_ListObject`'s inline
/// Integer/Float/Bytes/Ascii storage `block` and `len`. writeanalyze reads
/// the same set, so an effect names a dotted leaf exactly when the trace's
/// field descr is that leaf.
pub(crate) fn is_flattened_storage_leaf(owner: &str, nested: &str, leaf: &str) -> bool {
    matches!(leaf, "block" | "len")
        && matches!(
            nested,
            "int_items" | "float_items" | "bytes_items" | "ascii_items"
        )
        && owner.rsplit("::").next() == Some("W_ListObject")
}

/// Drop Deref wrappers and the by-value storage field (`int_items` /
/// `float_items` / …) so the write base is the list pointer. Pair of
/// [`Lowering::flatten_list_storage_field`].
pub(in crate::front::mir) fn peel_nested_storage_place(inner: Place) -> Place {
    let mut cur = inner;
    loop {
        match cur.kind {
            PlaceKind::Projection(next, ProjectionElem::Atom(s)) if s == "Deref" => {
                cur = *next;
            }
            PlaceKind::Projection(next, ProjectionElem::Tagged(_)) => return *next,
            other => {
                cur.kind = other;
                return cur;
            }
        }
    }
}

/// Stable short label for an [`Rvalue::Aggregate`]'s [`Field`]
/// projection payload. Charon encodes `Field` as `[{"Adt"|"Tuple": ...}, idx]`,
/// where `idx` is the field's position. We project to
/// `<container>_<idx>` so synthetic FieldDescriptors stay readable.
pub(in crate::front::mir) fn field_label_from_payload(payload: &serde_json::Value) -> String {
    if let Some(arr) = payload.as_array()
        && arr.len() == 2
    {
        let container = arr[0]
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .unwrap_or_else(|| "Field".into());
        let idx = arr[1].as_u64().unwrap_or(u64::MAX);
        return format!("{container}_{idx}");
    }
    "field".into()
}

/// Return `(trait_leaf_ident, method_leaf_ident)` when the FunDecl's
/// raw `NameSeg` vec ends in two consecutive `Ident` segments — the
/// Charon shape for a trait method declaration (e.g.
/// `pyre_interpreter::shared_opcode::SharedOpcodeHandler::push_value`).
/// The penultimate Ident is the trait name, the leaf the method
/// name.
///
/// Distinct from [`Lowering::impl_method_owner`], which looks for an
/// `Impl` `NameSeg::Other` segment preceding the leaf — that arm
/// fires for inherent / trait-impl methods Charon already resolved
/// at extraction time.  Trait method declarations have no `Impl`
/// segment because the body is the trait's default impl.
///
/// Used by the `CallKind::Trait` arm of
/// [`Lowering::call_target_segments`] to emit
/// `CallTarget::FunctionPath { segments: [trait_leaf, method_leaf], specialization: crate::parse::GraphSpecialization::Regular}`,
/// matching the direct-path key
/// `register_function_graph(direct_path, …)` at `lib.rs`
/// (`extract_trait_impls`'s `<default methods of <Trait>>` branch).
pub(in crate::front::mir) fn trait_method_owner(fd: &FunDecl) -> Option<(String, String)> {
    let segs = &fd.item_meta.name;
    if segs.len() < 2 {
        return None;
    }
    let leaf = match segs.last()? {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    let parent = match &segs[segs.len() - 2] {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    Some((parent, leaf))
}

/// Whether the call's trait reference leaves its implementation to be
/// selected at run time.
///
/// Charon records where the reference came from in its `kind`: `Dyn` for a
/// trait object, whose vtable is the only thing that names the target, as
/// against `Clause` (a bound on the enclosing generic, fixed by whoever
/// instantiated it), `TraitImpl` (an impl named outright) and `BuiltinOrAuto`.
/// Only the first is a dispatch; each of the others stands for exactly one
/// implementation, and a family of one has nothing to select.
///
/// `rpbc.py` splits the same two cases on whether the row's function
/// came out constant — `if isinstance(vlist[0], Constant): v =
/// hop.genop('direct_call', vlist, ...)`, and only the `else` arm appends the
/// row of graphs and emits `indirect_call`.
///
/// The reference arrives either inline or hash-consed: Charon writes
/// `Value: [id, body]` at the first occurrence and `Deduplicated:
/// id` afterwards. [`traitref_unwrap`] resolves both, and resolves them in a
/// loop — either spelling can nest inside the other, so peeling one of each
/// in a fixed order would leave a wrapper standing and make the `kind` read
/// miss. Missing it declines the call, which is safe but would make the
/// classification depend on how Charon happened to serialize the reference.
pub(in crate::front::mir) fn trait_ref_is_dyn(trait_ref: &serde_json::Value, llbc: &Llbc) -> bool {
    let Some(resolved) = traitref_unwrap(trait_ref, llbc, 0) else {
        return false;
    };
    resolved.get("kind").and_then(serde_json::Value::as_str) == Some("Dyn")
}

/// Resolve a call on a trait object as dynamic dispatch.
///
/// Charon keeps a call such as `self.head()` inside a trait default method as
/// `CallKind::Trait` pointing at the declaration of `head`. When the receiver
/// is a trait object and that declaration has no body, the direct
/// `[Trait, method]` path has nothing to execute; the receiver's
/// implementation family must select the target.
///
/// A reference the caller's instantiation already resolved is excluded, even
/// though its declaration is equally bodyless: the implementation is unique,
/// so the direct path names the impl's own graph and the call stays inside
/// the inline closure. See [`trait_ref_is_dyn`].
///
/// A bodyless declaration that takes no receiver is excluded too. `rpbc.py`
/// dispatches a methods-PBC on the instance the call is bound to, and the
/// lowering below reads that instance off the call's first argument; an
/// associated function has none, and the type it would be selected by is
/// fixed at the call site rather than carried by a value. Classifying one
/// here hands the lowering an argument list whose first entry is either
/// absent or an ordinary parameter standing in for a receiver.
pub(in crate::front::mir) fn abstract_trait_call_target(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<(String, String)> {
    let CallKind::Trait(value) = &reg.kind else {
        return None;
    };
    let payload = value.as_array()?;
    if !trait_ref_is_dyn(payload.first()?, llbc) {
        return None;
    }
    let (trait_path, method_leaf) = trait_payload_path_and_method(value, llbc)?;
    let decl_id = trait_ref_decl_id(payload.first()?, llbc)?;
    // A `--monomorphize` trait declaration carries no method rows, so there
    // is no signature to read. The receiver is then guaranteed by the
    // reference kind: a method without `self` is callable on a trait object
    // only under `where Self: Sized`, which a `dyn Trait` never satisfies.
    if let Some(method) = llbc
        .trait_by_id(decl_id)?
        .methods
        .get(payload.get(1)?.as_u64()? as usize)
    {
        let inputs = method.pointer("/skip_binder/signature/inputs")?;
        if inputs.as_array().is_none_or(|rows| rows.is_empty()) {
            return None;
        }
    }
    let default_method_path = format!("{trait_path}::{method_leaf}");
    let has_body = llbc.iter_local_fns().any(|fd| {
        let path = fd.item_meta.name_path();
        (path == default_method_path || path.ends_with(&format!("::{default_method_path}")))
            && fd.has_unstructured_body()
    });
    if has_body {
        return None;
    }
    let trait_leaf = trait_path.rsplit("::").next()?.to_string();
    Some((trait_leaf, method_leaf))
}

/// Return the static graph path for an inherent method on a scalar type.
///
/// Its receiver is a scalar in the low-level model, so routing through
/// `CallTarget::Method` would perform an attribute lookup on an integer. The
/// Rust call is statically resolved already; use the registered impl graph
/// directly and keep the scalar receiver as its first argument.
///
/// This covers both `#[repr(transparent)]` wrappers and payload-free enums.
/// The latter are represented by their integer discriminant in the translated
/// graph.  PyPy's corresponding buffer requests are integer `BUF_*` flags
/// (`pypy/interpreter/baseobjspace.py`); Rust packages the same
/// decisions as inherent helpers on a scalar tag.  Those helpers remain
/// ordinary statically selected functions, not Python method lookups on the
/// integer.
pub(in crate::front::mir) fn scalar_inherent_method_path(
    reg: &RegularCall,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<Vec<String>> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let declaration = llbc.fn_by_id(*id)?;
    let (owner, method) = impl_method_owner_for_fundecl(llbc, declaration)?;
    let receiver = declaration.signature.inputs.first()?;
    let node = strip_ty_wrappers(tyref_node(receiver, llbc)?, llbc)?;
    let owner_id = adt_node_def_id(node)?;
    let owner_decl = llbc.type_by_id(owner_id)?;
    let is_payload_free_enum = matches!(
        &owner_decl.kind,
        TypeDeclKind::Enum(variants) if variants.iter().all(|variant| variant.fields.is_empty())
    );
    if !owner_decl.is_repr_transparent() && !is_payload_free_enum {
        return None;
    }
    // A transparent GC handle is a Ref instance (`gc_ref_pointer`), not
    // a scalar. This intercept would emit a FunctionPath without a
    // fun-decl id, so `install_gc_mut_ref_call` could not wrap `&mut
    // self`, and `&self` would miss MethodDesc.func_args.
    if gc_ref_pointer(
        &TyRef::Other(node.clone()),
        llbc,
        tombstoned,
        gc_struct_ids,
        0,
        raw_alloc_flavor_structs,
    ) {
        return None;
    }
    let mut path = crate::model::split_qualified_path(&owner);
    path.push(instance_leaf(llbc, declaration).unwrap_or(method));
    Some(path)
}

/// Compact identifier for a `CallKind::Trait` payload — the triple
/// `[trait_ref, method_idx, decl_id]`. We project to
/// `trait<decl_id>::m<method_idx>` so the synthesised path is small
/// and deterministic.  Falls back to `unknown` if the shape is
/// unexpected; callers should fail-loud on `unknown` if downstream
/// dispatch needs the actual impl.
pub(in crate::front::mir) fn trait_call_label(v: &serde_json::Value) -> String {
    if let Some(arr) = v.as_array() {
        let method_idx = arr.get(1).and_then(Value::as_u64).unwrap_or(u64::MAX);
        return format!("m{method_idx}");
    }
    "unknown".to_string()
}

/// `(full trait name path, method leaf)` from
/// `CallKind::Trait([trait_ref, method_idx])`.
pub(in crate::front::mir) fn trait_payload_path_and_method(
    v: &serde_json::Value,
    llbc: &Llbc,
) -> Option<(String, String)> {
    let arr = v.as_array()?;
    let decl_id = trait_ref_decl_id(arr.first()?, llbc)?;
    let method_leaf = llbc
        .trait_method_name(decl_id, arr.get(1)?.as_u64()?)?
        .to_string();
    let trait_path = llbc.trait_by_id(decl_id)?.item_meta.name_path();
    Some((trait_path, method_leaf))
}

/// `Vec<T>` as trait `Clone::clone` → the `list` builtin
/// (`rtype_bltn_list` / `rlist.py` `ll_copy`) when T's clone is the
/// identity copy. `ll_arraycopy` never calls `T::clone`, so a
/// non-identity element (`Vec<Vec<U>>`, `Vec<String>`) stays residual.
pub(in crate::front::mir) fn specialized_trait_clone_list_target(
    v: &serde_json::Value,
    llbc: &Llbc,
) -> Option<(Vec<String>, Option<(String, String, Option<String>)>)> {
    if !is_trait_clone_clone(v, llbc) {
        return None;
    }
    let node = trait_clone_self_node(v, llbc)?;
    let self_ty = serde_json::from_value::<TyRef>(node.clone()).ok()?;
    tyref_vec_clone_is_ll_copy(&self_ty, llbc).then(|| (vec!["list".to_string()], None))
}

/// `CallKind::Trait` payload of `core::clone::Clone::clone`.
pub(in crate::front::mir) fn is_trait_clone_clone(v: &serde_json::Value, llbc: &Llbc) -> bool {
    if let Some((trait_path, method_leaf)) = trait_payload_path_and_method(v, llbc)
        && trait_path == "core::clone::Clone"
        && method_leaf == "clone"
    {
        return true;
    }
    let Some(arr) = v.as_array() else {
        return false;
    };
    if arr.get(1).and_then(serde_json::Value::as_u64) != Some(0) {
        return false;
    }
    traitref_unwrap(arr.first().unwrap_or(v), llbc, 0)
        .and_then(|tref| tref.pointer("/kind/BuiltinOrAuto/builtin_data"))
        .and_then(serde_json::Value::as_str)
        == Some("Clone")
}

/// Owned `Self` of a trait `Clone::clone` obligation, from the trait-ref
/// generic `types[0]`. A leading `&` is peeled so a tuple `Self` is the
/// tuple, not the borrow (`tyref_is_tuple` does not peel refs).
pub(in crate::front::mir) fn trait_clone_self_node<'l>(
    v: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    let arr = v.as_array()?;
    let tref = traitref_unwrap(arr.first()?, llbc, 0)?;
    let ty = crate::front::clause_spec::trait_decl_pred(tref)?
        .pointer("/generics/types")?
        .as_array()?
        .first()?;
    let node = strip_ty_indirections(ty, llbc)?;
    if let Some(pointee) = node
        .as_object()
        .and_then(|o| o.get("Ref"))
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.get(1))
    {
        strip_ty_indirections(pointee, llbc)
    } else {
        Some(node)
    }
}

pub(in crate::front::mir) fn trait_ref_decl_id(
    trait_ref: &serde_json::Value,
    llbc: &Llbc,
) -> Option<u64> {
    let resolved = traitref_unwrap(trait_ref, llbc, 0)?;
    crate::front::clause_spec::trait_decl_pred(resolved)
        .and_then(|pred| pred.get("id"))
        .and_then(serde_json::Value::as_u64)
}

/// Strip the leading crate-name segment from a Charon `name_path()`.
/// Charon prefixes every fully-qualified path with the crate name
/// (`pyre_interpreter::frame::eval_loop_jit`); functions are named
/// relative to their module root instead (`frame::eval_loop_jit` for a
/// non-empty `module_path`, or the bare leaf for `module_path == ""`)
/// so `register_function_graph_alias` (lib.rs) can walk
/// `{bare, crate::*, pyre_interpreter::*, pyre_object::*, jit_artifact::*}`
/// aliases off the same `func.name`.
pub(crate) fn strip_crate_prefix(path: &str) -> String {
    match path.split_once("::") {
        Some((_crate, rest)) => rest.to_string(),
        // single-segment name (rare — top-level item without crate
        // prefix in some Charon outputs): leave as-is.
        None => path.to_string(),
    }
}

/// True for the two REFERENCE items-base accessors — the ones whose block
/// lays `Ptr` items after its length header.
///
/// Split out of [`graph_is_items_block_base_accessor`] rather than spelled
/// twice, because brick 1 (the accessor-body collapse) and brick 3 (the
/// element `*base.add(i)`) must recognise the same accessor or the pointer
/// means different things at the two ends: brick 1 rewrites the accessor to
/// return the block HEADER, and only brick 3's `base_size`-bearing array descr
/// adds the item offset back. An accessor collapsed by the first and declined
/// by the second leaves a residual `.add` striding from the LENGTH WORD — a
/// silent one-element mis-index, not a missed optimisation. One list, two
/// call sites.
///
/// Matched on the module-qualified path so a leaf collision in another module
/// cannot widen the gate, and crate-prefix-independent so the same accessor
/// matches whether reached from `pyre_object`, `pyre_interpreter`, `jit_artifact`'s
/// monomorphized copy — or a non-pyre host crate that lays its block out this
/// way. The prefix carries no information the suffix does not: what makes the
/// collapse sound is the accessor's CONTRACT (return the interior pointer for
/// descr-based consumption), which the module-qualified name states and the
/// crate name does not.
///
/// Compared with [`path_ends_with_segments`] and not with `str::ends_with`. On
/// a `::`-joined path `ends_with` is a SUBSTRING test, not a path test: it
/// accepts `some_host::my_object_array::items_block_items_base`, because
/// `object_array::…` is a byte suffix of `my_object_array::…` while naming a
/// different module. A module that merely *ends in* `object_array` would then
/// have its accessor collapsed to the block header on the strength of its
/// spelling. The segment-aware comparator already existed in this file; these
/// gates were the ones not using it.
pub(in crate::front::mir) fn is_object_items_block_base_accessor(name: &str) -> bool {
    path_ends_with_segments(name, "object_array::items_block_items_base")
        || path_ends_with_segments(name, "object_array::items_block_items_ptr")
        || path_ends_with_segments(name, "tupleobject::tuple_items_base")
}

pub(in crate::front::mir) fn is_typed_items_block_base_accessor(name: &str) -> bool {
    path_ends_with_segments(name, "rlist::typed_items_block_items_base")
}

/// A call to an accessor whose body brick 1 collapses to its receiver: the
/// translated result is the block argument itself (`l.ll_items()` is
/// `l.items`).
pub(crate) fn items_block_accessor_returns_its_block(name: &str) -> bool {
    is_object_items_block_base_accessor(name) || is_typed_items_block_base_accessor(name)
}

/// Scalar storage adapters for PyPy's `GcArray(Signed|Float)` list items.
/// These spell the physical header offset locally because their empty Rust
/// arrays need a non-null zero-length-slice address; for every live element
/// access the block is non-null and the translated value is still `l.items`.
pub(in crate::front::mir) fn is_typed_array_base_adapter(name: &str) -> bool {
    path_ends_with_segments(name, "int_array::<Impl>::base")
        || path_ends_with_segments(name, "float_array::<Impl>::base")
}

pub(in crate::front::mir) fn is_string_array_base_adapter(name: &str) -> bool {
    path_ends_with_segments(name, "bytes_array::<Impl>::base")
        || path_ends_with_segments(name, "unicode_array::<Impl>::base")
}

/// `BytesArray` / `UnicodeArray` methods share `items_block_items_base`
/// with object lists, but their items are `SomeString` (`BytesListStrategy`
/// / `AsciiListStrategy`, `externalvsinternal(..., gcref=True)`). Narrowing
/// that accessor to `[PyObject]` there unions `Instance(PyObject)` with
/// `String` at `ll_setitem_fast`. Brick 3 already recovers `[str]` for
/// those graphs (`string_items_elem_ptr_add`).
pub(in crate::front::mir) fn graph_is_string_items_owner(name: &str) -> bool {
    name.split("::")
        .any(|seg| matches!(path_segment_ident(seg), "bytes_array" | "unicode_array"))
}

/// True for the `ItemsBlock` / `TypedItemsBlock` items-base accessor bodies
/// whose `.add(*_ITEMS_OFFSET)` the front-end collapses to the
/// receiver (see [`Lowering::is_items_block_base_ptr_add`]).
///
/// The typed (scalar) accessor is here and NOT in
/// [`is_object_items_block_base_accessor`]: its block holds `u64` words, so a
/// `*base.add(i)` through it is not a reference element store and brick 3 must
/// not lower it as one. That separation is what keeps the shared-list
/// invariant above from pulling a scalar block into the `Ref`-typed array ops
/// — the element side is refused by [`is_object_ref_items_ptr`] as well, so the
/// two gates agree by construction rather than by coincidence.
/// The `is_string_array_base_adapter` arm is currently INERT through the sole
/// consumer [`Lowering::is_items_block_base_ptr_add`], which also requires a
/// `ptr::add` / `wrapping_add` in the same body: `BytesArray::base` /
/// `UnicodeArray::base` are `items_block_items_base(self.block)`, a call, and
/// carry no `.add` to collapse.  It is kept so the predicate answers the
/// question its name asks for the whole family, and it becomes live the
/// moment either body spells the offset itself — but do not read a passing
/// `offset_leaf` assertion over those two graphs as evidence that it fired.
pub(in crate::front::mir) fn graph_is_items_block_base_accessor(name: &str) -> bool {
    is_object_items_block_base_accessor(name)
        || is_typed_items_block_base_accessor(name)
        || is_typed_array_base_adapter(name)
        || is_string_array_base_adapter(name)
}

/// One path segment, with a raw-identifier prefix removed.  `r#struct` and
/// `struct` name the same module: rustc's canonical `DefPath` ident (what
/// Charon records, and therefore what a `name_path()` carries) drops the
/// `r#`, while a path spelled by `module_path!()` on the host side keeps it.
pub(in crate::front::mir) fn path_segment_ident(seg: &str) -> &str {
    seg.strip_prefix("r#").unwrap_or(seg)
}

/// Whole-path equality, segment by segment, ignoring raw-identifier
/// prefixes on either side.
pub(in crate::front::mir) fn path_eq_ignoring_raw(a: &str, b: &str) -> bool {
    let mut a = a.split("::");
    let mut b = b.split("::");
    loop {
        match (a.next(), b.next()) {
            (None, None) => return true,
            (Some(x), Some(y)) if path_segment_ident(x) == path_segment_ident(y) => {}
            _ => return false,
        }
    }
}

/// `key` is a proper `::`-boundary suffix of `path`, compared the same way.
pub(in crate::front::mir) fn path_has_suffix_ignoring_raw(path: &str, key: &str) -> bool {
    let mut path = path.rsplit("::");
    let mut key = key.rsplit("::");
    loop {
        let Some(k) = key.next() else {
            // Key exhausted at a segment boundary; require at least one
            // more segment so this stays a *proper* suffix (whole-path
            // equality is the caller's other arm).
            return path.next().is_some();
        };
        match path.next() {
            Some(p) if path_segment_ident(p) == path_segment_ident(k) => {}
            _ => return false,
        }
    }
}

/// `key` names the trailing `::` segments of `path` — as a PATH, not as a
/// substring. The drop-in for `path.ends_with(key)` wherever `key` is a
/// `::`-joined path rather than a bare leaf.
///
/// Both arms are needed. [`path_has_suffix_ignoring_raw`] deliberately demands
/// a *proper* suffix — at least one segment of `path` left over — so on its own
/// it would reject a `path` that is exactly `key`, which `ends_with` accepted.
/// Pairing it with [`path_eq_ignoring_raw`] leaves the substring collision as
/// the only input whose answer changes.
pub(in crate::front::mir) fn path_ends_with_segments(path: &str, key: &str) -> bool {
    path_eq_ignoring_raw(path, key) || path_has_suffix_ignoring_raw(path, key)
}

pub(in crate::front::mir) fn value_type_is_int_bank(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Int | ValueType::Unsigned)
}

/// `rlib/longlong2float.py` `float2longlong` / `longlong2float`.
/// Trailing segments are exactly `longlong2float::{float2longlong,longlong2float}`
/// and the banks are f64↔i64; a look-alike module is not this pair.
pub(in crate::front::mir) fn host_longlong2float_llop(
    path: &str,
    src: Option<&ValueType>,
    dst: &ValueType,
) -> Option<bool> {
    let to_float = if path_ends_with_segments(path, "longlong2float::longlong2float") {
        true
    } else if path_ends_with_segments(path, "longlong2float::float2longlong") {
        false
    } else {
        return None;
    };
    let src_ok = match src {
        Some(ValueType::Int | ValueType::Unsigned) if to_float => true,
        Some(ValueType::Float) if !to_float => true,
        None => true,
        _ => false,
    };
    let dst_ok = if to_float {
        matches!(dst, ValueType::Float)
    } else {
        value_type_is_int_bank(dst)
    };
    (src_ok && dst_ok).then_some(to_float)
}

/// `lltype.cast_ptr_to_int` / `cast_int_to_ptr`. Trailing segments are
/// exactly `lltype::{cast_ptr_to_int,cast_int_to_ptr}` and the banks are
/// Ref↔Int; a bare leaf or a look-alike module stays a call.
pub(in crate::front::mir) fn host_lltype_cast_llop(
    path: &str,
    src: Option<&ValueType>,
    dst: &ValueType,
) -> Option<(&'static str, ValueType)> {
    if path_ends_with_segments(path, "lltype::cast_ptr_to_int") {
        let src_ok = matches!(src, Some(ValueType::Ref(_)) | None);
        return (src_ok && value_type_is_int_bank(dst))
            .then_some(("cast_ptr_to_int", ValueType::Int));
    }
    if path_ends_with_segments(path, "lltype::cast_int_to_ptr") {
        let src_ok = match src {
            Some(ty) => value_type_is_int_bank(ty),
            None => true,
        };
        return (src_ok && matches!(dst, ValueType::Ref(_)))
            .then_some(("cast_int_to_ptr", dst.clone()));
    }
    None
}

/// `majit_rlib::rthread::threadlocalref_get_int` /
/// `threadlocalref_store_int` — the llop intrinsics
/// `ThreadLocalField.getraw` / `setraw` lower to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThreadLocalRefLlop {
    Get,
    Store,
}

/// The full path, crate included, is `majit_rlib::rthread::threadlocalref_get_int`
/// or `majit_rlib::rthread::threadlocalref_store_int`; a look-alike in another
/// crate or module stays a call.
pub(crate) fn host_rthread_threadlocalref_llop(path: &str) -> Option<ThreadLocalRefLlop> {
    match path {
        "majit_rlib::rthread::threadlocalref_get_int" => Some(ThreadLocalRefLlop::Get),
        "majit_rlib::rthread::threadlocalref_store_int" => Some(ThreadLocalRefLlop::Store),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
