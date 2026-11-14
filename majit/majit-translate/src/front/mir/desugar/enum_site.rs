//! Hand lowering of ADT / enum callees.

use super::*;

/// `PartialEq::eq` / `ne` on a named ADT owner.
///
/// `trait_default_call_segments` rewrites a provided `PartialEq::ne` to
/// `[owner…, leaf]`. The Method spelling carries the same leaf on the
/// impl owner. Shared by the niche-`Option` fold and the tagged-`Result`
/// structural derive.
fn partial_eq_eq_ne_leaf(
    segments: &[String],
    method_hint: Option<&(String, String, Option<String>)>,
    owner_path: &[&str],
    owner_leaf: &str,
) -> Option<&'static str> {
    let from_path = if fmt_path_ends_with_owner_method(
        segments,
        owner_path,
        &["core", "cmp", "PartialEq"],
        "eq",
    ) {
        Some("eq")
    } else if fmt_path_ends_with_owner_method(
        segments,
        owner_path,
        &["core", "cmp", "PartialEq"],
        "ne",
    ) {
        Some("ne")
    } else {
        None
    };
    // `CallControl.graphs_from` / `FunctionReprBase.call` call the
    // function the pointer names. Only `core::cmp::PartialEq`'s
    // `eq`/`ne` lower through this matcher; an extension trait's
    // same-named method stays a call.
    let from_method = method_hint.and_then(|(owner, name, trait_identity)| {
        let identity = trait_identity.as_ref()?;
        (owner == owner_leaf
            && matches!(name.as_str(), "eq" | "ne")
            && trait_impl_identity_names_trait(
                &format!("<impl {identity}>"),
                &["core", "cmp", "PartialEq"],
            ))
        .then_some(if name == "ne" { "ne" } else { "eq" })
    });
    from_path.or(from_method)
}

fn diamond_arm_sources(carried: &[Variable], extras: &[Variable]) -> Vec<Variable> {
    let mut sources = carried.to_vec();
    for extra in extras {
        if !sources.contains(extra) {
            sources.push(extra.clone());
        }
    }
    sources
}

fn push_block_op(graph: &mut FunctionGraph, block: BlockId, kind: OpKind) -> Variable {
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind,
    });
    res
}

/// How one Result/Option/struct payload compares, decided before the
/// diamond is built. A payload that cannot be classified makes the
/// whole `Result` `eq`/`ne` decline (`Ok(false)`), graph untouched.
#[derive(Clone)]
enum PayloadEqPlan {
    /// Fieldless variant: `eq` is True, `ne` is False.
    NoPayload,
    String,
    Scalar,
    /// Impl FunDecl with an `Unstructured` body (`ClassDesc.lookup`).
    /// `ne` is PartialEq's provided method (`not eq`) unless the impl
    /// itself defines `ne` with a body.
    FunDecl {
        eq_id: u64,
        eq_segments: Vec<String>,
        ne: Option<(u64, Vec<String>)>,
    },
    /// `#[derive(PartialEq)]` body is not in the LLBC; the TypeDecl
    /// fields AND-chain in declaration order. Presence of a
    /// `StructuralPartialEq` impl is the derive marker.
    Structural {
        owner: String,
        fields: Vec<StructuralFieldPlan>,
    },
    /// Nested `Option` (None=0 / Some=1) or `Result` (Ok=0 / Err=1).
    Tagged {
        owner: String,
        arm0_owner: String,
        arm1_owner: String,
        arm0_ty: ValueType,
        arm1_ty: ValueType,
        arm0_raw_u8: bool,
        arm1_raw_u8: bool,
        arm0: Box<PayloadEqPlan>,
        arm1: Box<PayloadEqPlan>,
        arm0_has: bool,
        arm1_has: bool,
    },
}

#[derive(Clone)]
struct StructuralFieldPlan {
    name: String,
    ty: ValueType,
    plan: Box<PayloadEqPlan>,
}

fn fundecl_path_leaf(fd: &FunDecl) -> Option<&str> {
    fd.item_meta
        .name_path_str()
        .rsplit("::")
        .next()
        .filter(|leaf| !leaf.is_empty())
}

fn emit_eq_false(graph: &mut FunctionGraph, block: BlockId, val: Variable) -> Variable {
    let f = push_block_op(graph, block, OpKind::ConstBool(false));
    push_block_op(
        graph,
        block,
        OpKind::BinOp {
            op: "eq".to_string(),
            lhs: val,
            rhs: f,
            result_ty: ValueType::Int,
        },
    )
}

fn emit_named_field_read(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    owner: &str,
    name: &str,
    ty: ValueType,
) -> Variable {
    let field = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(field.clone()),
        kind: OpKind::FieldRead {
            base: recv,
            field: crate::model::FieldDescriptor::new(name, Some(owner.to_string())),
            ty,
            pure: true,
        },
    });
    field
}

fn remap_payload_goto_args(
    link_args: &[Variable],
    placeholder: &Variable,
    produced: &Variable,
    same_sources: &[Variable],
    same_inputs: &[Variable],
    arm_inputs: &[Variable],
    arm_exit: &[Variable],
) -> Vec<Variable> {
    link_args
        .iter()
        .map(|v| {
            if v == placeholder {
                produced.clone()
            } else {
                let mid = crate::front::option_variant::map_source(same_sources, same_inputs, v)
                    .expect("diamond edge arg carried");
                let mid = crate::front::option_variant::map_source(same_inputs, arm_inputs, &mid)
                    .expect("nested diamond edge arg carried");
                crate::front::option_variant::map_source(arm_inputs, arm_exit, &mid)
                    .expect("payload-cmp exit arg carried")
            }
        })
        .collect()
}

impl<'a> Lowering<'a> {
    /// Named fields of a struct lltype, after peeling a borrow. `None`
    /// when the declaration is missing or opaque, so a wrapper call
    /// declines instead of aliasing the guard.
    pub(in crate::front::mir) fn adt_struct_fields(
        &self,
        ty: &TyRef,
    ) -> Option<Vec<(String, ValueType)>> {
        let peeled = self.tyref_peel_ref_to_pointee(ty);
        let ty = peeled.as_ref().unwrap_or(ty);
        let id = self.tyref_adt_def_id(ty)?;
        let td = self.llbc.type_by_id(id)?;
        if td.published_struct_copy {
            return None;
        }
        let TypeDeclKind::Struct(fields) = &td.kind else {
            return None;
        };
        Some(
            fields
                .iter()
                .filter_map(|field| {
                    let name = field.name.clone()?;
                    Some((name, self.value_type_of(&field.ty)))
                })
                .collect(),
        )
    }

    /// The ADT `def_id` behind a signature [`TyRef`], whether inline
    /// or routed through the dedup table.  `None` for non-ADT shapes.
    pub(in crate::front::mir) fn tyref_adt_def_id(&self, ty: &TyRef) -> Option<u64> {
        match ty {
            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
            TyRef::Dedup { id } => self.llbc.dedup_to_adt_def_id(*id),
        }
    }

    /// One concrete type argument from an ADT signature type.  Charon stores
    /// `Result<X, E>` as `Adt.generics.types = [X, E]`; deserialize the chosen
    /// node back through [`TyRef`] so deduplicated and inline literal forms use
    /// the same width-atom helpers as top-level signature inputs/outputs.
    pub(in crate::front::mir) fn tyref_adt_type_arg(
        &self,
        ty: &TyRef,
        index: usize,
    ) -> Option<TyRef> {
        let adt = tyref_node(ty, self.llbc)?
            .as_object()?
            .get("Adt")?
            .as_object()?;
        let node = type_decl_ref_generics(adt, self.llbc)?
            .as_object()?
            .get("types")?
            .as_array()?
            .get(index)?;
        serde_json::from_value(node.clone()).ok()
    }

    /// The ADT `def_id` behind a signature [`TyRef`], peeling `Ref` /
    /// `RawPtr` wrappers and dedup / hash-cons indirections first — a
    /// `bool::then` closure env arrives as `&closure`.  Mirrors the
    /// wrapper-peeling loop in [`Self::first_input_is_adt`].  `None` for a
    /// non-ADT pointee.
    pub(in crate::front::mir) fn tyref_ref_adt_def_id(&self, ty: &TyRef) -> Option<u64> {
        let mut v: &serde_json::Value = match ty {
            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        loop {
            let obj = v.as_object()?;
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                v = self.llbc.dedup_body(id)?;
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
                continue;
            }
            if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
                v = arr.first()?;
                continue;
            }
            return inline_adt_def_id(v);
        }
    }

    /// The full name path of the ADT behind a signature [`TyRef`], peeling
    /// the same wrappers [`Self::tyref_ref_adt_def_id`] does — a desugared
    /// `it.next()` passes its iterator as `&mut Iter<'_, T>`, so the receiver
    /// reaches [`iterator_adds_a_reference`] behind one reference.
    pub(in crate::front::mir) fn tyref_ref_adt_path(&self, ty: &TyRef) -> Option<String> {
        let id = self.tyref_ref_adt_def_id(ty)?;
        Some(self.llbc.type_by_id(id)?.item_meta.name_path())
    }

    /// Peel `Ref` / `RawPtr` wrappers (through dedup / hash-cons
    /// indirections) to the pointee as an owned [`TyRef`] — the by-value shape
    /// the `Option` owner resolvers expect.  Returns `ty` itself (cloned) when
    /// it is not a reference.  Used to feed an `&Option<..>` receiver (e.g.
    /// `is_none`/`is_some`, which take `&self`) into
    /// [`Self::resolve_option_consumer_owners`], whose payload / suffix reads
    /// need the un-wrapped `Option<..>`.  Same wrapper set as
    /// [`Self::tyref_ref_adt_def_id`], which resolves to a `def_id` instead.
    pub(in crate::front::mir) fn tyref_peel_ref_to_pointee(&self, ty: &TyRef) -> Option<TyRef> {
        let mut v: &serde_json::Value = match ty {
            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        loop {
            let obj = v.as_object()?;
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                v = self.llbc.dedup_body(id)?;
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
                continue;
            }
            if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
                v = arr.first()?;
                continue;
            }
            return serde_json::from_value(v.clone()).ok();
        }
    }

    /// The `Option`'s payload type `T` (its first `generics.types` entry)
    /// projected to a [`ValueType`] — the `Some::__pos_0` field kind a tagged
    /// Option construction writes.  `None`
    /// when the type is not an ADT head carrying a first type argument.
    pub(in crate::front::mir) fn tyref_option_payload_value_type(
        &self,
        ty: &TyRef,
    ) -> Option<ValueType> {
        let body = match ty {
            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        let inner = type_decl_ref_generics(body.get("Adt")?.as_object()?, self.llbc)?
            .get("types")?
            .get(0)?;
        let inner_ty = TyRef::Other(inner.clone());
        if tyref_is_shared_u8_slice(&inner_ty, self.llbc)
            || tyref_is_bare_u8_slice(&inner_ty, self.llbc)
        {
            return Some(ValueType::Int);
        }
        Some(tyref_enum_payload_value_type(
            &inner_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ))
    }

    /// The RPython instance class carried by an `Option<T>` payload when `T`
    /// is a raw/reference pointer to a registered GC object.  `ValueType`
    /// records only the register bank, so a `Ref(None)` payload needs that
    /// class on the Some-arm identity (`Some.__pos_0` of a niche Option)
    /// and on remaining closure-rewrite tuple writes.  This preserves
    /// the concrete field annotation that RPython's `InstanceRepr._setup_repr`
    /// consumes (`rpython/rtyper/rclass.py`).
    ///
    /// One pointer word is the class of its pointee (`Option<*mut T>`,
    /// `Option<&T>`). A remaining pointer after that one word
    /// (`Option<&*mut T>`, `Option<&&T>`) is the address of a slot, not
    /// class `T`: [`strip_ty_wrappers`] would peel the extra `Ref`, and
    /// [`raw_ptr_pointee_class_root_with`] would peel the extra `RawPtr`.
    pub(in crate::front::mir) fn option_payload_instance_class_root(
        &self,
        option_ty: &TyRef,
    ) -> Option<String> {
        // An int-bank niche (`Option<&RawEnum>`, `Option<&RawStruct>`,
        // `Option<fn>`) is Address / SomePtr (`ty_is_raw_address` /
        // `tyref_niche_option_payload_paint`). `rclass.py`
        // `InstanceRepr.convert_const(None)` is `null_instance` of a GC
        // class; `rnone.py` `rtype_is_None` is `ptr_iszero` of that
        // pointer. Promoting the Address to Instance is the
        // `Ptr ∪ Instance` union (`project_struct_rows`).
        if self.option_niche_is_int_bank(option_ty) {
            return None;
        }
        let payload = strip_ty_indirections(
            type_decl_ref_generics(
                tyref_node(option_ty, self.llbc)?
                    .as_object()?
                    .get("Adt")?
                    .as_object()?,
                self.llbc,
            )?
            .get("types")?
            .get(0)?,
            self.llbc,
        )?;
        if let Some(root) =
            raw_ptr_pointee_class_root_with(payload, self.llbc, self.tombstoned_leaves)
        {
            return Some(root);
        }
        if option_payload_remaining_pointer(payload, self.llbc) {
            return None;
        }
        let pointee = payload.as_object()?.get("Ref")?.as_array()?.get(1)?;
        let pointee = strip_ty_indirections(pointee, self.llbc)?;
        let pointee = type_node_peel_aliases(pointee, self.llbc).unwrap_or(pointee);
        adt_node_class_root_with(pointee, self.llbc, self.tombstoned_leaves)
    }

    /// Class root of an `RBigInt` / `&RBigInt` call result, the instance
    /// `annotationoftype` builds with `getuniqueclassdef`.
    pub(in crate::front::mir) fn rbigint_result_class_root(
        &self,
        dest_ty: &TyRef,
    ) -> Option<String> {
        tyref_is_rbigint(dest_ty, self.llbc).then(|| {
            tyref_input_class_root(dest_ty, self.llbc, self.tombstoned_leaves)
                .unwrap_or_else(|| "RBigInt".to_string())
        })
    }

    /// The per-instantiation `Option` enum root for a residual-call
    /// destination `dest_ty` that is an `Option<*mut RegisteredStruct>`
    /// (`Option<PyObjectRef>`) or an `Option<SplitEligibleEnum>`
    /// (`Option<Result<*mut PyObject, PyError>>`), or `None` when `dest_ty`
    /// is not an Option whose payload keys a suffixed constructor classdef.
    ///
    /// A `dont_look_inside` residual returning `Option<*mut PyObject>` has
    /// its result kind erased to the `ref` GCREF token, so the call result
    /// arrives classdef-less; the caller's `if let Some(x) = ..` then reads
    /// `__discriminant` off a bare pointer and panics.  Minting the SAME
    /// root a constructed `Some(..)`/`None` of this instantiation mints
    /// (`Option`'s `name_path` + the destination `<X>` suffix, exactly as
    /// [`Self::resolve_option_dest_owners`] / the tagged-pair ctor derive it)
    /// lets the narrowed result share the constructed Option's ClassDef, so
    /// the residual and constructed Options unify at a phi merge and the
    /// discriminant / payload reads resolve.
    ///
    /// The payload distinguisher accepts the Option's first type argument
    /// when it is either a raw pointer whose pointee resolves through
    /// [`raw_ptr_pointee_class_root`] (`Some` for `Option<*mut PyObject>`),
    /// or a split-eligible enum whose head [`adt_head_instantiation_suffix`]
    /// suffixes (`Some` for `Option<Result<..>>` — the same instantiation the
    /// inlined constructor mints a suffixed `Some`/`None` classdef for).
    /// `None` for `Option<u64>` (no raw pointer, not an enum), `Option<*mut
    /// u8>` (primitive pointee, no registered root) and a non-split enum arg
    /// — all stay classdef-less GCREF, unchanged. A remaining pointer after
    /// one `Ref` (`Option<&*mut T>` / `Option<&&T>`) is the address of a
    /// slot, not class `T` (`rdict.py` `ll_get`): [`strip_ty_wrappers`]
    /// would extra-peel it, so [`option_payload_remaining_pointer`] bails
    /// first and the dest stays classdef-less GCREF.
    pub(in crate::front::mir) fn option_residual_narrow_root(
        &self,
        dest_ty: &TyRef,
    ) -> Option<String> {
        if !crate::front::result_exc::tyref_is_option(dest_ty, self.llbc) {
            return None;
        }
        // Distinguisher: the Option's payload must be either a raw pointer
        // whose pointee resolves to a registered struct root, or a
        // split-eligible enum (`Option<Result<..>>`) — both cases the
        // constructor path mints a suffixed classdef for, so the residual
        // receiver must key the same suffixed root instead of a bare GCREF.
        // Foreign value payloads (`BigInt` / `Wtf8Buf`, no registered root;
        // a non-split enum arg) still bail, keeping them classdef-less.
        // A remaining pointer after one Ref is a slot address, not class T.
        let adt = tyref_node(dest_ty, self.llbc)?
            .as_object()?
            .get("Adt")?
            .as_object()?;
        let payload = type_decl_ref_generics(adt, self.llbc)?
            .get("types")?
            .get(0)?;
        if option_payload_remaining_pointer(payload, self.llbc) {
            return None;
        }
        let payload_node = strip_ty_wrappers(payload, self.llbc)?;
        let payload_ok =
            raw_ptr_pointee_class_root_with(payload_node, self.llbc, self.tombstoned_leaves)
                .is_some()
                || payload_node
                    .as_object()
                    .and_then(|o| o.get("Adt"))
                    .and_then(serde_json::Value::as_object)
                    .and_then(|adt| {
                        adt_head_instantiation_suffix(adt, self.llbc, self.tombstoned_leaves)
                    })
                    .is_some();
        if !payload_ok {
            return None;
        }
        // The narrow root is the Option enum instantiation itself — the same
        // spelling a static `Some(..)` construction of this instantiation mints.
        // An empty suffix is the template ClassDef; its `Some.__pos_0` is the
        // typevar and getattr cannot succeed (`bookkeeper.py` `getuniqueclassdef`).
        let suffix = tyref_enum_instantiation_suffix(dest_ty, self.llbc, self.tombstoned_leaves);
        if suffix.is_empty() {
            return None;
        }
        let def_id = self.tyref_adt_def_id(dest_ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        Some(format!("{}{}", td.item_meta.name_path_str(), suffix))
    }

    /// Class root of a one-word niche `Option<&T>` / `Option<&mut T>` /
    /// `Option<*mut T>`: the registered ADT pointee, except a GC `&mut T`
    /// payload which is the MutRef cell (`gc_mut_ref_input_stamp` /
    /// `tyref_option_gc_mut_ref_cell_root`).
    ///
    /// Rust uses the null pointer niche for this Option shape. It therefore
    /// maps to RPython's nullable `SomeInstance`, not to an Option
    /// container class with `__discriminant` / `__pos_0` fields. `None`
    /// (`option_niche_null_cast`) and the Some-arm identity share this
    /// root so a `None` of `Option<&mut T>` unions with a MutRef wrap
    /// (`model.py` `SomeInstance.__init__` / `binaryop.py` `union`).
    pub(in crate::front::mir) fn option_niche_payload_class_root(
        &self,
        dest_ty: &TyRef,
    ) -> Option<String> {
        if !self.tyref_is_niche_option_ptr(dest_ty) {
            return None;
        }
        // A GC `&mut T` payload is the MutRef cell, not `T`.
        tyref_option_gc_mut_ref_cell_root(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
        .or_else(|| self.option_payload_instance_class_root(dest_ty))
    }

    /// Target projection for the null half of a nullable Option whose payload
    /// the source translator has changed from a Rust container to an RPython
    /// value repr. `null_mut()` begins as a classdef-less pointer; string and
    /// list comparisons require the same `StringRepr` / `ListRepr` as their
    /// receiver, exactly as `convert_const(None)` is applied on the selected
    /// repr upstream. Ordinary pointer payloads already share InstanceRepr and
    /// need no projection.
    pub(in crate::front::mir) fn option_niche_null_cast(
        &self,
        option_ty: &TyRef,
    ) -> Option<(String, ValueType)> {
        // `Option<&[u8]>` is a two-word nullable window, so
        // `tyref_is_niche_option_ptr` is false. The None arm is still a
        // null `CCHARP` data word, not `convert_const(None)` on
        // `StringRepr`.
        if tyref_is_option_of_shared_u8_slice(option_ty, self.llbc) {
            return Some(("&[u8]".to_string(), ValueType::Int));
        }
        if !self.tyref_is_niche_option_ptr(option_ty) {
            return None;
        }
        let adt = tyref_node(option_ty, self.llbc)?
            .as_object()?
            .get("Adt")?
            .as_object()?;
        let payload = type_decl_ref_generics(adt, self.llbc)?
            .get("types")?
            .get(0)?;
        let stripped = strip_ty_wrappers(payload, self.llbc)?;
        let rendered = charon_type_value_to_ast_string(stripped, self.llbc, 0);
        let payload_ty = tyref_enum_payload_value_type(
            &TyRef::Other(payload.clone()),
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        // RPython has one string type. Every [`tyref_is_option_of_string_value`]
        // payload paints `Str` here (`&str` / `Utf8Str` / `Wtf8` / `Wtf8Buf`).
        // A rust-heap `String` payload still paints `Ref` so `ll_strconcat`
        // does not consume `{cap, ptr, len}`. The `None` of `Option<String>`
        // / `Option<&String>` is that same null Ref, not a `STR` pointer.
        // `PathBuf` / `OsString` is `rpath.py` `s_Str0`; `None` of
        // `Option<PathBuf>` is that same null Ref (`StringRepr.convert_const(None)`).
        if tyref_is_option_of_string_value(option_ty, self.llbc)
            || matches!(payload_ty, ValueType::Str)
        {
            return Some((rendered, ValueType::Str));
        }
        if tyref_is_string_or_os_path_family(&TyRef::Other(payload.clone()), self.llbc) {
            return Some((rendered, payload_ty));
        }
        if rendered.starts_with("Vec<") || rendered.starts_with("VecDeque<") {
            return Some((rendered.clone(), ValueType::Ref(Some(rendered))));
        }
        // `&mut [u8]` / `*mut [u8]` is a nullable list (`tyref_is_u8_list`,
        // `tyref_niche_option_payload_paint`). Charon stringifies both
        // `&[u8]` and `&mut [u8]` as `[u8]`; the Mut Ref node is the
        // gate (`type_node_is_mut_ref`). `None` is `convert_const(None)`
        // on that `ListRepr` (`rlist.py`). Shared `&[u8]` already
        // returned as a window above.
        if tyref_is_u8_list(&TyRef::Other(payload.clone()), self.llbc)
            || (type_node_is_mut_ref(payload, self.llbc)
                && (rendered.starts_with('[') || rendered.starts_with("Slice<")))
        {
            return Some((rendered.clone(), ValueType::Ref(Some(rendered))));
        }
        // Shared `&[PyObjectRef]` is one GcArray word; `None` is
        // `convert_const(None)` on that `ListRepr` (`rlist.py`
        // `FixedSizeListRepr`). A mut slice of the same items is the
        // `{ptr, len}` pair and is not this niche.
        if let Some(root) =
            object_pointer_gcarray_list_root(&TyRef::Other(payload.clone()), self.llbc)
        {
            return Some((root.clone(), ValueType::Ref(Some(root))));
        }
        // Pointer-niche `Option<&RawStruct>` is the nullable `Ptr`
        // (`tyref_niche_option_payload_paint` /
        // `raw_struct_ptr_annotation`). `history.getkind` of that `Ptr`
        // is `'int'`. Painting `None` as `Ref` left `Result::ok` to
        // union a classdef-less Instance with the Ok `Int` (`unionof`).
        if let Some(root) =
            niche_option_storage_raw_struct_root(option_ty, self.llbc, self.tombstoned_leaves)
        {
            return Some((root, ValueType::Int));
        }
        // `None` of a nullable instance is RPython's `None`, which unions
        // into `SomeInstance(T, can_be_none=True)` and keeps the class
        // (`model.py` `unionof`). The bare `null_mut()` is classdef-less.
        // Narrow the null to the same class the Some arm occupies
        // (`rnone.py` `pairtype(NoneRepr, Repr).convert_from_to`):
        // MutRef cell for a GC `&mut T`, payload class otherwise.
        if let Some(root) = self.option_niche_payload_class_root(option_ty) {
            return Some((root.clone(), ValueType::Ref(Some(root))));
        }
        None
    }

    /// `__pos_0` of an enum whose payload is a nullable instance pointer
    /// (`Result<Option<*mut T>, _>::Ok`, or `Result<*mut T, _>`).
    ///
    /// The niche `Option` is the pointer itself (`option_niche_null_cast`).
    /// A bare field read annotates as a classdef-less instance, and
    /// `unionof` with the already-narrowed null arm then drops the class.
    /// The read wants the same `__cast_instance_intrinsic` narrow.
    pub(in crate::front::mir) fn enum_payload_is_nullable_instance_ptr(
        &self,
        place_ty: &TyRef,
    ) -> bool {
        // Tagged `Option<Option<&Raw>>::__pos_0` is the inner int-bank
        // niche word. Recasting it through `W_ROOT` is
        // `Int ∪ Ref(PyObject)` at the flatten return (`rnone.py`
        // `rtype_is_None` / `rclass.py` `convert_const(None)`: one None
        // per pointer, the outer tag cannot reuse it).
        if self.option_niche_is_int_bank(place_ty) {
            return false;
        }
        if self
            .option_niche_null_cast(place_ty)
            .is_some_and(|(root, ty)| {
                // `Result<Option<*mut T>>::Ok.__pos_0` recasts a nullable
                // GC instance through W_ROOT. A string/path payload is
                // `rstr.py` `SomeString` / `rpath.py` `_posix_rnormpath`
                // `s_Str0` (`StringRepr.convert_const(None)`); W_ROOT
                // would plant `Ref(PyObject)` onto that word.
                matches!(
                    &ty,
                    ValueType::Ref(Some(payload))
                        if !root.starts_with("Vec<")
                            && !root.starts_with("VecDeque<")
                            && !crate::front::mir::is_str_family_class_root(&root)
                            && !crate::front::mir::is_str_family_class_root(payload)
                )
            })
        {
            return true;
        }
        registered_gc_ptr_class_root_with(
            place_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
        .is_some()
    }

    /// Class of a nullable instance-pointer payload (`&mut T`, `*mut T`,
    /// niche `Option<&mut T>`). An instance pointer keeps that class's
    /// classdef (`bookkeeper.py` `getuniqueclassdef`; `rclass.py`
    /// `InstanceRepr`); only an explicit upcast goes to the base
    /// (`annlowlevel.py` `cast_base_ptr_to_instance`).
    pub(in crate::front::mir) fn nullable_instance_payload_class_root(
        &self,
        payload_ty: &TyRef,
    ) -> Option<String> {
        self.option_niche_payload_class_root(payload_ty)
            .or_else(|| {
                registered_gc_ptr_class_root_with(
                    payload_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            })
            .or_else(|| {
                enum_payload_instance_class_root(payload_ty, self.llbc, self.tombstoned_leaves)
            })
    }

    /// Resolve a destination `Option` into its `(option_owner, some_owner,
    /// payload_ty)` — the enum root ctor owner, the `Some` variant
    /// payload-field owner, and the payload `ValueType`.  `None` when the
    /// destination is not a resolvable `Option`.
    pub(in crate::front::mir) fn resolve_option_dest_owners(
        &self,
        dest_ty: &TyRef,
    ) -> Option<(String, String, ValueType)> {
        let def_id = self.tyref_adt_def_id(dest_ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        // Suffix the enum root with the destination `Option<X>`'s `<X>` so a
        // construction mints the same per-instantiation root a static
        // `Some(..)` mints; `some_owner` then derives `Option<X>::Some`.
        let option_owner = format!(
            "{}{}",
            td.item_meta.name_path_str(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc, self.tombstoned_leaves)
        );
        let some_owner = Self::tagged_pair_payload_owner(td, &option_owner, 1)?;
        let payload_ty = self.tyref_option_payload_value_type(dest_ty)?;
        Some((option_owner, some_owner, payload_ty))
    }

    /// Resolve the `(option_owner, some_owner, payload_ty)` an `Option`
    /// *consumer* pass (`expect`, checked unsigned arithmetic) must spell so its
    /// `__discriminant` / `__pos_0` reads key the SAME classdef the value's
    /// *producer* wrote.
    ///
    /// Agreement is by construction: the suffixed producers (`slice_first`,
    /// `from_size_align`) mint a
    /// per-instantiation `Option<X>` root through
    /// [`Self::resolve_option_dest_owners`], so the consumer reuses that
    /// exact derivation, including checked unsigned arithmetic. Its generated
    /// Some/None constructors must identify the same existing classes as a
    /// source-level Some(value), not create a separate bare Option payload.
    /// A niche
    /// one-word niche `Option` has no aggregate
    /// `__discriminant` / `__pos_0` (the arms use a pointer null-test and an
    /// identity payload), so its owners are never read — keep them bare, which
    /// also mirrors [`Self::resolve_option_dest_owners`]'s own aggregate
    /// applicability.
    pub(in crate::front::mir) fn resolve_option_consumer_owners(
        &self,
        recv_ty: &TyRef,
    ) -> Option<(String, String, ValueType)> {
        let payload_ty = self.tyref_option_payload_value_type(recv_ty)?;
        if self.tyref_is_niche_option_ptr(recv_ty) {
            let def_id = self.tyref_adt_def_id(recv_ty)?;
            let td = self.llbc.type_by_id(def_id)?;
            let option_owner = td.item_meta.name_path();
            let some_owner = Self::tagged_pair_payload_owner(td, &option_owner, 1)?;
            return Some((option_owner, some_owner, payload_ty));
        }
        self.resolve_option_dest_owners(recv_ty)
    }

    /// Resolve a recognized `<[T]>::first(slice)` call into a
    /// [`crate::front::slice_first::SliceFirstSite`] — the `Option` enum root +
    /// `Some` variant owners the length-checked diamond post-pass spells its
    /// arms with.  The post-pass accepts both representations that
    /// `lower_call` can produce: a reference-payload result followed by
    /// `__cast_instance_intrinsic`, and a value-payload `Option<u8>` / `Option<i64>`
    /// with no trailing cast.  Reuse the consumer-owner spelling so unsigned
    /// payloads stay on the bare `Option` root used by checked arithmetic,
    /// while other payloads keep their per-instantiation class.  `None` when
    /// the destination is not a resolvable `Option`.
    pub(in crate::front::mir) fn recognize_slice_first_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::slice_first::SliceFirstSite> {
        let (option_owner, some_owner, payload_ty) =
            self.resolve_option_consumer_owners(dest_ty)?;
        let niche = self.tyref_is_niche_option_ptr(dest_ty);
        let payload_narrow_root = niche
            .then(|| self.option_niche_payload_class_root(dest_ty))
            .flatten();
        Some(crate::front::slice_first::SliceFirstSite {
            result_var: result_var.clone(),
            access: crate::front::slice_first::SliceAccess::First,
            option_owner,
            some_owner,
            payload_ty,
            niche,
            niche_null_cast: self.option_niche_null_cast(dest_ty),
            payload_narrow_root,
            array_type_id: None,
            object_array_type_id: None,
            string_byte_view: false,
        })
    }

    /// Whether the receiver of a recognized slice call is an `as_bytes()` view
    /// of a string rather than a GC array, i.e. carries a `StringRepr` length
    /// and a `__string_byte_getitem` element read.
    pub(in crate::front::mir) fn slice_receiver_is_string_byte_view(
        &self,
        arg_locals: &[Option<usize>],
        _first_arg_ty: Option<&TyRef>,
    ) -> bool {
        arg_locals
            .first()
            .copied()
            .flatten()
            .is_some_and(|local| self.string_byte_view_locals.contains(&local))
    }

    /// Stamp byte-view / ARRAY identity onto a `first` / `last` site.
    /// `as_bytes()` aliases the receiver to the `StringRepr`, so the
    /// successful arm must emit `__string_byte_getitem`; any other
    /// proven element spelling rides on `array_type_id` the way
    /// [`Self::recognize_slice_get_site`] does.
    pub(in crate::front::mir) fn annotate_slice_first_site(
        &self,
        site: &mut crate::front::slice_first::SliceFirstSite,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        element: Option<(ValueType, Option<String>)>,
    ) {
        let string_byte_view = self.slice_receiver_is_string_byte_view(arg_locals, first_arg_ty);
        if string_byte_view {
            site.string_byte_view = true;
            site.array_type_id = None;
        } else if let Some((_item_ty, array_type_id)) = element {
            site.array_type_id = array_type_id;
        }
    }

    /// Resolve a recognized `<[T]>::get(slice, i)` call into a
    /// [`crate::front::slice_get::SliceGetSite`] — the same `Option` enum root
    /// plus `Some` variant owners [`Self::recognize_slice_first_site`] resolves,
    /// plus the `SliceIndex` instantiation gate `first` has no need of.
    ///
    /// The call site has already proved the `SliceIndex` instantiation is
    /// scalar through [`Self::is_slice_get_scalar_call`], using either the
    /// operand Place type or (for a literal index) the method generic.  Resolve
    /// the destination with the same consumer-owner rule used by checked
    /// arithmetic and `Option` consumers: reference payloads may carry the
    /// trailing narrowing cast the post-pass absorbs, while scalar payloads
    /// (`Option<u8>`, `Option<i64>`, …) remain value-banked and have no cast.
    pub(in crate::front::mir) fn recognize_slice_get_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
        item_ty: ValueType,
        array_type_id: Option<String>,
    ) -> Option<crate::front::slice_get::SliceGetSite> {
        let Some((option_owner, some_owner, _reference_payload_ty)) =
            self.resolve_option_consumer_owners(dest_ty)
        else {
            crate::decline::record_named(
                crate::decline::gate::SLICE_GET_SITE,
                "dest-not-a-resolvable-option",
                &self.graph.name,
            );
            return None;
        };
        crate::decline::observe_accept(
            crate::decline::gate::SLICE_GET_SITE,
            "site-recognized (ACCEPT, not a decline)",
            &self.graph.name,
        );
        let niche = self.tyref_is_niche_option_ptr(dest_ty);
        let payload_narrow_root = niche
            .then(|| self.option_niche_payload_class_root(dest_ty))
            .flatten();
        Some(crate::front::slice_get::SliceGetSite {
            result_var: result_var.clone(),
            option_owner,
            some_owner,
            payload_ty: item_ty,
            array_type_id,
            object_array_type_id: None,
            niche,
            niche_null_cast: self.option_niche_null_cast(dest_ty),
            payload_narrow_root,
            payload_is_element_addr: false,
            element_byte_size: 0,
        })
    }

    /// Resolve a recognized carrier `Result<T, PyError>` combinator
    /// (`unwrap_or` / `expect` / `unwrap`) into a
    /// [`crate::front::result_exc::CarrierCombinatorSite`].  `None` when the
    /// receiver is not the exception-carrier Result — plain Option / Result
    /// unwrap and expect are regular calls of the extracted core bodies,
    /// and `unwrap_or` stays a residual call whose core body is translated.
    pub(in crate::front::mir) fn recognize_carrier_combinator_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
        catch: crate::front::result_exc::CarrierCatchKind,
    ) -> Option<crate::front::result_exc::CarrierCombinatorSite> {
        let recv_ty = recv_ty?;
        let peeled = self.peel_to_option_or_result(recv_ty);
        let recv_ty = peeled.as_ref().unwrap_or(recv_ty);
        if !crate::front::result_exc::tyref_is_result_of_carrier(
            recv_ty,
            self.llbc,
            self.static_addrs.error_carrier,
        ) {
            return None;
        }
        let payload_ty = if crate::front::result_exc::tyref_result_ok_is_unit(recv_ty, self.llbc) {
            ValueType::Void
        } else {
            self.tyref_option_payload_value_type(recv_ty)?
        };
        Some(crate::front::result_exc::CarrierCombinatorSite {
            result_var: result_var.clone(),
            payload_ty,
            catch,
        })
    }

    /// Resolve `unwrap_or` on `Result<T, RBigIntError>` — the OverflowError
    /// catch of `jit_bigint_toint` (`rpython/rlib/rbigint.py` `rbigint.toint`).
    /// `None` when the receiver is not that Result.
    pub(in crate::front::mir) fn recognize_overflow_combinator_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
        catch: crate::front::result_exc::CarrierCatchKind,
    ) -> Option<crate::front::result_exc::CarrierCombinatorSite> {
        let recv_ty = recv_ty?;
        let peeled = self.peel_to_option_or_result(recv_ty);
        let recv_ty = peeled.as_ref().unwrap_or(recv_ty);
        if !crate::front::result_exc::tyref_is_result_of_rbigint_error(recv_ty, self.llbc) {
            return None;
        }
        let payload_ty = self.tyref_option_payload_value_type(recv_ty)?;
        Some(crate::front::result_exc::CarrierCombinatorSite {
            result_var: result_var.clone(),
            payload_ty,
            catch,
        })
    }

    /// Resolve the type-owned parts of an `Option::ok_or_else` continuation
    /// that `front::checked_arith` may fuse into a signed `*_ovf` edge.  The
    /// receiver and destination gates keep unrelated same-named methods out;
    /// the post-pass proves that this receiver is the checked result.
    pub(in crate::front::mir) fn recognize_checked_arith_ok_or_else_site(
        &self,
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        dest_ty: &TyRef,
        result_var: &Variable,
        closure_arg: Variable,
    ) -> Option<crate::front::checked_arith::CheckedArithOkOrElseSite> {
        if !crate::front::result_exc::tyref_is_option(recv_ty?, self.llbc)
            || !crate::front::result_exc::tyref_is_result(dest_ty, self.llbc)
        {
            return None;
        }
        let env_def_id = self.tyref_ref_adt_def_id(env_ty?)?;
        let call_once_owner = closure_env_call_once_owner(self.llbc.type_by_id(env_def_id)?);
        let result_suffix = crate::front::result_exc::tyref_result_instantiation_suffix(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
        )
        .unwrap_or_default();
        let ok_ty = self.tyref_adt_type_arg(dest_ty, 0)?;
        let err_ty = self.tyref_adt_type_arg(dest_ty, 1)?;
        Some(crate::front::checked_arith::CheckedArithOkOrElseSite {
            result_var: result_var.clone(),
            closure_arg,
            call_once_owner,
            result_suffix,
            ok_payload_ty: tyref_enum_payload_value_type(
                &ok_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            err_payload_ty: tyref_enum_payload_value_type(
                &err_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
        })
    }

    /// Resolve an unsigned `checked_{add,sub,mul}` call whose result is an
    /// `Option<T>` into a
    /// [`crate::front::checked_arith_uint::CheckedArithUintSite`] — the
    /// `Option` enum root + `Some` variant owners, the payload type, and the
    /// operand atom width, keyed off the destination `Option<T>` type.
    /// Mirrors [`Self::resolve_option_consumer_owners`] so the virtualized
    /// `Option` this pass builds shares field descrs with downstream `?` /
    /// match reads.
    pub(in crate::front::mir) fn recognize_checked_arith_uint_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
        payload_hint: Option<&TyRef>,
        bits: u32,
    ) -> Option<crate::front::checked_arith_uint::CheckedArithUintSite> {
        if !crate::front::result_exc::tyref_is_option(dest_ty, self.llbc) {
            return None;
        }
        let (mut option_owner, mut some_owner, payload_ty) =
            self.resolve_option_dest_owners(dest_ty)?;
        // The dest place is often a bare `Option` type_id. The payload
        // type (FunDecl output, or a `usize` operand of `checked_add`)
        // still names the instantiation the `Some` constructor and the
        // `?` getattr must share (`bookkeeper.py` `getuniqueclassdef`).
        self.instantiate_option_owners_from_payload(
            dest_ty,
            payload_hint,
            &mut option_owner,
            &mut some_owner,
        );
        Some(crate::front::checked_arith_uint::CheckedArithUintSite {
            opt: result_var.clone(),
            option_owner,
            some_owner,
            payload_ty,
            bits,
        })
    }

    /// Resolve a `Result::ok()` call whose result is an `Option<Layout>` into a
    /// [`crate::front::from_size_align::FromSizeAlignSite`] — the `Option` enum
    /// root + `Some` variant owners, the payload `ValueType`, and the nested
    /// `Layout` ADT owner, keyed off the `.ok()` destination `Option<Layout>`
    /// type.  Gated on the payload being `core::alloc::layout::Layout` so only
    /// the `from_size_align(..).ok()` shape is recorded; the post-pass validates
    /// the preceding `from_size_align` residual.  The `Option` root is
    /// **instantiation-suffixed** (`resolve_option_dest_owners`):
    /// the enclosing `try_typed_items_block_layout` also builds
    /// `checked_{mul,add}` `Option<usize>` values, and a bare `Option` root
    /// would share one `core.option.Option::Some::__pos_0` field with them,
    /// unioning `Integer` with `Instance(Layout)` (annotator UnionError).  The
    /// suffix gives `Option<Layout>` its own ClassDef and `__pos_0` field.
    pub(in crate::front::mir) fn recognize_from_size_align_ok_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::from_size_align::FromSizeAlignSite> {
        if !crate::front::result_exc::tyref_is_option(dest_ty, self.llbc) {
            return None;
        }
        let (option_owner, some_owner, payload_ty) = self.resolve_option_dest_owners(dest_ty)?;
        // The `Some` payload must be the opaque `core::alloc::layout::Layout`
        // ADT — this gates recording to the `from_size_align(..).ok()` shape.
        let payload_ty_ref = self.tyref_adt_type_arg(dest_ty, 0)?;
        let layout_def_id = self.tyref_adt_def_id(&payload_ty_ref)?;
        let layout_owner = self.llbc.type_by_id(layout_def_id)?.item_meta.name_path();
        if !crate::front::from_size_align::is_layout_adt_owner(&layout_owner) {
            return None;
        }
        Some(crate::front::from_size_align::FromSizeAlignSite {
            ok_result: result_var.clone(),
            option_owner,
            some_owner,
            payload_ty,
            layout_owner,
        })
    }

    /// Resolve a `Result::expect(res, msg)` call whose result is a by-value
    /// `Layout` into a
    /// [`crate::front::from_size_align::FromSizeAlignExpectSite`] — the
    /// by-value sibling of [`Self::recognize_from_size_align_ok_site`].  Gated
    /// on the destination being `core::alloc::layout::Layout` so only the
    /// `from_size_align(..).expect(..)` shape is recorded; the post-pass
    /// validates the preceding `from_size_align` residual + folded-const align
    /// before mutating.  `None` (leaving the residual call) otherwise.
    pub(in crate::front::mir) fn recognize_from_size_align_expect_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::from_size_align::FromSizeAlignExpectSite> {
        let layout_def_id = self.tyref_adt_def_id(dest_ty)?;
        let layout_owner = self.llbc.type_by_id(layout_def_id)?.item_meta.name_path();
        if !crate::front::from_size_align::is_layout_adt_owner(&layout_owner) {
            return None;
        }
        Some(crate::front::from_size_align::FromSizeAlignExpectSite {
            result_var: result_var.clone(),
            layout_owner,
        })
    }

    /// Resolve a `Layout::from_size_align(size, align)` call whose result is
    /// `Result<Layout, LayoutError>` into a
    /// [`crate::front::from_size_align::FromSizeAlignCallSite`].  Gated on
    /// the Ok payload being the Layout ADT so only this producer is
    /// recorded; the post-pass validates the residual Call is still the
    /// last op before mutating.  `None` (leaving the residual Call)
    /// otherwise.  Twin of [`Self::recognize_checked_arith_uint_site`]:
    /// owners come from the destination `Result` type while it is still in
    /// hand.
    pub(in crate::front::mir) fn recognize_from_size_align_call_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::from_size_align::FromSizeAlignCallSite> {
        let dest = self.peel_to_option_or_result(dest_ty);
        let dest_ty = dest.as_ref().unwrap_or(dest_ty);
        if !crate::front::result_exc::tyref_is_result(dest_ty, self.llbc) {
            return None;
        }
        let ok_ty = crate::front::result_exc::tyref_result_ok(dest_ty, self.llbc)?;
        let layout_def_id = self
            .tyref_adt_def_id(&ok_ty)
            .or_else(|| self.tyref_ref_adt_def_id(&ok_ty))?;
        let layout_owner = self.llbc.type_by_id(layout_def_id)?.item_meta.name_path();
        if !crate::front::from_size_align::is_layout_adt_owner(&layout_owner) {
            return None;
        }
        let (result_owner, ok_owner, _err_owner, payload_ty, _, _) =
            self.resolve_result_owners(dest_ty)?;
        Some(crate::front::from_size_align::FromSizeAlignCallSite {
            result_var: result_var.clone(),
            result_owner,
            ok_owner,
            layout_owner,
            payload_ty,
        })
    }

    /// Resolve a `Layout::size` / `Layout::align` call whose receiver peels
    /// to the Layout ADT into a
    /// [`crate::front::from_size_align::LayoutAccessorSite`].  `None` when
    /// the receiver is not Layout (leaving the residual Call).
    pub(in crate::front::mir) fn recognize_layout_accessor_site(
        &self,
        field: &'static str,
        first_arg_ty: Option<&TyRef>,
        result_var: &Variable,
    ) -> Option<crate::front::from_size_align::LayoutAccessorSite> {
        let layout_owner = self.tyref_ref_adt_path(first_arg_ty?)?;
        if !crate::front::from_size_align::is_layout_adt_owner(&layout_owner) {
            return None;
        }
        Some(crate::front::from_size_align::LayoutAccessorSite {
            result_var: result_var.clone(),
            field,
            layout_owner,
        })
    }

    /// Dest type of a bare `Option` place with the payload as its type
    /// argument. [`Self::resolve_option_dest_owners`] then reads the
    /// dest type's generic args and [`Self::tagged_pair_payload_owner`]
    /// (`v.name`) for the `Some` owner.
    pub(in crate::front::mir) fn option_dest_with_payload(
        &self,
        dest_ty: &TyRef,
        payload: &TyRef,
    ) -> Option<TyRef> {
        let def_id = self.tyref_adt_def_id(dest_ty)?;
        let payload_node = tyref_node(payload, self.llbc)?.clone();
        Some(TyRef::Other(serde_json::json!({
            "Adt": {
                "id": def_id,
                "generics": {
                    "regions": [],
                    "types": [payload_node],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })))
    }

    /// When the dest place dropped its generics, rebuild the dest type
    /// from the payload `TyRef` and intern owners through
    /// [`Self::resolve_option_dest_owners`].
    pub(in crate::front::mir) fn instantiate_option_owners_from_payload(
        &self,
        dest_ty: &TyRef,
        payload_hint: Option<&TyRef>,
        option_owner: &mut String,
        some_owner: &mut String,
    ) {
        if !tyref_enum_instantiation_suffix(dest_ty, self.llbc, self.tombstoned_leaves).is_empty() {
            return;
        }
        let Some(hint) = payload_hint else {
            return;
        };
        if tyref_is_type_parameter(hint, self.llbc) {
            return;
        }
        let Some(inst) = self.option_dest_with_payload(dest_ty, hint) else {
            return;
        };
        if tyref_enum_instantiation_suffix(&inst, self.llbc, self.tombstoned_leaves).is_empty() {
            return;
        }
        if let Some((owner, some, _)) = self.resolve_option_dest_owners(&inst) {
            *option_owner = owner;
            *some_owner = some;
        }
    }

    /// `Result<_, E>` inside a function returning an exception sum is that
    /// sum's `?`, not a value `Try::branch`. The enclosing return is local 0.
    fn enclosing_sum_claims_error(&self, recv_ty: &TyRef) -> bool {
        let Some(output) = self.body.locals.locals.first() else {
            return false;
        };
        let Some(sum) = crate::front::result_exc::exception_sum_of_result(
            &output.ty,
            self.llbc,
            self.static_addrs.error_carrier,
        ) else {
            return false;
        };
        let Some(err) = crate::front::result_exc::tyref_result_err(recv_ty, self.llbc) else {
            return false;
        };
        crate::front::result_exc::exception_sum_claims_error(
            &sum,
            &tyref_to_ast_string(&err, self.llbc),
            self.static_addrs.error_carrier.carrier_path,
        )
    }

    fn result_try_excluded(&self, recv_ty: &TyRef) -> bool {
        crate::front::result_exc::tyref_lowers_as_result_exc(
            recv_ty,
            self.llbc,
            self.static_addrs.error_carrier,
        ) || self.enclosing_sum_claims_error(recv_ty)
    }

    /// Resolve `Try::branch(res)` where `res: Result<T, E>` and `E` is not the
    /// exception carrier.  Carrier Results stay on `result_exc`; a miss here
    /// leaves the residual `branch` call. An exception sum claims a marked
    /// residual the same way: `From` is the identity and the instance is raised.
    pub(in crate::front::mir) fn recognize_result_try_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
    ) -> Option<ResultTrySite> {
        let recv_ty = self.peel_to_option_or_result(recv_ty?)?;
        if !crate::front::result_exc::tyref_is_result(&recv_ty, self.llbc)
            || self.result_try_excluded(&recv_ty)
        {
            return None;
        }
        let (result_owner, ok_owner, err_owner, ok_ty, err_ty, err_class) =
            self.resolve_result_owners(&recv_ty)?;
        Some(ResultTrySite {
            branch_result_var: result_var.clone(),
            result_owner,
            ok_owner,
            err_owner,
            ok_ty,
            err_ty,
            ok_is_raw_u8: crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                .is_some_and(|ty| tyref_is_raw_u8_ptr(&ty, self.llbc)),
            err_is_raw_u8: crate::front::result_exc::tyref_result_err(&recv_ty, self.llbc)
                .is_some_and(|ty| tyref_is_raw_u8_ptr(&ty, self.llbc)),
            err_class,
            recv_err_ast: crate::front::result_exc::tyref_result_err(&recv_ty, self.llbc)
                .map(|ty| tyref_to_ast_string(&ty, self.llbc))
                .unwrap_or_default(),
        })
    }

    /// `Result::branch` whose operand and `ControlFlow` result share a
    /// Charon type-decl layout is already that value: emit `same_as`.
    /// A missing layout stays a call for the later discriminant match.
    /// A `Result<T, PyError>` keeps its `branch`: `front::result_exc`
    /// rewires that `?` diamond onto the exception edge.
    pub(in crate::front::mir) fn rewrite_equal_layout_result_branch(
        &self,
        op_kind: OpKind,
        recv_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> OpKind {
        let Some(recv_ty) = recv_ty else {
            return op_kind;
        };
        if !crate::front::result_exc::tyref_is_result(recv_ty, self.llbc)
            || self.result_try_excluded(recv_ty)
        {
            return op_kind;
        }
        let Some(result_layout) = self.layout_of_tyref(recv_ty) else {
            return op_kind;
        };
        let Some(flow_layout) = self.layout_of_tyref(dest_ty) else {
            return op_kind;
        };
        rewrite_result_branch_for_layouts(op_kind, Some(&result_layout), Some(&flow_layout))
    }

    /// Record each `Result::branch` variant payload from the Result and
    /// ControlFlow type decls. The later match reads these instead of
    /// assuming every payload is a reference.
    pub(in crate::front::mir) fn stamp_result_branch_payloads(
        &self,
        kind: OpKind,
        recv_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> OpKind {
        let OpKind::Call {
            target,
            args,
            result_ty,
        } = kind
        else {
            return kind;
        };
        let CallTarget::Method {
            name,
            receiver_root,
            ..
        } = &target
        else {
            return OpKind::Call {
                target,
                args,
                result_ty,
            };
        };
        if name != "branch"
            || !receiver_root.as_deref().is_some_and(|root| {
                majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(root))
                    == "Result"
            })
        {
            return OpKind::Call {
                target,
                args,
                result_ty,
            };
        }
        let Some(recv_ty) = recv_ty else {
            return OpKind::Call {
                target,
                args,
                result_ty,
            };
        };
        if !crate::front::result_exc::tyref_is_result(recv_ty, self.llbc) {
            return OpKind::Call {
                target,
                args,
                result_ty,
            };
        }
        // `Result<T, E>::branch` returns
        // `ControlFlow<Result<Infallible, E>, T>`. `Break`'s payload is
        // that `Result`, not `E`. Split-eligible Result instantiations
        // annotate as that class (`bookkeeper.py` `getuniqueclassdef`),
        // so `tyref_to_value_type_with` banks `Result<Infallible, E>` as
        // `Ref(Some("Result<Infallible,E>"))`. That kind is not `E`. The
        // match builds `Err(e)` when the two kinds differ. rustc lays
        // this `Result` out as `E` with no discriminant, because
        // `Ok(Infallible)` is uninhabited; the translator does not
        // collapse the kind to `E`.
        let payloads = crate::model::ResultBranchPayloads {
            ok: adt_variant_payload_type(
                recv_ty,
                "Ok",
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            err: adt_variant_payload_type(
                recv_ty,
                "Err",
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            continue_ty: adt_variant_payload_type(
                dest_ty,
                "Continue",
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            break_ty: adt_variant_payload_type(
                dest_ty,
                "Break",
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
        };
        OpKind::Call {
            target: target.with_branch_payloads(payloads),
            args,
            result_ty,
        }
    }

    pub(in crate::front::mir) fn layout_of_tyref(
        &self,
        ty: &TyRef,
    ) -> Option<majit_charon_reader::ullbc::TypeLayout> {
        let def_id = self
            .tyref_adt_def_id(ty)
            .or_else(|| self.tyref_ref_adt_def_id(ty))?;
        let target = std::env::var("TARGET").unwrap_or_default();
        self.llbc
            .type_by_id(def_id)?
            .layout_for_target(self.llbc, &target)
    }

    pub(in crate::front::mir) fn resolve_result_return_owners(
        &self,
        output_ty: &TyRef,
    ) -> Option<ResultTryReturnOwners> {
        let output_ty = self.peel_to_option_or_result(output_ty)?;
        if !crate::front::result_exc::tyref_is_result(&output_ty, self.llbc) {
            return None;
        }
        let (result_owner, _ok_owner, err_owner, _ok_ty, err_ty, _err_class) =
            self.resolve_result_owners(&output_ty)?;
        Some(ResultTryReturnOwners {
            result_owner,
            err_owner,
            err_ty,
            err_ast: crate::front::result_exc::tyref_result_err(&output_ty, self.llbc)
                .map(|ty| tyref_to_ast_string(&ty, self.llbc))
                .unwrap_or_default(),
        })
    }

    pub(in crate::front::mir) fn resolve_result_owners(
        &self,
        ty: &TyRef,
    ) -> Option<(String, String, String, ValueType, ValueType, Option<String>)> {
        let ok_tyref = crate::front::result_exc::tyref_result_ok(ty, self.llbc)?;
        let err_tyref = crate::front::result_exc::tyref_result_err(ty, self.llbc)?;
        let def_id = self.tyref_adt_def_id(ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        let result_owner = format!(
            "{}{}",
            td.item_meta.name_path_str(),
            tyref_enum_instantiation_suffix(ty, self.llbc, self.tombstoned_leaves)
        );
        let ok_owner = Self::tagged_pair_payload_owner(td, &result_owner, 0)?;
        let err_owner = Self::tagged_pair_payload_owner(td, &result_owner, 1)?;
        Some((
            result_owner,
            ok_owner,
            err_owner,
            tyref_enum_payload_value_type(
                &ok_tyref,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            tyref_enum_payload_value_type(
                &err_tyref,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            enum_payload_instance_class_root(&err_tyref, self.llbc, self.tombstoned_leaves),
        ))
    }

    pub(in crate::front::mir) fn peel_to_option_or_result(&self, ty: &TyRef) -> Option<TyRef> {
        let peeled = self
            .tyref_peel_ref_to_pointee(ty)
            .unwrap_or_else(|| clone_tyref(ty));
        if crate::front::result_exc::tyref_is_option(&peeled, self.llbc)
            || crate::front::result_exc::tyref_is_result(&peeled, self.llbc)
        {
            Some(peeled)
        } else {
            None
        }
    }

    pub(in crate::front::mir) fn recognize_disc_combinator_site(
        &self,
        kind: DiscCombinator,
        call_args: &[LinkArg],
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        fn_item_decl: Option<&FunDecl>,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<DiscCombinatorSite> {
        let recv_ty = self.peel_to_option_or_result(recv_ty?)?;
        let is_option = crate::front::result_exc::tyref_is_option(&recv_ty, self.llbc);
        let is_result = crate::front::result_exc::tyref_is_result(&recv_ty, self.llbc);
        match kind {
            DiscCombinator::OptionFilter
            | DiscCombinator::OptionOkOr
            | DiscCombinator::OptionOkOrElse
                if !is_option =>
            {
                return None;
            }
            DiscCombinator::ResultMap
            | DiscCombinator::ResultAndThen
            | DiscCombinator::ResultUnwrapOrElse
            | DiscCombinator::ResultOrElse
            | DiscCombinator::ResultOk
            | DiscCombinator::ResultErr
            | DiscCombinator::ResultIsOk
            | DiscCombinator::ResultIsErr
                if !is_result =>
            {
                return None;
            }
            _ => {}
        }
        let carrier_recv = is_result
            && crate::front::result_exc::tyref_is_result_of_carrier(
                &recv_ty,
                self.llbc,
                self.static_addrs.error_carrier,
            );
        if carrier_recv
            && matches!(
                kind,
                DiscCombinator::ResultUnwrapOrElse
                    | DiscCombinator::ResultOrElse
                    | DiscCombinator::ResultOk
                    | DiscCombinator::ResultErr
            )
        {
            // Carrier Results whose combinator would rebuild a Result shell
            // are `result_exc`'s domain.  `map`/`and_then` still lower, as an
            // if/else on the receiver built before `result_exc` runs.
            return None;
        }

        let mut site = DiscCombinatorSite {
            kind,
            result_var: result_var.clone(),
            call_args: call_args.to_vec(),
            recv_owner: String::new(),
            recv_tag0_owner: String::new(),
            recv_tag1_owner: String::new(),
            payload0_ty: ValueType::Ref(None),
            payload1_ty: ValueType::Ref(None),
            payload0_class: None,
            payload1_class: None,
            payload0_is_raw_u8: false,
            payload1_is_raw_u8: false,
            result_owner: String::new(),
            result_tag0_owner: String::new(),
            result_tag1_owner: String::new(),
            result_payload0_ty: ValueType::Ref(None),
            result_payload1_ty: ValueType::Ref(None),
            result_payload0_class: None,
            result_payload1_class: None,
            call_once_owner: String::new(),
            fn_item_segments: None,
            fn_item_ctor: None,
            args_tuple_suffix: String::new(),
            call_result_ty: ValueType::Ref(None),
            call_result_class: None,
            carrier_recv,
            recv_niche: false,
            scalar_niche: false,
            result_niche: false,
            result_scalar_niche: false,
            niche_null_cast: None,
            int_null: None,
        };

        if is_option {
            let (option_owner, some_owner, mut payload_ty) =
                self.resolve_option_consumer_owners(&recv_ty)?;
            site.recv_owner = option_owner;
            site.recv_tag1_owner = some_owner;
            site.payload1_class = self.option_payload_instance_class_root(&recv_ty);
            site.recv_niche = self.tyref_is_niche_option_ptr(&recv_ty);
            let scalar_kind = if site.recv_niche {
                None
            } else {
                option_nonzero_word_kind(&recv_ty, self.llbc)
            };
            site.scalar_niche = scalar_kind.is_some();
            if let Some(kind) = scalar_kind {
                payload_ty = kind;
            }
            site.payload1_ty = payload_ty;
            site.payload1_is_raw_u8 =
                crate::front::result_exc::tyref_option_payload(&recv_ty, self.llbc)
                    .is_some_and(|ty| tyref_is_raw_u8_ptr(&ty, self.llbc));
            site.niche_null_cast = self.option_niche_null_cast(&recv_ty);
            site.int_null = self.option_niche_int_null(&recv_ty);
        } else {
            let (owner, ok_owner, err_owner, ok_ty, err_ty, err_class) =
                self.resolve_result_owners(&recv_ty)?;
            site.recv_owner = owner;
            site.recv_tag0_owner = ok_owner;
            site.recv_tag1_owner = err_owner;
            site.payload0_ty = ok_ty;
            site.payload1_ty = err_ty;
            site.payload0_is_raw_u8 =
                crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                    .is_some_and(|ty| tyref_is_raw_u8_ptr(&ty, self.llbc));
            site.payload1_is_raw_u8 =
                crate::front::result_exc::tyref_result_err(&recv_ty, self.llbc)
                    .is_some_and(|ty| tyref_is_raw_u8_ptr(&ty, self.llbc));
            site.payload0_class = crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                .and_then(|ty| {
                    enum_payload_instance_class_root(&ty, self.llbc, self.tombstoned_leaves)
                });
            site.payload1_class = err_class;
        }

        // Prefer a closure ADT (`call_once(env, (x,))`); a function item has no
        // ADT def id and is called directly, or built when it is a constructor.
        let mut fn_item = None;
        if kind.needs_closure() {
            match env_ty.and_then(|ty| self.tyref_ref_adt_def_id(ty)) {
                Some(env_def_id) => {
                    site.call_once_owner =
                        closure_env_call_once_owner(self.llbc.type_by_id(env_def_id)?);
                }
                None => {
                    let fd = fn_item_decl?;
                    let segments = fundecl_fn_item_segments(self.llbc, fd);
                    if segments.is_empty() {
                        return None;
                    }
                    site.fn_item_segments = Some(segments);
                    fn_item = Some(fd);
                }
            }
        }

        match kind {
            DiscCombinator::OptionFilter => {
                let (option_owner, some_owner, payload_ty) =
                    self.resolve_option_consumer_owners(dest_ty)?;
                site.result_owner = option_owner;
                site.result_tag1_owner = some_owner;
                site.result_payload1_ty = payload_ty;
                site.result_payload1_class = self.option_payload_instance_class_root(dest_ty);
                site.result_niche = self.tyref_is_niche_option_ptr(dest_ty);
                site.result_scalar_niche =
                    !site.result_niche && tyref_option_nonzero_scalar(dest_ty, self.llbc);
                if let Some(cast) = self.option_niche_null_cast(dest_ty) {
                    site.niche_null_cast = Some(cast);
                }
                site.call_result_ty = ValueType::Bool;
                site.args_tuple_suffix = option_payload_tuple_suffix(&recv_ty, self.llbc);
            }
            DiscCombinator::OptionOkOr | DiscCombinator::OptionOkOrElse => {
                let dest = self.peel_to_option_or_result(dest_ty)?;
                let (owner, ok_owner, err_owner, ok_ty, err_ty, err_class) =
                    self.resolve_result_owners(&dest)?;
                site.result_owner = owner;
                site.result_tag0_owner = ok_owner;
                site.result_tag1_owner = err_owner;
                site.result_payload0_ty = ok_ty;
                site.result_payload1_ty = err_ty.clone();
                site.result_payload0_class = site.payload1_class.clone();
                if kind == DiscCombinator::OptionOkOrElse {
                    // `F: FnOnce() -> E`.  The None arm calls `f` and wraps
                    // `E` as `Err`.
                    site.result_payload1_class = err_class.clone();
                    site.call_result_ty = err_ty;
                    site.call_result_class = err_class;
                } else {
                    site.result_payload1_class = env_ty
                        .and_then(|ty| {
                            self.tyref_peel_ref_to_pointee(ty)
                                .or_else(|| Some(clone_tyref(ty)))
                        })
                        .and_then(|ty| {
                            enum_payload_instance_class_root(&ty, self.llbc, self.tombstoned_leaves)
                        });
                }
            }
            DiscCombinator::ResultMap | DiscCombinator::ResultAndThen => {
                let dest = self.peel_to_option_or_result(dest_ty)?;
                if !crate::front::result_exc::tyref_is_result(&dest, self.llbc) {
                    return None;
                }
                let (owner, ok_owner, err_owner, ok_ty, err_ty, err_class) =
                    self.resolve_result_owners(&dest)?;
                site.result_owner = owner;
                site.result_tag0_owner = ok_owner;
                site.result_tag1_owner = err_owner;
                site.result_payload0_ty = ok_ty.clone();
                site.result_payload1_ty = err_ty;
                site.result_payload1_class = err_class;
                if kind == DiscCombinator::ResultMap {
                    site.call_result_ty = ok_ty;
                    site.call_result_class = crate::front::result_exc::tyref_result_ok(
                        &dest, self.llbc,
                    )
                    .and_then(|ty| {
                        enum_payload_instance_class_root(&ty, self.llbc, self.tombstoned_leaves)
                    });
                    site.result_payload0_class = site.call_result_class.clone();
                    site.args_tuple_suffix =
                        crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                            .map(|ty| payload_tuple_suffix(&ty, self.llbc))
                            .unwrap_or_default();
                } else {
                    site.call_result_ty = self.value_type_of(&dest);
                    site.args_tuple_suffix =
                        crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                            .map(|ty| payload_tuple_suffix(&ty, self.llbc))
                            .unwrap_or_default();
                }
            }
            DiscCombinator::ResultUnwrapOrElse => {
                site.call_result_ty = self.value_type_of(dest_ty);
                site.args_tuple_suffix =
                    crate::front::result_exc::tyref_result_err(&recv_ty, self.llbc)
                        .map(|ty| payload_tuple_suffix(&ty, self.llbc))
                        .unwrap_or_default();
            }
            DiscCombinator::ResultOrElse => {
                let dest = self.peel_to_option_or_result(dest_ty)?;
                let (owner, ok_owner, err_owner, ok_ty, err_ty, err_class) =
                    self.resolve_result_owners(&dest)?;
                site.result_owner = owner;
                site.result_tag0_owner = ok_owner;
                site.result_tag1_owner = err_owner;
                site.result_payload0_ty = ok_ty;
                site.result_payload1_ty = err_ty;
                site.result_payload0_class = site.payload0_class.clone();
                site.result_payload1_class = err_class;
                site.call_result_ty = self.value_type_of(&dest);
                site.args_tuple_suffix =
                    crate::front::result_exc::tyref_result_err(&recv_ty, self.llbc)
                        .map(|ty| payload_tuple_suffix(&ty, self.llbc))
                        .unwrap_or_default();
            }
            DiscCombinator::ResultOk | DiscCombinator::ResultErr => {
                let dest = self.peel_to_option_or_result(dest_ty)?;
                if !crate::front::result_exc::tyref_is_option(&dest, self.llbc) {
                    return None;
                }
                let (option_owner, some_owner, payload_ty) =
                    self.resolve_option_consumer_owners(&dest)?;
                site.result_owner = option_owner;
                site.result_tag1_owner = some_owner;
                site.result_payload1_ty = payload_ty;
                site.result_payload1_class = self.option_payload_instance_class_root(&dest);
            }
            DiscCombinator::ResultIsOk | DiscCombinator::ResultIsErr => {}
        }
        if let Some(fd) = fn_item
            && fd.is_adt_constructor()
        {
            // The type the constructor builds: the callable's result.
            let built = match kind {
                DiscCombinator::ResultMap => crate::front::result_exc::tyref_result_ok(
                    &self.peel_to_option_or_result(dest_ty)?,
                    self.llbc,
                )?,
                DiscCombinator::ResultAndThen | DiscCombinator::ResultOrElse => {
                    self.peel_to_option_or_result(dest_ty)?
                }
                DiscCombinator::ResultUnwrapOrElse => dest_ty.clone(),
                _ => return None,
            };
            site.fn_item_ctor = Some(self.constructor_fn_item_shape(fd, &built)?);
        }
        Some(site)
    }

    /// Resolve `Option::ok_or_else(opt, closure) -> Result<T, E>` for the
    /// combined Option-select + immediate-`?` exception rewrite.
    ///
    /// The destination must use the configured exception carrier; this keeps
    /// ordinary `ok_or_else` values and non-Python error domains as residual
    /// Rust combinators.  The post-pass additionally proves the compiler's
    /// private `Try::branch` diamond before it mutates the graph.
    pub(in crate::front::mir) fn recognize_option_ok_or_else_try_site(
        &self,
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::result_exc::OptionOkOrElseTrySite> {
        let recv_ty = recv_ty?;
        if !crate::front::result_exc::tyref_is_option(recv_ty, self.llbc)
            || !crate::front::result_exc::tyref_is_result_of_carrier(
                dest_ty,
                self.llbc,
                self.static_addrs.error_carrier,
            )
        {
            return None;
        }
        let (option_owner, some_owner, mut payload_ty) =
            self.resolve_option_consumer_owners(recv_ty)?;
        let scalar_kind = option_nonzero_word_kind(recv_ty, self.llbc);
        let scalar_niche = scalar_kind.is_some();
        if let Some(kind) = scalar_kind {
            payload_ty = kind;
        }
        let env_def_id = self.tyref_ref_adt_def_id(env_ty?)?;
        let call_once_owner = closure_env_call_once_owner(self.llbc.type_by_id(env_def_id)?);
        let error_ty = crate::front::result_exc::tyref_result_err(dest_ty, self.llbc)
            .map(|ty| self.value_type_of(&ty))
            .unwrap_or(ValueType::Ref(None));
        Some(crate::front::result_exc::OptionOkOrElseTrySite {
            result_var: result_var.clone(),
            option_owner,
            some_owner,
            call_once_owner,
            payload_ty,
            error_ty,
            niche: self.tyref_is_niche_option_ptr(recv_ty),
            scalar_niche,
        })
    }

    /// Resolve a dest-is-carrier `map_err` into the owners and payload
    /// types needed by the LastException rewrite in `front::result_map_err`.
    pub(in crate::front::mir) fn recognize_result_map_err_site(
        &self,
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        fn_item: Option<&[String]>,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::result_map_err::ResultMapErrSite> {
        let recv_ty = recv_ty?;
        let recv_ok = crate::front::result_exc::tyref_result_ok(recv_ty, self.llbc)?;
        let recv_err = crate::front::result_exc::tyref_result_err(recv_ty, self.llbc)?;
        let dest_ok = crate::front::result_exc::tyref_result_ok(dest_ty, self.llbc)?;
        let dest_err = crate::front::result_exc::tyref_result_err(dest_ty, self.llbc)?;

        let recv_decl = self.llbc.type_by_id(self.tyref_adt_def_id(recv_ty)?)?;
        let dest_decl = self.llbc.type_by_id(self.tyref_adt_def_id(dest_ty)?)?;
        let receiver_owner = format!(
            "{}{}",
            recv_decl.item_meta.name_path_str(),
            tyref_enum_instantiation_suffix(recv_ty, self.llbc, self.tombstoned_leaves)
        );
        let result_owner = format!(
            "{}{}",
            dest_decl.item_meta.name_path_str(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc, self.tombstoned_leaves)
        );
        let receiver_ok_owner = Self::tagged_pair_payload_owner(recv_decl, &receiver_owner, 0)?;
        let receiver_err_owner = Self::tagged_pair_payload_owner(recv_decl, &receiver_owner, 1)?;
        let result_err_owner = Self::tagged_pair_payload_owner(dest_decl, &result_owner, 1)?;
        // `map_err(named_fn)` passes a function item, which has no closure
        // ADT. `map_err(|e| …)` passes the closure env.
        let (call_once_owner, fn_item_segments, closure_env_is_trivially_dropless) =
            if let Some(env_ty) = env_ty {
                let env_decl = self.llbc.type_by_id(self.tyref_ref_adt_def_id(env_ty)?)?;
                (
                    closure_env_call_once_owner(env_decl),
                    None,
                    self.type_decl_is_trivially_dropless(env_decl, 0),
                )
            } else {
                let segments = fn_item?.to_vec();
                (String::new(), Some(segments), true)
            };

        Some(crate::front::result_map_err::ResultMapErrSite {
            result_var: result_var.clone(),
            receiver_owner,
            receiver_ok_owner,
            receiver_err_owner,
            result_owner,
            result_err_owner,
            call_once_owner,
            ok_ty: if crate::front::result_exc::tyref_result_ok_is_unit(recv_ty, self.llbc)
                || is_unit_type(&recv_ok, self.llbc)
                || tyref_is_void_zst(&recv_ok, self.llbc)
            {
                // `()` is `lltype.Void`. `tyref_to_value_type` and
                // `result_exc` `unit_ok` / `widen_unit_return_to_void`
                // keep unit Ok as Void (`exceptiontransform.py`).
                // `tyref_result_ok_is_unit` reads the Ok slot as `"()"`.
                ValueType::Void
            } else {
                tyref_enum_payload_value_type(
                    &recv_ok,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            },
            ok_class_root: enum_payload_instance_class_root(
                &dest_ok,
                self.llbc,
                self.tombstoned_leaves,
            ),
            err_ty: tyref_enum_payload_value_type(
                &recv_err,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            err_class_root: enum_payload_instance_class_root(
                &recv_err,
                self.llbc,
                self.tombstoned_leaves,
            ),
            mapped_err_ty: tyref_enum_payload_value_type(
                &dest_err,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            mapped_err_class_root: enum_payload_instance_class_root(
                &dest_err,
                self.llbc,
                self.tombstoned_leaves,
            ),
            args_tuple_suffix: payload_tuple_suffix(&recv_err, self.llbc),
            fn_item_segments,
            closure_env_is_trivially_dropless,
        })
    }

    /// Conservative `needs_drop::<ClosureEnv>() == false` projection for the
    /// compiler-generated closure types consumed by `Result::map_err`.
    pub(in crate::front::mir) fn type_decl_is_trivially_dropless(
        &self,
        decl: &TypeDecl,
        depth: usize,
    ) -> bool {
        type_decl_is_trivially_dropless(decl, self.llbc, depth)
    }

    pub(in crate::front::mir) fn tyref_is_trivially_dropless(
        &self,
        ty: &TyRef,
        depth: usize,
    ) -> bool {
        tyref_is_trivially_dropless(ty, self.llbc, depth)
    }

    /// `true` when `ty` resolves to a fieldless (C-like) enum — at least
    /// one variant and every variant carrying zero payload fields.  Such
    /// an enum is represented by-value as its discriminant integer, so
    /// `Rvalue::Discriminant` on it is the identity on that value (see the
    /// `Discriminant` lowering).  Payload-carrying enums return `false`
    /// and keep the `__discriminant` field read against their aggregate
    /// `Ref` base.
    pub(in crate::front::mir) fn tyref_is_fieldless_enum(&self, ty: &TyRef) -> bool {
        self.tyref_adt_def_id(ty)
            .and_then(|def_id| self.llbc.type_by_id(def_id))
            .is_some_and(|td| type_decl_is_fieldless_enum(td, self.llbc))
    }

    /// [`Self::tyref_is_fieldless_enum`] through a `&` borrow — the shape a
    /// derived trait method sees, since `PartialEq::eq(&self, &other)` types
    /// both operands as references.  A fieldless enum is modelled by-value as
    /// its discriminant integer, and `Rvalue::Ref` is the identity on it, so
    /// the borrow carries no representation of its own.
    pub(in crate::front::mir) fn tyref_is_borrowed_fieldless_enum(&self, ty: &TyRef) -> bool {
        self.tyref_ref_adt_def_id(ty)
            .and_then(|def_id| self.llbc.type_by_id(def_id))
            .is_some_and(|td| type_decl_is_fieldless_enum(td, self.llbc))
    }

    /// Niche `Option` whose remaining word is an int-bank address.
    ///
    /// `rnone.py` `rtype_is_None` compares an operand against a null of its
    /// own lltype (`Ptr` → `ptr_iszero`, Address / non-GC funcptr →
    /// `int_is_zero`); `rmodel.py` `rtype_is_` never mixes a GC pointer with
    /// an Address. [`tyref_to_value_type_with`] already puts `Option<fn>`,
    /// a raw-address niche ([`ty_is_raw_address`], e.g. `Option<&RawEnum>`),
    /// and a pointer-niche `Option<&RawStruct>` ([`tyref_niche_option_payload_paint`])
    /// in `Int`; a GC pointer / `SomeString` niche stays `Ref`.
    pub(in crate::front::mir) fn option_niche_is_int_bank(&self, option_ty: &TyRef) -> bool {
        if !self.tyref_is_niche_option_ptr(option_ty) {
            return false;
        }
        matches!(
            self.value_type_of(option_ty),
            ValueType::Int | ValueType::Unsigned
        )
    }

    /// Int-bank null of a niche `Option`: `null_fn` with the `fn(..)`
    /// spelling, or `null_raw` with the raw pointee's class root
    /// (`lltype.nullptr` / `llmemory.NULL`). A storage-raw struct
    /// borrow Option is Int (`tyref_niche_option_payload_paint`) and
    /// takes the first arm; the struct arm is the fallback when that
    /// paint is not yet the Option's value type. `null_mut` is a GC
    /// Instance and recasting it to `Ptr(Struct)` is refused
    /// (`pairtype(GCRefRepr, Repr).convert_from_to` / `rgcref.py`;
    /// `try_lower_raw_adt_ptr_null`). `None` when the
    /// niche is a GC / string Ref (`push_niche_null`).
    pub(in crate::front::mir) fn option_niche_int_null(
        &self,
        option_ty: &TyRef,
    ) -> Option<crate::model::NicheIntNull> {
        if self.option_niche_is_int_bank(option_ty) {
            // `executioncontext.py` `profilefunc`: `None` is `s_None`, not
            // `lltype.nullptr(FuncType)`. A null_fn `SomePtr` cannot union
            // with the field's `SomePBC`.
            if tyref_is_profilefunc_family(option_ty, self.llbc) {
                return None;
            }
            if tyref_option_payload_is_fn_ptr(option_ty, self.llbc)
                || tyref_node(option_ty, self.llbc)
                    .is_some_and(|node| type_node_is_fn_ptr(node, self.llbc))
            {
                return Some(crate::model::NicheIntNull::FnPtr(tyref_fn_ptr_spelling(
                    option_ty, self.llbc,
                )));
            }
            let seed = crate::front::result_exc::tyref_option_payload(option_ty, self.llbc)
                .unwrap_or_else(|| option_ty.clone());
            return Some(crate::model::NicheIntNull::RawAddress(
                raw_address_owner_root(
                    &seed,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
            ));
        }
        if self.payload_storage_raw_pointee_is_struct(option_ty)
            && let Some(root) = self.payload_storage_raw_struct_root(option_ty)
        {
            return Some(crate::model::NicheIntNull::RawAddress(Some(root)));
        }
        None
    }

    /// `core::option::<Impl>::is_some` / `is_none` on a niche `Option<fn>`.
    ///
    /// `rnone.py` `rtype_is_None` compares the word with `lltype.nullptr`
    /// of its `FuncType` (`null_fn`). `is_none` is `flowcontext.py`
    /// `cmp_is`: the `is_` op left as a value. `is_some` is `cmp_is_not`,
    /// which is `not_` of that `is_`. `not_` runs `guessbool` on the
    /// `is_` result, so the result is this block's exitswitch and each
    /// exit continues with a constant (`False` on the is-None exit,
    /// `True` on the other). Nothing computes `not` as a value. A
    /// non-fn niche stays a call: `Option<PyObjectRef>::is_some` is
    /// still the tagged discriminant (`callable_w`).
    pub(in crate::front::mir) fn lower_fn_niche_option_predicate(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let Some(name) = regular_call_name_path(reg, self.llbc) else {
            return Ok(false);
        };
        let is_some = matches!(
            name.as_str(),
            "core::option::<Impl>::is_some"
                | "core::option::Option::is_some"
                | "option::Option::is_some"
        );
        let is_none = matches!(
            name.as_str(),
            "core::option::<Impl>::is_none"
                | "core::option::Option::is_none"
                | "option::Option::is_none"
        );
        if (!is_some && !is_none) || call.args.len() != 1 {
            return Ok(false);
        }
        let Some(arg_ty) = operand_tyref(&call.args[0]) else {
            return Ok(false);
        };
        let owned = tyref_shared_borrow_option_pointee(arg_ty, self.llbc);
        let option_ty = owned.as_ref().unwrap_or(arg_ty);
        if !tyref_option_payload_is_fn_ptr(option_ty, self.llbc) {
            return Ok(false);
        }
        let Some(int_null) = self.option_niche_int_null(option_ty) else {
            return Ok(false);
        };
        if !matches!(int_null, crate::model::NicheIntNull::FnPtr(_)) {
            return Ok(false);
        }
        let recv = self.resolve_operand(mir_bb, call.args[0].clone())?;
        let bb_id = self.emit_bb(mir_bb);
        let nullc = int_null.push(&mut self.graph, bb_id);
        let v_is = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(v_is.clone()),
            kind: OpKind::BinOp {
                op: "is_".to_string(),
                lhs: recv,
                rhs: nullc,
                result_ty: ValueType::Int,
            },
        });
        if is_some {
            // `cmp_is_not` / `not_` / `guessbool`: the `is_` is the
            // exitswitch. The True exit (word is None) carries
            // `Const(false)`; the False exit carries `Const(true)`.
            self.close_fn_niche_is_some_fork(mir_bb, bb_id, dest_local, target, v_is)?;
        } else {
            // `cmp_is`: the `is_` itself is the value that flows out.
            self.local_var[dest_local] = Some(LocalValue::One(v_is));
            let target_bb = self.block_id[target];
            let link_args = self.edge_args(mir_bb, target)?;
            self.graph.set_goto(bb_id, target_bb, link_args);
        }
        Ok(true)
    }

    /// `Constant(bool, lltype.Bool)` for one `is_some` link slot.
    fn fn_niche_is_some_link_const(is_some: bool) -> crate::model::LinkArg {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        crate::model::LinkArg::Const(crate::flowspace::model::Constant::with_concretetype(
            crate::flowspace::model::ConstValue::Bool(is_some),
            LowLevelType::Bool,
        ))
    }

    /// Close `is_some` the way `flowcontext.py` `guessbool` closes the
    /// block inside `not_` (`cmp_is_not`).
    ///
    /// False link first, then True, matching `guessbool`'s
    /// `[False, True]` eggs and [`FunctionGraph::set_branch`]. The
    /// constant on `dest_local` is `not` of that exit
    /// (`Constant(..., lltype.Bool)`, the same concretetype
    /// `optimize_goto_if_not` stamps). Every other edge arg is shared.
    /// `set_branch` cannot express a per-link constant and always wraps
    /// `bool()`, so the links are [`Link::new_mixed`].
    ///
    /// When `dest_local` is not one word of the target's inputargs, two
    /// intermediate blocks each `ConstBool` the dest and `set_goto` the
    /// target — the graph `guessbool` produces before `join_blocks`.
    fn close_fn_niche_is_some_fork(
        &mut self,
        mir_bb: usize,
        bb_id: crate::model::BlockId,
        dest_local: usize,
        target: usize,
        v_is: Variable,
    ) -> Result<(), LowerError> {
        let inputs = self.target_input_locals(target)?;
        let slot = inputs.iter().position(|local| *local == dest_local);
        let view = self.string_byte_view_locals.contains(&dest_local)
            || self.list_view_locals.contains(&dest_local);
        if slot.is_none() || view {
            return self.close_fn_niche_is_some_eggs(mir_bb, bb_id, dest_local, target, v_is);
        }
        let slot = slot.expect("dest_local is a target input");
        // `edge_args` reads `local_var`. The placeholder only reserves
        // the dest slot; the links below replace it and the variable
        // is not an operation result, so it is not a flowing value.
        let placeholder = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.local_var[dest_local] = Some(LocalValue::One(placeholder.clone()));
        let link_vars = self.edge_args(mir_bb, target)?;
        if link_vars.get(slot) != Some(&placeholder) {
            return Err(LowerError::Unsupported(
                "Option<fn>::is_some dest is not one link word".to_string(),
            ));
        }
        let target_bb = self.block_id[target];
        let mut false_args: Vec<crate::model::LinkArg> = link_vars
            .iter()
            .cloned()
            .map(crate::model::LinkArg::Value)
            .collect();
        let mut true_args = false_args.clone();
        for (index, var) in link_vars.iter().enumerate() {
            if var == &placeholder {
                // False exit of `is_`: the word is not None, so `is_some`.
                false_args[index] = Self::fn_niche_is_some_link_const(true);
                // True exit of `is_`: the word is None, so not `is_some`.
                true_args[index] = Self::fn_niche_is_some_link_const(false);
            }
        }
        assert_eq!(
            false_args.len(),
            self.graph.block(target_bb).inputargs.len(),
            "is_some fork args must match the target inputargs"
        );
        let false_link = crate::model::Link::new_mixed(
            false_args,
            target_bb,
            Some(crate::model::ExitCase::Bool(false)),
        )
        .with_llexitcase_from_exitcase();
        let true_link = crate::model::Link::new_mixed(
            true_args,
            target_bb,
            Some(crate::model::ExitCase::Bool(true)),
        )
        .with_llexitcase_from_exitcase();
        self.graph.set_control_flow_metadata(
            bb_id,
            Some(crate::model::ExitSwitch::Value(v_is)),
            vec![false_link, true_link],
        );
        Ok(())
    }

    /// Two `guessbool` eggs. Each binds `dest_local` to its own
    /// `ConstBool` and gotos the call target. Other edge args are
    /// threaded as egg inputargs so the goto's operands are defined
    /// in the egg (`close_call_diamond`).
    ///
    /// The eggs are recorded on `exception_cascade_blocks` so
    /// `lower_framestate` walks through them to the MIR target
    /// (`framestate_mir_successors`) and does not rewrite their links
    /// in pass 2. Otherwise pass 1 would `stub_dead_block` the target.
    fn close_fn_niche_is_some_eggs(
        &mut self,
        mir_bb: usize,
        bb_id: crate::model::BlockId,
        dest_local: usize,
        target: usize,
        v_is: Variable,
    ) -> Result<(), LowerError> {
        let placeholder = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.local_var[dest_local] = Some(LocalValue::One(placeholder.clone()));
        let link_vars = self.edge_args(mir_bb, target)?;
        let target_bb = self.block_id[target];
        let mut carried: Vec<Variable> = Vec::new();
        for var in &link_vars {
            if var != &placeholder && !carried.contains(var) {
                carried.push(var.clone());
            }
        }
        let (false_egg, false_inputs) = self.graph.create_block_with_arg_vars(carried.len());
        let (true_egg, true_inputs) = self.graph.create_block_with_arg_vars(carried.len());
        self.emit_fn_niche_is_some_egg(
            false_egg,
            &false_inputs,
            true,
            &link_vars,
            &placeholder,
            &carried,
            target_bb,
        );
        self.emit_fn_niche_is_some_egg(
            true_egg,
            &true_inputs,
            false,
            &link_vars,
            &placeholder,
            &carried,
            target_bb,
        );
        let false_link = crate::model::Link::from_variables(
            &self.graph,
            carried.clone(),
            false_egg,
            Some(crate::model::ExitCase::Bool(false)),
        )
        .with_llexitcase_from_exitcase();
        let true_link = crate::model::Link::from_variables(
            &self.graph,
            carried,
            true_egg,
            Some(crate::model::ExitCase::Bool(true)),
        )
        .with_llexitcase_from_exitcase();
        self.graph.set_control_flow_metadata(
            bb_id,
            Some(crate::model::ExitSwitch::Value(v_is)),
            vec![false_link, true_link],
        );
        self.exception_cascade_blocks.push(false_egg);
        self.exception_cascade_blocks.push(true_egg);
        Ok(())
    }

    fn emit_fn_niche_is_some_egg(
        &mut self,
        egg: crate::model::BlockId,
        inputs: &[Variable],
        is_some_value: bool,
        link_vars: &[Variable],
        placeholder: &Variable,
        carried: &[Variable],
        target_bb: crate::model::BlockId,
    ) {
        let produced = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(egg).operations.push(SpaceOperation {
            result: Some(produced.clone()),
            kind: OpKind::ConstBool(is_some_value),
        });
        let args = remap_diamond_goto_args(link_vars, placeholder, &produced, carried, inputs);
        self.graph.set_goto(egg, target_bb, args);
    }

    /// `true` when `ty` is represented as a one-word nullable `Option` in
    /// the translated RPython model:
    /// `Option<NonNull<T>>`, `Option<Box<T>>`, `Option<fn(..)>`, `Option<&mut T>`,
    /// `Option<&T>` with a thin (Sized) ADT pointee, or a raw pointer to a
    /// thin nominal ADT. It also includes the exact Rust container spellings
    /// that translate to one RPython GC pointer: `Option<Vec<T>>` of a
    /// multi-word item becomes a nullable `SomeList`, a shared
    /// `Option<&[PyObjectRef]>` is the same nullable GcArray list
    /// (`rlist.py` `FixedSizeListRepr` / `ll_fixed_items`), and
    /// `Option<&str>` / `Option<Cow<str-family>>` (and the other
    /// string-family payloads [`node_is_string_value`] /
    /// [`tyref_is_cow_str`] already project) become a nullable
    /// `SomeString`. A rust-heap `Option<String>` paints `Ref`.
    /// The JIT models each in ONE pointer word (`None` = null, `Some(p)` =
    /// the non-null payload pointer), so
    /// `Discriminant` on it is a pointer-null test (`base != null`) and the
    /// `Some` payload read is the identity on that pointer — no aggregate
    /// `__discriminant` / `__pos_0` field exists. This mirrors the
    /// fieldless-enum by-value model ([`Self::tyref_is_fieldless_enum`]).
    ///
    /// Raw-pointer payloads are accepted only when their pointee resolves to
    /// a nominal ADT with a concrete layout. Such pointers carry the object
    /// aggregate address in these graphs, and `Some(p)` therefore aliases the
    /// base pointer. A scalar-pointer payload remains excluded because it may
    /// use `Some(null)` as a state distinct from `None`; collapsing both to
    /// the null word would merge those states. A nominal unsized pointee is
    /// excluded as well: Charon uses the fat-pointer representation for that
    /// raw-pointer shape, so it is not a single payload word.
    ///
    /// Tuple payloads are EXCLUDED, and the exclusion is load-bearing.
    /// Inside a lowered graph a constructed tuple is a heap aggregate, so
    /// `Some(tuple)` looks like a pointer word — but every payload accepted
    /// here is a genuine one-word Rust niche, and `Option<(A, B)>` is not:
    /// Rust lays it out as a discriminant plus the elements. Modelling it as
    /// one nullable word therefore breaks wherever such an `Option` is
    /// produced or consumed outside the lowered graphs, and two tuple shapes
    /// have no aggregate address inside them either — the unit `()` and the
    /// checked-arithmetic `(numeric, bool)` destination, whose local carries
    /// the scalar `.0` value ([`tyref_checked_binop_value_type`]).
    /// `niche_option_tuple_remains_aggregate` pins the exclusion; collapsing
    /// it regressed 12 fixtures on both native backends, one of them on
    /// output. Mutable references are included explicitly: Rust
    /// guarantees their non-null representation, and generated
    /// `#[pyre_class]::from_obj` returns `Option<&mut Self>`. Shared
    /// references `&T` are included when the pointee is a thin (Sized)
    /// nominal ADT — equally a one-word null niche — via
    /// [`type_node_shared_ref_pointee`] gated on the pointee's resolved layout
    /// size being concrete. Fat shared references are otherwise excluded:
    /// `&[T]` / `&dyn` carry no ADT def-id, and an unsized nominal newtype (a
    /// DST such as `Wtf8 { bytes: [u8] }`) DOES carry a def-id but has no
    /// concrete layout size — both are two-word references with no one-word
    /// niche payload. String-family fat refs (`&str`, `&[u8]`, `&Wtf8`) and
    /// by-value `Wtf8Buf` return from the early
    /// [`tyref_is_option_of_string_value`] arm: the front has already
    /// replaced that Rust `(data, len)` spelling with RPython's single
    /// `SomeString` value. PyPy `BuiltinCode.docstring` is exactly
    /// `SomeString(can_be_None=True)` (`gateway.py:getdocstring`), not
    /// a tag + payload aggregate.
    ///
    /// `front::iter_next` interaction (shared arm only). `Iterator::next()`
    /// returns `Option<&T>`, and `front::iter_next` rewrites its
    /// `__discriminant` match diamond into the native `[__iter_next]` op by
    /// matching a literal `FieldRead __discriminant` + `Value` exitswitch
    /// (`iter_next.rs` `rewire_one_next_site`). The `Discriminant` fold below
    /// (`Rvalue::Discriminant`, this file) turns that into an identity-null
    /// bool switch when the recognizer accepts the receiver — for a
    /// shared thin-ADT ref that would fire BEFORE iter_next runs and leave it
    /// no diamond to match, so the residual `next()` call would survive as an
    /// unregistered callee (a census Skip, not a miscompile). Currently
    /// unreached: every `.iter()` for-loop that reaches iter_next iterates a
    /// raw pointer / tuple / int element (none a thin nominal ADT), so
    /// the shared arm never fires on a `next()` result — verified by a clean
    /// census (0 `slice::iter::Iter::next` heads) and 370/370 check.py. The
    /// `&mut` arm never had this exposure: iterator receivers come from
    /// `.iter()` (`core::slice::…::iter`, a SHARED borrow), never `iter_mut`.
    /// The raw-pointer arm matches only a direct payload, not `&RawPtr`, so
    /// it does not expand this iter-next exposure.
    /// If a `.iter()` loop over `&[SizedStruct]` ever appears, teach
    /// iter_next the null-test shape (its matcher must accept the
    /// `niche_disc_vars` bool switch) BEFORE relying on the fold there.
    pub(in crate::front::mir) fn tyref_is_niche_option_ptr(&self, ty: &TyRef) -> bool {
        tyref_is_niche_option(ty, self.llbc)
    }

    /// `true` when `option_ty` is a niche nullable `Option<P>` and `P`
    /// compares by address: a raw pointer (`*mut T` / `*const T`),
    /// `NonNull<T>`, or a function pointer. Under the niche model `None`
    /// is null and `Some(p)` is `p`, so derived equality is address
    /// identity of the two words.
    ///
    /// `&T` compares the pointee by value, `Box<T>` compares by value, and
    /// a string payload compares string contents. A leading reference is
    /// not peeled off `P` before that test, so `Option<&NonNull<T>>` stays
    /// out with the other `&T` payloads. An aggregate Option is not a
    /// niche and stays out too.
    pub(in crate::front::mir) fn option_niche_payload_compares_by_address(
        &self,
        option_ty: &TyRef,
    ) -> bool {
        if !self.tyref_is_niche_option_ptr(option_ty) {
            return false;
        }
        let Some(payload) = tyref_node(option_ty, self.llbc)
            .and_then(|node| node.as_object())
            .and_then(|m| m.get("Adt"))
            .and_then(|a| a.as_object())
            .and_then(|a| type_decl_ref_generics(a, self.llbc))
            .and_then(|g| g.get("types"))
            .and_then(|t| t.as_array())
            .and_then(|t| t.first())
        else {
            return false;
        };
        // The same unique trait-projection peel `tyref_is_niche_option_ptr`
        // applies before recognising the payload, so `Option<H::Value>`
        // bound to `*mut PyObject` is that raw pointer.
        let resolved_assoc = trait_payload_node(payload, self.llbc)
            .and_then(|node| trait_assoc_projection_target(node, self.llbc));
        let resolved_body = resolved_assoc.as_ref().and_then(|ty| self.tyref_body(ty));
        let payload = resolved_body.unwrap_or(payload);
        if type_node_is_fn_ptr(payload, self.llbc)
            || type_node_raw_ptr_pointee(payload, self.llbc).is_some()
        {
            return true;
        }
        let Some(node) = strip_ty_indirections(payload, self.llbc) else {
            return false;
        };
        let Some(def_id) = adt_node_def_id(node) else {
            return false;
        };
        self.llbc
            .type_by_id(def_id)
            .is_some_and(|td| td.item_meta.name_path_str() == "core::ptr::non_null::NonNull")
    }

    /// `true` when `option_ty` is a niche nullable `Option<P>` whose
    /// paint is `ValueType::Str` ([`tyref_is_option_of_string_value`]):
    /// builtin `str`, `Utf8Str` / `BytesBlock`, shared `&str` /
    /// `&Wtf8` / `&Wtf8Buf`, by-value `Wtf8Buf` / `Utf8Str`, or
    /// `Cow<str-family>`.
    /// `Option<&[u8]>` is the two-word `tyref_is_u8_window`, not a niche,
    /// so it stays out. Under the niche model `None` is the null string
    /// and `Some(c)` is the string `c` itself, so derived equality is
    /// string contents — `rstr.py` `AbstractStringRepr.rtype_eq` /
    /// `rtype_ne` → `ll_streq`. A rust-heap `String` / `&String` paints
    /// `Ref` and stays out: `BinOp eq` on two `Ref`s is instance identity
    /// (`rclass.py` `rtype_eq` → `ptr_eq`).
    pub(in crate::front::mir) fn option_niche_payload_compares_as_string(
        &self,
        option_ty: &TyRef,
    ) -> bool {
        self.tyref_is_niche_option_ptr(option_ty)
            && tyref_is_option_of_string_value(option_ty, self.llbc)
    }

    /// Operand of `<Option<P> as PartialEq>::eq` / `ne`: `&Option<P>`
    /// peeled to the niche Option, whose payload compares by address or
    /// as a string.
    pub(in crate::front::mir) fn option_niche_eq_ne_operand(&self, ty: &TyRef) -> bool {
        self.tyref_peel_ref_to_pointee(ty).is_some_and(|peeled| {
            self.option_niche_payload_compares_by_address(&peeled)
                || self.option_niche_payload_compares_as_string(&peeled)
        })
    }

    /// `<Option<P> as PartialEq>::eq` / `ne` on a niche nullable Option.
    /// Shared by the FunctionPath spelling `option::Option::eq` / `ne`
    /// and the `impl_method_owner` Method spelling (`name_path`
    /// `core::option::<Impl>::eq`, `CallTarget::Method { name: "eq" }`).
    /// Operands arrive as `&Option<P>`; peel that one reference.
    /// `Rvalue::Ref` aliases the borrow to the niche word (`Some(p)`
    /// is `p`). Two payload families share this `BinOp`:
    ///
    /// * Address identity (`*mut T` / `*const T`, `NonNull<T>`,
    ///   function pointer) — `rclass.py` `rtype_eq` / `rtype_ne`
    ///   → `ptr_eq` / `ptr_ne`; `rptr.py` `PtrRepr.rtype_eq`.
    /// * String-value paint (`tyref_is_option_of_string_value`:
    ///   `Option<&str>` / `Option<Wtf8Buf>` / `Option<Cow<str>>`
    ///   and the other `Str` Options) — `rstr.py`
    ///   `AbstractStringRepr.rtype_eq` / `rtype_ne` →
    ///   `ll_streq`, which already treats a null operand
    ///   (`if not s1 or not s2: return s1 == s2`).
    ///
    /// A rust-heap `String` / `&String` paints `Ref`; `&T` (pointee
    /// by value), `Box<T>` (by value), and a non-niche aggregate
    /// Option stay residual.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_option_niche_eq_ne(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        method_hint: Option<&(String, String, Option<String>)>,
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        second_arg_ty: Option<&TyRef>,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if args.len() != 2 {
            return Ok(false);
        }
        let Some(leaf) =
            partial_eq_eq_ne_leaf(segments, method_hint, &["option", "Option"], "Option")
        else {
            return Ok(false);
        };
        if ![first_arg_ty, second_arg_ty]
            .iter()
            .all(|ty| ty.is_some_and(|ty| self.option_niche_eq_ne_operand(ty)))
        {
            return Ok(false);
        }
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: leaf.to_string(),
                lhs: args[0].clone(),
                rhs: args[1].clone(),
                result_ty: ValueType::Int,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Operand of `<Result<T, E> as PartialEq>::eq` / `ne`: `&Result<T, E>`
    /// peeled to `core::result::Result`. Keyed on the ADT path, never on
    /// a caller symbol.
    pub(in crate::front::mir) fn result_eq_ne_ty(&self, ty: Option<&TyRef>) -> Option<TyRef> {
        let peeled = self.peel_to_option_or_result(ty?)?;
        crate::front::result_exc::tyref_is_result(&peeled, self.llbc).then_some(peeled)
    }

    fn result_variant_carries_payload(&self, ty: &TyRef, disc: i64) -> bool {
        let Some(def_id) = self.tyref_adt_def_id(ty) else {
            return true;
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return true;
        };
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return true;
        };
        let variant = variants
            .iter()
            .find(|v| v.discriminant_i64() == Some(disc))
            .or_else(|| usize::try_from(disc).ok().and_then(|idx| variants.get(idx)));
        variant.map(|v| !v.fields.is_empty()).unwrap_or(true)
    }

    fn classify_payload_eq(
        &self,
        ty: &TyRef,
        carries: bool,
        stack: &mut Vec<u64>,
    ) -> Option<PayloadEqPlan> {
        if !carries {
            return Some(PayloadEqPlan::NoPayload);
        }
        let peeled = self.tyref_peel_ref_to_pointee(ty);
        let ty = peeled.as_ref().unwrap_or(ty);
        if self.tyref_compares_as_string(ty) || tyref_strips_to_str(ty, self.llbc) {
            return Some(PayloadEqPlan::String);
        }
        if self.scalar_cmp_bank(ty).is_some()
            || self.tyref_is_fieldless_enum(ty)
            || self.tyref_is_borrowed_fieldless_enum(ty)
        {
            return Some(PayloadEqPlan::Scalar);
        }
        if let Some(plan) = self.classify_fun_decl_payload_eq(ty) {
            return Some(plan);
        }
        if crate::front::result_exc::tyref_is_option(ty, self.llbc) {
            return self.classify_option_payload_eq(ty, stack);
        }
        if crate::front::result_exc::tyref_is_result(ty, self.llbc) {
            return self.classify_result_payload_eq(ty, stack);
        }
        self.classify_structural_payload_eq(ty, stack)
    }

    /// Unique `PartialEq` impl whose `eq` FunDecl has an `Unstructured`
    /// body. `unique_trait_impl_for_adt` / `ClassDesc.lookup`.
    fn classify_fun_decl_payload_eq(&self, ty: &TyRef) -> Option<PayloadEqPlan> {
        let adt_id = self.tyref_adt_def_id(ty)?;
        let trait_id = unique_trait_decl_id(self.llbc, "core::cmp::PartialEq")?;
        let impl_id = unique_trait_impl_for_adt(self.llbc, trait_id, adt_id)?;
        let eq_fn = crate::front::clause_spec::impl_method_fn_id(self.llbc, impl_id, 0)?;
        let eq_fd = self.llbc.fn_by_id(eq_fn)?;
        if !crate::front::clause_spec::decl_has_unstructured_body(eq_fd) {
            return None;
        }
        let eq_leaf = fundecl_path_leaf(eq_fd)?;
        let eq_segments = spec_segments(self.llbc, eq_fd, eq_leaf);
        let ne =
            crate::front::clause_spec::impl_method_fn_id(self.llbc, impl_id, 1).and_then(|id| {
                let fd = self.llbc.fn_by_id(id)?;
                if fd.is_trait_default()
                    || !crate::front::clause_spec::decl_has_unstructured_body(fd)
                {
                    return None;
                }
                let leaf = fundecl_path_leaf(fd)?;
                Some((id, spec_segments(self.llbc, fd, leaf)))
            });
        Some(PayloadEqPlan::FunDecl {
            eq_id: eq_fn,
            eq_segments,
            ne,
        })
    }

    fn classify_option_payload_eq(
        &self,
        ty: &TyRef,
        stack: &mut Vec<u64>,
    ) -> Option<PayloadEqPlan> {
        if self.option_niche_payload_compares_as_string(ty) {
            return Some(PayloadEqPlan::String);
        }
        if self.option_niche_payload_compares_by_address(ty) {
            return Some(PayloadEqPlan::Scalar);
        }
        let (option_owner, some_owner, some_ty) = self.resolve_option_dest_owners(ty)?;
        let payload = crate::front::result_exc::tyref_option_payload(ty, self.llbc)?;
        let has = self.result_variant_carries_payload(ty, 1);
        let arm1 = Box::new(self.classify_payload_eq(&payload, has, stack)?);
        Some(PayloadEqPlan::Tagged {
            owner: option_owner.clone(),
            arm0_owner: option_owner,
            arm1_owner: some_owner,
            arm0_ty: ValueType::Int,
            arm1_ty: some_ty,
            arm0_raw_u8: false,
            arm1_raw_u8: tyref_is_raw_u8_ptr(&payload, self.llbc),
            arm0: Box::new(PayloadEqPlan::NoPayload),
            arm1,
            arm0_has: false,
            arm1_has: has,
        })
    }

    fn classify_result_payload_eq(
        &self,
        ty: &TyRef,
        stack: &mut Vec<u64>,
    ) -> Option<PayloadEqPlan> {
        let (result_owner, ok_owner, err_owner, ok_ty, err_ty, _) =
            self.resolve_result_owners(ty)?;
        let ok_tyref = crate::front::result_exc::tyref_result_ok(ty, self.llbc)?;
        let err_tyref = crate::front::result_exc::tyref_result_err(ty, self.llbc)?;
        let ok_has = self.result_variant_carries_payload(ty, 0);
        let err_has = self.result_variant_carries_payload(ty, 1);
        let arm0 = Box::new(self.classify_payload_eq(&ok_tyref, ok_has, stack)?);
        let arm1 = Box::new(self.classify_payload_eq(&err_tyref, err_has, stack)?);
        Some(PayloadEqPlan::Tagged {
            owner: result_owner,
            arm0_owner: ok_owner,
            arm1_owner: err_owner,
            arm0_ty: ok_ty,
            arm1_ty: err_ty,
            arm0_raw_u8: tyref_is_raw_u8_ptr(&ok_tyref, self.llbc),
            arm1_raw_u8: tyref_is_raw_u8_ptr(&err_tyref, self.llbc),
            arm0,
            arm1,
            arm0_has: ok_has,
            arm1_has: err_has,
        })
    }

    fn classify_structural_payload_eq(
        &self,
        ty: &TyRef,
        stack: &mut Vec<u64>,
    ) -> Option<PayloadEqPlan> {
        let adt_id = self.tyref_adt_def_id(ty)?;
        if !self.tyref_has_structural_partial_eq(adt_id) {
            return None;
        }
        let td = self.llbc.type_by_id(adt_id)?;
        let TypeDeclKind::Struct(fields) = &td.kind else {
            return None;
        };
        if stack.contains(&adt_id) {
            return None;
        }
        stack.push(adt_id);
        let owner = td.item_meta.name_path().to_string();
        let mut planned = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let name = field
                .name
                .clone()
                .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(index));
            let field_ty = tyref_to_value_type_with(
                &field.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            );
            let plan = self.classify_payload_eq(&field.ty, true, stack);
            let Some(plan) = plan else {
                stack.pop();
                return None;
            };
            planned.push(StructuralFieldPlan {
                name,
                ty: field_ty,
                plan: Box::new(plan),
            });
        }
        stack.pop();
        Some(PayloadEqPlan::Structural {
            owner,
            fields: planned,
        })
    }

    fn tyref_has_structural_partial_eq(&self, adt_id: u64) -> bool {
        let Some(trait_id) = unique_trait_decl_id(self.llbc, "core::marker::StructuralPartialEq")
        else {
            return false;
        };
        unique_trait_impl_for_adt(self.llbc, trait_id, adt_id).is_some()
    }

    /// Payload `PartialEq` of one Result variant. `&str` is string
    /// `BinOp` (`rstr.py` `AbstractStringRepr.rtype_eq` / `rtype_ne` →
    /// `ll_streq`). A scalar or fieldless enum is the same integer
    /// `BinOp`. An impl with an `Unstructured` body is that FunDecl
    /// (`ClassDesc.lookup`); a derived body missing from the LLBC is
    /// the TypeDecl field AND-chain. Anything else has already
    /// declined the lowering.
    fn emit_result_payload_cmp(
        &mut self,
        block: BlockId,
        thread: &[Variable],
        leaf: &str,
        lhs: Variable,
        rhs: Variable,
        payload_owner: &str,
        payload_ty: &ValueType,
        payload_tyref: &TyRef,
        plan: &PayloadEqPlan,
    ) -> (Variable, BlockId, Vec<Variable>) {
        if matches!(plan, PayloadEqPlan::NoPayload) {
            let val = push_block_op(&mut self.graph, block, OpKind::ConstBool(leaf == "eq"));
            return (val, block, thread.to_vec());
        }
        let raw_u8 = tyref_is_raw_u8_ptr(payload_tyref, self.llbc);
        let a = emit_payload_read(
            &mut self.graph,
            block,
            lhs,
            payload_owner,
            payload_ty.clone(),
            raw_u8,
        );
        let b = emit_payload_read(
            &mut self.graph,
            block,
            rhs,
            payload_owner,
            payload_ty.clone(),
            raw_u8,
        );
        self.emit_payload_eq_plan(block, thread, leaf, a, b, plan)
    }

    fn emit_payload_eq_plan(
        &mut self,
        block: BlockId,
        thread: &[Variable],
        leaf: &str,
        lhs: Variable,
        rhs: Variable,
        plan: &PayloadEqPlan,
    ) -> (Variable, BlockId, Vec<Variable>) {
        match plan {
            PayloadEqPlan::NoPayload => {
                let val = push_block_op(&mut self.graph, block, OpKind::ConstBool(leaf == "eq"));
                (val, block, thread.to_vec())
            }
            PayloadEqPlan::String => {
                stamp_as_rstr_str(&lhs);
                stamp_as_rstr_str(&rhs);
                let val = push_block_op(
                    &mut self.graph,
                    block,
                    OpKind::BinOp {
                        op: leaf.to_string(),
                        lhs,
                        rhs,
                        result_ty: ValueType::Int,
                    },
                );
                (val, block, thread.to_vec())
            }
            PayloadEqPlan::Scalar => {
                let val = push_block_op(
                    &mut self.graph,
                    block,
                    OpKind::BinOp {
                        op: leaf.to_string(),
                        lhs,
                        rhs,
                        result_ty: ValueType::Int,
                    },
                );
                (val, block, thread.to_vec())
            }
            PayloadEqPlan::FunDecl {
                eq_id,
                eq_segments,
                ne,
            } => {
                let (fn_id, segments, negate) = if leaf == "ne" {
                    if let Some((ne_id, ne_segments)) = ne {
                        (*ne_id, ne_segments.clone(), false)
                    } else {
                        (*eq_id, eq_segments.clone(), true)
                    }
                } else {
                    (*eq_id, eq_segments.clone(), false)
                };
                let val = push_block_op(
                    &mut self.graph,
                    block,
                    OpKind::Call {
                        target: CallTarget::function_path(segments).with_fun_decl_id(fn_id),
                        args: crate::model::call_args(vec![lhs, rhs]),
                        result_ty: ValueType::Int,
                    },
                );
                let val = if negate {
                    emit_eq_false(&mut self.graph, block, val)
                } else {
                    val
                };
                (val, block, thread.to_vec())
            }
            PayloadEqPlan::Structural { owner, fields } => {
                self.emit_structural_eq(block, thread, leaf, lhs, rhs, owner, fields)
            }
            PayloadEqPlan::Tagged {
                owner,
                arm0_owner,
                arm1_owner,
                arm0_ty,
                arm1_ty,
                arm0_raw_u8,
                arm1_raw_u8,
                arm0,
                arm1,
                arm0_has,
                arm1_has,
            } => self.emit_tagged_eq(
                block,
                thread,
                leaf,
                lhs,
                rhs,
                owner,
                arm0_owner,
                arm1_owner,
                arm0_ty,
                arm1_ty,
                *arm0_raw_u8,
                *arm1_raw_u8,
                arm0,
                arm1,
                *arm0_has,
                *arm1_has,
            ),
        }
    }

    fn emit_structural_eq(
        &mut self,
        block: BlockId,
        thread: &[Variable],
        leaf: &str,
        lhs: Variable,
        rhs: Variable,
        owner: &str,
        fields: &[StructuralFieldPlan],
    ) -> (Variable, BlockId, Vec<Variable>) {
        let mut live = thread.to_vec();
        if !live.contains(&lhs) {
            live.push(lhs.clone());
        }
        if !live.contains(&rhs) {
            live.push(rhs.clone());
        }
        let mut sources = live.clone();
        let mut block = block;
        let mut acc: Option<Variable> = None;
        for field in fields {
            let lhs_now = crate::front::option_variant::map_source(&sources, &live, &lhs)
                .expect("structural eq lhs still live");
            let rhs_now = crate::front::option_variant::map_source(&sources, &live, &rhs)
                .expect("structural eq rhs still live");
            let fa = emit_named_field_read(
                &mut self.graph,
                block,
                lhs_now,
                owner,
                &field.name,
                field.ty.clone(),
            );
            let fb = emit_named_field_read(
                &mut self.graph,
                block,
                rhs_now,
                owner,
                &field.name,
                field.ty.clone(),
            );
            let (cmp, next_block, next_live) =
                self.emit_payload_eq_plan(block, &live, "eq", fa, fb, &field.plan);
            if let Some(prev) = acc.as_ref() {
                acc = crate::front::option_variant::map_source(&live, &next_live, prev).or(acc);
            }
            sources = live;
            block = next_block;
            live = next_live;
            acc = Some(match acc {
                None => cmp,
                Some(prev) => push_block_op(
                    &mut self.graph,
                    block,
                    OpKind::BinOp {
                        op: "and".to_string(),
                        lhs: prev,
                        rhs: cmp,
                        result_ty: ValueType::Int,
                    },
                ),
            });
        }
        let mut val =
            acc.unwrap_or_else(|| push_block_op(&mut self.graph, block, OpKind::ConstBool(true)));
        if leaf == "ne" {
            val = emit_eq_false(&mut self.graph, block, val);
        }
        let exit_thread = if live.len() >= thread.len() {
            live[..thread.len()].to_vec()
        } else {
            thread.to_vec()
        };
        (val, block, exit_thread)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn emit_tagged_eq(
        &mut self,
        entry: BlockId,
        thread: &[Variable],
        leaf: &str,
        lhs: Variable,
        rhs: Variable,
        owner: &str,
        arm0_owner: &str,
        arm1_owner: &str,
        arm0_ty: &ValueType,
        arm1_ty: &ValueType,
        arm0_raw_u8: bool,
        arm1_raw_u8: bool,
        arm0: &PayloadEqPlan,
        arm1: &PayloadEqPlan,
        arm0_has: bool,
        arm1_has: bool,
    ) -> (Variable, BlockId, Vec<Variable>) {
        let d1 = emit_enum_disc_read(&mut self.graph, entry, lhs.clone(), owner);
        let d2 = emit_enum_disc_read(&mut self.graph, entry, rhs.clone(), owner);
        let discs_ne = push_block_op(
            &mut self.graph,
            entry,
            OpKind::BinOp {
                op: "ne".to_string(),
                lhs: d1,
                rhs: d2,
                result_ty: ValueType::Int,
            },
        );
        let mixed_sources = diamond_arm_sources(thread, &[]);
        let same_sources = diamond_arm_sources(thread, &[lhs.clone(), rhs.clone()]);
        let (mixed_bb, mixed_inputs) = self.graph.create_block_with_arg_vars(mixed_sources.len());
        let (same_bb, same_inputs) = self.graph.create_block_with_arg_vars(same_sources.len());
        let mixed_val = push_block_op(&mut self.graph, mixed_bb, OpKind::ConstBool(leaf == "ne"));
        let lhs_same = crate::front::option_variant::map_source(&same_sources, &same_inputs, &lhs)
            .expect("tagged eq lhs threaded into same-variant arm");
        let rhs_same = crate::front::option_variant::map_source(&same_sources, &same_inputs, &rhs)
            .expect("tagged eq rhs threaded into same-variant arm");
        let d_same = emit_enum_disc_read(&mut self.graph, same_bb, lhs_same.clone(), owner);
        let (arm0_bb, arm0_inputs) = self.graph.create_block_with_arg_vars(same_inputs.len());
        let (arm1_bb, arm1_inputs) = self.graph.create_block_with_arg_vars(same_inputs.len());
        let arm0_lhs =
            crate::front::option_variant::map_source(&same_inputs, &arm0_inputs, &lhs_same)
                .expect("tagged eq lhs threaded into arm0");
        let arm0_rhs =
            crate::front::option_variant::map_source(&same_inputs, &arm0_inputs, &rhs_same)
                .expect("tagged eq rhs threaded into arm0");
        let arm1_lhs =
            crate::front::option_variant::map_source(&same_inputs, &arm1_inputs, &lhs_same)
                .expect("tagged eq lhs threaded into arm1");
        let arm1_rhs =
            crate::front::option_variant::map_source(&same_inputs, &arm1_inputs, &rhs_same)
                .expect("tagged eq rhs threaded into arm1");
        let (arm0_val, arm0_exit, arm0_exit_thread) = if arm0_has {
            let a = emit_payload_read(
                &mut self.graph,
                arm0_bb,
                arm0_lhs,
                arm0_owner,
                arm0_ty.clone(),
                arm0_raw_u8,
            );
            let b = emit_payload_read(
                &mut self.graph,
                arm0_bb,
                arm0_rhs,
                arm0_owner,
                arm0_ty.clone(),
                arm0_raw_u8,
            );
            self.emit_payload_eq_plan(arm0_bb, &arm0_inputs, leaf, a, b, arm0)
        } else {
            let val = push_block_op(&mut self.graph, arm0_bb, OpKind::ConstBool(leaf == "eq"));
            (val, arm0_bb, arm0_inputs.clone())
        };
        let (arm1_val, arm1_exit, arm1_exit_thread) = if arm1_has {
            let a = emit_payload_read(
                &mut self.graph,
                arm1_bb,
                arm1_lhs,
                arm1_owner,
                arm1_ty.clone(),
                arm1_raw_u8,
            );
            let b = emit_payload_read(
                &mut self.graph,
                arm1_bb,
                arm1_rhs,
                arm1_owner,
                arm1_ty.clone(),
                arm1_raw_u8,
            );
            self.emit_payload_eq_plan(arm1_bb, &arm1_inputs, leaf, a, b, arm1)
        } else {
            let val = push_block_op(&mut self.graph, arm1_bb, OpKind::ConstBool(leaf == "eq"));
            (val, arm1_bb, arm1_inputs.clone())
        };
        let (join, join_inputs) = self.graph.create_block_with_arg_vars(thread.len() + 1);
        let join_thread = join_inputs[..thread.len()].to_vec();
        let join_val = join_inputs[thread.len()].clone();
        let mixed_thread: Vec<Variable> = thread
            .iter()
            .map(|v| {
                crate::front::option_variant::map_source(&mixed_sources, &mixed_inputs, v)
                    .expect("tagged mixed thread")
            })
            .collect();
        let mut mixed_goto = mixed_thread;
        mixed_goto.push(mixed_val);
        self.graph.set_goto(mixed_bb, join, mixed_goto);
        let arm0_thread: Vec<Variable> = thread
            .iter()
            .map(|v| {
                let mid = crate::front::option_variant::map_source(&same_sources, &same_inputs, v)
                    .expect("tagged arm0 thread");
                let mid =
                    crate::front::option_variant::map_source(&same_inputs, &arm0_inputs, &mid)
                        .expect("tagged arm0 inner thread");
                crate::front::option_variant::map_source(&arm0_inputs, &arm0_exit_thread, &mid)
                    .expect("tagged arm0 exit thread")
            })
            .collect();
        let mut arm0_goto = arm0_thread;
        arm0_goto.push(arm0_val);
        self.graph.set_goto(arm0_exit, join, arm0_goto);
        let arm1_thread: Vec<Variable> = thread
            .iter()
            .map(|v| {
                let mid = crate::front::option_variant::map_source(&same_sources, &same_inputs, v)
                    .expect("tagged arm1 thread");
                let mid =
                    crate::front::option_variant::map_source(&same_inputs, &arm1_inputs, &mid)
                        .expect("tagged arm1 inner thread");
                crate::front::option_variant::map_source(&arm1_inputs, &arm1_exit_thread, &mid)
                    .expect("tagged arm1 exit thread")
            })
            .collect();
        let mut arm1_goto = arm1_thread;
        arm1_goto.push(arm1_val);
        self.graph.set_goto(arm1_exit, join, arm1_goto);
        self.graph.set_branch(
            same_bb,
            d_same,
            arm1_bb,
            same_inputs.clone(),
            arm0_bb,
            same_inputs,
        );
        self.graph.set_branch(
            entry,
            discs_ne,
            mixed_bb,
            mixed_sources,
            same_bb,
            same_sources,
        );
        (join_val, join, join_thread)
    }

    /// Discriminant of a `core::result::Result` `AggregateKind::Adt`.
    /// Keyed on the Result TypeDecl path plus the variant's declared
    /// tag; any other ADT is `None`.
    fn result_aggregate_kind_disc(&self, kind: &Value, dest_ty: &TyRef) -> Option<i64> {
        let peeled = self.peel_to_option_or_result(dest_ty)?;
        if !crate::front::result_exc::tyref_is_result(&peeled, self.llbc) {
            return None;
        }
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => type_decl_ref_adt_id(head.as_object()?)?,
        };
        let td = self.llbc.type_by_id(type_id)?;
        if td.item_meta.name_path() != "core::result::Result" {
            return None;
        }
        let variant_idx = adt.get(1).and_then(Value::as_u64)?;
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return None;
        };
        variants.get(variant_idx as usize)?.discriminant_i64()
    }

    /// Discriminant of an `Operand::Const` Result ADT, including a
    /// `Ref`/`Ptr` wrapper of that ADT. Other const kinds stay `None`.
    fn result_const_expr_disc(&self, value: &Value) -> Option<i64> {
        let ty = self.llbc.const_expr_ty(value)?;
        let tyref: TyRef = serde_json::from_value(ty.clone()).ok()?;
        let peeled = self.peel_to_option_or_result(&tyref)?;
        if !crate::front::result_exc::tyref_is_result(&peeled, self.llbc) {
            return None;
        }
        let mut kind = self.llbc.const_expr_kind(value)?;
        for _ in 0..4 {
            let Some(obj) = kind.as_object() else {
                break;
            };
            if let Some(arr) = obj.get("Ref").and_then(Value::as_array) {
                kind = self.llbc.const_expr_kind(arr.first()?)?;
                continue;
            }
            if let Some(arr) = obj.get("Ptr").and_then(Value::as_array) {
                kind = self.llbc.const_expr_kind(arr.get(1)?)?;
                continue;
            }
            break;
        }
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        let variant_idx = adt.first()?.as_u64()?;
        let def_id = self.tyref_adt_def_id(&peeled)?;
        let td = self.llbc.type_by_id(def_id)?;
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return None;
        };
        variants.get(variant_idx as usize)?.discriminant_i64()
    }

    fn unique_local_assign(&self, local: usize) -> Option<(&Rvalue, &TyRef)> {
        if self.multi_assigned_locals.contains(&local) {
            return None;
        }
        let mut found = None;
        for bb in &self.body.body {
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                if matches!(place.kind, PlaceKind::Local(i) if i as usize == local) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((rvalue, &place.ty));
                }
            }
        }
        found
    }

    fn result_disc_place_local(place: &Place) -> Option<usize> {
        match &place.kind {
            PlaceKind::Local(i) => Some(*i as usize),
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                match &inner.kind {
                    PlaceKind::Local(i) => Some(*i as usize),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Compile-time Result discriminant of a single-assignment local,
    /// walking `Use` / `Ref` aliases to a Result `Aggregate` or const ADT.
    fn result_local_const_disc(&self, start: usize) -> Option<i64> {
        let mut local = start;
        for _ in 0..8 {
            if self.multi_assigned_locals.contains(&local) {
                return None;
            }
            if let Some(&tag) = self.const_discriminant_locals.get(&local) {
                let ty = &self.body.locals.locals.get(local)?.ty;
                return self
                    .peel_to_option_or_result(ty)
                    .filter(|t| crate::front::result_exc::tyref_is_result(t, self.llbc))
                    .and(Some(tag));
            }
            let (rvalue, dest_ty) = self.unique_local_assign(local)?;
            match rvalue {
                Rvalue::Aggregate(kind, _) => {
                    return self.result_aggregate_kind_disc(kind, dest_ty);
                }
                Rvalue::Use(Operand::Const(value), _) => {
                    return self.result_const_expr_disc(value);
                }
                Rvalue::Use(Operand::Copy(place) | Operand::Move(place), _)
                | Rvalue::Ref { place, .. }
                | Rvalue::RawPtr { place, .. } => {
                    local = Self::result_disc_place_local(place)?;
                }
                _ => return None,
            }
        }
        None
    }

    fn result_operand_const_disc(
        &self,
        local: Option<usize>,
        operand: Option<&Operand>,
    ) -> Option<i64> {
        match operand {
            Some(Operand::Const(value)) => self.result_const_expr_disc(value),
            Some(Operand::Copy(place) | Operand::Move(place)) => self
                .result_local_const_disc(local.or_else(|| Self::result_disc_place_local(place))?),
            _ => self.result_local_const_disc(local?),
        }
    }

    /// `<Result<T, E> as PartialEq>::eq` / `ne`. Twin of
    /// [`Self::try_lower_option_niche_eq_ne`]: same `PartialEq` path
    /// rewrite, tagged-sum layout instead of a niche word.
    ///
    /// The derived body is a discriminant switch (`rclass.py`
    /// `InstanceRepr._setup_repr`: tag on the enum root, payload on the
    /// variant subclass). Mixed tags → `ne` is True / `eq` is False;
    /// both `Ok` compare `T`; both `Err` compare `E`. Result convention
    /// is Ok=0 / Err=1, so `bool(disc)` takes Err as the true arm.
    ///
    /// A compile-time discriminant on one operand (`Aggregate` of
    /// `core::result::Result`, a `Use`/`Ref` of that local, or a const
    /// ADT) makes the other same-variant arm unreachable, so that
    /// payload plan is not required. Both operands non-constant still
    /// need both plans.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_result_eq_ne(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        method_hint: Option<&(String, String, Option<String>)>,
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        second_arg_ty: Option<&TyRef>,
        dest_local: usize,
        target: usize,
        arg_locals: &[Option<usize>],
        arg_operands: &[Operand],
    ) -> Result<bool, LowerError> {
        if args.len() != 2 {
            return Ok(false);
        }
        let Some(leaf) =
            partial_eq_eq_ne_leaf(segments, method_hint, &["result", "Result"], "Result")
        else {
            return Ok(false);
        };
        let Some(lhs_ty) = self.result_eq_ne_ty(first_arg_ty) else {
            return Ok(false);
        };
        if second_arg_ty.is_some() && self.result_eq_ne_ty(second_arg_ty).is_none() {
            return Ok(false);
        }
        let Some((result_owner, ok_owner, err_owner, ok_ty, err_ty, _)) =
            self.resolve_result_owners(&lhs_ty)
        else {
            return Ok(false);
        };
        let Some(ok_tyref) = crate::front::result_exc::tyref_result_ok(&lhs_ty, self.llbc) else {
            return Ok(false);
        };
        let Some(err_tyref) = crate::front::result_exc::tyref_result_err(&lhs_ty, self.llbc) else {
            return Ok(false);
        };
        let ok_has_payload = self.result_variant_carries_payload(&lhs_ty, 0);
        let err_has_payload = self.result_variant_carries_payload(&lhs_ty, 1);
        let lhs_const = self
            .result_operand_const_disc(arg_locals.first().copied().flatten(), arg_operands.first());
        let rhs_const = self
            .result_operand_const_disc(arg_locals.get(1).copied().flatten(), arg_operands.get(1));
        let skip_ok = matches!(lhs_const, Some(1)) || matches!(rhs_const, Some(1));
        let skip_err = matches!(lhs_const, Some(0)) || matches!(rhs_const, Some(0));
        if skip_ok && skip_err {
            return Ok(false);
        }
        let mut stack = Vec::new();
        let ok_plan = if skip_ok {
            None
        } else {
            match self.classify_payload_eq(&ok_tyref, ok_has_payload, &mut stack) {
                Some(plan) => Some(plan),
                None => return Ok(false),
            }
        };
        stack.clear();
        let err_plan = if skip_err {
            None
        } else {
            match self.classify_payload_eq(&err_tyref, err_has_payload, &mut stack) {
                Some(plan) => Some(plan),
                None => return Ok(false),
            }
        };

        if skip_ok || skip_err {
            let const_tag = if skip_err { 0 } else { 1 };
            let runtime_idx = if lhs_const.is_none() {
                0
            } else if rhs_const.is_none() {
                1
            } else {
                0
            };
            let plan = if skip_err {
                ok_plan
                    .as_ref()
                    .expect("const Ok keeps the Ok payload plan")
            } else {
                err_plan
                    .as_ref()
                    .expect("const Err keeps the Err payload plan")
            };
            let payload_owner = if skip_err { &ok_owner } else { &err_owner };
            let payload_ty = if skip_err { &ok_ty } else { &err_ty };
            let payload_tyref = if skip_err { &ok_tyref } else { &err_tyref };
            let bb_id = self.emit_bb(mir_bb);
            let d_runtime = emit_enum_disc_read(
                &mut self.graph,
                bb_id,
                args[runtime_idx].clone(),
                &result_owner,
            );
            let tag = push_block_op(&mut self.graph, bb_id, OpKind::ConstInt(const_tag));
            let discs_ne = push_block_op(
                &mut self.graph,
                bb_id,
                OpKind::BinOp {
                    op: "ne".to_string(),
                    lhs: d_runtime,
                    rhs: tag,
                    result_ty: ValueType::Int,
                },
            );
            let placeholder = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.local_var[dest_local] = Some(LocalValue::One(placeholder.clone()));
            let target_bb = self.block_id[target];
            let link_args = self.edge_args(mir_bb, target)?;
            let mut carried: Vec<Variable> = Vec::new();
            for v in &link_args {
                if *v != placeholder && !carried.contains(v) {
                    carried.push(v.clone());
                }
            }
            let mixed_sources = diamond_arm_sources(&carried, &[]);
            let match_sources = diamond_arm_sources(&carried, &[args[0].clone(), args[1].clone()]);
            let (mixed_bb, mixed_inputs) =
                self.graph.create_block_with_arg_vars(mixed_sources.len());
            let (match_bb, match_inputs) =
                self.graph.create_block_with_arg_vars(match_sources.len());
            let mixed_val =
                push_block_op(&mut self.graph, mixed_bb, OpKind::ConstBool(leaf == "ne"));
            let lhs_match =
                crate::front::option_variant::map_source(&match_sources, &match_inputs, &args[0])
                    .expect("Result eq/ne lhs threaded into const-disc match arm");
            let rhs_match =
                crate::front::option_variant::map_source(&match_sources, &match_inputs, &args[1])
                    .expect("Result eq/ne rhs threaded into const-disc match arm");
            let (match_val, match_exit, match_exit_thread) = self.emit_result_payload_cmp(
                match_bb,
                &match_inputs,
                leaf,
                lhs_match,
                rhs_match,
                payload_owner,
                payload_ty,
                payload_tyref,
                plan,
            );
            let (join, join_inputs) = self.graph.create_block_with_arg_vars(link_args.len());
            let mixed_args = remap_diamond_goto_args(
                &link_args,
                &placeholder,
                &mixed_val,
                &mixed_sources,
                &mixed_inputs,
            );
            self.graph.set_goto(mixed_bb, join, mixed_args);
            let match_args = remap_payload_goto_args(
                &link_args,
                &placeholder,
                &match_val,
                &match_sources,
                &match_inputs,
                &match_inputs,
                &match_exit_thread,
            );
            self.graph.set_goto(match_exit, join, match_args);
            self.graph.set_branch(
                bb_id,
                discs_ne,
                mixed_bb,
                mixed_sources,
                match_bb,
                match_sources,
            );
            let mut remap: std::collections::HashMap<u64, Variable> =
                std::collections::HashMap::new();
            for (src, dst) in link_args.iter().zip(join_inputs.iter()) {
                remap.insert(src.id(), dst.clone());
            }
            for value in self.local_var.iter_mut().flatten() {
                value.remap_words(&remap);
            }
            self.remap_forwarded_provenance(&remap);
            self.emit_block.set(mir_bb, join);
            let join_args = self.edge_args(mir_bb, target)?;
            self.graph.set_goto(join, target_bb, join_args);
            return Ok(true);
        }

        let ok_plan = ok_plan.expect("both-runtime Result eq/ne classified Ok");
        let err_plan = err_plan.expect("both-runtime Result eq/ne classified Err");

        let bb_id = self.emit_bb(mir_bb);
        let d1 = emit_enum_disc_read(&mut self.graph, bb_id, args[0].clone(), &result_owner);
        let d2 = emit_enum_disc_read(&mut self.graph, bb_id, args[1].clone(), &result_owner);
        let discs_ne = push_block_op(
            &mut self.graph,
            bb_id,
            OpKind::BinOp {
                op: "ne".to_string(),
                lhs: d1,
                rhs: d2,
                result_ty: ValueType::Int,
            },
        );

        let placeholder = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.local_var[dest_local] = Some(LocalValue::One(placeholder.clone()));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;

        let mut carried: Vec<Variable> = Vec::new();
        for v in &link_args {
            if *v != placeholder && !carried.contains(v) {
                carried.push(v.clone());
            }
        }
        let mixed_sources = diamond_arm_sources(&carried, &[]);
        let same_sources = diamond_arm_sources(&carried, &[args[0].clone(), args[1].clone()]);
        let (mixed_bb, mixed_inputs) = self.graph.create_block_with_arg_vars(mixed_sources.len());
        let (same_bb, same_inputs) = self.graph.create_block_with_arg_vars(same_sources.len());

        let mixed_val = push_block_op(&mut self.graph, mixed_bb, OpKind::ConstBool(leaf == "ne"));
        let lhs_same =
            crate::front::option_variant::map_source(&same_sources, &same_inputs, &args[0])
                .expect("Result eq/ne lhs threaded into same-variant arm");
        let rhs_same =
            crate::front::option_variant::map_source(&same_sources, &same_inputs, &args[1])
                .expect("Result eq/ne rhs threaded into same-variant arm");
        let d_same = emit_enum_disc_read(&mut self.graph, same_bb, lhs_same.clone(), &result_owner);

        let (ok_bb, ok_inputs) = self.graph.create_block_with_arg_vars(same_inputs.len());
        let (err_bb, err_inputs) = self.graph.create_block_with_arg_vars(same_inputs.len());
        let ok_lhs = crate::front::option_variant::map_source(&same_inputs, &ok_inputs, &lhs_same)
            .expect("Result eq/ne lhs threaded into Ok arm");
        let ok_rhs = crate::front::option_variant::map_source(&same_inputs, &ok_inputs, &rhs_same)
            .expect("Result eq/ne rhs threaded into Ok arm");
        let err_lhs =
            crate::front::option_variant::map_source(&same_inputs, &err_inputs, &lhs_same)
                .expect("Result eq/ne lhs threaded into Err arm");
        let err_rhs =
            crate::front::option_variant::map_source(&same_inputs, &err_inputs, &rhs_same)
                .expect("Result eq/ne rhs threaded into Err arm");
        let (ok_val, ok_exit, ok_exit_thread) = self.emit_result_payload_cmp(
            ok_bb, &ok_inputs, leaf, ok_lhs, ok_rhs, &ok_owner, &ok_ty, &ok_tyref, &ok_plan,
        );
        let (err_val, err_exit, err_exit_thread) = self.emit_result_payload_cmp(
            err_bb,
            &err_inputs,
            leaf,
            err_lhs,
            err_rhs,
            &err_owner,
            &err_ty,
            &err_tyref,
            &err_plan,
        );

        let (join, join_inputs) = self.graph.create_block_with_arg_vars(link_args.len());
        let mixed_args = remap_diamond_goto_args(
            &link_args,
            &placeholder,
            &mixed_val,
            &mixed_sources,
            &mixed_inputs,
        );
        self.graph.set_goto(mixed_bb, join, mixed_args);
        let ok_args = remap_payload_goto_args(
            &link_args,
            &placeholder,
            &ok_val,
            &same_sources,
            &same_inputs,
            &ok_inputs,
            &ok_exit_thread,
        );
        self.graph.set_goto(ok_exit, join, ok_args);
        let err_args = remap_payload_goto_args(
            &link_args,
            &placeholder,
            &err_val,
            &same_sources,
            &same_inputs,
            &err_inputs,
            &err_exit_thread,
        );
        self.graph.set_goto(err_exit, join, err_args);
        self.graph.set_branch(
            same_bb,
            d_same,
            err_bb,
            same_inputs.clone(),
            ok_bb,
            same_inputs,
        );
        self.graph.set_branch(
            bb_id,
            discs_ne,
            mixed_bb,
            mixed_sources,
            same_bb,
            same_sources,
        );

        let mut remap: std::collections::HashMap<u64, Variable> = std::collections::HashMap::new();
        for (src, dst) in link_args.iter().zip(join_inputs.iter()) {
            remap.insert(src.id(), dst.clone());
        }
        for value in self.local_var.iter_mut().flatten() {
            value.remap_words(&remap);
        }
        self.remap_forwarded_provenance(&remap);
        self.emit_block.set(mir_bb, join);
        let join_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(join, target_bb, join_args);
        Ok(true)
    }

    /// Emit a null pointer for a nullable `Option` (`None`) as a
    /// `core::ptr::null_mut()` call, returning its result `Variable`. A Rust
    /// container projected to an RPython string/list is then narrowed to that
    /// projected repr before it is returned or compared.
    ///
    /// `null_mut()` is preferred over a bare `ConstRefNull`: the annotator's
    /// `ptr_null_constant` types it as a classdef-less nullable `SomeInstance`
    /// (joinable with any typed pointer receiver), and the rtyper's
    /// `rtype_ptr_null` recovers the null via `convert_const(None)` on the
    /// *result repr* — so the null adapts to whatever `InstanceRepr` /
    /// `PtrRepr` the consuming slot carries.  A `ConstRefNull` is a fixed
    /// GCREF `_ptr`, which fails to union (annotate) and fails to convert
    /// (rtype) against an `InstanceRepr(PyObject)` at a `ptr_ne` or a return
    /// merge.  Used by both the `Discriminant` null-test fold and the niche
    /// `Aggregate` `None` construction fold so every niche-Option null shares
    /// one repr-adaptive source.
    pub(in crate::front::mir) fn push_niche_null_ptr(
        &mut self,
        mir_bb: usize,
        option_ty: &TyRef,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        // Overlay raw-struct payload (`*mut SomeRawStruct`): the Some
        // arm is `cast_int_to_ptr(PTR, addr)` from
        // `emit_dest_overlay_cast` → `dest_overlay_ptr_lltype`. None is
        // the same PTR with `oddint == 0` (`cast_int_to_ptr` /
        // `ann_cast_int_to_ptr` → `SomePtr(PTRTYPE)`).
        if let Some(payload) = crate::front::result_exc::tyref_option_payload(option_ty, self.llbc)
            && let Some(ptr) = self.dest_overlay_ptr_lltype(&payload)
        {
            return emit_typed_nullptr(&mut self.graph, bb_id, ptr);
        }
        // `Option<*mut c_void>` / `*mut c_void` is `rffi.VOIDP`
        // (`rffi.py` `VOIDP`). None is `cast_int_to_ptr(VOIDP, 0)`
        // (`lltype.py` `cast_int_to_ptr` when `oddint == 0` /
        // `ann_cast_int_to_ptr` → `SomePtr(VOIDP)`). This arm runs
        // before `option_niche_int_null` so the int-bank `null_raw` /
        // `llmemory.NULL` arm does not take the VOIDP word.
        if tyref_is_raw_c_void_ptr(option_ty, self.llbc)
            || crate::front::result_exc::tyref_option_payload(option_ty, self.llbc)
                .is_some_and(|payload| tyref_is_raw_c_void_ptr(&payload, self.llbc))
        {
            return emit_voidp_nullptr(&mut self.graph, bb_id);
        }
        // `executioncontext.py` `profilefunc`: `None` is `Constant(None)`
        // (`s_None`), not `null_fn` / `null_mut`.
        if tyref_is_profilefunc_family(option_ty, self.llbc) {
            return self.graph.push_const_none(bb_id);
        }
        // Int-banked niche (`Option<fn>` / raw Address): `rnone.py`
        // `rtype_is_None` compares against a null of the operand's own
        // lltype (`ptr_iszero` / `int_is_zero` / `adr_eq`). `null_mut()`
        // is a GC ref and would emit `is_` on mixed `ir` operands.
        // `Option<fn>` keeps `null_fn` with the `fn(..)` spelling; a
        // raw-address niche uses `null_raw` of the pointee (`lltype.nullptr`)
        // or `llmemory.NULL` when the payload has no named type.
        if let Some(int_null) = self.option_niche_int_null(option_ty) {
            return int_null.push(&mut self.graph, bb_id);
        }
        // `null()` / `null_mut()` of `*u8` is `lltype.nullptr(CCHARP.TO)`
        // (`rffi.py` `CCHARP`), a typed null Ptr constant.
        if tyref_is_raw_u8_ptr(option_ty, self.llbc) {
            return emit_ccharp_nullptr(&mut self.graph, bb_id);
        }
        self.graph
            .push_niche_null(bb_id, self.option_niche_null_cast(option_ty).as_ref())
    }
}

/// [`Llbc::has_explicit_drop_impl`] for `def_id`.
pub(in crate::front::mir) fn type_decl_has_explicit_drop(llbc: &Llbc, def_id: u64) -> bool {
    llbc.has_explicit_drop_impl(def_id, |ty| resolve_tyexpr_to_adt_def_id_free(llbc, ty))
}

/// Conservative `needs_drop::<T>() == false` projection. The same
/// predicate a [`TermKind::Drop`] of a local of this type uses to
/// decide the destructor is a no-op.
pub(in crate::front::mir) fn type_decl_is_trivially_dropless(
    decl: &TypeDecl,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 16 || type_decl_has_explicit_drop(llbc, decl.def_id) {
        return false;
    }
    if decl.published_struct_copy {
        return false;
    }
    match &decl.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => fields
            .iter()
            .all(|field| tyref_is_trivially_dropless(&field.ty, llbc, depth + 1)),
        TypeDeclKind::Enum(variants) => variants.iter().all(|variant| {
            variant
                .fields
                .iter()
                .all(|field| tyref_is_trivially_dropless(&field.ty, llbc, depth + 1))
        }),
        TypeDeclKind::Alias(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => false,
    }
}

pub(in crate::front::mir) fn tyref_is_trivially_dropless(
    ty: &TyRef,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 16 {
        return false;
    }
    let Some(node) = tyref_node(ty, llbc) else {
        return false;
    };
    if node.get("Scalar").is_some()
        || node.get("Ref").is_some()
        || node.get("RawPtr").is_some()
        || node.get("FnDef").is_some()
        || node.get("FnPtr").is_some()
    {
        return true;
    }
    let Some(def_id) = strip_ty_wrappers(node, llbc).and_then(adt_node_def_id) else {
        return false;
    };
    llbc.type_by_id(def_id)
        .is_some_and(|decl| type_decl_is_trivially_dropless(decl, llbc, depth + 1))
}
