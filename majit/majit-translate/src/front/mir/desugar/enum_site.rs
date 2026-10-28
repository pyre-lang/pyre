//! Hand lowering of ADT / enum / code-flag callees.

use super::*;

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
                    Some((
                        name,
                        tyref_to_value_type_with(
                            &field.ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                    ))
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
    /// records only the register bank, so a `Ref(None)` payload needs its
    /// source class recorded on the closure-rewrite site until the synthesized
    /// writer can emit the ordinary instance-narrowing cast.  This preserves
    /// the concrete field annotation that RPython's `InstanceRepr._setup_repr`
    /// consumes (`rpython/rtyper/rclass.py`).
    pub(in crate::front::mir) fn option_payload_instance_class_root(
        &self,
        option_ty: &TyRef,
    ) -> Option<String> {
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
        let pointee = payload.as_object()?.get("Ref")?.as_array()?.get(1)?;
        let pointee = strip_ty_wrappers(pointee, self.llbc)?;
        raw_ptr_pointee_class_root_with(pointee, self.llbc, self.tombstoned_leaves)
            .or_else(|| adt_node_class_root_with(pointee, self.llbc, self.tombstoned_leaves))
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
    /// [`Self::resolve_bool_then_option_dest`] / the tagged-pair ctor derive it)
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
    /// — all stay classdef-less GCREF, unchanged.
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
        let adt = tyref_node(dest_ty, self.llbc)?
            .as_object()?
            .get("Adt")?
            .as_object()?;
        let payload = type_decl_ref_generics(adt, self.llbc)?
            .get("types")?
            .get(0)?;
        let payload_node = strip_ty_wrappers(payload, self.llbc)?;
        let payload_ok =
            raw_ptr_pointee_class_root_with(payload_node, self.llbc, self.tombstoned_leaves)
                .is_some()
                || payload_node
                    .as_object()
                    .and_then(|o| o.get("Adt"))
                    .and_then(serde_json::Value::as_object)
                    .and_then(|adt| adt_head_instantiation_suffix(adt, self.llbc))
                    .is_some();
        if !payload_ok {
            return None;
        }
        // The narrow root is the Option enum instantiation itself — the same
        // spelling a static `Some(..)` construction of this instantiation mints.
        // An empty suffix is the template ClassDef; its `Some.__pos_0` is the
        // typevar and getattr cannot succeed (`bookkeeper.py` `getuniqueclassdef`).
        let suffix = tyref_enum_instantiation_suffix(dest_ty, self.llbc);
        if suffix.is_empty() {
            return None;
        }
        let def_id = self.tyref_adt_def_id(dest_ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        Some(format!("{}{}", td.item_meta.name_path(), suffix))
    }

    /// Registered ADT pointee of a one-word niche `Option<&T>` /
    /// `Option<&mut T>` / `Option<*mut T>`.
    ///
    /// Rust uses the null pointer niche for this Option shape. It therefore
    /// maps to RPython's nullable `SomeInstance(T)`, not to an Option
    /// container class with `__discriminant` / `__pos_0` fields.
    pub(in crate::front::mir) fn option_niche_payload_class_root(
        &self,
        dest_ty: &TyRef,
    ) -> Option<String> {
        if !self.tyref_is_niche_option_ptr(dest_ty) {
            return None;
        }
        self.option_payload_instance_class_root(dest_ty)
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
        if tyref_is_option_of_string_value(option_ty, self.llbc)
            || matches!(payload_ty, ValueType::Str)
        {
            return Some((rendered, ValueType::Str));
        }
        if tyref_is_string_family(&TyRef::Other(payload.clone()), self.llbc) {
            return Some((rendered, payload_ty));
        }
        if rendered.starts_with("Vec<") || rendered.starts_with("VecDeque<") {
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
        // `None` of a maybe-null `W_Root` is RPython's `None`, which unions
        // into `SomeInstance(W_Root, can_be_none=True)` and keeps the class
        // (`model.py` `unionof`).  The bare `null_mut()` is classdef-less, so
        // a `return None` beside a `Some(w)` widened the whole return to the
        // classdef-less top and carried it into every container the value
        // reached.  Narrow the null to the `PyObject` pointee the payload
        // (`NonNull<PyObject>`, `*mut PyObject`, `&PyObject`) names.
        let pointee = rendered
            .trim_end_matches('>')
            .rsplit(['<', ' ', '&'])
            .next()
            .unwrap_or_default();
        if pointee == "PyObject" || pointee.ends_with("::PyObject") {
            return Some(("pyobject::PyObject".to_string(), ValueType::Ref(None)));
        }
        None
    }

    /// `__pos_0` of an enum whose payload is a nullable `PyObject` pointer
    /// (`Result<Option<*mut PyObject>, _>::Ok`, or `Result<*mut PyObject, _>`).
    ///
    /// The niche `Option` is the pointer itself (`option_niche_null_cast`).
    /// A bare field read annotates as a classdef-less instance, and
    /// `unionof` with the already-narrowed `PY_NULL` arm then drops the
    /// `PyObject` class. The read wants the same
    /// `__cast_instance_intrinsic("PyObject")` narrow.
    pub(in crate::front::mir) fn enum_payload_is_nullable_instance_ptr(
        &self,
        place_ty: &TyRef,
    ) -> bool {
        if self
            .option_niche_null_cast(place_ty)
            .is_some_and(|(root, _)| root == "pyobject::PyObject")
        {
            return true;
        }
        tyref_node(place_ty, self.llbc)
            .and_then(|node| strip_ty_wrappers(node, self.llbc))
            .and_then(|node| {
                raw_ptr_pointee_class_root_with(node, self.llbc, self.tombstoned_leaves)
            })
            .as_deref()
            == Some("PyObject")
    }

    /// Resolve a destination `Option` into its `(option_owner, some_owner,
    /// payload_ty)` — the enum root ctor owner, the `Some` variant
    /// payload-field owner, and the payload `ValueType`.  `None` when the
    /// destination is not a resolvable `Option`.
    pub(in crate::front::mir) fn resolve_bool_then_option_dest(
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
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let some_owner = Self::tagged_pair_payload_owner(td, &option_owner, 1)?;
        let payload_ty = self.tyref_option_payload_value_type(dest_ty)?;
        Some((option_owner, some_owner, payload_ty))
    }

    /// Resolve the `(option_owner, some_owner, payload_ty)` an `Option`
    /// *consumer* pass (`option_try` `?`, `expect`) must spell so its
    /// `__discriminant` / `__pos_0` reads key the SAME classdef the value's
    /// *producer* wrote.
    ///
    /// Agreement is by construction: the suffixed producers (`map_or`,
    /// `slice_first`, `from_size_align`) mint a
    /// per-instantiation `Option<X>` root through
    /// [`Self::resolve_bool_then_option_dest`], so the consumer reuses that
    /// exact derivation, including checked unsigned arithmetic. Its generated
    /// Some/None constructors must identify the same existing classes as a
    /// source-level Some(value), not create a separate bare Option payload.
    /// A niche
    /// one-word niche `Option` has no aggregate
    /// `__discriminant` / `__pos_0` (the arms use a pointer null-test and an
    /// identity payload), so its owners are never read — keep them bare, which
    /// also mirrors [`Self::resolve_bool_then_option_dest`]'s own aggregate
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
        self.resolve_bool_then_option_dest(recv_ty)
    }

    /// Resolve the enclosing function's return `Option` owner for the
    /// `?`-None construction ([`crate::front::option_try`]).  It must MATCH the
    /// spelling of the suffixed producers whose `Some` values reach the same
    /// returnblock — otherwise a suffixed `Some` and a bare `None` union to the
    /// bare template, payload-erasing the return.  Reuses
    /// [`Self::resolve_option_consumer_owners`], so a ref/str/float-payload
    /// return uses the source constructor identity; only the niche-pointer
    /// representation has no aggregate owner. `None` when the output is not a
    /// resolvable `Option` — the `?` rewrite then declines all sites.
    pub(in crate::front::mir) fn resolve_option_return_owner(
        &self,
        output_ty: &TyRef,
    ) -> Option<String> {
        if !crate::front::result_exc::tyref_is_option(output_ty, self.llbc) {
            return None;
        }
        let (option_owner, _some_owner, _payload_ty) =
            self.resolve_option_consumer_owners(output_ty)?;
        Some(option_owner)
    }

    /// Resolve a recognized `bool::then(cond, closure_env)` call into a
    /// [`crate::front::bool_then::BoolThenSite`]. `None` (leaving the
    /// residual call) when the destination is not a resolvable `Option` or
    /// the closure env type does not resolve to an ADT.
    pub(in crate::front::mir) fn recognize_bool_then_site(
        &self,
        dest_ty: &TyRef,
        env_ty: Option<&TyRef>,
        result_var: &Variable,
    ) -> Option<crate::front::bool_then::BoolThenSite> {
        let (option_owner, some_owner, mut payload_ty) =
            self.resolve_bool_then_option_dest(dest_ty)?;
        let niche = self.tyref_is_niche_option_ptr(dest_ty);
        let scalar_kind = option_nonzero_word_kind(dest_ty, self.llbc);
        let scalar_niche = scalar_kind.is_some();
        if let Some(kind) = scalar_kind {
            payload_ty = kind;
        }
        let payload_narrow_root = niche
            .then(|| self.option_niche_payload_class_root(dest_ty))
            .flatten();
        let env_def_id = self.tyref_ref_adt_def_id(env_ty?)?;
        let env_td = self.llbc.type_by_id(env_def_id)?;
        let call_once_owner = closure_env_call_once_owner(env_td);
        Some(crate::front::bool_then::BoolThenSite {
            result_var: result_var.clone(),
            call_once_owner,
            option_owner,
            some_owner,
            payload_ty,
            niche,
            scalar_niche,
            niche_null_cast: self.option_niche_null_cast(dest_ty),
            payload_narrow_root,
        })
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
    /// + `Some` variant owners [`Self::recognize_slice_first_site`] resolves,
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
        })
    }

    /// Resolve a recognized carrier `Result<T, PyError>` combinator
    /// (`unwrap_or` / `expect` / `unwrap`) into a
    /// [`crate::front::result_exc::CarrierCombinatorSite`].  `None` when the
    /// receiver is not the exception-carrier Result — `expect` / `unwrap`
    /// then go to [`Self::recognize_unwrap_site`] /
    /// [`Self::recognize_expect_site`], and `unwrap_or` stays a residual
    /// call whose core body is translated.
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

    /// Resolve a recognized `Option::unwrap(opt)` / plain-error
    /// `Result::unwrap(res)` call into an
    /// [`crate::front::option_unwrap::UnwrapSite`] — the enum root, payload
    /// variant owner, payload type, and discriminant polarity the guard
    /// post-pass needs. `Result<T, PyError>` remains the exception transform's
    /// domain and is deliberately excluded. `None` leaves the residual call.
    pub(in crate::front::mir) fn recognize_unwrap_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
    ) -> Option<crate::front::option_unwrap::UnwrapSite> {
        let recv_ty = recv_ty?;
        let is_option = crate::front::result_exc::tyref_is_option(recv_ty, self.llbc);
        let (enum_owner, payload_owner, payload_on_disc_true, niche) = if is_option {
            let (option_owner, some_owner, _) = self.resolve_option_consumer_owners(recv_ty)?;
            (
                option_owner,
                some_owner,
                true,
                self.tyref_is_niche_option_ptr(recv_ty),
            )
        } else if crate::front::result_exc::tyref_is_result(recv_ty, self.llbc)
            && !crate::front::result_exc::tyref_is_result_of_carrier(
                recv_ty,
                self.llbc,
                self.static_addrs.error_carrier,
            )
        {
            let def_id = self.tyref_adt_def_id(recv_ty)?;
            let td = self.llbc.type_by_id(def_id)?;
            // Match the per-instantiation owner minted by checked conversion
            // producers (`Result<usize, TryFromIntError>`, etc.). A bare
            // Result root would split the producer and consumer classdefs.
            let enum_owner = format!(
                "{}{}",
                td.item_meta.name_path(),
                tyref_enum_instantiation_suffix(recv_ty, self.llbc)
            );
            let payload_owner = Self::tagged_pair_payload_owner(td, &enum_owner, 0)?;
            (enum_owner, payload_owner, false, false)
        } else {
            return None;
        };
        let payload_ty = self.tyref_option_payload_value_type(recv_ty)?;
        let scalar_niche = is_option && !niche && tyref_option_nonzero_scalar(recv_ty, self.llbc);
        Some(crate::front::option_unwrap::UnwrapSite {
            result_var: result_var.clone(),
            enum_owner,
            payload_owner,
            payload_ty,
            payload_on_disc_true,
            niche,
            int_null: self.option_niche_int_null(recv_ty),
            scalar_niche,
            niche_null_cast: self.option_niche_null_cast(recv_ty),
        })
    }

    /// Resolve a recognized `Option::expect(opt, msg)` call into an
    /// [`crate::front::option_expect::ExpectSite`] — the two-arg sibling of
    /// [`Self::recognize_unwrap_site`].  Guards on `Option` (not `Result`:
    /// `Result::expect` shares the name but its `Ok`/`Err` tags do not match
    /// `Some`=1/`None`=0).  The owners route through
    /// [`Self::resolve_option_consumer_owners`] so the synthesized
    /// `__discriminant` / `__pos_0` reads key the same classdef the value's
    /// producer wrote.  `None` (leaving the residual call) when the receiver is
    /// not a resolvable `Option`.
    pub(in crate::front::mir) fn recognize_expect_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
        opt_var: Option<&Variable>,
    ) -> Option<crate::front::option_expect::ExpectSite> {
        let recv_ty = recv_ty?;
        if !crate::front::result_exc::tyref_is_option(recv_ty, self.llbc) {
            return None;
        }
        let (mut option_owner, mut some_owner, payload_ty) =
            self.resolve_option_consumer_owners(recv_ty)?;
        let mut niche = self.tyref_is_niche_option_ptr(recv_ty);
        self.recover_option_instantiation(
            recv_ty,
            opt_var,
            &mut option_owner,
            &mut some_owner,
            &mut niche,
        );
        let scalar_niche = !niche && tyref_option_nonzero_scalar(recv_ty, self.llbc);
        Some(crate::front::option_expect::ExpectSite {
            result_var: result_var.clone(),
            option_owner,
            some_owner,
            payload_ty,
            niche,
            int_null: self.option_niche_int_null(recv_ty),
            scalar_niche,
            niche_null_cast: self.option_niche_null_cast(recv_ty),
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
        let result_suffix =
            crate::front::result_exc::tyref_result_instantiation_suffix(dest_ty, self.llbc)
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

    /// Resolve a recognized `Try::branch(opt)` call where `opt: Option<T>`
    /// into an [`crate::front::option_try::OptionTrySite`].  `None` leaves the
    /// residual call untouched when the receiver is not a resolvable `Option`.
    /// Resolve an unsigned `usize::checked_{add,mul}` call whose result is an
    /// `Option<usize>` into a
    /// [`crate::front::checked_arith_uint::CheckedArithUintSite`] — the
    /// `Option` enum root + `Some` variant owners and the payload type, keyed
    /// off the destination `Option<usize>` type.  Mirrors
    /// [`Self::recognize_option_try_site`]'s owner derivation so the
    /// virtualized `Option` this pass builds shares field descrs with the
    /// downstream `?` reads (`front::option_try`).
    pub(in crate::front::mir) fn recognize_checked_arith_uint_site(
        &self,
        dest_ty: &TyRef,
        result_var: &Variable,
        payload_hint: Option<&TyRef>,
    ) -> Option<crate::front::checked_arith_uint::CheckedArithUintSite> {
        if !crate::front::result_exc::tyref_is_option(dest_ty, self.llbc) {
            return None;
        }
        let (mut option_owner, mut some_owner, payload_ty) =
            self.resolve_bool_then_option_dest(dest_ty)?;
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
        })
    }

    /// Resolve a `Result::ok()` call whose result is an `Option<Layout>` into a
    /// [`crate::front::from_size_align::FromSizeAlignSite`] — the `Option` enum
    /// root + `Some` variant owners, the payload `ValueType`, and the nested
    /// `Layout` ADT owner, keyed off the `.ok()` destination `Option<Layout>`
    /// type.  Gated on the payload being `core::alloc::layout::Layout` so only
    /// the `from_size_align(..).ok()` shape is recorded; the post-pass validates
    /// the preceding `from_size_align` residual.  The `Option` root is
    /// **instantiation-suffixed** (`resolve_bool_then_option_dest`):
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
        let (option_owner, some_owner, payload_ty) = self.resolve_bool_then_option_dest(dest_ty)?;
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

    /// Dest type of the producer of `var`: the callee `FunDecl` output
    /// (the same substitution `lower_call` uses when the dest place is a
    /// bare `Option` type_id). Recasts and forwarded successor inputargs
    /// are identity; chase them to that output.
    pub(in crate::front::mir) fn recover_option_dest_ty(&self, var: &Variable) -> Option<TyRef> {
        self.recover_option_dest_ty_depth(var, 0)
    }

    pub(in crate::front::mir) fn recover_option_dest_ty_depth(
        &self,
        var: &Variable,
        depth: usize,
    ) -> Option<TyRef> {
        if depth > 8 {
            return None;
        }
        for block in &self.graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() != Some(var) {
                    continue;
                }
                if let Some(src) = recast_operand(&op.kind) {
                    return self.recover_option_dest_ty_depth(src, depth + 1);
                }
                if let OpKind::Call { target, .. } = &op.kind
                    && let Some(id) = target.fun_decl_id()
                    && let Some(fd) = self.llbc.fn_by_id(id)
                {
                    return Some(clone_tyref(&fd.signature.output));
                }
                return None;
            }
        }
        // `lower_call` closes the producer block, so `opt?` reads the dest
        // as a successor inputarg. Chase the forwarded predecessor arg to
        // the FunDecl output the producer recast onto.
        for (bi, block) in self.graph.blocks.iter().enumerate() {
            let Some(pos) = block.inputargs.iter().position(|v| v == var) else {
                continue;
            };
            for pred in &self.graph.blocks {
                for link in &pred.exits {
                    if link.target.0 != bi {
                        continue;
                    }
                    let Some(LinkArg::Value(src)) = link.args.get(pos) else {
                        continue;
                    };
                    if let Some(ty) = self.recover_option_dest_ty_depth(src, depth + 1) {
                        return Some(ty);
                    }
                }
            }
        }
        None
    }

    /// When `recv_ty` is a bare `Option` type_id, recover the dest type
    /// the producer interned (`FunDecl` output, or a `checked_*` site
    /// already keyed off that dest type) so `__pos_0` keys that ClassDef
    /// (`bookkeeper.py` `getuniqueclassdef(cls)`). Variant owners come
    /// from [`Self::tagged_pair_payload_owner`] (`v.name`).
    pub(in crate::front::mir) fn recover_option_instantiation(
        &self,
        recv_ty: &TyRef,
        opt_var: Option<&Variable>,
        option_owner: &mut String,
        some_owner: &mut String,
        niche: &mut bool,
    ) {
        if *niche || !tyref_enum_instantiation_suffix(recv_ty, self.llbc).is_empty() {
            return;
        }
        let Some(opt_var) = opt_var else {
            return;
        };
        if let Some((owner, some)) = self.checked_arith_uint_owners(opt_var) {
            *option_owner = owner;
            *some_owner = some;
            return;
        }
        let Some(dest) = self.recover_option_dest_ty(opt_var) else {
            return;
        };
        if crate::front::result_exc::tyref_is_option(&dest, self.llbc) {
            let payload = crate::front::result_exc::tyref_option_payload(&dest, self.llbc);
            if payload
                .as_ref()
                .is_some_and(|ty| tyref_is_type_parameter(ty, self.llbc))
            {
                return;
            }
            if tyref_enum_instantiation_suffix(&dest, self.llbc).is_empty()
                && !self.tyref_is_niche_option_ptr(&dest)
            {
                return;
            }
            if let Some((owner, some, _)) = self.resolve_option_consumer_owners(&dest) {
                *option_owner = owner;
                *some_owner = some;
                *niche = self.tyref_is_niche_option_ptr(&dest);
            }
        } else {
            *niche = true;
        }
    }

    pub(in crate::front::mir) fn checked_arith_uint_owners(
        &self,
        var: &Variable,
    ) -> Option<(String, String)> {
        self.checked_arith_uint_owners_depth(var, 0)
    }

    pub(in crate::front::mir) fn checked_arith_uint_owners_depth(
        &self,
        var: &Variable,
        depth: usize,
    ) -> Option<(String, String)> {
        if depth > 8 {
            return None;
        }
        if let Some(site) = self
            .checked_arith_uint_sites
            .iter()
            .find(|site| site.opt == *var)
        {
            return Some((site.option_owner.clone(), site.some_owner.clone()));
        }
        for block in &self.graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() != Some(var) {
                    continue;
                }
                let src = recast_operand(&op.kind)?;
                return self.checked_arith_uint_owners_depth(src, depth + 1);
            }
        }
        for (bi, block) in self.graph.blocks.iter().enumerate() {
            let Some(pos) = block.inputargs.iter().position(|v| v == var) else {
                continue;
            };
            for pred in &self.graph.blocks {
                for link in &pred.exits {
                    if link.target.0 != bi {
                        continue;
                    }
                    let Some(LinkArg::Value(src)) = link.args.get(pos) else {
                        continue;
                    };
                    if let Some(owners) = self.checked_arith_uint_owners_depth(src, depth + 1) {
                        return Some(owners);
                    }
                }
            }
        }
        None
    }

    /// Dest type of a bare `Option` place with the payload as its type
    /// argument. [`Self::resolve_bool_then_option_dest`] then reads the
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
    /// [`Self::resolve_bool_then_option_dest`].
    pub(in crate::front::mir) fn instantiate_option_owners_from_payload(
        &self,
        dest_ty: &TyRef,
        payload_hint: Option<&TyRef>,
        option_owner: &mut String,
        some_owner: &mut String,
    ) {
        if !tyref_enum_instantiation_suffix(dest_ty, self.llbc).is_empty() {
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
        if tyref_enum_instantiation_suffix(&inst, self.llbc).is_empty() {
            return;
        }
        if let Some((owner, some, _)) = self.resolve_bool_then_option_dest(&inst) {
            *option_owner = owner;
            *some_owner = some;
        }
    }

    pub(in crate::front::mir) fn recognize_option_try_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
        opt_var: Option<&Variable>,
    ) -> Option<crate::front::option_try::OptionTrySite> {
        let recv_ty = recv_ty?;
        let recv_ty_owned = self
            .peel_to_option_or_result(recv_ty)
            .unwrap_or_else(|| clone_tyref(recv_ty));
        if !crate::front::result_exc::tyref_is_option(&recv_ty_owned, self.llbc) {
            return None;
        }
        // The branched `__discriminant` / `__pos_0` reads key the SAME classdef
        // the branched Option's producer wrote — per-instantiation for a
        // ref/str/float payload, bare for the `Int`/`Unsigned` carve-out.  A
        // cross-payload `?` (branch `Option<usize>`, return `Option<Layout>`)
        // is fine: the branched owners here are independent of the enclosing
        // function's return owner (`front::option_try` reads the Some arm from
        // these owners and builds the None arm from the return owner).
        let (mut option_owner, mut some_owner, payload_ty) =
            self.resolve_option_consumer_owners(&recv_ty_owned)?;
        let mut niche = self.tyref_is_niche_option_ptr(&recv_ty_owned);
        // The dest place is often a bare `Option` type_id. The payload
        // type (FunDecl output, or a `usize` operand of `checked_add`)
        // still names the instantiation the `Some` constructor and the
        // `?` getattr must share (`bookkeeper.py` `getuniqueclassdef`).
        let payload_hint =
            crate::front::result_exc::tyref_option_payload(&recv_ty_owned, self.llbc);
        self.instantiate_option_owners_from_payload(
            &recv_ty_owned,
            payload_hint.as_ref(),
            &mut option_owner,
            &mut some_owner,
        );
        // A producer whose dest place lost its generics still recasts the
        // value onto the FunDecl output's instantiation (or onto the
        // payload class of a pointer-niche Option). Route the `?` onto
        // that same ClassDef so getattr `__pos_0` does not hit the
        // template `Some` whose payload is the typevar. `lower_call`
        // closes the producer block, so chase a forwarded inputarg.
        self.recover_option_instantiation(
            &recv_ty_owned,
            opt_var,
            &mut option_owner,
            &mut some_owner,
            &mut niche,
        );
        let scalar_niche = !niche && tyref_option_nonzero_scalar(&recv_ty_owned, self.llbc);
        Some(crate::front::option_try::OptionTrySite {
            branch_result_var: result_var.clone(),
            option_owner,
            some_owner,
            payload_ty,
            niche,
            int_null: self.option_niche_int_null(&recv_ty_owned),
            scalar_niche,
            niche_null_cast: self.option_niche_null_cast(&recv_ty_owned),
        })
    }

    /// Resolve `Try::branch(res)` where `res: Result<T, E>` and `E` is not the
    /// exception carrier.  Carrier Results stay on `result_exc`; a miss here
    /// leaves the residual `branch` call.
    pub(in crate::front::mir) fn recognize_result_try_site(
        &self,
        recv_ty: Option<&TyRef>,
        result_var: &Variable,
    ) -> Option<ResultTrySite> {
        let recv_ty = self.peel_to_option_or_result(recv_ty?)?;
        if !crate::front::result_exc::tyref_is_result(&recv_ty, self.llbc)
            || crate::front::result_exc::tyref_is_result_of_carrier(
                &recv_ty,
                self.llbc,
                self.static_addrs.error_carrier,
            )
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
            || crate::front::result_exc::tyref_is_result_of_carrier(
                recv_ty,
                self.llbc,
                self.static_addrs.error_carrier,
            )
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
        if name != "branch" || !receiver_root.as_deref().unwrap_or("").ends_with("Result") {
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
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(ty, self.llbc)
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
                    site.call_result_ty = tyref_to_value_type_with(
                        &dest,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    site.args_tuple_suffix =
                        crate::front::result_exc::tyref_result_ok(&recv_ty, self.llbc)
                            .map(|ty| payload_tuple_suffix(&ty, self.llbc))
                            .unwrap_or_default();
                }
            }
            DiscCombinator::ResultUnwrapOrElse => {
                site.call_result_ty = tyref_to_value_type_with(
                    dest_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
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
                site.call_result_ty = tyref_to_value_type_with(
                    &dest,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
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

    /// Resolve a recognized `Option::map_or(opt, default, callable)` call into a
    /// [`crate::front::option_map_or::MapOrSite`] — the `Option` enum root +
    /// `Some` variant owners, the closure env's `call_once` owner or the
    /// function-item path, the payload type `T` (the `Some::__pos_0` read the
    /// callable consumes), and the result type `U` (the `call_once` / select
    /// result).  `None` (leaving the residual call) when the receiver is not a
    /// resolvable `Option` — guarding on `Option` since `Result::map_or` shares
    /// the name but has different tags — or the callable is neither a closure
    /// ADT nor a named function item.
    pub(in crate::front::mir) fn recognize_map_or_site(
        &self,
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        fn_item_segments: Option<Vec<String>>,
        fn_item_decl: Option<&FunDecl>,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::option_map_or::MapOrSite> {
        let recv_ty = recv_ty?;
        if !crate::front::result_exc::tyref_is_option(recv_ty, self.llbc) {
            return None;
        }
        let def_id = self.tyref_adt_def_id(recv_ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        // Suffix the enum root with the receiver `Option<X>`'s `<X>` so the
        // per-instantiation root a static `Some(..)` mints is reused here;
        // `some_owner` then derives `Option<X>::Some`.
        let option_owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(recv_ty, self.llbc)
        );
        let some_owner = Self::tagged_pair_payload_owner(td, &option_owner, 1)?;
        let payload_ty = self.tyref_option_payload_value_type(recv_ty)?;
        let payload_class_root = self.option_payload_instance_class_root(recv_ty);
        // Prefer a closure ADT (`call_once(env, (x,))`).  A function item has
        // no ADT def id — `tyref_ref_adt_def_id` misses — so fall through to
        // the `FnDef` path captured from the operand (or from `env_ty` when
        // the item was spilled to a local).
        let fn_item_segments = fn_item_segments
            .or_else(|| env_ty.and_then(|ty| tyref_fn_def_call_segments(ty, self.llbc)));
        let (call_once_owner, args_tuple_suffix, fn_item_segments) =
            if let Some(env_def_id) = env_ty.and_then(|ty| self.tyref_ref_adt_def_id(ty)) {
                let env_td = self.llbc.type_by_id(env_def_id)?;
                (
                    closure_env_call_once_owner(env_td),
                    option_payload_tuple_suffix(recv_ty, self.llbc),
                    None,
                )
            } else {
                let segments = fn_item_segments.filter(|s| !s.is_empty())?;
                (String::new(), String::new(), Some(segments))
            };
        let fn_item_ctor = match fn_item_decl {
            Some(fd) if fn_item_segments.is_some() && fd.is_adt_constructor() => {
                Some(self.constructor_fn_item_shape(fd, dest_ty)?)
            }
            _ => None,
        };
        let result_ty = tyref_to_value_type_with(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        let niche = self.tyref_is_niche_option_ptr(recv_ty);
        let scalar_kind = if niche {
            None
        } else {
            option_nonzero_word_kind(recv_ty, self.llbc)
        };
        let scalar_niche = scalar_kind.is_some();
        let payload_ty = scalar_kind.unwrap_or(payload_ty);
        Some(crate::front::option_map_or::MapOrSite {
            result_var: result_var.clone(),
            option_owner,
            some_owner,
            call_once_owner,
            fn_item_segments,
            fn_item_ctor,
            payload_ty,
            payload_class_root,
            result_ty,
            args_tuple_suffix,
            niche,
            int_null: self.option_niche_int_null(recv_ty),
            scalar_niche,
            niche_null_cast: self.option_niche_null_cast(recv_ty),
        })
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
            .map(|ty| {
                tyref_to_value_type_with(
                    &ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            })
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

    /// Resolve `Result<T, E>::map_err(result, closure) -> Result<T, F>` into
    /// the owners and payload types needed by `front::result_map_err`.
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
            recv_decl.item_meta.name_path(),
            tyref_enum_instantiation_suffix(recv_ty, self.llbc)
        );
        let result_owner = format!(
            "{}{}",
            dest_decl.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let receiver_ok_owner = Self::tagged_pair_payload_owner(recv_decl, &receiver_owner, 0)?;
        let receiver_err_owner = Self::tagged_pair_payload_owner(recv_decl, &receiver_owner, 1)?;
        let result_ok_owner = Self::tagged_pair_payload_owner(dest_decl, &result_owner, 0)?;
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
            result_ok_owner,
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
        if depth > 16 || self.type_decl_has_explicit_drop(decl.def_id) {
            return false;
        }
        if decl.published_struct_copy {
            return false;
        }
        match &decl.kind {
            TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => fields
                .iter()
                .all(|field| self.tyref_is_trivially_dropless(&field.ty, depth + 1)),
            TypeDeclKind::Enum(variants) => variants.iter().all(|variant| {
                variant
                    .fields
                    .iter()
                    .all(|field| self.tyref_is_trivially_dropless(&field.ty, depth + 1))
            }),
            TypeDeclKind::Alias(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => false,
        }
    }

    pub(in crate::front::mir) fn tyref_is_trivially_dropless(
        &self,
        ty: &TyRef,
        depth: usize,
    ) -> bool {
        if depth > 16 {
            return false;
        }
        let Some(node) = tyref_node(ty, self.llbc) else {
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
        let Some(def_id) = strip_ty_wrappers(node, self.llbc).and_then(adt_node_def_id) else {
            return false;
        };
        self.llbc
            .type_by_id(def_id)
            .is_some_and(|decl| self.type_decl_is_trivially_dropless(decl, depth + 1))
    }

    pub(in crate::front::mir) fn type_decl_has_explicit_drop(&self, def_id: u64) -> bool {
        self.llbc.has_explicit_drop_impl(def_id, |ty| {
            resolve_tyexpr_to_adt_def_id_free(self.llbc, ty)
        })
    }

    /// Resolve a recognized `Option::map`/`and_then`/`unwrap_or_else(opt,
    /// callable)` call into a
    /// [`crate::front::option_closure_select::ClosureSelectSite`] — the `Option`
    /// enum root + `Some` variant owners, the callable (closure env
    /// `call_once` owner, or function-item `FunctionPath` segments), the
    /// payload type `T`, and the call result type (`U` for `map`, `Option<U>`
    /// for `and_then`, `T` for `unwrap_or_else`).  `None` (leaving the residual
    /// call) when the receiver is not a resolvable `Option`, the callable is
    /// neither a closure ADT nor a named function item, or (for `map`) the
    /// result is not an `Option`.
    pub(in crate::front::mir) fn recognize_closure_select_site(
        &self,
        kind: crate::front::option_closure_select::ClosureCombinator,
        recv_ty: Option<&TyRef>,
        env_ty: Option<&TyRef>,
        fn_item_segments: Option<Vec<String>>,
        fn_item_decl: Option<&FunDecl>,
        dest_ty: &TyRef,
        result_var: &Variable,
    ) -> Option<crate::front::option_closure_select::ClosureSelectSite> {
        use crate::front::option_closure_select::ClosureCombinator;
        let recv_ty = recv_ty?;
        let recv_ty = self
            .tyref_peel_ref_to_pointee(recv_ty)
            .unwrap_or_else(|| clone_tyref(recv_ty));
        if !crate::front::result_exc::tyref_is_option(&recv_ty, self.llbc) {
            return None;
        }
        let def_id = self.tyref_adt_def_id(&recv_ty)?;
        let td = self.llbc.type_by_id(def_id)?;
        // Suffix the enum root with the receiver `Option<X>`'s `<X>` so the
        // per-instantiation root a static `Some(..)` mints is reused here;
        // `some_owner` then derives `Option<X>::Some`.
        let option_owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(&recv_ty, self.llbc)
        );
        let some_owner = Self::tagged_pair_payload_owner(td, &option_owner, 1)?;
        let mut payload_ty = self.tyref_option_payload_value_type(&recv_ty)?;
        let recv_scalar = option_nonzero_word_kind(&recv_ty, self.llbc);
        let recv_is_scalar = recv_scalar.is_some();
        if let Some(kind) = recv_scalar {
            payload_ty = kind;
        }
        let payload_class_root = self.option_payload_instance_class_root(&recv_ty);
        // Prefer a closure ADT (`call_once(env, (x,))`).  A function item has
        // no ADT def id — `tyref_ref_adt_def_id` misses — so fall through to
        // the `FnDef` path captured from the operand (or from `env_ty` when
        // the item was spilled to a local).
        let fn_item_segments = fn_item_segments
            .or_else(|| env_ty.and_then(|ty| tyref_fn_def_call_segments(ty, self.llbc)));
        let (call_once_owner, args_tuple_suffix, fn_item_segments) =
            if let Some(env_def_id) = env_ty.and_then(|ty| self.tyref_ref_adt_def_id(ty)) {
                let env_td = self.llbc.type_by_id(env_def_id)?;
                (
                    closure_env_call_once_owner(env_td),
                    option_payload_tuple_suffix(&recv_ty, self.llbc),
                    None,
                )
            } else {
                let segments = fn_item_segments.filter(|s| !s.is_empty())?;
                (String::new(), String::new(), Some(segments))
            };
        // The type the closure's `call_once` returns: `map`'s dest is
        // `Option<U>` and its closure returns `U` (the dest payload);
        // `and_then`'s dest is `Option<U>` returned directly; `or_else`'s dest
        // is `Option<T>` and its closure returns that same `Option<T>`;
        // `unwrap_or_else`'s dest is the bare `T`; `is_some_and` closures and
        // dests are both `bool`.
        let call_result_tyref = match kind {
            ClosureCombinator::Map => {
                crate::front::result_exc::tyref_option_payload(dest_ty, self.llbc)?
            }
            ClosureCombinator::AndThen
            | ClosureCombinator::OrElse
            | ClosureCombinator::UnwrapOrElse
            | ClosureCombinator::IsSomeAnd => clone_tyref(dest_ty),
        };
        let mut call_result_ty = tyref_to_value_type_with(
            &call_result_tyref,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        // A constructor function item builds its aggregate in the arm; one
        // whose shape does not resolve declines the rewrite.
        let fn_item_ctor = match fn_item_decl {
            Some(fd) if fn_item_segments.is_some() && fd.is_adt_constructor() => {
                Some(self.constructor_fn_item_shape(fd, &call_result_tyref)?)
            }
            _ => None,
        };
        // A constructor builds its `Result` value; it never raises.
        let call_once_result_exc = (fn_item_ctor.is_none()
            && crate::front::result_exc::tyref_is_result_of_carrier(
                &call_result_tyref,
                self.llbc,
                self.static_addrs.error_carrier,
            ))
        .then(|| {
            let suffix = crate::front::result_exc::tyref_result_instantiation_suffix(
                &call_result_tyref,
                self.llbc,
            );
            let payload_ty =
                crate::front::result_exc::tyref_result_ok(&call_result_tyref, self.llbc)
                    .map(|ty| {
                        tyref_to_value_type_with(
                            &ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    })
                    .unwrap_or(ValueType::Ref(None));
            (suffix, payload_ty)
        });
        let niche = self.tyref_is_niche_option_ptr(&recv_ty);
        let mut fieldless_none_tag =
            tyref_option_fieldless_niche(&recv_ty, self.llbc).map(|niche| niche.none_tag);
        if recv_is_scalar {
            fieldless_none_tag = Some(0);
        }
        // `map`/`and_then` BUILD their result `Option<U>`; the other combinators
        // build none.  `U` is the dest payload, not the receiver's `T`, so the
        // built variant must key the dest's own classdef — otherwise two `map`s
        // over the same receiver type with different closure results share one
        // `__pos_0` slot, the collision `recognize_from_size_align_ok_site`
        // documents at its own site.  `resolve_option_consumer_owners` is the
        // derivation every downstream reader of the built value (`unwrap_or`,
        // `?`, `expect`) uses, so producing anything else payload-erases the
        // read; a dest that is not a resolvable `Option` declines the whole
        // rewrite and leaves the residual call.
        let (result_option_owner, result_some_owner, result_niche) = match kind {
            ClosureCombinator::Map | ClosureCombinator::AndThen => {
                let (owner, some, _payload) = self.resolve_option_consumer_owners(dest_ty)?;
                (owner, some, self.tyref_is_niche_option_ptr(dest_ty))
            }
            _ => (option_owner.clone(), some_owner.clone(), niche),
        };
        let dest_scalar = match kind {
            ClosureCombinator::Map | ClosureCombinator::AndThen => {
                option_nonzero_word_kind(dest_ty, self.llbc)
            }
            _ if recv_is_scalar => Some(payload_ty.clone()),
            _ => None,
        };
        let dest_is_scalar = dest_scalar.is_some();
        if matches!(kind, ClosureCombinator::Map)
            && let Some(kind) = dest_scalar
        {
            call_result_ty = kind;
        }
        let result_fieldless_none_tag = match kind {
            ClosureCombinator::Map | ClosureCombinator::AndThen if dest_is_scalar => Some(0),
            ClosureCombinator::Map | ClosureCombinator::AndThen => {
                tyref_option_fieldless_niche(dest_ty, self.llbc).map(|niche| niche.none_tag)
            }
            _ => fieldless_none_tag,
        };
        let niche_null_cast = self.option_niche_null_cast(&recv_ty);
        let result_niche_null_cast = match kind {
            ClosureCombinator::Map | ClosureCombinator::AndThen => {
                self.option_niche_null_cast(dest_ty)
            }
            _ => niche_null_cast.clone(),
        };
        Some(crate::front::option_closure_select::ClosureSelectSite {
            kind,
            result_var: result_var.clone(),
            option_owner,
            some_owner,
            result_option_owner,
            result_some_owner,
            result_niche,
            result_int_null: match kind {
                ClosureCombinator::Map | ClosureCombinator::AndThen => {
                    self.option_niche_int_null(dest_ty)
                }
                _ => None,
            },
            result_niche_null_cast,
            result_fieldless_none_tag,
            call_once_owner,
            fn_item_segments,
            fn_item_ctor,
            payload_ty,
            payload_class_root,
            call_result_ty,
            args_tuple_suffix,
            niche,
            int_null: self.option_niche_int_null(&recv_ty),
            niche_null_cast,
            fieldless_none_tag,
            call_once_result_exc,
        })
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
    /// an Address. [`tyref_to_value_type_with`] already puts `Option<fn>` and
    /// a raw-address niche ([`ty_is_raw_address`], e.g. `Option<&RawEnum>`)
    /// in `Int`; a GC pointer / `SomeString` niche stays `Ref`.
    pub(in crate::front::mir) fn option_niche_is_int_bank(&self, option_ty: &TyRef) -> bool {
        if !self.tyref_is_niche_option_ptr(option_ty) {
            return false;
        }
        matches!(
            tyref_to_value_type_with(
                option_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            ValueType::Int | ValueType::Unsigned
        )
    }

    /// Int-bank null of a niche `Option`: `null_fn` with the `fn(..)`
    /// spelling, or `null_raw` with the raw pointee's class root
    /// (`lltype.nullptr` / `llmemory.NULL`). `None` when the niche is
    /// ref-banked (`push_niche_null`).
    pub(in crate::front::mir) fn option_niche_int_null(
        &self,
        option_ty: &TyRef,
    ) -> Option<crate::model::NicheIntNull> {
        if !self.option_niche_is_int_bank(option_ty) {
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
        Some(crate::model::NicheIntNull::RawAddress(
            raw_address_owner_root(
                &seed,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
        ))
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
    /// `Option<&str>` / `Option<String>` (and the other string-family
    /// payloads [`node_is_string_value`] already projects) become a
    /// nullable `SomeString`.
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
            .is_some_and(|td| td.item_meta.name_path() == "core::ptr::non_null::NonNull")
    }

    /// `true` when `option_ty` is a niche nullable `Option<P>` whose
    /// paint is `ValueType::Str` ([`tyref_is_option_of_string_value`]):
    /// builtin `str`, `Utf8Str` / `BytesBlock`, shared `&str` /
    /// `&Wtf8` / `&Wtf8Buf`, or by-value `Wtf8Buf` / `Utf8Str`.
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
    ///   `Option<&str>` / `Option<Wtf8Buf>` and
    ///   the other `Str` Options) — `rstr.py`
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
        let from_path = if fmt_path_ends_with_owner_method(
            segments,
            &["option", "Option"],
            &["core", "cmp", "PartialEq"],
            "eq",
        ) {
            Some("eq")
        } else if fmt_path_ends_with_owner_method(
            segments,
            &["option", "Option"],
            &["core", "cmp", "PartialEq"],
            "ne",
        ) {
            Some("ne")
        } else {
            None
        };
        // `CallControl.graphs_from` / `FunctionReprBase.call` call the
        // function the pointer names. Only `core::cmp::PartialEq`'s
        // `eq`/`ne` lower to this `BinOp`; an extension trait's
        // same-named method on `Option` stays a call.
        let from_method = method_hint.and_then(|(owner, name, trait_identity)| {
            let identity = trait_identity.as_ref()?;
            (owner == "Option"
                && matches!(name.as_str(), "eq" | "ne")
                && trait_impl_identity_names_trait(
                    &format!("<impl {identity}>"),
                    &["core", "cmp", "PartialEq"],
                ))
            .then_some(name.as_str())
        });
        let Some(leaf) = from_path.or(from_method) else {
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
        self.graph
            .push_niche_null(bb_id, self.option_niche_null_cast(option_ty).as_ref())
    }

    /// Lower `i64::checked_neg()` (`core::num::<Impl>::checked_neg` —
    /// core fn bodies are Opaque in the LLBC, so the `Call` form is
    /// permanently unliftable) to a decomposed ovfcheck shape.
    /// Upstream `translator/simplify.py transform_ovfcheck`
    /// rewrites `ovfcheck(-x)` into the op's `_ovf` variant
    /// (`flowspace/operation.py ovfchecked`, registered by
    /// `operation.py add_operator('neg', ..., ovf=True)`), whose
    /// OverflowError edge the caller branches on.  Rust spells that
    /// ovfcheck as `checked_neg()` + a `Some`/`None` match, so the
    /// equivalent decomposition writes the destination `Option<i64>`
    /// as a synthetic aggregate (the same transparent-ctor +
    /// `FieldWrite` chain `Rvalue::Aggregate` emits):
    /// `__discriminant = ne(v, i64::MIN)` — negation overflows only
    /// at `i64::MIN`, and `Option`'s `None`/`Some` tags are 0/1 — and
    /// payload `__pos_0 = neg(v)` (wrapping; the `None` arm never
    /// reads it).  The downstream discriminant switch and payload
    /// downcast then lower through the ordinary enum FieldRead paths.
    /// The `i64::MIN` sentinel is only correct at word width, so the
    /// lowering is gated on a word-sized signed operand (`i64`/`isize`)
    /// — a narrower `checked_neg` overflows at its own narrower `MIN`
    /// and keeps the generic `Call` form (none arise today; the live
    /// callers are `neg`'s `int_value` and `functional`'s `step`,
    /// both `i64`).
    /// Lower `{i64,u64}::wrapping_{add,sub,mul}`
    /// (`core::num::<Impl>::wrapping_*`, Opaque in the LLBC like every
    /// core fn) to the native `BinOp("add"/"sub"/"mul")`.  `Signed`
    /// arithmetic is modular machine arithmetic — `rint.py rtype_add
    /// rtype_add` emits `int_add` with no overflow check, and `int_add`
    /// wraps (`rarithmetic.py intmask` semantics) — so the wrapping
    /// method IS the plain llop.  A narrower receiver lowers to that word
    /// op followed by a truncation back to its width — `int_and` for an
    /// unsigned target, `int_signext` for a signed one — the way `rint.py`
    /// `cast_primitive` and `jtransform.py` `_int_to_int_cast` narrow.
    ///
    /// Both integer banks count, the lesson [`vec_index_type_is_scalar`]
    /// already carries: `usize` serializes as `{"Integer": {"Unsigned": "Usize"}}`, which
    /// `tyref_literal_int_atom` does not see at all, so an `Int`-only
    /// test silently declined every unsigned counter (`keys_version`,
    /// `clear_gen`) and left the enclosing graph blocked on the call.
    /// Unsigned is the same machine op under a different rtyper
    /// dispatch: `rint.py`'s `opprefix` makes it `uint_add`, and
    /// `jtransform.py` `Transformer` renames `uint_{add,sub,mul}` straight
    /// back to `int_{add,sub,mul}` for the JIT.
    ///
    /// The result carries the receiver's signedness rather than a flat
    /// `Int`.  `union_type` widens `Int ∪ Unsigned` to `Unknown`
    /// (`binaryop.py` UnionError), so annotating a `usize` sum as
    /// `Int` would poison the merge where it meets the unsigned field
    /// read it came from.
    /// `CodeFlags::{contains,intersects,bitor}` as the integer operations
    /// `bitflags` expands them to: `contains` is `(self & other) == other`,
    /// `intersects` is `(self & other) != 0`, `bitor` is `self | other`.
    /// `contains` / `intersects` take `&self`, a borrow of the transparent
    /// scalar, so their receiver is the word `*self` reads; `bitor` and the
    /// `other` operand are by value. `bits` is that same `*self` word: the
    /// integer `bits()` returns, not a residual FunctionPath (the impl is
    /// outside the extracted crates; `jtransform.py`
    /// `Transformer.handle_residual_call` needs a real function pointer).
    /// Bit ops use the `bitand` / `bitor` labels and `Unsigned` result a
    /// `u32` `&` / `|` lowers to. A compare uses `eq` / `ne` with the `Int`
    /// result a `u32` `==` / `!=` lowers to, because that destination is
    /// `bool`. The flag constants are the `Unsigned` values
    /// [`code_flags_const`] emits.
    pub(in crate::front::mir) fn try_lower_code_flags_method(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        arg_locals: &[Option<usize>],
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let Some((leaf, flags_ty)) = code_flags_method(self.llbc, reg) else {
            return Ok(false);
        };
        if leaf == "bits" {
            let Some(Some(receiver)) = arg_locals.first() else {
                return Ok(false);
            };
            let place = Place {
                kind: PlaceKind::Projection(
                    Box::new(Place {
                        kind: PlaceKind::Local(*receiver as u64),
                        ty: unit_tyref(),
                    }),
                    ProjectionElem::Atom("Deref".to_string()),
                ),
                ty: flags_ty,
            };
            let res = self.resolve_place(mir_bb, place)?;
            self.local_var[dest_local] = Some(LocalValue::One(res));
            let bb_id = self.emit_bb(mir_bb);
            let target_bb = self.block_id[target];
            let link_args = self.edge_args(mir_bb, target)?;
            self.graph.set_goto(bb_id, target_bb, link_args);
            return Ok(true);
        }
        let [lhs, rhs] = args else {
            return Ok(false);
        };
        let lhs = if leaf == "bitor" {
            lhs.clone()
        } else {
            let Some(Some(receiver)) = arg_locals.first() else {
                return Ok(false);
            };
            let place = Place {
                kind: PlaceKind::Projection(
                    Box::new(Place {
                        kind: PlaceKind::Local(*receiver as u64),
                        ty: unit_tyref(),
                    }),
                    ProjectionElem::Atom("Deref".to_string()),
                ),
                ty: flags_ty,
            };
            self.resolve_place(mir_bb, place)?
        };
        let rhs = rhs.clone();
        let bb_id = self.emit_bb(mir_bb);
        let res = match leaf {
            "contains" => {
                let masked = self.emit_uint_binop(mir_bb, "bitand", lhs, rhs.clone());
                self.emit_int_binop(bb_id, "eq", masked, rhs)
            }
            "intersects" => {
                let masked = self.emit_uint_binop(mir_bb, "bitand", lhs, rhs);
                let zero = self.emit_const_uint(mir_bb, 0);
                self.emit_int_binop(bb_id, "ne", masked, zero)
            }
            "bitor" => self.emit_uint_binop(mir_bb, "bitor", lhs, rhs),
            _ => return Ok(false),
        };
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }
}
