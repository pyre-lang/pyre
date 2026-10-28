//! Hand lowering of std/core/pyre callees (`emit_w_str_utf8_read` … `nonnull_new_identity_alias`).

use super::*;

impl<'a> Lowering<'a> {
    /// `w_obj._utf8` — narrow the `PyObjectRef` receiver to
    /// `W_UnicodeObject` and read its `value` field, the rstr `STR`
    /// payload.
    pub(in crate::front::mir) fn emit_w_str_utf8_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
    ) -> Variable {
        let narrowed = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(narrowed.clone()),
            kind: crate::model::cast_instance_call("W_UnicodeObject", base),
        });
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::FieldRead {
                base: narrowed,
                field: FieldDescriptor::new("value", Some("W_UnicodeObject".to_string()))
                    .with_owner_id(Some(majit_ir::descr::StructId::from_canonical(
                        "unicodeobject::W_UnicodeObject",
                    )))
                    .with_base_is_deref(true),
                ty: ValueType::Str,
                pure: false,
            },
        });
        result
    }

    /// `arr[i]` / `arr[i] = v` on a workspace fixed-array type —
    /// resolves to its `Index::index` / `IndexMut::index_mut` impl,
    /// whose body bottoms out at raw-slice construction
    /// (`as_mut_slice` → `from_raw_parts`) with no graph lowering.
    /// The structs are length-prefixed GcArray layouts (see
    /// `FixedObjectArray`), so the callsite IS RPython's
    /// getarrayitem/setarrayitem on the receiver and is devirtualized
    /// to `ArrayRead`/`ArrayWrite` by the caller.
    pub(in crate::front::mir) fn is_workspace_index_call(&self, reg: &RegularCall) -> bool {
        is_workspace_index_regular(reg, self.llbc)
    }

    /// `v[i]` / `v[i] = x` on a `Vec<T>` — its `<Vec<T> as Index>::index`
    /// (read) or `<Vec<T> as IndexMut>::index_mut` (write) impl.  Unlike
    /// the workspace fixed arrays ([`is_workspace_index_call`]), the
    /// receiver is a resized `ListRepr` (`length` + `items` block), so the
    /// element load goes through the rtyper's `getitem`
    /// (`AbstractBaseListRepr.rtype_getitem`, which loads the `items` field
    /// before indexing).  The caller lowers the read to `ArrayRead` and the
    /// paired `*p = v` write to `ArrayWrite`; the flowspace `getitem` /
    /// `setitem` they become are repr-dispatched, so the resized
    /// indirection is added by `rtype_getitem` / `rtype_setitem` rather than
    /// the front end.  Delegates to the free [`is_vec_index_regular`] so the
    /// same gate is shared with the deferred-write liveness pre-pass.
    pub(in crate::front::mir) fn is_vec_index_call(
        &self,
        reg: &RegularCall,
        index_ty: Option<&TyRef>,
    ) -> bool {
        is_vec_index_regular_with_callsite(
            reg,
            index_ty,
            self.llbc,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
    }

    /// `core::slice::index::SliceIndex<usize>::index(_mut)` — scalar slice
    /// indexing.  RPython represents the same operation directly as
    /// `getitem` / `setitem`; the Rust trait shim is opaque in Charon and must
    /// not survive as a residual call.  Range implementations share the same
    /// name, so accept only an integer-bank second argument. `usize` is the
    /// ordinary spelling, but Charon may preserve an integer alias instead of
    /// the literal atom; [`vec_index_type_is_scalar`] performs the same
    /// representation test used by `Vec::index`. RangeFrom / RangeTo remain
    /// Ref-bank values and continue through `front::slice_index`'s bounded
    /// getslice rewrites.
    pub(in crate::front::mir) fn is_slice_scalar_index_call(
        &self,
        reg: &RegularCall,
        index_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let is_index = self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::index::<Impl>::index" | "core::slice::index::<Impl>::index_mut"
            )
        });
        let callsite_index_is_scalar = index_ty.is_some_and(|ty| {
            vec_index_type_is_scalar(
                ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
        }) || reg
            .generics
            .get("types")
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.get(1))
            .and_then(|ty| serde_json::from_value::<TyRef>(ty.clone()).ok())
            .is_some_and(|ty| {
                vec_index_type_is_scalar(
                    &ty,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            });
        is_index && callsite_index_is_scalar
    }

    /// `<[T]>::get::<I>(slice, index)` / `get_mut` with a scalar `I`.  For a
    /// local index, MIR supplies the type through the operand Place; for a
    /// literal index it only survives in Charon's method generics (`[T, I]`).
    /// Preserve that distinction so `get(1)` takes the guarded-item Option
    /// diamond while `get(1..)` continues through `front::slice_index` as a
    /// subslice.  The FunDecl spelling is `core::slice::<Impl>::get{,_mut}`
    /// (and the `alloc::slice` twin); `core::slice::index::<Impl>::get` is
    /// not this function.
    pub(in crate::front::mir) fn is_slice_get_scalar_call(
        &self,
        reg: &RegularCall,
        index_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let is_get = self.llbc.fn_by_id(*id).is_some_and(|fd| {
            crate::front::slice_get::is_slice_get_name_path(&fd.item_meta.name_path())
        });
        is_get
            && callsite_or_generic_index_is_scalar(
                reg,
                index_ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
    }

    /// `true` when `reg` is `<[T]>::get_mut` (not `get`).  `get_mut` of a
    /// thin pointer is the same two-argument diamond as `get`; `get_mut` of
    /// a scalar is not recorded (`slice_get_element_may_record`).
    pub(in crate::front::mir) fn is_slice_get_mut_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            crate::front::slice_get::is_slice_get_mut_name_path(&fd.item_meta.name_path())
        })
    }

    /// `true` when `reg` is `core::slice::<Impl>::first` or `last`.
    /// Those share `get`'s `T` generic, so this gates the
    /// `slice_get_element` walk.
    pub(in crate::front::mir) fn is_slice_first_or_last_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::<Impl>::first" | "core::slice::<Impl>::last"
            )
        })
    }

    /// ARRAY identity proof for the scalar-index `<[T]>::get` rewrite.
    ///
    /// This is the `get` counterpart of the `Index::index` element gate: a
    /// scalar carries an element spelling so the descr computes its exact
    /// stride; a thin pointer is already one word and needs no identity. Any
    /// other inline Rust aggregate declines — including a string or tuple
    /// element, which is one GC pointer only inside a list the translation
    /// builds, never inside the host slice this call reads.
    /// `AtomicU64` (`portal_diag_bump`) and `Option<Entry>` (`RDict` slots)
    /// are that decline: an `ArrayRead` would copy the aggregate, and the
    /// consumer mutates the live slot (`fetch_add`, `*slot = …`).
    /// The outer `Option` is the proof, while the inner one is the optional
    /// ARRAY identity (`Some(None)` means the proven thin-pointer case).
    pub(in crate::front::mir) fn slice_get_element(
        &self,
        reg: &RegularCall,
    ) -> Option<(ValueType, Option<String>)> {
        let element_ty = reg
            .generics
            .get("types")
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.first())
            .and_then(|ty| serde_json::from_value::<TyRef>(ty.clone()).ok())?;
        let element = tyref_node(&element_ty, self.llbc)?;
        // `get` returns `Option<&T>`, but its devirtualized ArrayRead is the
        // same RPython getarrayitem operation as `Index::index`: it produces
        // T, not a Rust slot reference.  Read the bank from the call's T
        // generic before looking at the reference-wrapped destination.
        let item_ty = tyref_to_value_type_with(
            &element_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        // Identity only for an object pointer or a scalar element spelling.
        // A fallback type string must not hand an inline aggregate or an
        // unproven pointer an ARRAY identity. A thin pointer is one word
        // and carries no identity.
        if json_ty_is_objectptr(element, self.llbc) {
            let array_type_id = slice_array_type_id("*mut PyObject")
                .expect("an object-pointer slice names the object gcarray");
            return Some((item_ty, Some(array_type_id)));
        }
        if let Some(spelling) = json_ty_scalar_element_spelling(element, self.llbc) {
            let array_type_id =
                slice_array_type_id(&spelling).unwrap_or_else(|| format!("[{spelling}]"));
            return Some((item_ty, Some(array_type_id)));
        }
        json_ty_is_thin_pointer_element(element, self.llbc).then_some((item_ty, None))
    }

    pub(in crate::front::mir) fn is_slice_scalar_index_mut_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::slice::index::<Impl>::index_mut")
    }

    /// `<[T]>::swap(s, a, b)` (`core::slice::<Impl>::swap`) — an
    /// in-place element exchange through a `&mut [T]`.  The slice base
    /// is the same `FixedObjectArray` shape `is_workspace_index_call`
    /// reads, so the callsite lowers to the getarrayitem/setarrayitem
    /// decomposition rather than residualizing the Opaque core body.
    pub(in crate::front::mir) fn is_slice_swap_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::slice::<Impl>::swap")
    }

    /// Element type of a `core::slice::<Impl>::swap` call, read off the
    /// call's own type arguments (`generics.types[0]`) the way
    /// [`Self::tyref_adt_type_arg`] reads an ADT's.  The swap decomposition
    /// synthesizes `ArrayRead`/`ArrayWrite` pairs the MIR never spells, and
    /// `swap` is matched by name alone, so the element kind has to be the
    /// SLICE's rather than a fixed guess.
    ///
    /// Returns the `TyRef` itself, not a [`ValueType`], because the two
    /// consumers need different strengths of answer.  The item kind may use
    /// `tyref_to_value_type`'s `Ref(None)` fallback — that is what the
    /// decomposition assumed unconditionally before, so an unresolved
    /// element is no worse than the old behaviour.  The array identity may
    /// NOT: Charon runs `monomorphize:false`, so a call inside a generic
    /// body (`listsort::sort_with<T>` reaching `swap` through
    /// `reverse_slice`) carries a `TypeVar` here, and every unresolved shape
    /// lands on that same `Ref(None)`.  Reading the fallback as "this is the
    /// object block" would stamp the object-array descr onto the `i64` /
    /// `f64` / `usize` sort paths, aliasing them onto the real object array
    /// and giving them a pointer itemsize — 4 bytes on wasm32, where the
    /// scalar banks stride 8.
    pub(in crate::front::mir) fn slice_swap_elem_tyref(&self, reg: &RegularCall) -> Option<TyRef> {
        reg.generics
            .get("types")
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.first())
            .and_then(|node| serde_json::from_value::<TyRef>(node.clone()).ok())
    }

    /// `FixedObjectArray::set_ref(self, index, value)` — the GC-published
    /// element store whose body hand-rolls the write barrier.  Recognised as
    /// the receiver of a `setarrayitem_gc` collapse (`ll_setitem_fast`,
    /// rlist.py); the barrier is a backend rewrite, not a traced call.
    pub(in crate::front::mir) fn is_object_array_set_ref_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "pyre_object::object_array::<Impl>::set_ref"
        })
    }

    /// `object_array::items_block_set_ref(block, index, value)`, the
    /// `ItemsBlock` element store whose body hand-rolls the write barrier.
    pub(in crate::front::mir) fn is_items_block_set_ref_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "pyre_object::object_array::items_block_set_ref"
        })
    }

    /// Source-level write-barrier helpers in front of a following `setfield`.
    /// `typeobject::type_write_barrier` for `W_TypeObject` fields.
    pub(in crate::front::mir) fn is_source_write_barrier_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "pyre_object::typeobject::type_write_barrier"
        })
    }

    /// Address-taken `[T; N]` field whose result is `var`.
    ///
    /// `direct_arrayitems` (`lltype.py`) is that field's address. The
    /// header is the field-read base; the byte offset is the field's
    /// layout offset (`array_items_base` for a length-prefixed array).
    /// `get_array_token` reads `carray.items.offset` off the actual
    /// array, so element alignment comes from the source local and
    /// header. `dest_ty` is the `.cast()` dest and is only the
    /// instantiated element when those source types have no layout.
    /// An address-taken array field with no recoverable items offset
    /// stops translation rather than emit a different address.
    pub(in crate::front::mir) fn taken_array_field_address(
        &self,
        bb_id: BlockId,
        var: &Variable,
        dest_ty: Option<&TyRef>,
    ) -> Result<Option<(Variable, u64)>, LowerError> {
        let found = {
            let block = self.graph.block(bb_id);
            let mut found = None;
            for op in &block.operations {
                let OpKind::FieldRead { base, field, .. } = &op.kind else {
                    continue;
                };
                if op.result.as_ref() != Some(var) || !field.taken_by_address {
                    continue;
                }
                let Some(owner) = field.owner_root.clone() else {
                    continue;
                };
                found = Some((base.clone(), owner, field.name.clone()));
                break;
            }
            found
        };
        let Some(found) = found else {
            return Ok(None);
        };
        let instantiated_align = Self::array_tail_instantiated_elem_align(
            self.instantiated_elem_align_of_var(var, &found.2),
            self.instantiated_elem_align_of_var(&found.0, &found.2),
            dest_ty.and_then(|ty| self.instantiated_elem_align_from_tyref(ty, &found.2)),
        );
        let offset = Self::array_field_items_offset(
            self.array_field_tail_offset(&found.1, &found.2, instantiated_align),
            &found.1,
            &found.2,
        )?;
        Ok(Some((found.0, offset)))
    }

    /// Alignment of the array element named by MIR local `var`.
    pub(in crate::front::mir) fn instantiated_elem_align_of_var(
        &self,
        var: &Variable,
        field_name: &str,
    ) -> Option<usize> {
        for (i, slot) in self.local_var.iter().enumerate() {
            let Some(value) = slot else {
                continue;
            };
            let bound = match value {
                LocalValue::One(v) | LocalValue::ItemAddr(v) => v,
                LocalValue::Pair(v, _) => v,
                LocalValue::SliceIter { items, .. } => items,
            };
            if bound == var {
                return self.instantiated_elem_align_from_tyref(
                    &self.body.locals.locals.get(i)?.ty,
                    field_name,
                );
            }
        }
        None
    }

    /// Element alignment of an instantiated array-tail field, or of the
    /// `.cast()` dest / array element at a use site. Type variables (the
    /// generic declaration) are `None`.
    pub(in crate::front::mir) fn instantiated_elem_align_from_tyref(
        &self,
        ty: &TyRef,
        field_name: &str,
    ) -> Option<usize> {
        self.instantiated_elem_align_from_json(&tyref_json(ty), field_name)
    }

    pub(in crate::front::mir) fn instantiated_elem_align_from_json(
        &self,
        ty: &serde_json::Value,
        field_name: &str,
    ) -> Option<usize> {
        let body = tyexpr_body(self.llbc, ty)?;
        if let Some(raw) = body.get("RawPtr").and_then(serde_json::Value::as_array) {
            return self.instantiated_elem_align_from_json(raw.first()?, field_name);
        }
        if let Some(reference) = body.get("Ref").and_then(serde_json::Value::as_array) {
            return self.instantiated_elem_align_from_json(reference.get(1)?, field_name);
        }
        if let Some(arr) = body.get("Array").and_then(serde_json::Value::as_array) {
            return self.instantiated_elem_align_from_json(arr.first()?, field_name);
        }
        let adt = body.get("Adt")?.as_object()?;
        let td = self.llbc.type_by_id(type_decl_ref_adt_id(adt)?)?;
        if let TypeDeclKind::Struct(fields) = &td.kind
            && let Some(field) = fields
                .iter()
                .find(|field| field.name.as_deref() == Some(field_name))
            && let Some(node) = tyref_node(&field.ty, self.llbc)
            && let Some(stripped) = strip_ty_wrappers(node, self.llbc)
            && stripped
                .get("Array")
                .and_then(serde_json::Value::as_array)
                .is_some()
        {
            let subst = type_decl_ref_generics(adt, self.llbc)
                .and_then(|g| g.get("types"))
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            let (_, align) = mem_size_align(self.llbc, &tyref_json(&field.ty), &subst, 0)?;
            return usize::try_from(align).ok().filter(|&a| a > 0);
        }
        // `size_align_const_from_tyexpr` only resolves `Value` /
        // `Deduplicated` wrappers. A spec copy replaces those with a
        // plain `Adt` that already carries the instantiation.
        mem_size_align(self.llbc, body, &[], 0)
            .and_then(|(_, align)| usize::try_from(align).ok())
            .filter(|&a| a > 0)
    }

    /// Byte offset of the `[T; N]` field `field_name` on struct leaf
    /// `owner`. Every matching declaration must agree. A missing layout
    /// with the array at field 1 (the `usize` length word of
    /// `GcEntries`) is that word rounded up to `T`'s alignment
    /// (`array_items_base`). The declaration of a generic struct has no
    /// layout and unresolved type variables, so `instantiated_elem_align`
    /// is the use-site `T`. Unknown element alignment is `None`.
    pub(in crate::front::mir) fn array_field_tail_offset(
        &self,
        owner: &str,
        field_name: &str,
        instantiated_elem_align: Option<usize>,
    ) -> Option<u64> {
        let leaf = owner
            .rsplit("::")
            .next()
            .unwrap_or(owner)
            .split('<')
            .next()
            .unwrap_or(owner);
        let target = std::env::var("TARGET").unwrap_or_default();
        let mut offset: Option<u64> = None;
        let mut index: Option<usize> = None;
        let mut elem_align: Option<u64> = None;
        let mut saw = false;
        for td in self.llbc.iter_type_decls() {
            let name = td.item_meta.name_path();
            let td_leaf = name.rsplit("::").next().unwrap_or("");
            if td_leaf.split('<').next().unwrap_or(td_leaf) != leaf {
                continue;
            }
            let TypeDeclKind::Struct(fields) = &td.kind else {
                continue;
            };
            let Some(idx) = fields
                .iter()
                .position(|field| field.name.as_deref() == Some(field_name))
            else {
                continue;
            };
            let Some(node) = tyref_node(&fields[idx].ty, self.llbc) else {
                continue;
            };
            let Some(stripped) = strip_ty_wrappers(node, self.llbc) else {
                continue;
            };
            let Some(array) = stripped.get("Array").and_then(serde_json::Value::as_array) else {
                continue;
            };
            saw = true;
            match index {
                None => index = Some(idx),
                Some(prev) if prev != idx => return None,
                Some(_) => {}
            }
            if let Some(elem) = array.first()
                && let Some(align) = self
                    .size_align_const_from_tyexpr(true, elem)
                    .and_then(|a| u64::try_from(a).ok())
                    .filter(|&a| a > 0)
            {
                match elem_align {
                    None => elem_align = Some(align),
                    Some(prev) if prev != align => return None,
                    Some(_) => {}
                }
            }
            if let Some(layout) = td.layout_for_target(self.llbc, &target) {
                let off = layout.struct_field_offset(idx)?;
                match offset {
                    None => offset = Some(off),
                    Some(prev) if prev != off => return None,
                    Some(_) => {}
                }
            }
        }
        if let Some(off) = offset {
            return Some(off);
        }
        if saw && index == Some(1) {
            let align = elem_align
                .and_then(|a| usize::try_from(a).ok())
                .or(instantiated_elem_align);
            return Self::array_field_tail_offset_fallback(
                crate::layout::target_word_size(),
                align,
            );
        }
        None
    }

    /// Items offset of a length-prefixed `[T; 0]` sitting at field 1 when
    /// Charon recorded no layout. Items start at the length word rounded
    /// up to `T`'s alignment (`array_items_base` / `symbolic.get_array_token`).
    /// Unknown element alignment is `None`.
    pub(in crate::front::mir) fn array_field_tail_offset_fallback(
        word: usize,
        elem_align: Option<usize>,
    ) -> Option<u64> {
        let align = elem_align.filter(|&a| a > 0)?;
        Some(crate::codewriter::call::array_items_base(word, align) as u64)
    }

    /// `get_array_token` reads `carray.items.offset` off the actual array
    /// type. Source local and header win over a `.cast()` dest so a
    /// differently aligned reinterpret cannot change the items address.
    pub(in crate::front::mir) fn array_tail_instantiated_elem_align(
        source_local: Option<usize>,
        source_header: Option<usize>,
        dest: Option<usize>,
    ) -> Option<usize> {
        source_local.or(source_header).or(dest)
    }

    /// `direct_arrayitems` builds a pointer to item zero of the actual
    /// array. Unknown items offset stops translation rather than select
    /// a different address operation.
    pub(in crate::front::mir) fn array_field_items_offset(
        offset: Option<u64>,
        owner: &str,
        field: &str,
    ) -> Result<u64, LowerError> {
        offset.ok_or_else(|| {
            LowerError::Unsupported(format!(
                "direct_arrayitems: items offset of {owner}.{field} is unknown"
            ))
        })
    }

    /// `direct_arrayitems` (`lltype.py`) as `cast_ptr_to_int` + add +
    /// `cast_int_to_ptr` + `cast_instance`. The integer is not recorded
    /// in `cast_ptr_to_int_src`: the round-trip fold would drop `offset`.
    /// Offset 0 is the header itself (`GcArray` items at byte 0).
    pub(in crate::front::mir) fn emit_array_tail_item_ptr(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        offset: u64,
        root: &str,
    ) -> Variable {
        let raw = if offset == 0 {
            base
        } else {
            let addr = push_cast_ptr_to_int(&mut self.graph, bb_id, base);
            let off = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
            let off_n = i64::try_from(offset).expect("array field offset fits in Signed");
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(off.clone()),
                kind: OpKind::ConstInt(off_n),
            });
            let sum = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(sum.clone()),
                kind: OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: addr,
                    rhs: off,
                    result_ty: ValueType::Int,
                },
            });
            let raw = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(raw.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: [
                            "rpython",
                            "rtyper",
                            "lltypesystem",
                            "lltype",
                            "cast_int_to_ptr",
                        ]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![sum]),
                    result_ty: ValueType::Ref(None),
                },
            });
            raw
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: crate::model::cast_instance_call(root, raw),
        });
        res
    }

    /// True when `*mut u8 as *mut i64` (or `*mut f64`) must be `rffi.cast`
    /// onto `CArrayPtr` rather than an alias of the GC header.
    ///
    /// `index_words_ptr` returns that raw address; look-inside of the
    /// alias would feed `SomeInstance` to the caller's `direct_ptradd`
    /// (`ann_direct_ptradd`, `rffi.py` `ptradd`). An already-int raw
    /// scalar (`*const usize as *mut usize`, `HashStateStorage::as_mut_ptr`)
    /// stays in the int bank. An items-base accessor (`IntArray::base`)
    /// and a function that stores through the same spelling
    /// (`set_sizehint_state_value`) keep the header so getarrayitem /
    /// setarrayitem still fire.
    pub(in crate::front::mir) fn should_emit_rffi_cast_identity_to_raw_scalar(
        &self,
        dest_ty: &TyRef,
        src_ty: Option<&TyRef>,
    ) -> bool {
        if graph_is_items_block_base_accessor(&self.graph.name) {
            return false;
        }
        if raw_scalar_address_value_type(dest_ty, self.llbc).is_none() {
            return false;
        }
        if src_ty.is_some_and(|ty| raw_scalar_address_value_type(ty, self.llbc).is_some()) {
            return false;
        }
        if !src_ty.is_some_and(|ty| {
            matches!(
                tyref_to_value_type_with(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs
                ),
                ValueType::Ref(_)
            )
        }) {
            return false;
        }
        self.body
            .locals
            .locals
            .first()
            .is_some_and(|ret| raw_scalar_address_value_type(&ret.ty, self.llbc).is_some())
    }

    /// `rffi.cast(CArrayPtr(T), p)` of a GC / erased byte pointer onto a
    /// raw scalar address (`*mut i64`, `*mut f64`). `__cast_address_intrinsic`
    /// (`p as *mut u8`) answers classdef-less `SomeInstance`; `ann_direct_ptradd`
    /// needs `SomePtr` (`lltype.py` `ann_direct_ptradd`, `rffi.py` `ptradd`).
    /// `cast_ptr_to_int` then `cast_int_to_ptr` is that cast (`lltype.py`).
    pub(in crate::front::mir) fn emit_rffi_cast_to_raw_scalar_ptr(
        &mut self,
        bb_id: BlockId,
        ptr: Variable,
        dest_ty: &TyRef,
    ) -> Variable {
        let item = tyref_peel_one_raw_ptr_node(dest_ty, self.llbc)
            .or_else(|| tyref_peel_one_ref_node(dest_ty, self.llbc))
            .and_then(|pointee| json_ty_byte_size(pointee, self.llbc))
            .map(item_lltype_for_size)
            .unwrap_or(crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed);
        emit_rffi_cast_to_carray_ptr(&mut self.graph, bb_id, ptr, &item)
    }

    /// Pointer reinterprets `*const T::cast_mut` / `*mut T::cast_const`
    /// / `<ptr>::cast` — address-preserving const↔mut flips and pointee
    /// retypes.  The same i64 machine repr as the receiver, so the JIT
    /// models them as identity, the receiver-method twin of the
    /// `CastKind::RawPtr` Rvalue (`same_as`, [`cast_label_from_payload`]).
    /// Gating on a `RawPtr` self excludes unrelated inherent `cast`
    /// methods on non-pointer owners.
    pub(in crate::front::mir) fn is_ptr_identity_cast(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let leaf = fd
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("")
            .to_string();
        if !matches!(leaf.as_str(), "cast" | "cast_mut" | "cast_const") {
            return false;
        }
        fd.signature.inputs.first().is_some_and(|t| {
            tyref_node(t, self.llbc)
                .and_then(|n| strip_ty_wrappers(n, self.llbc))
                .and_then(serde_json::Value::as_object)
                .is_some_and(|o| o.contains_key("RawPtr"))
        })
    }

    /// Destination is `*mut u8` / `*const u8` / `*mut CHAR` and the source
    /// is an integer-kinded raw pointer (`*mut c_void` from `malloc`).
    /// `lltype.py` `ann_cast_int_to_ptr` / `rffi.cast` onto `CCHARP`.
    pub(in crate::front::mir) fn identity_cast_int_raw_ptr_to_ccharp(
        &self,
        dest_ty: &TyRef,
        src_ty: Option<&TyRef>,
    ) -> bool {
        if !tyref_is_raw_byte_ptr(dest_ty, self.llbc) {
            return false;
        }
        let Some(src_ty) = src_ty else {
            return false;
        };
        if !tyref_is_raw_pointer(src_ty, self.llbc) || tyref_is_raw_byte_ptr(src_ty, self.llbc) {
            return false;
        }
        matches!(
            tyref_to_value_type_with(
                src_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            ValueType::Int | ValueType::Unsigned
        )
    }

    /// Recognize `<core::sync::atomic::Atomic*>::load(&self, ordering)`.
    /// This is only method recognition; `lower_call` must check ordering
    /// separately before applying the existing Relaxed fold.
    /// The atomic types are layout-transparent
    /// over their inner scalar/pointer (asserted for the `PyType`
    /// `subclassrange_*` / `instantiate` vtable fields), so the JIT
    /// models the load as that inner value: [`tyref_atomic_inner_value_type`]
    /// types the `Atomic*` field as its inner [`ValueType`], the `&self`
    /// `Ref` rvalue aliases to that field read, and this aliases the load
    /// destination to the receiver.  Gating on an atomic receiver
    /// excludes unrelated inherent `load` methods, and the method name
    /// never reaches the rtyper as a `ptr.getattr`.
    /// `core::mem::replace(&mut place, new)`, `swap`, and `take`.
    ///
    /// RPython has no `mem::replace`. The move is `old = place; place = new`:
    /// `getarrayitem_gc` then `setarrayitem_gc`, or `getfield_gc` then
    /// `setfield_gc`. No recorded place leaves the residual call.
    pub(in crate::front::mir) fn try_lower_mem_exchange(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        arg_locals: &[Option<usize>],
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let Some(name) = regular_call_name_path(reg, self.llbc) else {
            return Ok(false);
        };
        if !(name.starts_with("core::mem::") || name.starts_with("std::mem::")) {
            return Ok(false);
        }
        let leaf = name.rsplit("::").next();
        let arg_address = |this: &Self, index: usize| -> Option<Variable> {
            let local = arg_locals.get(index).copied().flatten()?;
            match this.local_var.get(local)? {
                Some(LocalValue::One(var)) => Some(var.clone()),
                _ => None,
            }
        };
        match leaf {
            Some("replace") if args.len() == 2 => {
                let Some(slot) = self.mem_slot(arg_locals.first().copied().flatten()) else {
                    return Ok(false);
                };
                let address = arg_address(self, 0);
                // `*p = new` where `p` is not an `index_mut` alias is one word
                // at that address: `raw_load` then `raw_store`
                // (`rewrite_op_raw_load` / `rewrite_op_raw_store`). A
                // multi-word pointee is the same move field by field
                // (`rffi.py` `_get_structcopy_fn`): each Charon field offset,
                // not one residual copy. A `__deref_write` call has no bound
                // address.
                if let Some(place) = self.bare_deref_place(&slot) {
                    let old = if let Some(old) =
                        self.exchange_deref_word(mir_bb, &place, args[1].clone())?
                    {
                        old
                    } else if let Some(old) =
                        self.exchange_deref_aggregate(mir_bb, &place, args[1].clone(), address)?
                    {
                        old
                    } else {
                        return Ok(false);
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(old));
                } else {
                    let old = self.read_mem_slot(mir_bb, &slot)?;
                    self.write_mem_slot(mir_bb, slot, args[1].clone())?;
                    self.local_var[dest_local] = Some(LocalValue::One(old));
                }
            }
            Some("swap") if args.len() == 2 => {
                let Some(slot0) = self.mem_slot(arg_locals.first().copied().flatten()) else {
                    return Ok(false);
                };
                let Some(slot1) = self.mem_slot(arg_locals.get(1).copied().flatten()) else {
                    return Ok(false);
                };
                let address0 = arg_address(self, 0);
                let address1 = arg_address(self, 1);
                if let (Some(place0), Some(place1)) =
                    (self.bare_deref_place(&slot0), self.bare_deref_place(&slot1))
                    && self.move_plan(&place0.ty).is_some()
                {
                    let Some(old0) =
                        self.read_moved_aggregate(mir_bb, &place0, address0.clone())?
                    else {
                        return Ok(false);
                    };
                    let Some(old1) =
                        self.read_moved_aggregate(mir_bb, &place1, address1.clone())?
                    else {
                        return Ok(false);
                    };
                    if !self.store_moved_aggregate(mir_bb, &place0, &old1, address0)?
                        || !self.store_moved_aggregate(mir_bb, &place1, &old0, address1)?
                    {
                        return Ok(false);
                    }
                } else {
                    let old0 = self.read_mem_slot(mir_bb, &slot0)?;
                    let old1 = self.read_mem_slot(mir_bb, &slot1)?;
                    self.write_mem_slot(mir_bb, slot0, old1)?;
                    self.write_mem_slot(mir_bb, slot1, old0)?;
                }
                let bb_id = self.emit_bb(mir_bb);
                self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
            }
            Some("take") if args.len() == 1 => {
                if let Some(acc) =
                    self.string_accumulator_of_take_arg(arg_locals.first().copied().flatten())
                {
                    // `std::mem::take(&mut acc)` on a String / Wtf8Buf
                    // accumulator is `old = place; place = empty`
                    // (`rbuilder.py` `StringBuilder.build` then
                    // `StringBuilder()`).  The generic mem-slot path
                    // cannot rebind an SSA string local (`zero_for_mem_slot`
                    // refuses `ValueType::Str`).
                    let taken = if self.builder_mode && self.accumulator_is_builder_mode(acc) {
                        let built = self.resolve_builder_build(mir_bb, acc)?;
                        let bb_id = self.emit_bb(mir_bb);
                        let fresh = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(fresh.clone()),
                            kind: OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec![
                                        crate::runtime_names::shims::STRINGBUILDER_NEW.to_string(),
                                    ],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: Vec::new(),
                                result_ty: ValueType::Ref(None),
                            },
                        });
                        self.local_var[acc] = Some(LocalValue::One(fresh));
                        built
                    } else {
                        let Some(old) = self
                            .local_var
                            .get(acc)
                            .and_then(Option::as_ref)
                            .and_then(|value| value.one().ok())
                        else {
                            return Ok(false);
                        };
                        let bb_id = self.emit_bb(mir_bb);
                        let empty = emit_str_const(&mut self.graph, bb_id, "");
                        self.local_var[acc] = Some(LocalValue::One(empty));
                        old
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(taken));
                } else {
                    let Some(slot) = self.mem_slot(arg_locals.first().copied().flatten()) else {
                        return Ok(false);
                    };
                    let address = arg_address(self, 0);
                    if let Some(place) = self.bare_deref_place(&slot)
                        && self.move_plan(&place.ty).is_some()
                    {
                        // `mem::take` is `replace(dest, <T as Default>::default())`.
                        // Resolve that call before any read: a missing impl
                        // leaves the whole `take` residual and emits nothing.
                        let Some(target) = self.default_call_for_take(reg, &place.ty) else {
                            return Ok(false);
                        };
                        let result_ty = tyref_to_value_type_with(
                            &place.ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        );
                        let bb_id = self.emit_bb(mir_bb);
                        let fresh = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(fresh.clone()),
                            kind: OpKind::Call {
                                target,
                                args: Vec::new(),
                                result_ty,
                            },
                        });
                        let Some(old) =
                            self.exchange_deref_aggregate(mir_bb, &place, fresh, address)?
                        else {
                            return Ok(false);
                        };
                        self.local_var[dest_local] = Some(LocalValue::One(old));
                    } else {
                        let Some(zero) = self.zero_for_mem_slot(mir_bb, &slot) else {
                            return Ok(false);
                        };
                        let old = self.read_mem_slot(mir_bb, &slot)?;
                        self.write_mem_slot(mir_bb, slot, zero)?;
                        self.local_var[dest_local] = Some(LocalValue::One(old));
                    }
                }
            }
            _ => return Ok(false),
        }
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `<T as Default>::default` bound by `take`'s trait obligation.
    ///
    /// The call carries the `TraitImpl` of `core::default::Default` for
    /// `T`. That impl's `default` method is the assoc-fn target
    /// `call_target_segments` would emit for a direct call: no receiver,
    /// so a `FunctionPath` of `[owner, default]`.
    pub(in crate::front::mir) fn default_call_for_take(
        &self,
        reg: &RegularCall,
        ty: &TyRef,
    ) -> Option<CallTarget> {
        let want = self.tyref_adt_def_id(ty)?;
        let refs = reg.generics.get("trait_refs")?.as_array()?;
        for raw in refs {
            let Some(node) = traitref_unwrap(raw, self.llbc, 0) else {
                continue;
            };
            let Some(impl_id) = node
                .get("kind")
                .and_then(|kind| kind.get("TraitImpl"))
                .and_then(|impl_row| impl_row.get("id"))
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let Some(trait_id) = crate::front::clause_spec::trait_decl_pred(node)
                .and_then(|pred| pred.get("id"))
                .and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let Some(trait_path) = self
                .llbc
                .trait_by_id(trait_id)
                .map(|decl| decl.item_meta.name_path())
            else {
                continue;
            };
            if trait_path != "core::default::Default" {
                continue;
            }
            let Some(ti) = self.llbc.trait_impl_by_id(impl_id) else {
                continue;
            };
            let Some(self_ty) = ti
                .get("impl_trait")
                .and_then(|row| row.get("generics"))
                .and_then(|generics| generics.get("types"))
                .and_then(serde_json::Value::as_array)
                .and_then(|types| types.first())
            else {
                continue;
            };
            if self.resolve_tyexpr_to_adt_def_id(self_ty) != Some(want) {
                continue;
            }
            let Some(methods) = ti.get("methods").and_then(serde_json::Value::as_array) else {
                continue;
            };
            for method in methods {
                let Some(fid) = method
                    .get("skip_binder")
                    .and_then(|binder| binder.get("id"))
                    .and_then(serde_json::Value::as_u64)
                else {
                    continue;
                };
                let Some(fd) = self.llbc.fn_by_id(fid) else {
                    continue;
                };
                if fd.item_meta.name_path().rsplit("::").next() != Some("default") {
                    continue;
                }
                return Some(self.assoc_fn_call_target(fd));
            }
        }
        None
    }

    /// Call target of an impl-owned function, matching `call_target_segments`
    /// for a direct call of `fd`.
    pub(in crate::front::mir) fn assoc_fn_call_target(&self, fd: &FunDecl) -> CallTarget {
        let method_hint = self.impl_method_owner(fd);
        let segments = if method_hint.is_none()
            && let Some((_, leaf)) = impl_method_owner_for_fundecl(self.llbc, fd)
        {
            spec_segments(self.llbc, fd, &leaf)
        } else {
            fd.item_meta
                .name_path()
                .split("::")
                .map(str::to_string)
                .collect()
        };
        let mut target = match method_hint {
            Some((owner_root, leaf, _)) => CallTarget::method(leaf, Some(owner_root)),
            None => CallTarget::FunctionPath {
                segments,
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
        };
        target = target.with_fun_decl_id(fd.def_id);
        if matches!(target, CallTarget::Method { .. }) {
            target = target.with_resolved_path(registered_path_for_fun_decl(self.llbc, fd));
        }
        target
    }

    pub(in crate::front::mir) fn mem_slot(&self, local: Option<usize>) -> Option<MemSlot> {
        let local = local?;
        if let Some(place) = self.atomic_ref_place.get(&local) {
            return Some(MemSlot::Place(place.clone()));
        }
        if self.index_elem_alias.contains_key(&local)
            && self.local_var.get(local).and_then(Option::as_ref).is_some()
        {
            return Some(MemSlot::Index(local));
        }
        None
    }

    /// `&mut *borrow` where `borrow` is itself `&mut place` is that place.
    /// A bare `Deref` that is not a field, an array element, or a raw word
    /// declines the graph (`emit_projection_write`).
    pub(in crate::front::mir) fn concrete_borrow_place(&self, mut place: Place) -> Place {
        for _ in 0..8 {
            let PlaceKind::Projection(inner, elem) = &place.kind else {
                break;
            };
            let ProjectionElem::Atom(name) = elem else {
                break;
            };
            if name != "Deref" {
                break;
            }
            let PlaceKind::Local(local) = inner.kind else {
                break;
            };
            let Some(next) = self.atomic_ref_place.get(&(local as usize)).cloned() else {
                break;
            };
            place = next;
        }
        place
    }

    /// `*p` that is not an `index_mut` element and not a reborrow of a field.
    pub(in crate::front::mir) fn bare_deref_place(&self, slot: &MemSlot) -> Option<Place> {
        let MemSlot::Place(place) = slot else {
            return None;
        };
        let place = self.concrete_borrow_place(place.clone());
        let PlaceKind::Projection(inner, elem) = &place.kind else {
            return None;
        };
        let ProjectionElem::Atom(name) = elem else {
            return None;
        };
        if name != "Deref" {
            return None;
        }
        let PlaceKind::Local(local) = inner.kind else {
            return None;
        };
        if self.index_elem_alias.contains_key(&(local as usize)) {
            return None;
        }
        Some(place)
    }

    /// One-word pointee: `raw_load` of the old value, then `raw_store` of
    /// `new`. `None` means the pointee is not one word and the call stays.
    pub(in crate::front::mir) fn exchange_deref_word(
        &mut self,
        mir_bb: usize,
        place: &Place,
        new_value: Variable,
    ) -> Result<Option<Variable>, LowerError> {
        // `*p` of a cell parameter exchanges the cell's `value` field.
        // Raw memory cannot hold the GC reference.
        if let Some(cell) = self.deref_cell_param_local(place)
            && let Some(root) = self.gc_mut_ref_param_root(cell)
        {
            let base = self.defined_local(mir_bb, cell)?;
            let old = self.emit_gc_mut_ref_field_read(mir_bb, base.clone(), &root);
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base,
                    field: FieldDescriptor::new("value", Some(root)),
                    value: LinkArg::Value(new_value),
                    ty: ValueType::Ref(None),
                },
            });
            return Ok(Some(old));
        }
        let PlaceKind::Projection(inner, _) = &place.kind else {
            return Ok(None);
        };
        let Some((item_ty, itemsize, is_item_signed)) = self.raw_word_descr(&place.ty) else {
            return Ok(None);
        };
        let base = self.resolve_place(mir_bb, (**inner).clone())?;
        let bb_id = self.emit_bb(mir_bb);
        let offset = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(offset.clone()),
            kind: OpKind::ConstInt(0),
        });
        let old = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(old.clone()),
            kind: OpKind::RawLoad {
                base: base.clone(),
                offset: offset.clone(),
                item_ty: item_ty.clone(),
                itemsize,
                is_item_signed,
            },
        });
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::RawStore {
                base,
                offset,
                value: new_value,
                item_ty,
                itemsize,
                is_item_signed,
            },
        });
        Ok(Some(old))
    }

    /// Read `place`'s fields into a fresh aggregate, then store `new_value`'s
    /// fields over it. `Ok(None)` leaves the residual call.
    ///
    /// `rffi.py` `_get_structcopy_fn` copies an inline struct one field at a
    /// time. The offsets are Charon's `variant_layouts`, so a 16-byte enum
    /// (`Dynamic`'s `Union`) moves as those fields rather than one word.
    pub(in crate::front::mir) fn exchange_deref_aggregate(
        &mut self,
        mir_bb: usize,
        place: &Place,
        new_value: Variable,
        address: Option<Variable>,
    ) -> Result<Option<Variable>, LowerError> {
        let Some(old) = self.read_moved_aggregate(mir_bb, place, address.clone())? else {
            return Ok(None);
        };
        if !self.store_moved_aggregate(mir_bb, place, &new_value, address)? {
            return Ok(None);
        }
        Ok(Some(old))
    }

    pub(in crate::front::mir) fn read_moved_aggregate(
        &mut self,
        mir_bb: usize,
        place: &Place,
        address: Option<Variable>,
    ) -> Result<Option<Variable>, LowerError> {
        let Some(plan) = self.move_plan(&place.ty) else {
            return Ok(None);
        };
        if !plan.arms.is_empty() {
            return self.copy_enum_switch(mir_bb, place, &plan, None, address);
        }
        let base = self.deref_base(mir_bb, place, address)?;
        let mut parts = Vec::with_capacity(plan.spans.len());
        for span in &plan.spans {
            parts.push(self.emit_span_read(mir_bb, &base, span));
        }
        Ok(Some(self.emit_span_aggregate(mir_bb, &plan, &parts)))
    }

    pub(in crate::front::mir) fn store_moved_aggregate(
        &mut self,
        mir_bb: usize,
        place: &Place,
        value: &Variable,
        address: Option<Variable>,
    ) -> Result<bool, LowerError> {
        let Some(plan) = self.move_plan(&place.ty) else {
            return Ok(false);
        };
        if !plan.arms.is_empty() {
            self.copy_enum_switch(mir_bb, place, &plan, Some(value.clone()), address)?;
            return Ok(true);
        }
        let base = self.deref_base(mir_bb, place, address)?;
        for span in &plan.spans {
            let part = self.emit_span_read(mir_bb, value, span);
            self.emit_span_write(mir_bb, &base, span, part);
        }
        Ok(true)
    }

    /// Move `src` into `slot` when `slot` is the address of an inline
    /// aggregate at offset 0 (`Dynamic.__pos_0`, `HeldCell.0`).
    pub(in crate::front::mir) fn write_aggregate_into(
        &mut self,
        mir_bb: usize,
        slot: &Variable,
        plan: &MovePlan,
        src: &Variable,
    ) -> Result<(), LowerError> {
        if plan.arms.is_empty() {
            for span in &plan.spans {
                let part = self.emit_span_read(mir_bb, src, span);
                self.emit_span_write(mir_bb, slot, span, part);
            }
            return Ok(());
        }
        self.store_enum_variant(mir_bb, slot, src, plan)
    }

    /// Switch on `__discriminant` and copy that variant's fields into
    /// `slot`. One builder for an inline field store and for
    /// `copy_enum_switch`'s store.
    pub(in crate::front::mir) fn store_enum_variant(
        &mut self,
        mir_bb: usize,
        slot: &Variable,
        src: &Variable,
        plan: &MovePlan,
    ) -> Result<(), LowerError> {
        let discr = plan
            .spans
            .first()
            .expect("enum move plan carries __discriminant");
        let head = self.emit_bb(mir_bb);
        let tag = self.emit_span_read(mir_bb, src, discr);
        let join = self.graph.create_block();
        let mut links = Vec::with_capacity(plan.arms.len());
        for arm in &plan.arms {
            let arm_bb = self.graph.create_block();
            self.emit_block.set(mir_bb, arm_bb);
            for span in &arm.spans {
                let part = self.emit_span_read(mir_bb, src, span);
                self.emit_span_write(mir_bb, slot, span, part);
            }
            self.emit_span_write(mir_bb, slot, discr, tag.clone());
            self.graph.set_goto(arm_bb, join, Vec::new());
            links.push(
                Link::from_variables(
                    &self.graph,
                    Vec::new(),
                    arm_bb,
                    Some(ExitCase::Const(ConstValue::Int(arm.discr))),
                )
                .with_prevblock(head)
                .with_llexitcase_from_exitcase(),
            );
        }
        self.graph.block_mut(head).exitswitch = Some(ExitSwitch::Value(tag));
        self.graph.closeblock(head, links);
        self.emit_block.set(mir_bb, join);
        Ok(())
    }

    pub(in crate::front::mir) fn struct_field_offset_from_payload(
        &self,
        base_ty: &TyRef,
        payload: &serde_json::Value,
    ) -> Option<u64> {
        let (variant_idx, field_idx) = indexed_field_parts(payload)?;
        if variant_idx.is_some() {
            return None;
        }
        let type_id = type_decl_ref_adt_id(self.field_base_adt(base_ty)?)?;
        let td = self.llbc.type_by_id(type_id)?;
        let target = std::env::var("TARGET").unwrap_or_default();
        td.layout_for_target(self.llbc, &target)?
            .struct_field_offset(field_idx)
    }

    /// Whole-enum move: switch on `__discriminant`, then copy that
    /// variant's fields. `new_value` stores into `place`; `None` reads
    /// `place` and returns the join phi.
    pub(in crate::front::mir) fn copy_enum_switch(
        &mut self,
        mir_bb: usize,
        place: &Place,
        plan: &MovePlan,
        new_value: Option<Variable>,
        address: Option<Variable>,
    ) -> Result<Option<Variable>, LowerError> {
        if new_value.is_some() {
            let slot = self.deref_base(mir_bb, place, address)?;
            let src = new_value.expect("enum store has a source");
            self.store_enum_variant(mir_bb, &slot, &src, plan)?;
            return Ok(None);
        }
        let discr = plan
            .spans
            .first()
            .expect("enum move plan carries __discriminant");
        let head = self.emit_bb(mir_bb);
        let slot = self.deref_base(mir_bb, place, address)?;
        let tag = self.emit_span_read(mir_bb, &slot, discr);
        let (join, vars) = self.graph.create_block_with_arg_vars(1);
        let phi = vars.into_iter().next().expect("enum move phi");
        let mut links = Vec::with_capacity(plan.arms.len());
        for arm in &plan.arms {
            let arm_bb = self.graph.create_block();
            self.emit_block.set(mir_bb, arm_bb);
            let mut parts = Vec::with_capacity(1 + arm.spans.len());
            parts.push(tag.clone());
            for span in &arm.spans {
                parts.push(self.emit_span_read(mir_bb, &slot, span));
            }
            let mut spans = Vec::with_capacity(parts.len());
            spans.push(discr.clone());
            spans.extend(arm.spans.iter().cloned());
            let agg_plan = MovePlan {
                ctor_id: plan.ctor_id,
                spans,
                arms: Vec::new(),
            };
            let agg = self.emit_span_aggregate(mir_bb, &agg_plan, &parts);
            self.graph.set_goto(arm_bb, join, vec![agg]);
            links.push(
                Link::from_variables(
                    &self.graph,
                    Vec::new(),
                    arm_bb,
                    Some(ExitCase::Const(ConstValue::Int(arm.discr))),
                )
                .with_prevblock(head)
                .with_llexitcase_from_exitcase(),
            );
        }
        self.graph.block_mut(head).exitswitch = Some(ExitSwitch::Value(tag));
        self.graph.closeblock(head, links);
        self.emit_block.set(mir_bb, join);
        Ok(Some(phi))
    }

    pub(in crate::front::mir) fn deref_base(
        &mut self,
        mir_bb: usize,
        place: &Place,
        address: Option<Variable>,
    ) -> Result<Variable, LowerError> {
        let PlaceKind::Projection(inner, _) = &place.kind else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: aggregate exchange place is not a deref"
            )));
        };
        // `&*p` aliases `p`. The `&mut` argument's own variable is that
        // address; the inner local is not live in the replace block.
        if let PlaceKind::Local(i) = &inner.kind
            && self
                .local_var
                .get(*i as usize)
                .and_then(|var| var.as_ref())
                .is_none()
            && let Some(address) = address
        {
            return Ok(address);
        }
        self.resolve_place(mir_bb, (**inner).clone())
    }

    /// Charon fields of a multi-word inline value. A transparent newtype
    /// (`Dynamic` over `Union`) contributes the inner type's fields. A
    /// struct field and an enum field are both `getfield`/`setfield`.
    /// Overlapping enum fields keep the wider span. A wide JIT tag must
    /// not swallow the payload that sits in the next bytes.
    pub(in crate::front::mir) fn move_plan(&self, ty: &TyRef) -> Option<MovePlan> {
        let id = self.tyref_adt_def_id(ty)?;
        let td = self.llbc.type_by_id(id)?;
        if td.published_struct_copy {
            return None;
        }
        let target = std::env::var("TARGET").unwrap_or_default();
        let layout = td.layout_for_target(self.llbc, &target)?;
        let size = layout.size?;
        if size <= 8 {
            return None;
        }
        match &td.kind {
            TypeDeclKind::Struct(fields) => {
                if td.is_repr_transparent() && fields.len() == 1 {
                    return self.move_plan(&fields[0].ty);
                }
                let name_path = td.item_meta.name_path();
                let owner = class_identity_spelling(&name_path, self.tombstoned_leaves);
                let owner_id =
                    majit_ir::descr::StructId::from_canonical(&strip_crate_prefix(&name_path));
                let mut spans = Vec::with_capacity(fields.len());
                for (i, field) in fields.iter().enumerate() {
                    let offset = layout.struct_field_offset(i)?;
                    // `rffi.py` `_get_structcopy_fn.copyfn` copies a
                    // ContainerType field by recursing into its leaves.
                    // A nested container has no recursive plan yet, so
                    // the whole type is declined. Proven scalars stay
                    // one span (`struct_field_span_raw`).
                    let (_, itemsize, _) = self.struct_field_span_raw(&field.ty)?;
                    if itemsize == 0 || itemsize > 8 {
                        return None;
                    }
                    let name = field
                        .name
                        .clone()
                        .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(i));
                    let value_ty = tyref_to_value_type_with(
                        &field.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    spans.push(MoveSpan {
                        offset,
                        bytes: itemsize,
                        kind: SpanKind::Field {
                            name,
                            owner: owner.clone(),
                            owner_id,
                            ty: value_ty,
                        },
                    });
                }
                if spans.is_empty() {
                    return None;
                }
                Some(MovePlan {
                    ctor_id: id,
                    spans,
                    arms: Vec::new(),
                })
            }
            TypeDeclKind::Enum(variants) => {
                let name_path = td.item_meta.name_path();
                let leaf = class_identity_spelling(&name_path, self.tombstoned_leaves);
                let canon = strip_crate_prefix(&name_path);
                let base_id = majit_ir::descr::StructId::from_canonical(&canon);
                // A union payload has no single kind (`rclass.py` never
                // bit-copies a pointer into an int). The move is a switch
                // on `__discriminant` and a typed copy of the live variant.
                let (Some(disc_offset), Some(int_ty)) =
                    (layout.discriminant_offset(), layout.discriminant_int_type())
                else {
                    return None;
                };
                let itemsize = int_type_byte_width(int_ty) as usize;
                if itemsize == 0 || itemsize > 8 {
                    return None;
                }
                let signed = int_ty.starts_with('i');
                let item_ty = if signed {
                    ValueType::Int
                } else {
                    ValueType::Unsigned
                };
                let discr = MoveSpan {
                    offset: disc_offset,
                    bytes: itemsize,
                    kind: SpanKind::Field {
                        name: "__discriminant".to_string(),
                        owner: leaf.clone(),
                        owner_id: base_id,
                        ty: item_ty,
                    },
                };
                let mut arms = Vec::with_capacity(variants.len());
                for (vidx, variant) in variants.iter().enumerate() {
                    let Some(discr_value) = variant.discriminant_i64() else {
                        return None;
                    };
                    let variant_owner = format!("{leaf}::{}", variant.name);
                    let variant_id = majit_ir::descr::StructId::from_canonical(&format!(
                        "{canon}::{}",
                        variant.name
                    ));
                    let mut candidates: Vec<MoveSpan> = Vec::new();
                    for (i, field) in variant.fields.iter().enumerate() {
                        let Some(offset) = layout.field_offset(vidx, i) else {
                            return None;
                        };
                        let Some((_, itemsize, _)) = self.span_raw_for_ty(&field.ty) else {
                            return None;
                        };
                        if itemsize == 0 {
                            continue;
                        }
                        if itemsize > 8 {
                            return None;
                        }
                        let name = field.name.clone().unwrap_or_else(|| {
                            majit_charon_reader::ullbc::positional_field_name(i)
                        });
                        let value_ty = tyref_to_value_type_with(
                            &field.ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        );
                        candidates.push(MoveSpan {
                            offset,
                            bytes: itemsize,
                            kind: SpanKind::Field {
                                name,
                                owner: variant_owner.clone(),
                                owner_id: variant_id,
                                ty: value_ty,
                            },
                        });
                    }
                    candidates.sort_by(|left, right| {
                        left.offset
                            .cmp(&right.offset)
                            .then(right.bytes.cmp(&left.bytes))
                    });
                    let mut arm_spans: Vec<MoveSpan> = Vec::new();
                    for candidate in candidates {
                        let start = candidate.offset;
                        let end = start + candidate.bytes as u64;
                        let overlaps = arm_spans.iter().any(|kept| {
                            let kept_end = kept.offset + kept.bytes as u64;
                            start < kept_end && kept.offset < end
                        });
                        if !overlaps {
                            arm_spans.push(candidate);
                        }
                    }
                    arms.push(EnumMoveArm {
                        discr: discr_value,
                        spans: arm_spans,
                    });
                }
                if arms.is_empty() {
                    return None;
                }
                Some(MovePlan {
                    ctor_id: id,
                    spans: vec![discr],
                    arms,
                })
            }
            _ => None,
        }
    }

    pub(in crate::front::mir) fn emit_span_read(
        &mut self,
        mir_bb: usize,
        base: &Variable,
        span: &MoveSpan,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let SpanKind::Field {
            name,
            owner,
            owner_id,
            ty,
        } = &span.kind;
        let kind = OpKind::FieldRead {
            base: base.clone(),
            field: crate::model::FieldDescriptor::new(name.clone(), Some(owner.clone()))
                .with_owner_id(Some(*owner_id)),
            ty: ty.clone(),
            pure: false,
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind,
        });
        result
    }

    pub(in crate::front::mir) fn emit_span_write(
        &mut self,
        mir_bb: usize,
        base: &Variable,
        span: &MoveSpan,
        value: Variable,
    ) {
        let bb_id = self.emit_bb(mir_bb);
        let SpanKind::Field {
            name,
            owner,
            owner_id,
            ty,
        } = &span.kind;
        let kind = OpKind::FieldWrite {
            base: base.clone(),
            field: crate::model::FieldDescriptor::new(name.clone(), Some(owner.clone()))
                .with_owner_id(Some(*owner_id)),
            value: crate::model::LinkArg::Value(value),
            ty: ty.clone(),
        };
        self.graph
            .block_mut(bb_id)
            .operations
            .push(SpaceOperation { result: None, kind });
    }

    /// Copy `src` into `dest` one inner leaf at a time.
    ///
    /// `src` is the inlined struct value; `dest` is the outer object.
    /// Leaves use the inner struct's own descriptors, matching a nested
    /// `FieldRead` of `frame.header.leaf` whose base is the outer object.
    pub(in crate::front::mir) fn emit_inlined_struct_leaf_copy(
        &mut self,
        mir_bb: usize,
        dest: &Variable,
        src: &Variable,
        nested: &TypeDecl,
    ) {
        let leaves = collect_by_value_struct_leaves(
            nested,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            16,
            &self.raw_alloc_flavor_structs,
        );
        let bb_id = self.emit_bb(mir_bb);
        for (leaf, leaf_ty) in leaves {
            let read = self.graph.alloc_value_var();
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(read.clone()),
                kind: OpKind::FieldRead {
                    base: src.clone(),
                    field: leaf.clone(),
                    ty: leaf_ty.clone(),
                    pure: false,
                },
            });
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: dest.clone(),
                    field: leaf,
                    value: LinkArg::Value(read),
                    ty: leaf_ty,
                },
            });
        }
    }

    /// Fresh aggregate holding `parts`, in `plan.spans` order. The
    /// constructor is the same empty `malloc` marker an `Rvalue::Aggregate`
    /// emits; the parts are the field stores.
    pub(in crate::front::mir) fn emit_span_aggregate(
        &mut self,
        mir_bb: usize,
        plan: &MovePlan,
        parts: &[Variable],
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let td = self
            .llbc
            .type_by_id(plan.ctor_id)
            .expect("move plan type is in the LLBC");
        let name_path = td.item_meta.name_path();
        let mut segments: Vec<String> = name_path.split("::").map(str::to_string).collect();
        let leaf = segments.pop().unwrap_or_default();
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let owner = if segments.is_empty() {
            leaf.clone()
        } else {
            format!("{}::{leaf}", segments.join("::"))
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(segments, leaf),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner)),
            },
        });
        for (span, part) in plan.spans.iter().zip(parts.iter()) {
            self.emit_span_write(mir_bb, &result, span, part.clone());
        }
        result
    }

    /// Proven scalar width: a `json_ty_raw_store_descr` word, a thin
    /// pointer, or `core::sync::atomic::Atomic<T>` classified by `T`
    /// (`tyref_atomic_arg`). Nested ADTs are left to the caller.
    pub(in crate::front::mir) fn span_raw_scalar_ptr_or_atomic(
        &self,
        ty: &TyRef,
    ) -> Option<(ValueType, usize, bool)> {
        if let Some(node) = tyref_node(ty, self.llbc)
            && let Some(descr) = json_ty_raw_store_descr(node, self.llbc)
        {
            return Some(descr);
        }
        if tyref_is_copy_scalar_or_thin_ptr(ty, self.llbc) {
            return Some((
                ValueType::Ref(None),
                crate::layout::target_word_size(),
                false,
            ));
        }
        if let Some(inner) = tyref_atomic_arg(ty, self.llbc)
            && let Some(descr) = json_ty_raw_store_descr(inner, self.llbc)
        {
            return Some(descr);
        }
        None
    }

    /// Byte width of one struct field for a whole-value move
    /// (`rffi.py` `_get_structcopy_fn.copyfn`). A nested container
    /// (struct, tuple, data-carrying enum, union) is declined until
    /// `move_plan` can recurse into its leaves the way `copyfn` does.
    /// A fieldless (C-like) enum is its scalar tag.
    pub(in crate::front::mir) fn struct_field_span_raw(
        &self,
        ty: &TyRef,
    ) -> Option<(ValueType, usize, bool)> {
        self.span_raw_scalar_ptr_or_atomic(ty)
            .or_else(|| fieldless_enum_raw_store_descr(ty, self.llbc))
    }

    /// Byte width of one field for a whole-value move. A literal uses
    /// its Charon size. A thin pointer or a small ADT (a `Box`, a C-like
    /// tag) is that many bytes, so a 16-byte `Union` still moves field by
    /// field when a payload is not an integer.
    ///
    /// `core::sync::atomic::Atomic<T>` is Opaque and carries no Charon
    /// layout; the stored word is `T` (`tyref_atomic_arg`), classified
    /// with the same `json_ty_raw_store_descr` the Scalar arm uses
    /// (`tyref_atomic_inner_value_type` paints the FieldWrite that way).
    pub(in crate::front::mir) fn span_raw_for_ty(
        &self,
        ty: &TyRef,
    ) -> Option<(ValueType, usize, bool)> {
        if let Some(descr) = self.span_raw_scalar_ptr_or_atomic(ty) {
            return Some(descr);
        }
        let id = self.tyref_adt_def_id(ty)?;
        let td = self.llbc.type_by_id(id)?;
        let target = std::env::var("TARGET").unwrap_or_default();
        let size = td.layout_for_target(self.llbc, &target)?.size?;
        if size == 0 || size > 8 {
            return None;
        }
        Some((ValueType::Unsigned, size as usize, false))
    }

    /// `majit_ir::llop::gc_load_indexed::<T, _>(base, index, scale, base_ofs)`
    /// is `llop.gc_load_indexed(T, base, index, scale, base_ofs)`. The item
    /// is described from the destination type `T`; `scale` and `base_ofs`
    /// must be literals (`rewrite_op_gc_load_indexed` raises
    /// `NotImplementedError` otherwise), and `T` is not a GC reference.
    pub(in crate::front::mir) fn lower_llop_gc_load_indexed(
        &mut self,
        mir_bb: usize,
        bb_id: crate::model::BlockId,
        args: &[Variable],
        arg_operands: &[Operand],
        dest_ty: &TyRef,
    ) -> Result<Variable, LowerError> {
        let ([base, index, ..], [_, _, Operand::Const(scale), Operand::Const(base_ofs)]) =
            (args, arg_operands)
        else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: gc_load_indexed: 'scale' and 'base_ofs' should be constants"
            )));
        };
        let (Some(scale), Some(base_ofs)) = (
            const_operand_index(self.llbc, scale),
            const_operand_index(self.llbc, base_ofs),
        ) else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: gc_load_indexed: 'scale' and 'base_ofs' should be constants"
            )));
        };
        let Some((item_ty, itemsize, is_item_signed)) = self.raw_word_descr(dest_ty) else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: gc_load_indexed: no item descr for the result type"
            )));
        };
        let result_ct = match item_ty {
            ValueType::Int | ValueType::Unsigned => crate::model::ConcreteType::Signed,
            ValueType::Float => crate::model::ConcreteType::Float,
            _ => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: gc_load_indexed of a GC reference"
                )));
            }
        };
        let loaded = self.graph.alloc_value_var_with_type(result_ct);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(loaded.clone()),
            kind: OpKind::GcLoadIndexed {
                base: base.clone(),
                index: index.clone(),
                scale: scale as i64,
                base_ofs: base_ofs as i64,
                item_ty,
                itemsize,
                is_item_signed,
            },
        });
        Ok(loaded)
    }

    pub(in crate::front::mir) fn raw_word_descr(
        &self,
        ty: &TyRef,
    ) -> Option<(ValueType, usize, bool)> {
        if !tyref_is_copy_scalar_or_thin_ptr(ty, self.llbc) {
            return None;
        }
        json_ty_raw_store_descr(tyref_node(ty, self.llbc)?, self.llbc)
    }

    pub(in crate::front::mir) fn read_mem_slot(
        &mut self,
        mir_bb: usize,
        slot: &MemSlot,
    ) -> Result<Variable, LowerError> {
        match slot {
            MemSlot::Place(place) => {
                self.resolve_place(mir_bb, self.concrete_borrow_place(place.clone()))
            }
            MemSlot::Index(local) => self
                .local_var
                .get(*local)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: indexed mem slot lost its element"
                    ))
                })?
                .one(),
        }
    }

    pub(in crate::front::mir) fn write_mem_slot(
        &mut self,
        mir_bb: usize,
        slot: MemSlot,
        value: Variable,
    ) -> Result<(), LowerError> {
        match slot {
            MemSlot::Place(place) => {
                let place = self.concrete_borrow_place(place);
                match place.kind {
                    PlaceKind::Local(i) => {
                        self.local_var[i as usize] = Some(LocalValue::One(value));
                        Ok(())
                    }
                    PlaceKind::Projection(inner, elem) => self.emit_projection_write(
                        mir_bb,
                        *inner,
                        elem,
                        LinkArg::Value(value),
                        &place.ty,
                    ),
                    _ => Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: mem place is not a local or a projection"
                    ))),
                }
            }
            MemSlot::Index(local) => {
                let place = Place {
                    kind: PlaceKind::Local(local as u64),
                    ty: unit_tyref(),
                };
                self.emit_projection_write(
                    mir_bb,
                    place,
                    ProjectionElem::Atom("Deref".to_string()),
                    LinkArg::Value(value),
                    &unit_tyref(),
                )
            }
        }
    }

    /// `Default` for `mem::take` when the slot is a scalar zero.
    /// A non-scalar stays a residual call: its `Default` is not a literal.
    pub(in crate::front::mir) fn zero_for_mem_slot(
        &mut self,
        mir_bb: usize,
        slot: &MemSlot,
    ) -> Option<Variable> {
        let vt = match slot {
            MemSlot::Place(place) => tyref_to_value_type_with(
                &place.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            MemSlot::Index(local) => self.index_elem_alias.get(local)?.item_ty.clone(),
        };
        let bb_id = self.emit_bb(mir_bb);
        let kind = match vt {
            ValueType::Int => OpKind::ConstInt(0),
            ValueType::Unsigned => OpKind::ConstUInt(0),
            ValueType::Bool => OpKind::ConstBool(false),
            ValueType::Float => OpKind::ConstFloat(0.0f64.to_bits()),
            ValueType::Void => return Some(self.emit_unit(bb_id)),
            _ => return None,
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind,
        });
        Some(res)
    }

    pub(in crate::front::mir) fn is_atomic_load(&self, reg: &RegularCall) -> bool {
        self.is_atomic_method(reg, "load")
    }

    /// The receiver variable is a `FieldRead`, or a fresh block input
    /// copied from one, of quasi-immutable `w_globals`.
    ///
    /// `w_code_get_w_globals` computes `&raw (*code).w_globals` in one
    /// block and calls `AtomicPtr::load` in the successor. `FrameState.copy`
    /// gives that successor a fresh Variable, so the load's receiver is not
    /// the `FieldRead` result. [`Self::input_copied_from`] is that copy.
    ///
    /// The marker is the table harvested once for this artefact
    /// (`self.immutable_fields`). A missing owner, a missing marker, a
    /// field other than `w_globals`, a merge phi, or a rank that is not
    /// quasi fails closed and the load keeps the unsupported-ordering error.
    pub(in crate::front::mir) fn atomic_load_receiver_is_quasi_w_globals(
        &self,
        recv: &Variable,
    ) -> bool {
        let mut var_id = recv.id();
        let mut seen = std::collections::HashSet::new();
        let (owner, name) = loop {
            if !seen.insert(var_id) || seen.len() > 64 {
                return false;
            }
            let found = self.graph.blocks.iter().find_map(|block| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::FieldRead { field, .. }
                        if op
                            .result
                            .as_ref()
                            .is_some_and(|result| result.id() == var_id) =>
                    {
                        Some((field.owner_root.clone(), field.name.clone()))
                    }
                    _ => None,
                })
            });
            if let Some(found) = found {
                break found;
            }
            match self.input_copied_from.get(&var_id).copied() {
                Some(next) => var_id = next,
                None => return false,
            }
        };
        let Some(owner) = owner else {
            return false;
        };
        if !majit_ir::descr::is_w_globals_field_name(&name) {
            return false;
        }
        self.immutable_fields.get(&owner).is_some_and(|fields| {
            fields
                .iter()
                .any(|(field_name, rank)| field_name == &name && rank.is_quasi_immutable())
        })
    }

    /// `<core::sync::atomic::Atomic*>::store(&self, value, ordering)` — the
    /// write twin of [`Lowering::is_atomic_load`].  A relaxed store to a
    /// layout-transparent atomic is the same machine store a plain field
    /// write emits, so `lower_call` lowers it to the same `FieldWrite`.
    pub(in crate::front::mir) fn is_atomic_store(&self, reg: &RegularCall) -> bool {
        self.is_atomic_method(reg, "store")
    }

    /// The `core::sync::atomic::Ordering` variant name an `Rvalue::Aggregate`
    /// builds, or `None` when the rvalue is not one.
    ///
    /// Keyed on the enum's own path and read back by variant NAME rather than
    /// index, so a reordering of `Ordering`'s variants cannot silently turn a
    /// `Release` into a `Relaxed`.
    pub(in crate::front::mir) fn atomic_ordering_variant(&self, rvalue: &Rvalue) -> Option<String> {
        let Rvalue::Aggregate(kind, _) = rvalue else {
            return None;
        };
        // Same head shapes `resolve_aggregate_adt` decodes: a bare `type_id`
        // or a full `TypeDeclRef` object.
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => type_decl_ref_adt_id(head.as_object()?)?,
        };
        let td = self.llbc.type_by_id(type_id)?;
        if td.item_meta.name_path() != "core::sync::atomic::Ordering" {
            return None;
        }
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return None;
        };
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64)? as usize;
        Some(variants.get(variant_idx)?.name.clone())
    }

    /// `<core::sync::atomic::Atomic*>::<method>(&self, ..)` — an inherent
    /// method of a std atomic, keyed on the receiver's type so a user type
    /// with a `load` / `store` method of its own does not match.
    pub(in crate::front::mir) fn is_atomic_method(&self, reg: &RegularCall, method: &str) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let name = fd.item_meta.name_path();
        if name.rsplit("::").next() != Some(method) {
            return false;
        }
        fd.signature.inputs.first().is_some_and(|t| {
            adt_path_of_tyref(t, self.llbc).is_some_and(|p| {
                p.contains("::sync::atomic::")
                    && p.rsplit("::")
                        .next()
                        .is_some_and(|leaf| leaf.starts_with("Atomic"))
            })
        })
    }

    /// `core::sync::atomic::Atomic*::from_ptr(p)` is a layout-transparent
    /// pointer view.  Match the exact core associated function and require an
    /// atomic output wrapper so unrelated `from_ptr` constructors (notably
    /// `CStr::from_ptr`) keep their own semantics.
    pub(in crate::front::mir) fn is_atomic_from_ptr_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "core::sync::atomic::<Impl>::from_ptr"
                && tyref_atomic_inner_value_type(
                    &fd.signature.output,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
                .is_some()
        })
    }

    /// `std::sync::OnceLock::get_or_init` whose receiver is a
    /// [`OpKind::ConstRefAddr`] taken from a `refs` row. Those rows register
    /// the initialized object, not the `OnceLock` container, so the only
    /// receivers that qualify are statics the host already initialized.
    pub(in crate::front::mir) fn is_prebuilt_once_lock_get_or_init(
        &self,
        reg: &RegularCall,
        recv: &Variable,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let is_get_or_init = self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let name = fd.item_meta.name_path();
            name.contains("::once_lock::") && name.rsplit("::").next() == Some("get_or_init")
        });
        is_get_or_init
            && self
                .graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| {
                    op.result.as_ref() == Some(recv)
                        && matches!(op.kind, OpKind::ConstRefAddr(addr)
                            if self.static_addrs.refs.iter().any(|(_, a)| *a == addr))
                })
    }

    /// `core::sync::atomic::Atomic*::new(v)` — the wrapper constructor over
    /// the inner scalar the field translates as.  Keyed on the exact core
    /// associated function and an atomic output wrapper, like
    /// [`Self::is_atomic_from_ptr_identity`].
    pub(in crate::front::mir) fn is_atomic_new_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "core::sync::atomic::<Impl>::new"
                && tyref_atomic_inner_value_type(
                    &fd.signature.output,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
                .is_some()
        })
    }

    /// `<*mut T>::add` / `<*const T>::add` inside an `ItemsBlock` or
    /// `TypedItemsBlock` items-base accessor (`items_block_items_base` /
    /// `items_block_items_ptr` / `typed_items_block_items_base`), whose
    /// whole body is `(block as *mut u8).add(*_ITEMS_OFFSET)` — the items
    /// pointer of a GC array backing a list or rbigint digit array.
    ///
    /// The runtime reads list items through a `gcarray` descr whose
    /// `base_size` already folds in the header offset
    /// (`pyobject_gcarray_descr` base_size = `ITEMS_BLOCK_ITEMS_OFFSET`),
    /// so the JIT's items base *is* the header pointer.  The accessor
    /// body therefore collapses to its receiver, dropping the
    /// `core::ptr::mut_ptr::<Impl>::add` call (no graph lowering, not a
    /// registered callee).
    ///
    /// The soundness boundary is the *enclosing accessor*, not the
    /// `.add` callee or its constant offset: a `.add(NAMED_OFFSET)`
    /// whose interior pointer is dereferenced in place — `runtime_ops`
    /// reading `*(p.add(PYFRAME_W_BUILTIN_OFFSET))`, or
    /// `w_tuple_getitem_known`'s `*base.add(idx)` — must keep its offset.
    /// Only an accessor that *returns* the interior pointer for
    /// descr-based consumption is safe to alias to its receiver.
    pub(in crate::front::mir) fn is_items_block_base_ptr_add(&self, reg: &RegularCall) -> bool {
        graph_is_items_block_base_accessor(&self.graph.name)
            && (regular_call_is_ptr_add(reg, self.llbc)
                || (is_typed_array_base_adapter(&self.graph.name)
                    && regular_call_is_ptr_wrapping_add(reg, self.llbc)))
    }

    pub(in crate::front::mir) fn is_container_items_view_from_raw_parts(
        &self,
        reg: &RegularCall,
    ) -> bool {
        if !(self.graph.name.ends_with("rbigint::<Impl>::digits")
            || self.graph.name.ends_with("rbigint::<Impl>::digits_mut"))
        {
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut"
            )
        })
    }

    /// Return the owning `BytesBlock` for the exact
    /// `bytes_block_chars(block)` raw-slice constructor.
    ///
    /// RPython's `STR` is a length-prefixed GC struct with a variable-sized
    /// `chars` array (`lltypesystem/rstr.py`). pyre's `BytesBlock` is that
    /// `STR` (`type BytesBlock = Utf8Str`): the lifted value is the header,
    /// not the interior `chars` address. Matching the defining FieldRead
    /// also proves this is that layout adapter rather than an unrelated
    /// raw slice in the helper.
    pub(in crate::front::mir) fn bytes_block_header_from_raw_parts(
        &self,
        bb_id: BlockId,
        reg: &RegularCall,
        chars: &Variable,
    ) -> Option<Variable> {
        if !self.graph_is_bytes_block_chars_or_string_view() {
            return None;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        if self.llbc.fn_by_id(*id)?.item_meta.name_path() != "core::slice::raw::from_raw_parts" {
            return None;
        }
        self.graph
            .block(bb_id)
            .operations
            .iter()
            .rev()
            .find_map(|op| match (&op.result, &op.kind) {
                (Some(result), OpKind::FieldRead { base, field, .. })
                    if result == chars
                        && field.name == "chars"
                        && field.owner_root.as_deref() == Some("bytesobject::BytesBlock")
                        && field.taken_by_address =>
                {
                    Some(base.clone())
                }
                _ => None,
            })
    }

    /// Whether this is the sole Rust-only forwarding query inside
    /// `pyre_object::gc_hook::try_gc_current_object_address`.
    ///
    /// RPython never emits such a source-level helper: its GC transformer
    /// reloads every live GC variable from the shadow stack after a safepoint.
    /// Keep the match graph-scoped so runtime consumers that deliberately
    /// normalize a stored raw address are not rewritten as identities.
    pub(in crate::front::mir) fn is_gc_current_object_address_adapter(
        &self,
        reg: &RegularCall,
    ) -> bool {
        if !self.is_gc_current_object_address_adapter_graph() {
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "majit_gc::gc_current_object_address")
    }

    pub(in crate::front::mir) fn is_gc_current_object_address_adapter_graph(&self) -> bool {
        self.graph.name == "pyre_object::gc_hook::try_gc_current_object_address"
    }

    /// `*items_block_items_base(block).add(idx)` — a list / tuple
    /// element load (`x = *p`) or store (`*p = v`) through an
    /// `ItemsBlock` items pointer.  brick 1 rewrites the accessor to
    /// return the block *header*, so the runtime reaches `items[idx]`
    /// through a `gcarray` descr whose `base_size` folds in
    /// `ITEMS_BLOCK_ITEMS_OFFSET`; this lowers the dereferenced `.add`
    /// as that getarrayitem / setarrayitem — the same decomposition as
    /// the workspace [`Lowering::is_workspace_index_call`] path, but
    /// reached through a raw `.add` rather than `Index::index`.
    ///
    /// Five conditions, cheapest first (so the body scans run only on
    /// genuine candidates): the callee is `<ptr>::add`; two arguments;
    /// the receiver points at a managed reference
    /// ([`is_object_ref_items_ptr`]); the index is the second argument
    /// (a local or a constant element count — `*base.add(1)` is
    /// `l.items[1]`); the base traces to an items-base accessor
    /// ([`base_traces_to_items_block_accessor`]) — so the header
    /// `base_size` re-add lands on `items[0]`; and the `.add` result is
    /// dereferenced exactly once and never escapes as a raw pointer
    /// ([`add_dest_used_only_as_single_deref`]). Brick 1's byte-offset
    /// add lives in the accessor graph and its dest escapes, so that
    /// gate keeps it out.
    pub(in crate::front::mir) fn is_list_items_elem_ptr_add(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> bool {
        is_list_items_elem_ptr_add_parts(
            reg,
            args_len,
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            dest_local,
            self.body,
            self.llbc,
        )
    }

    /// PyPy's unwrapped byte / ASCII list storage.  Rust reaches the shared
    /// `ItemsBlock` through `BytesArray::base` / `UnicodeArray::base`, but the
    /// source-level element remains `str`: `AbstractUnwrappedStrategy`
    /// appends the result of `bytes_w` / `utf8_w`, and
    /// `externalvsinternal(..., gcref=True)` alone changes its physical
    /// storage to GCREF.  Return the logical `[str]` ARRAY identity so the
    /// ordinary ListRepr performs that recast.
    pub(in crate::front::mir) fn string_items_elem_ptr_add(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> Option<(ValueType, String)> {
        is_string_items_elem_ptr_add_parts(
            reg,
            args_len,
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            dest_local,
            self.body,
            self.llbc,
        )
        .then(|| (ValueType::Str, STRING_GCREF_GCARRAY_TYPE_ID.to_string()))
    }

    /// Scalar `GcArray(Signed|Float)` counterpart of
    /// [`Self::is_list_items_elem_ptr_add`]. Returns the RPython item bank and
    /// ARRAY identity when the pointer traces to
    /// `typed_items_block_items_base`, names an `i64`/`f64` element, and is
    /// consumed by one immediate dereference.
    pub(in crate::front::mir) fn typed_items_elem_ptr_add(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> Option<(ValueType, String)> {
        if !is_typed_items_elem_ptr_add_parts(
            reg,
            args_len,
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            dest_local,
            self.body,
            self.llbc,
        ) {
            return None;
        }
        raw_ptr_typed_items_element(first_arg_ty?, self.llbc)
    }

    /// `*(&[u8]).as_ptr().add(i)` — `ll_stritem_nonneg` on a shared
    /// `&[u8]`, `getarrayitem` on `&mut [u8]`.
    pub(in crate::front::mir) fn u8_slice_elem_ptr_add(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> Option<(ValueType, String)> {
        if !is_u8_slice_elem_ptr_add_parts(
            reg,
            args_len,
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            arg_locals.get(1).copied().flatten(),
            dest_local,
            self.body,
            self.llbc,
        ) {
            return None;
        }
        raw_ptr_u8_slice_element(first_arg_ty?, self.llbc)
    }

    /// `<*const T>::cast_mut` / `<*mut T>::cast_const` — pointer casts that
    /// change only const/mut, never the pointee type.  The JIT does not
    /// model the mut/const distinction (`Ref` / `RawPtr` lower to a
    /// same-Variable alias), so a `p.cast_mut()` callsite binds
    /// its destination straight to the pointer argument instead of
    /// emitting a call to core's raw-pointer method (which has no graph
    /// lowering and is not a registered callee).
    ///
    /// The pointee-changing `<ptr>::cast::<U>()` is NOT matched here: it
    /// routes to [`is_ptr_identity_cast`], which narrows a
    /// `cast::<RegisteredStruct>()` to `SomeInstance(root)` (the same
    /// `cast_pointer` shape as `obj as *const W_Foo`) instead of leaving
    /// the destination classdef-less and blocking the downstream getattr.
    pub(in crate::front::mir) fn is_noop_ptr_cast(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::ptr::const_ptr::<Impl>::cast_mut" | "core::ptr::mut_ptr::<Impl>::cast_const"
            )
        })
    }

    /// A thin-pointer `Deref::deref` / `DerefMut::deref_mut` whose
    /// dereferenced `&T` is a registered struct, resolved to the pointee
    /// struct root.  Valid only for a bodyless declaration whose handle's
    /// single pointer word *is* the pointee address, so `*p` is a
    /// zero-offset reinterpret: `Box<T>` (no header — `Unique<T>` points
    /// straight at `T`) and single-field transparent wrappers
    /// (`{UnsafeCell<T>}`).  For these the caller lowers the deref to the
    /// `cast_pointer(T, p)` downcast marker (`cast_pointer_marker_op`),
    /// which yields `SomeInstance(T)` (lltype.py) independent of the
    /// receiver's (classdef-less) annotation — the same shape emitted for
    /// `obj as *const W_Foo`.
    ///
    /// A declaration that carries an unstructured body in this LLBC is an
    /// ordinary method.  The body reads the pointee out of the handle, so
    /// the caller lowers the call on the normal path and that body is
    /// lowered and inlined like any other callee.  This returns `None`
    /// for those.
    ///
    /// `Rc<T>` / `Arc<T>` are the exception among bodyless handles: their
    /// word points at a refcount header, so the pointee sits at a non-zero
    /// offset and `cast_pointer` would reinterpret the header.  They are
    /// subtracted by owner-type leaf; an unresolved owner keeps the ordinary
    /// thin-pointer treatment, since the dereferenced `&T` must still
    /// resolve a registered struct root below.
    ///
    /// Returns `None` when the call is not a `deref` / `deref_mut` leaf,
    /// the callee has a body, or the dereferenced type is not a named ADT
    /// (slice / `str` derefs resolve no struct root — their `&[T]` /
    /// `&str` value model is the receiver's, narrowed by the list / string
    /// reprs, not a pointer downcast).
    pub(in crate::front::mir) fn deref_cast_root(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        let np = fd.item_meta.name_path();
        if !(np.ends_with("::deref") || np.ends_with("::deref_mut")) {
            return None;
        }
        // The body is the impl.  A one-word reinterpret would skip it.
        if crate::front::clause_spec::decl_has_unstructured_body(fd) {
            return None;
        }
        if let Some(leaf) = deref_impl_owner_leaf(self.llbc, fd)
            && matches!(leaf.as_str(), "Rc" | "Arc")
        {
            return None;
        }
        tyref_class_root_with(dest_ty, self.llbc, self.tombstoned_leaves)
    }

    /// `<String as Deref>::deref(&self) -> &str` / `<Vec<T> as
    /// Deref>::deref(&self) -> &[T]` (and their `deref_mut`).  Unlike the
    /// bodyless one-word handles [`Self::deref_cast_root`] reinterprets as
    /// a typed struct pointer, the owning-container deref returns the same
    /// value the lifted model already carries: Rust `String`/`&str` both
    /// lower to the immutable rpy_string and `Vec<T>`/`&[T]` both to the
    /// rpy_list, so `*s` is identity.  `deref_cast_root` returns `None`
    /// here (a `str`/slice target has no class root), so without this the
    /// callsite falls through to a `CallTarget::Method` `deref` getattr the
    /// rtyper cannot route on the classdef-less receiver.  Bind the
    /// destination to the argument instead, the same alias shape as the
    /// blanket-`into` identity above.
    pub(in crate::front::mir) fn is_container_identity_deref(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let np = fd.item_meta.name_path();
        if !(np.ends_with("::deref") || np.ends_with("::deref_mut")) {
            return false;
        }
        deref_impl_owner_leaf(self.llbc, fd)
            .is_some_and(|leaf| matches!(leaf.as_str(), "String" | "Vec"))
    }

    /// A resolved `core::borrow::Borrow::borrow` whose impl is one of the
    /// core/alloc identity views and whose method has no extracted body.
    /// A `Clause` ref and a local impl with a body stay calls.
    pub(in crate::front::mir) fn is_std_borrow_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Trait(payload) = &reg.kind else {
            return false;
        };
        let Some(impl_id) = crate::front::clause_spec::resolved_trait_impl_id(payload, self.llbc)
        else {
            return false;
        };
        let Some((fn_id, _)) = crate::front::clause_spec::trait_impl_method(payload, self.llbc)
        else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(fn_id) else {
            return false;
        };
        if fd.unstructured().is_some() {
            return false;
        }
        if fd.item_meta.name_path().rsplit("::").next() != Some("borrow") {
            return false;
        }
        let Some(ti) = self.llbc.trait_impls_raw().get(impl_id as usize) else {
            return false;
        };
        let Some(trait_id) = ti
            .pointer("/impl_trait/id")
            .and_then(serde_json::Value::as_u64)
        else {
            return false;
        };
        let Some(td) = self.llbc.trait_by_id(trait_id) else {
            return false;
        };
        if td.item_meta.name_path() != "core::borrow::Borrow" {
            return false;
        }
        let Some(types) = ti
            .pointer("/impl_trait/generics/types")
            .and_then(serde_json::Value::as_array)
        else {
            return false;
        };
        let (Some(self_ty), Some(borrowed)) = (types.first(), types.get(1)) else {
            return false;
        };
        crate::front::std_identity::is_identity_borrow_pair(self_ty, borrowed, self.llbc)
    }

    /// `<String as AsRef<str>>::as_ref(&self) -> &str`,
    /// `String::as_str(&self) -> &str`, `Wtf8::as_str(&self) -> &str`, and
    /// `<String as Borrow<str>>::borrow(&self) -> &str`, and pyre's
    /// `as_str_unchecked(&Wtf8) -> &str` — every one
    /// returns a `&str` view of the same string, an identity in the
    /// lifted value model (Rust `String`/`&str`/`str`/`Wtf8`/`Wtf8Buf`
    /// all lower to the immutable rpy_string).  `as_str_unchecked` names
    /// the view `Wtf8::as_str` gives once its validity arm has been taken
    /// separately: `as_str` itself returns `Result<&str, Utf8Error>`,
    /// whose destination does not strip to `str`, so it fails the gate
    /// below and stays a wall.  Without the intercept the
    /// call keeps a `CallTarget::Method` `as_ref` getattr the rtyper
    /// cannot route on the classdef-less string receiver (the `Cannot
    /// find attribute "as_ref" on UnicodeString` wall).  Bind the
    /// destination to the receiver instead, the same alias shape as the
    /// container deref above.  Gated on the receiver being a string value
    /// (`tyref_is_string_value`: `str`/`String`/`Wtf8`/`Wtf8Buf`), more
    /// precise than the impl owner-leaf, plus the `str`-typed result so
    /// `AsRef<[u8]>` is the `as_bytes` byte view
    /// ([`Self::is_string_as_bytes_identity`]) and `AsRef<OsStr>` /
    /// `AsRef<Path>` keep their ordinary lowering.
    pub(in crate::front::mir) fn is_string_to_str_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> bool {
        let leaf_owned: Option<String> = match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => self.llbc.fn_by_id(*id).and_then(|fd| {
                fd.item_meta
                    .name_path()
                    .rsplit("::")
                    .next()
                    .map(str::to_string)
            }),
            CallKind::Trait(payload) => {
                trait_payload_path_and_method(payload, self.llbc).map(|(_, method)| method)
            }
            _ => None,
        };
        let Some(leaf) = leaf_owned else {
            return false;
        };
        if !matches!(
            leaf.as_str(),
            "as_ref" | "as_str" | "as_str_unchecked" | "borrow"
        ) {
            return false;
        }
        if !first_arg_ty.is_some_and(|ty| tyref_is_string_family(ty, self.llbc)) {
            return false;
        }
        tyref_strips_to_str(dest_ty, self.llbc)
    }

    /// `utf8_payload_wtf8(*const UnicodeValueStorage) -> &Wtf8` — the STR
    /// header is already the string value (`Utf8Str` maps to `ValueType::Str`).
    pub(in crate::front::mir) fn is_utf8_payload_wtf8_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        fd.item_meta.name_path().rsplit("::").next() == Some("utf8_payload_wtf8")
    }

    /// `prebuilt_str_storage(&str) -> *mut UnicodeValueStorage` — the
    /// prebuilt rstr `STR` of a literal (`CONST_STR_CACHE`), the value its
    /// `&str` already lowers to.
    pub(in crate::front::mir) fn is_prebuilt_str_storage_identity(
        &self,
        reg: &RegularCall,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        matches!(
            fd.item_meta.name_path().as_str(),
            "pyre_object::unicodeobject::prebuilt_str_storage"
                | "pyre_object::unicodeobject::prebuilt_wtf8_storage"
        )
    }

    /// `utf8_payload_bytes(*const UnicodeValueStorage) -> &[u8]` — byte view
    /// of the same rstr `STR` header.
    pub(in crate::front::mir) fn is_utf8_payload_bytes_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        fd.item_meta.name_path().rsplit("::").next() == Some("utf8_payload_bytes")
    }

    pub(in crate::front::mir) fn is_w_str_storage(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path().rsplit("::").next() == Some("w_str_storage"))
    }

    pub(in crate::front::mir) fn is_w_bytes_data(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path().rsplit("::").next() == Some("w_bytes_data"))
    }

    pub(in crate::front::mir) fn is_w_bytes_block(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path().rsplit("::").next() == Some("w_bytes_block"))
    }

    pub(in crate::front::mir) fn is_bytes_block_chars(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path().rsplit("::").next() == Some("bytes_block_chars")
        })
    }

    pub(in crate::front::mir) fn graph_is_bytes_block_chars_or_string_view(&self) -> bool {
        self.graph
            .name
            .rsplit("::")
            .next()
            .unwrap_or(&self.graph.name)
            == "bytes_block_chars"
    }

    /// `W_BytesObject.data` as `rstr.py` `STR` (`W_BytesObject._value`).
    pub(in crate::front::mir) fn emit_w_bytes_data_str_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
    ) -> Variable {
        self.emit_w_bytes_data_field_read(bb_id, base, ValueType::Str)
    }

    /// Copy a `STR` into a list of unsigned (`Vec<u8>`). `list(str)` yields
    /// `Char`. Inverse of `rstr.py` `LLHelpers.ll_join_chars`.
    pub(in crate::front::mir) fn emit_ll_str_to_u8_list(
        &mut self,
        bb_id: BlockId,
        s: Variable,
    ) -> Variable {
        let dest = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(dest.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path([crate::front::join_chars::LL_STR2UINTLIST]),
                args: crate::model::call_args(vec![s]),
                result_ty: ValueType::Ref(None),
            },
        });
        dest
    }

    /// `{chars address, length}` of a `Ptr(STR)` header. Owner → window is a
    /// borrow: `rffi.get_raw_address_of_string` (`rgc.can_move` →
    /// `rgc.move_out_of_nursery`, then `offsetof(STR, 'chars')`) plus
    /// `__strlen`.
    pub(in crate::front::mir) fn emit_str_to_window(
        &mut self,
        mir_bb: usize,
        header: Variable,
    ) -> (Variable, Variable) {
        let bb_id = self.emit_bb(mir_bb);
        let data = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        stamp_as_ccharp(&data);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING.to_string(),
                    ],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![header.clone()]),
                result_ty: ValueType::Int,
            },
        });
        let len = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(len.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__strlen".to_string()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![header]),
                result_ty: ValueType::Int,
            },
        });
        (data, len)
    }

    /// Items address of a resizable byte list
    /// (`rgc.nonmoving_raw_ptr_for_resizable_list`). Result is CCHARP.
    pub(in crate::front::mir) fn emit_list_items_ccharp(
        &mut self,
        mir_bb: usize,
        list: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let data = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        stamp_as_ccharp(&data);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        crate::runtime_names::shims::NONMOVING_RAW_PTR_FOR_RESIZABLE_LIST
                            .to_string(),
                    ],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![list]),
                result_ty: ValueType::Int,
            },
        });
        self.ensure_window_ccharp(mir_bb, data)
    }

    /// `{items address, length}` of a `Vec<u8>` / `&mut [u8]` list. Owner →
    /// window is a borrow (`rgc.nonmoving_raw_ptr_for_resizable_list`).
    /// A fixed-size list (`Box<[u8]>` / `[u8; N]`) has no raw pointer:
    /// `rgc._check_resizable_list_of_chars` forces `listdef.resize()`.
    pub(in crate::front::mir) fn emit_list_to_window(
        &mut self,
        mir_bb: usize,
        list: Variable,
        src_ty: Option<&TyRef>,
    ) -> Result<(Variable, Variable), LowerError> {
        if let Some(len) = self.window_len_companion(&list) {
            return Ok((self.ensure_window_ccharp(mir_bb, list), len));
        }
        let bb_id = self.emit_bb(mir_bb);
        if let Some(data) =
            self.window_fat_component(bb_id, &list, crate::model::VecFieldPart::FatData)
        {
            let len = self
                .window_fat_component(bb_id, &list, crate::model::VecFieldPart::FatLen)
                .unwrap_or_else(|| {
                    let len = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(len.clone()),
                        kind: OpKind::ArrayLen {
                            base: list.clone(),
                            array_type_id: Some("[u8]".to_string()),
                            nolength: false,
                        },
                    });
                    len
                });
            return Ok((self.ensure_window_ccharp(mir_bb, data), len));
        }
        if src_ty.is_some_and(|ty| tyref_is_fixed_u8_list(ty, self.llbc)) {
            return Err(LowerError::Unsupported(
                "fixed-size list has no nonmoving raw ptr (rgc._check_resizable_list_of_chars)"
                    .to_string(),
            ));
        }
        let data = self.emit_list_items_ccharp(mir_bb, list.clone());
        let bb_id = self.emit_bb(mir_bb);
        // `<Vec<u8>>::len()` on a `SomeList` receiver is `__len`
        // ([`Self::is_container_len`]; `rlist.py` `ll_length`).
        let len = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(len.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__len".to_string()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![list]),
                result_ty: ValueType::Int,
            },
        });
        Ok((data, len))
    }

    /// Window data word is `CCHARP` (int-kind address) on every link.
    /// A GC fat box is split; a GC header is `rffi.cast(CCHARP, p)`;
    /// a Signed address is `cast_int_to_ptr(CCHARP, int)` (`lltype.py`
    /// `ann_cast_int_to_ptr`) so merge does not union `Ptr(Array)` with
    /// `Integer`.
    pub(in crate::front::mir) fn ensure_window_ccharp(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
    ) -> Variable {
        if self.is_fat_box_var(&ptr) {
            if let Ok((data, _)) = self.split_window_value(mir_bb, ptr.clone()) {
                return data;
            }
        }
        if ptr.concretetype().as_ref().is_some_and(lltype_is_ccharp) {
            return ptr;
        }
        if self.var_is_gc_ref_value(&ptr) {
            let bb_id = self.emit_bb(mir_bb);
            return emit_rffi_cast_to_carray_ptr(
                &mut self.graph,
                bb_id,
                ptr,
                &crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Char,
            );
        }
        let kind = ptr.concretetype().as_ref().map(crate::model::getkind);
        if matches!(kind, Some(crate::model::ConcreteType::Signed))
            && !ptr.concretetype().as_ref().is_some_and(lltype_is_ccharp)
        {
            let bb_id = self.emit_bb(mir_bb);
            emit_cast_int_to_ccharp(&mut self.graph, bb_id, ptr)
        } else {
            ptr
        }
    }

    /// `usize` / raw pointer whose window data word is not yet `CCHARP`.
    pub(in crate::front::mir) fn mir_ty_is_raw_ptr_or_int(&self, ty: Option<&TyRef>) -> bool {
        let Some(ty) = ty else {
            return false;
        };
        if tyref_is_raw_pointer(ty, self.llbc) {
            return true;
        }
        matches!(
            tyref_to_value_type_with(
                ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            ValueType::Int | ValueType::Unsigned
        )
    }

    /// `from_raw_parts{,_mut}` data word: cast a raw pointer / integer to
    /// `CCHARP` at this producer (`lltype.py` `ann_cast_int_to_ptr`).
    pub(in crate::front::mir) fn emit_window_from_raw_parts_ptr(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        ptr_ty: Option<&TyRef>,
    ) -> Variable {
        if self.mir_ty_is_raw_ptr_or_int(ptr_ty)
            && !ptr.concretetype().as_ref().is_some_and(lltype_is_ccharp)
        {
            let bb_id = self.emit_bb(mir_bb);
            emit_cast_int_to_ccharp(&mut self.graph, bb_id, ptr)
        } else {
            self.ensure_window_ccharp(mir_bb, ptr)
        }
    }

    /// Length word of a window data `var`, if this frame already holds it.
    pub(in crate::front::mir) fn window_len_companion(&self, var: &Variable) -> Option<Variable> {
        for (local, slot) in self.local_var.iter().enumerate() {
            let Some(value) = slot.as_ref() else {
                continue;
            };
            match value {
                LocalValue::Pair(ptr, len) if ptr == var => return Some(len.clone()),
                LocalValue::One(ptr) if ptr == var => {
                    let shadow = self.len_shadow.get(local).copied().flatten()?;
                    match self.local_var.get(shadow).and_then(|slot| slot.as_ref()) {
                        Some(LocalValue::One(len)) => return Some(len.clone()),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Window → STR copy (`rffi.py` `charpsize2str`,
    /// `majit_rlib::rffi::buffer::charpsize2str`).
    pub(in crate::front::mir) fn emit_charpsize2str(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        len: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        self.push_charpsize2str(bb_id, ptr, len)
    }

    pub(in crate::front::mir) fn push_charpsize2str(
        &mut self,
        bb_id: BlockId,
        ptr: Variable,
        len: Variable,
    ) -> Variable {
        let header = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(header.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path([
                    "majit_rlib",
                    "rffi",
                    "buffer",
                    "charpsize2str",
                ]),
                args: crate::model::call_args(vec![ptr, len]),
                result_ty: ValueType::Str,
            },
        });
        stamp_as_rstr_str(&header);
        header
    }

    /// `rffi.charpsize2str` / `charp2str`: a raw `*mut u8` argument of a
    /// function that returns a byte list or a `Result` whose `Ok` payload is one
    /// is `CCHARP`, using the same predicate the callee's Input uses
    /// (`tyref_raw_u8_ptr_param_is_ccharp`). Stamp the actual so
    /// `getcalldescr` sees Int-kind, matching the unique graph Input.
    pub(in crate::front::mir) fn coerce_ccharp_byte_list_call_args(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        mut args: Vec<Variable>,
    ) -> Vec<Variable> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return args;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return args;
        };
        let ret_is_u8_list = tyref_returns_u8_list(&fd.signature.output, self.llbc);
        if !ret_is_u8_list {
            return args;
        }
        for (i, ty) in fd.signature.inputs.iter().enumerate() {
            if i < args.len() && tyref_raw_u8_ptr_param_is_ccharp(ty, ret_is_u8_list, self.llbc) {
                let coerced = self.ensure_window_ccharp(mir_bb, args[i].clone());
                stamp_as_ccharp(&coerced);
                args[i] = coerced;
            }
        }
        args
    }

    /// Window → unsigned list (`rffi.py` `charpsize2str` → `Vec<u8>`).
    /// Direct call of the `rffi::buffer::charpsize2str` graph (crate-stripped
    /// `registered_path_for_fun_decl` / `spec_segments` spelling). Attach the
    /// FunDecl id when this artefact loaded it. The result is the list, not
    /// `STR`.
    pub(in crate::front::mir) fn emit_charpsize2str_list(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        len: Variable,
    ) -> Variable {
        let dest = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        let fun_decl_id = self.fun_decl_id_path_ends_with(&["rffi", "buffer", "charpsize2str"]);
        let segments = fun_decl_id
            .and_then(|id| self.llbc.fn_by_id(id))
            .map(|fd| registered_path_for_fun_decl(self.llbc, fd).segments)
            .unwrap_or_else(|| {
                vec![
                    "rffi".to_string(),
                    "buffer".to_string(),
                    "charpsize2str".to_string(),
                ]
            });
        let mut target = CallTarget::FunctionPath {
            segments,
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        };
        if let Some(fun_decl_id) = fun_decl_id {
            target = target.with_fun_decl_id(fun_decl_id);
        }
        let ptr = {
            let coerced = self.ensure_window_ccharp(mir_bb, ptr);
            stamp_as_ccharp(&coerced);
            coerced
        };
        let bb_id = self.emit_bb(mir_bb);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(dest.clone()),
            kind: OpKind::Call {
                target,
                args: crate::model::call_args(vec![ptr, len]),
                result_ty: ValueType::Ref(None),
            },
        });
        dest
    }

    /// One-byte `raw_load` at `data + index` (`CCHARP`).
    ///
    /// The result is `Signed` (`rstr.py` `CHAR`): `Unknown` banks as `'r'`
    /// and `rewrite_op_raw_load` would wrap the byte in `cast_int_to_ptr`.
    pub(in crate::front::mir) fn emit_window_raw_load(
        &mut self,
        mir_bb: usize,
        data: Variable,
        index: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let loaded = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(loaded.clone()),
            kind: OpKind::RawLoad {
                base: data,
                offset: index,
                item_ty: ValueType::Unsigned,
                itemsize: 1,
                is_item_signed: false,
            },
        });
        loaded
    }

    /// Byte-stride `data + k` of a window (`direct_ptradd` of `Char`).
    pub(in crate::front::mir) fn window_ptr_add(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        count: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let item = item_lltype_for_size(1);
        let addr = push_direct_ptradd(&mut self.graph, bb_id, ptr, count, &item);
        self.ensure_window_ccharp(mir_bb, addr)
    }

    /// Base of `xs[i]` whose item is a window byte (`s[i]` or `(*s)[i]`).
    pub(in crate::front::mir) fn place_is_window_index_base(&self, place: &Place) -> bool {
        if self.place_is_list_view(place) {
            return false;
        }
        if tyref_is_u8_window(&place.ty, self.llbc)
            || tyref_is_shared_u8_slice(&place.ty, self.llbc)
        {
            return true;
        }
        if let PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) = &place.kind
            && atom == "Deref"
            && tyref_is_u8_window(&inner.ty, self.llbc)
        {
            return true;
        }
        false
    }

    /// `(ptr, len)` of a window operand, `None` when `op` is not a window.
    pub(in crate::front::mir) fn window_operand(
        &mut self,
        mir_bb: usize,
        op: &Operand,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Const(_) => return Ok(None),
        };
        if place_is_bare_global(place) {
            return Ok(None);
        }
        let window_place = tyref_is_u8_window(&place.ty, self.llbc)
            || matches!(
                &place.kind,
                PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                    if atom == "Deref" && tyref_is_u8_window(&inner.ty, self.llbc)
            )
            || matches!(
                &place.kind,
                PlaceKind::Projection(inner, ProjectionElem::Tagged(v))
                    if (v.get("Field").is_some() || v.get("Downcast").is_some())
                        && tyref_is_u8_window(&inner.ty, self.llbc)
            );
        if window_place {
            if let Some((ptr, len)) = self.pair_place_parts(mir_bb, place)? {
                return Ok(Some((self.ensure_window_ccharp(mir_bb, ptr), len)));
            }
        }
        if self.operand_is_string_byte_view(op) || self.operand_is_list_view(op) {
            return Ok(None);
        }
        if !window_place {
            return Ok(None);
        }
        if let PlaceKind::Local(local) = place.kind {
            let var = self.resolve_place(mir_bb, place.clone())?;
            if self.local_holds_str_view(local as usize) || self.var_is_str_repr(&var) {
                let (ptr, len) = self.emit_str_to_window(mir_bb, var);
                return Ok(Some((self.ensure_window_ccharp(mir_bb, ptr), len)));
            }
            return Ok(Some(self.owner_to_window(mir_bb, var, Some(&place.ty))?));
        }
        Err(LowerError::Unsupported(format!(
            "bb{mir_bb}: window read through a projection"
        )))
    }

    /// `(ptr, len)` of an rvalue flowing into a window dest, or `None`
    /// when the rvalue is not a window / owner conversion this lowering
    /// knows.
    pub(in crate::front::mir) fn window_rvalue_parts(
        &mut self,
        mir_bb: usize,
        rvalue: &Rvalue,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        match rvalue {
            Rvalue::Use(op, _) => {
                if operand_tyref(op).is_some_and(|ty| tyref_is_u8_window(ty, self.llbc)) {
                    return self.window_operand(mir_bb, op);
                }
                if let Some(parts) = self.window_operand(mir_bb, op)? {
                    return Ok(Some(parts));
                }
                let Some(var) = (match op {
                    Operand::Copy(place) | Operand::Move(place) if !place_is_bare_global(place) => {
                        self.resolve_place(mir_bb, place.clone()).ok()
                    }
                    _ => None,
                }) else {
                    return Ok(None);
                };
                Ok(Some(self.owner_to_window(
                    mir_bb,
                    var,
                    operand_tyref(op),
                )?))
            }
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => match &place.kind {
                PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                    if place_is_bare_global(inner) {
                        return Ok(None);
                    }
                    if let Some(parts) = self.pair_place_parts(mir_bb, inner)? {
                        return Ok(Some(parts));
                    }
                    let var = self.resolve_place(mir_bb, inner.as_ref().clone())?;
                    Ok(Some(self.owner_to_window(mir_bb, var, Some(&inner.ty))?))
                }
                _ => {
                    if place_is_bare_global(place) {
                        return Ok(None);
                    }
                    if let Some(parts) = self.pair_place_parts(mir_bb, place)? {
                        return Ok(Some(parts));
                    }
                    if let Some(parts) = self.u8_array_unsize_source(mir_bb, place) {
                        return Ok(Some(parts));
                    }
                    let var = self.resolve_place(mir_bb, place.clone())?;
                    Ok(Some(self.owner_to_window(mir_bb, var, Some(&place.ty))?))
                }
            },
            Rvalue::UnaryOp(cast, Operand::Copy(src) | Operand::Move(src))
                if cast_kind_is_unsize(cast) =>
            {
                Ok(self.u8_array_unsize_source(mir_bb, src))
            }
            Rvalue::Cast(cast, op, _) => match op {
                Operand::Copy(src) | Operand::Move(src) if cast_kind_is_unsize(cast) => {
                    if let Some(parts) = self.u8_array_unsize_source(mir_bb, src) {
                        return Ok(Some(parts));
                    }
                    self.window_operand(mir_bb, op)
                }
                _ => self.window_operand(mir_bb, op),
            },
            _ => Ok(None),
        }
    }

    /// Split a fat window SSA (`FieldRead` or residual call result) into
    /// `{data, len}`.
    pub(in crate::front::mir) fn split_window_value(
        &mut self,
        mir_bb: usize,
        fat: Variable,
    ) -> Result<(Variable, Variable), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        if let Some(data) =
            self.window_fat_component(bb_id, &fat, crate::model::VecFieldPart::FatData)
            && let Some(len) =
                self.window_fat_component(bb_id, &fat, crate::model::VecFieldPart::FatLen)
        {
            return Ok((self.ensure_window_ccharp(mir_bb, data), len));
        }
        let data = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(data.clone()),
            kind: OpKind::FieldRead {
                base: fat.clone(),
                field: FieldDescriptor::new("data", None)
                    .with_vec_part(crate::model::VecFieldPart::FatData)
                    .with_owner_declared_gc(Some(false)),
                ty: ValueType::Unsigned,
                pure: false,
            },
        });
        let len = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(len.clone()),
            kind: OpKind::FieldRead {
                base: fat,
                field: FieldDescriptor::new("len", None)
                    .with_vec_part(crate::model::VecFieldPart::FatLen)
                    .with_owner_declared_gc(Some(false)),
                ty: ValueType::Int,
                pure: false,
            },
        });
        Ok((self.ensure_window_ccharp(mir_bb, data), len))
    }

    /// Convert a STR / list / fat-box owner into a window pair.
    /// Decide by the value's representation first: a STR header is
    /// `emit_str_to_window` whatever the place's static type is.
    pub(in crate::front::mir) fn owner_to_window(
        &mut self,
        mir_bb: usize,
        var: Variable,
        src_ty: Option<&TyRef>,
    ) -> Result<(Variable, Variable), LowerError> {
        if let Some(len) = self.window_len_companion(&var) {
            return Ok((self.ensure_window_ccharp(mir_bb, var), len));
        }
        if var.concretetype().as_ref().is_some_and(lltype_is_ccharp) {
            return Err(LowerError::Unsupported(
                "owner_to_window: window data word without a length".to_string(),
            ));
        }
        if self.var_is_str_repr(&var)
            || self.var_has_str_value_type(&var)
            || src_ty.is_some_and(|ty| tyref_is_w1_str(ty, self.llbc))
        {
            let (ptr, len) = self.emit_str_to_window(mir_bb, var);
            return Ok((self.ensure_window_ccharp(mir_bb, ptr), len));
        }
        if self.list_view_local_of_var(&var).is_some() {
            return self.emit_list_view_to_window(mir_bb, var);
        }
        if src_ty.is_some_and(|ty| tyref_is_u8_list(ty, self.llbc)) {
            return self.emit_list_to_window(mir_bb, var, src_ty);
        }
        if src_ty.is_some_and(|ty| tyref_is_u8_window(ty, self.llbc)) {
            let bb_id = self.emit_bb(mir_bb);
            if let Some(data) =
                self.window_fat_component(bb_id, &var, crate::model::VecFieldPart::FatData)
            {
                let len = self
                    .window_fat_component(bb_id, &var, crate::model::VecFieldPart::FatLen)
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: window fat box has no length word"
                        ))
                    })?;
                return Ok((data, len));
            }
            // A window is SomeTuple([SomePtr(CCHARP), SomeInteger])
            // (`model.py SomeTuple`, `rffi.py CCHARP`), so an already-split
            // buffer is read as its two fields (`rtuple.py TupleRepr.getitem`).
            // `var` addresses the `{data, len}` slot; the FieldRead
            // {FatData, FatLen} fallback `pair_place_parts` uses for a
            // non-Local window loads those two words.
            return self.split_window_value(mir_bb, var);
        }
        Err(LowerError::Unsupported(format!(
            "bb{mir_bb}: owner_to_window: unclassified owner"
        )))
    }

    /// `[u8; N]` / `&[u8; N]` unsized onto a window: `{array, N}`.
    pub(in crate::front::mir) fn u8_array_unsize_source(
        &mut self,
        mir_bb: usize,
        src: &Place,
    ) -> Option<(Variable, Variable)> {
        let PlaceKind::Local(local) = src.kind else {
            return None;
        };
        let array_ty = ref_pointee_ty(&src.ty, self.llbc).unwrap_or_else(|| clone_tyref(&src.ty));
        let (item, len) = tyref_raw_array_parts(&array_ty, self.llbc)?;
        if json_ty_scalar_element_spelling(item, self.llbc).as_deref() != Some("u8") {
            return None;
        }
        let ptr = self
            .local_var
            .get(local as usize)
            .and_then(|slot| slot.as_ref())
            .and_then(|value| value.one().ok())?;
        let bb_id = self.emit_bb(mir_bb);
        let ptr = push_cast_ptr_to_int(&mut self.graph, bb_id, ptr);
        let ptr = self.ensure_window_ccharp(mir_bb, ptr);
        let len = self.emit_const_uint(mir_bb, len);
        Some((ptr, len))
    }

    /// Subslice of a window: `{data + start, end - start}` (byte stride).
    pub(in crate::front::mir) fn window_subslice(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        len: Variable,
        range: &Variable,
        range_is_full: bool,
        checked: bool,
    ) -> Result<(Variable, Variable), LowerError> {
        if range_is_full {
            return Ok((self.ensure_window_ccharp(mir_bb, ptr), len));
        }
        let (start, end) = if let Some(site) = self
            .slice_index_range_sites
            .iter()
            .find(|site| site.range_result == *range)
        {
            (Some(site.start.clone()), site.end.clone())
        } else if let Some(site) = self
            .slice_index_rangefrom_sites
            .iter()
            .find(|site| site.range_result == *range)
        {
            (Some(site.start.clone()), len.clone())
        } else if let Some(site) = self
            .slice_index_rangeto_sites
            .iter()
            .find(|site| site.range_result == *range)
        {
            (None, site.end.clone())
        } else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: window subslice by a range this body did not build"
            )));
        };
        let Some(start) = start else {
            if !checked {
                return Ok((self.ensure_window_ccharp(mir_bb, ptr), end));
            }
            let in_bounds = self.emit_uint_binop(mir_bb, "le", end.clone(), len);
            let carried = self.fork_in_bounds(mir_bb, in_bounds, &[ptr, end])?;
            return Ok((
                self.ensure_window_ccharp(mir_bb, carried[0].clone()),
                carried[1].clone(),
            ));
        };
        let (ptr, start, end) = if checked {
            let start_le_end = self.emit_uint_binop(mir_bb, "le", start.clone(), end.clone());
            let end_le_len = self.emit_uint_binop(mir_bb, "le", end.clone(), len);
            let in_bounds = self.emit_uint_binop(mir_bb, "bitand", start_le_end, end_le_len);
            let carried = self.fork_in_bounds(mir_bb, in_bounds, &[ptr, start, end])?;
            (carried[0].clone(), carried[1].clone(), carried[2].clone())
        } else {
            (ptr, start, end)
        };
        let ptr = self.window_ptr_add(mir_bb, ptr, start.clone());
        let len = self.emit_uint_binop(mir_bb, "sub", end, start);
        Ok((ptr, len))
    }

    /// Proven window-view identity. `jtransform.handle_regular_call` keeps
    /// the target graph; only these callees may drop the call. Deref is
    /// [`Self::is_container_identity_deref`], borrow is
    /// [`Self::is_std_borrow_identity`].
    pub(in crate::front::mir) fn is_window_view_identity(&self, reg: &RegularCall) -> bool {
        self.is_container_slice_identity(reg)
            || self.is_container_identity_deref(reg)
            || self.is_std_borrow_identity(reg)
    }

    /// Dest is a window local: STR→WINDOW / LIST→WINDOW / `(p, n)` /
    /// identity / residual fat split. Never `ll_join_chars`.
    pub(in crate::front::mir) fn lower_window_dest_call(
        &mut self,
        mir_bb: usize,
        name: &str,
        args: &[Variable],
        arg_tys: &[Option<TyRef>],
        pair_lens: &[(usize, Variable)],
        reg: &RegularCall,
        dest_ty: &TyRef,
        arg_is_range_full: bool,
    ) -> Result<(Variable, Variable), LowerError> {
        if matches!(
            name,
            "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut"
        ) && args.len() == 2
        {
            let ptr_ty = arg_tys.first().and_then(Option::as_ref).map(|ty| {
                tyref_to_value_type_with(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            });
            let ptr = match ptr_ty {
                Some(ty @ (ValueType::Int | ValueType::Unsigned)) => {
                    self.retype_word(mir_bb, args[0].clone(), &ty, &ValueType::Unsigned)
                }
                Some(ValueType::Ref(_)) => {
                    let bb_id = self.emit_bb(mir_bb);
                    let (retype, ptr) =
                        push_ptr_to_unsigned_cast(&mut self.graph, bb_id, args[0].clone());
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(ptr.clone()),
                        kind: retype,
                    });
                    ptr
                }
                _ => args[0].clone(),
            };
            return Ok((
                self.emit_window_from_raw_parts_ptr(
                    mir_bb,
                    ptr,
                    arg_tys.first().and_then(Option::as_ref),
                ),
                args[1].clone(),
            ));
        }
        if is_slice_from_ref_name(name) && args.len() == 1 {
            let ptr_ty = arg_tys.first().and_then(Option::as_ref).map(|ty| {
                tyref_to_value_type_with(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            });
            let ptr = match ptr_ty {
                Some(ty @ (ValueType::Int | ValueType::Unsigned)) => {
                    self.retype_word(mir_bb, args[0].clone(), &ty, &ValueType::Unsigned)
                }
                Some(ValueType::Ref(_)) => {
                    let bb_id = self.emit_bb(mir_bb);
                    let (retype, ptr) =
                        push_ptr_to_unsigned_cast(&mut self.graph, bb_id, args[0].clone());
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(ptr.clone()),
                        kind: retype,
                    });
                    ptr
                }
                _ => args[0].clone(),
            };
            let len = self.emit_const_uint(mir_bb, 1);
            return Ok((self.ensure_window_ccharp(mir_bb, ptr), len));
        }
        if matches!(
            name,
            "core::slice::index::<Impl>::index"
                | "core::slice::index::<Impl>::index_mut"
                | "core::slice::<Impl>::get"
                | "alloc::vec::<Impl>::index"
                | "alloc::vec::<Impl>::index_mut"
                | "core::array::<Impl>::index"
                | "core::array::<Impl>::index_mut"
        ) && args.len() == 2
        {
            if let Some(len) = pair_lens
                .iter()
                .find(|(index, _)| *index == 0)
                .map(|(_, len)| len.clone())
                .or_else(|| self.window_len_companion(&args[0]))
            {
                return self.window_subslice(
                    mir_bb,
                    args[0].clone(),
                    len,
                    &args[1],
                    arg_is_range_full,
                    true,
                );
            }
        }
        if let Some((0, len)) = pair_lens.first().cloned()
            && self.is_window_view_identity(reg)
        {
            return Ok((self.ensure_window_ccharp(mir_bb, args[0].clone()), len));
        }
        let first_ty = arg_tys.first().and_then(Option::as_ref);
        let w1_str = first_ty.is_some_and(|ty| {
            tyref_is_string_value(ty, self.llbc)
                || tyref_raw_ptr_pointee_is_string_value(ty, self.llbc)
        });
        let str_src = args.len() == 1
            && ((self.is_string_as_bytes_identity(reg, first_ty, dest_ty) && w1_str)
                || self.is_utf8_payload_bytes_identity(reg)
                || self.is_bytes_block_chars(reg)
                || self.is_w_bytes_data(reg)
                || self.is_w_bytes_block(reg)
                || self.is_w_str_storage(reg));
        if str_src {
            let header = if self.is_w_bytes_data(reg) || self.is_w_bytes_block(reg) {
                let bb_id = self.emit_bb(mir_bb);
                self.emit_w_bytes_data_str_read(bb_id, args[0].clone())
            } else if self.is_w_str_storage(reg) {
                let bb_id = self.emit_bb(mir_bb);
                self.emit_w_str_utf8_read(bb_id, args[0].clone())
            } else {
                args[0].clone()
            };
            let (ptr, len) = self.emit_str_to_window(mir_bb, header);
            return Ok((self.ensure_window_ccharp(mir_bb, ptr), len));
        }
        if args.len() == 1
            && (self.is_container_slice_identity(reg)
                || self.is_container_identity_deref(reg)
                || (first_ty.is_some_and(|ty| tyref_is_u8_list(ty, self.llbc))
                    && self.is_window_view_identity(reg)))
        {
            return self.emit_list_to_window(mir_bb, args[0].clone(), first_ty);
        }
        if args.len() == 1 && self.list_view_local_of_var(&args[0]).is_some() {
            return self.emit_list_view_to_window(mir_bb, args[0].clone());
        }
        if args.len() == 2
            && first_ty.is_some_and(|ty| tyref_is_u8_list(ty, self.llbc))
            && matches!(
                name,
                "alloc::vec::<Impl>::index"
                    | "alloc::vec::<Impl>::index_mut"
                    | "core::slice::index::<Impl>::index"
                    | "core::slice::index::<Impl>::index_mut"
            )
        {
            let (ptr, len) = self.emit_list_to_window(mir_bb, args[0].clone(), first_ty)?;
            return self.window_subslice(mir_bb, ptr, len, &args[1], arg_is_range_full, true);
        }
        if args.len() == 1
            && first_ty
                .is_some_and(|ty| tyref_is_w1_str(ty, self.llbc) || tyref_is_u8_list(ty, self.llbc))
            && self.is_window_view_identity(reg)
        {
            return Ok(self.owner_to_window(mir_bb, args[0].clone(), first_ty)?);
        }
        // A callee whose output is a window uses the two-word return ABI:
        // the caller allocates `ret_len_slot_local` (`SLICE_LEN_SLOT_NEW`),
        // passes it, and reads the length (`SLICE_LEN_SLOT_TAKE`). Same
        // path as `lower_pair_call` residual pair-return.
        if regular_call_fun_decl_id(&reg.kind)
            .and_then(|id| self.llbc.fn_by_id(id))
            .is_some_and(|fd| self.pair_or_window_fun_inputs(fd, &reg.generics).1)
        {
            let slot = self.emit_path_call(
                mir_bb,
                majit_ir::rvec::SLICE_LEN_SLOT_NEW,
                vec![],
                ValueType::Unsigned,
            );
            let target_op = self.pair_call_target(mir_bb, reg, args, pair_lens)?;
            let mut call_args = splice_pair_lens(args.to_vec(), pair_lens.to_vec());
            call_args.push(slot.clone());
            let ptr = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(ptr.clone()),
                kind: OpKind::Call {
                    target: target_op,
                    args: crate::model::call_args(call_args),
                    result_ty: ValueType::Unsigned,
                },
            });
            let len = self.emit_path_call(
                mir_bb,
                majit_ir::rvec::SLICE_LEN_SLOT_TAKE,
                vec![slot],
                ValueType::Unsigned,
            );
            return Ok((self.ensure_window_ccharp(mir_bb, ptr), len));
        }
        let target_op = self.pair_call_target(mir_bb, reg, args, pair_lens)?;
        let call_args = splice_pair_lens(args.to_vec(), pair_lens.to_vec());
        let fat = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let bb_id = self.emit_bb(mir_bb);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(fat.clone()),
            kind: OpKind::Call {
                target: target_op,
                args: crate::model::call_args(call_args),
                result_ty: ValueType::Int,
            },
        });
        self.fat_box_vars.push(fat.clone());
        self.split_window_value(mir_bb, fat)
    }

    /// Two `FieldWrite`s of a window pair into a fat `{data, len}` field.
    pub(in crate::front::mir) fn emit_window_field_write(
        &mut self,
        mir_bb: usize,
        inner: Place,
        elem: ProjectionElem,
        ptr: Variable,
        len: Variable,
    ) -> Result<(), LowerError> {
        let ProjectionElem::Tagged(v) = elem else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: window field write through a non-field projection"
            )));
        };
        let field_payload = v.as_object().and_then(|m| m.get("Field")).ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: window field write without a Field payload"
            ))
        })?;
        let field_base_ty = clone_tyref(&inner.ty);
        let base_is_deref = matches!(
            &inner.kind,
            PlaceKind::Projection(_, ProjectionElem::Atom(atom)) if atom == "Deref"
        );
        let base = self.resolve_place(mir_bb, inner)?;
        let bb_id = self.emit_bb(mir_bb);
        let field = match self.resolve_adt_field(&field_base_ty, field_payload) {
            Some((owner_root, field_name, _, owner_id, host_index)) => {
                FieldDescriptor::new(field_name, Some(owner_root))
                    .with_owner_id(owner_id)
                    .with_owner_declared_gc(Some(false))
                    .with_base_is_deref(base_is_deref)
                    .with_host_index(host_index)
            }
            None => {
                let idx = self.positional_field_index(field_payload).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: window field write: unresolved field index"
                    ))
                })?;
                FieldDescriptor::new(format!("__pos_{idx}"), None)
                    .with_owner_declared_gc(Some(false))
                    .with_base_is_deref(base_is_deref)
            }
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: base.clone(),
                field: field
                    .clone()
                    .with_vec_part(crate::model::VecFieldPart::FatData),
                value: LinkArg::Value(ptr),
                ty: ValueType::Unsigned,
            },
        });
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base,
                field: field.with_vec_part(crate::model::VecFieldPart::FatLen),
                value: LinkArg::Value(len),
                ty: ValueType::Int,
            },
        });
        Ok(())
    }

    pub(in crate::front::mir) fn emit_w_bytes_data_field_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        ty: ValueType,
    ) -> Variable {
        let narrowed = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(narrowed.clone()),
            kind: crate::model::cast_instance_call("W_BytesObject", base),
        });
        let field = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(field.clone()),
            kind: OpKind::FieldRead {
                base: narrowed,
                field: FieldDescriptor::new("data", Some("W_BytesObject".to_string()))
                    .with_owner_id(Some(majit_ir::descr::StructId::from_canonical(
                        "bytesobject::W_BytesObject",
                    )))
                    .with_base_is_deref(true),
                ty,
                pure: false,
            },
        });
        field
    }

    /// View no-op: `as_*` whose receiver and dest share a representation
    /// class ([`is_as_view_noop`]). `str::bytes` is the same byte view
    /// and is not an `as_*` name, so it stays a separate arm.
    ///
    /// Sound only for len / equality / iteration consumers.  A scalar
    /// index (`as_bytes()[i]`) lowers to `getitem` on `StringRepr`, which
    /// yields a `Char`, not the `u8` the Rust source expects; there is no
    /// `(CharRepr, IntegerRepr)` eq/arithmetic pairtype, so such a use
    /// fails rtype and residualizes rather than miscompiling.  Do NOT add
    /// a `(CharRepr, IntegerRepr)` eq arm without lowering a real
    /// byte-slice view first.
    pub(in crate::front::mir) fn is_string_as_bytes_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> bool {
        if is_as_view_noop(self.llbc, reg, first_arg_ty, dest_ty)
            && first_arg_ty.is_some_and(|ty| tyref_is_w1_str_receiver(ty, self.llbc))
        {
            return true;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next();
        // `str::bytes` is the same byte view as `as_bytes`: the iterator
        // yields `u8` and the string is already its bytes. `str::chars`
        // yields Unicode scalars (`Utf8StringIterator`), so it is not a
        // byte view and keeps the ordinary call.
        // `<String as AsRef<[u8]>>::as_ref` is `as_bytes`.
        // A Rust-heap `String` receiver keeps the pre-existing `as_bytes`
        // identity for len / eq / iteration consumers; its window conversion
        // is a separate follow-up.
        let byte_view = leaf == Some("as_bytes")
            || (leaf == Some("bytes") && path.contains("str"))
            || is_as_ref_byte_view(leaf, dest_ty, self.llbc);
        if !byte_view {
            return false;
        }
        first_arg_ty.is_some_and(|ty| {
            tyref_is_string_family(ty, self.llbc) || tyref_is_w1_str(ty, self.llbc)
        })
    }

    /// `alloc::str` methods that are already `StringRepr` rtype methods.
    /// `to_ascii_lowercase` is the ASCII case fold `ll_lower` performs on the
    /// byte string. `replace` is `ll_replace_chr_chr` (receiver, pattern, repl).
    pub(in crate::front::mir) fn string_repr_method_name(
        &self,
        reg: &RegularCall,
        argc: usize,
        first_arg_ty: Option<&TyRef>,
    ) -> Option<&'static str> {
        if !first_arg_ty.is_some_and(|ty| tyref_is_string_value(ty, self.llbc)) {
            return None;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let path = self.llbc.fn_by_id(*id)?.item_meta.name_path();
        let leaf = path.rsplit("::").next()?;
        match (leaf, argc) {
            ("to_ascii_lowercase", 1) if path.contains("str") => Some("lower"),
            ("replace", 3) if path.contains("str") => Some("replace"),
            _ => None,
        }
    }

    /// `Wtf8::from_bytes_unchecked` — the inverse of `Wtf8::as_bytes`.
    /// The destination is the string; the operand is its byte storage.
    pub(in crate::front::mir) fn is_wtf8_from_bytes_identity(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next();
        // `str::from_utf8_unchecked(&[u8]) -> &str` is the same byte-for-byte
        // view as `Wtf8::from_bytes_unchecked`: the destination is the string.
        if leaf != Some("from_bytes_unchecked") && leaf != Some("from_utf8_unchecked") {
            return false;
        }
        tyref_is_string_value(dest_ty, self.llbc)
    }

    /// Inherent checked `Wtf8::from_bytes(&[u8]) -> Option<&Wtf8>`.
    /// Distinct from [`Self::is_wtf8_from_bytes_identity`]: that arm is
    /// only `from_bytes_unchecked` / `from_utf8_unchecked`.
    pub(in crate::front::mir) fn is_wtf8_from_bytes_checked(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let owner_ok = impl_method_owner_for_fundecl(self.llbc, fd).is_some_and(|(owner, leaf)| {
            leaf == "from_bytes" && owner.rsplit("::").next() == Some("Wtf8")
        });
        if !owner_ok {
            return false;
        }
        crate::front::result_exc::tyref_is_option(dest_ty, self.llbc)
            && self
                .tyref_adt_type_arg(dest_ty, 0)
                .is_some_and(|payload| tyref_is_string_family(&payload, self.llbc))
    }

    /// `alloc::string::<Impl>::from_utf8_lossy` whose dest is `Cow<str>`.
    pub(in crate::front::mir) fn is_string_from_utf8_lossy(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        fd.item_meta.name_path().ends_with("::from_utf8_lossy") && self.tyref_is_cow_str(dest_ty)
    }

    /// `alloc::borrow::Cow<B>` whose borrowed form `B` is string-family
    /// (`Cow<str>` / `Cow<&str>`).
    ///
    /// `Deref` takes `&self`, so the receiver is `&Cow<str>`.
    /// [`Self::tyref_adt_type_arg`] reads the Adt node and does not peel
    /// `Ref`; [`adt_path_of_tyref`] already does, which is why the path
    /// check would pass and the payload check would not.
    pub(in crate::front::mir) fn tyref_is_cow_str(&self, ty: &TyRef) -> bool {
        let Some(path) = adt_path_of_tyref(ty, self.llbc) else {
            return false;
        };
        if !path.ends_with("::Cow") {
            return false;
        }
        let peeled = self.tyref_peel_ref_to_pointee(ty);
        let adt_ty = peeled.as_ref().unwrap_or(ty);
        self.tyref_adt_type_arg(adt_ty, 0).is_some_and(|payload| {
            tyref_is_string_family(&payload, self.llbc) || tyref_strips_to_str(&payload, self.llbc)
        })
    }

    pub(in crate::front::mir) fn fun_decl_id_path_ends_with(&self, suffix: &[&str]) -> Option<u64> {
        let mut found = None;
        for fd in self.llbc.iter_fun_decls() {
            let segs: Vec<&str> = fd.item_meta.name_path_str().split("::").collect();
            if !segs.ends_with(suffix) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(fd.def_id);
        }
        found
    }

    /// `pyre_object::rutf8::check_utf8` (`rutf8.py check_utf8`).
    pub(in crate::front::mir) fn rutf8_check_utf8_fun_decl_id(&self) -> Option<u64> {
        self.fun_decl_id_path_ends_with(&["rutf8", "check_utf8"])
    }

    /// `typedef::decode_utf8_with_errors` (`unicodehelper.py
    /// _str_decode_utf8_slowpath` / `str_decode_utf8` replace).
    pub(in crate::front::mir) fn decode_utf8_with_errors_fun_decl_id(&self) -> Option<u64> {
        self.fun_decl_id_path_ends_with(&["typedef", "decode_utf8_with_errors"])
    }

    pub(in crate::front::mir) fn emit_fun_decl_call(
        &mut self,
        bb_id: BlockId,
        fun_decl_id: u64,
        segments: Vec<String>,
        args: Vec<Variable>,
        result_ty: ValueType,
    ) -> Variable {
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let target = CallTarget::FunctionPath {
            segments,
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        }
        .with_fun_decl_id(fun_decl_id);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::Call {
                target,
                args: crate::model::call_args(args),
                result_ty,
            },
        });
        result
    }

    pub(in crate::front::mir) fn stamp_successor_local_as_rstr_str(
        &mut self,
        target: usize,
        dest_local: usize,
    ) {
        if !self.block_id.is_lowered(target) {
            return;
        }
        let row = self.block_id.row(target);
        let entry = self.block_entry_local_var[row].unpack();
        if let Some(Some(var)) = entry.get(dest_local) {
            stamp_as_rstr_str(var);
        }
        // The dest is already a string byte view in this block
        // (`from_utf8_lossy` / niche `from_bytes`).  Successor `Deref` /
        // `&*cow` hydrates `block_entry_string_byte_view_locals`; without
        // the mark a projected `&Cow<str>` keeps the residual Cow deref.
        if !self.block_byte_view_conflict[row].contains(dest_local) {
            self.block_byte_view_seen[row].insert(dest_local);
            self.block_entry_string_byte_view_locals[row].insert(dest_local);
        }
    }

    /// Split a terminator-call block on `disc`. `set_branch` wraps
    /// `bool()`, so a Result tag 0 (Ok) takes the false arm and Err
    /// takes the true arm.
    ///
    /// Arms join a fresh block, then that join is the emit cursor for
    /// this MIR block (`emit_block.set`) and the goto to the MIR
    /// successor — the slice-bounds fork shape.  Framestate pass 1
    /// unions `emit_bb`'s exits, so a direct arm→successor edge would
    /// drop the MIR successor out of the pending set.
    ///
    /// `true_raises` closes the Err arm with
    /// `set_raise_implicit(AssertionError)` (`regular_call_is_panicking_abort`
    /// / `TermKind::Abort` / `rtyper/debug.py ll_assert`) and does not
    /// goto the join.  `unicodehelper.py str_decode_utf8` replace never
    /// returns Err.  W1 leaves this false: its Err arm is the real
    /// `Option::None`.
    #[expect(
        clippy::too_many_arguments,
        reason = "Diamond wiring matches the emit-time Result unwrap; a context object would hide the arm extras"
    )]
    pub(in crate::front::mir) fn close_call_diamond(
        &mut self,
        mir_bb: usize,
        target: usize,
        dest_local: usize,
        disc: Variable,
        mark_string: bool,
        true_raises: bool,
        true_extras: &[Variable],
        false_extras: &[Variable],
        true_produce: impl FnOnce(&mut Self, BlockId, &[Variable]) -> Variable,
        false_produce: impl FnOnce(&mut Self, BlockId, &[Variable]) -> Variable,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let placeholder = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        if mark_string {
            stamp_as_rstr_str(&placeholder);
            if !self.string_byte_view_locals.contains(&dest_local) {
                self.string_byte_view_locals.push(dest_local);
            }
        }
        self.local_var[dest_local] = Some(LocalValue::One(placeholder.clone()));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;

        let mut carried: Vec<Variable> = Vec::new();
        for v in &link_args {
            if *v != placeholder && !carried.contains(v) {
                carried.push(v.clone());
            }
        }

        let mut true_sources = Vec::new();
        if !true_raises {
            true_sources = carried.clone();
            for extra in true_extras {
                if !true_sources.contains(extra) {
                    true_sources.push(extra.clone());
                }
            }
        }
        let mut false_sources = carried.clone();
        for extra in false_extras {
            if !false_sources.contains(extra) {
                false_sources.push(extra.clone());
            }
        }

        let (true_bb, true_inputs) = self.graph.create_block_with_arg_vars(true_sources.len());
        let (false_bb, false_inputs) = self.graph.create_block_with_arg_vars(false_sources.len());

        let true_mapped: Vec<Variable> = if true_raises {
            Vec::new()
        } else {
            true_extras
                .iter()
                .map(|v| {
                    crate::front::option_variant::map_source(&true_sources, &true_inputs, v)
                        .expect("true-arm extra threaded")
                })
                .collect()
        };
        let false_mapped: Vec<Variable> = false_extras
            .iter()
            .map(|v| {
                crate::front::option_variant::map_source(&false_sources, &false_inputs, v)
                    .expect("false-arm extra threaded")
            })
            .collect();

        let true_val = if true_raises {
            None
        } else {
            Some(true_produce(self, true_bb, &true_mapped))
        };
        let false_val = false_produce(self, false_bb, &false_mapped);

        let (join, join_inputs) = self.graph.create_block_with_arg_vars(link_args.len());
        if let Some(true_val) = true_val {
            let true_args = remap_diamond_goto_args(
                &link_args,
                &placeholder,
                &true_val,
                &true_sources,
                &true_inputs,
            );
            self.graph.set_goto(true_bb, join, true_args);
        } else {
            self.graph.set_raise_implicit(true_bb, "AssertionError");
        }
        let false_args = remap_diamond_goto_args(
            &link_args,
            &placeholder,
            &false_val,
            &false_sources,
            &false_inputs,
        );
        self.graph.set_goto(false_bb, join, false_args);
        self.graph
            .set_branch(bb_id, disc, true_bb, true_sources, false_bb, false_sources);

        // Same forwarded-word remap as the slice-bounds fork, not an
        // optimizer side table: join phis replace the head Variables so
        // framestate `getstate` after the split is defined on `join`.
        let mut remap: std::collections::HashMap<u64, Variable> = std::collections::HashMap::new();
        for (src, dst) in link_args.iter().zip(join_inputs.iter()) {
            remap.insert(src.id(), dst.clone());
        }
        for value in self.local_var.iter_mut().flatten() {
            value.remap_words(&remap);
        }
        self.remap_forwarded_provenance(&remap);
        if mark_string {
            if let Some(LocalValue::One(var)) = &self.local_var[dest_local] {
                stamp_as_rstr_str(var);
            }
            self.stamp_successor_local_as_rstr_str(target, dest_local);
        }
        self.emit_block.set(mir_bb, join);
        let join_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(join, target_bb, join_args);
        Ok(())
    }

    /// Checked `Wtf8::from_bytes` → `rutf8.check_utf8(bytes, true)` then
    /// Ok → `Option::Some` of the unchecked byte identity, Err → `None`.
    /// `rutf8.py check_utf8` / `unicodehelper.py str_decode_utf8` fast arm.
    ///
    /// `check_utf8`'s `&[u8]` formal is a window (`pair_or_window_fun_inputs` /
    /// `pair_len_shadows_of`), so the emitted call is `{data: CCHARP, len}`
    /// plus `allow_surrogates`. A one-word STR owner is split the same way
    /// other `&[u8]` callees are. Dest `&Wtf8` is `rstr.STR`; the Ok arm
    /// is that header (`charpsize2str` of the window, or the STR itself).
    #[expect(
        clippy::too_many_arguments,
        reason = "Mirrors the other try_lower_* call intercepts"
    )]
    pub(in crate::front::mir) fn try_lower_wtf8_from_bytes_checked(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if !self.is_wtf8_from_bytes_checked(reg, dest_ty) {
            return Ok(false);
        }
        let (data, len, ok_str) = match args {
            [bytes] => {
                if let Some(len) = self.window_len_companion(bytes) {
                    (self.ensure_window_ccharp(mir_bb, bytes.clone()), len, None)
                } else if self.var_is_str_repr(bytes) || self.var_has_str_value_type(bytes) {
                    let (data, len) = self.emit_str_to_window(mir_bb, bytes.clone());
                    (data, len, Some(bytes.clone()))
                } else {
                    return Ok(false);
                }
            }
            [data, len] => (
                self.ensure_window_ccharp(mir_bb, data.clone()),
                len.clone(),
                None,
            ),
            _ => return Ok(false),
        };
        let Some(check_id) = self.rutf8_check_utf8_fun_decl_id() else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(check_id) else {
            return Ok(false);
        };
        let segments = registered_path_for_fun_decl(self.llbc, fd).segments;
        let output_ty = clone_tyref(&fd.signature.output);
        let Some((result_owner, _, _, _, _, _)) = self.resolve_result_owners(&output_ty) else {
            return Ok(false);
        };
        let niche = self.tyref_is_niche_option_ptr(dest_ty);
        let tagged_owners = if niche {
            None
        } else {
            match self.resolve_bool_then_option_dest(dest_ty) {
                Some(owners) => Some(owners),
                None => return Ok(false),
            }
        };
        let bb_id = self.emit_bb(mir_bb);
        let allow = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(allow.clone()),
            kind: OpKind::ConstBool(true),
        });
        // Window pair + `allow_surrogates=True`: `rutf8.py check_utf8`.
        let result_var = self.emit_fun_decl_call(
            bb_id,
            check_id,
            segments,
            vec![data.clone(), len.clone(), allow],
            ValueType::Ref(Some(result_owner.clone())),
        );
        let disc = emit_enum_disc_read(&mut self.graph, bb_id, result_var, &result_owner);
        let option_owner = tagged_owners.as_ref().map(|(o, _, _)| o.clone());
        let some_owner = tagged_owners.as_ref().map(|(_, s, _)| s.clone());
        let payload_ty = tagged_owners
            .as_ref()
            .map(|(_, _, t)| t.clone())
            .unwrap_or(ValueType::Str);
        let ok_from_window = ok_str.is_none();
        let ok_extras: Vec<Variable> = match ok_str {
            Some(header) => vec![header],
            None => vec![data, len],
        };
        // Result Ok=0 is the false arm of `bool(disc)`.
        self.close_call_diamond(
            mir_bb,
            target,
            dest_local,
            disc,
            niche,
            false,
            &[],
            &ok_extras,
            |this, err_bb, _| {
                if niche {
                    let cast = this.option_niche_null_cast(dest_ty);
                    this.graph.push_niche_null(err_bb, cast.as_ref())
                } else {
                    crate::front::option_variant::emit_option_variant(
                        &mut this.graph,
                        err_bb,
                        option_owner.as_ref().expect("tagged Option owner"),
                        0,
                        None,
                    )
                }
            },
            |this, ok_bb, mapped| {
                let header = if ok_from_window {
                    this.push_charpsize2str(ok_bb, mapped[0].clone(), mapped[1].clone())
                } else {
                    let header = mapped[0].clone();
                    stamp_as_rstr_str(&header);
                    header
                };
                if niche {
                    header
                } else {
                    crate::front::option_variant::emit_option_variant(
                        &mut this.graph,
                        ok_bb,
                        option_owner.as_ref().expect("tagged Option owner"),
                        1,
                        Some((
                            some_owner.as_ref().expect("tagged Some owner").as_str(),
                            header,
                            payload_ty.clone(),
                        )),
                    )
                }
            },
        )?;
        Ok(true)
    }

    /// `String::from_utf8_lossy` → `decode_utf8_with_errors(bytes, "replace")`
    /// then Ok payload as a plain string (`unicodehelper.py str_decode_utf8`
    /// replace / U+FFFD). Dest is not a Cow ADT.
    #[expect(
        clippy::too_many_arguments,
        reason = "Mirrors the other try_lower_* call intercepts"
    )]
    pub(in crate::front::mir) fn try_lower_from_utf8_lossy(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if args.len() != 1 || !self.is_string_from_utf8_lossy(reg, dest_ty) {
            return Ok(false);
        }
        let Some(decode_id) = self.decode_utf8_with_errors_fun_decl_id() else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(decode_id) else {
            return Ok(false);
        };
        let segments = registered_path_for_fun_decl(self.llbc, fd).segments;
        let output_ty = clone_tyref(&fd.signature.output);
        let Some((result_owner, ok_owner, _, _, _, _)) = self.resolve_result_owners(&output_ty)
        else {
            return Ok(false);
        };
        let bytes = args[0].clone();
        let bb_id = self.emit_bb(mir_bb);
        let replace = emit_str_const(&mut self.graph, bb_id, "replace");
        let result_var = self.emit_fun_decl_call(
            bb_id,
            decode_id,
            segments,
            vec![bytes, replace],
            ValueType::Ref(Some(result_owner.clone())),
        );
        let disc = emit_enum_disc_read(&mut self.graph, bb_id, result_var.clone(), &result_owner);
        // Result Ok=0 is the false arm of `bool(disc)`. Err is unreachable:
        // `unicodehelper.py str_decode_utf8` replace never returns Err, so
        // the true arm is `RaiseImplicit(AssertionError)` (`rtyper/debug.py
        // ll_assert`) and does not join.
        self.close_call_diamond(
            mir_bb,
            target,
            dest_local,
            disc,
            true,
            true,
            &[],
            std::slice::from_ref(&result_var),
            |_, _, _| unreachable!("unicodehelper.py str_decode_utf8 replace never returns Err"),
            |this, ok_bb, mapped| {
                let payload = emit_payload_read(
                    &mut this.graph,
                    ok_bb,
                    mapped[0].clone(),
                    &ok_owner,
                    ValueType::Str,
                );
                stamp_as_rstr_str(&payload);
                payload
            },
        )?;
        Ok(true)
    }

    /// `<str as ToString>::to_string` / `<String as ToString>::to_string`
    /// (`alloc::string::<Impl>::to_string`) — a `String` clone of a value
    /// that is already a string in the lifted model (`String`/`&str`/`str`
    /// all lower to the immutable rpy_string), so it is an identity.  The
    /// path alone cannot tell the string specialization from the blanket
    /// `impl<T: Display> ToString for T` (both monomorphize to the same
    /// `<Impl>::to_string`), so gate on the receiver type: fold only when
    /// the argument is itself string-family.  A non-string `to_string`
    /// (an integer's `ll_int2dec`) has a receiver outside the family and
    /// keeps its ordinary lowering.
    pub(in crate::front::mir) fn is_to_string_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        if !fd.item_meta.name_path().ends_with("::to_string") {
            return false;
        }
        first_arg_ty.is_some_and(|ty| tyref_is_string_family(ty, self.llbc))
    }

    pub(in crate::front::mir) fn is_to_string_path(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path().ends_with("::to_string"))
    }

    /// `Wtf8::to_wtf8_buf(&self) -> Wtf8Buf`, `<str as ToOwned>::to_owned
    /// (&self) -> String`, and a string-family `<_ as Clone>::clone(&self)`
    /// — an owned copy of a value that is already a string in the lifted
    /// model (`String`/`&str`/`str`/`Wtf8`/`Wtf8Buf` all lower to the
    /// immutable rpy_string), so it is an identity on the receiver, the same
    /// alias shape as `to_string`.  `clone` / `to_owned` are generic leaves
    /// shared by every `Clone` / `ToOwned` impl, so gate on the receiver
    /// being a string value (`tyref_is_string_value`): a `Vec`/`BigInt`/`[T]`
    /// (`<[T]>::to_owned -> Vec<T>`) receiver is outside the family and keeps
    /// its ordinary lowering.  A receiver-string `clone`/`to_owned`/
    /// `to_wtf8_buf` returns the same string family, so the receiver gate
    /// alone fixes the result value model; no destination check is needed.
    pub(in crate::front::mir) fn is_string_clone_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let np = fd.item_meta.name_path();
        let Some(leaf) = np.rsplit("::").next() else {
            return false;
        };
        if !matches!(leaf, "to_wtf8_buf" | "to_owned" | "clone") {
            return false;
        }
        first_arg_ty.is_some_and(|ty| tyref_is_string_family(ty, self.llbc))
    }

    /// `core::clone::impls::<Impl>::clone` on a Copy scalar or thin-pointer
    /// pointee is the value itself.  A trait `Clone::clone` on the same
    /// kinds, or on a zero-sized Copy type (`BuildHasherDefault<_>`,
    /// `PhantomData<_>`), is that same identity: `ll_dict_copy`
    /// (`rordereddict.py`) has no hasher object, so a Copy/ZST clone is
    /// the value.  A non-Copy clone (Vec, a named struct) keeps its
    /// ordinary residual call: the path is not enough.
    pub(in crate::front::mir) fn is_copy_scalar_or_thin_ptr_clone(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> bool {
        if !self.is_copy_or_trait_clone_call(reg) {
            return false;
        }
        self.clone_identity_pointee_node(reg, first_arg_ty)
            .is_some_and(|pointee| json_ty_is_copy_or_zst_clone_identity(pointee, self.llbc))
    }

    pub(in crate::front::mir) fn is_copy_or_trait_clone_call(&self, reg: &RegularCall) -> bool {
        match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => self.llbc.fn_by_id(*id).is_some_and(|fd| {
                is_core_clone_impls_clone_path(fd.item_meta.name_path().as_str())
            }),
            CallKind::Trait(v) => is_trait_clone_clone(v, self.llbc),
            _ => false,
        }
    }

    /// Pointee of `&self` for `Clone::clone`, or the `Self` type in a
    /// trait-ref `Clone` obligation.
    pub(in crate::front::mir) fn clone_identity_pointee_node<'l>(
        &'l self,
        reg: &'l RegularCall,
        first_arg_ty: Option<&'l TyRef>,
    ) -> Option<&'l serde_json::Value> {
        if let CallKind::Trait(v) = &reg.kind
            && let Some(node) = trait_clone_self_node(v, self.llbc)
        {
            return Some(node);
        }
        first_arg_ty.and_then(|ty| tyref_clone_pointee_node(ty, self.llbc))
    }

    pub(in crate::front::mir) fn trait_clone_receiver_ty(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> Option<TyRef> {
        let node = self.clone_identity_pointee_node(reg, first_arg_ty)?;
        serde_json::from_value(node.clone()).ok()
    }

    /// Trait `Clone::clone` of `Vec<T>` is `rlist.py` `ll_copy` (the `list`
    /// builtin `<[T]>::to_vec` retargets to) when T's clone is the identity
    /// copy — `ll_arraycopy` never calls `T::clone`.  A tuple of Copy/ZST
    /// and such `Vec` fields is field-wise clone plus the tuple Aggregate
    /// ctor — `KwargsDictStrategy.copy`
    /// `self.erase((keys_w[:], values_w[:]))` (`kwargsdict.py`).  Anything
    /// else stays residual.
    pub(in crate::front::mir) fn try_lower_trait_clone(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
        dest_local: usize,
        args: &[Variable],
        target: usize,
    ) -> Result<bool, LowerError> {
        let CallKind::Trait(_) = &reg.kind else {
            return Ok(false);
        };
        if !self.is_copy_or_trait_clone_call(reg) {
            return Ok(false);
        }
        let Some(recv) = args.first().cloned() else {
            return Ok(false);
        };
        let Some(self_ty) = self.trait_clone_receiver_ty(reg, first_arg_ty) else {
            return Ok(false);
        };
        if tyref_vec_clone_is_ll_copy(&self_ty, self.llbc) {
            self.emit_vec_list_clone(mir_bb, dest_local, recv, dest_ty, target)?;
            return Ok(true);
        }
        if tyref_is_tuple(&self_ty, self.llbc) && self.tuple_fields_are_copy_or_vec(&self_ty) {
            let tuple_ty = if tyref_is_tuple(dest_ty, self.llbc) {
                dest_ty
            } else {
                &self_ty
            };
            return self.emit_tuple_clone(mir_bb, dest_local, recv, tuple_ty, target);
        }
        Ok(false)
    }

    pub(in crate::front::mir) fn tuple_fields_are_copy_or_vec(&self, ty: &TyRef) -> bool {
        let mut idx = 0;
        loop {
            let Some(elem_ty) = tyref_tuple_element(ty, idx, self.llbc) else {
                return idx > 0;
            };
            if !self.ty_is_copy_or_vec_clone(&elem_ty) {
                return false;
            }
            idx += 1;
        }
    }

    pub(in crate::front::mir) fn ty_is_copy_or_vec_clone(&self, ty: &TyRef) -> bool {
        if tyref_vec_clone_is_ll_copy(ty, self.llbc) {
            return true;
        }
        tyref_node(ty, self.llbc)
            .and_then(|node| strip_ty_indirections(node, self.llbc))
            .is_some_and(|node| json_ty_is_copy_or_zst_clone_identity(node, self.llbc))
    }

    pub(in crate::front::mir) fn emit_vec_list_clone(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        recv: Variable,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<(), LowerError> {
        let res = self.emit_list_copy(mir_bb, recv, dest_ty);
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    pub(in crate::front::mir) fn emit_list_copy(
        &mut self,
        mir_bb: usize,
        recv: Variable,
        dest_ty: &TyRef,
    ) -> Variable {
        let result_ty = tyref_to_value_type_with(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        self.emit_path_call(mir_bb, "list", vec![recv], result_ty)
    }

    pub(in crate::front::mir) fn emit_tuple_field_read(
        &mut self,
        mir_bb: usize,
        base: Variable,
        tuple_ty: &TyRef,
        idx: usize,
        elem_ty: &TyRef,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let owner = tyref_positional_aggregate_root(tuple_ty, self.llbc)
            .unwrap_or_else(|| format!("Tuple{}", tyref_tuple_suffix(tuple_ty, self.llbc)));
        let ty = tuple_element_field_value_type(
            elem_ty,
            Some(clone_tyref(elem_ty)),
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new(format!("__pos_{idx}"), Some(owner)),
                ty,
                pure: false,
            },
        });
        res
    }

    pub(in crate::front::mir) fn emit_tuple_clone(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        recv: Variable,
        tuple_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let mut cloned = Vec::new();
        let mut elem_tys = Vec::new();
        let mut idx = 0;
        loop {
            let Some(elem_ty) = tyref_tuple_element(tuple_ty, idx, self.llbc) else {
                break;
            };
            let field = self.emit_tuple_field_read(mir_bb, recv.clone(), tuple_ty, idx, &elem_ty);
            let value = if tyref_is_vec_value(&elem_ty, self.llbc) {
                self.emit_list_copy(mir_bb, field, &elem_ty)
            } else {
                field
            };
            cloned.push(value);
            elem_tys.push(elem_ty);
            idx += 1;
        }
        if cloned.is_empty() {
            return Ok(false);
        }
        let kind = serde_json::json!({ "Tuple": serde_json::Value::Null });
        let Ok(shape) = self.aggregate_shape(&kind, tuple_ty, cloned.len()) else {
            return Ok(false);
        };
        let AggregateShape::Ctor(ctor) = shape else {
            return Ok(false);
        };
        let bb_id = self.emit_bb(mir_bb);
        let tuple = self.emit_tuple_ctor_with_typed_fields(bb_id, &ctor, &cloned, &elem_tys);
        self.local_var[dest_local] = Some(LocalValue::One(tuple));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `core::default::<Impl>::default` / `core::ptr::mut_ptr::<Impl>::default`
    /// for a scalar or thin-pointer destination: the zero of that kind.
    pub(in crate::front::mir) fn is_scalar_or_ptr_default(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        is_core_default_path(fd.item_meta.name_path().as_str())
            && tyref_is_copy_scalar_or_thin_ptr(dest_ty, self.llbc)
    }

    /// `core::tuple::<Impl>::default` whose destination is a non-unit
    /// tuple whose every element has a Default the front end already
    /// lowers (primitive scalar zero, unit / `PhantomData`, empty `Vec`,
    /// nested tuple).
    pub(in crate::front::mir) fn is_tuple_default(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        is_core_tuple_default_path(fd.item_meta.name_path().as_str())
            && tyref_is_tuple(dest_ty, self.llbc)
            && self.can_emit_default_of_ty(dest_ty)
    }

    /// `CallKind::Trait` naming `core::default::Default::default`. Distinct
    /// from a Fun Regular path of the same spelling (the lookalike).
    pub(in crate::front::mir) fn is_trait_default_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Trait(v) = &reg.kind else {
            return false;
        };
        trait_payload_path_and_method(v, self.llbc).is_some_and(|(trait_path, method_leaf)| {
            trait_path == "core::default::Default" && method_leaf == "default"
        })
    }

    /// `CallKind::Trait` naming `core::ops::function::FnOnce::call_once` /
    /// `Fn::call` / `FnMut::call_mut`. The generated shim for a function item.
    pub(in crate::front::mir) fn is_fn_family_call_once(&self, reg: &RegularCall) -> bool {
        let CallKind::Trait(v) = &reg.kind else {
            return false;
        };
        trait_payload_path_and_method(v, self.llbc).is_some_and(|(trait_path, method_leaf)| {
            matches!(
                (trait_path.as_str(), method_leaf.as_str()),
                ("core::ops::function::FnOnce", "call_once")
                    | ("core::ops::function::Fn", "call")
                    | ("core::ops::function::FnMut", "call_mut")
            )
        })
    }

    /// rustc's `FnOnce`/`Fn`/`FnMut` shim for a function item unpacks
    /// `Args` and tail-calls the `FnDef`. Rewrite that trait call into
    /// the ordinary `Fun` call so argument conversion runs on the named
    /// function's own inputs. A closure ADT, an arity mismatch, a const
    /// non-unit tuple, or unreadable generics declines and the trait
    /// call stays as it is.
    pub(in crate::front::mir) fn fn_item_call_once_direct_call(
        &self,
        call: &CallPayload,
    ) -> Option<CallPayload> {
        let CallFunc::Regular(reg) = &call.func else {
            return None;
        };
        if !self.is_fn_family_call_once(reg) {
            return None;
        }
        if call.args.len() != 2 {
            return None;
        }
        let fn_item = operand_fn_item_call(self.llbc, call.args.first())?;
        let CallKind::Fun(FunId::Regular { id }) = &fn_item.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        if fn_item.generics.as_object().is_none() {
            return None;
        }
        let n = fd.signature.inputs.len();
        let args = if n == 0 {
            let unit = match &call.args[1] {
                Operand::Copy(p) | Operand::Move(p) => is_unit_type(&p.ty, self.llbc),
                Operand::Const(_) => const_operand_tyref(&call.args[1], self.llbc)
                    .is_some_and(|ty| is_unit_type(&ty, self.llbc)),
            };
            if !unit {
                return None;
            }
            Vec::new()
        } else {
            let (is_move, place) = match &call.args[1] {
                Operand::Copy(p) => (false, p),
                Operand::Move(p) => (true, p),
                Operand::Const(_) => return None,
            };
            if tyref_tuple_arity(&place.ty, self.llbc) != n {
                return None;
            }
            let mut args = Vec::with_capacity(n);
            for i in 0..n {
                let ty = tyref_tuple_element(&place.ty, i, self.llbc)?;
                // Extracted LLBC spells a variant-less tuple field as
                // `"Field":[null, idx]` (`build/llbc/majit-rlib.ullbc`).
                let field = Place {
                    kind: PlaceKind::Projection(
                        Box::new(place.clone()),
                        ProjectionElem::Tagged(serde_json::json!({
                            "Field": [serde_json::Value::Null, i]
                        })),
                    ),
                    ty,
                };
                args.push(if is_move {
                    Operand::Move(field)
                } else {
                    Operand::Copy(field)
                });
            }
            args
        };
        Some(CallPayload {
            func: CallFunc::Regular(fn_item),
            args,
            dest: call.dest.clone(),
        })
    }

    /// True when [`Self::emit_default_of_ty`] can build `ty` without a
    /// residual `Default` call. Checked before any op is pushed so a
    /// later unknown element does not leave a half-built tuple in the
    /// block.
    pub(in crate::front::mir) fn can_emit_default_of_ty(&self, ty: &TyRef) -> bool {
        if tyref_is_tuple(ty, self.llbc) {
            let mut idx = 0;
            loop {
                let Some(elem_ty) = tyref_tuple_element(ty, idx, self.llbc) else {
                    return idx > 0;
                };
                if !self.can_emit_default_of_ty(&elem_ty) {
                    return false;
                }
                idx += 1;
            }
        }
        // A primitive scalar's default is zero. A thin pointer is not
        // accepted: `Box<T>::default` allocates a `T::default()`, not null.
        // A rust-heap `String` is `{cap, ptr, len}` and is not the empty
        // rstr `STR` this path emits.
        tyref_default_is_unit(ty, self.llbc)
            || tyref_is_primitive_scalar(ty, self.llbc)
            || tyref_is_vec_value(ty, self.llbc)
            || tyref_is_string_value(ty, self.llbc)
    }

    /// Default of `ty`: primitive scalar zero, unit / `PhantomData`,
    /// empty `Vec` (`vec::Vec::new` → `newlist` / `ll_newemptylist`),
    /// empty rstr `STR` (`Wtf8` / `Wtf8Buf` / `str`), or
    /// `TupleRepr.newtuple` of those.
    pub(in crate::front::mir) fn emit_default_of_ty(
        &mut self,
        bb_id: BlockId,
        ty: &TyRef,
    ) -> Option<Variable> {
        if tyref_is_tuple(ty, self.llbc) {
            return self.emit_tuple_of_element_defaults(bb_id, ty);
        }
        if tyref_default_is_unit(ty, self.llbc) {
            return Some(self.emit_unit(bb_id));
        }
        if tyref_is_primitive_scalar(ty, self.llbc) {
            return self.emit_zero_constant_of_ty(bb_id, ty);
        }
        if tyref_is_vec_value(ty, self.llbc) {
            return Some(self.emit_vec_new(bb_id, ty));
        }
        if tyref_is_string_value(ty, self.llbc) {
            return Some(emit_str_const(&mut self.graph, bb_id, ""));
        }
        None
    }

    pub(in crate::front::mir) fn emit_tuple_of_element_defaults(
        &mut self,
        bb_id: BlockId,
        ty: &TyRef,
    ) -> Option<Variable> {
        let mut args = Vec::new();
        let mut elem_tys = Vec::new();
        let mut idx = 0;
        loop {
            let Some(elem_ty) = tyref_tuple_element(ty, idx, self.llbc) else {
                break;
            };
            args.push(self.emit_default_of_ty(bb_id, &elem_ty)?);
            elem_tys.push(elem_ty);
            idx += 1;
        }
        if args.is_empty() {
            return None;
        }
        // `TupleRepr.newtuple`: malloc of the tuple GcStruct plus one
        // setfield per item. The frontend spelling is the same
        // SyntheticTransparentCtor + FieldWrite chain a Charon tuple
        // Aggregate already uses. Raw `OpKind::NewTuple` is the
        // high-level `space.newtuple` op; when a graph still drops to
        // the legacy walker it reaches the assembler as
        // `newtuple/<kinds>>r`, and the production blackhole only
        // wires `newtuple/rr>r`.
        let kind = serde_json::json!({ "Tuple": serde_json::Value::Null });
        let shape = self.aggregate_shape(&kind, ty, args.len()).ok()?;
        let AggregateShape::Ctor(ctor) = shape else {
            return None;
        };
        Some(self.emit_tuple_ctor_with_typed_fields(bb_id, &ctor, &args, &elem_tys))
    }

    /// `emit_aggregate_ctor` plus per-item `FieldWrite.ty` from the
    /// element type. Tuple items may be Int (a `Vec` header) rather
    /// than Ref; a hardcoded Ref write would emit `setfield_gc_r` of
    /// an integer word.
    pub(in crate::front::mir) fn emit_tuple_ctor_with_typed_fields(
        &mut self,
        bb_id: BlockId,
        ctor: &AggregateCtor,
        args: &[Variable],
        elem_tys: &[TyRef],
    ) -> Variable {
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: ctor.target.clone(),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(ctor.result_owner.clone())),
            },
        });
        for (i, field) in ctor.fields.iter().enumerate() {
            if field.void {
                continue;
            }
            let Some(value) = args.get(i) else {
                continue;
            };
            let mut ty = elem_tys
                .get(i)
                .map(|elem_ty| {
                    tyref_to_value_type_with(
                        elem_ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    )
                })
                .unwrap_or(ValueType::Ref(None));
            if matches!(ty, ValueType::Void) {
                continue;
            }
            ty = crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                &ctor.result_owner,
                &field.name,
                ty,
            );
            let value = match &field.narrow_root {
                Some(root) => {
                    let narrowed = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(narrowed.clone()),
                        kind: crate::model::cast_instance_call(root, value.clone()),
                    });
                    narrowed
                }
                None => value.clone(),
            };
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: res.clone(),
                    field: crate::model::FieldDescriptor {
                        name: field.name.clone(),
                        owner_root: Some(ctor.result_owner.clone()),
                        owner_id: ctor.owner_id,
                        // Built by value: the store is not a dereference of a
                        // live container (`base_is_local_aggregate`).
                        base_is_deref: Some(false),
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: crate::model::LinkArg::Value(value),
                    ty,
                },
            });
        }
        res
    }

    /// Empty `Vec::new()`, with the constructor an ordinary call selects
    /// (`rust_vec_std_call`): a `Vec` of one-word items is the
    /// `RustVecRepr` raw header, built by its `ll_vec_*` `NewEmpty`
    /// helper; any other `Vec` keeps `vec::Vec::new`, which
    /// `flowspace_adapter` rewrites to `newlist()` / `ll_newemptylist`.
    /// A raw `NewList` in a graph that still drops to the legacy walker
    /// has no blackhole handler (`newlist/r>r`).
    pub(in crate::front::mir) fn emit_vec_new(
        &mut self,
        bb_id: BlockId,
        dest_ty: &TyRef,
    ) -> Variable {
        let vec_new: Vec<String> = ["vec", "Vec", "new"].map(str::to_string).into();
        let segments = match rust_vec_std_call(&vec_new, 0, dest_ty, None, self.llbc) {
            Some((op, kind)) => rust_vec_helper_segments(op, kind),
            None => vec_new,
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path(segments),
                args: crate::model::call_args(vec![]),
                result_ty: tyref_to_value_type_with(
                    dest_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
            },
        });
        res
    }

    pub(in crate::front::mir) fn is_slice_iter_copied(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
    ) -> bool {
        let Ok((emitted, _)) = self.call_target_segments(mir_bb, reg, None) else {
            return false;
        };
        emitted.as_slice() == ["slice", "iter", "Iter", "copied"]
    }

    pub(in crate::front::mir) fn is_slice_iter_position(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
    ) -> bool {
        let Ok((emitted, _)) = self.call_target_segments(mir_bb, reg, None) else {
            return false;
        };
        emitted.as_slice() == ["slice", "iter", "Iter", "position"]
            || (emitted.last().map(String::as_str) == Some("position")
                && emitted.iter().any(|s| s == "Iter" || s == "iterator"))
    }

    /// Haystack STR of a `["core","slice","iter"]` iterator that was
    /// copied through `ll_str2uintlist`, or whose receiver is already a
    /// STR view. A STR-view `iter().position` look-ahead aliases the
    /// iterator local to the STR itself.
    pub(in crate::front::mir) fn str_origin_of_slice_iter(
        &self,
        iter: &Variable,
    ) -> Option<Variable> {
        if self.var_is_str_repr(iter) {
            return Some(iter.clone());
        }
        let recv = self.slice_iter_receiver(iter)?;
        if let Some(s) = self.str_from_uintlist_copy(&recv) {
            return Some(s);
        }
        self.var_is_str_repr(&recv).then_some(recv)
    }

    /// Successor terminator is `Iter::position` with a `|&b| b == c`
    /// predicate, so the iterator is never walked.
    pub(in crate::front::mir) fn mir_target_is_const_eq_iter_position(
        &self,
        target: usize,
    ) -> bool {
        let Some(bb) = self.body.body.get(target) else {
            return false;
        };
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(self.llbc) else {
            return false;
        };
        let CallFunc::Regular(reg) = &call.func else {
            return false;
        };
        if call.args.len() != 2 || !self.is_slice_iter_position(target, reg) {
            return false;
        }
        let pred_ty = match call.args.get(1) {
            Some(Operand::Copy(p) | Operand::Move(p)) => clone_tyref(&p.ty),
            Some(op @ Operand::Const(_)) => {
                let Some(ty) = const_operand_tyref(op, self.llbc) else {
                    return false;
                };
                ty
            }
            _ => return false,
        };
        self.const_eq_u8_of_predicate(&pred_ty).is_some()
    }

    pub(in crate::front::mir) fn slice_iter_receiver(&self, iter: &Variable) -> Option<Variable> {
        let op = self.producer_op(iter)?;
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op.kind
        else {
            return None;
        };
        if segments.as_slice() != ["core", "slice", "iter"] || args.len() != 1 {
            return None;
        }
        args[0].as_variable().cloned()
    }

    pub(in crate::front::mir) fn str_from_uintlist_copy(&self, var: &Variable) -> Option<Variable> {
        let op = self.producer_op(var)?;
        let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op.kind
        else {
            return None;
        };
        if segments.as_slice() != [crate::front::join_chars::LL_STR2UINTLIST] || args.len() != 1 {
            return None;
        }
        args[0].as_variable().cloned()
    }

    pub(in crate::front::mir) fn producer_op(&self, var: &Variable) -> Option<&SpaceOperation> {
        let (block_id, idx) = resolve_to_producer_op(&self.graph, var)?;
        self.graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .and_then(|block| block.operations.get(idx))
    }

    /// Constant byte a predicate closure compares its item against
    /// (`|&byte| byte == c`). The needle lives in the closure FunDecl
    /// body; captures are empty.
    pub(in crate::front::mir) fn const_eq_u8_of_predicate(&self, pred_ty: &TyRef) -> Option<u8> {
        let def_id = self
            .tyref_adt_def_id(pred_ty)
            .or_else(|| self.tyref_ref_adt_def_id(pred_ty))?;
        let td = self.llbc.type_by_id(def_id)?;
        if !type_decl_is_closure_env(td) {
            return None;
        }
        for fd in self.llbc.iter_fun_decls() {
            if !fd.has_unstructured_body() {
                continue;
            }
            let Some(first) = fd.signature.inputs.first() else {
                continue;
            };
            let first_id = self
                .tyref_adt_def_id(first)
                .or_else(|| self.tyref_ref_adt_def_id(first));
            if first_id != Some(def_id) {
                continue;
            }
            let Some(body) = fd.unstructured() else {
                continue;
            };
            if let Some(byte) = const_eq_u8_in_unstructured(self.llbc, &body) {
                return Some(byte);
            }
        }
        None
    }

    /// `iter().position(|&b| b == c)` of a STR-view slice iterator is
    /// `str.find(c)` (`rpython_str_find_char`) wrapped as `Option<usize>`.
    pub(in crate::front::mir) fn try_lower_str_view_iter_position(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
        pred_ty: Option<&TyRef>,
    ) -> Result<bool, LowerError> {
        if args.len() != 2 || !self.is_slice_iter_position(mir_bb, reg) {
            return Ok(false);
        }
        let Some(haystack) = self.str_origin_of_slice_iter(&args[0]) else {
            return Ok(false);
        };
        let Some(pred_ty) = pred_ty else {
            return Ok(false);
        };
        let Some(byte) = self.const_eq_u8_of_predicate(pred_ty) else {
            return Ok(false);
        };
        let Some((option_owner, some_owner, payload_ty)) =
            self.resolve_bool_then_option_dest(dest_ty)
        else {
            return Ok(false);
        };
        let bb_id = self.emit_bb(mir_bb);
        // Needle is a 1-char STR. `emit_str_const` is the `__str_const`
        // define the adapter pre-folds to `Constant('text')`; a raw
        // `ConstStr` is skipped (`translate_op`) without that fold, so
        // `find`'s needle is used before definition.
        let needle = emit_str_const(&mut self.graph, bb_id, &char::from(byte).to_string());
        let start = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(start.clone()),
            kind: OpKind::ConstInt(0),
        });
        let idx = crate::front::str_find::emit_rpython_str_find_char(
            &mut self.graph,
            bb_id,
            &[haystack, needle, start],
        )
        .map_err(LowerError::Unsupported)?;
        let neg1 = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(neg1.clone()),
            kind: OpKind::ConstInt(-1),
        });
        let found = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(found.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: idx.clone(),
                rhs: neg1,
                result_ty: ValueType::Bool,
            },
        });
        let option_owner_some = option_owner.clone();
        self.close_call_diamond(
            mir_bb,
            target,
            dest_local,
            found,
            false,
            false,
            std::slice::from_ref(&idx),
            &[],
            move |this, some_bb, mapped| {
                crate::front::option_variant::emit_option_variant(
                    &mut this.graph,
                    some_bb,
                    &option_owner_some,
                    1,
                    Some((some_owner.as_str(), mapped[0].clone(), payload_ty.clone())),
                )
            },
            move |this, none_bb, _| {
                crate::front::option_variant::emit_option_variant(
                    &mut this.graph,
                    none_bb,
                    &option_owner,
                    0,
                    None,
                )
            },
        )?;
        Ok(true)
    }

    pub(in crate::front::mir) fn is_const_ptr_as_ref(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let segs: Vec<&str> = path.split("::").collect();
        matches!(
            segs.as_slice(),
            ["core", "ptr", "const_ptr", "<Impl>", "as_ref"]
        )
    }

    /// Zero-argument std `default` whose destination is a void ZST.
    ///
    /// `name_path` spells the impl as `core::hash::<Impl>::default`. The
    /// layout size, not the `Self` name, is the gate: a zero-sized
    /// `core`/`std`/`alloc` `default` is `lltype.Void`
    /// (`ObjectDictStrategy.get_empty_storage`). A `Default` with a
    /// payload stays a call, and a pyre `default` stays a call.
    pub(in crate::front::mir) fn is_void_zst_std_default(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        if !tyref_is_void_zst(dest_ty, self.llbc)
            || tyref_keeps_empty_class_instance(dest_ty, self.llbc, self.tombstoned_leaves)
        {
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let root = path.split("::").next().unwrap_or("");
        path.rsplit("::").next() == Some("default") && matches!(root, "core" | "std" | "alloc")
    }

    /// `<*mut T>::add` / `<*const T>::add` / `::sub` when the pointee has a
    /// known byte size, so the count can be scaled to a byte offset before
    /// `lltype.direct_ptradd`.
    pub(in crate::front::mir) fn ptr_add_pointee_size(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> Option<i64> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        let path = fd.item_meta.name_path();
        if !is_core_ptr_add_path(path.as_str()) && !is_core_ptr_sub_path(path.as_str()) {
            return None;
        }
        let pointee = first_arg_ty.and_then(|ty| {
            tyref_peel_one_raw_ptr_node(ty, self.llbc)
                .or_else(|| tyref_peel_one_ref_node(ty, self.llbc))
        })?;
        json_ty_byte_size(pointee, self.llbc)
    }

    pub(in crate::front::mir) fn ptr_offset_is_sub(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| is_core_ptr_sub_path(fd.item_meta.name_path().as_str()))
    }

    /// `(header as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)` — `s.chars`
    /// (`rstr.py` `STR`). The add is on a `u8` pointer, the peeled base is
    /// the low-level string header (`*const Utf8Str` or `&Utf8Str`), and
    /// the dest is a `u8` pointer. `sub` and any other pointee keep
    /// `rtype_direct_ptradd`.
    pub(in crate::front::mir) fn is_string_header_chars_byte_ptr_add(
        &self,
        reg: &RegularCall,
        arg_locals: &[Option<usize>],
        arg_operands: &[Operand],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> bool {
        if self.ptr_offset_is_sub(reg) {
            return false;
        }
        if !tyref_is_raw_u8_ptr(dest_ty, self.llbc) {
            return false;
        }
        if !first_arg_ty.is_some_and(|ty| tyref_is_raw_u8_ptr(ty, self.llbc)) {
            return false;
        }
        if !arg_operands
            .get(1)
            .is_some_and(|op| operand_is_lowlevel_string_chars_offset(self.body, self.llbc, op))
        {
            return false;
        }
        arg_locals
            .first()
            .copied()
            .flatten()
            .and_then(|local| peel_copy_and_cast(self.llbc, self.body, local))
            .and_then(|base| self.body.locals.locals.get(base))
            .is_some_and(|decl| tyref_is_lowlevel_string_header_ptr(&decl.ty, self.llbc))
    }

    /// Later intercepts in this same `RegularCall` match already lower
    /// some `.add` calls: items-base accessors alias to the header
    /// (brick 1), string-array remove aliases the interior pointer,
    /// and a `.add` consumed by one deref becomes getarrayitem
    /// (brick 3).  Those must run instead of `direct_ptradd`.
    pub(in crate::front::mir) fn ptr_add_has_later_intercept(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> bool {
        (args_len == 2 && self.is_items_block_base_ptr_add(reg))
            || (args_len == 2
                && self.string_array_remove_owner().is_some()
                && regular_call_is_ptr_add(reg, self.llbc))
            || self
                .string_items_elem_ptr_add(reg, args_len, arg_locals, first_arg_ty, dest_local)
                .is_some()
            || self.is_list_items_elem_ptr_add(reg, args_len, arg_locals, first_arg_ty, dest_local)
            || self
                .typed_items_elem_ptr_add(reg, args_len, arg_locals, first_arg_ty, dest_local)
                .is_some()
            || self
                .u8_slice_elem_ptr_add(reg, args_len, arg_locals, first_arg_ty, dest_local)
                .is_some()
    }

    /// `core::ptr::write` of a Copy scalar or thin pointer.  An aggregate
    /// write stays a residual call so `lower_struct_ptr_writes` can still
    /// turn a constructor-shaped operand into field stores.
    pub(in crate::front::mir) fn ptr_write_store_descr(
        &self,
        reg: &RegularCall,
        value_ty: Option<&TyRef>,
    ) -> Option<(ValueType, usize, bool)> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        if !is_core_ptr_write_path(fd.item_meta.name_path().as_str()) {
            return None;
        }
        let ty = value_ty?;
        if !tyref_is_copy_scalar_or_thin_ptr(ty, self.llbc) {
            return None;
        }
        json_ty_raw_store_descr(tyref_node(ty, self.llbc)?, self.llbc)
    }

    pub(in crate::front::mir) fn emit_zero_constant_of_ty(
        &mut self,
        bb_id: BlockId,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        if !tyref_is_copy_scalar_or_thin_ptr(dest_ty, self.llbc) {
            return None;
        }
        let node = tyref_node(dest_ty, self.llbc)
            .and_then(|node| strip_ty_indirections(node, self.llbc))?;
        // `T::default` for a thin pointer is otherwise a GCREF
        // `ConstRefNull`. `fn` / `Option<fn>` are `Ptr(FuncType)`, and
        // `history.getkind` puts that in the int bank — the same answer
        // [`tyref_to_value_type_with`] gives. A ref null copied into that
        // inputarg is the `Move` from Ref to Int.
        let value_ty = tyref_to_value_type_with(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        if matches!(value_ty, ValueType::Int)
            && (type_node_is_fn_ptr(node, self.llbc)
                || tyref_option_payload_is_fn_ptr(dest_ty, self.llbc))
        {
            return Some(self.graph.push_null_fn_ptr_with_spelling(
                bb_id,
                tyref_fn_ptr_spelling(dest_ty, self.llbc).as_deref(),
            ));
        }
        let kind = if json_ty_is_thin_pointer_element(node, self.llbc)
            && !matches!(value_ty, ValueType::Int | ValueType::Unsigned)
        {
            OpKind::ConstRefNull
        } else {
            match value_ty {
                ValueType::Int => OpKind::ConstInt(0),
                ValueType::Unsigned => OpKind::ConstUInt(0),
                ValueType::Bool => OpKind::ConstBool(false),
                ValueType::Float => OpKind::ConstFloat(0),
                ValueType::SingleFloat => OpKind::ConstSingleFloat(0),
                ValueType::Int128 => OpKind::ConstInt128(0),
                ValueType::UInt128 => OpKind::ConstUInt128(0),
                ValueType::Ref(_) | ValueType::Str => OpKind::ConstRefNull,
                ValueType::Void => OpKind::ConstNone,
                ValueType::StringBuilder | ValueType::State | ValueType::Unknown => {
                    return None;
                }
            }
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind,
        });
        Some(res)
    }

    /// `pyre_object::gc_storage::gc_alloc_storage_box::<Wtf8Buf>` is the
    /// physical Rust owner for a value that the translated model already
    /// represents as one RPython string.  Admit only the exact string-family
    /// input and exactly-one-raw-pointer string-family result; other storage
    /// boxes retain their allocation body.
    pub(in crate::front::mir) fn is_string_storage_box_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        fd.item_meta.name_path() == "pyre_object::gc_storage::gc_alloc_storage_box"
            && first_arg_ty.is_some_and(|ty| tyref_is_string_value(ty, self.llbc))
            && tyref_raw_ptr_pointee_is_string_value(dest_ty, self.llbc)
    }

    /// `Option::as_ref` / `Option::<&T>::copied` /
    /// `Option::<T: Clone>::cloned` — changes only whether a pointer-niche
    /// payload is borrowed or read through that borrow.  In the lifted value
    /// model a `&T` and a pointer-shaped `T` are the same one GC pointer word
    /// (`slice_first`'s payload note: the front has no pointer-to-slot op, so a
    /// `.first()` payload is already the element VALUE, and `Option<&T>` /
    /// `Option<T>` share the niche discriminant + one-word representation), so
    /// all three are identities on the receiver.  All three are gated to a
    /// pointer-niche receiver and destination: an arbitrary `Option<T>` can be
    /// wider than `Option<&T>` and must keep its real projection.  A nominal
    /// `Copy`/`Clone` struct is what separates the two for `copied` / `cloned`
    /// -- `Option<&Foo>` is one nullable word while `Option<Foo>` is a
    /// discriminant plus the fields, so aliasing the destination to the
    /// receiver would have later projections read the payload at the wrong
    /// representation.  Without the intercept the call keeps a
    /// `CallTarget::Method` (`as_ref`/`copied`/`cloned`) getattr the rtyper
    /// cannot route on the classdef-less `Option` receiver (the `getattr
    /// (opt, "copied")` census wall — e.g. `_codecs::lookup_codec`'s
    /// `args.first().copied()`).  Bind the destination to the receiver
    /// instead, the same alias shape as `to_string` / `NonNull::as_ptr`.
    ///
    /// Gated on BOTH the receiver and the destination: a slice/iterator
    /// `.copied()` or a `Vec::clone` — a real copy, not an identity — has a
    /// non-`Option` receiver/dest, which the niche predicate rejects on its
    /// first line, and keeps its ordinary lowering.  (`Option::clone` on a
    /// `Copy` payload is itself an identity, so the `cloned` arm is sound for
    /// the `Option` family.)  The receiver is peeled first because `as_ref`
    /// takes `&Option<T>` while `copied` / `cloned` take the `Option<&T>` by
    /// value; [`Self::tyref_peel_ref_to_pointee`] returns a non-reference
    /// unchanged.
    pub(in crate::front::mir) fn is_option_value_identity(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let np = fd.item_meta.name_path();
        let Some(leaf) = np.rsplit("::").next() else {
            return false;
        };
        if !matches!(leaf, "as_ref" | "copied" | "cloned") {
            return false;
        }
        let receiver_is_option = first_arg_ty.is_some_and(|ty| {
            crate::front::result_exc::tyref_is_option(ty, self.llbc)
                || crate::front::result_exc::tyref_is_option_ref(ty, self.llbc)
        });
        if !receiver_is_option {
            return false;
        }
        let Some(receiver_ty) = first_arg_ty.and_then(|ty| self.tyref_peel_ref_to_pointee(ty))
        else {
            return false;
        };
        self.tyref_is_niche_option_ptr(&receiver_ty) && self.tyref_is_niche_option_ptr(dest_ty)
    }

    /// `alloc::fmt::format(args)` (the `format!` macro's String producer)
    /// — the call whose `format_args!` argument the #277 recognizer
    /// back-traces.  `write!`/`writeln!` lower to `fmt::Write::write_fmt`
    /// (a `Result`), never `fmt::format`, so the two-segment tail keeps
    /// the match precise.
    pub(in crate::front::mir) fn is_fmt_format_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let np = fd.item_meta.name_path();
            let segs: Vec<String> = np.split("::").map(str::to_string).collect();
            fmt_path_ends_with(&segs, &["fmt", "format"])
        })
    }

    /// `fmt::Arguments::from_str(s)` — the no-placeholder `format_args!`
    /// constructor a constant message lowers to (`debug_assert!` /
    /// `assert!` / `panic!` with a literal, and `format!("literal")`).
    /// Its sole argument is the `&'static str` literal itself (an already
    /// string-family value in the lifted model), so the `Arguments` is the
    /// constant string: the caller aliases the destination to that operand
    /// instead of emitting the unregistered ctor.  The result then flows
    /// either to a residualized panic (`Abort` → `RaiseImplicit`) or to
    /// `alloc::fmt::format` (handled by [`Self::traces_to_str_const`]).
    ///
    /// `from_str_nonconst(s) { Arguments::from_str(s) }` is the identical
    /// constructor the compiler emits when the no-placeholder `format_args!`
    /// sits in a non-`const` body (the `#[new]`/setter gateways) — same
    /// `&'static str` argument, same constant-string `Arguments` — so it
    /// takes the same alias.
    pub(in crate::front::mir) fn is_arguments_from_str(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        // `from_str` is an inherent-impl associated function on
        // `core::fmt::Arguments`, so its `name_path()` carries an `<Impl>`
        // placeholder, not the owner name — resolve the owner the same way
        // `call_target_segments` spells the `["fmt", "Arguments",
        // "from_str"]` `FunctionPath` (via `impl_method_owner_for_fundecl`).
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            impl_method_owner_for_fundecl(self.llbc, fd).is_some_and(|(owner, leaf)| {
                (leaf == "from_str" || leaf == "from_str_nonconst")
                    && owner.ends_with("fmt::Arguments")
            })
        })
    }

    /// Whether `var` is produced by a `__str_const` synthetic call (a
    /// string literal — see [`Lowering::emit_constant`]), following the
    /// cross-block back-trace.  Used to recognize `alloc::fmt::format`
    /// applied to a constant message (a folded `Arguments::from_str`),
    /// whose render is the constant itself.
    pub(in crate::front::mir) fn traces_to_str_const(&self, var: &Variable) -> bool {
        use crate::model::{CallTarget, OpKind};
        resolve_to_producer_op(&self.graph, var)
            .and_then(|(b, i)| {
                self.graph
                    .blocks
                    .iter()
                    .find(|blk| blk.id == b)
                    .and_then(|blk| blk.operations.get(i))
                    .map(|op| &op.kind)
            })
            .is_some_and(|kind| {
                matches!(
                    kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.first().map(String::as_str) == Some("__str_const")
                )
            })
    }

    /// The blanket `impl<I: Iterator> IntoIterator for I`
    /// (`core::iter::traits::collect`) — its `into_iter(self) -> I`
    /// returns the receiver unchanged, so a `for` desugar's `into_iter`
    /// callsite aliases its argument instead of calling core's identity
    /// body (an unregistered callee).  Container impls
    /// (`Vec`/array/`Range`) live under other module paths, so the exact
    /// path match selects only the reflexive blanket.
    pub(in crate::front::mir) fn is_reflexive_into_iter(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "core::iter::traits::collect::<Impl>::into_iter"
        })
    }

    /// The concrete container `IntoIterator` impls — `<&[T] as
    /// IntoIterator>::into_iter` and the `Vec` / array forms.  Unlike the
    /// reflexive blanket (`is_reflexive_into_iter`), the receiver type
    /// (the container) differs from the destination (a fresh iterator), so
    /// they cannot be identity-aliased; the caller lowers them to the
    /// `iter` operation on the container instead of an unregistered
    /// `FunctionPath` callee.  `<[T]>::iter` / `iter_mut` take the same
    /// bridge with `fun_decl_id: None`: the FunDecl input is unsubstituted
    /// `&[T]`, which would paint a STR-view receiver as List.
    pub(in crate::front::mir) fn is_concrete_iter_constructor(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::iter::<Impl>::into_iter"
                    | "alloc::vec::<Impl>::into_iter"
                    // `core::array::<Impl>` is the `&[T; N]` by-reference
                    // `IntoIterator` (delegates to slice iter); `core::array::
                    // iter::<Impl>` is the by-value `[T; N]` form
                    // (`core/src/array/iter.rs`, yielding owned `T`). Both walk
                    // the same element sequence — an identity on the list model.
                    | "core::array::<Impl>::into_iter"
                    | "core::array::iter::<Impl>::into_iter"
                    // `alloc/src/boxed/iter.rs` — the `Box<[T]>` forms, by
                    // value and by reference; the latter delegates to slice
                    // iter. A boxed slice is the same element sequence as the
                    // slice it owns (`project_struct_field_type` strips the
                    // `Box`), so both walk the receiver's list.
                    | "alloc::boxed::iter::<Impl>::into_iter"
                    | "core::slice::<Impl>::iter"
                    | "core::slice::<Impl>::iter_mut"
            )
        })
    }

    /// `core::str::<Impl>::chars` / `core::str::chars` on a string-family
    /// receiver (`tyref_is_string_family`), including autoderef
    /// `String` → `str`.  Yields Unicode scalars (`Utf8StringIterator`),
    /// so it is not a byte-view identity and not a slice `iter`.
    pub(in crate::front::mir) fn is_str_chars_constructor(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next();
        if leaf != Some("chars") || !path.contains("str") {
            return false;
        }
        first_arg_ty.is_some_and(|ty| tyref_is_string_family(ty, self.llbc))
    }

    /// `core::str::<Impl>::as_ptr` / `core::str::as_ptr` on a
    /// string-family receiver (`tyref_is_string_family` /
    /// `tyref_strips_to_str`).  `as_ptr` contains the substring
    /// `"str"`, so the owner test is a path *segment* `== "str"`.
    pub(in crate::front::mir) fn is_str_as_ptr(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        const_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next();
        if leaf != Some("as_ptr") || !path.split("::").any(|seg| seg == "str") {
            return false;
        }
        first_arg_ty.or(const_arg_ty).is_some_and(|ty| {
            tyref_is_string_family(ty, self.llbc) || tyref_strips_to_str(ty, self.llbc)
        })
    }

    /// `core::str::<Impl>::len` on a string-family receiver is
    /// `len(s)` = `ll_strlen` (`rstr.py AbstractStringRepr.rtype_len`),
    /// planted as the `__strlen` marker the rtyper adapter and
    /// `jtransform` both lower. Slice/Vec/Wtf8 `len` have their own
    /// arms.
    pub(in crate::front::mir) fn is_str_len(
        &self,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
        const_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next();
        if leaf != Some("len") || !path.split("::").any(|seg| seg == "str") {
            return false;
        }
        first_arg_ty.or(const_arg_ty).is_some_and(|ty| {
            tyref_is_string_family(ty, self.llbc) || tyref_strips_to_str(ty, self.llbc)
        })
    }

    /// `core::str::<Impl>::contains` / `core::str::contains`.
    /// Distinct from `Range::contains`, slice `contains`, and flag-set
    /// `contains`: those owners have no `str` path segment.  The
    /// haystack place type is not consulted — `FMTS.contains(spec)`
    /// materialises a `Ref` of the static whose Dedup id can share a
    /// number with a layout node, so `tyref_is_string_family` is not a
    /// reliable gate here.
    pub(in crate::front::mir) fn is_str_contains_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let path = fd.item_meta.name_path();
        let leaf = path
            .rsplit("::")
            .next()
            .map(crate::front::clause_spec::unspecialized_leaf);
        leaf == Some("contains") && path.split("::").any(|seg| seg == "str")
    }

    /// `str::contains`'s `Pattern` generic, monomorphised.  `char` (and
    /// a code-point Variable that rustc still types as `char`) becomes a
    /// UTF-8 string needle; `&str` / `&String` stay a string needle.
    pub(in crate::front::mir) fn str_contains_needle_kind(
        &self,
        reg: &RegularCall,
        second_arg_ty: Option<&TyRef>,
        second_const_arg_ty: Option<&TyRef>,
    ) -> Option<StrContainsNeedleKind> {
        let mut tys: Vec<TyRef> = Vec::new();
        if let Some(ty) = str_contains_pattern_generic(reg, self.llbc) {
            tys.push(ty);
        }
        if let Some(ty) = second_arg_ty {
            tys.push(ty.clone());
        }
        if let Some(ty) = second_const_arg_ty {
            tys.push(ty.clone());
        }
        if tys.iter().any(|ty| tyref_is_char(ty, self.llbc)) {
            return Some(StrContainsNeedleKind::Char);
        }
        if tys
            .iter()
            .any(|ty| tyref_is_string_family(ty, self.llbc) || tyref_strips_to_str(ty, self.llbc))
        {
            return Some(StrContainsNeedleKind::Str);
        }
        // After `str::chars` → utf8iter the live needle is a nonneg
        // SomeInteger code point while the MIR Pattern is still `char`.
        if tys.iter().any(|ty| tyref_is_integer_scalar(ty, self.llbc)) {
            return Some(StrContainsNeedleKind::Char);
        }
        None
    }

    /// `<[T]>::len` / `Vec::len` — the container element count.  Lowered
    /// to the `__len` operation so the receiver's list annotation
    /// supplies the length through the rtyper's `len` op, the same
    /// routing [`is_concrete_iter_constructor`] gives the container
    /// `iter`.
    ///
    /// The length-prefixed containers `IntArray` / `FloatArray`
    /// (`pyre-object/src/int_array.rs` / `float_array.rs`) expose the same
    /// inherent `len()` whose body reads the nested LIST `length` word
    /// (`rlist.py` `("length", Signed)`).  The receiver is the
    /// `{ block, len }` struct, not a `SomeList`, so the read is that
    /// field ([`Self::typed_array_len_owner`]), not `ListRepr.rtype_len`.
    /// Entering the body dead-ends at `AtomicUsize::load`; recognising
    /// the call emits a `FieldRead` of `len`
    /// (`jtransform.py` `_handle_list_call` `list.int_len`).
    ///
    /// `FixedObjectArray::len` is *not* here: it goes to [`OpKind::ArrayLen`]
    /// via [`Self::is_object_array_len`], because its receiver is the
    /// virtualizable array and a `__len` call would pass that array as a
    /// call argument. `Vec::len` stays here: `rlist.ll_length` reads
    /// `l.length`, and a field named `"len"` is not that attribute.
    ///
    /// The FunDecl path is the container (`alloc::vec::<Impl>::len` /
    /// `IntArray` / `FloatArray`). `rordereddict::RDict::len` is a
    /// different declaration, so it cannot take this arm;
    /// `rordereddict.rtype_len` / `ll_dict_len` records
    /// `d.num_live_items` from its own body.
    pub(in crate::front::mir) fn is_container_len(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "alloc::vec::<Impl>::len"
                    | "pyre_object::int_array::<Impl>::len"
                    | "pyre_object::float_array::<Impl>::len"
            )
        })
    }

    /// `Vec::set_len(v, n)` — `l.length = n` (`ll_set_length`).
    pub(in crate::front::mir) fn is_vec_set_len(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let np = fd.item_meta.name_path();
            np.rsplit("::").next() == Some("set_len")
                && np.split("::").any(|seg| seg == "vec" || seg == "Vec")
        })
    }

    /// `items_block_capacity` — the `ItemsBlock` capacity header read,
    /// `len(l.items)` of the GcArray.
    pub(in crate::front::mir) fn is_items_block_capacity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "pyre_object::object_array::items_block_capacity"
        })
    }

    /// Object-strategy `ListStrategy.unerase` of an `ItemsBlock`
    /// (`rerased.py`). The field stays erased; the use site narrows it
    /// to the type-derived items list (`rlist.py` `ll_setitem_fast`).
    pub(in crate::front::mir) fn unerase_object_strategy_items_block(
        &mut self,
        bb_id: BlockId,
        block: Variable,
        root: &str,
    ) -> Variable {
        self.narrow_value_to_instance_root(bb_id, block.into(), root)
            .as_variable()
            .expect("ListStrategy.unerase of object items stays a Variable")
            .clone()
    }

    /// `items_block_items_base` — the `ItemsBlock` items pointer. On a
    /// fixed list the operand is already that array (`ll_fixed_items`).
    pub(in crate::front::mir) fn is_items_block_items_base(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "pyre_object::object_array::items_block_items_base"
        })
    }

    /// The operand's lowleveltype is a fixed-list GcArray
    /// (`[*mut PyObject]` / `FixedObjectArray`), so `len` of it is
    /// `ll_fixed_length`. Follows copies to the producing place; the
    /// decision is [`type_spelling_is_fixed_object_array`] on that
    /// place's published type, not the field it was read from.
    pub(in crate::front::mir) fn local_is_fixed_object_array(&self, mut local: usize) -> bool {
        for _ in 0..8 {
            let mut producers = 0usize;
            let mut followed: Option<usize> = None;
            let mut is_fixed = false;
            for (bb_idx, bb) in self.body.body.iter().enumerate() {
                if !self.forward_reached[bb_idx] {
                    continue;
                }
                for stmt in &bb.statements {
                    let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                        continue;
                    };
                    if !matches!(&place.kind, PlaceKind::Local(i) if *i as usize == local) {
                        continue;
                    }
                    producers += 1;
                    let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = rvalue else {
                        continue;
                    };
                    if let PlaceKind::Local(i) = &src.kind {
                        followed = Some(*i as usize);
                    } else if self.place_lowlevel_is_fixed_object_array(src) {
                        is_fixed = true;
                    }
                }
            }
            if producers != 1 {
                return false;
            }
            if is_fixed {
                return true;
            }
            match followed {
                Some(next) => local = next,
                None => return false,
            }
        }
        false
    }

    /// Published lowleveltype of `place`. A field uses
    /// [`published_struct_field_layout`]; any other place uses its own
    /// layout spelling.
    pub(in crate::front::mir) fn place_lowlevel_is_fixed_object_array(
        &self,
        place: &Place,
    ) -> bool {
        let spelling = if let PlaceKind::Projection(base, ProjectionElem::Tagged(field)) =
            &place.kind
            && let Some(payload) = field.as_object().and_then(|obj| obj.get("Field"))
            && let Some((_, name, ty, _, _)) = self.resolve_adt_field(&base.ty, payload)
            && let Some(type_path) = self.field_decl_path(&base.ty)
        {
            published_struct_field_layout(
                &type_path,
                &name,
                &ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
        } else {
            tyref_to_field_layout_string(
                &place.ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
        };
        type_spelling_is_fixed_object_array(&spelling)
    }

    pub(in crate::front::mir) fn field_decl_path(&self, base_ty: &TyRef) -> Option<String> {
        let type_id = type_decl_ref_adt_id(self.field_base_adt(base_ty)?)?;
        Some(self.llbc.type_by_id(type_id)?.item_meta.name_path())
    }

    /// `FixedObjectArray::len` (`pyre-object/src/object_array.rs`) — the
    /// `PyFrame.locals_cells_stack_w` `Ptr(GcArray(PyObjectRef))` accessor.
    /// Its body reads the array's own length header, so the call *is*
    /// `arraylen_gc`; see the `ArrayLen` emission site for why this one
    /// cannot share `is_container_len`'s `__len` routing.
    pub(in crate::front::mir) fn is_object_array_len(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "pyre_object::object_array::<Impl>::len")
    }

    /// Owner leaf of `IntArray::len` / `FloatArray::len` — the nested
    /// LIST `length` field (`rlist.py` `GcStruct("list", ("length", Signed),
    /// …)`).  `None` for `Vec::len`, which stays on `__len`.
    pub(in crate::front::mir) fn typed_array_len_owner(
        &self,
        reg: &RegularCall,
    ) -> Option<&'static str> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
            "pyre_object::int_array::<Impl>::len" => Some("IntArray"),
            "pyre_object::float_array::<Impl>::len" => Some("FloatArray"),
            _ => None,
        }
    }

    /// `as_bytes()` aliases the dest to the `W_UnicodeObject` and marks
    /// that local; a later `slice::len` / `slice::is_empty` often
    /// receives `Copy(*local)` rather than the local itself, so the
    /// mark must follow the projection the same way `Rvalue::Len` does.
    /// Last-write-wins for a call dest aliased to `args[0]`.  Inherit the
    /// mark when the receiver is already a string byte view (`x = x.as_slice()`
    /// after `as_bytes`); drop it when the receiver is an ordinary `&[u8]`.
    pub(in crate::front::mir) fn alias_dest_to_arg0(
        &mut self,
        dest_local: usize,
        arg: Variable,
        inherit_byte_view: bool,
    ) {
        if inherit_byte_view {
            if !self.string_byte_view_locals.contains(&dest_local) {
                self.string_byte_view_locals.push(dest_local);
            }
        } else {
            self.string_byte_view_locals
                .retain(|&local| local != dest_local);
        }
        self.local_var[dest_local] = Some(LocalValue::One(arg));
    }

    pub(in crate::front::mir) fn alias_dest_to_arg0_inherit(
        &mut self,
        dest_local: usize,
        arg: Variable,
        arg_locals: &[Option<usize>],
    ) {
        let inherit = arg_locals
            .first()
            .copied()
            .flatten()
            .is_some_and(|local| self.string_byte_view_locals.contains(&local));
        self.alias_dest_to_arg0(dest_local, arg, inherit);
    }

    pub(in crate::front::mir) fn operand_is_string_byte_view(&self, op: &Operand) -> bool {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self
                .string_byte_view_locals
                .iter()
                .any(|&local| place_references_local(place, local)),
            Operand::Const(_) => false,
        }
    }

    /// ARRAY identity of a `<[T]>::len` / `is_empty` receiver whose `T` is
    /// an object pointer. Every other slice keeps the identity-less length
    /// read. The identity string itself comes from [`slice_array_type_id`].
    /// `<[PyObject]>::len` / `is_empty` sees the list, so `len` is
    /// `AbstractListRepr.rtype_len`. A classdef-less receiver takes
    /// `len_SomeInstance`, which looks up `__len__`.
    pub(in crate::front::mir) fn object_slice_arraylen_base(
        &mut self,
        bb_id: BlockId,
        reg: &RegularCall,
        base: Variable,
    ) -> Variable {
        if self.slice_object_array_type_id(reg).is_none() {
            return base;
        }
        self.narrow_value_to_instance_root(bb_id, crate::model::LinkArg::Value(base), "[PyObject]")
            .as_variable()
            .expect("an object-pointer slice base stays a Variable")
            .clone()
    }

    pub(in crate::front::mir) fn slice_object_array_type_id(
        &self,
        reg: &RegularCall,
    ) -> Option<String> {
        self.slice_swap_elem_tyref(reg)
            .is_some_and(|ty| output_type_is_objectptr(&ty, self.llbc))
            .then(|| {
                slice_array_type_id("*mut PyObject")
                    .expect("object-pointer slice names the object gcarray")
            })
    }

    pub(in crate::front::mir) fn is_slice_len(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::slice::<Impl>::len")
    }

    pub(in crate::front::mir) fn is_slice_is_empty(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::slice::<Impl>::is_empty")
    }

    /// `Vec::as_slice` / `<[T]>::as_slice` — a borrowed slice view of the
    /// same elements over shared storage, so it is an identity on the list
    /// model and the callsite aliases its receiver instead of leaving the
    /// unregistered method callee.
    ///
    /// `FixedObjectArray::as_slice` is the same view over a receiver
    /// that is already the object list, so the alias applies.
    /// `IntArray` / `FloatArray` are `{ block, len }` structs
    /// (`rlist.py` `("length", Signed)` plus `items`). Their `len` stays
    /// the `FieldRead` in [`Self::is_container_len`]
    /// ([`Self::typed_array_len_owner`]); the view is
    /// [`Self::numeric_array_slice_view_owner`] (`block`), not an alias
    /// of the struct — aliasing it makes the following index
    /// `getitem_SomeInstance`. Entering `from_raw_parts` dead-ends:
    /// that intrinsic is unregistered.
    ///
    /// `as_mut_slice` is matched alongside `as_slice`: the mutable view
    /// addresses the receiver's own storage, so a write through the alias
    /// *is* a write to the receiver, which is exactly what the list model
    /// wants.
    ///
    /// `<[T]>::to_vec` is deliberately NOT matched: it allocates an
    /// independent list, so aliasing it to the receiver would let a later
    /// mutation of the copy (or of the original) be observed through the
    /// other — `let mut c = s.to_vec(); c.push(x)` must not touch `s`.  It
    /// needs a real copy lowering (alloc + element copy), not an alias.
    pub(in crate::front::mir) fn is_container_slice_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::<Impl>::as_slice"
                    | "core::slice::<Impl>::as_mut_slice"
                    | "alloc::vec::<Impl>::as_slice"
                    | "alloc::vec::<Impl>::as_mut_slice"
                    | "pyre_object::object_array::<Impl>::as_slice"
                    | "pyre_object::object_array::<Impl>::as_mut_slice"
            )
        })
    }

    /// Return the source owner for the string-list items-view adapters.
    /// RPython's `ll_items(l)` reads the `items` field from the resized list;
    /// pyre's source spelling calls that field `block`.
    pub(in crate::front::mir) fn string_array_slice_view_owner(
        &self,
        reg: &RegularCall,
    ) -> Option<&'static str> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
            "pyre_object::bytes_array::<Impl>::as_slice"
            | "pyre_object::bytes_array::<Impl>::as_mut_slice" => Some("BytesArray"),
            "pyre_object::unicode_array::<Impl>::as_slice"
            | "pyre_object::unicode_array::<Impl>::as_mut_slice" => Some("UnicodeArray"),
            _ => None,
        }
    }

    /// `IntArray::as_slice` / `FloatArray::as_slice` — the `block`
    /// pointer, narrowed to `[i64]` / `[f64]`.
    pub(in crate::front::mir) fn numeric_array_slice_view_owner(
        &self,
        reg: &RegularCall,
    ) -> Option<(&'static str, &'static str)> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
            "pyre_object::int_array::<Impl>::as_slice"
            | "pyre_object::int_array::<Impl>::as_mut_slice" => Some(("IntArray", "[i64]")),
            "pyre_object::float_array::<Impl>::as_slice"
            | "pyre_object::float_array::<Impl>::as_mut_slice" => Some(("FloatArray", "[f64]")),
            _ => None,
        }
    }

    /// Owner of the two Rust storage adapters whose `remove` body is the
    /// concrete spelling of RPython `rlist.ll_delitem_nonneg`.
    pub(in crate::front::mir) fn string_array_remove_owner(&self) -> Option<&'static str> {
        match self.graph.name.as_str() {
            "pyre_object::bytes_array::<Impl>::remove" => Some("BytesArray"),
            "pyre_object::unicode_array::<Impl>::remove" => Some("UnicodeArray"),
            _ => None,
        }
    }

    /// `std::ptr::copy` inside the exact string-list `remove` adapters.  The
    /// caller reconstructs the logical array and indices and emits the
    /// RPython `rgc.ll_arraymove` marker; no arbitrary raw copy qualifies.
    pub(in crate::front::mir) fn is_string_array_remove_ptr_copy(&self, reg: &RegularCall) -> bool {
        if self.string_array_remove_owner().is_none() {
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::ptr::copy")
    }

    /// `core::ptr::copy_nonoverlapping` / `std::ptr::copy_nonoverlapping`.
    pub(in crate::front::mir) fn is_ptr_copy_nonoverlapping(&self, reg: &RegularCall) -> bool {
        regular_call_name_path(reg, self.llbc).is_some_and(|path| {
            matches!(
                path.as_str(),
                "core::ptr::copy_nonoverlapping" | "std::ptr::copy_nonoverlapping"
            )
        })
    }

    /// The body-side `from_raw_parts` half of
    /// [`Self::string_array_slice_view_owner`].
    pub(in crate::front::mir) fn is_string_array_view_from_raw_parts(
        &self,
        reg: &RegularCall,
    ) -> bool {
        if !(self.graph.name.ends_with("bytes_array::<Impl>::as_slice")
            || self
                .graph
                .name
                .ends_with("bytes_array::<Impl>::as_mut_slice")
            || self.graph.name.ends_with("unicode_array::<Impl>::as_slice")
            || self
                .graph
                .name
                .ends_with("unicode_array::<Impl>::as_mut_slice"))
        {
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut"
            )
        })
    }

    pub(in crate::front::mir) fn emit_string_array_items_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        owner: &str,
    ) -> Variable {
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let canonical_owner = match owner {
            "BytesArray" => "bytes_array::BytesArray",
            "UnicodeArray" => "unicode_array::UnicodeArray",
            _ => unreachable!("string array owner is closed"),
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("block", Some(owner.to_string())).with_owner_id(Some(
                    majit_ir::descr::StructId::from_canonical(canonical_owner),
                )),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        result
    }

    /// The logical `l.length` of a string-list storage record — the second
    /// word the `from_raw_parts` view fold drops.  Upstream reads it as the
    /// list's own field (`ll_length`, `rlist.py`), strictly apart
    /// from the array header's `ll_fixed_length` / `arraylen_gc`.
    pub(in crate::front::mir) fn emit_string_array_len_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        owner: &str,
    ) -> Variable {
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let canonical_owner = match owner {
            "BytesArray" => "bytes_array::BytesArray",
            "UnicodeArray" => "unicode_array::UnicodeArray",
            _ => unreachable!("string array owner is closed"),
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("len", Some(owner.to_string())).with_owner_id(Some(
                    majit_ir::descr::StructId::from_canonical(canonical_owner),
                )),
                ty: ValueType::Unsigned,
                pure: false,
            },
        });
        result
    }

    pub(in crate::front::mir) fn record_string_array_view_len(
        &mut self,
        dest_local: usize,
        logical_len: Variable,
    ) {
        if !self
            .string_array_view_locals
            .iter()
            .any(|(local, _)| *local == dest_local)
        {
            self.string_array_view_locals
                .push((dest_local, logical_len));
        }
    }

    pub(in crate::front::mir) fn record_numeric_array_view_len(
        &mut self,
        dest_local: usize,
        logical_len: Variable,
    ) {
        if !self
            .numeric_array_view_locals
            .iter()
            .any(|(local, _)| *local == dest_local)
        {
            self.numeric_array_view_locals
                .push((dest_local, logical_len));
        }
    }

    pub(in crate::front::mir) fn array_view_logical_len_for_place(
        &self,
        place: &Place,
    ) -> Option<Variable> {
        self.string_array_view_locals
            .iter()
            .chain(self.numeric_array_view_locals.iter())
            .find(|(local, _)| place_references_local(place, *local))
            .map(|(_, len)| len.clone())
    }

    pub(in crate::front::mir) fn array_view_logical_len_for_var(
        &self,
        arg: &Variable,
    ) -> Option<Variable> {
        self.string_array_view_locals
            .iter()
            .chain(self.numeric_array_view_locals.iter())
            .find(|(local, _)| {
                self.local_var
                    .get(*local)
                    .and_then(Option::as_ref)
                    .and_then(|current| current.one().ok())
                    .is_some_and(|current| current == *arg)
            })
            .map(|(_, len)| len.clone())
    }

    /// `l.length` as a value defined in `bb_id`.
    ///
    /// View creation records that word outside `local_var`, so
    /// `FrameState.getoutputargs` / `flowcontext.py` `make_next_block`
    /// do not carry it. A later `.len()` / `Rvalue::Len` must not name
    /// the predecessor Variable on a link (`checkgraph` `definevar`).
    pub(in crate::front::mir) fn bind_array_view_len_in_block(
        &mut self,
        bb_id: BlockId,
        len_var: Variable,
    ) -> Variable {
        bind_array_view_len_in_block(&mut self.graph, bb_id, len_var)
    }

    pub(in crate::front::mir) fn is_from_raw_parts_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut"
            )
        })
    }

    /// Body-side `from_raw_parts` of [`Self::numeric_array_slice_view_owner`].
    pub(in crate::front::mir) fn numeric_array_view_from_raw_parts_root(
        &self,
        reg: &RegularCall,
    ) -> Option<&'static str> {
        if !self.is_from_raw_parts_call(reg) {
            return None;
        }
        if self.graph.name.ends_with("int_array::<Impl>::as_slice")
            || self.graph.name.ends_with("int_array::<Impl>::as_mut_slice")
        {
            Some("[i64]")
        } else if self.graph.name.ends_with("float_array::<Impl>::as_slice")
            || self
                .graph
                .name
                .ends_with("float_array::<Impl>::as_mut_slice")
        {
            Some("[f64]")
        } else {
            None
        }
    }

    /// `IntArray` / `FloatArray` items pointer. Same `block` field as
    /// [`Self::emit_string_array_items_read`].
    pub(in crate::front::mir) fn emit_numeric_array_items_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        owner: &str,
    ) -> Variable {
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let canonical_owner = match owner {
            "IntArray" => "int_array::IntArray",
            "FloatArray" => "float_array::FloatArray",
            _ => unreachable!("numeric array owner is closed"),
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("block", Some(owner.to_string())).with_owner_id(Some(
                    majit_ir::descr::StructId::from_canonical(canonical_owner),
                )),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        result
    }

    /// The logical `l.length` of an `IntArray` / `FloatArray` storage
    /// record. Upstream reads it as the list's own field (`ll_length`),
    /// strictly apart from `ll_fixed_length` / `arraylen_gc`.
    pub(in crate::front::mir) fn emit_numeric_array_len_read(
        &mut self,
        bb_id: BlockId,
        base: Variable,
        owner: &str,
    ) -> Variable {
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let canonical_owner = match owner {
            "IntArray" => "int_array::IntArray",
            "FloatArray" => "float_array::FloatArray",
            _ => unreachable!("numeric array owner is closed"),
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("len", Some(owner.to_string())).with_owner_id(Some(
                    majit_ir::descr::StructId::from_canonical(canonical_owner),
                )),
                ty: ValueType::Int,
                pure: false,
            },
        });
        result
    }

    /// `UnsafeCell::{new, into_inner, get, get_mut}`: the cell word itself.
    pub(in crate::front::mir) fn is_unsafecell_word_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        // `UnsafeCell<T>` is `#[repr(transparent)]`. `new`, `into_inner`,
        // `get`, and `get_mut` are the cell word — the same identity as
        // `<[T]>::as_ptr`. A bare `core::cell::get_mut`, which Charon
        // produces when it drops the type, is not enough: `RefCell::get_mut`
        // spells the same and is not the cell word. `RefCell` carries a
        // borrow counter ahead of the value, so aliasing it would skip that
        // field. The owner has to be `UnsafeCell`, either as a path segment
        // or as the inherent-impl ADT (`core::cell::<Impl>::new`, whose
        // `name_path` drops `UnsafeCell`).
        let leaf_is_word = |leaf: &str| matches!(leaf, "new" | "into_inner" | "get" | "get_mut");
        let owner_is_cell = |owner: &str| owner == "UnsafeCell" || owner.ends_with("::UnsafeCell");
        let path = fd.item_meta.name_path();
        if let Some((owner, leaf)) = path.rsplit_once("::")
            && leaf_is_word(leaf)
            && owner_is_cell(owner)
        {
            return true;
        }
        impl_method_owner_for_fundecl(self.llbc, fd)
            .is_some_and(|(owner, leaf)| leaf_is_word(&leaf) && owner_is_cell(&owner))
    }

    /// `<[T]>::as_ptr` — the data-pointer projection of a slice receiver.  In
    /// the erased value-model a `&[T]` collapses to a single GC array pointer
    /// (its length is read from the array header by `ArrayLen`, not a companion
    /// fat-pointer word), so the slice's data pointer IS the receiver value.
    /// Alias the result to the receiver — the same identity fold as
    /// `NonNull::as_ptr` (`nonnull_new_identity_alias`) and `as_slice` above —
    /// instead of leaving an unregistered `core::slice::<Impl>::as_ptr`
    /// residual whose only prior handling was a `FOREIGN_STDLIB_EXTERNALS`
    /// `Address` annotation with no host address.
    ///
    /// This is a fixed-array identity ONLY: it mirrors `ll_fixed_items(l) = l`
    /// (`rlist.py`, a `FixedSizeListRepr` IS its items array).  A resized
    /// list / `Vec` reaches its items buffer through `ll_items(l) = l.items`
    /// (`rlist.py` `ll_items`, a `getfield`), so `alloc::vec::<Impl>::as_ptr`
    /// is NOT an identity on the receiver and must not fold here —
    /// `rust_vec_std_call` lowers it to `ll_vec_items_*` / `ll_vec_as_ptr`.
    /// `core::ptr::{const_ptr,mut_ptr}::<Impl>::is_null` — Charon's name
    /// for `<*const T>::is_null` / `<*mut T>::is_null`.  The impl module
    /// leaf is `const_ptr` or `mut_ptr` and the method leaf is `is_null`;
    /// the `Impl` segment sits between them.  Mutability does not change
    /// the null test (`p is None`).
    pub(in crate::front::mir) fn is_raw_ptr_is_null(&self, reg: &RegularCall) -> bool {
        self.is_raw_ptr_method(reg, "is_null")
    }

    /// `<*const T>::as_ref` / `<*mut T>::as_ref` / `as_mut`.
    ///
    /// The return is `Option<&T>` with the null niche: the pointer word is
    /// the option. Same alias as [`Self::is_noop_ptr_cast`]. A null test
    /// stays on the `Option` match (`p is None`), not on a residual call
    /// into `core::ptr`.
    pub(in crate::front::mir) fn is_raw_ptr_as_ref(&self, reg: &RegularCall) -> bool {
        self.is_raw_ptr_method(reg, "as_ref") || self.is_raw_ptr_method(reg, "as_mut")
    }

    pub(in crate::front::mir) fn is_raw_ptr_method(
        &self,
        reg: &RegularCall,
        leaf_name: &str,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let segs = &fd.item_meta.name;
        let Some(last_idx) = segs
            .iter()
            .rposition(|s| matches!(s, NameSeg::Ident { .. }))
        else {
            return false;
        };
        let leaf = match &segs[last_idx] {
            NameSeg::Ident { ident: (s, _) } => s.as_str(),
            _ => return false,
        };
        if leaf != leaf_name || last_idx < 2 {
            return false;
        }
        let module_leaf = match &segs[last_idx - 2] {
            NameSeg::Ident { ident: (s, _) } => s.as_str(),
            _ => return false,
        };
        matches!(module_leaf, "const_ptr" | "mut_ptr")
            && matches!(&segs[last_idx - 1], NameSeg::Other(v) if v.get("Impl").is_some())
    }

    pub(in crate::front::mir) fn is_container_as_ptr_identity(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::<Impl>::as_ptr" | "core::slice::<Impl>::as_mut_ptr"
            )
        })
    }

    /// `core::hint::must_use(value)` — the identity wrapper
    /// (`pub const fn must_use<T>(value: T) -> T`) the compiler inserts to
    /// carry an `unused_must_use` lint through macro-generated code.  It
    /// returns its argument unchanged and `core` has no graph body for it
    /// (an unregistered callee), so the callsite aliases its argument
    /// instead of calling the missing identity body.
    /// `core::mem::forget(value)` deliberately suppresses destructor glue.
    /// `core::mem::drop(value)` is [`Self::is_mem_drop`]: a `Move` of a local
    /// lowers as the [`TermKind::Drop`] of that place, and a type with no
    /// drop glue is a no-op.  A type whose glue this front end cannot lower
    /// stays a residual call.
    pub(in crate::front::mir) fn is_mem_forget(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::mem::forget")
    }

    /// `majit_ir::icf::icf_identity_token::<ID>()` — empty `nomem` asm
    /// natively / a volatile const read on wasm32, returning `()`.
    /// `ItemMeta::name_path` is the template path, so every const generic
    /// instance is this one item. genc `FunctionCodeGenerator` never
    /// merges graphs, and the flow graph has no operation for it.
    pub(in crate::front::mir) fn is_icf_identity_token(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "majit_ir::icf::icf_identity_token")
    }

    /// `core::mem::drop(value)` runs `value`'s destructor at the call.
    pub(in crate::front::mir) fn is_mem_drop(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::mem::drop")
    }

    pub(in crate::front::mir) fn is_hint_must_use(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::hint::must_use")
    }

    /// `RBigInt::translated_alias(self)` is the Rust ownership spelling of an
    /// RPython reference assignment. The host shallow-clones the two-word
    /// handle, while the translated representation is already one GC
    /// reference and must reuse that exact reference (`x + 0`, `x << 0`,
    /// nonnegative `abs`, and the other upstream `return self` fast paths).
    ///
    /// Keep this separate from ordinary `Clone::clone`: `rbigint.neg` uses
    /// that operation to construct a fresh handle before changing `_size`.
    pub(in crate::front::mir) fn is_rbigint_translated_alias(
        &self,
        reg: &RegularCall,
        receiver_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let path = fd.item_meta.name_path();
            // Charon spells an inherent impl method owned by another crate
            // either `rbigint::RBigInt::translated_alias` or
            // `rbigint::<Impl>::translated_alias`, depending on which LLBC
            // contains the callsite.  The receiver's exact nominal type is
            // the stable identity across both forms.  Do not use the output:
            // dependent-crate LLBC represents that foreign return as an
            // opaque projection, which is precisely why the marker used to
            // escape as a classdef-less method call.
            path.ends_with("::translated_alias")
                && receiver_ty.is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
        })
    }

    /// A callsite of the reflexive `impl<T> From<T> for T`
    /// (`core::convert::<Impl>::from`), whose whole conversion is a
    /// `T -> T` identity.  `core` carries no graph body, so without this
    /// the callsite skips as an unregistered `FunctionPath`; aliasing the
    /// destination to the argument is the `From` direction of the
    /// identity [`Self::blanket_into_devirt`] already reports for `Into`
    /// through [`IntoDevirt::Identity`].
    ///
    /// The gate is the *declared* signature rather than the instantiated
    /// one: the reflexive impl declares `from(T) -> T`, so its input and
    /// output resolve to the same type expression whatever `T` is bound
    /// to at the callsite, and no generic substitution has to be walked.
    /// A conversion between two distinct types declares two distinct
    /// expressions and keeps the ordinary call — it has a body worth
    /// entering, or crosses a repr boundary the carrier models.
    ///
    /// The live callsite is `pyopcode`'s `u32::from(format.get(op_arg))`:
    /// [`Self::oparg_value_alias`] already binds `Arg::<T>::get` to its
    /// `OpArg` argument, so the `From` sees a `u32` on both sides and
    /// rustc selects the reflexive impl.
    pub(in crate::front::mir) fn is_reflexive_from(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        if fd.item_meta.name_path() != "core::convert::<Impl>::from" {
            return false;
        }
        let Some(src) = fd.signature.inputs.first() else {
            return false;
        };
        match (self.tyref_body(src), self.tyref_body(&fd.signature.output)) {
            (Some(src_ty), Some(dst_ty)) => src_ty == dst_ty,
            _ => false,
        }
    }

    /// A callsite of a numeric `From` whose source and destination are both
    /// integer-kinded (`core::convert::num::<Impl>::from`) — `u32::from(u16)`
    /// and its family.
    ///
    /// `jtransform.py rewrite_op_cast_bool_to_int` deletes every one of these outright:
    /// `rewrite_op_cast_char_to_int`, `cast_int_to_uint`, `cast_uint_to_int`
    /// and the rest are each `pass`, because a cast between two integer
    /// primitives is a no-op once both live in the same `Int` register kind.
    /// This is the same conclusion for the same reason, reached at the
    /// callsite rather than at an op because `core` carries no graph body and
    /// the call would otherwise stay residual.
    ///
    /// Narrowing needs no separate rule: `From` between numerics is
    /// implemented only where the conversion is lossless, so a source that is
    /// wider than its destination never selects this impl.
    ///
    /// 128-bit is excluded on both sides. [`ValueType::Int128`] /
    /// [`ValueType::UInt128`] are not the register kind — `OpKind::ConstInt128`
    /// says so where it is declared — so aliasing across that boundary would
    /// put a value in a bank that cannot hold it.
    ///
    /// Float is excluded by the same test: `f64::from(i32)` is a real
    /// conversion (`cast_int_to_float`), not an alias.
    pub(in crate::front::mir) fn is_widening_int_from(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        if fd.item_meta.name_path() != "core::convert::num::<Impl>::from" {
            return false;
        }
        let Some(src) = fd.signature.inputs.first() else {
            return false;
        };
        let register_int = |ty: &TyRef| {
            matches!(
                tyref_to_value_type_with(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs
                ),
                ValueType::Int | ValueType::Unsigned
            )
        };
        register_int(src) && register_int(&fd.signature.output)
    }

    /// `f64::is_nan(self)` — `core` has no graph body (Opaque), so the
    /// callsite would skip as an unregistered `FunctionPath`.  `is_nan`
    /// is `value != value` (`rfloat.isnan`), so the caller lowers it to
    /// a reflexive `ne` `BinOp`; the float operand makes the rtyper pick
    /// `float_ne`, which (unlike `int_ne`) carries no `n(x, x) => 0`
    /// reflexive fold, preserving the NaN-only truth value.
    pub(in crate::front::mir) fn is_f64_is_nan(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_nan")
    }

    /// `f64::abs(self)` — `core` has no graph body (Opaque).  `rfloat`
    /// / `lloperation.py float_abs` is the corresponding foldable llop.
    pub(in crate::front::mir) fn is_f64_abs(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::abs")
    }

    /// `core::num::<Impl>::abs` on a word-sized signed integer (`i64` /
    /// `isize`), the `Signed` repr `rint.py rtype_abs` lowers to `int_abs`.
    /// A narrower integer wraps at its own `MIN`, which the word op does
    /// not reproduce, so it is left to the generic call path.
    pub(in crate::front::mir) fn is_word_int_abs(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        fd.item_meta.name_path() == "core::num::<Impl>::abs"
            && fd.signature.inputs.first().is_some_and(|src| {
                matches!(self.tyref_literal_int_atom(src), Some("I64" | "Isize"))
            })
    }

    /// Opaque `f64` math leaves that `ll_math.py` registers as C
    /// llexternals.  Returns `(arity, math_* leaf)`.
    pub(in crate::front::mir) fn f64_ll_math_leaf(
        &self,
        reg: &RegularCall,
    ) -> Option<(usize, &'static str)> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let path = self.llbc.fn_by_id(*id)?.item_meta.name_path();
        let leaf = path
            .strip_prefix("core::f64::<Impl>::")
            .or_else(|| path.strip_prefix("std::f64::<Impl>::"))?;
        crate::translator::rtyper::lltypesystem::module::ll_math::f64_method_llexternal(leaf)
    }

    /// `f64::is_finite(self)` — `core` has no graph body (Opaque), so the
    /// callsite would skip as an unregistered `FunctionPath`.  `is_finite`
    /// is `(x - x) == 0.0`: finite `x` gives `0.0 == 0.0` = true, while ±inf
    /// and NaN both make `x - x` NaN so the equality is false.  The float
    /// `sub` carries no reflexive `sub(x, x) => 0` fold (unlike the integer
    /// bank), preserving the inf/NaN truth value.
    pub(in crate::front::mir) fn is_f64_is_finite(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_finite")
    }

    /// Rust's opaque infinity predicate maps to ll_math.py ll_math_isinf.
    pub(in crate::front::mir) fn is_f64_is_infinite(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_infinite")
    }

    /// `f64::from_bits(bits)` — `core` has no graph body (Opaque), so map the
    /// callsite to RPython's `longlong2float.longlong2float` external operation.
    /// Its rtyper specialization emits `convert_longlong_bytes_to_float`, the
    /// exact inverse of `float2longlong`.
    ///
    /// Matched as a `::`-segment suffix, not a substring: `str::contains`
    /// / `str::ends_with("::from_bits")` also accept Grain's
    /// `position_from_bits` and bitflags `from_bits`, which are not the
    /// f64 bitcast (`path_ends_with_segments`).
    pub(in crate::front::mir) fn is_f64_from_bits(&self, reg: &RegularCall) -> bool {
        self.f64_inherent_method(reg, "from_bits")
    }

    /// `f64::to_bits(self)` — the reverse of [`is_f64_from_bits`].
    pub(in crate::front::mir) fn is_f64_to_bits(&self, reg: &RegularCall) -> bool {
        self.f64_inherent_method(reg, "to_bits")
    }

    /// Inherent `f64` method at `core::f64::<Impl>::{method}` (or `std::`).
    pub(in crate::front::mir) fn f64_inherent_method(
        &self,
        reg: &RegularCall,
        method: &str,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let path = fd.item_meta.name_path();
            // Charon spells an inherent `f64` method either as
            // `core::f64::<Impl>::to_bits` or as `core::f64::to_bits`.
            path.strip_prefix("core::f64::<Impl>::")
                .or_else(|| path.strip_prefix("std::f64::<Impl>::"))
                .or_else(|| path.strip_prefix("core::f64::"))
                .or_else(|| path.strip_prefix("std::f64::"))
                == Some(method)
        })
    }

    /// `f64::to_int_unchecked::<i64>` — truncation toward zero, undefined
    /// outside the signed range, which is `cast_float_to_int`.  An i128
    /// dest, an f32 receiver, or a saturating `as` helper is not this
    /// spelling.
    pub(in crate::front::mir) fn is_f64_to_int_unchecked_i64(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
        first_arg_ty: Option<&TyRef>,
    ) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        if self.tyref_literal_int_atom(dest_ty) != Some("I64") {
            return false;
        }
        let path = fd.item_meta.name_path();
        match path.as_str() {
            "core::f64::<Impl>::to_int_unchecked" | "std::f64::<Impl>::to_int_unchecked" => true,
            "core::convert::num::<Impl>::to_int_unchecked"
            | "core::convert::num::FloatToInt::to_int_unchecked" => {
                first_arg_ty.is_some_and(|ty| self.tyref_literal_float_atom(ty) == Some("F64"))
            }
            _ => false,
        }
    }

    /// `transmute::<f64, i64>` / `transmute::<i64, f64>` is the same
    /// bitcast as `float2longlong` / `longlong2float`.  Emit the llop
    /// itself so a leaf the walker descends does not residualize a
    /// typed call.
    pub(in crate::front::mir) fn emit_float_bytes_llop(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        operand: crate::flowspace::model::Variable,
        to_float: bool,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let (op, result_ty) = if to_float {
            ("convert_longlong_bytes_to_float", ValueType::Float)
        } else {
            ("convert_float_bytes_to_longlong", ValueType::Int)
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::UnaryOp {
                op: op.to_string(),
                operand,
                result_ty,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `core::intrinsics::saturating_add` / `saturating_sub`
    /// (`lloperation.py` `LL_OPERATIONS`; signed clamp via `rint.py`
    /// `rtype_add_ovf` / `rtype_sub_ovf`). Emit a residual intrinsic Call
    /// and record the site for the post-lowering clamp rewrite.
    ///
    /// The rewrite must not run here. `lower_framestate` unions only the
    /// just-closed exits (`flowcontext.py` `make_next_block`). Splitting
    /// the Call's goto onto fresh non-MIR diamond arms leaves the MIR
    /// `target` with an empty entry state, so Pass 1 `stub_dead_block`s
    /// it (`set_raise("mir-dead")`) and the wrapper becomes a raise-only
    /// stub. `saturating_mul` already rewrites after the body lowers;
    /// this follows that contract.
    pub(in crate::front::mir) fn emit_saturating_intrinsic(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        lhs: crate::flowspace::model::Variable,
        rhs: crate::flowspace::model::Variable,
        dest_ty: &TyRef,
        is_add: bool,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let leaf = if is_add {
            "saturating_add"
        } else {
            "saturating_sub"
        };
        let result_ty = if self.tyref_literal_uint_atom(dest_ty).is_some() {
            ValueType::Unsigned
        } else {
            ValueType::Int
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "intrinsics".into(), leaf.into()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![lhs, rhs]),
                result_ty,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res.clone()));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);

        let signed = self
            .tyref_literal_int_atom(dest_ty)
            .is_some_and(crate::front::checked_arith::is_ovf_width_int_atom);
        let uint_atom = self.tyref_literal_uint_atom(dest_ty);
        let word_sized =
            uint_atom.is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom);
        if is_add {
            let unsigned_max = if signed || word_sized {
                None
            } else {
                uint_atom.and_then(|atom| {
                    crate::front::saturating_intrinsic::unsigned_atom_max(
                        atom,
                        crate::layout::target_word_size(),
                    )
                })
            };
            if signed || word_sized || unsigned_max.is_some() {
                self.saturating_add_sites.push(
                    crate::front::saturating_intrinsic::SaturatingAddSite {
                        result_var: res,
                        signed,
                        unsigned_max,
                    },
                );
            }
        } else {
            let unsigned_ok = uint_atom.is_some_and(|atom| atom != "U128");
            if signed || unsigned_ok {
                self.saturating_sub_sites.push(
                    crate::front::saturating_intrinsic::SaturatingSubSite {
                        result_var: res,
                        signed,
                    },
                );
            }
        }
        Ok(())
    }

    /// `lltype.cast_ptr_to_int` / `cast_int_to_ptr` as the host call
    /// `rtype_cast_ptr_to_int` / `rtype_cast_int_to_ptr` accept.
    /// `normalize_unary_op_name` rejects a unary op of either name.
    pub(in crate::front::mir) fn emit_host_cast_llop(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        operand: crate::flowspace::model::Variable,
        op: &str,
        result_ty: ValueType,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let concretetype = if op == "cast_int_to_ptr" {
            crate::model::ConcreteType::GcRef
        } else {
            crate::model::ConcreteType::Signed
        };
        let res = self.graph.alloc_value_var_with_type(concretetype);
        FunctionGraph::set_concretetype_of_inline(
            &operand,
            if op == "cast_int_to_ptr" {
                crate::model::ConcreteType::Signed
            } else {
                crate::model::ConcreteType::GcRef
            },
        );
        if op == "cast_ptr_to_int" {
            self.cast_ptr_to_int_src
                .insert(res.clone(), operand.clone());
        }
        let kind = match op {
            "cast_int_to_ptr" => cast_int_to_ptr_call(operand),
            "cast_ptr_to_int" => cast_ptr_to_int_call(operand),
            _ => OpKind::UnaryOp {
                op: op.to_string(),
                operand,
                result_ty,
            },
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind,
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `llop.threadlocalref_get(c_offset, c_loop_invariant)`
    /// (`rthread.py ThreadLocalField.getraw`). Result type is the
    /// projection's `ValueType` (`FIELDTYPE`).
    pub(in crate::front::mir) fn emit_threadlocalref_get(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        offset: crate::flowspace::model::Variable,
        loop_invariant: crate::flowspace::model::Variable,
        result_ty: ValueType,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        if let Some(shell) = crate::codewriter::annotation_state::valuetype_to_someshell(&result_ty)
        {
            *res.annotation.borrow_mut() = Some(std::rc::Rc::new(shell));
        }
        FunctionGraph::set_concretetype_of_inline(&offset, crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: "threadlocalref_get".to_string(),
                lhs: offset,
                rhs: loop_invariant,
                result_ty,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `llop.threadlocalref_store(c_offset, value)`
    /// (`rthread.py ThreadLocalField.setraw`). Upstream has no
    /// `rewrite_op_threadlocalref_store`; jtransform residualizes this
    /// as a plain call.
    pub(in crate::front::mir) fn emit_threadlocalref_store(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        offset: crate::flowspace::model::Variable,
        value: crate::flowspace::model::Variable,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
        *res.annotation.borrow_mut() = Some(std::rc::Rc::new(
            crate::translator::rtyper::llannotation::lltype_to_annotation(
                crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
            ),
        ));
        FunctionGraph::set_concretetype_of_inline(&offset, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&value, crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: "threadlocalref_store".to_string(),
                lhs: offset,
                rhs: value,
                result_ty: ValueType::Void,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// True when `var` is a `ConstInt` / `ConstUInt` already emitted in
    /// this graph. Used to refuse `threadlocalref_get` projection when
    /// the offset is not a call-site constant.
    pub(in crate::front::mir) fn var_is_const_int(
        &self,
        var: &crate::flowspace::model::Variable,
    ) -> bool {
        self.graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                op.result.as_ref() == Some(var)
                    && matches!(op.kind, OpKind::ConstInt(_) | OpKind::ConstUInt(_))
            })
        })
    }

    /// Rewrite a 64-bit float/int bitcast to the `longlong2float` pair.
    /// `to_float` is `longlong2float.longlong2float`; otherwise
    /// `longlong2float.float2longlong`.
    pub(in crate::front::mir) fn emit_longlong2float_call(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        operand: crate::flowspace::model::Variable,
        to_float: bool,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let (leaf, result_ty) = if to_float {
            ("longlong2float", ValueType::Float)
        } else {
            ("float2longlong", ValueType::Int)
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["longlong2float".to_string(), leaf.to_string()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![operand]),
                result_ty,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `f64::is_sign_negative(self)` — `core` has no graph body (Opaque), so the
    /// callsite would skip as an unregistered `FunctionPath`.  `is_sign_negative`
    /// is `float2longlong(x) < 0`: reinterpreting the f64 bits as `i64`, the sign
    /// bit is the MSB, so a negative `i64` means the sign bit is set (true for
    /// negatives, `-0.0`, and sign-bit-set NaN).  A pure `x < 0.0` would miss
    /// `-0.0` and NaN, so it must be the bit reinterpret.
    pub(in crate::front::mir) fn is_f64_is_sign_negative(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_sign_negative")
    }

    /// `f64::is_sign_positive(self)` — the positive half of the same sign-bit
    /// primitive as [`Self::is_f64_is_sign_negative`]. Reinterpret the f64 as
    /// a signed i64 and test `>= 0`; a numerical comparison with zero would
    /// incorrectly classify `-0.0` and sign-bit-set NaN.
    pub(in crate::front::mir) fn is_f64_is_sign_positive(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_sign_positive")
    }

    /// `majit_metainterp::jit::promote(x)` / the lower-layer
    /// `majit_ir::jit::promote(x)` decorator carrier =
    /// `hint(x, promote=True)` (`rlib/jit.py`), with the `promote_string`
    /// (`:118`) and `promote_unicode` (`:124`) siblings.  All wrappers carry their
    /// flag by name (each body is a bare `hint(x)`), so the callsite is
    /// recognised by the wrapper path, not the body.  Returns the
    /// synthesised `hint_*` marker leaf for the matched wrapper so the
    /// residual `OpKind::Call` reaches the matching `jtransform`
    /// `rewrite_op_hint` arm.
    ///
    /// `promote_string`/`promote_unicode` route to their own
    /// `hint_promote_string`/`hint_promote_unicode` markers (preserving the
    /// upstream hint-kind distinction in the IR) even though `jtransform`
    /// lowers all three through the same `<kind>_guard_value` family: pyre
    /// interpreter strings are `W_UnicodeObject` GC refs, not
    /// `Ptr(rstr.STR)`, so the string `guard_value` collapses to the
    /// ref-kind `r_guard_value` (see `jtransform::rewrite_op_hint`).
    pub(in crate::front::mir) fn jit_promote_marker(
        &self,
        reg: &RegularCall,
    ) -> Option<&'static str> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
            "majit_metainterp::jit::promote" | "majit_ir::jit::promote" => Some("hint_promote"),
            "majit_metainterp::jit::promote_string" => Some("hint_promote_string"),
            "majit_metainterp::jit::promote_unicode" => Some("hint_promote_unicode"),
            _ => None,
        }
    }

    /// Devirtualize a callsite of the blanket
    /// `impl<T, U: From<T>> Into<U> for T` (`core::convert::<Impl>::into`).
    ///
    /// The callsite's `generics.trait_refs` carries the resolved
    /// `U: From<T>` obligation as a trait ref whose `trait_decl_ref`
    /// names `core::convert::From` and whose `kind` is
    /// `TraitImpl { id }` — the def_id of the selected `impl From<T>
    /// for U`.  Two outcomes:
    ///
    /// - The obligation's decl-ref type args are equal (`T == U`):
    ///   the reflexive `impl<T> From<T> for T` was selected and the
    ///   whole conversion is a `T -> T` identity —
    ///   [`IntoDevirt::Identity`].
    /// - Otherwise the impl's `methods` table binds the single `From`
    ///   method to the concrete `from` FunDecl. Its owner, trait-impl id,
    ///   and method form the devirtualized call target —
    ///   [`IntoDevirt::Target`]. The id is load-bearing: several
    ///   `impl From<_> for Dynamic` bodies share the same owner/method pair.
    ///   `from` is an associated function (no `self` receiver), so the caller
    ///   must keep the `FunctionPath` shape (a `CallTarget::Method` hint would
    ///   bind the *argument* as a receiver).
    ///
    /// Returns `None` (caller keeps the blanket-into path) when the
    /// obligation is unresolved (`kind` is a clause/builtin rather
    /// than `TraitImpl`) or any table lookup misses.
    pub(in crate::front::mir) fn blanket_into_devirt(
        &self,
        reg: &RegularCall,
    ) -> Option<IntoDevirt> {
        if !self.callee_is_blanket_into(reg) {
            return None;
        }
        let trait_refs = reg.generics.get("trait_refs")?.as_array()?;
        for tref in trait_refs {
            let Some(tref) = traitref_unwrap(tref, self.llbc, 0) else {
                continue;
            };
            let Some(decl) = crate::front::clause_spec::trait_decl_pred(tref) else {
                continue;
            };
            let Some(decl_id) = decl.get("id").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            let is_from = self
                .llbc
                .trait_by_id(decl_id)
                .is_some_and(|td| td.item_meta.name_path() == "core::convert::From");
            if !is_from {
                continue;
            }
            // `U: From<T>` decl-ref generics carry `[U, T]`; equal
            // args select the reflexive blanket impl.  Compare by
            // hash-cons id so an inline `Value: [id, …]`
            // matches its `Deduplicated: id` reference.
            let types = decl.get("generics")?.get("types")?.as_array()?;
            if types.len() == 2 {
                let reflexive = match (ty_dedup_key(&types[0]), ty_dedup_key(&types[1])) {
                    (Some(a), Some(b)) => a == b,
                    _ => types[0] == types[1],
                };
                if reflexive {
                    return Some(IntoDevirt::Identity);
                }
            }
            let impl_id = traitref_impl_id(tref, self.llbc, 0)?;
            let ti = self.llbc.trait_impls_raw().get(impl_id as usize)?;
            let fn_id = ti.get("methods")?.as_array()?.iter().find_map(|m| {
                let tm = m.get("kind")?.get("TraitMethod")?.as_array()?;
                if tm.first()?.as_u64()? != decl_id {
                    return None;
                }
                m.get("skip_binder")?.get("id")?.as_u64()
            })?;
            let fd = self.llbc.fn_by_id(fn_id)?;
            let (owner, method) = impl_method_owner_for_fundecl(self.llbc, fd)?;
            let identity = crate::front::clause_spec::render_trait_impl(self.llbc, impl_id)?;
            let path = crate::parse::CallPath::from_segments(trait_impl_identity_path(
                &owner, &identity, &method,
            ));
            return Some(IntoDevirt::Target(path.segments));
        }
        None
    }

    /// `msg.into()` on a generic parameter bound `T: Into<String>` or
    /// `T: Into<Wtf8Buf>` — a `CallKind::Trait` whose trait ref is a
    /// *clause* (no resolved impl for [`Self::blanket_into_devirt`] to
    /// read).  The blanket `impl<T, U: From<T>> Into<U> for T` makes the
    /// result `U::from(self)`; for a string-family target the conversion
    /// is identity in the lifted value model (`String`, `&str`, `Wtf8`
    /// and `Wtf8Buf` all lower to the immutable rpy_string), so the
    /// caller may alias the destination to the argument.  The callsite's
    /// `dest` type *is* the trait ref's target type argument, so it is
    /// the only payload field consulted besides the trait identity.
    pub(in crate::front::mir) fn trait_clause_into_string_identity(
        &self,
        reg: &RegularCall,
        dest_ty: &TyRef,
    ) -> bool {
        let CallKind::Trait(v) = &reg.kind else {
            return false;
        };
        let Some(traitref) = v.as_array().and_then(|a| a.first()) else {
            return false;
        };
        let Some(trait_id) = traitref_decl_id(traitref, self.llbc, 0) else {
            return false;
        };
        let is_into = self
            .llbc
            .trait_by_id(trait_id)
            .is_some_and(|td| td.item_meta.name_path() == "core::convert::Into");
        is_into && tyref_is_string_value(dest_ty, self.llbc)
    }

    /// Whether the FunDecl's first signature input is the given ADT,
    /// possibly behind reference / raw-pointer layers (`self: Owner`,
    /// `&Owner`, `&mut Owner`, `*const/mut Owner`).  Distinguishes a
    /// real method from an associated function of the same impl block.
    pub(in crate::front::mir) fn first_input_is_adt(&self, fd: &FunDecl, adt_def_id: u64) -> bool {
        let Some(first) = fd.signature.inputs.first() else {
            return false;
        };
        let Some(mut v) = self.tyref_body(first) else {
            return false;
        };
        loop {
            let Some(obj) = v.as_object() else {
                return false;
            };
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                match self.llbc.dedup_body(id) {
                    Some(b) => {
                        v = b;
                        continue;
                    }
                    None => return false,
                }
            }
            if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            // `{"Ref": [region, ty, kind]}` / `{"RawPtr": [ty, kind]}`.
            if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
                match arr.get(1) {
                    Some(inner) => {
                        v = inner;
                        continue;
                    }
                    None => return false,
                }
            }
            if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
                match arr.first() {
                    Some(inner) => {
                        v = inner;
                        continue;
                    }
                    None => return false,
                }
            }
            return inline_adt_def_id(v)
                .is_some_and(|id| same_nominal_adt(self.llbc, id, adt_def_id));
        }
    }

    /// Decode the receiver type's ADT `def_id` from an `Impl` NameSeg
    /// payload.  Two shapes:
    ///
    /// - **InherentImplBlock**: `{"Ty": {"skip_binder": <TyExpr>}}` where
    ///   `<TyExpr>` is the type expression of `Self` in the impl block.
    ///   It can be inline (`Value: [id, body]`) or
    ///   deduplicated (`Deduplicated: id`).  When inline, the body
    ///   carries the ADT def_id directly (`{"Adt": {"id": <def_id>}}`);
    ///   when deduplicated, we consult [`Self::dedup_to_adt_def_id`]
    ///   which lazy-builds a per-LLBC `dedup_id → adt_def_id` index
    ///   from the inline forms scattered across the LLBC.
    ///
    /// - **TraitImplBlock**: `{"Trait": <trait_impl_id>}` — indirect
    ///   through the opaque `trait_impls` array to find the impl's
    ///   first concrete type argument, then resolve through the same
    ///   inline-or-dedup path.
    pub(in crate::front::mir) fn resolve_impl_owner_adt_def_id(
        &self,
        impl_payload: &serde_json::Value,
    ) -> Option<u64> {
        let obj = impl_payload.as_object()?;
        if let Some(ty) = obj.get("Ty") {
            let sb = ty.as_object()?.get("skip_binder")?;
            return self.resolve_tyexpr_to_adt_def_id(sb);
        }
        if let Some(trait_impl_id) = obj.get("Trait").and_then(serde_json::Value::as_u64) {
            let trait_impls = self.llbc.trait_impls_raw();
            let ti = trait_impls.get(trait_impl_id as usize)?;
            let first_ty = ti
                .as_object()?
                .get("impl_trait")?
                .as_object()?
                .get("generics")?
                .as_object()?
                .get("types")?
                .as_array()?
                .first()?;
            return self.resolve_tyexpr_to_adt_def_id(first_ty);
        }
        None
    }

    /// Resolve a Charon type expression to the underlying ADT
    /// `def_id`, whether the expression is inline
    /// (`Value: [_, body]`) or deduplicated
    /// (`Deduplicated: id`).  Returns `None` for non-ADT shapes
    /// (primitives, references, tuples).
    pub(in crate::front::mir) fn resolve_tyexpr_to_adt_def_id(
        &self,
        ty: &serde_json::Value,
    ) -> Option<u64> {
        resolve_tyexpr_to_adt_def_id_free(self.llbc, ty)
    }

    /// Alias `Arg::<T>::get(self, arg: OpArg) -> T`
    /// (`rustpython_compiler_core::bytecode::instruction`, instruction.rs)
    /// to its `OpArg` argument.  `Arg<T>` is the zero-sized oparg marker
    /// ([`tyref_is_bytecode_arg_marker`]) and `OpArg` is the transparent
    /// `struct OpArg(u32)` newtype, both modeled as a plain integer (the
    /// raw operand) — see [`tyref_is_oparg`].
    /// The body is `T::try_from(u32::from(arg)).unwrap()`, and every
    /// `OpArgType` (`VarNum` / `VarNums` / `u32`) is a transparent newtype
    /// over `u32`, so the result's bits equal `u32::from(arg)` — the
    /// argument's own integer value.  Returning `arg` (`args[1]`) directly
    /// is the identity the marker `self` is dropped around.
    ///
    /// Without this the `.get` method call lowers to `getattr(Integer,
    /// "get")` on the integer `self` marker — an attribute the annotator
    /// cannot type (the dominant `Cannot find attribute "get" on Integer`
    /// wall behind `complete_pending_blocks` for the opcode-dispatch
    /// handler family, which all read their operand via `<field>.get(op_arg)`).
    ///
    /// Identified by the second input being the `oparg::OpArg` type: no
    /// other `.get` (slice / `Vec` / map) takes an `OpArg`, and the
    /// concrete `OpArg` resolves robustly where the generic `self: Arg<T>`
    /// marker would not.  Returns `None` for any other `.get` so those
    /// keep the `Call` form.
    pub(in crate::front::mir) fn oparg_arg_get_alias(
        &self,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        if segments.last().map(String::as_str) != Some("get") {
            return None;
        }
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        // The `OpArg` argument (`inputs[1]` — `args[1]`, after the
        // `self` receiver) is the operand value.
        let oparg_ty = fd.signature.inputs.get(1)?;
        // `OpArg` is `Opaque` in the LLBC (external crate), so resolve it
        // by qualified name rather than structural shape.
        if !adt_path_of_tyref(oparg_ty, self.llbc).is_some_and(|p| p.ends_with("oparg::OpArg")) {
            return None;
        }
        // A fieldless-enum result (`Arg::<SpecialMethod>::get`) must keep
        // its `Ref` enum shape.  Aliasing to the bare integer operand
        // would make the downstream `match`'s `Rvalue::Discriminant` read
        // a `__discriminant` field off an integer base — a
        // `getfield_gc_i_pure/id>i` opname no blackhole handler covers.
        // The generic `fd.signature.output` is the type parameter `T`, so
        // key off the call's concrete destination type instead.  Only the
        // int-newtype results (`VarNum` / `u32`, whose bits *are* the
        // operand) alias; the enum keeps the canonical `…/rd>i` read.
        if self.tyref_is_fieldless_enum(dest_ty) {
            return None;
        }
        args.get(1).cloned()
    }

    /// Alias the transparent `OpArgType` conversions to their single
    /// argument.  Each oparg newtype is `#[repr(transparent)]` over
    /// `u32` and the fieldless oparg enums carry their operand as the
    /// discriminant, so — once the operand is modeled as an integer
    /// ([`tyref_is_oparg`], [`oparg_arg_get_alias`]) — every conversion
    /// below is the identity on that integer:
    ///   * the inherent `as_u32` / `as_usize` extractors and the
    ///     `from_u32` constructor (`newtype_oparg!`), and
    ///   * the `From` conversions (`u32::from(oparg)` for a newtype, the
    ///     `__discriminant` read for a fieldless enum).
    ///     Gated on the defining impl living in `bytecode::oparg` so the
    ///     generic `from` / `as_u32` / `as_usize` names cannot match
    ///     unrelated types.
    pub(in crate::front::mir) fn oparg_value_alias(
        &self,
        segments: &[String],
        args: &[Variable],
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        let [first, .., module, impl_seg, leaf] = segments else {
            return None;
        };
        if first.as_str() != "rustpython_compiler_core"
            || module.as_str() != "oparg"
            || impl_seg.as_str() != "<Impl>"
            || !matches!(leaf.as_str(), "as_u32" | "as_usize" | "from_u32" | "from")
        {
            return None;
        }
        Some(arg.clone())
    }

    /// Identity-passthrough stdlib wrappers that return their sole
    /// argument unchanged: `core::hint::must_use` (the `#[must_use]` lint
    /// shim, `must_use<T>(x: T) -> T { x }`) and `core::convert::identity`.
    /// Aliasing the destination to the argument keeps the result kind
    /// following the operand — both are generic over `T` — where a
    /// residual `Call` Skips "not registered" and a fixed-type stub would
    /// mistype a non-ref argument.  Mirrors the `Rvalue::Use` no-copy
    /// alias.  `must_use` is the immediate consumer of every
    /// `alloc::fmt::format` result, so leaving it residual blocks the
    /// whole error-message helper from rtyping.
    pub(in crate::front::mir) fn identity_passthrough_alias(
        &self,
        segments: &[String],
        args: &[Variable],
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        // `fmt::Arguments::from_str(s)` is the no-placeholder `format_args!`
        // — the constructed `Arguments` renders to exactly its `&str`
        // argument, so the rendered value *is* that string.  Alias it to
        // `s`, dropping the graph-less `Arguments::from_str` extern: the
        // literal-message chains it heads dead-end in synthetic panic
        // `Option`s pyre replaces with host exception constants, or thread
        // into `Write::write_fmt` (whose own graph-less extern keeps those
        // subjects residual) — no `alloc::fmt::format` consumes one.  This
        // is the same transparent passthrough `must_use` / `identity` take.
        // `from_str_nonconst(s) { Arguments::from_str(s) }` is the identical
        // constructor emitted for a no-placeholder `format_args!` in a
        // non-`const` body (the C-module `#[new]`/setter gateways), so it
        // aliases the same way.
        if fmt_path_ends_with(segments, &["Arguments", "from_str"])
            || fmt_path_ends_with(segments, &["Arguments", "from_str_nonconst"])
        {
            return Some(arg.clone());
        }
        // `rlib/debug.py check_not_access_directly` — upstream's
        // `specialize_call` is `hop.inputarg(hop.args_r[0], arg=0)`, i.e. the
        // identity once the annotator's assert has run. The assert half is
        // registered in `annotator/builtin.rs`; this is the other half, and
        // without it the marker would sit as a residual call on the caller's
        // path instead of vanishing the way it does upstream.
        if fmt_path_ends_with(segments, &["debug", "check_not_access_directly"]) {
            return Some(arg.clone());
        }
        let [a, b, c] = segments else {
            return None;
        };
        let is_identity = a == "core"
            && ((b == "hint" && c == "must_use") || (b == "convert" && c == "identity"));
        is_identity.then(|| arg.clone())
    }

    /// The `NonNull<T>` niche identities.  A `NonNull<T>` modelling an object
    /// pointer is represented by its raw `*mut T`, so both constructors and
    /// the pointer accessor are the identity on their single operand:
    ///
    /// - `NonNull::new(p) -> Option<NonNull<T>>` — the niche `Option`
    ///   represents `None` as null and `Some(p)` as the pointer, so the
    ///   constructed value *is* `p`.  Gated on the destination being that
    ///   niche `Option<NonNull>` (`tyref_is_niche_option_ptr`).
    /// - `NonNull::as_ptr(self) -> *mut T` — reads the pointer back out, which
    ///   is the receiver itself.
    ///
    /// Aliasing drops the graph-less `NonNull` externs (otherwise "not
    /// registered" in the rtyper census), mirroring the payload/discriminant
    /// niche folds.
    pub(in crate::front::mir) fn nonnull_new_identity_alias(
        &self,
        segments: &[String],
        args: &[Variable],
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        if fmt_path_ends_with(segments, &["non_null", "NonNull", "new"])
            && self.tyref_is_niche_option_ptr(dest_ty)
        {
            return Some(arg.clone());
        }
        if fmt_path_ends_with(segments, &["non_null", "NonNull", "as_ptr"]) {
            return Some(arg.clone());
        }
        None
    }
}
