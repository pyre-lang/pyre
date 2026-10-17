//! Statement / rvalue / place → SpaceOperation (`rpython/flowspace/operation.py`).

use super::*;

impl<'a> Lowering<'a> {
    /// `ptr::add` consumed only as `(*dest).field` on a struct pointee.
    ///
    /// Records the GcArray header and the index, aliases the destination
    /// to the item pointer, and links the call edge. The field use emits
    /// `getinteriorfield` / `setinteriorfield`
    /// (`rewrite_op_getinteriorfield`). A miss returns `Ok(false)` so
    /// `direct_ptradd` still owns byte, integer, and `sub` offsets.
    /// Declines without inserting the alias when the recovered header has
    /// no Variable in this block.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::front::mir) fn try_lower_struct_field_ptr_add(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let Some(traced) = struct_field_ptr_add_trace(
            reg,
            args.len(),
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            arg_locals.get(1).copied().flatten(),
            dest_local,
            self.body,
            self.llbc,
            self.tombstoned_leaves,
        ) else {
            return Ok(false);
        };
        // `realias_operand` reads the local's binding at the field use.
        // A write after `ptr::add` would name a different element or header.
        let watched = match &traced.header {
            TracedPtrAddHeader::EntriesItem { local } => [*local, traced.index_local],
            TracedPtrAddHeader::EntryPtr { recv_local, .. } => [*recv_local, traced.index_local],
        };
        if locals_assigned_reachable(self.body, self.llbc, target, &watched) {
            return Ok(false);
        };
        let index_var = args.get(1).cloned().ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: struct-field ptr::add lost its index operand"
            ))
        })?;
        let header = match traced.header {
            TracedPtrAddHeader::EntriesItem { local } => {
                let Some(var) = self.local_var.get(local).cloned().flatten() else {
                    return Ok(false);
                };
                InteriorHeader::Local {
                    local,
                    var: var.one()?,
                }
            }
            TracedPtrAddHeader::EntryPtr {
                recv_local,
                recv_ty,
            } => {
                let Some(var) = self.local_var.get(recv_local).cloned().flatten() else {
                    return Ok(false);
                };
                let Some((field, ty)) = self.entries_field_descriptor(&recv_ty) else {
                    return Ok(false);
                };
                InteriorHeader::Entries {
                    recv_local,
                    recv_var: var.one()?,
                    field,
                    ty,
                }
            }
        };
        let base = args.first().cloned().ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: struct-field ptr::add lost its pointer operand"
            ))
        })?;
        self.interior_field_alias.insert(
            dest_local,
            InteriorFieldAlias {
                header,
                index_local: traced.index_local,
                index_var,
                array_type_id: traced.array_type_id,
            },
        );
        self.alias_dest_to_arg0_inherit(dest_local, base, arg_locals);
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `*p.add(i)` where `p` is the data pointer of a `&[u8]` byte string.
    ///
    /// `<[u8]>::as_ptr` aliases that pointer to the string and marks it.
    /// The `.add` is the index (`rstr.py` `ll_stritem_nonneg`), so the
    /// call is `__string_byte_getitem` and the following `*p` is that
    /// byte. A pointer that escapes, a write, an `i8` pointee, or
    /// `add(LOWLEVEL_STRING_CHARS_OFFSET)` stays `direct_ptradd`.
    /// Items-block `.add`s keep their own intercepts.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::front::mir) fn try_lower_byte_view_ptr_add(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        arg_locals: &[Option<usize>],
        index_operand: Option<&Operand>,
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if args.len() != 2 || !regular_call_is_ptr_add(reg, self.llbc) {
            return Ok(false);
        }
        let Some(first_ty) = first_arg_ty else {
            return Ok(false);
        };
        if !tyref_is_raw_u8_ptr(first_ty, self.llbc) {
            return Ok(false);
        }
        let Some(base_local) = arg_locals.first().copied().flatten() else {
            return Ok(false);
        };
        if !self.string_byte_view_locals.contains(&base_local) {
            return Ok(false);
        }
        if index_operand
            .is_some_and(|op| operand_is_lowlevel_string_chars_offset(self.body, self.llbc, op))
        {
            return Ok(false);
        }
        if self.ptr_add_has_later_intercept(reg, args.len(), arg_locals, first_arg_ty, dest_local) {
            return Ok(false);
        }
        if !add_dest_used_only_as_single_read_deref(self.llbc, self.body, dest_local) {
            return Ok(false);
        }
        let bb_id = self.block_id[mir_bb];
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__string_byte_getitem".to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![args[0].clone(), args[1].clone()]),
                result_ty: ValueType::Int,
            },
        });
        // The paired `*p` must not `RawLoad` the byte already read.
        // `array_type_id` stays unset: the read-only gate excludes a
        // store, and a string has no `setitem` (`rstr.py` `ll_str2mutable`).
        self.index_elem_alias.insert(
            dest_local,
            IndexElemAlias {
                base_local: Some(base_local),
                base_var: args[0].clone(),
                index_local: arg_locals.get(1).copied().flatten(),
                index_var: args[1].clone(),
                item_ty: ValueType::Int,
                array_type_id: None,
                rust_vec: None,
                list_root: None,
                items_owner: None,
                no_element: false,
            },
        );
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `FieldRead entries` descriptor for an `entry_ptr` receiver.
    ///
    /// The field is looked up by name on the struct, not taken as index 0.
    /// `resolve_adt_field` peels `&RDict`, so the result is the `*mut
    /// GcEntries` header `getinteriorfield` adds `array_items_base` to.
    pub(in crate::front::mir) fn entries_field_descriptor(
        &self,
        recv_ty: &TyRef,
    ) -> Option<(FieldDescriptor, ValueType)> {
        let index = struct_field_index_named(recv_ty, self.llbc, "entries")?;
        let payload = serde_json::json!([null, index]);
        let (owner_root, field_name, field_ty, owner_id, _host_index) =
            self.resolve_adt_field(recv_ty, &payload)?;
        let field = FieldDescriptor::new(field_name, Some(owner_root))
            .with_owner_id(owner_id)
            .with_base_is_deref(true);
        let ty = tyref_to_value_type_with(
            &field_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        Some((field, ty))
    }

    /// The GcArray object a recorded struct-field `ptr::add` addresses.
    ///
    /// `Local` re-resolves the header MIR local. `Entries` emits
    /// `FieldRead entries` (`pure: false`) and returns that result, which
    /// is the base the following interior op must use.
    pub(in crate::front::mir) fn interior_header_base(
        &mut self,
        bb_id: BlockId,
        alias: &InteriorFieldAlias,
    ) -> Result<Variable, LowerError> {
        match &alias.header {
            InteriorHeader::Local { local, var } => {
                Ok(self.realias_operand(Some(*local), var.clone()))
            }
            InteriorHeader::Entries {
                recv_local,
                recv_var,
                field,
                ty,
            } => {
                let recv = self.realias_operand(Some(*recv_local), recv_var.clone());
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: OpKind::FieldRead {
                        base: recv,
                        field: field.clone(),
                        ty: ty.clone(),
                        pure: false,
                    },
                });
                Ok(res)
            }
        }
    }

    /// `(*add).field` when `add` is a recorded struct-field `ptr::add`.
    ///
    /// Returns `Ok(None)` when `inner` is not exactly `*Local` of such a
    /// destination, so the ordinary `FieldRead` still runs. The interior
    /// base is the recovered header, ignoring the item-pointer binding
    /// `alias_dest_to_arg0_inherit` left on the destination.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::front::mir) fn try_emit_interior_field_read(
        &mut self,
        mir_bb: usize,
        inner: &Place,
        owner_root: &str,
        field_name: &str,
        owner_id: &Option<majit_ir::descr::StructId>,
        field_ty: &TyRef,
        place_ty: &TyRef,
        container_is_enum: bool,
        owner_is_closure_env: bool,
    ) -> Result<Option<Variable>, LowerError> {
        let PlaceKind::Projection(pre, ProjectionElem::Atom(name)) = &inner.kind else {
            return Ok(None);
        };
        if name != "Deref" {
            return Ok(None);
        }
        let PlaceKind::Local(local) = pre.kind else {
            return Ok(None);
        };
        let Some(alias) = self.interior_field_alias.get(&(local as usize)).cloned() else {
            return Ok(None);
        };
        let bb_id = self.emit_bb(mir_bb);
        let base = self.interior_header_base(bb_id, &alias)?;
        let index = self.realias_operand(Some(alias.index_local), alias.index_var);
        let item_ty = adt_field_read_value_type(
            place_ty,
            field_ty,
            container_is_enum,
            owner_is_closure_env,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let mut field = FieldDescriptor::new(field_name, Some(owner_root.to_string()))
            .with_owner_id(*owner_id)
            .with_base_is_deref(true)
            .with_inline_vec(
                field_ty_is_inline_vec(field_ty, self.llbc)
                    || field_ty_is_inline_vec(place_ty, self.llbc),
            );
        if tyref_rust_vec_item_kind(field_ty, self.llbc).is_some()
            || tyref_rust_vec_item_kind(place_ty, self.llbc).is_some()
        {
            field = field.with_taken_by_address(true);
        }
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::InteriorFieldRead {
                base,
                index,
                field,
                item_ty,
                array_type_id: Some(alias.array_type_id),
            },
        });
        Ok(Some(res))
    }

    /// `(*add).field = value` for a recorded struct-field `ptr::add`.
    ///
    /// `InteriorFieldWrite.value` is a [`Variable`]. A pooled
    /// `LinkArg::Const` (`field_write_inline_const`) is materialised with
    /// [`link_arg_as_alias_op`] first. An unpooled constant is an error,
    /// not a `FieldWrite` of the item pointer.
    pub(in crate::front::mir) fn emit_interior_field_write(
        &mut self,
        mir_bb: usize,
        bb_id: BlockId,
        dest_local: usize,
        field: FieldDescriptor,
        value: LinkArg,
        item_ty: ValueType,
    ) -> Result<(), LowerError> {
        let Some(alias) = self.interior_field_alias.get(&dest_local).cloned() else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: interior field write lost its ptr::add alias"
            )));
        };
        let base = self.interior_header_base(bb_id, &alias)?;
        let index = self.realias_operand(Some(alias.index_local), alias.index_var);
        let value_var = match &value {
            LinkArg::Value(var) => var.clone(),
            LinkArg::Const(_) => {
                let op = link_arg_as_alias_op(&value, &item_ty).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: interior field write constant is not a pooled \
                         int, bool, or float"
                    ))
                })?;
                let var = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(var.clone()),
                    kind: op,
                });
                var
            }
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::InteriorFieldWrite {
                base,
                index,
                field,
                value: value_var,
                item_ty,
                array_type_id: Some(alias.array_type_id),
            },
        });
        Ok(())
    }

    /// If `elem` writes a struct field and `rvalue` is a plain pooled
    /// constant matching the field's value bank — integer / bool for an
    /// int-kind field, float for a float-kind field — return the matching
    /// [`ConstValue`] so the `FieldWrite` carries it inline
    /// (`setfield_gc_i/rcd` short form or `/rid` pool slot for an int,
    /// `setfield_gc_f/rfd` pool slot for a float) rather than forcing it
    /// into a register with a `ConstInt` / `ConstFloat` + copy.
    ///
    /// RPython keeps the `Constant` box as a `setfield_gc` argument
    /// (codewriter args are `AbstractValue`, never pre-materialised) and
    /// leaves the short-vs-pool encoding to the assembler
    /// (`assembler.py` `emit_const`).  The assembler `FieldWrite` arm derives
    /// the value kind from the constant via `constant_kind` and routes a
    /// float to a pooled `f` slot (`assembler.rs` `emit_const_f`), which
    /// the walker resolves through the constants window of `registers_f`.
    ///
    /// Ref-kind constants are not pooled at this layer: a string / char /
    /// fn-pointer constant lowers to a runtime `Call`
    /// (`build_rvalue`'s `DecodedConst::Str` / `FnPath` arms), not a
    /// poolable `ConstPtr`, so it keeps the materialised path.  A plain
    /// `Deref` (`*p = v`) also keeps it, lowering to a `__deref_write`
    /// Call whose args must be Variables; an `index_mut`-aliased `Deref`
    /// and a `[i]` subscript both lower to `setarrayitem_gc_*`, whose
    /// value operand can stay an inline `Constant` like `setfield_gc_*`
    /// (jtransform.py `rewrite_op_setarrayitem` passes `op.args[2]` verbatim).
    pub(in crate::front::mir) fn field_write_inline_const(
        &self,
        inner: &Place,
        elem: &ProjectionElem,
        rvalue: &Rvalue,
        dest_ty: &TyRef,
    ) -> Option<ConstValue> {
        // A `Field`, a `[i]` subscript, or an `index_mut`-aliased `Deref`
        // lowers to a `setfield_gc_*` / `setarrayitem_gc_*` whose value
        // operand can stay an inline `Constant`.  A plain `Deref` lowers
        // to a `__deref_write` Call (Variable args only) and is excluded.
        let inlinable_target = match elem {
            // A pair slice item write is a helper call taking a Variable.
            ProjectionElem::Tagged(v)
                if v.get("Index").is_some() && self.item_seq_of_place(inner).is_some() =>
            {
                false
            }
            ProjectionElem::Tagged(v) => v
                .as_object()
                .is_some_and(|m| m.contains_key("Field") || m.contains_key("Index")),
            ProjectionElem::Atom(s) if s == "Deref" => matches!(
                &inner.kind,
                PlaceKind::Local(i) if self.index_elem_alias.contains_key(&(*i as usize))
            ),
            _ => false,
        };
        if !inlinable_target {
            return None;
        }
        // Only a bare constant operand qualifies; a computed rvalue still
        // flows through `build_rvalue`.
        let Rvalue::Use(Operand::Const(value), _) = rvalue else {
            return None;
        };
        // Inline only a genuinely pooled constant matched to the target's
        // value bank: int / bool into an int-kind target (`setfield_gc_i`
        // / `setarrayitem_gc_i`; `getkind(Bool) == 'int'`), float into a
        // float-kind target (`setfield_gc_f`).  A `Bool`-typed target
        // (e.g. a `bool` array element) banks `i` just like an `Int`
        // field.  `Str` / `FnPath` constants lower to a `Call`, not a
        // poolable `ConstPtr`, so they fall through to the materialised
        // path.
        match (
            tyref_to_value_type_with(
                dest_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            decode_constant(self.llbc, value).ok()?,
        ) {
            (ValueType::Int, DecodedConst::Int(n)) => Some(ConstValue::Int(n)),
            (ValueType::Int, DecodedConst::UniChar(code)) => {
                char::from_u32(code).map(|c| ConstValue::UniStr(c.to_string()))
            }
            // Charon records a generic integer 0 as unsigned. `intmask`
            // (`rarithmetic.py`) is the signed word; `allocated: 0` on
            // `W_ListObject` is `isize` (`rlist.py` `ll_newlist`).
            (ValueType::Int, DecodedConst::UInt(n)) if n <= i64::MAX as u64 => {
                Some(ConstValue::Int(n as i64))
            }
            (ValueType::Unsigned, DecodedConst::UInt(n)) => Some(ConstValue::UInt(n)),
            (ValueType::Unsigned, DecodedConst::Int(n)) => Some(ConstValue::UInt(n as u64)),
            (ValueType::Int | ValueType::Bool, DecodedConst::Bool(b)) => Some(ConstValue::Bool(b)),
            (ValueType::Float, DecodedConst::Float(bits)) => Some(ConstValue::Float(bits)),
            _ => None,
        }
    }

    /// Emit the side-effectful write op for an `Assign` whose dest is
    /// a `Projection(inner, elem)`. `value` is the freshly computed
    /// rvalue.  `dest_ty` is the projected place's own `TyRef` — the
    /// field type AFTER generic substitution, mirroring the typed
    /// `FieldRead` arm in `resolve_place` (the declaration-side field
    /// ty is the generic param for generic ADTs, which
    /// `tyref_to_value_type` can only degrade to `Ref(None)`).
    /// `heaptracker.py all_fielddescrs` flattens a by-value nested
    /// struct into dotted leaves on the outer GC owner. `W_ListObject`
    /// stores Integer/Float items as an inline `IntArray` /
    /// `FloatArray`; a read or write of `.block` / `.len` is therefore
    /// `getfield_gc` / `setfield_gc(l, int_items.block)` — the same field
    /// `_handle_list_call` and `rlist.py` `l.items` / `l.items = newitems`
    /// use. Without the flatten, Charon emits `FieldRead`/`FieldWrite(block)`
    /// on the nested `IntArray` copy, so heap CSE of `int_items.block`
    /// misses the access and a `getfield` of `IntArray.block` at offset 0
    /// loads the typed-block capacity header.
    pub(in crate::front::mir) fn flatten_list_storage_field(
        &self,
        inner: &Place,
        elem: &ProjectionElem,
    ) -> Option<(String, String, Option<majit_ir::descr::StructId>)> {
        let leaf_payload = match elem {
            ProjectionElem::Tagged(v) => v.as_object().and_then(|m| m.get("Field"))?,
            _ => return None,
        };
        let (_, leaf_name, _, _, _) = self.resolve_adt_field(&inner.ty, leaf_payload)?;
        if !matches!(leaf_name.as_str(), "block" | "len") {
            return None;
        }
        let mut cur = inner;
        loop {
            match &cur.kind {
                PlaceKind::Projection(next, ProjectionElem::Atom(s)) if s == "Deref" => {
                    cur = next;
                }
                PlaceKind::Projection(nested_base, ProjectionElem::Tagged(v)) => {
                    let nested_payload = v.as_object().and_then(|m| m.get("Field"))?;
                    let (owner, nested_name, _, owner_id, _) =
                        self.resolve_adt_field(&nested_base.ty, nested_payload)?;
                    if is_flattened_storage_leaf(&owner, &nested_name, &leaf_name) {
                        return Some((owner, format!("{nested_name}.{leaf_name}"), owner_id));
                    }
                    return None;
                }
                _ => return None,
            }
        }
    }

    /// The GC pointer a `p as usize` chain produced, if `var` is that
    /// address integer or an `r_uint` retype of it.
    pub(in crate::front::mir) fn address_cast_pointer(&self, var: &Variable) -> Option<Variable> {
        if let Some(orig) = self.cast_ptr_to_int_src.get(var) {
            return Some(orig.clone());
        }
        if let Some(orig) = self.raw_address_uint_src.get(var) {
            return Some(orig.clone());
        }
        let (block_id, idx) = resolve_to_producer_op(&self.graph, var)?;
        let op = self
            .graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)?
            .operations
            .get(idx)?;
        if op
            .result
            .as_ref()
            .is_some_and(|result| self.cast_ptr_to_int_src.contains_key(result))
        {
            return self.cast_ptr_to_int_src.get(op.result.as_ref()?).cloned();
        }
        if op
            .result
            .as_ref()
            .is_some_and(|result| self.raw_address_uint_src.contains_key(result))
        {
            return self.raw_address_uint_src.get(op.result.as_ref()?).cloned();
        }
        // `hint` is `same_as` of its value (`flowspace_adapter`,
        // `jtransform.py` `rewrite_op_hint`). `Function.getcode` is
        // `promote(self.code as usize) as PyObject` (`jit.py` `promote`):
        // the promote result is a fresh variable, so the map above misses
        // unless the walk steps through the hint.
        let hint_value = match &op.kind {
            OpKind::Hint { value, .. } => Some(value.clone()),
            _ => None,
        };
        if let Some(value) = hint_value {
            return self.address_cast_pointer(&value);
        }
        let OpKind::Call { target, args, .. } = &op.kind else {
            return None;
        };
        let is_address_retype = match target {
            CallTarget::FunctionPath { segments, .. } => segments.last().is_some_and(|leaf| {
                // `hint_promote` is `same_as` of its argument
                // (`jtransform.py` `rewrite_op_hint`). `Function.getcode`
                // promotes the address word, so the int→ptr half has to
                // see the pointer the word came from.
                matches!(leaf.as_str(), "r_uint" | "cast_ptr_to_int" | "hint_promote")
            }),
            _ => false,
        };
        if !is_address_retype {
            return None;
        }
        let arg = args.first()?.as_variable()?;
        self.address_cast_pointer(arg)
    }

    /// `var` or the op that produced it (across one inputarg link) is a
    /// GCREF word built from an address cast.
    pub(in crate::front::mir) fn var_tracks_pointer_word(&self, var: &Variable) -> bool {
        if self.pointer_word_vars.contains(var) {
            return true;
        }
        let Some((block_id, idx)) = resolve_to_producer_op(&self.graph, var) else {
            return false;
        };
        self.graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .and_then(|block| block.operations.get(idx))
            .is_some_and(|op| {
                crate::model::cast_instance_root(&op.kind) == Some("GCREF")
                    || op
                        .result
                        .as_ref()
                        .is_some_and(|result| self.pointer_word_vars.contains(result))
            })
    }

    pub(in crate::front::mir) fn var_tracks_pointer_word_array(&self, var: &Variable) -> bool {
        if self.pointer_word_arrays.contains(var) {
            return true;
        }
        let Some((block_id, idx)) = resolve_to_producer_op(&self.graph, var) else {
            return false;
        };
        self.graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .and_then(|block| block.operations.get(idx))
            .and_then(|op| op.result.as_ref())
            .is_some_and(|result| self.pointer_word_arrays.contains(result))
    }

    pub(in crate::front::mir) fn emit_projection_write(
        &mut self,
        mir_bb: usize,
        inner: Place,
        elem: ProjectionElem,
        mut value: LinkArg,
        dest_ty: &TyRef,
    ) -> Result<(), LowerError> {
        // `rewrite_op_setfield` drops a void store (`getkind == 'v'`).
        // `RDict.hash_builder` is `BuildHasherDefault<H>` (`rordereddict.py`
        // has no hasher field).
        if tyref_is_void_zst(dest_ty, self.llbc) {
            return Ok(());
        }
        // `(*s)[i] = value` over a pair slice, or `a[i] = value` over a
        // raw-buffer array, writes through the item pointer.
        if let ProjectionElem::Tagged(v) = &elem
            && let Some(index_payload) = v.get("Index")
            && let Some(seq) = self.item_seq_of_place(&inner)
        {
            let value = value.as_variable().cloned().ok_or_else(|| {
                LowerError::Schema(format!("bb{mir_bb}: pair slice item write of a constant"))
            })?;
            let value_ty = tyref_to_value_type_with(
                dest_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            );
            return self.pair_setitem(mir_bb, seq, &index_payload.clone(), value, &value_ty);
        }
        if let Some((owner, flat_name, owner_id)) = self.flatten_list_storage_field(&inner, &elem) {
            let list_place = peel_nested_storage_place(inner);
            let base = self.resolve_place(mir_bb, list_place)?;
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base,
                    field: FieldDescriptor::new(flat_name, Some(owner)).with_owner_id(owner_id),
                    value,
                    ty: tyref_to_value_type_with(
                        dest_ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ),
                },
            });
            return Ok(());
        }
        // A plain `*p = v` (`__deref_write` Call below) reads a
        // materialised register: its Call args must be Variables, so a
        // constant operand must already have been forced to a Variable.
        // `field_write_inline_const` only mints a `LinkArg::Const` for the
        // setfield / setarrayitem targets (Field, `[i]`, aliased Deref),
        // never the plain-Deref Call, so this expect never trips.
        let value_var = |v: &LinkArg| {
            v.as_variable()
                .expect("__deref_write carries a materialised Variable value")
                .clone()
        };
        let inner_local = match &inner.kind {
            PlaceKind::Local(i) => Some(*i as usize),
            _ => None,
        };
        // `*slot = v` over a pair-slice item address (`IterMut` payload,
        // `first` / `get` of a mut slice): write through `ptr` at index 0.
        if matches!(elem, ProjectionElem::Atom(ref s) if s == "Deref")
            && let Some(local) = inner_local
            && matches!(
                self.local_var.get(local).and_then(|slot| slot.as_ref()),
                Some(LocalValue::ItemAddr(_))
            )
            && let Some((kind, item_ty)) = self.item_addr_slot_kind(local)
        {
            let addr = self.item_addr_var(mir_bb, local)?;
            let value = value.as_variable().cloned().ok_or_else(|| {
                LowerError::Schema(format!("bb{mir_bb}: pair slice item write of a constant"))
            })?;
            let index = self.emit_const_uint(mir_bb, 0);
            self.emit_slice_setitem(mir_bb, addr, index, kind, value, &item_ty);
            return Ok(());
        }
        // Container shape, captured before `inner` is consumed below —
        // the write-side counterpart of the `resolve_place` Field arm.
        let base_is_deref = inner.is_deref_projection();
        // `(*dest).field` where `dest` was bound by `ptr::add`. Captured
        // before `resolve_place` collapses the deref to the item pointer:
        // the interior base is the GcArray header, not that pointer.
        let interior_dest = struct_field_deref_local(&inner, &elem);
        // RPython writeanalyze keys an array effect by the base box's
        // concrete ARRAY type (`op.args[0].concretetype`). Charon keeps the
        // same owner on the pre-projection Place, so preserve it before
        // consuming `inner`.
        let (value_local, index_spelling) = index_identity_inputs(&inner, self.llbc);
        let (projection_array_type_id, projection_array_nolength) =
            fixed_array_index_identity(value_local, &index_spelling);
        // `*p = v` on a `*mut T` is `raw_store` (`rewrite_op_raw_store`).
        // A `&mut T` is a field or array place, not an integer address.
        let deref_base_is_raw_ptr = tyref_node(&inner.ty, self.llbc)
            .and_then(|node| strip_ty_wrappers(node, self.llbc))
            .and_then(|node| node.as_object())
            .is_some_and(|obj| obj.contains_key("RawPtr"));
        // `(*p).field = v` now projects `{"Field": [null, index]}` (or
        // `[variant, index]`). The read side resolves that index through
        // the base type; capture the type before `inner` is consumed.
        let field_base_ty = clone_tyref(&inner.ty);
        let base = self.resolve_place(mir_bb, inner)?;
        let bb_id = self.emit_bb(mir_bb);
        let op = match &elem {
            ProjectionElem::Atom(s) if s == "Deref" => {
                if let Some(alias) = inner_local
                    .and_then(|i| self.index_elem_alias.get(&i))
                    .cloned()
                {
                    // `*p = val` where `p` was bound by a
                    // devirtualized workspace `index_mut` call is the
                    // write half of `arr[i] = val` — emit the array
                    // write directly (setarrayitem).  The index call
                    // terminates its own block, so the base/index
                    // operands are typically rebound to fresh inputarg
                    // Variables here; re-resolve them through
                    // `local_var` by their source local and fall back
                    // to the recorded Variable for a constant operand.
                    let base_var = alias.base_var.clone();
                    let mut arr = self.realias_operand(alias.base_local, alias.base_var);
                    let idx = self.realias_operand(alias.index_local, alias.index_var);
                    // The read already used `ll_items` (`block`). Preferring
                    // the wrapper local here would cast `BytesArray` to
                    // `[str]` without that field (`ll_setitem_fast`).
                    if let Some(owner) = alias.items_owner
                        && alias.base_local.is_some()
                    {
                        arr = self.emit_string_array_items_read(bb_id, arr, owner);
                    }
                    if let Some(kind) = alias.rust_vec {
                        let op = OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: rust_vec_helper_segments(
                                    majit_ir::rvec::VecOp::SetItem,
                                    kind,
                                ),
                                fun_decl_id: None,
                            },
                            args: vec![LinkArg::from(arr), LinkArg::from(idx), value.clone()],
                            result_ty: ValueType::Void,
                        };
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: None,
                            kind: op,
                        });
                        return Ok(());
                    }
                    if alias.array_type_id.as_deref() == Some(STRING_GCREF_GCARRAY_TYPE_ID) {
                        // The manual Rust shadow-stack reload is the internal
                        // GCREF word. At the source graph boundary RPython's
                        // rlist recasts it to the external StringRepr before
                        // setitem; the typer lowers this marker back to the
                        // exact cast_opaque_ptr operation.
                        value = self.narrow_value_to_instance_root(bb_id, value, "str");
                    } else if let Some(item) =
                        alias.list_root.as_deref().and_then(list_spelling_item)
                    {
                        // Same external-item boundary as `FixedObjectArray::set_ref`:
                        // setitem must see the element class before the ListDef
                        // generalizes (`ll_setitem_fast`, `rlist.py`).
                        value = self.narrow_value_to_instance_root(bb_id, value, item);
                    }
                    let cast_root = if let Some(root) = alias.list_root.as_deref() {
                        Some(root)
                    } else {
                        match alias.array_type_id.as_deref() {
                            Some("[i64]" | "[f64]") => alias.array_type_id.as_deref(),
                            Some(id) if id == STRING_GCREF_GCARRAY_TYPE_ID => Some(id),
                            _ => None,
                        }
                    };
                    let arr = if let Some(root) = cast_root {
                        self.narrow_value_to_instance_root(bb_id, LinkArg::Value(arr), root)
                            .as_variable()
                            .expect("a materialized typed-items base stays a Variable")
                            .clone()
                    } else {
                        arr
                    };
                    let stored_is_pointer_word = value
                        .as_variable()
                        .is_some_and(|var| self.var_tracks_pointer_word(var));
                    if stored_is_pointer_word {
                        self.pointer_word_arrays.insert(arr.clone());
                        self.pointer_word_arrays.insert(base_var);
                    }
                    let item_ty = if stored_is_pointer_word {
                        ValueType::Ref(None)
                    } else {
                        alias.item_ty.clone()
                    };
                    OpKind::ArrayWrite {
                        base: arr,
                        index: idx,
                        value: value.clone(),
                        item_ty,
                        array_type_id: alias.array_type_id.clone(),
                        nolength: crate::front::typestr::nolength_from_array_type_id(
                            alias.array_type_id.as_deref(),
                        ),
                    }
                } else if let Some(owner) = self.string_array_remove_owner() {
                    // Tail of RPython `ll_delitem_nonneg`: after
                    // `ll_arraymove`, clear `l[newlength]` to the null GC
                    // pointer before shrinking the logical length.  Rust
                    // spells this one site as `*p = null_mut()`, but the
                    // translated operation is a setitem on the same
                    // GcArray(GCREF) as the move.
                    let list = self
                        .local_var
                        .get(1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: string-array remove lost self before null clear"
                            ))
                        })?
                        .one()?;
                    let block = self.emit_string_array_items_read(bb_id, list.clone(), owner);
                    let arr = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            LinkArg::Value(block),
                            STRING_GCREF_GCARRAY_TYPE_ID,
                        )
                        .as_variable()
                        .expect("a string-items view stays a Variable")
                        .clone();
                    let length = self.emit_string_array_len_read(bb_id, list, owner);
                    let one = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(one.clone()),
                        kind: OpKind::ConstUInt(1),
                    });
                    let newlength = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(newlength.clone()),
                        kind: OpKind::BinOp {
                            op: "sub".to_string(),
                            lhs: length,
                            rhs: one,
                            result_ty: ValueType::Unsigned,
                        },
                    });
                    value = self.narrow_value_to_instance_root(bb_id, value, "str");
                    OpKind::ArrayWrite {
                        base: arr,
                        index: newlength,
                        value: value.clone(),
                        item_ty: ValueType::Str,
                        array_type_id: Some(STRING_GCREF_GCARRAY_TYPE_ID.to_string()),
                        nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                            STRING_GCREF_GCARRAY_TYPE_ID,
                        )),
                    }
                } else if let Some(local) = inner_local
                    && !self.index_elem_alias.contains_key(&local)
                    && let Some((item_ty, array_type_id)) =
                        raw_ptr_typed_items_element(&field_base_ty, self.llbc)
                    && base_traces_to_typed_items_block_accessor(self.body, local, self.llbc)
                {
                    // `*(typed_items_block_items_base(..) as *mut i64) = v`
                    // is `setarrayitem` on `GcArray(Signed|Float)` at index 0,
                    // the write half of the `getarrayitem` the matching deref
                    // read emits. `heap.py` `emitting_operation` treats
                    // `raw_store` as having no effect on GC array caches, so
                    // a later `getarrayitem` would reuse the pre-store word.
                    // `.add(idx)` stores already take the `index_elem_alias`
                    // arm above.
                    let arr = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            LinkArg::Value(base.clone()),
                            &array_type_id,
                        )
                        .as_variable()
                        .expect("a materialized typed-items base stays a Variable")
                        .clone();
                    let index = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(index.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    OpKind::ArrayWrite {
                        base: arr,
                        index,
                        value: value.clone(),
                        item_ty,
                        array_type_id: Some(array_type_id.clone()),
                        nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                            array_type_id.as_str(),
                        )),
                    }
                } else if let Some(local) = inner_local
                    && let Some(referent) = self.atomic_ref_place.get(&local).map(clone_place)
                {
                    // `*p = v` where `p` was bound as `&mut place` is that
                    // place's store: `setfield` / `setarrayitem`
                    // (`rewrite_op_setfield`), the same place
                    // `concrete_borrow_place` recovers for `mem::replace`.
                    let place = self.concrete_borrow_place(referent);
                    match place.kind {
                        PlaceKind::Projection(field_inner, elem) => {
                            let same_deref = matches!(
                                (&field_inner.kind, &elem),
                                (PlaceKind::Local(i), ProjectionElem::Atom(name))
                                    if *i as usize == local && name == "Deref"
                            );
                            if !same_deref {
                                let ty = clone_tyref(&place.ty);
                                return self.emit_projection_write(
                                    mir_bb,
                                    *field_inner,
                                    elem,
                                    value,
                                    &ty,
                                );
                            }
                        }
                        PlaceKind::Local(i) => {
                            self.local_var[i as usize] = Some(LocalValue::One(value_var(&value)));
                            return Ok(());
                        }
                        _ => {}
                    }
                    match self.deref_word_store_or_decline(
                        mir_bb,
                        bb_id,
                        inner_local,
                        base,
                        value_var(&value),
                        dest_ty,
                        deref_base_is_raw_ptr,
                    )? {
                        Some(op) => op,
                        None => return Ok(()),
                    }
                } else {
                    // A store this front end cannot name as a field, an
                    // array element, or a raw word must not become a
                    // symbolic `__deref_write`: that path has no code
                    // address. Decline the graph so the caller residualizes
                    // the helper that contains the store.
                    match self.deref_word_store_or_decline(
                        mir_bb,
                        bb_id,
                        inner_local,
                        base,
                        value_var(&value),
                        dest_ty,
                        deref_base_is_raw_ptr,
                    )? {
                        Some(op) => op,
                        None => return Ok(()),
                    }
                }
            }
            ProjectionElem::Tagged(v) => {
                if let Some(field_payload) = v.as_object().and_then(|m| m.get("Field")) {
                    // A multi-word inline field is the aggregate, not a
                    // pointer. Storing the temporary's address into the
                    // first word overwrites the discriminant.
                    if self.struct_field_offset_from_payload(&field_base_ty, field_payload)
                        == Some(0)
                        && let Some(plan) = self.move_plan(dest_ty)
                        && !plan.arms.is_empty()
                        && let LinkArg::Value(src) = &value
                    {
                        self.write_aggregate_into(mir_bb, &base, &plan, src)?;
                        return Ok(());
                    }
                    // A by-value store of an inlined struct is one setfield
                    // per inner leaf (`heaptracker.py all_fielddescrs`
                    // recursion; `jtransform.py rewrite_op_getsubstruct`).
                    // The typeptr leaf stays for `rewrite_op_setfield` /
                    // `is_typeptr_getset` to drop on a vtable object.
                    if let Some(nested) = tyref_by_value_struct_decl(
                        dest_ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ) {
                        let src = value.as_variable().cloned().ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: by-value store of an inlined struct field needs a variable"
                            ))
                        })?;
                        self.emit_inlined_struct_leaf_copy(mir_bb, &base, &src, nested);
                        return Ok(());
                    }
                    // Resolve the field through its TypeDecl exactly
                    // like the read side (`resolve_place` Field arm):
                    // the descriptor must carry the same
                    // (`field_name`, `owner_root`) shape or no
                    // downstream owner-keyed consumer can match it —
                    // jtransform's vable matcher rejects a `None`
                    // owner against an owner-rooted config, so e.g.
                    // `PyFrame.valuestackdepth` writes stayed plain
                    // `setfield` while the paired reads rewrote to
                    // `getfield_vable`.  The Adt case derives the
                    // written value's kind from `dest_ty` (a Ref field
                    // must not be stamped `Int`); the bare-label
                    // fallback keeps non-Adt containers lowering as
                    // before.
                    let (field, ty) = match self.resolve_adt_field(&field_base_ty, field_payload) {
                        Some((owner_root, field_name, _field_ty, owner_id, host_index)) => {
                            // Rust type-checks every assignment against the
                            // projected place's substituted field type.  A
                            // reference or raw pointer to a named ADT is
                            // therefore the same instance slot as a field
                            // initialized by an aggregate: preserve that
                            // class on ordinary `self.field = value` writes
                            // too.  Otherwise one pointer-shaped value
                            // generalizes the session-global ClassDef attr to
                            // the raw-pointer repr before a later getattr is
                            // rtyped.  RPython keeps the declared
                            // SomeInstance class on every assignment (and a
                            // nullable pointer merely sets `can_be_None`).
                            value = self.narrow_typed_ref_field_value(bb_id, value, Some(dest_ty));
                            // `PyError.0 = pin_root(...)`: the declared field
                            // is `PyObjectRef`, the class row is the inner
                            // object (`*mut PyErrorObject`). Reads of
                            // `self.0` already recast through
                            // `cast_off_error_carrier`; the store must land
                            // on that same inner class, or setattr unions
                            // `PyObject ∪ PyErrorObject`.
                            value =
                                self.recast_error_carrier_field_store(bb_id, &field_base_ty, value);
                            let ty =
                                crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                                    &owner_root,
                                    &field_name,
                                    tyref_to_value_type_with(
                                        dest_ty,
                                        self.llbc,
                                        self.tombstoned_leaves,
                                        self.gc_struct_ids,
                                        &self.raw_alloc_flavor_structs,
                                    ),
                                );
                            if let Some(dest) = interior_dest
                                && self.interior_field_alias.contains_key(&dest)
                            {
                                let field = FieldDescriptor::new(
                                    field_name.clone(),
                                    Some(owner_root.clone()),
                                )
                                .with_owner_id(owner_id)
                                .with_owner_declared_gc(container_declared_gc(
                                    &field_base_ty,
                                    self.llbc,
                                    self.gc_struct_ids,
                                    self.immutable_fields,
                                ))
                                .with_base_is_deref(true)
                                .with_host_index(host_index);
                                self.emit_interior_field_write(
                                    mir_bb,
                                    bb_id,
                                    dest,
                                    field,
                                    value.clone(),
                                    ty.clone(),
                                )?;
                                return Ok(());
                            }
                            (
                                FieldDescriptor::new(field_name, Some(owner_root))
                                    .with_owner_id(owner_id)
                                    .with_owner_declared_gc(container_declared_gc(
                                        &field_base_ty,
                                        self.llbc,
                                        self.gc_struct_ids,
                                        self.immutable_fields,
                                    ))
                                    .with_base_is_deref(base_is_deref)
                                    .with_host_index(host_index),
                                ty,
                            )
                        }
                        None => {
                            if let Some((owner, name, ty, word)) =
                                self.opaque_scalar_field(&field_base_ty, dest_ty, field_payload)
                            {
                                (
                                    FieldDescriptor::new(name, Some(owner))
                                        .with_base_is_deref(base_is_deref)
                                        .with_scalar_word(word),
                                    ty,
                                )
                            } else {
                                (
                                    FieldDescriptor::new(
                                        field_label_from_payload(field_payload),
                                        None,
                                    ),
                                    ValueType::Int,
                                )
                            }
                        }
                    };
                    if let Some(owner) = field.owner_root.as_deref() {
                        value = self.coerce_gcref_null_store(bb_id, owner, &field.name, value);
                    }
                    value =
                        self.coerce_signed_int_store(bb_id, &ty, Some(field.name.as_str()), value);
                    OpKind::FieldWrite {
                        base,
                        field,
                        value,
                        ty,
                    }
                } else if let Some(index_payload) = v.as_object().and_then(|m| m.get("Index")) {
                    let idx_var = self.index_offset_var(mir_bb, index_payload)?;
                    // The element place type of `GcRef` peels to the inner
                    // `usize`. A word built from an address cast is
                    // `Ptr(GCREF)`, same as `GcRef::NULL`, so the store's
                    // item kind has to be that pointer.
                    let stored_is_pointer_word = value
                        .as_variable()
                        .is_some_and(|var| self.var_tracks_pointer_word(var));
                    if stored_is_pointer_word {
                        self.pointer_word_arrays.insert(base.clone());
                    }
                    let item_ty = if stored_is_pointer_word {
                        ValueType::Ref(None)
                    } else {
                        tyref_to_value_type_with(
                            dest_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    };
                    OpKind::ArrayWrite {
                        base,
                        index: idx_var,
                        value: value.clone(),
                        item_ty,
                        array_type_id: projection_array_type_id,
                        nolength: projection_array_nolength,
                    }
                } else {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: ProjectionElem::Tagged write not handled: {v}"
                    )));
                }
            }
            ProjectionElem::Atom(s) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: ProjectionElem::Atom({s}) write not handled"
                )));
            }
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: op,
        });
        Ok(())
    }

    /// `*p = v` when `p` is not a field or array reborrow.
    ///
    /// A multi-word pointee is the field-wise move `exchange_deref_aggregate`
    /// already emits for `mem::replace` (`rffi.py` `_get_structcopy_fn`).
    /// A one-word raw pointer is `raw_store` at offset 0
    /// (`rewrite_op_raw_store`). A `TypedItemsBlock` word is `setarrayitem`
    /// and is emitted before this helper (`emit_projection_write`).
    /// Anything else declines the graph: a symbolic `__deref_write` has no
    /// code address.
    ///
    /// `Ok(None)` means the field writes were already pushed.
    pub(in crate::front::mir) fn deref_word_store_or_decline(
        &mut self,
        mir_bb: usize,
        bb_id: BlockId,
        inner_local: Option<usize>,
        base: Variable,
        value: Variable,
        dest_ty: &TyRef,
        deref_base_is_raw_ptr: bool,
    ) -> Result<Option<OpKind>, LowerError> {
        // A GC reference behind `&mut` lives in a one-field GC cell for
        // the duration of the borrow. Copy-in/copy-out is exact because
        // the borrow is exclusive and does not outlive the call. Raw
        // memory cannot hold a GC reference, so `*p = v` writes the
        // cell's `value` field.
        if let Some(local) = inner_local
            && let Some(root) = self.gc_mut_ref_param_root(local)
        {
            return Ok(Some(OpKind::FieldWrite {
                base,
                field: FieldDescriptor::new("value", Some(root)),
                value: LinkArg::Value(value),
                ty: ValueType::Ref(None),
            }));
        }
        if let Some(local) = inner_local {
            let place = Place {
                kind: PlaceKind::Projection(
                    Box::new(Place {
                        kind: PlaceKind::Local(local as u64),
                        ty: unit_tyref(),
                    }),
                    ProjectionElem::Atom("Deref".to_string()),
                ),
                ty: clone_tyref(dest_ty),
            };
            if self.move_plan(&place.ty).is_some()
                && self.store_moved_aggregate(mir_bb, &place, &value, Some(base.clone()))?
            {
                return Ok(None);
            }
        }
        if deref_base_is_raw_ptr
            && let Some((item_ty, itemsize, is_item_signed)) = self.raw_word_descr(dest_ty)
        {
            let offset = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(offset.clone()),
                kind: OpKind::ConstInt(0),
            });
            return Ok(Some(OpKind::RawStore {
                base,
                offset,
                value,
                item_ty,
                itemsize,
                is_item_signed,
            }));
        }
        let _ = (bb_id, base);
        Err(LowerError::Unsupported(format!(
            "bb{mir_bb}: deref store is not a field, array element, or raw word"
        )))
    }

    pub(in crate::front::mir) fn local_decl_is_primitive(&self, local: usize) -> bool {
        self.body
            .locals
            .locals
            .get(local)
            .is_some_and(|decl| tyref_is_primitive_scalar(&decl.ty, self.llbc))
    }

    /// rustc `&mut x` → `*mut T` is `_r = &mut x; _p = &raw mut (*_r)`.
    /// `_p` is a raw pointer whose value is still the aliased word.
    pub(in crate::front::mir) fn raw_scalar_arg_aliases_primitive_word(
        &self,
        arg_local: Option<usize>,
    ) -> bool {
        self.resolved_borrow_place(arg_local)
            .is_some_and(|place| self.place_holds_aliased_primitive_word(&place))
    }

    /// A place that *holds* the scalar word: a primitive local, or a
    /// field of a by-value aggregate. A remaining `Deref` is an address
    /// that already names the word (`&raw mut (*p)` of a real pointer).
    pub(in crate::front::mir) fn place_holds_aliased_primitive_word(&self, place: &Place) -> bool {
        if !tyref_is_primitive_scalar(&place.ty, self.llbc) {
            return false;
        }
        let mut current = place;
        for _ in 0..8 {
            match &current.kind {
                PlaceKind::Local(_) => return true,
                PlaceKind::Projection(inner, ProjectionElem::Tagged(v))
                    if v.get("Field").is_some() =>
                {
                    current = inner;
                }
                _ => return false,
            }
        }
        false
    }

    /// Preserve the declared RPython reference repr of a typed field value.
    ///
    /// RPython represents an instance attribute as `SomeInstance(classdef)`;
    /// when nullable, the same annotation simply carries `can_be_None=True`.
    /// Rust spells those slots as `&NamedAdt` or `*mut/*const NamedAdt`, whose
    /// producer initially has pointer annotation.  A pointer to an RPython
    /// array is the same boundary: `rutf8.UTF8_INDEX_STORAGE` is a `GcArray`,
    /// while Rust carries the exact owner as `*mut Vec<Utf8LocElem>`.  The
    /// `_setup_repr_llfields` types the virtualizable token as GCREF.
    /// A host `0` is `ConstPtr(NULL)`, not an int store into that field.
    pub(in crate::front::mir) fn coerce_gcref_null_store(
        &mut self,
        bb_id: BlockId,
        owner: &str,
        field_name: &str,
        value: LinkArg,
    ) -> LinkArg {
        use crate::translator::rtyper::lltypesystem::lltype;
        let is_gcref = crate::virtualizable_decl::virtualizable_llfield_type(owner, field_name)
            .is_some_and(|ty| ty == *lltype::GCREF);
        if !is_gcref || !crate::model::link_arg_is_int_zero(&self.graph, &value) {
            return value;
        }
        match self.graph.push_op_var(bb_id, OpKind::ConstRefNull, true) {
            Some(null) => LinkArg::Value(null),
            None => value,
        }
    }

    /// Unsigned 0 stored into a signed field is `intmask` (`rarithmetic.py`).
    /// Charon records a generic integer 0 as `ConstUInt`; `W_ListObject.allocated`
    /// is `isize` (`rlist.py` `ll_newlist`). Fused boxing FieldWrite can carry
    /// `Ref(None)` for that store; the field name still intmasks.
    pub(in crate::front::mir) fn coerce_signed_int_store(
        &mut self,
        bb_id: BlockId,
        dest_kind: &ValueType,
        field_name: Option<&str>,
        value: LinkArg,
    ) -> LinkArg {
        let signed_dest = matches!(dest_kind, ValueType::Int) || field_name == Some("allocated");
        if !signed_dest {
            return value;
        }
        let n = match &value {
            LinkArg::Const(c) => {
                use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
                // Charon may leave `concretetype` unset until a later stamp.
                // `intmask` is the signed word either way (`ll_newlist`).
                let already_signed = matches!(
                    c.concretetype,
                    Some(LowLevelType::Signed)
                        | Some(LowLevelType::SignedLongLong)
                        | Some(LowLevelType::SignedLongLongLong)
                );
                match &c.value {
                    crate::flowspace::model::ConstValue::Int(_) if already_signed => return value,
                    crate::flowspace::model::ConstValue::Int(n) => *n,
                    crate::flowspace::model::ConstValue::UInt(n) => *n as i64,
                    crate::flowspace::model::ConstValue::Int128(v)
                        if *v >= i64::MIN as i128 && *v <= i64::MAX as i128 =>
                    {
                        *v as i64
                    }
                    _ => return value,
                }
            }
            LinkArg::Value(var) => {
                let Some(n) = self
                    .graph
                    .blocks
                    .iter()
                    .flat_map(|block| &block.operations)
                    .find_map(|op| {
                        (op.result.as_ref() == Some(var))
                            .then_some(match &op.kind {
                                OpKind::ConstUInt(n) if *n <= i64::MAX as u64 => Some(*n as i64),
                                OpKind::ConstUInt128(n) if *n <= i64::MAX as u128 => {
                                    Some(*n as i64)
                                }
                                _ => None,
                            })
                            .flatten()
                    })
                else {
                    return value;
                };
                n
            }
        };
        match self.graph.push_op_var(bb_id, OpKind::ConstInt(n), true) {
            Some(signed) => LinkArg::Value(signed),
            None => value,
        }
    }

    /// Declared field bank of `dest_ty.name`, after wrapper peel.
    pub(in crate::front::mir) fn named_field_value_type(
        &self,
        dest_ty: &TyRef,
        field_name: &str,
    ) -> Option<ValueType> {
        let node =
            tyref_node(dest_ty, self.llbc).and_then(|node| strip_ty_wrappers(node, self.llbc))?;
        let id = adt_node_def_id(node)?;
        let td = self.llbc.type_by_id(id)?;
        let fields = type_decl_struct_fields(td, self.llbc)?;
        let field = fields.iter().enumerate().find_map(|(i, f)| {
            let name = f
                .name
                .clone()
                .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(i));
            (name == field_name).then_some(f)
        })?;
        Some(tyref_to_value_type_with(
            &field.ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ))
    }

    /// Rust field type-check is the proof needed to narrow that producer
    /// before `setattr`; this is an identity cast for non-null values and a
    /// typed null for null values.
    pub(in crate::front::mir) fn narrow_typed_ref_field_value(
        &mut self,
        bb_id: BlockId,
        value: LinkArg,
        declared_ty: Option<&TyRef>,
    ) -> LinkArg {
        let Some(root) = self.typed_ref_field_narrow_root(declared_ty) else {
            return value;
        };
        self.narrow_value_to_instance_root(bb_id, value, &root)
    }

    /// The instance root a value stored into a field declared `declared_ty`
    /// is narrowed to, or `None` when the declaration names no class.
    pub(in crate::front::mir) fn typed_ref_field_narrow_root(
        &self,
        declared_ty: Option<&TyRef>,
    ) -> Option<String> {
        let ty = declared_ty?;
        // A fieldless enum is its discriminant integer (`tyref_to_value_type`
        // / `type_decl_is_fieldless_enum`). A store of that word — including
        // through a borrow, which is the identity on the tag — is already
        // the int; `cast_pointer` has no integer operand
        // (`rclass.py` `pairtype(InstanceRepr, InstanceRepr).convert_from_to`).
        // Same decision as [`Self::instance_narrow_for_type`].
        if tyref_fieldless_enum_pointee(ty, self.llbc).is_some() {
            return None;
        }
        // A `repr(transparent)` scalar wrapper's borrow is the word
        // (`tyref_transparent_inner_value_type` peels `&W` the same way
        // [`tyref_to_value_type_with`] does). Narrowing to the wrapper
        // class would mint `Ptr(W)` next to the Integer the by-value
        // site already stores, and `setattr("__pos_0")` cannot union them.
        // A pointer newtype whose inner is already a Ref stays a class
        // instance: skip only the integer/float/bool word.
        if let Some(inner) = tyref_transparent_inner_value_type(
            ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ) && value_type_is_transparent_scalar_word(&inner)
        {
            return None;
        }
        let node = strip_ty_indirections(tyref_node(ty, self.llbc)?, self.llbc)?;
        if let Some(reference) = node.as_object()?.get("Ref") {
            let pointee = reference.as_array()?.get(1)?;
            return adt_node_class_root_with(
                strip_ty_indirections(pointee, self.llbc)?,
                self.llbc,
                self.tombstoned_leaves,
            );
        }
        raw_ptr_pointee_class_root_with(node, self.llbc, self.tombstoned_leaves)
            .or_else(|| raw_ptr_pointee_container_root(node, self.llbc))
    }

    /// Re-express a GC-reference value at a statically-known RPython
    /// instance boundary.  Rust raw pointers erase the pointee in
    /// [`ValueType::Ref(None)`], while the corresponding RPython value keeps
    /// its `SomeInstance(classdef)` (for example `PyObjectRef` is W_Root).
    /// The marker is a same-bank identity and rtypes to `cast_opaque_ptr`.
    pub(in crate::front::mir) fn narrow_value_to_instance_root(
        &mut self,
        bb_id: BlockId,
        value: LinkArg,
        root: &str,
    ) -> LinkArg {
        let LinkArg::Value(value) = value else {
            // Ref-kind constants are materialized calls in this frontend;
            // only scalar constants can remain inline in a FieldWrite.
            return value;
        };
        let narrowed = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(narrowed.clone()),
            kind: crate::model::cast_instance_call(root, value),
        });
        LinkArg::Value(narrowed)
    }

    /// Extract the `offset` operand from an `Index { offset, from_end }`
    /// projection element and resolve it to a Variable. `from_end` is
    /// ignored: backwards-from-end indexing only appears in slice patterns
    /// at the moment, and the lowering uses the offset Variable directly.
    pub(in crate::front::mir) fn index_offset_var(
        &mut self,
        mir_bb: usize,
        index_payload: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        let offset = index_payload
            .as_object()
            .and_then(|m| m.get("offset"))
            .ok_or_else(|| {
                LowerError::Schema(format!(
                    "bb{mir_bb}: Index projection missing offset: {index_payload}"
                ))
            })?
            .clone();
        let op: Operand = serde_json::from_value(offset)
            .map_err(|e| LowerError::Schema(format!("bb{mir_bb}: Index offset decode: {e}")))?;
        self.resolve_operand(mir_bb, op)
    }

    /// Store into an error-carrier field (`self.0` on `PyError`) as the
    /// inner object class, matching [`Self::cast_off_error_carrier`].
    pub(in crate::front::mir) fn recast_error_carrier_field_store(
        &mut self,
        bb_id: BlockId,
        field_base_ty: &TyRef,
        value: LinkArg,
    ) -> LinkArg {
        if !self.ty_is_error_carrier(field_base_ty) {
            return value;
        }
        let Some(inner) = self.error_carrier_inner_class_root(field_base_ty) else {
            return value;
        };
        self.narrow_value_to_instance_root(bb_id, value, &inner)
    }

    /// Whether `ty` is the configured exception carrier (`PyError`).
    pub(in crate::front::mir) fn ty_is_error_carrier(&self, ty: &TyRef) -> bool {
        let spec = self.static_addrs.error_carrier.carrier_path;
        if spec.is_empty() {
            return false;
        }
        let Some(path) = adt_path_of_tyref(ty, self.llbc) else {
            return false;
        };
        path == spec || strip_crate_prefix(&path) == strip_crate_prefix(spec)
    }

    pub(in crate::front::mir) fn error_carrier_class_root(&self, ty: &TyRef) -> Option<String> {
        if !self.ty_is_error_carrier(ty) {
            return None;
        }
        tyref_class_root_with(ty, self.llbc, self.tombstoned_leaves)
    }

    /// Class root of the carrier's payload: `Deref::Target` (`PyErrorObject`).
    /// The declared field is `PyObjectRef` (`*mut PyObject`); projecting that
    /// class has no common annotator base with the handle, which subclasses
    /// the object class (`rclass.py` MRO through `OperationError`).
    pub(in crate::front::mir) fn error_carrier_inner_class_root(
        &self,
        ty: &TyRef,
    ) -> Option<String> {
        if !self.ty_is_error_carrier(ty) {
            return None;
        }
        transparent_deref_target_class(
            ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
    }

    /// `PyError(ptr)` is one pointer word. The class is the handle's
    /// own leaf (`error::PyError`), a subclass of `Deref::Target`.
    pub(in crate::front::mir) fn retag_error_carrier(
        &mut self,
        wrapper_ty: &TyRef,
        value: Variable,
    ) -> (Option<OpKind>, Variable) {
        let Some(root) = self.error_carrier_class_root(wrapper_ty) else {
            return (None, value);
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        (Some(crate::model::cast_instance_call(root, value)), res)
    }

    /// `self.0` on the carrier: the same word, typed as the inner object.
    pub(in crate::front::mir) fn cast_off_error_carrier(
        &mut self,
        mir_bb: usize,
        wrapper_ty: &TyRef,
        base: Variable,
    ) -> Variable {
        let Some(root) = self.error_carrier_inner_class_root(wrapper_ty) else {
            return base;
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let bb_id = self.emit_bb(mir_bb);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: crate::model::cast_instance_call(root, base),
        });
        res
    }

    /// Build the IR for an Rvalue. Returns `(op, result_var)` — `op` is
    /// the `OpKind` to push onto the current block, `result_var` is the
    /// Variable the destination local should be bound to. `op` is
    /// `None` for trivial copies (no op pushed, the existing Variable
    /// is reused).
    pub(in crate::front::mir) fn build_rvalue(
        &mut self,
        mir_bb: usize,
        rvalue: Rvalue,
        dest_ty: &TyRef,
    ) -> Result<(Option<OpKind>, Variable), LowerError> {
        match rvalue {
            Rvalue::Use(operand, _) => {
                let v = self.resolve_operand(mir_bb, operand)?;
                // Plain use — reuse the operand's Variable without
                // emitting a copy op. RPython does the same: a flow
                // graph never has a redundant `same_as` between two
                // Variables that already alias.
                Ok((None, v))
            }
            Rvalue::BinaryOp(op_json, lhs, rhs) => {
                // Peek kinds before `resolve_operand` consumes the
                // operands. Ordered pointer compares must emit
                // `lltype.cast_ptr_to_int` first (`rbuiltin.py
                // rtype_cast_ptr_to_int`); `rptr.py` has only eq/ne.
                let lhs_kind = self.operand_value_kind(&lhs);
                let rhs_kind = self.operand_value_kind(&rhs);
                let lhs_raw_addr = operand_is_declared_raw(
                    &lhs,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                let rhs_raw_addr = operand_is_declared_raw(
                    &rhs,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                // Rust `==` / `!=` on raw pointers compares addresses: the
                // flowspace `is_` (`rptr.py` `rtype_is_` → `ptr_eq`), not the
                // value `eq` a string operand would dispatch to `ll_streq`.
                let raw_ref_pointer = |op: &Operand, kind: &Option<ValueType>| {
                    matches!(op, Operand::Copy(place) | Operand::Move(place)
                        if tyref_is_raw_pointer(&place.ty, self.llbc))
                        && kind.as_ref().is_some_and(|k| value_type_bank(k) == 1)
                };
                let pointer_identity =
                    raw_ref_pointer(&lhs, &lhs_kind) && raw_ref_pointer(&rhs, &rhs_kind);
                let lhs_v = self.resolve_operand(mir_bb, lhs)?;
                let rhs_v = self.resolve_operand(mir_bb, rhs)?;
                let mut op_label = binop_label(&op_json)?;
                // Preserve the source scalar width through flow-graph
                // construction.  PyPy's rbigint `_x_divrem` performs its
                // intermediates in `LONG_TYPE` / `ULONG_TYPE`
                // (`SignedLongLongLong` / `UnsignedLongLongLong`), so
                // collapsing an i128/u128 destination to the word-sized
                // `Int` carrier here loses the exact RPython low-level type
                // before the rtyper can see it. f64 similarly stays Float.
                // Charon represents a checked arithmetic result as
                // `(numeric, bool)`, but this lowering deliberately binds the
                // destination local to the scalar `.0` value (see
                // `binop_result_locals`). Project that numeric element's Rust
                // type before choosing the annotation shell: in particular,
                // a `usize` subtraction is an unsigned/non-negative value by
                // type, even though its MIR destination is a tuple.
                let scalar_ty =
                    tyref_checked_binop_value_type(dest_ty, self.llbc).unwrap_or_else(|| {
                        tyref_to_value_type_with(
                            dest_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    });
                let result_ty = match scalar_ty {
                    ValueType::Float => ValueType::Float,
                    ValueType::Unsigned => ValueType::Unsigned,
                    ValueType::Int128 => ValueType::Int128,
                    ValueType::UInt128 => ValueType::UInt128,
                    _ => ValueType::Int,
                };
                // Rust `/` (MIR `Div`) lowers to the `floordiv` opname
                // (`canonical_binop_label`) — the correct label for
                // integer truncating division. Over floats `/` is true
                // division; `pairtype(SomeFloat, SomeFloat)` registers
                // `truediv` but not `floordiv` (`binaryop.py`),
                // the same distinction RPython's flowspace makes by
                // emitting the `truediv` opname for `/` at graph
                // construction. Re-label the float result so the prepass
                // annotator dispatches `truediv(SomeFloat, SomeFloat)`
                // instead of walling on the unregistered `floordiv`.
                // Downstream is unchanged: `codewriter::jtransform` funnels
                // both labels to `float_truediv` and the rtyper maps
                // `(FloatRepr, "truediv")` / `(FloatRepr, "div")` to the
                // same `rtype_template("truediv")`.
                if result_ty == ValueType::Float && op_label == "floordiv" {
                    op_label = "truediv".to_string();
                }
                // `getkind` reads the result variable, not `result_ty`.
                // `Unknown` is kind `'r'`, so an `i64` add was emitted as
                // `int_add/ii>r`, a key no blackhole handler has. A scalar
                // result is Signed / Float (`history.py getkind`). A
                // non-scalar destination keeps `Unknown`.
                let result_concrete = match &scalar_ty {
                    ValueType::Float => crate::model::ConcreteType::Float,
                    ValueType::Int | ValueType::Unsigned | ValueType::Bool => {
                        crate::model::ConcreteType::Signed
                    }
                    _ => crate::model::ConcreteType::Unknown,
                };
                let res = self.graph.alloc_value_var_with_type(result_concrete);
                // Integer ops that have no pointer form (`rptr.py` is
                // only eq/ne) must see Signed addresses:
                // `cast_ptr_to_int(p) < cast_ptr_to_int(q)`, and the
                // same for `%` / `/`. Equality stays a pointer compare
                // (`ptr_eq` / `instance_ptr_eq` in jtransform).
                let (lhs_v, rhs_v) =
                    if int_binop_needs_ptr_to_int(&op_label, lhs_kind.as_ref(), rhs_kind.as_ref())
                        || (matches!(
                            op_label.as_str(),
                            "lt" | "le" | "gt" | "ge" | "mod" | "floordiv" | "div"
                        ) && (lhs_raw_addr || rhs_raw_addr))
                    {
                        let bb_id = self.emit_bb(mir_bb);
                        let lhs_v = if matches!(lhs_kind, Some(ValueType::Ref(_))) || lhs_raw_addr {
                            let orig = lhs_v.clone();
                            let v = push_cast_ptr_to_int(&mut self.graph, bb_id, lhs_v);
                            self.cast_ptr_to_int_src.insert(v.clone(), orig);
                            v
                        } else {
                            lhs_v
                        };
                        let rhs_v = if matches!(rhs_kind, Some(ValueType::Ref(_))) || rhs_raw_addr {
                            let orig = rhs_v.clone();
                            let v = push_cast_ptr_to_int(&mut self.graph, bb_id, rhs_v);
                            self.cast_ptr_to_int_src.insert(v.clone(), orig);
                            v
                        } else {
                            rhs_v
                        };
                        // Raw-pointer ordering is unsigned. `cast_ptr_to_int`
                        // is signed, so retype both words and use `uint_*`.
                        if matches!(op_label.as_str(), "lt" | "le" | "gt" | "ge")
                            && (lhs_raw_addr || rhs_raw_addr)
                        {
                            let lhs_v = push_r_uint(&mut self.graph, bb_id, lhs_v);
                            let rhs_v = push_r_uint(&mut self.graph, bb_id, rhs_v);
                            op_label = match op_label.as_str() {
                                "lt" => "uint_lt",
                                "le" => "uint_le",
                                "gt" => "uint_gt",
                                "ge" => "uint_ge",
                                other => other,
                            }
                            .to_string();
                            (lhs_v, rhs_v)
                        } else {
                            (lhs_v, rhs_v)
                        }
                    } else {
                        (lhs_v, rhs_v)
                    };
                if pointer_identity && matches!(op_label.as_str(), "eq" | "ne") {
                    // `a is not b` has no operation of its own; it is
                    // `is_(a, b) == False`, the shape `Option::is_some`
                    // lowers to.
                    if op_label == "ne" {
                        let bb_id = self.emit_bb(mir_bb);
                        let is_res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        let false_var = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(is_res.clone()),
                            kind: OpKind::BinOp {
                                op: "is_".to_string(),
                                lhs: lhs_v,
                                rhs: rhs_v,
                                result_ty: ValueType::Int,
                            },
                        });
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(false_var.clone()),
                            kind: OpKind::ConstBool(false),
                        });
                        return Ok((
                            Some(OpKind::BinOp {
                                op: "eq".to_string(),
                                lhs: is_res,
                                rhs: false_var,
                                result_ty: ValueType::Int,
                            }),
                            res,
                        ));
                    }
                    op_label = "is_".to_string();
                }
                Ok((
                    Some(OpKind::BinOp {
                        op: op_label,
                        lhs: lhs_v,
                        rhs: rhs_v,
                        result_ty,
                    }),
                    res,
                ))
            }
            // `UnaryOp(op, operand)` — `Neg`, `Not`, casts.  Arithmetic
            // `Neg` / `Not` lower to `OpKind::UnaryOp` with a canonical
            // snake_case label so the assembler reaches the wired
            // `int_neg` / `int_invert` handlers instead of inventing a
            // synthetic `int_unary.*` opname.  `Cast(...)` is handled
            // separately below: same-bank casts alias the operand, and a
            // bank-crossing cast lowers to a `simple_call` against the
            // matching host cast callable (see the in-arm comment).
            Rvalue::UnaryOp(op_json, operand) => {
                // A `Cast(..)` reinterprets the operand. When the operand
                // and the destination share a register bank (ptr→ptr,
                // int→int of any width, float→float, `Unsize`, `FnPtr` —
                // every cast that keeps the i64/f64 carrier in place) the
                // JIT models it as `same_as`, so alias the operand without
                // emitting an op.  A bank-CHANGING cast (`int↔ptr`,
                // `int↔float`) must move the value into the destination
                // bank.  The rtyper retired every typed cast opname from
                // the unary-op path (`normalize_unary_op_name` accepts only
                // `neg` / `bool` / `invert` / `same_as`), so a bank
                // crossing lowers to `simple_call(<host_callable>, v)` —
                // `lltype.cast_int_to_ptr` / `lltype.cast_ptr_to_int` for
                // `int↔ptr`, the bare `float` / `int` builtins for
                // `int↔float` — whose rtyper hooks emit the low-level
                // `cast_*` op.  Pure-aliasing those would leave e.g. an
                // `as_usize() as *mut T` value in the Int bank where a
                // later GcRef merge expects a Ref, tripping the assembler's
                // per-bank cross-check.  The bank decision reads the
                // operand's place type and the destination type directly,
                // so it is independent of which Charon `CastKind` tag
                // encodes the conversion.  Genuine `Neg` / `Not` arithmetic
                // keeps a real scalar `OpKind::UnaryOp`.
                if unary_op_is_cast(&op_json) {
                    // A constant `usize as i64` (`#receiver_slots as i64`)
                    // has no place type, so the unsigned→signed arm below
                    // never sees it and the alias keeps `ConstUInt`.
                    // `gt` then mixes that unsigned constant with
                    // `rarithmetic.intmask`'s signed result
                    // (`rtype_compare_template`). Emit the wrapped signed
                    // word before the unsigned constant is materialized.
                    // Narrow integer destinations still need a mask.
                    if let Operand::Const(value) = &operand
                        && let Some(atom) = self.tyref_literal_int_atom(dest_ty)
                        && matches!(atom, "I64" | "Isize")
                        && let Ok(DecodedConst::UInt(bits)) = decode_constant(self.llbc, value)
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                        return Ok((Some(OpKind::ConstInt(bits as i64)), res));
                    }
                    // Signedness-aware source classification.  Both the
                    // value projection and the attribute projection preserve
                    // `uN` as Unsigned; the latter remains the source of
                    // truth for wrapped/field place types.
                    let arg = self.resolve_operand(mir_bb, operand.clone())?;
                    if cast_kind_is_unsize(&op_json)
                        && let Some((op, res)) = self.object_array_unsize_newlist(dest_ty, &arg)
                    {
                        return Ok((Some(op), res));
                    }
                    // A transparent address-cast word is a GCREF pointer even
                    // though its place type is the inner `usize`. Reading it
                    // back (`roots[i].0 as *mut _`) must not emit
                    // `cast_int_to_ptr`: that paints an integer, and
                    // `direct_ptradd` then rejects the digit base.
                    let pointer_word = self.pointer_word_vars.contains(&arg);
                    let src_attr = if pointer_word {
                        Some(ValueType::Ref(None))
                    } else {
                        match &operand {
                            Operand::Copy(p) | Operand::Move(p) => {
                                Some(tyref_to_attr_value_type_with(
                                    &p.ty,
                                    self.llbc,
                                    self.tombstoned_leaves,
                                    self.gc_struct_ids,
                                    &self.raw_alloc_flavor_structs,
                                ))
                            }
                            Operand::Const(_) => None,
                        }
                    };
                    let src_kind = if pointer_word {
                        Some(ValueType::Ref(None))
                    } else {
                        self.operand_value_kind(&operand)
                    };
                    let src_root = if pointer_word {
                        None
                    } else {
                        self.operand_class_root(&operand)
                    };
                    let src_is_prebuilt_once_value = match &operand {
                        Operand::Copy(p) | Operand::Move(p) => self
                            .prebuilt_once_value_locals
                            .iter()
                            .any(|&local| place_references_local(p, local)),
                        Operand::Const(_) => false,
                    };
                    let src_int = match &operand {
                        Operand::Copy(p) | Operand::Move(p) => {
                            int_cast_size_and_sign(&p.ty, self.llbc)
                        }
                        Operand::Const(_) => None,
                    };
                    let dst_kind = tyref_to_value_type_with(
                        dest_ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    // An integer cast to a narrower unsigned type keeps only
                    // the low bytes. The JIT carries every integer in one
                    // machine word, so the truncation has to be an explicit
                    // `int_and` (`jtransform.py _int_to_int_cast`); aliasing
                    // the operand would leave the high bits in place.
                    if let (Some(src), Some(dst)) =
                        (src_int, int_cast_size_and_sign(dest_ty, self.llbc))
                        && let Some(IntToIntCast::And(mask)) =
                            int_to_int_cast(src, dst, crate::layout::target_word_size() as u64)
                    {
                        let bb_id = self.emit_bb(mir_bb);
                        // `and_(r_uint, r_uint)` keeps the Unsigned
                        // annotation the destination has; a signed operand
                        // is retyped first through the same `r_uint` marker
                        // the same-width signedness flip below uses.
                        let lhs = if src.1 {
                            arg
                        } else {
                            let unsigned = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(unsigned.clone()),
                                kind: OpKind::Call {
                                    target: CallTarget::FunctionPath {
                                        segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                                            .into_iter()
                                            .map(str::to_string)
                                            .collect(),
                                        fun_decl_id: None,
                                    },
                                    args: crate::model::call_args(vec![arg]),
                                    result_ty: ValueType::Unsigned,
                                },
                            });
                            unsigned
                        };
                        let mask_var = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(mask_var.clone()),
                            kind: OpKind::ConstUInt(mask),
                        });
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((
                            Some(OpKind::BinOp {
                                op: "and".to_string(),
                                lhs,
                                rhs: mask_var,
                                result_ty: ValueType::Unsigned,
                            }),
                            res,
                        ));
                    }
                    // The Rust-only current-address adapter is erased as a
                    // whole GCREF identity.  Its `ptr -> usize -> ptr`
                    // round-trip exists only to call the host GC query; once
                    // that query is replaced by RPython's root-reload model,
                    // retaining either bank crossing would manufacture an
                    // address integer absent from the upstream graph.
                    if self.is_gc_current_object_address_adapter_graph() {
                        return Ok((None, arg));
                    }
                    // A prebuilt singleton kept in a `OnceLock<usize>` reads
                    // back as the registered object itself
                    // ([`Lowering::prebuilt_once_value_locals`]); the word it
                    // is stored as is already that GC reference.
                    if src_is_prebuilt_once_value && matches!(dst_kind, ValueType::Ref(_)) {
                        return Ok((None, arg));
                    }
                    let src_width = match &operand {
                        Operand::Copy(place) | Operand::Move(place) => {
                            self.literal_int_width(&place.ty)
                        }
                        Operand::Const(_) => None,
                    };
                    if let (Some(src_w), Some(dst_w)) = (src_width, self.literal_int_width(dest_ty))
                        && let Some(lowered) =
                            self.materialize_int_cast(mir_bb, src_w, dst_w, arg.clone())
                    {
                        return Ok(lowered);
                    }
                    // Signedness-flipping int cast (`w_tuple_len(obj) as i64`)
                    // — aliasing keeps the source `r_uint` annotation on the
                    // signed destination, tripping the SomeInteger signedness
                    // `UnionError` (`binaryop.py union`) when the length
                    // meets a signed index.  Route the unsigned→signed flip
                    // through `rarithmetic.intmask` (the RPython spelling of
                    // this re-type): `rtype_intmask` coerces to `lltype.Signed`
                    // — identity on the i64 carrier — so the value is unchanged
                    // and the result re-types Signed.  Resolves via the Layer-3
                    // `HOST_ENV.import_module(rpython.rlib.rarithmetic)
                    // .module_get(intmask)` path (`flowspace_adapter.rs`).
                    // Rust `as char` is `builtin_unichr` → `SomeUnicodeCodePoint`
                    // (`IntegerRepr.rtype_unichr`); `builtin_chr` is the
                    // byte-char (`SomeChar`) path.  Reserved `__majit_unichr`
                    // like `__majit_range`: a bare `["unichr"]` binds the
                    // leaf-keyed `ctypeprim::unichr` host.
                    if tyref_is_char(dest_ty, self.llbc) {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((Some(unichr_call(arg)), res));
                    }
                    if matches!(src_attr, Some(ValueType::Unsigned))
                        && matches!(
                            tyref_to_attr_value_type_with(
                                dest_ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs
                            ),
                            ValueType::Int
                        )
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((
                            Some(OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: ["rpython", "rlib", "rarithmetic", "intmask"]
                                        .into_iter()
                                        .map(str::to_string)
                                        .collect(),
                                    fun_decl_id: None,
                                },
                                args: crate::model::call_args(vec![arg]),
                                result_ty: ValueType::Int,
                            }),
                            res,
                        ));
                    }
                    // An Int-banked pointer — `history.getkind` of
                    // `Ptr(FuncType)`, or of `Ptr(Struct(raw) "RustVec")` —
                    // would otherwise take the signedness-flip `r_uint` arm
                    // below as a bare integer retype.  The source is still a
                    // pointer: `rbuiltin.py` `gen_cast` of a Ptr origin to
                    // an unsigned primitive is `cast_ptr_to_int` then
                    // `gen_cast(..., TGT=Unsigned)`, and a signed dest is
                    // just `cast_ptr_to_int` (`rtype_cast_ptr_to_int`
                    // resulttype=Signed).  Dest matching goes through
                    // `int_cast_size_and_sign` so an `as fn` dest
                    // (Int-banked, not a Scalar Integer) is not claimed.
                    // `cast_ptr_to_int_src` is left out: that table folds
                    // `p as usize as *mut T` back to a GC pointer
                    // (`jtransform.py rewrite_op_cast_opaque_ptr`), and
                    // neither a function pointer nor a RustVec header is a
                    // GCREF — recording one would put an Int-banked value
                    // into a Ref dest.
                    let src_is_ptr_to_int_origin = operand_tyref(&operand)
                        .is_some_and(|ty| tyref_is_ptr_to_int_cast_origin(ty, self.llbc));
                    if src_is_ptr_to_int_origin {
                        let dest_int = int_cast_size_and_sign(dest_ty, self.llbc);
                        let bb_id = self.block_id[mir_bb];
                        if dest_int.is_some_and(|(_, unsigned)| unsigned) {
                            let (retype, result) =
                                push_ptr_to_unsigned_cast(&mut self.graph, bb_id, arg);
                            return Ok((Some(retype), result));
                        }
                        if dest_int.is_some() {
                            let signed = push_cast_ptr_to_int(&mut self.graph, bb_id, arg);
                            return Ok((None, signed));
                        }
                    }
                    // The opposite signedness flip is RPython's
                    // `rarithmetic.r_uint(v)`: it keeps the machine word
                    // bits but changes the annotation/repr to Unsigned.
                    if matches!(src_attr, Some(ValueType::Int))
                        && matches!(
                            tyref_to_attr_value_type_with(
                                dest_ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs
                            ),
                            ValueType::Unsigned
                        )
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        // `history.getkind` already banks `Ptr` of a Raw struct
                        // as `int`, so `p as usize` is this `r_uint` and not
                        // `cast_ptr_to_int`. `GcRef(p as usize)` still retypes
                        // that address to `Ptr(GCREF)`; record the source the
                        // same way the Ref arm does.
                        let source_is_raw_address = match &operand {
                            Operand::Copy(place) | Operand::Move(place) => ty_is_raw_address(
                                &place.ty,
                                self.llbc,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            ),
                            Operand::Const(_) => false,
                        };
                        if source_is_raw_address {
                            // `r_uint` takes `Unsigned`. A raw pointer is
                            // `SomePtr`, so sign the address first.
                            if let Some((op, res)) =
                                self.raw_ptr_to_unsigned(mir_bb, true, dest_ty, &arg)
                            {
                                return Ok((Some(op), res));
                            }
                        }
                        return Ok((
                            Some(OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                                        .into_iter()
                                        .map(str::to_string)
                                        .collect(),
                                    fun_decl_id: None,
                                },
                                args: crate::model::call_args(vec![arg]),
                                result_ty: ValueType::Unsigned,
                            }),
                            res,
                        ));
                    }
                    // Pointer -> unsigned-word cast is two operations in
                    // RPython, not a `cast_ptr_to_int` whose result is merely
                    // labelled Unsigned.  `rbuiltin.py`'s `gen_cast` (which
                    // `rtype_cast_primitive` delegates to) first emits
                    // `cast_ptr_to_int(..., resulttype=Signed)` and then
                    // `gen_cast(..., TGT=Unsigned)`.  Rust's `p as usize`
                    // reaches this arm directly, so materialise that same
                    // sequence: the inner call produces the orthodox Signed
                    // address integer and `r_uint` performs the second,
                    // same-word retype.  Omitting it leaves `p as usize`
                    // annotated `SomeInteger()` and makes it fail to join a
                    // genuine `0usize` (`SomeInteger(unsigned=True)`).
                    if matches!(src_kind, Some(ValueType::Ref(_)))
                        && matches!(dst_kind, ValueType::Unsigned)
                    {
                        let bb_id = self.emit_bb(mir_bb);
                        let orig = arg.clone();
                        let (retype, result) =
                            push_ptr_to_unsigned_cast(&mut self.graph, bb_id, arg);
                        self.cast_ptr_to_int_src.insert(result.clone(), orig);
                        return Ok((Some(retype), result));
                    }
                    // `ann_cast_int_to_ptr(PtrT, s_int)` keeps dest PTRTYPE.
                    // Overlay `as *const T` must not alias or fold back to
                    // the GC orig (`rewrite_op_cast_opaque_ptr`); getattr
                    // terminator / w_cls resolves on that SomePtr.
                    let overlay_bb = self.emit_bb(mir_bb);
                    if let Some(raw) = self.emit_dest_overlay_cast(
                        overlay_bb,
                        src_kind.as_ref(),
                        arg.clone(),
                        dest_ty,
                    ) {
                        return Ok((None, raw));
                    }
                    // `p as usize as *mut T` is `lltype.cast_opaque_ptr`.
                    // Fold the int→ptr half back to the original GC pointer
                    // so OptEarlyForce does not force_box a virtual
                    // W_IntObject (`jtransform.py rewrite_op_cast_opaque_ptr`).
                    // A flavor-raw dest is int-banked (`rclass.py getgcflavor`);
                    // folding would leave a GC `*const ()` actual against
                    // FUNC.ARGS Int. Fall through to `ptr_cast_marker`.
                    if matches!(dst_kind, ValueType::Ref(_))
                        && flavor_raw_pointer_class_root(
                            dest_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            &self.raw_alloc_flavor_structs,
                        )
                        .is_none()
                        && let Some(orig) = self.cast_ptr_to_int_src.get(&arg).cloned()
                    {
                        return Ok((None, orig));
                    }
                    // `promote(p as usize) as *mut T` (`jit.py` `promote`,
                    // `Function.getcode`). The promote result is a new
                    // variable, so the map above misses. Recover the
                    // pointer and narrow it to the destination class.
                    // Returning the pre-cast `*const ()` leaves a
                    // classdef-less instance beside `Instance(PyObject)`.
                    if matches!(dst_kind, ValueType::Ref(_))
                        && let Some(orig) = self.address_cast_pointer(&arg)
                    {
                        match self.ptr_cast_marker(
                            Some(&ValueType::Ref(None)),
                            None,
                            dest_ty,
                            &orig,
                        ) {
                            Some((op, res)) => return Ok((Some(op), res)),
                            None => return Ok((None, orig)),
                        }
                    }
                    // An integer word cast to a non-byte raw scalar stays
                    // in the int bank. `history.py` `getkind` banks a raw
                    // `Ptr` as `int`, including `*mut f64`, so
                    // `usize as *const usize` aliases that word. A `Ref`
                    // source still reaches [`Self::ptr_cast_marker`].
                    // A function that *returns* `*mut i64` (`index_words_ptr`)
                    // is `rffi.cast` onto `CArrayPtr`; aliasing would keep
                    // `__cast_address_intrinsic`'s Instance for the caller's
                    // `direct_ptradd`. An items-base accessor or a store
                    // through the same spelling keeps the GcArray header.
                    if matches!(src_kind, Some(ValueType::Ref(_)))
                        && self.should_emit_rffi_cast_identity_to_raw_scalar(
                            dest_ty,
                            operand_tyref(&operand),
                        )
                    {
                        let bb_id = self.block_id[mir_bb];
                        let raw = self.emit_rffi_cast_to_raw_scalar_ptr(bb_id, arg, dest_ty);
                        return Ok((None, raw));
                    }
                    let dst_kind = if matches!(src_kind, Some(ValueType::Int | ValueType::Unsigned))
                        && let Some(raw) = raw_scalar_address_value_type(dest_ty, self.llbc)
                    {
                        raw
                    } else {
                        dst_kind
                    };
                    return Ok(
                        match src_kind
                            .as_ref()
                            .and_then(|s| cast_call_segments(s, &dst_kind))
                        {
                            Some(segments) => {
                                let res = self
                                    .graph
                                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                                if matches!(src_kind, Some(ValueType::Ref(_)))
                                    && matches!(dst_kind, ValueType::Int | ValueType::Unsigned)
                                {
                                    self.cast_ptr_to_int_src.insert(res.clone(), arg.clone());
                                }
                                (
                                    Some(OpKind::Call {
                                        target: CallTarget::function_path(segments),
                                        args: crate::model::call_args(vec![arg]),
                                        result_ty: dst_kind,
                                    }),
                                    res,
                                )
                            }
                            // Same bank (or a bank pair with no host cast
                            // callable): alias the operand — except a
                            // ptr→ptr cast whose destination pointee type
                            // the annotator must honour, which travels as a
                            // marker call ([`Lowering::ptr_cast_marker`],
                            // shared with the `Rvalue::Cast` arm).
                            None => {
                                match self.ptr_cast_marker(
                                    src_kind.as_ref(),
                                    src_root.as_deref(),
                                    dest_ty,
                                    &arg,
                                ) {
                                    Some((op, res)) => (Some(op), res),
                                    None => match self.raw_owner_retype(
                                        src_kind.as_ref(),
                                        src_root.as_deref(),
                                        dest_ty,
                                        &arg,
                                    ) {
                                        Some((op, res)) => (Some(op), res),
                                        None => match self.raw_ptr_to_signed(
                                            mir_bb,
                                            operand_is_declared_raw(
                                                &operand,
                                                self.llbc,
                                                self.gc_struct_ids,
                                                &self.raw_alloc_flavor_structs,
                                            ),
                                            dest_ty,
                                            &arg,
                                        ) {
                                            Some((op, res)) => (Some(op), res),
                                            None => (None, arg),
                                        },
                                    },
                                }
                            }
                        },
                    );
                }
                // Rust's `!` is bitwise complement on integers (`invert`)
                // but LOGICAL negation on `bool`.  `BoolRepr` has no
                // logical invert — it inherits `IntegerRepr`'s bitwise
                // `~` — so `!bool` lowered to `invert` is an op the rtyper
                // rejects.  Lower `!bool` to `eq(b, False)`, the orthodox
                // logical negation, leaving integer `!` on the `invert`
                // path.
                let operand_is_bool =
                    matches!(self.operand_value_kind(&operand), Some(ValueType::Bool));
                let op_label = unary_op_label(&op_json)?;
                let arg = self.resolve_operand(mir_bb, operand)?;
                if op_label == "invert" && operand_is_bool {
                    let false_var = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let bb_id = self.emit_bb(mir_bb);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(false_var.clone()),
                        kind: OpKind::ConstBool(false),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: arg,
                            rhs: false_var,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // Preserve f64 and 128-bit arithmetic destinations exactly.
                // The latter match RPython's LONG_TYPE/ULONG_TYPE carrier
                // through annotation/rtyping even though the JIT codewriter
                // deliberately has no 128-bit register kind.
                let result_ty = match tyref_to_value_type_with(
                    dest_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ) {
                    ValueType::Float => ValueType::Float,
                    ValueType::Unsigned => ValueType::Unsigned,
                    ValueType::Int128 => ValueType::Int128,
                    ValueType::UInt128 => ValueType::UInt128,
                    _ => ValueType::Int,
                };
                Ok((
                    Some(OpKind::UnaryOp {
                        op: op_label,
                        operand: arg,
                        result_ty,
                    }),
                    res,
                ))
            }
            // `Ref { place, ... }` — references in MIR are pointer-typed
            // aliases of the referent. The JIT does not model lifetimes,
            // and downstream consumers (codewriter, regalloc) operate on
            // the value flowing through the reference, not the reference
            // itself. Aliasing the dest local to the referent Variable
            // keeps the IR small, treating `&x` as a same-Variable copy.
            Rvalue::Ref { place, kind, .. } => {
                // `&mut *p` of a cell parameter is that cell. The reborrow
                // does not load `value`.
                if borrow_kind_is_exclusive(&kind)
                    && let Some(cell) = self.reborrow_cell_param(&place)
                {
                    let v = self
                        .local_var
                        .get(cell)
                        .and_then(Clone::clone)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: reborrow of GC reference cell {cell} before its definition"
                            ))
                        })?
                        .one()?;
                    return Ok((None, v));
                }
                if let Some(address) = self.lower_inline_substruct_address(mir_bb, &place)? {
                    return Ok((None, address));
                }
                // A `StringBuilder` is a GC reference (`STRINGBUILDERPTR`):
                // `&mut self._s` is the `getfield` of that reference, not the
                // address of an inline substructure. A transparent scalar
                // (`CodeFlags`) borrow is the word, not an address.
                let projection = self.place_borrow_takes_address(&place)
                    && !tyref_is_string_builder(&place.ty, self.llbc);
                let before = self.graph.block(self.emit_bb(mir_bb)).operations.len();
                // A Deref reborrow of an existing `&[u8]` keeps the identity.
                let v = self.resolve_place(mir_bb, place)?;
                self.mark_place_address_of(mir_bb, projection, before, &v);
                Ok((None, v))
            }
            // `RawPtr { place, ... }` — `&raw const x` / `&raw mut x`.
            // Same aliasing model as `Ref`: the JIT treats raw pointers
            // and references identically at the IR level (lifetime
            // tracking lives outside the JIT).
            // `&raw const *p` is the address `p`. Resolving the deref
            // would `RawLoad` the pointee (`rewrite_op_raw_load`).
            Rvalue::RawPtr { place, .. } => {
                if let Some(inner) = self.raw_primitive_reborrow_inner(&place) {
                    let v = self.resolve_place(mir_bb, inner)?;
                    return Ok((None, v));
                }
                if let Some(address) = self.lower_inline_substruct_address(mir_bb, &place)? {
                    return Ok((None, address));
                }
                let projection = self.place_borrow_takes_address(&place);
                let before = self.graph.block(self.emit_bb(mir_bb)).operations.len();
                let v = self.resolve_place(mir_bb, place)?;
                self.mark_place_address_of(mir_bb, projection, before, &v);
                Ok((None, v))
            }
            // `Repeat(elem, elem_ty, count)` — `[v; N]` literal. The
            // decodable shape follows upstream `transform.py` and
            // `rlist.py`: `newlist(fill)` followed by `mul(list, count)`,
            // which the transform pass collapses to `alloc_and_set`.
            // `do_fixed_newlist_clear`'s `new_array_clear` is that later
            // pass, so the repeat reaches the rtyper as
            // `__array_repeat(fill, count)`. The transparent `Array` ctor
            // is not equivalent: `Repeat` carries one fill value and a
            // count, not per-slot `__pos_N` writes.
            //
            // The result is the list object (`newlist` / `OpKind::NewList`
            // is `ValueType::Ref`). `rtype_alloc_and_set` concretetypes
            // that pointer `GcRef`. Declaring `Int` makes the legacy
            // walker `Signed` on the same variable. The dest's
            // `Array<T;N>` is the ARRAY identity `fixed_repeat_identity`
            // reads; `Ref(None)` leaves the rewrite to `LinkClasses`
            // matching a later index of the same SSA, which address-of
            // (`&mut mapping`) splits.
            Rvalue::Repeat(elem, _elem_ty, count, _) => {
                let arg = self.resolve_operand(mir_bb, elem)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let args = match self.emit_constant(mir_bb, &count) {
                    Ok(count) => vec![arg, count],
                    Err(_) => vec![arg],
                };
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec!["__array_repeat".to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: repeat_array_result_ty(dest_ty, self.llbc),
                    }),
                    res,
                ))
            }
            // `ShallowInitBox(elem, ty)` — `Box::new` half-construction
            // marker. The MIR emits this followed by an `Assign(*box,
            // value)` that fills the box contents. Modeled as a
            // synthetic 1-arg constructor call carrying the element.
            Rvalue::ShallowInitBox(elem, _ty) => {
                let arg = self.resolve_operand(mir_bb, elem)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("Box"),
                        args: crate::model::call_args(vec![arg]),
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `Cast(kind, operand, target_ty)` — numeric/pointer
            // coercion. The JIT does not track narrow integer widths,
            // so reuse the alias path: the cast result Variable is the
            // same as the operand Variable. `as` casts that do not
            // change the JIT-visible kind collapse this way.
            Rvalue::Cast(kind, operand, ty) => {
                // Classify the source BEFORE `resolve_operand` consumes
                // `operand` (mirrors the `UnaryOp::Cast` twin above): only a
                // Ref source may narrow or erase below.
                let src_kind = self.operand_value_kind(&operand);
                let src_root = self.operand_class_root(&operand);
                let src_raw_addr = operand_is_declared_raw(
                    &operand,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                let src_int = operand_tyref(&operand).and_then(|src| self.literal_int_width(src));
                let dst_int = self.literal_int_width(&ty);
                let v = self.resolve_operand(mir_bb, operand)?;
                if cast_kind_is_unsize(&kind)
                    && let Some((op, res)) = self.object_array_unsize_newlist(dest_ty, &v)
                {
                    return Ok((Some(op), res));
                }
                // A same-bank ptr→ptr cast keeps the i64 pointer carrier in
                // place, so it would alias — but the pointee type it
                // reinterprets to is load-bearing for the annotator, in both
                // directions ([`Lowering::ptr_cast_marker`]). `lltype.cast_pointer`
                // (`lltype.py`) is pointer-to-pointer. UnaryOp Cast already
                // calls `ptr_cast_marker` without a Charon kind gate;
                // `ptr_cast_marker` requires both sides Ref, so an int-to-ptr
                // Scalar still falls through. A flavor-raw dest must reach
                // the marker even when Charon spells the kind as something
                // other than `RawPtr` (`*const () as *const CodeObject`).
                let dest_flavor_raw = flavor_raw_pointer_class_root(
                    &ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    &self.raw_alloc_flavor_structs,
                )
                .is_some();
                if (cast_kind_is_raw_ptr(&kind) || dest_flavor_raw)
                    && let Some((op, res)) = self
                        .ptr_cast_marker(src_kind.as_ref(), src_root.as_deref(), &ty, &v)
                        .or_else(|| {
                            self.raw_owner_retype(src_kind.as_ref(), src_root.as_deref(), &ty, &v)
                        })
                {
                    return Ok((Some(op), res));
                }
                if let (Some(src), Some(dst)) = (src_int, dst_int)
                    && let Some(lowered) = self.materialize_int_cast(mir_bb, src, dst, v.clone())
                {
                    return Ok(lowered);
                }
                if let Some((op, res)) = self.raw_ptr_to_signed(mir_bb, src_raw_addr, &ty, &v) {
                    return Ok((Some(op), res));
                }
                if let Some((op, res)) = self.raw_ptr_to_unsigned(mir_bb, src_raw_addr, &ty, &v) {
                    return Ok((Some(op), res));
                }
                // `*mut Raw as *mut u8` / a GC pointer, and the reverse, cross
                // the int and ref banks. Aliasing would reuse one variable in
                // both argument lists.
                let dst_kind = tyref_to_value_type_with(
                    &ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                let dest_raw_addr = ty_is_raw_address(
                    &ty,
                    self.llbc,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                let crosses_raw_ref = (src_raw_addr && matches!(dst_kind, ValueType::Ref(_)))
                    || (matches!(src_kind, Some(ValueType::Ref(_))) && dest_raw_addr);
                if crosses_raw_ref
                    && let Some(src) = src_kind.as_ref()
                    && let Some(segments) = cast_call_segments(src, &dst_kind)
                {
                    let converted = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    if matches!(src, ValueType::Ref(_)) {
                        self.cast_ptr_to_int_src
                            .insert(converted.clone(), v.clone());
                    }
                    let bank = OpKind::Call {
                        target: CallTarget::function_path(segments),
                        args: crate::model::call_args(vec![v]),
                        result_ty: dst_kind,
                    };
                    // `cast_ptr_to_int` is `SomeInteger`. A following field
                    // read needs `SomePtr` of the destination raw owner,
                    // still in the int bank.
                    if matches!(src, ValueType::Ref(_))
                        && let Some(dest_root) = raw_address_owner_root(
                            &ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    {
                        self.graph.push_op_with_result_var(
                            self.emit_bb(mir_bb),
                            bank,
                            converted.clone(),
                        );
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((
                            Some(crate::model::cast_instance_call_result(
                                dest_root,
                                converted,
                                ValueType::Int,
                            )),
                            res,
                        ));
                    }
                    // `cast_int_to_ptr` is an opaque ref. A following field
                    // read needs the destination owner's class.
                    if src_raw_addr
                        && !matches!(src, ValueType::Ref(_))
                        && let Some(dest_root) =
                            tyref_class_root_with(&ty, self.llbc, self.tombstoned_leaves)
                    {
                        self.graph.push_op_with_result_var(
                            self.emit_bb(mir_bb),
                            bank,
                            converted.clone(),
                        );
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((
                            Some(crate::model::cast_instance_call(dest_root, converted)),
                            res,
                        ));
                    }
                    return Ok((Some(bank), converted));
                }
                Ok((None, v))
            }
            // `Len(place)` — slice / array length. Synthetic 1-arg
            // call; needs no descriptor for now.
            Rvalue::Len(place) => {
                // A pair slice's length is its second word, a raw-buffer
                // array's its type's length. A window's length is `FatLen`.
                if tyref_is_u8_window(&place.ty, self.llbc)
                    && let Some((_, len)) = self.pair_place_parts(mir_bb, &place)?
                {
                    return Ok((None, len));
                }
                if let Some(seq) = self.item_seq_of_place(&place) {
                    return Ok((None, self.item_seq_len(mir_bb, seq)?));
                }
                // A string-list or numeric-list items view is one word — the
                // `GcArray` header — so `__len` over it is `ll_fixed_length`
                // / `arraylen_gc`, the CAPACITY.  `ll_length` reads the
                // list's own `l.length` (`ll_length`), which the fold
                // recorded; answer with that, and keep the two apart
                // exactly as upstream does.
                if let Some(logical_len) = self.array_view_logical_len_for_place(&place) {
                    let bb_id = self.emit_bb(mir_bb);
                    return Ok((None, self.bind_array_view_len_in_block(bb_id, logical_len)));
                }
                let len_leaf = if self
                    .string_byte_view_locals
                    .iter()
                    .any(|local| place_references_local(&place, *local))
                {
                    "__strlen"
                } else {
                    "__len"
                };
                let base = self.resolve_place(mir_bb, place)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec![len_leaf.to_string()],
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![base]),
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `NullaryOp(op, ty)` — `SizeOf(T)`, `AlignOf(T)`, etc.
            // 0-arg synthetic Call carrying the op name.
            Rvalue::NullaryOp(op_json, _ty) => {
                let op_name = if let Some(s) = op_json.as_str() {
                    s.to_string()
                } else if let Some(obj) = op_json.as_object() {
                    obj.keys()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| "nullary".into())
                } else {
                    "nullary".into()
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::function_path([format!("__nullary_{op_name}")]),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `Aggregate(kind, operands)` — tuple / struct / enum-variant
            // / array construction. A unique, unescaped named struct is
            // `malloc(GcStruct)` plus one `setfield` per member; the
            // constructor call is a temporary marker so boxing fusion and
            // dead-aggregate sweep still see the construct-on-stack
            // spelling, then [`scalar_replace_named_struct_aggregates`]
            // replaces it with per-field SSA and a `New` per by-value
            // escape. Transparent newtype wrappers stay a no-op alias of
            // their inner operand. Enum variants, tuples, and arrays keep
            // `CallTarget::SyntheticTransparentCtor` for the later rewrites
            // that still match that shape.
            Rvalue::Aggregate(kind, operands) => {
                let shape = self
                    .aggregate_shape(&kind, dest_ty, operands.len())
                    .map_err(|msg| LowerError::Unsupported(format!("bb{mir_bb}: {msg}")))?;
                match shape {
                    AggregateShape::Discriminant(tag) => {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        Ok((Some(OpKind::ConstInt(tag)), res))
                    }
                    AggregateShape::UnsignedZero => {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        Ok((Some(OpKind::ConstUInt(0)), res))
                    }
                    // `Constant(None, Void)`, defined as `emit_unit` defines it.
                    AggregateShape::Void => {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                        Ok((Some(OpKind::ConstNone), res))
                    }
                    AggregateShape::NicheNull => {
                        let nullc = self.push_niche_null_ptr(mir_bb, dest_ty);
                        Ok((None, nullc))
                    }
                    AggregateShape::Operand { index, transparent } => {
                        let operand = operands
                            .into_iter()
                            .nth(index)
                            .expect("aggregate_shape checked the operand index");
                        let value = self.resolve_operand(mir_bb, operand)?;
                        // `GcRef(p as usize)`: the operand is the address integer
                        // from `cast_ptr_to_int` + `r_uint`, but the list item
                        // kind is the `GcRef::NULL` `ConstRefNull`
                        // (`Ptr(GCREF)`). Keep that pointer word.
                        // `llmemory.cast_ptr_to_adr` / `cast_opaque_ptr(..., GCREF)`
                        // is the same retype; `__cast_instance_intrinsic("GCREF")`
                        // annotates `SomePtr(GCREF)`.
                        if transparent && let Some(orig) = self.address_cast_pointer(&value) {
                            let res = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            self.pointer_word_vars.insert(res.clone());
                            return Ok((
                                Some(crate::model::cast_instance_call("GCREF", orig)),
                                res,
                            ));
                        }
                        Ok(self.retag_error_carrier(dest_ty, value))
                    }
                    AggregateShape::Ctor(ctor) => {
                        // A pair-slice operand is its pointer here and its
                        // length beside it; the field stores both words.
                        // Owner → window is only for a window field
                        // (`FmtArg::Bytes.__pos_0: &[u8]`). A `(&String,)`
                        // format interpolant stays the rust-heap header.
                        let mut arg_vars: Vec<Variable> = Vec::with_capacity(operands.len());
                        let mut pair_lens: Vec<Option<Variable>> =
                            Vec::with_capacity(operands.len());
                        for (i, op) in operands.into_iter().enumerate() {
                            if let Some((ptr, len)) = self.pair_value_operand(mir_bb, &op)? {
                                arg_vars.push(ptr);
                                pair_lens.push(Some(len));
                                continue;
                            }
                            if let Some((ptr, len)) = self.window_operand(mir_bb, &op)? {
                                arg_vars.push(ptr);
                                pair_lens.push(Some(len));
                                continue;
                            }
                            let field_is_window =
                                ctor.fields.get(i).is_some_and(|field| field.window);
                            // A STR-view `&[u8]` stored into a window field
                            // (`FmtArg::Bytes.__pos_0`) or into a Result/enum
                            // payload whose operand is that slice (`Result<[u8],
                            // PyError>::Ok`) borrows `{chars, length}` here
                            // (`rffi.get_nonmovingbuffer`). The sibling window
                            // operand already writes CCHARP + len; leaving the
                            // STR header as `__pos_0` unions Integer ∪ List.
                            if self.operand_is_string_byte_view(&op)
                                && (field_is_window
                                    || operand_tyref(&op)
                                        .is_some_and(|ty| tyref_is_u8_window(ty, self.llbc)))
                            {
                                let src_ty = operand_tyref(&op).map(clone_tyref);
                                let var = self.resolve_operand(mir_bb, op)?;
                                let (ptr, len) =
                                    self.owner_to_window(mir_bb, var, src_ty.as_ref())?;
                                arg_vars.push(self.ensure_window_ccharp(mir_bb, ptr));
                                pair_lens.push(Some(len));
                                continue;
                            }
                            if self.operand_is_list_view(&op)
                                && (field_is_window
                                    || operand_tyref(&op)
                                        .is_some_and(|ty| tyref_is_u8_window(ty, self.llbc)))
                            {
                                let var = self.resolve_operand(mir_bb, op)?;
                                let (ptr, len) = self.emit_list_view_to_window(mir_bb, var)?;
                                arg_vars.push(self.ensure_window_ccharp(mir_bb, ptr));
                                pair_lens.push(Some(len));
                                continue;
                            }
                            if field_is_window
                                && operand_tyref(&op).is_some_and(|src_ty| {
                                    tyref_is_string_value(src_ty, self.llbc)
                                        || tyref_raw_ptr_pointee_is_string_value(src_ty, self.llbc)
                                        || tyref_is_u8_list(src_ty, self.llbc)
                                })
                            {
                                let src_ty = operand_tyref(&op).map(clone_tyref);
                                let var = self.resolve_operand(mir_bb, op)?;
                                let (ptr, len) =
                                    self.owner_to_window(mir_bb, var, src_ty.as_ref())?;
                                arg_vars.push(ptr);
                                pair_lens.push(Some(len));
                                continue;
                            }
                            arg_vars.push(self.resolve_operand(mir_bb, op)?);
                            pair_lens.push(None);
                        }
                        if pair_lens.iter().any(Option::is_some)
                            && aggregate_ctor_name(&kind) == "Array"
                        {
                            return Err(LowerError::Unsupported(format!(
                                "bb{mir_bb}: pair slice item of an array aggregate"
                            )));
                        }
                        let bb_id = self.emit_bb(mir_bb);
                        // A host `0` stored into a GCREF virtualizable field is
                        // `ConstPtr(NULL)`, not an int (`coerce_gcref_null_store`).
                        for (i, arg) in arg_vars.iter_mut().enumerate() {
                            let Some(field) = ctor.fields.get(i) else {
                                continue;
                            };
                            let coerced = self.coerce_gcref_null_store(
                                bb_id,
                                &ctor.result_owner,
                                &field.name,
                                LinkArg::Value(arg.clone()),
                            );
                            if let Some(value) = coerced.as_variable() {
                                *arg = value.clone();
                            }
                        }
                        for (i, arg) in arg_vars.iter_mut().enumerate() {
                            let Some(field) = ctor.fields.get(i) else {
                                continue;
                            };
                            let Some(field_ty) = self.named_field_value_type(dest_ty, &field.name)
                            else {
                                continue;
                            };
                            let field_ty =
                                crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                                    &ctor.result_owner,
                                    &field.name,
                                    field_ty,
                                );
                            let coerced = self.coerce_signed_int_store(
                                bb_id,
                                &field_ty,
                                Some(field.name.as_str()),
                                LinkArg::Value(arg.clone()),
                            );
                            if let Some(value) = coerced.as_variable() {
                                *arg = value.clone();
                            }
                        }
                        let res = emit_aggregate_ctor_with_pairs(
                            &mut self.graph,
                            bb_id,
                            &ctor,
                            &arg_vars,
                            &pair_lens,
                        );
                        // Capture an exclusive int `Range` aggregate
                        // (`for _ in a..b`) so `front::range_iter`
                        // can divert it to the orthodox `range()` builtin +
                        // list-iter (`builtin.builtin_range`) — the
                        // `next`-diamond then folds the loop, instead of the
                        // union-failing shared `core.ops.range.Range` classdef.  Only
                        // int/uint element ranges are recorded; the divert itself is
                        // consumer-gated in the post-pass, so a range read by a
                        // slice-index / field read keeps this ctor + FieldWrite path.
                        if ctor.owner_path.as_slice() == ["core", "ops", "range"]
                            && ctor
                                .ctor_name
                                .split('<')
                                .next()
                                .unwrap_or(ctor.ctor_name.as_str())
                                == "Range"
                            && arg_vars.len() == 2
                            && self.aggregate_head_is_int_range(&kind)
                        {
                            // `0..n` is `Range<usize>` after the per-instantiation
                            // suffix. The for-loop divert matches the ctor result,
                            // not the leaf spelling, and replaces that value with
                            // `range()` before any consumer reads the struct.
                            self.range_iter_new_sites.push(
                                crate::front::range_iter::RangeNewSite {
                                    result_var: res.clone(),
                                    kind: crate::front::range_iter::RangeSiteKind::Exclusive,
                                },
                            );
                            self.slice_index_range_sites.push(
                                crate::front::slice_index::SliceIndexRangeSite {
                                    range_result: res.clone(),
                                    start: arg_vars[0].clone(),
                                    end: arg_vars[1].clone(),
                                },
                            );
                        }
                        if ctor.owner_path.as_slice() == ["core", "ops", "range"]
                            && ctor
                                .ctor_name
                                .split('<')
                                .next()
                                .unwrap_or(ctor.ctor_name.as_str())
                                == "RangeTo"
                            && arg_vars.len() == 1
                        {
                            self.slice_index_rangeto_sites.push(
                                crate::front::slice_index::SliceIndexRangeToSite {
                                    range_result: res.clone(),
                                    end: arg_vars[0].clone(),
                                },
                            );
                        }
                        if ctor.owner_path.as_slice() == ["core", "ops", "range"]
                            && ctor
                                .ctor_name
                                .split('<')
                                .next()
                                .unwrap_or(ctor.ctor_name.as_str())
                                == "RangeFrom"
                            && arg_vars.len() == 1
                        {
                            self.slice_index_rangefrom_sites.push(
                                crate::front::slice_index::SliceIndexRangeFromSite {
                                    range_result: res.clone(),
                                    start: arg_vars[0].clone(),
                                },
                            );
                        }
                        Ok((None, res))
                    }
                }
            }
            // `Discriminant(place)` — read the integer tag of an enum
            // value. Modeled as a synthetic `FieldRead` of an
            // `__discriminant` field: tag access is morally a pure
            // field read at the bit level, and reusing the existing
            // `FieldRead` shape keeps the IR closed under the opkind
            // catalogue (per `front/mod.rs` rule — no new OpKinds in
            // this layer).  The read is keyed to the enum base identity
            // (`owner_root = module::Enum`, `owner_id = StructId`) so it
            // resolves at the tag's real byte position — the enum base is
            // registered in `exact_layouts` with `__discriminant` at
            // `discriminant_offset()`.  A tag at offset 0 (the common
            // case) registers 0, identical to the prior heuristic.  An
            // unresolvable place type falls back to the unowned read.
            Rvalue::Discriminant(place) => {
                // The `Option<&T>` a pair-slice `get` yields is the item's
                // address, 0 for `None`.
                if let PlaceKind::Local(opt) = place.kind
                    && matches!(
                        self.local_var
                            .get(opt as usize)
                            .and_then(|slot| slot.as_ref()),
                        Some(LocalValue::ItemAddr(_))
                    )
                {
                    let addr = self.item_addr_var(mir_bb, opt as usize)?;
                    let zero = self.emit_const_uint(mir_bb, 0);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.niche_disc_vars.insert(res.id());
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "ne".to_string(),
                            lhs: addr,
                            rhs: zero,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                // `Option<&[u8]>` is a nullable window: `None` is a null
                // data word. The discriminant is that null test, not a
                // `__discriminant` field of a tagged aggregate. The data
                // word is CCHARP (`Ptr(Array Char)`); `p is None` is
                // `rnone.py` `rtype_is_None` → `ptr_iszero`, the same
                // `is_(opt, null)` then `eq(_, False)` sequence the
                // niche-pointer Discriminant arm emits (`push_niche_is_some`).
                // `Constant(None)` is Void (`pairtype(Repr, VoidRepr)`);
                // integer 0 is not a pointer null (`lltype.nullptr`).
                if tyref_is_u8_window(&place.ty, self.llbc)
                    && let Some((data, _)) = self.pair_place_parts(mir_bb, &place)?
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                    let is_none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let false_var = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let bb_id = self.emit_bb(mir_bb);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(none.clone()),
                        kind: OpKind::ConstNone,
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(is_none.clone()),
                        kind: OpKind::BinOp {
                            op: "is_".to_string(),
                            lhs: data,
                            rhs: none,
                            result_ty: ValueType::Int,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(false_var.clone()),
                        kind: OpKind::ConstBool(false),
                    });
                    self.niche_disc_vars.insert(res.id());
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: is_none,
                            rhs: false_var,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                // `owner_root` is the annotation-side classdef key: a
                // reference-payload enum instantiation reads its tag off
                // the per-instantiation base class (`Option<Result<…>>`),
                // the same spelling the constructor / receiver / field-read
                // sites project, so the `__discriminant` read and the
                // constructor's `setattr` land on ONE classdef per
                // instantiation (`enum_variant_narrowing_knowntypedata`
                // then mints matching variant subclasses). `owner_id` is the
                // independent physical-layout identity and preserves every
                // concrete generic argument.
                let (owner_root, owner_id) = match self.tyref_adt_class_root(&place.ty) {
                    Some(class_root) => {
                        let canon = decl_path_for_tombstone(&class_root, self.tombstoned_leaves);
                        let sid = self.tyref_adt_layout_id(&place.ty).unwrap_or_else(|| {
                            let bare = majit_ir::descr::strip_instantiation_suffix(&canon);
                            majit_ir::descr::StructId::from_canonical(bare)
                        });
                        (Some(canon), Some(sid))
                    }
                    None => (None, None),
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // A local bound directly to its payload by a decomposed
                // always-`Ok` conversion (`try_lower_usize_try_from`)
                // carries no runtime enum object — no `__discriminant`
                // field exists to read.  Its tag is the recorded
                // constant, so fold to `ConstInt(tag)` (mirrors
                // `expect_on_const_ok`, which aliases the payload).
                if let PlaceKind::Local(i) = &place.kind
                    && let Some(&tag) = self.const_discriminant_locals.get(&(*i as usize))
                {
                    return Ok((Some(OpKind::ConstInt(tag)), res));
                }
                // `Discriminant` of an INLINE enum field (`self.strategy`):
                // the field is stored by value, so the container holds the
                // enum's bytes directly — there is no pointer to follow.
                // Reading `&self.strategy` as a `getfield_gc_r` would load
                // the field's first word as a bogus pointer (the deref then
                // faults on the tag read).  When the tag sits at the field's
                // base (the C-like repr, `inline_enum_field_disc_offset ==
                // Some(0)`), the field's own offset already addresses the
                // tag, so read the tag in ONE getfield from the container at
                // the field offset — matching the runtime `strategy` field
                // descr (offset 32, size 1, Int) the production fold uses.
                // A non-zero tag offset falls through to the generic read.
                let inline_enum_field =
                    if let PlaceKind::Projection(f_base, ProjectionElem::Tagged(v)) = &place.kind
                        && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                        && let Some((f_owner_root, f_name, f_ty, f_owner_id, f_host_index)) =
                            self.resolve_adt_field(&f_base.ty, field_payload)
                        && self.inline_fieldless_enum_field_tag0(&f_ty)
                    {
                        Some((f_owner_root, f_name, f_owner_id, f_host_index))
                    } else {
                        None
                    };
                if let Some((f_owner_root, f_name, f_owner_id, f_host_index)) = inline_enum_field {
                    let PlaceKind::Projection(inner, _) = place.kind else {
                        unreachable!("inline_enum_field implies a Projection place")
                    };
                    let base = self.resolve_place(mir_bb, *inner)?;
                    return Ok((
                        Some(OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(f_name, Some(f_owner_root))
                                .with_owner_id(f_owner_id)
                                .with_host_index(f_host_index),
                            ty: ValueType::Int,
                            pure: true,
                        }),
                        res,
                    ));
                }
                // A standalone fieldless (C-like) enum is represented
                // by-value as its discriminant integer, so
                // `Discriminant(enum)` is the identity on that value — alias
                // the place value with no field read (mirroring `Rvalue::Use`
                // / the const-tag fold above).  Modelling the value as the
                // int is the orthodox representation (RPython has no enum
                // type; a fieldless enum is a named `int`), and it means the
                // downstream switch reads the int directly rather than a
                // `__discriminant` field off an aggregate base that carries
                // no rtype clsfield.  The INLINE-field case is handled above
                // (the tag is read from the container at the field offset),
                // so this only fires for a standalone enum value.
                //
                // This identity must win over `lower_raw_pointer_discriminant`:
                // a borrow of a fieldless enum is the same tag
                // (`tyref_is_borrowed_fieldless_enum`), and loading the tag
                // *from* that word treats the integer as an address
                // (`rewrite_op_raw_load`).
                if self.tyref_is_fieldless_enum(&place.ty)
                    || self.tyref_is_borrowed_fieldless_enum(&place.ty)
                {
                    let base = self.resolve_place(mir_bb, place)?;
                    return Ok((None, base));
                }
                // `*p` on a raw address is the enum's bytes, not the
                // explicit sum shell. Load the host tag and decode it to
                // the variant index `host_variant_index` returns;
                // `SwitchInt` on this `Discriminant` compares that index.
                if let Some(decoded) = self.lower_raw_pointer_discriminant(mir_bb, &place)? {
                    return Ok((None, decoded));
                }
                // A one-word niche `Option` is represented by its payload
                // pointer: `None`
                // = null, `Some(p)` = the non-null pointer.  Its
                // discriminant is the pointer-null test `base != null`
                // (`None` = 0, `Some` = 1), not a `__discriminant` field off
                // a two-word aggregate base (there is none).  Emit the null
                // pointer, then compare — a `ne` on two `Ref` operands
                // lowers to `ptr_ne` (`jtransform`), and its `Int` result
                // feeds the match switch exactly as a `__discriminant` read
                // would.  The null is a `null_mut()` call, not a
                // `ConstRefNull`: `null_mut()` rtypes to `convert_const(None)`
                // on the result repr (`rbuiltin::rtype_ptr_null`), so it
                // adapts to the `base` pointer's own `InstanceRepr` and the
                // `ptr_ne` sees two same-repr operands.  A `ConstRefNull` is a
                // fixed GCREF `_ptr`, which cannot rtype against an
                // `InstanceRepr(PyObject)` base at the `ne`.  Modelling the
                // maybe-null pointer this way is the orthodox representation
                // (RPython has no `Option`; a maybe-null `Ptr` tests
                // `ptr_nonzero`).
                if self.tyref_is_niche_option_ptr(&place.ty) {
                    let option_ty = clone_tyref(&place.ty);
                    let base = self.resolve_place(mir_bb, place)?;
                    let nullc = self.push_niche_null_ptr(mir_bb, &option_ty);
                    // Preserve PyPy's identity test: first compute
                    // `base is None`, then negate that bool for the Rust
                    // discriminant (`Some` = 1). `ne` would dispatch to value
                    // inequality on StringRepr/ListRepr, which is not a null
                    // test (and FixedSizeListRepr deliberately has no such
                    // operation). The final bool must close a `SwitchInt`
                    // with `ExitCase::Bool`, so record its result id.
                    let is_none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let false_var = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let bb_id = self.emit_bb(mir_bb);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(is_none.clone()),
                        kind: OpKind::BinOp {
                            op: "is_".to_string(),
                            lhs: base,
                            rhs: nullc,
                            result_ty: ValueType::Int,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(false_var.clone()),
                        kind: OpKind::ConstBool(false),
                    });
                    self.niche_disc_vars.insert(res.id());
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: is_none,
                            rhs: false_var,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                // `Option<E>` where `E` is a densely numbered fieldless enum
                // is also a one-word niche value.  Rust stores `Some(e)` as
                // `e` and reserves the first value after E's dense range for
                // `None`; there is no aggregate whose `__discriminant` can be
                // read.  Keep the physical scalar in the graph and turn the
                // source-level discriminant into the same boolean shape as
                // the nullable-pointer Option above (`Some` = scalar != niche).
                if let Some(niche) = tyref_option_fieldless_niche(&place.ty, self.llbc) {
                    let base = self.resolve_place(mir_bb, place)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(none.clone()),
                        kind: OpKind::ConstInt(niche.none_tag),
                    });
                    self.niche_disc_vars.insert(res.id());
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "ne".to_string(),
                            lhs: base,
                            rhs: none,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                // `Option<NonZero*>`: `None` is the zero the payload forbids.
                // The class root is the spelling `tyref_adt_class_root` already
                // resolved (`core::option::Option<NonZero<usize>>`). The
                // payload walk misses some hash-consed generics that this
                // suffix still names.
                let nonzero_kind = option_nonzero_word_kind_deref_receiver(&place.ty, self.llbc);
                if let Some(kind) = nonzero_kind {
                    let unsigned = kind == ValueType::Unsigned;
                    let base = self.resolve_place(mir_bb, place)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(none.clone()),
                        kind: if unsigned {
                            OpKind::ConstUInt(0)
                        } else {
                            OpKind::ConstInt(0)
                        },
                    });
                    self.niche_disc_vars.insert(res.id());
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "ne".to_string(),
                            lhs: base,
                            rhs: none,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                let disc_owner_gc = container_declared_gc(
                    &place.ty,
                    self.llbc,
                    self.gc_struct_ids,
                    self.immutable_fields,
                );
                let base = self.resolve_place(mir_bb, place)?;
                Ok((
                    Some(OpKind::FieldRead {
                        base,
                        field: crate::model::FieldDescriptor {
                            name: "__discriminant".to_string(),
                            owner_root,
                            owner_id,
                            base_is_deref: None,
                            taken_by_address: false,
                            inline_vec: false,
                            vec_part: None,
                            owner_declared_gc: disc_owner_gc,
                            host_index: None,
                            scalar_word: None,
                        },
                        ty: ValueType::Int,
                        pure: true,
                    }),
                    res,
                ))
            }
            other => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Rvalue::{} not yet supported",
                rvalue_variant_name(&other)
            ))),
        }
    }

    /// Address of `*p` when `p` is one raw pointer. `None` for any other place.
    pub(in crate::front::mir) fn raw_deref_address(
        &mut self,
        mir_bb: usize,
        place: &Place,
    ) -> Result<Option<Variable>, LowerError> {
        let PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) = &place.kind else {
            return Ok(None);
        };
        if elem != "Deref"
            || !ty_is_raw_address(
                &inner.ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
        {
            return Ok(None);
        }
        Ok(Some(self.resolve_place(mir_bb, (**inner).clone())?))
    }

    /// `Discriminant(*p)` and `Discriminant((*p).field)` when the enum is
    /// a Raw host layout with a tag. `None` leaves the explicit-shell read.
    pub(in crate::front::mir) fn lower_raw_pointer_discriminant(
        &mut self,
        mir_bb: usize,
        place: &Place,
    ) -> Result<Option<Variable>, LowerError> {
        let Some(tag) = self.raw_enum_host_tag(&place.ty) else {
            return Ok(None);
        };
        let (base, extra_offset) = match &place.kind {
            PlaceKind::Projection(inner, ProjectionElem::Atom(elem))
                if elem == "Deref"
                    && ty_is_raw_address(
                        &inner.ty,
                        self.llbc,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ) =>
            {
                (self.resolve_place(mir_bb, (**inner).clone())?, 0u64)
            }
            PlaceKind::Projection(inner, ProjectionElem::Tagged(v))
                if let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some(field_off) =
                        self.struct_field_offset_from_payload(&inner.ty, field_payload)
                    && let Some(addr) = self.raw_deref_address(mir_bb, inner)? =>
            {
                (addr, field_off)
            }
            _ => return Ok(None),
        };
        let Some(total) = extra_offset.checked_add(tag.offset) else {
            return Ok(None);
        };
        let Some(total) = i64::try_from(total).ok() else {
            return Ok(None);
        };
        let offset = self.emit_i64(mir_bb, total);
        let raw = self.emit_raw_tag_load(mir_bb, base, offset, (tag.bits / 8) as usize);
        Ok(Some(self.emit_host_variant_index(mir_bb, raw, &tag)))
    }

    /// Host tag of the enum `ty` names, when it fits in one int-bank load.
    pub(in crate::front::mir) fn raw_enum_host_tag(
        &self,
        ty: &TyRef,
    ) -> Option<majit_charon_reader::ullbc::TagLayout> {
        let body = self.tyref_adt_body(ty)?;
        let def_id = inline_adt_def_id(body)?;
        let td = self.llbc.type_by_id(def_id)?;
        if !matches!(&td.kind, TypeDeclKind::Enum(_)) {
            return None;
        }
        let target = std::env::var("TARGET").unwrap_or_default();
        let tag = td.layout_for_target(self.llbc, &target)?.tag()?;
        let word_bits = (crate::layout::target_word_size() * 8) as u32;
        if tag.bits == 0 || tag.bits > word_bits || tag.bits % 8 != 0 {
            return None;
        }
        i64::try_from(tag.offset).ok()?;
        Some(tag)
    }

    pub(in crate::front::mir) fn emit_i64(&mut self, mir_bb: usize, value: i64) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        self.graph
            .push_op_var(bb_id, OpKind::ConstInt(value), true)
            .expect("ConstInt produces a value")
    }

    pub(in crate::front::mir) fn emit_binop(
        &mut self,
        mir_bb: usize,
        op: &str,
        lhs: Variable,
        rhs: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        self.graph
            .push_op_var(
                bb_id,
                OpKind::BinOp {
                    op: op.to_string(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .expect("BinOp produces a value")
    }

    /// Zero-extending load of the tag word. `host_variant_index` masks to
    /// `tag.bits` before the niche subtract, so the read is unsigned.
    pub(in crate::front::mir) fn emit_raw_tag_load(
        &mut self,
        mir_bb: usize,
        base: Variable,
        offset: Variable,
        itemsize: usize,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        self.graph
            .push_op_var(
                bb_id,
                OpKind::RawLoad {
                    base,
                    offset,
                    item_ty: ValueType::Int,
                    itemsize,
                    is_item_signed: false,
                },
                true,
            )
            .expect("raw tag load produces a value")
    }

    /// `mask(bits, raw)` from `host_variant_index`. A full int-bank word
    /// needs no mask.
    pub(in crate::front::mir) fn mask_to_tag_width(
        &mut self,
        mir_bb: usize,
        raw: Variable,
        bits: u32,
    ) -> Variable {
        if bits >= 64 {
            return raw;
        }
        let mask = self.emit_i64(mir_bb, (1i64 << bits) - 1);
        self.emit_binop(mir_bb, "and", raw, mask)
    }

    /// Variant index of a raw tag word. Direct tags that are already
    /// `0..n` are the masked word. Any other direct list is an equality
    /// sum. A niche is `host_variant_index`'s unsigned compare:
    /// `relative = mask(raw - niche_start)`, then
    /// `untagged + in_span * ((start + relative) - untagged)` where
    /// `in_span` is `uint_le(relative, end - start)` (0 or 1).
    pub(in crate::front::mir) fn emit_host_variant_index(
        &mut self,
        mir_bb: usize,
        raw: Variable,
        tag: &majit_charon_reader::ullbc::TagLayout,
    ) -> Variable {
        use majit_charon_reader::ullbc::TagEncoding;

        let masked = self.mask_to_tag_width(mir_bb, raw, tag.bits);
        match &tag.encoding {
            TagEncoding::Direct { tags } if Self::tag_words_are_variant_indices(tags) => masked,
            TagEncoding::Direct { tags } => {
                self.emit_direct_tag_index(mir_bb, masked, tag.bits, tags)
            }
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => self.emit_niche_tag_index(
                mir_bb,
                masked,
                tag.bits,
                *untagged_variant,
                niche_variants,
                *niche_start,
            ),
        }
    }

    pub(in crate::front::mir) fn emit_direct_tag_index(
        &mut self,
        mir_bb: usize,
        masked: Variable,
        bits: u32,
        tags: &[u128],
    ) -> Variable {
        let mut acc = self.emit_i64(mir_bb, 0);
        for (index, known) in tags.iter().enumerate() {
            if index == 0 {
                continue;
            }
            let known_c = self.emit_i64(mir_bb, Self::tag_word_const(bits, *known));
            let is_here = self.emit_binop(mir_bb, "eq", masked.clone(), known_c);
            let term = if index == 1 {
                is_here
            } else {
                let index_c = self.emit_i64(mir_bb, index as i64);
                self.emit_binop(mir_bb, "mul", is_here, index_c)
            };
            acc = self.emit_binop(mir_bb, "add", acc, term);
        }
        acc
    }

    pub(in crate::front::mir) fn emit_niche_tag_index(
        &mut self,
        mir_bb: usize,
        masked: Variable,
        bits: u32,
        untagged_variant: usize,
        niche_variants: &std::ops::RangeInclusive<usize>,
        niche_start: u128,
    ) -> Variable {
        let start_c = self.emit_i64(mir_bb, Self::tag_word_const(bits, niche_start));
        let diff = self.emit_binop(mir_bb, "sub", masked, start_c);
        let relative = self.mask_to_tag_width(mir_bb, diff, bits);
        let start = *niche_variants.start();
        let end = *niche_variants.end();
        let span = self.emit_i64(mir_bb, (end - start) as i64);
        let in_span = self.emit_binop(mir_bb, "uint_le", relative.clone(), span);
        let index_base = self.emit_i64(mir_bb, start as i64);
        let niche_index = self.emit_binop(mir_bb, "add", index_base, relative);
        let untagged = self.emit_i64(mir_bb, untagged_variant as i64);
        let delta = self.emit_binop(mir_bb, "sub", niche_index, untagged.clone());
        let scaled = self.emit_binop(mir_bb, "mul", in_span, delta);
        self.emit_binop(mir_bb, "add", untagged, scaled)
    }

    pub(in crate::front::mir) fn tag_words_are_variant_indices(tags: &[u128]) -> bool {
        tags.iter()
            .enumerate()
            .all(|(index, word)| *word == index as u128)
    }

    /// Low `bits` of `raw` as an int-bank constant. The niche subtract and
    /// the direct equality both run on that bit pattern.
    pub(in crate::front::mir) fn tag_word_const(bits: u32, raw: u128) -> i64 {
        let masked = if bits >= 64 {
            raw
        } else {
            raw & ((1u128 << bits) - 1)
        };
        masked as u64 as i64
    }

    pub(in crate::front::mir) fn resolve_operand(
        &mut self,
        mir_bb: usize,
        op: Operand,
    ) -> Result<Variable, LowerError> {
        match op {
            Operand::Copy(place) => self.resolve_place(mir_bb, place),
            Operand::Move(place) => {
                // A `move c` of a builder-mode accumulator is its single
                // `ll_build` materialisation point
                // (`accumulator_materialization_site_count == 1`). Emit the
                // `__majit_stringbuilder_build` marker
                // (`flowspace_adapter` → getattr("build") + simple_call →
                // `SomeString`) instead of moving the concat accumulator out.
                // Only a bare `Local(c)` move qualifies; a projection falls
                // through. Inert unless this function selected builder mode and
                // the local is an accumulator (read from the per-body table).
                if let PlaceKind::Local(i) = place.kind
                    && self.builder_mode
                    && self.accumulator_is_builder_mode(i as usize)
                {
                    return self.resolve_builder_build(mir_bb, i as usize);
                }
                self.resolve_place(mir_bb, place)
            }
            Operand::Const(value) => self.emit_constant(mir_bb, &value),
        }
    }

    /// Emit the `__majit_stringbuilder_build(c)` marker for the terminal
    /// `move c` of a builder-mode accumulator, returning the built
    /// `SomeString` Variable.  The builder Ref was minted at the ctor and
    /// mutated in place by the appends, so it is read (not rebound) here.
    pub(in crate::front::mir) fn resolve_builder_build(
        &mut self,
        mir_bb: usize,
        c: usize,
    ) -> Result<Variable, LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let builder = self.local_var[c]
            .as_ref()
            .ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "builder-mode accumulator local {c} unbound at its build site"
                ))
            })?
            .one()?;
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::STRINGBUILDER_BUILD.to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![builder]),
                result_ty: ValueType::Ref(None),
            },
        });
        Ok(result)
    }

    /// Encode a `String::push` / `Wtf8Buf::push` code-point piece as a
    /// one-code-point UTF-8 string (`rutf8.py unichr_as_utf8`,
    /// `allow_surrogates=True`).  An ASCII / Unicode-scalar constant
    /// becomes `__str_const`; a variable code point becomes
    /// `__majit_unichr_as_utf8`.
    pub(in crate::front::mir) fn emit_codepoint_as_utf8(
        &mut self,
        mir_bb: usize,
        piece: &Variable,
        piece_operand: Option<&Operand>,
    ) -> Result<Variable, LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        if let Some(Operand::Const(value)) = piece_operand
            && let Ok(decoded) = decode_constant(self.llbc, value)
        {
            let code = match decoded {
                DecodedConst::Int(n) => Some(n),
                DecodedConst::UInt(n) => i64::try_from(n).ok(),
                DecodedConst::UniChar(code) => Some(i64::from(code)),
                _ => None,
            };
            if let Some(code) = code
                && let Some(text) = unichr_as_utf8_text(code)
            {
                return Ok(emit_str_const(&mut self.graph, bb_id, &text));
            }
        }
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        stamp_as_rstr_str(&result);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![crate::runtime_names::shims::UNICHAR_AS_UTF8.to_string()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![piece.clone()]),
                result_ty: ValueType::Str,
            },
        });
        Ok(result)
    }

    /// If `arg_locals[0]` is the sole identity argument of a builder-mode
    /// accumulator (`as_bytes` / `deref` / `as_str` on that builder), emit
    /// `ll_build` (`AbstractStringBuilderRepr.rtype_method_build`) and
    /// return the finished `SomeString`.  The identity arm then aliases
    /// dest to that string.  `None` keeps the existing identity alias.
    pub(in crate::front::mir) fn builder_identity_string(
        &mut self,
        mir_bb: usize,
        arg_locals: &[Option<usize>],
    ) -> Result<Option<Variable>, LowerError> {
        if self.builder_mode
            && let Some(c) = arg_locals
                .first()
                .copied()
                .flatten()
                .and_then(|rt| self.accumulator_of_identity_arg(rt))
            && self.accumulator_is_builder_mode(c)
        {
            return self.resolve_builder_build(mir_bb, c).map(Some);
        }
        Ok(None)
    }

    /// The [`ValueType`] of a `Copy` / `Move` operand, read from the
    /// operand place's post-projection type.  Used by the `Cast`
    /// lowering to decide whether the cast crosses a register bank.
    /// Returns `None` for `Const` operands (a constant carries no place
    /// type here; a const-source cast aliases its operand).
    pub(in crate::front::mir) fn operand_value_kind(&self, op: &Operand) -> Option<ValueType> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => Some(tyref_to_value_type_with(
                &place.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )),
            Operand::Const(_) => None,
        }
    }

    /// The monomorphic-ADT class root of a `Copy` / `Move` operand's
    /// place type, or `None` for a `Const` operand / a pointee-less
    /// pointer.  Read by [`Self::ptr_cast_marker`] to tell a genuine
    /// type erasure (`*mut PyFrame` → `*mut u8`) apart from a cast whose
    /// source already carries no pointee class.
    pub(in crate::front::mir) fn operand_class_root(&self, op: &Operand) -> Option<String> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => {
                tyref_class_root_with(&place.ty, self.llbc, self.tombstoned_leaves)
            }
            Operand::Const(_) => None,
        }
    }

    /// `__cast_instance_intrinsic` for `raw as *mut T`.
    ///
    /// The constant is the class root: input `class_root`,
    /// `getuniqueclassdef_for_struct_root`, and the virtualizable set all
    /// key that leaf. The layout map and a constructor owner are the
    /// crate-relative declaration path of the same ADT. That path rides
    /// on the result type, so a later `ptr::write` compares owners by
    /// path. A class root that is already qualified, or that carries an
    /// instantiation suffix, stays the result type too.
    ///
    /// A GCREF source recast onto a tuple or list container is
    /// `KwargsDictStrategy.copy` unerase (`kwargsdict.py`):
    /// `lltype.cast_opaque_ptr` from GCREF to the tuple of two lists.
    /// Aliasing would leave `SomePtr(GCREF)` for a later `dstorage[0]`
    /// getitem (`rtuple.py` `rtype_getitem`). Named-struct dests keep
    /// the classdef arm regardless of source.
    pub(in crate::front::mir) fn instance_narrow_for_type(
        &self,
        dest_ty: &TyRef,
        arg: &Variable,
        src_root: Option<&str>,
    ) -> Option<OpKind> {
        // Fieldless enums are ints. A classdef here is the
        // `pair(rtype_eq) (InstanceRepr, IntegerRepr)` wall: the tag
        // compare becomes `eq` of a minted empty instance with `ConstInt`.
        if tyref_fieldless_enum_pointee(dest_ty, self.llbc).is_some() {
            return None;
        }
        // Same peel as [`tyref_to_value_type_with`]: a transparent scalar
        // wrapper (by value or through `&` / `&mut`) is the word, so a
        // call dest or pointer-cast must not mint `Ptr(W)`. A pointer
        // newtype whose inner is a Ref is still a class dest.
        if let Some(inner) = tyref_transparent_inner_value_type(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ) && value_type_is_transparent_scalar_word(&inner)
        {
            return None;
        }
        if let Some(root) = tyref_class_root_with(dest_ty, self.llbc, self.tombstoned_leaves) {
            let result_owner = self
                .cast_dest_layout_owner(dest_ty)
                .filter(|owner| {
                    !root.contains("::")
                        && !root.contains('<')
                        && owner.rsplit("::").next() == Some(root.as_str())
                })
                .unwrap_or_else(|| root.clone());
            return Some(if result_owner == root {
                crate::model::cast_instance_call(root, arg.clone())
            } else {
                crate::model::cast_instance_call_result(
                    root,
                    arg.clone(),
                    ValueType::Ref(Some(result_owner)),
                )
            });
        }
        if src_root.is_some_and(crate::annotator::bookkeeper::is_gcref_type_spelling) {
            let node = strip_ty_wrappers(tyref_node(dest_ty, self.llbc)?, self.llbc)?;
            if let Some(root) = raw_ptr_pointee_container_root(node, self.llbc) {
                return Some(crate::model::cast_instance_call(root, arg.clone()));
            }
        }
        None
    }

    /// Declaration path of `dest_ty`'s struct, reduced the way a
    /// constructor owner is. `None` when the pointee is not a struct.
    pub(in crate::front::mir) fn cast_dest_layout_owner(&self, dest_ty: &TyRef) -> Option<String> {
        let node = strip_ty_wrappers(tyref_node(dest_ty, self.llbc)?, self.llbc)?;
        let pointee = node
            .as_object()
            .and_then(|obj| obj.get("RawPtr"))
            .and_then(|raw| raw.as_array())
            .and_then(|raw| raw.first())
            .and_then(|pointee| strip_ty_wrappers(pointee, self.llbc))
            .unwrap_or(node);
        let id = adt_node_def_id(pointee)?;
        let decl = self.llbc.type_by_id(id)?;
        if !matches!(decl.kind, TypeDeclKind::Struct(_)) {
            return None;
        }
        Some(crate::model::crate_relative_struct_owner(
            &decl.item_meta.name_path(),
            self.graph.name.split("::").next(),
        ))
    }

    /// Dest PTRTYPE for a raw overlay struct (`*const MapNodeAbstractAttribute`
    /// / `*const MapNodeTerminatorView`). `ann_cast_int_to_ptr` returns
    /// `SomePtr(PTRTYPE)`; `SomePtr.getattr` resolves on the example fields.
    /// Decide by GC-ness, not class root: a raw (non-GC) `lltype.Struct` is
    /// never an instance (`lltype.py` `ann_cast_int_to_ptr`). A GC struct
    /// stays on [`Self::instance_narrow_for_type`]. `MapNode` is an Enum, so
    /// `as MapRef` stays 1-arg Opaque.
    pub(in crate::front::mir) fn dest_overlay_ptr_lltype(
        &self,
        dest_ty: &TyRef,
    ) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
        use crate::flowspace::model::ConstValue;
        use crate::translator::rtyper::lltypesystem::lltype::{
            GcKind, LowLevelType, Ptr, Struct, frozendict,
        };
        let pointee = tyref_peel_one_raw_ptr_node(dest_ty, self.llbc)?;
        // `ann_cast_int_to_ptr(PTRTYPE, s_int)` keeps PTRTYPE; whether that
        // is a raw `Ptr(Struct)` or a GC pointer is `PTRTYPE.TO._gckind`.
        let node = declared_raw_adt_node(
            pointee,
            self.llbc,
            self.gc_struct_ids,
            self.immutable_fields,
        )?;
        let id = adt_node_def_id(node)?;
        let decl = self.llbc.type_by_id(id)?;
        let TypeDeclKind::Struct(fields) = &decl.kind else {
            return None;
        };
        let name = crate::model::crate_relative_struct_owner(
            &decl.item_meta.name_path(),
            self.graph.name.split("::").next(),
        );
        let mut ll_fields = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let fname = field_decl_name(i, field);
            let fty = overlay_field_lltype(&field.ty, self.llbc, self.gc_struct_ids);
            ll_fields.push((fname, fty));
        }
        let mut body = Struct::registry_fields(&name, ll_fields, GcKind::Raw);
        // `_immutable_fields_` from the table harvested once for this
        // artefact (`harvest_immutable_fields_from_llbcs`), the same
        // source GC struct schemas use (`program.immutable_fields`).
        // `with_hints` rejects underscore names; overlay has `_discriminant`.
        if let Some(ranks) = type_decl_immutable_fields(decl, self.immutable_fields) {
            let mut dict = indexmap::IndexMap::new();
            for (fname, _) in ranks {
                dict.insert(ConstValue::byte_str(fname), ConstValue::Bool(true));
            }
            body._hints = frozendict::new(vec![(
                "immutable_fields".into(),
                ConstValue::Dict(std::sync::Arc::new(dict)),
            )]);
        }
        let ptr = Ptr::from_container_type(LowLevelType::Struct(Box::new(body))).ok()?;
        Some(LowLevelType::Ptr(Box::new(ptr)))
    }

    /// 2-arg `cast_int_to_ptr(PTRTYPE, addr)` when dest is a raw overlay.
    /// A Ref source goes through `cast_ptr_to_int` first (`rffi.cast`).
    pub(in crate::front::mir) fn emit_dest_overlay_cast(
        &mut self,
        bb_id: BlockId,
        src_kind: Option<&ValueType>,
        arg: Variable,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let ptr_ty = self.dest_overlay_ptr_lltype(dest_ty)?;
        let addr = if matches!(src_kind, Some(ValueType::Ref(_))) {
            push_cast_ptr_to_int(&mut self.graph, bb_id, arg)
        } else {
            arg
        };
        Some(emit_dest_typed_cast_int_to_ptr(
            &mut self.graph,
            bb_id,
            addr,
            ptr_ty,
        ))
    }

    /// The marker call a same-bank pointer-to-pointer reinterpret lowers
    /// to, or `None` when the cast simply aliases its operand.
    ///
    /// Two shapes share this decision, so both `Cast` lowering arms route
    /// through here:
    ///
    /// * **erase** — `p as *mut u8` drops the pointee class.  Aliasing
    ///   would keep `SomeInstance(PyFrame)` on a value whose declared type
    ///   is a bare byte pointer, so a callee taking `*mut u8` from
    ///   heterogeneous heap blocks (`gc_hook::try_gc_owns_object`,
    ///   `try_gc_write_barrier`) unions `PyFrame ∪ ItemsBlock` at its slot
    ///   0 and blocks with "cannot unify instances with no common base
    ///   class".  `llmemory.cast_ptr_to_adr` is the upstream erasure: a
    ///   GC ownership / write-barrier hook operates on a type-erased
    ///   `Address`, never on `Ptr(instance)`.  `__cast_address_intrinsic` is
    ///   that erasure — its annotator drops the classdef, which is the
    ///   annotation a pointee-less raw pointer already carries here, so
    ///   the erased callers and the callers that pass an untyped `*mut u8`
    ///   straight through agree on one annotation.
    /// * **narrow** — `obj as *const RegisteredStruct`; see
    ///   `__cast_instance_intrinsic` below.
    ///
    /// Both are pointer-to-pointer only: a `Ref` source is required, since
    /// an `addr_usize as *const Struct` reinterpret is `cast_int_to_ptr`
    /// territory and must alias instead.
    pub(in crate::front::mir) fn ptr_cast_marker(
        &mut self,
        src_kind: Option<&ValueType>,
        src_root: Option<&str>,
        dest_ty: &TyRef,
        arg: &Variable,
    ) -> Option<(OpKind, Variable)> {
        if !matches!(src_kind, Some(ValueType::Ref(_)))
            || !matches!(
                tyref_to_value_type_with(
                    dest_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs
                ),
                ValueType::Ref(_)
            )
        {
            return None;
        }
        let (segments, result_ty) =
            if src_root.is_some() && tyref_is_raw_byte_ptr(dest_ty, self.llbc) {
                (
                    vec![crate::runtime_names::shims::CAST_ADDRESS.to_string()],
                    ValueType::Ref(None),
                )
            } else {
                let kind = self.instance_narrow_for_type(dest_ty, arg, src_root)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                return Some((kind, res));
            };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        Some((
            OpKind::Call {
                target: CallTarget::function_path(segments),
                args: crate::model::call_args(vec![arg.clone()]),
                result_ty,
            },
            res,
        ))
    }

    /// `*mut A as *mut B` when both owners are declared Raw. Both words are
    /// `ValueType::Int`, so [`Self::ptr_cast_marker`] declines and an alias
    /// would keep `SomePtr(A)` on a value whose fields are B's. The marker
    /// result stays in the int bank; the annotator replaces the pointee.
    pub(in crate::front::mir) fn raw_owner_retype(
        &mut self,
        src_kind: Option<&ValueType>,
        src_root: Option<&str>,
        dest_ty: &TyRef,
        arg: &Variable,
    ) -> Option<(OpKind, Variable)> {
        if !matches!(src_kind, Some(ValueType::Int | ValueType::Unsigned)) {
            return None;
        }
        let dest_root = raw_address_owner_root(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )?;
        if src_root == Some(dest_root.as_str()) {
            return None;
        }
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        Some((
            crate::model::cast_instance_call_result(dest_root, arg.clone(), ValueType::Int),
            res,
        ))
    }

    /// `*mut Raw as isize` stays in the int bank, but the annotation must
    /// become `SomeInteger`. Aliasing would leave `SomePtr` on a signed op.
    pub(in crate::front::mir) fn raw_ptr_to_signed(
        &mut self,
        mir_bb: usize,
        src_is_raw: bool,
        dest_ty: &TyRef,
        arg: &Variable,
    ) -> Option<(OpKind, Variable)> {
        if !src_is_raw || !self.dest_is_plain_signed_int(dest_ty) {
            return None;
        }
        let word = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.cast_ptr_to_int_src.insert(word.clone(), arg.clone());
        let word_op = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: [
                    "rpython",
                    "rtyper",
                    "lltypesystem",
                    "lltype",
                    "cast_ptr_to_int",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![arg.clone()]),
            result_ty: ValueType::Int,
        };
        Some(self.narrow_raw_ptr_word(mir_bb, word_op, word, false, dest_ty))
    }

    pub(in crate::front::mir) fn dest_is_plain_signed_int(&self, ty: &TyRef) -> bool {
        if tyref_is_raw_pointer(ty, self.llbc)
            || ty_is_raw_address(
                ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
            || raw_scalar_address_value_type(ty, self.llbc).is_some()
        {
            return false;
        }
        matches!(
            tyref_to_value_type_with(
                ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs
            ),
            ValueType::Int | ValueType::Int128
        )
    }

    /// `*mut Raw as usize`. The word stays in the int bank; `r_uint` changes
    /// the annotation from `SomePtr` to unsigned. Same call the `UnaryOp::Cast`
    /// arm emits.
    pub(in crate::front::mir) fn raw_ptr_to_unsigned(
        &mut self,
        mir_bb: usize,
        src_is_raw: bool,
        dest_ty: &TyRef,
        arg: &Variable,
    ) -> Option<(OpKind, Variable)> {
        if !src_is_raw || !self.dest_is_plain_unsigned(dest_ty) {
            return None;
        }
        let bb_id = self.emit_bb(mir_bb);
        let signed = push_cast_ptr_to_int(&mut self.graph, bb_id, arg.clone());
        self.cast_ptr_to_int_src.insert(signed.clone(), arg.clone());
        let word = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.raw_address_uint_src.insert(word.clone(), arg.clone());
        let word_op = OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![signed]),
            result_ty: ValueType::Unsigned,
        };
        Some(self.narrow_raw_ptr_word(mir_bb, word_op, word, true, dest_ty))
    }

    /// A pointer-to-integer cast yields the machine word. A narrower
    /// destination then takes the same mask or sign-extend as
    /// [`Self::materialize_int_cast`].
    pub(in crate::front::mir) fn narrow_raw_ptr_word(
        &mut self,
        mir_bb: usize,
        word_op: OpKind,
        word: Variable,
        src_unsigned: bool,
        dest_ty: &TyRef,
    ) -> (OpKind, Variable) {
        let src_bits = (crate::layout::target_word_size() * 8) as u32;
        let Some((dst_bits, dst_unsigned)) = self.literal_int_width(dest_ty) else {
            return (word_op, word);
        };
        // On a 32-bit target the address is a machine word. `i64` /
        // `u64` are longlong, so the 64-bit box identity does not apply.
        // `longlongmask` / `ulonglongmask` coerce through `inputargs`.
        if dst_bits > 64 {
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(word.clone()),
                kind: word_op,
            });
            // `cast_ptr_to_int` yields `Signed`. `ulonglonglongmask` on
            // `Signed` sign-extends, so a wasm32 address at or above
            // `0x80000000` becomes negative. `r_uint` first, matching the
            // 64-bit widen below.
            let address = if src_unsigned {
                word
            } else {
                let unsigned_word = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(unsigned_word.clone()),
                    kind: rarithmetic_mask("r_uint", word, ValueType::Unsigned),
                });
                unsigned_word
            };
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            if dst_unsigned {
                return (
                    rarithmetic_mask("ulonglonglongmask", address, ValueType::UInt128),
                    res,
                );
            }
            // Zero-extend to `UnsignedLongLongLong`, then retype to signed i128.
            let widened = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(widened.clone()),
                kind: rarithmetic_mask("ulonglonglongmask", address, ValueType::UInt128),
            });
            return (
                rarithmetic_mask("longlonglongmask", widened, ValueType::Int128),
                res,
            );
        }
        if dst_bits > src_bits {
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(word.clone()),
                kind: word_op,
            });
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            if dst_unsigned {
                return (
                    rarithmetic_mask("ulonglongmask", word, ValueType::Unsigned),
                    res,
                );
            }
            // The address is an unsigned bit pattern. `cast_ptr_to_int`
            // yields `Signed`, and `ulonglongmask` on `Signed` selects
            // `cast_int_to_ulonglong` (`ullong_from_int`), which
            // sign-extends. `r_uint` first so the widen is
            // `ullong_from_uint`, then retype to signed longlong.
            let masked = if src_unsigned {
                word
            } else {
                let unsigned_word = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(unsigned_word.clone()),
                    kind: rarithmetic_mask("r_uint", word, ValueType::Unsigned),
                });
                unsigned_word
            };
            let widened = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(widened.clone()),
                kind: rarithmetic_mask("ulonglongmask", masked, ValueType::Unsigned),
            });
            return (
                rarithmetic_mask("longlongmask", widened, ValueType::Int),
                res,
            );
        }
        let Some(action) = int_cast_action(src_bits, src_unsigned, dst_bits, dst_unsigned) else {
            return (word_op, word);
        };
        if !matches!(action, IntCastAction::And(_) | IntCastAction::SignExt(_)) {
            return (word_op, word);
        }
        let bb_id = self.emit_bb(mir_bb);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(word.clone()),
            kind: word_op,
        });
        match self.materialize_int_cast(
            mir_bb,
            (src_bits, src_unsigned),
            (dst_bits, dst_unsigned),
            word.clone(),
        ) {
            Some((Some(op), res)) => (op, res),
            _ => (
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: word.clone(),
                    result_ty: if src_unsigned {
                        ValueType::Unsigned
                    } else {
                        ValueType::Int
                    },
                },
                word,
            ),
        }
    }

    pub(in crate::front::mir) fn dest_is_plain_unsigned(&self, ty: &TyRef) -> bool {
        if tyref_is_raw_pointer(ty, self.llbc)
            || ty_is_raw_address(
                ty,
                self.llbc,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
            || raw_scalar_address_value_type(ty, self.llbc).is_some()
        {
            return false;
        }
        matches!(
            tyref_to_value_type_with(
                ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs
            ),
            ValueType::Unsigned | ValueType::UInt128
        )
    }

    /// rmodel.py::pairtype(Repr, VoidRepr).convert_from_to returns an actual
    /// Constant(None, Void). Keep its definition in the input graph until the
    /// flowspace adapter inlines it; an undefined Void variable is not a
    /// constant and framestate must not propagate it as an initialized local.
    pub(in crate::front::mir) fn emit_unit(&mut self, bb_id: BlockId) -> Variable {
        let unit = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(unit.clone()),
            kind: OpKind::ConstNone,
        });
        unit
    }

    /// Keep a dyn-call / ICT-method receiver as a pointer-valued word.
    ///
    /// `resolve_operand` of `&ZST` yields `emit_unit` (Void). rustc still
    /// passes that pointer; RPython method `indirect_call` always has self
    /// (`FunctionReprBase.call`). `same_as` cannot recast Void: the
    /// flowspace adapter discards `result_ty`, the annotator preserves the
    /// operand annotation, and `rtype_same_as` emits the operand's original
    /// Void representation.
    ///
    /// Dyn call: the receiver is the fat pointer's DATA word, read from
    /// the same fat field the vtable word (`FatMeta`) is read from.
    /// Direct call to a ZST impl: a non-null aligned pointer constant
    /// (`ConstRefAddr`). Kind is Ref for the whole family: RPython
    /// `class DictStrategy(object)` and every subclass is a GC instance,
    /// so `history.getkind` of that Ptr is `'ref'`. rustc mixes ZST
    /// unit-struct impls with GC `ModuleDictStrategy`; one
    /// `indirect_call` has one FUNC type, so the family is one kind.
    /// A GC strategy stays a GC root (`walk_bh_regs` / `drag_out_root`).
    /// A ZST dummy in `registers_r` is a non-nursery constant
    /// (`walk_bh_regs` passes those through).
    pub(in crate::front::mir) fn keep_dyn_call_receiver(
        &mut self,
        mir_bb: usize,
        args: &mut [Variable],
    ) {
        let Some(recv) = args.first_mut() else {
            return;
        };
        let bb_id = self.emit_bb(mir_bb);
        if let Some(data) = self.dyn_receiver_data_word(bb_id, recv) {
            *recv = data;
            return;
        }
        if crate::model::FunctionGraph::concretetype_of(recv) == crate::model::ConcreteType::Void {
            *recv = self.emit_zst_self_address(bb_id);
        }
    }

    /// Fat-pointer DATA word of `recv`, or `None` when `recv` is not a
    /// fat field (direct ZST `&self` uses [`Self::emit_zst_self_address`]).
    pub(in crate::front::mir) fn dyn_receiver_data_word(
        &mut self,
        bb_id: crate::model::BlockId,
        recv: &Variable,
    ) -> Option<Variable> {
        emit_dyn_receiver_data_word(&mut self.graph, bb_id, recv)
    }

    /// Direct call to a ZST impl method: rustc `&ZST` is any non-null
    /// address aligned to `align_of::<T>()`. The callee never
    /// dereferences it. Word alignment is valid for every unit-struct
    /// strategy (`EmptyDictStrategy` is align-1). Emitted as
    /// `ConstRefAddr` so the first inputarg kind matches the GC impls
    /// in the same `indirect_call` family (`history.getkind` of a Gc
    /// Ptr is `'ref'`).
    pub(in crate::front::mir) fn emit_zst_self_address(
        &mut self,
        bb_id: crate::model::BlockId,
    ) -> Variable {
        let var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(var.clone()),
            kind: OpKind::ConstRefAddr(std::mem::size_of::<usize>() as i64),
        });
        var
    }

    /// Fold a `TraitConst` whose trait ref is one selected impl.
    ///
    /// `FunctionDesc.cachedgraph` builds one graph per instantiation and
    /// the copy substitutes the `Clause`, so the const names that impl.
    /// The impl's `consts[index]` is the `NamedConst` global;
    /// `const_eval_init_body` (and the `size_of` lane beside it) folds
    /// the initializer to the literal `history.getkind` banks as `int`.
    /// `CallControl.getcalldescr` then sees the same kind as the `usize`
    /// parameter. A `Clause` that still names no impl returns `None`:
    /// one shared template has no single value, and `decode_constant`
    /// keeps the opaque-string path. A zero-arg `__trait_const` call is
    /// not a registered graph, so `translate_op` would Skip the copy.
    pub(in crate::front::mir) fn resolved_trait_const_op(
        &self,
        value: &serde_json::Value,
    ) -> Option<OpKind> {
        let kind = self.llbc.const_expr_kind(value)?;
        let pair = kind.as_object()?.get("TraitConst")?.as_array()?;
        let trait_ref = pair.first()?;
        let index = usize::try_from(pair.get(1)?.as_u64()?).ok()?;
        let impl_id = traitref_impl_id(trait_ref, self.llbc, 0)?;
        let global_id = self
            .llbc
            .trait_impl_by_id(impl_id)?
            .get("consts")?
            .as_array()?
            .get(index)?
            .get("id")?
            .as_u64()?;
        let op = self
            .const_eval_global(global_id)
            .or_else(|| self.fold_size_const_global(global_id))
            .or_else(|| self.fold_transparent_int_const_global(global_id))?;
        match op {
            OpKind::ConstInt(_)
            | OpKind::ConstFnAddr { .. }
            | OpKind::ConstUInt(_)
            | OpKind::ConstInt128(_)
            | OpKind::ConstUInt128(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstFloat(_)
            | OpKind::ConstSingleFloat(_) => Some(op),
            _ => None,
        }
    }

    /// Decode a Charon `Operand::Const` value and emit the matching
    /// `OpKind::Const*` (or synthetic `Call` for non-primitive
    /// constants) operation on the current block, returning the fresh
    /// Variable that holds it.
    pub(in crate::front::mir) fn emit_constant(
        &mut self,
        mir_bb: usize,
        value: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        // A selected impl's associated const is that impl's literal.
        // `history.getkind` of the folded word matches the parameter;
        // a string sentinel would be `Ref` and `CallControl.getcalldescr`
        // would reject the call.
        if let Some(op) = self.resolved_trait_const_op(value) {
            let var = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(var.clone()),
                kind: op,
            });
            return Ok(var);
        }
        let decoded = decode_constant(self.llbc, value)?;
        let bb_id = self.emit_bb(mir_bb);
        Ok(self.emit_decoded_const(bb_id, decoded))
    }

    /// Emit `decoded` onto `bb_id`. A Unicode scalar is
    /// `bookkeeper.py` `immutablevalue` `SomeUnicodeCodePoint`: the
    /// integer code point plus `unichr`.
    pub(in crate::front::mir) fn emit_decoded_const(
        &mut self,
        bb_id: crate::model::BlockId,
        decoded: DecodedConst,
    ) -> Variable {
        if let DecodedConst::UniChar(code) = decoded {
            let code_v = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(code_v.clone()),
                kind: OpKind::ConstInt(i64::from(code)),
            });
            let var = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(var.clone()),
                kind: unichr_call(code_v),
            });
            return var;
        }
        let op = match decoded {
            DecodedConst::UniChar(_) => unreachable!("handled above"),
            DecodedConst::Int(n) => OpKind::ConstInt(n),
            DecodedConst::UInt(n) => OpKind::ConstUInt(n),
            DecodedConst::Int128(n) => OpKind::ConstInt128(n),
            DecodedConst::UInt128(n) => OpKind::ConstUInt128(n),
            DecodedConst::Bool(b) => OpKind::ConstBool(b),
            DecodedConst::Float(bits) => OpKind::ConstFloat(bits),
            DecodedConst::SingleFloat(bits) => OpKind::ConstSingleFloat(bits),
            // String / byte-string constants — no
            // ConstStr opkind exists; synthesise a 0-arg `Call` whose
            // path encodes the literal text so the IR stays stable.
            DecodedConst::Str(s) => OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), s],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![]),
                // A `&str` / `&[u8]` literal lowers to `Ptr(STR)` (getkind
                // `r`), so the synthetic call's result kind is a Ref, not an
                // Int.  The `__str_const` path is never registered: on the
                // trace pipeline the call residualises, and the flowspace
                // adapter pre-folds it to the upstream `Constant('text')`
                // shape (`flowspace_adapter.rs::is_str_const_define`).
                result_ty: ValueType::Ref(None),
            },
            // A function item used as a *value* rather than called; its value
            // is the function's address.  The define takes a synthetic head
            // for the same reason the `Str` arm above does: the path never
            // matches a registered graph, so the define is not mistaken for a
            // real 0-arg call to the function it names.  Both of
            // `getcalldescr`'s checks (`call.py`) run off the resolved callee
            // graph, and both reject this shape without the head — the
            // argument-kind comparison against the define's empty list
            // (`cycle_reduce_method`, one `Ref` parameter), and
            // `RESULT == FUNC.RESULT` for a 0-parameter callee carrying a
            // result token (`w_dict_new`, `dont_look_inside`, threaded through
            // `Option::unwrap_or_else`).  The callee's own segments stay in
            // the tail so the referenced function is still recoverable.
            //
            // Upstream materialises the address as `Constant(funcptr)` of
            // lltype `Ptr(FuncType)` (`rtyper.getcallable`, and
            // `sub_helper_funcptr_constant` for the sub-helper twins).
            // `FuncType._gckind` is `raw`, so `getkind` maps that pointer to
            // `int`, not to `ref` — the `Int` slot here IS that mapping rather
            // than a departure from it, and re-stamping it `Ref` would be the
            // deviation.  majit materialises a funcptr as its integer address
            // everywhere else as well (`jtransform.rs direct_funcptr_value`
            // emits `ConstFnAddr`, which the assembler encodes through the
            // `'i'` argcode), and the flowspace fold gives the define a
            // `Signed` legacy slot to match.
            DecodedConst::FnPath(segments) => {
                let mut synthetic = Vec::with_capacity(segments.len() + 1);
                synthetic.push(crate::model::FN_CONST_HEAD.to_string());
                synthetic.extend(segments);
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: synthetic,
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Int,
                }
            }
        };
        let var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(var.clone()),
            kind: op,
        });
        var
    }

    /// A safe `Index` / `IndexMut` / `.len()` of a `Vec` or slice takes the
    /// field's address (`&frame.items`). For a declared virtualizable array
    /// that address is the array the codewriter pairs with `getarrayitem`:
    /// clear the address mark so the field read enters `vable_array_vars`.
    /// An address taken for any other consumer keeps the mark.
    ///
    /// Returns whether `base` is that declared array field.
    pub(in crate::front::mir) fn release_declared_vable_array_address(
        &mut self,
        base: &Variable,
    ) -> bool {
        release_declared_vable_array_address(&mut self.graph, base)
    }

    #[allow(dead_code)] // virtualizable.py declared array field
    pub(in crate::front::mir) fn var_is_declared_vable_array(&self, var: &Variable) -> bool {
        self.graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                op.result.as_ref() == Some(var)
                    && matches!(
                        &op.kind,
                        OpKind::FieldRead { field, .. }
                            if crate::virtualizable_decl::is_declared_array_field(
                                field.owner_root.as_deref(),
                                &field.name,
                            )
                    )
            })
        })
    }

    /// `v[i]` / `v.len()` on a `Vec<T>`, wherever the `Vec` was reached:
    /// an inline field, a `&mut Vec` local, or a pointer already loaded
    /// from `Box<Vec<_>>`. An inline field is retargeted in place. Every
    /// other receiver is a getfield of the buffer or length word from that
    /// pointer (`rlist.ll_items` / `ll_length`).
    pub(in crate::front::mir) fn retarget_vec_part(
        &mut self,
        bb_id: BlockId,
        vec_var: &Variable,
        part: crate::model::VecFieldPart,
    ) -> Variable {
        retarget_vec_operand(&mut self.graph, bb_id, vec_var, part)
    }

    /// Data or length word of a fat `Box<[T]>` field read. `None` when
    /// `var` is not one of those reads (or a one-hop copy of one).
    pub(in crate::front::mir) fn fat_component(
        &mut self,
        bb_id: BlockId,
        var: &Variable,
        part: crate::model::VecFieldPart,
    ) -> Option<Variable> {
        if self.fat_box_vars.is_empty() {
            return None;
        }
        let marked = self.fat_box_vars.clone();
        retarget_fat_operand(&mut self.graph, &marked, bb_id, var, part)
    }

    /// Window `{data, len}` words of a fat box. Data is `CCHARP`.
    pub(in crate::front::mir) fn window_fat_component(
        &mut self,
        bb_id: BlockId,
        var: &Variable,
        part: crate::model::VecFieldPart,
    ) -> Option<Variable> {
        if self.fat_box_vars.is_empty() {
            return None;
        }
        let marked = self.fat_box_vars.clone();
        retarget_window_fat_operand(&mut self.graph, &marked, bb_id, var, part)
    }

    pub(in crate::front::mir) fn is_fat_box_var(&self, var: &Variable) -> bool {
        let peeled = peel_copy_var(&self.graph, var);
        self.fat_box_vars
            .iter()
            .any(|marked| marked == var || marked == &peeled)
    }

    /// True when `var` is a GC pointer / string, so a window pair must
    /// `cast_ptr_to_int` it into `CCHARP`. An `Input` of `Unsigned` is
    /// already the pair data word.
    pub(in crate::front::mir) fn var_is_gc_ref_value(&self, var: &Variable) -> bool {
        if crate::model::FunctionGraph::concretetype_of(var) == crate::model::ConcreteType::GcRef {
            return true;
        }
        let Some((block_id, idx)) = resolve_to_producer_op(&self.graph, var) else {
            return false;
        };
        let Some(op) = self
            .graph
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .and_then(|block| block.operations.get(idx))
        else {
            return false;
        };
        matches!(
            &op.kind,
            OpKind::FieldRead {
                ty: ValueType::Ref(_) | ValueType::Str,
                ..
            } | OpKind::Call {
                result_ty: ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder,
                ..
            } | OpKind::Input {
                ty: ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder,
                ..
            }
        )
    }
    /// Inner place of `&raw const *p` / `&raw mut *p` when that deref
    /// would `RawLoad` a primitive.
    ///
    /// The rvalue's value is the address. A pointer recorded on
    /// `atomic_ref_place` is a scalar alias, and resolving its deref
    /// already yields that scalar, so this returns `None`.
    pub(in crate::front::mir) fn raw_primitive_reborrow_inner(
        &self,
        place: &Place,
    ) -> Option<Place> {
        let PlaceKind::Projection(inner, ProjectionElem::Atom(name)) = &place.kind else {
            return None;
        };
        if name != "Deref" {
            return None;
        }
        if !tyref_is_raw_pointer(&inner.ty, self.llbc)
            || !tyref_is_primitive_scalar(&place.ty, self.llbc)
            || self.raw_word_descr(&place.ty).is_none()
        {
            return None;
        }
        if let PlaceKind::Local(local) = inner.kind
            && self.atomic_ref_place.contains_key(&(local as usize))
        {
            return None;
        }
        Some(clone_place(inner))
    }

    /// Whether `&<place>` / `&raw [mut] <place>` takes the address of a
    /// place, as opposed to reading the value one holds.
    ///
    /// `&*(*p).f` — a projection whose outermost element is `Deref` — yields
    /// the value the pointer in `(*p).f` holds, so the underlying read stays
    /// a value read and keeps its lowering.  `&(*p).f`, Field-last, is the
    /// real place-address and stays marked.
    ///
    /// The `Deref` test is spelled as `resolve_place` and
    /// `emit_projection_write` already spell it, applied one level out.
    /// Byte offset of `(*p).f` inside `*p` when `f` is a by-value struct
    /// stored inline, so `&(*p).f` is an interior address rather than a word
    /// the container holds.
    ///
    /// `jtransform.py rewrite_op_getsubstruct` lowers that access to
    /// `int_add(p, offsetof(STRUCT, f))`.  The generic `Rvalue::Ref` arm
    /// aliases a borrow to its referent's value, which for an inline struct
    /// reads the substructure's first word and hands it on as the address.
    /// The offset is Charon's layout for the build target, the same one the
    /// descr layer records.  `None` leaves the place to the generic arm: a
    /// non-deref container (a local aggregate), an enum payload, a field that
    /// is itself a pointer or scalar, a `core`/`alloc`/`std` type the front
    /// models by value, or a container without a resolved layout.
    pub(in crate::front::mir) fn inline_substruct_field_offset(
        &self,
        place: &Place,
    ) -> Option<u64> {
        let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
            return None;
        };
        let PlaceKind::Projection(_, ProjectionElem::Atom(deref)) = &inner.kind else {
            return None;
        };
        if deref != "Deref" {
            return None;
        }
        let payload = elem.as_object()?.get("Field")?;
        // Named `{"Adt": ...}` only. `resolve_adt_field` does not decode
        // that container, so a nonzero offset would otherwise collapse to
        // the parent pointer (`resolve_place`'s unmatched projection).
        // An index pair `[null, i]` / `[variant, i]` is a `FieldRead`.
        // `mark_place_address_of` sets `taken_by_address`, and
        // `rewrite_op_getsubstruct` turns that read into the interior
        // address. Returning the parent variable would give the field's
        // consumer the container's type; casting that GC pointer onto the
        // raw field struct is not `getsubstruct`.
        self.inline_substruct_named_adt_offset(place, payload)
    }

    /// `{"Adt": ...}` field container. `None` when the payload is an index
    /// pair or the field is not an inline struct address.
    pub(in crate::front::mir) fn inline_substruct_named_adt_offset(
        &self,
        place: &Place,
        payload: &serde_json::Value,
    ) -> Option<u64> {
        let payload = payload.as_array()?;
        let [container, field_idx] = payload.as_slice() else {
            return None;
        };
        let adt = container.as_object()?.get("Adt")?.as_array()?;
        if adt.get(1).is_some_and(|variant| !variant.is_null()) {
            return None;
        }
        let head = adt.first()?;
        let owner_id = match head.as_u64() {
            Some(id) => id,
            None => type_decl_ref_adt_id(head.as_object()?)?,
        };
        let owner = self.llbc.type_by_id(owner_id)?;
        if !matches!(owner.kind, TypeDeclKind::Struct(_)) {
            return None;
        }
        if !self.inline_substruct_field_is_address(&place.ty) {
            return None;
        }
        let target = std::env::var("TARGET").unwrap_or_default();
        let offset = owner
            .layout_for_target(self.llbc, &target)?
            .struct_field_offset(field_idx.as_u64()? as usize)?;
        Some(offset)
    }

    /// The borrowed field is a by-value struct, so its address is an
    /// interior pointer rather than a word the container holds.
    pub(in crate::front::mir) fn inline_substruct_field_is_address(
        &self,
        field_ty: &TyRef,
    ) -> bool {
        let Some(node) = tyref_node(field_ty, self.llbc) else {
            return false;
        };
        let decl_id = adt_node_def_id(node)
            .or_else(|| strip_ty_wrappers(node, self.llbc).and_then(adt_node_def_id));
        let Some(field_decl) = decl_id.and_then(|id| self.llbc.type_by_id(id)) else {
            return false;
        };
        let TypeDeclKind::Struct(fields) = &field_decl.kind else {
            return false;
        };
        if fields.is_empty() || field_decl.is_repr_transparent() {
            return false;
        }
        let field_path = field_decl.item_meta.name_path();
        !["core::", "alloc::", "std::"]
            .iter()
            .any(|prefix| field_path.starts_with(prefix))
    }

    /// Lower `&(*p).f` / `&raw (*p).f` for an inline struct field as
    /// `rewrite_op_getsubstruct` does.  Offset zero is `p` itself.  A nonzero
    /// offset takes the same deferred refusal `rewrite_op_getfield` gives a
    /// nonzero `getsubstruct`: an interior address cannot enter the Ref bank,
    /// so the graph aborts the trace when it reaches the access.
    pub(in crate::front::mir) fn lower_inline_substruct_address(
        &mut self,
        mir_bb: usize,
        place: &Place,
    ) -> Result<Option<Variable>, LowerError> {
        let Some(offset) = self.inline_substruct_field_offset(place) else {
            return Ok(None);
        };
        let PlaceKind::Projection(inner, _) = &place.kind else {
            unreachable!("inline_substruct_field_offset matched a projection");
        };
        let base = self.resolve_place(mir_bb, clone_place(inner))?;
        if offset != 0 {
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::Abort {
                    kind: crate::model::UnknownKind::UnsupportedExpr {
                        variant: crate::model::UnsupportedExprKind::RawAddr,
                    },
                },
            });
        }
        Ok(Some(base))
    }

    pub(in crate::front::mir) fn place_ref_is_address_of(place: &Place) -> bool {
        match &place.kind {
            PlaceKind::Projection(_, ProjectionElem::Atom(s)) if s == "Deref" => false,
            PlaceKind::Projection(..) => true,
            _ => false,
        }
    }

    /// `&T` of a transparent scalar is the word, so the field read stays a
    /// value. Any other projection borrow is the place address.
    ///
    /// An erased `OwnerRootGuard` is that one reference. Borrowing the
    /// option to match on it reads the word `new` stored; marking the
    /// read as an address would hand the consumer the slot instead.
    pub(in crate::front::mir) fn place_borrow_takes_address(&self, place: &Place) -> bool {
        if owner_root_guard::tyref_is_guard(&place.ty, self.llbc)
            || owner_root_guard::tyref_is_option_of_guard(&place.ty, self.llbc)
        {
            return false;
        }
        Self::place_ref_is_address_of(place)
            && tyref_transparent_inner_value_type(
                &place.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
            .is_none()
    }

    /// Record that the `Variable` just resolved for a place is the
    /// **address** of a field, not its value.
    ///
    /// The `Ref` / `RawPtr` arms alias the reference to its referent, so
    /// `&raw mut (*p).f` leaves an ordinary `FieldRead` behind.  A
    /// virtualizable field read is not kept as a value — the codewriter
    /// records the array in `vable_array_vars` and drops the op — so the
    /// address's consumer would be handed an undefined operand.  Flagging
    /// the descriptor lets the codewriter decline the virtualizable path
    /// for exactly this projection.
    ///
    /// Only the **outermost** projection is marked: it is the one whose
    /// result is the Variable the address stands for.  An inner step of
    /// `&(*p).a.b` is a genuine value read of `a` and keeps its lowering,
    /// so a nested read of a virtualizable field is not collaterally
    /// suppressed.
    ///
    /// Both guards are load-bearing, because a local's Variable *is* the
    /// Variable of whatever op produced it:
    ///
    /// - `projection` — `&mut x` on a bare local resolves to that local's
    ///   Variable with nothing emitted.  Matching on the last op's result
    ///   alone would then mark the op that produced `x`, which may be an
    ///   unrelated earlier field read.  Autoref makes this the common
    ///   case, not a corner: `self.method()` on `&mut self` is a
    ///   `Rvalue::Ref` over a local at every call site.
    /// - `before` — a projection that emits nothing (a bare `Deref`, or a
    ///   field the front resolves without a read) leaves the preceding op
    ///   as `last`, with the same false-marking risk.
    ///
    /// A false mark here is not inert: on a virtualizable field it would
    /// silently disable the protocol for a genuine access, which is the
    /// failure the `Option` default on `base_is_deref` exists to prevent.
    pub(in crate::front::mir) fn mark_place_address_of(
        &mut self,
        mir_bb: usize,
        projection: bool,
        before: usize,
        v: &Variable,
    ) {
        if !projection {
            return;
        }
        let bb_id = self.emit_bb(mir_bb);
        let block = self.graph.block_mut(bb_id);
        if block.operations.len() <= before {
            return;
        }
        if let Some(op) = block.operations.last_mut()
            && op.result.as_ref() == Some(v)
            && let OpKind::FieldRead { field, .. } = &mut op.kind
        {
            field.taken_by_address = true;
        }
    }

    /// Charon `ProjectionElem::Offset`: `ptr.offset(n)` is `&raw (*ptr).offset(bytes)`
    /// with `bytes` already `n * size_of::<T>()` (`offset_projection`).
    /// The old `BinOp::Offset` result was that address; emit unscaled
    /// `direct_ptradd` (Char item, byte shift) of the pointer plus `bytes`.
    pub(in crate::front::mir) fn resolve_offset_place(
        &mut self,
        mir_bb: usize,
        inner: Place,
        offset: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        let bytes_v = offset.get("bytes").ok_or_else(|| {
            LowerError::Schema(format!(
                "bb{mir_bb}: Offset projection missing bytes: {offset}"
            ))
        })?;
        let bytes_op: Operand = serde_json::from_value(bytes_v.clone())
            .map_err(|e| LowerError::Schema(format!("bb{mir_bb}: Offset bytes decode: {e}")))?;
        let bytes = self.resolve_operand(mir_bb, bytes_op)?;
        let ptr = match inner.kind {
            PlaceKind::Projection(ptr_place, ProjectionElem::Atom(ref s)) if s == "Deref" => {
                self.resolve_place(mir_bb, *ptr_place)?
            }
            _ => self.resolve_place(mir_bb, inner)?,
        };
        let bb_id = self.emit_bb(mir_bb);
        let entry_ty = ptr.concretetype();
        let result =
            push_direct_ptradd(&mut self.graph, bb_id, ptr, bytes, &item_lltype_for_size(1));
        Ok(restore_offset_pointer_type(
            &mut self.graph,
            bb_id,
            entry_ty,
            result,
        ))
    }

    pub(in crate::front::mir) fn resolve_place(
        &mut self,
        mir_bb: usize,
        place: Place,
    ) -> Result<Variable, LowerError> {
        self.resolve_place_as(mir_bb, place, false)
    }

    /// [`Self::resolve_place`].  A projection whose value is a pair slice
    /// is two words, so it resolves only for `pair_field`, the pointer-word
    /// read [`Self::pair_field_parts`] builds on.
    pub(in crate::front::mir) fn resolve_place_as(
        &mut self,
        mir_bb: usize,
        place: Place,
        pair_field: bool,
    ) -> Result<Variable, LowerError> {
        let place_ty = clone_tyref(&place.ty);
        // A pair slice's `PtrMetadata` is its length word. A window's is
        // the same length word (`FatLen`).
        if let PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) = &place.kind
            && atom == "PtrMetadata"
            && (self.pair_slice_kind(&inner.ty).is_some()
                || tyref_is_u8_window(&inner.ty, self.llbc))
            && let Some((_, len)) = self.pair_place_parts(mir_bb, inner)?
        {
            return Ok(len);
        }
        if !pair_field
            && matches!(place.kind, PlaceKind::Projection(..))
            && tyref_pair_field_kind(&place_ty, self.llbc).is_some()
        {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice projection read as one value"
            )));
        }
        // The `Some` payload of a pair-slice iterator's `next` is the item at
        // the index the range iterator yielded. The option local holds that
        // iterator's items and the `next` result.
        if let PlaceKind::Projection(inner, ProjectionElem::Tagged(v)) = &place.kind
            && v.get("Field").is_some()
            && let PlaceKind::Local(opt) = inner.kind
            && self
                .slice_next_iter
                .get(opt as usize)
                .and_then(|iter| *iter)
                .is_some()
        {
            return self.pair_iter_item(mir_bb, &place, opt as usize, &place_ty);
        }
        // The `Some` payload of a pair-slice `get` is the item at the address
        // the `Option` holds.
        if let PlaceKind::Projection(inner, ProjectionElem::Tagged(v)) = &place.kind
            && v.get("Field").is_some()
            && let PlaceKind::Local(opt) = inner.kind
            && matches!(
                self.local_var
                    .get(opt as usize)
                    .and_then(|slot| slot.as_ref()),
                Some(LocalValue::ItemAddr(_))
            )
            && let Some((kind, by_ref)) = self.option_pair_item(opt as usize)
        {
            let addr = self.item_addr_var(mir_bb, opt as usize)?;
            let item_ty = if by_ref {
                tyref_ref_pointee_value_type(
                    &place_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: slice item reference payload is not a reference"
                    ))
                })?
            } else {
                tyref_to_value_type_with(
                    &place_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            };
            let item_ty = self.with_item_class_root(item_ty, &place_ty);
            let zero = self.emit_const_uint(mir_bb, 0);
            let helper_ty = Self::pair_helper_item_ty(kind);
            let item = self.emit_path_call(
                mir_bb,
                majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::GetItem, kind),
                vec![addr, zero],
                helper_ty.clone(),
            );
            return Ok(self.retype_word(mir_bb, item, &helper_ty, &item_ty));
        }
        // `(*f)[i]` over a pair slice an aggregate field holds reads through
        // the field's item pointer.
        if let PlaceKind::Projection(inner, ProjectionElem::Tagged(v)) = &place.kind
            && let Some(index_payload) = v.get("Index")
            && let PlaceKind::Projection(slice, ProjectionElem::Atom(atom)) = &inner.kind
            && atom == "Deref"
            && !matches!(slice.kind, PlaceKind::Local(_))
            && let Some(kind) = self.pair_slice_kind(&slice.ty)
            && let Some((ptr, len)) = self.pair_place_parts(mir_bb, slice)?
        {
            let offset = self.index_offset_var(mir_bb, index_payload)?;
            let index = if index_payload
                .get("from_end")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                self.emit_index_from_end(mir_bb, len, offset)
            } else {
                offset
            };
            let item_ty = self.with_item_class_root(
                tyref_to_value_type_with(
                    &place_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
                &place_ty,
            );
            let helper_ty = Self::pair_helper_item_ty(kind);
            let item = self.emit_path_call(
                mir_bb,
                majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::GetItem, kind),
                vec![ptr, index],
                helper_ty.clone(),
            );
            return Ok(self.retype_word(mir_bb, item, &helper_ty, &item_ty));
        }
        // `(*s)[i]` over a pair slice, or `a[i]` over a raw-buffer array,
        // reads through the item pointer.
        if let PlaceKind::Projection(inner, ProjectionElem::Tagged(v)) = &place.kind
            && let Some(index_payload) = v.get("Index")
            && let Some(seq) = self.item_seq_of_place(inner)
        {
            let item_ty = self.with_item_class_root(
                tyref_to_value_type_with(
                    &place_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
                &place_ty,
            );
            return self.pair_getitem(mir_bb, seq, &index_payload.clone(), &item_ty);
        }
        match place.kind {
            // A pair slice is two words; only the pair-aware paths read it.
            PlaceKind::Local(i)
                if matches!(
                    self.local_var
                        .get(i as usize)
                        .and_then(|slot| slot.as_ref()),
                    Some(LocalValue::Pair(_, _))
                ) =>
            {
                // A window's one-word consumer sees the `CCHARP` data
                // word; the length stays in the shadow (`link_word`).
                // Word-sized pair slices still refuse a one-word read.
                let local = i as usize;
                if self
                    .body
                    .locals
                    .locals
                    .get(local)
                    .is_some_and(|loc| tyref_is_u8_window(&loc.ty, self.llbc))
                {
                    Ok(self.local_var[local]
                        .as_ref()
                        .expect("pair local is bound")
                        .link_word())
                } else {
                    self.local_var[local]
                        .as_ref()
                        .unwrap()
                        .one()
                        .map_err(|err| match err {
                            LowerError::Unsupported(msg) => {
                                LowerError::Unsupported(format!("bb{mir_bb}: {msg}"))
                            }
                            other => other,
                        })
                }
            }
            // A reference-item `Option<&T>` is one word: the item itself,
            // since a reference aliases its referent, or the null pointer
            // for `None`.
            PlaceKind::Local(i)
                if matches!(
                    self.local_var
                        .get(i as usize)
                        .and_then(|slot| slot.as_ref()),
                    Some(LocalValue::ItemAddr(_))
                ) && self
                    .option_pair_item(i as usize)
                    .is_some_and(|(kind, _)| kind == majit_ir::rvec::VecItemKind::Ref)
                    && self.tyref_is_niche_option_ptr(&place_ty) =>
            {
                let addr = self.item_addr_var(mir_bb, i as usize)?;
                let value_ty = self.with_item_class_root(
                    tyref_to_value_type_with(
                        &place_ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ),
                    &place_ty,
                );
                let helper_ty = Self::pair_helper_item_ty(majit_ir::rvec::VecItemKind::Ref);
                let null = self.push_niche_null_ptr(mir_bb, &place_ty);
                let null = self.retype_word(mir_bb, null, &value_ty, &helper_ty);
                let item = self.emit_path_call(
                    mir_bb,
                    majit_ir::rvec::slice_helper_path(
                        majit_ir::rvec::SliceOp::LoadOr,
                        majit_ir::rvec::VecItemKind::Ref,
                    ),
                    vec![addr, null],
                    helper_ty.clone(),
                );
                Ok(self.retype_word(mir_bb, item, &helper_ty, &value_ty))
            }
            PlaceKind::Local(i)
                if matches!(
                    self.local_var
                        .get(i as usize)
                        .and_then(|slot| slot.as_ref()),
                    Some(LocalValue::ItemAddr(_))
                ) =>
            {
                Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice item reference local {i} read as one value"
                )))
            }
            PlaceKind::Local(i)
                if self
                    .const_struct_fields
                    .get(i as usize)
                    .and_then(|slot| slot.as_ref())
                    .is_some() =>
            {
                Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: const struct local {i} read as one word"
                )))
            }
            PlaceKind::Local(i) => self
                .local_var
                .get(i as usize)
                .and_then(|slot| slot.as_ref())
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: read of MIR local {i} before any Assign — \
                         uninitialised local, not yet supported"
                    ))
                })?
                .one(),
            PlaceKind::Projection(inner, elem) => {
                if let Some(var) =
                    self.folded_const_struct_field(mir_bb, &inner, &elem, &place_ty)?
                {
                    return Ok(var);
                }
                // Adt-container `Field` projections emit a typed
                // `OpKind::FieldRead` carrying the field name and
                // `owner_root` so downstream consumers (codewriter
                // inlining + annotator GetAttr dispatch on
                // cross-procedural callers) get a resolvable
                // field/owner_root shape.
                //
                // Tuple-container `Field` projections split three ways.
                // A local bound by a positional `Rvalue::Aggregate`
                // (`positional_aggregate_locals`) carries a synthetic
                // ctor base with a `__pos_<N>` `FieldWrite` chain, so
                // its `.N` reads emit a symmetric `FieldRead __pos_<N>`.
                // A genuine Ref tuple (`__pos_<N>` block below) likewise
                // emits a typed `FieldRead`.  The `straight_line_add` /
                // AddChecked `(value, bool)` shape is the exception: it
                // lowers to a scalar `BinOp` (not an Aggregate), so its
                // `.0` collapses to the base Variable while the paired
                // `.1` Assert is dropped in `lower_statement` (a live
                // `.1` read fails loud — the overflow bit is unmodeled).
                //
                // Atom projections (`Deref` and others) still
                // collapse. `Deref` of a reference is a no-op: the
                // referent is already the value, and any other Atom
                // variant has no typed analogue today. A cell
                // parameter is the exception: `*p` reads the cell's
                // `value` field. `Deref` of a raw pointer to a
                // primitive is `raw_load` (below), not that alias.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some((owner_root, field_name, field_ty, owner_id, host_index)) =
                        self.resolve_adt_field(&inner.ty, field_payload)
                {
                    // Projecting the sized field of a `repr(transparent)`
                    // wrapper is the identity on its low-level value, including
                    // through a borrow: the borrow carries no representation of
                    // its own, the same way [`Self::tyref_is_borrowed_fieldless_enum`]
                    // treats a fieldless enum. A zero-sized marker in the same
                    // struct still has no bytes. Emitting a FieldRead would
                    // try to dereference that word as an aggregate base.
                    //
                    // `tyref_transparent_nonzst_field` is the defining-crate
                    // form (named field, `is_repr_transparent`). An Opaque
                    // or published copy has no `repr` and a named field
                    // (`__pos_0` is only the missing-row fallback), so the
                    // linked scalar bank [`tyref_transparent_inner_value_type`]
                    // is the same question: the non-ZST field is the word.
                    if let Some(inner_vt) = tyref_transparent_inner_value_type(
                        &inner.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ) {
                        if field_is_zst(&field_ty, self.llbc) {
                            return Ok(self.emit_unit(self.emit_bb(mir_bb)));
                        }
                        if let Some((_, name)) =
                            tyref_transparent_nonzst_field(&inner.ty, self.llbc)
                            && field_name != name
                        {
                            return Ok(self.emit_unit(self.emit_bb(mir_bb)));
                        }
                        // A linked-bank wrapper has no field list here
                        // (`tyref_transparent_nonzst_field` is None). A
                        // field that is not the scalar word is a zero-sized
                        // marker; loading it would `getfield_raw_r` the
                        // word as an aggregate.
                        let field_vt = tyref_to_value_type_with(
                            &field_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        );
                        if value_type_is_transparent_scalar_word(&inner_vt)
                            && !value_type_is_transparent_scalar_word(&field_vt)
                        {
                            return Ok(self.emit_unit(self.emit_bb(mir_bb)));
                        }
                        let wrapper_ty = clone_tyref(&inner.ty);
                        let base = self.resolve_place(mir_bb, *inner)?;
                        // The carrier word stays one pointer. Projecting
                        // its inner `PyObjectRef` has to name that class,
                        // or the handle and the object merge.
                        return Ok(self.cast_off_error_carrier(mir_bb, &wrapper_ty, base));
                    }
                    // A one-word niche `Option` is represented by its payload
                    // pointer, so the
                    // `Some` payload IS the base pointer — reading `Some.__pos_0`
                    // is the identity on it, not a field off a two-word
                    // aggregate (there is none, and its `__pos_0` attr has no
                    // rtype clsfield).  Alias the base value with no field read,
                    // mirroring the fieldless-enum discriminant collapse and the
                    // `Discriminant` niche fold above.
                    if field_name == "__pos_0" && self.tyref_is_niche_option_ptr(&inner.ty) {
                        return self.resolve_place(mir_bb, *inner);
                    }
                    // A niche u8-window `Option` (`Option<&[u8]>` / Charon
                    // `Option<[u8]>`) is the window itself: `None` is a null
                    // data word, `Some(w)` is `w` (`rnone.py` / `rclass.py`:
                    // a nullable value's non-None payload is the same word).
                    // `Some.__pos_0` is therefore the Option's data word, not
                    // a tagged-instance field. The `Discriminant` arm uses
                    // the same `tyref_is_u8_window` gate. The Option is a
                    // `{data, len}` pair, so the one-word projection is
                    // `data` via `pair_place_parts`, not `LocalValue::one`.
                    if field_name == "__pos_0" && tyref_is_u8_window(&inner.ty, self.llbc) {
                        if let Some((data, _)) = self.pair_place_parts(mir_bb, inner.as_ref())? {
                            return Ok(data);
                        }
                        return self.resolve_place(mir_bb, *inner);
                    }
                    // The payload of `Option<fieldless-enum>` is likewise the
                    // niche scalar itself.  A `Some.__pos_0` projection must
                    // therefore alias the Option value instead of treating
                    // that byte as an aggregate pointer and dereferencing it.
                    if field_name == "__pos_0"
                        && tyref_option_fieldless_niche(&inner.ty, self.llbc).is_some()
                    {
                        return self.resolve_place(mir_bb, *inner);
                    }
                    // `Option<NonZero*>` is that same integer word. `Some.0`
                    // is the word, not a field of an aggregate. Charon names
                    // the positional field `_0`; a missing name becomes
                    // `__pos_0`.
                    if (field_name == "_0" || field_name == "__pos_0")
                        && tyref_option_nonzero_scalar(&inner.ty, self.llbc)
                    {
                        return self.resolve_place(mir_bb, *inner);
                    }
                    // A zero-sized field has no runtime representation.
                    // `rclass.py InstanceRepr.getfield` emits `getfield`
                    // with `resulttype=r` where `r.lowleveltype is Void`
                    // for a void attribute, and `jtransform.py
                    // rewrite_op_getfield` drops a getfield whose result
                    // is Void. The read is a Void value for every base.
                    // Give the projected value the `Void` kind rather than
                    // materialising it in a value bank.  `getkind(lltype.Void)`
                    // is `'void'`, and both `call.py NON_VOID_ARGS` and
                    // `jtransform.py add_in_correct_list` drop a void argument, so the value
                    // takes neither a calldescr `arg_classes` slot nor an
                    // argument-bank entry.  That is what the callee's machine
                    // ABI expects: a zero-sized parameter occupies no argument
                    // register, so declaring a slot for it would shift every
                    // later argument by one register at the residual-call
                    // boundary.  A bank-classified value would take that slot —
                    // `Ref` is what an opaque zero-sized ADT falls back to.
                    // rmodel.py::pairtype(Repr, VoidRepr).convert_from_to
                    // returns an actual Constant(None, lltype.Void). Spell
                    // that constant with the legacy graph's ConstNone define
                    // so the adapter can preserve its identity through phi
                    // simplification; a fresh undefined Void variable loses
                    // the constant representative on outgoing links.
                    // A named empty class field is still an instance
                    // (`tyref_keeps_empty_class_instance`); the read is
                    // `InstanceRepr.getfield`, not Void.
                    if (tyref_is_void_zst(&place_ty, self.llbc)
                        && !tyref_keeps_empty_class_instance(
                            &place_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                        ))
                        || self.tyref_is_fieldless_enum(&inner.ty)
                        || self.tyref_is_borrowed_fieldless_enum(&inner.ty)
                    {
                        return Ok(self.emit_unit(self.emit_bb(mir_bb)));
                    }
                    // Narrow a classdef-less raw-pointer-deref base to
                    // `SomeInstance(<pointee root>)` before the field read.
                    // A `(*p).field` where `p: *mut/*const <Struct>` deref's to
                    // a value whose pointee class root `tyref_to_value_type`
                    // erases to `Ref(None)`; when such a value arrives as a
                    // callee-propagated argument (a non-parameter `PyObjectRef`
                    // bound through `recursivecall`), the downstream getattr
                    // blocks the annotator on a classdef-less instance. The
                    // field's declaring struct is authoritative, so this is a
                    // sound downcast — identity when the base already carries
                    // the class — the same `__cast_instance_intrinsic` narrow the
                    // ptr-identity-cast arm applies before its own field reads.
                    let narrow_root: Option<String> = match &inner.kind {
                        PlaceKind::Projection(pre, ProjectionElem::Atom(s)) if s == "Deref" => {
                            let stripped = tyref_node(&pre.ty, self.llbc)
                                .and_then(|n| strip_ty_wrappers(n, self.llbc));
                            // `(*p).f` on a raw address is `getfield_raw` at the
                            // host offset. A cast onto the gc shell is the Gc
                            // pointee's narrow.
                            if stripped.as_ref().is_some_and(|n| {
                                raw_ptr_node_is_declared_raw(n, self.llbc, self.gc_struct_ids)
                            }) {
                                None
                            } else {
                                stripped.and_then(|n| {
                                    raw_ptr_pointee_class_root_with(
                                        n,
                                        self.llbc,
                                        self.tombstoned_leaves,
                                    )
                                    .or_else(|| {
                                        let root = adt_node_class_root_with(
                                            n,
                                            self.llbc,
                                            self.tombstoned_leaves,
                                        )?;
                                        // `&ObjectKey` is the dict key the
                                        // `eq` body reads `.hash` from. Other
                                        // structs stay uncast: a blanket
                                        // narrow paints `Tuple` onto values
                                        // that are already tuples.
                                        (root == "ObjectKey").then_some(root)
                                    })
                                })
                            }
                        }
                        _ => None,
                    };
                    // Whether the container was reached through a pointer.
                    // Distinct from `narrow_root`, which is only `Some` for
                    // a *raw-pointer* deref whose pointee class resolves —
                    // a `&self` deref leaves it `None`, so it cannot stand
                    // in for this test.
                    let base_is_deref = inner.is_deref_projection();
                    // Whether this read projects an enum variant's payload,
                    // read off the container before the base resolve
                    // consumes `inner`.  It selects the payload projection
                    // below.
                    let container_is_enum = tyref_is_enum_free(&inner.ty, self.llbc);
                    // Captured before `resolve_place` moves `inner`. A
                    // reference-typed Result/Option shells classdef-less;
                    // the payload read casts to the variant class below.
                    // Carrier `Result<T, PyError>` is the exception transform.
                    // `owner_root` is the variant leaf (`Result<String,E>::Ok`),
                    // not the crate path. A leading `Result::` / `Result<` is
                    // that class; `::Result` covers a qualified spelling.
                    let enum_payload_owner = owner_root.starts_with("Result::")
                        || owner_root.starts_with("Result<")
                        || owner_root.starts_with("Option::")
                        || owner_root.starts_with("Option<")
                        || owner_root.contains("::Result::")
                        || owner_root.contains("::Result<")
                        || owner_root.contains("::Option::")
                        || owner_root.contains("::Option<");
                    let enum_payload_class = container_is_enum
                        && enum_payload_owner
                        && !crate::front::result_exc::tyref_is_result_of_carrier(
                            &inner.ty,
                            self.llbc,
                            self.static_addrs.error_carrier,
                        );
                    // A closure env is identified from the type decl's
                    // `src: Closure` origin, not from the `closure` name
                    // leaf.  Needed before `resolve_place` consumes `inner`.
                    let owner_is_closure_env = tyref_is_closure_env(&inner.ty, self.llbc);
                    if let Some(res) = self.try_emit_interior_field_read(
                        mir_bb,
                        &inner,
                        &owner_root,
                        &field_name,
                        &owner_id,
                        &field_ty,
                        &place_ty,
                        container_is_enum,
                        owner_is_closure_env,
                    )? {
                        return Ok(res);
                    }
                    if let Some((owner, flat_name, owner_id)) =
                        self.flatten_list_storage_field(&inner, &elem)
                    {
                        let list_place = peel_nested_storage_place(clone_place(&inner));
                        let base = self.resolve_place(mir_bb, list_place)?;
                        let bb_id = self.emit_bb(mir_bb);
                        let ty = tyref_to_value_type_with(
                            &place_ty,
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
                                field: FieldDescriptor::new(flat_name, Some(owner))
                                    .with_owner_id(owner_id)
                                    .with_base_is_deref(true),
                                ty,
                                pure: false,
                            },
                        });
                        return Ok(res);
                    }
                    let owner_declared_gc = container_declared_gc(
                        &inner.ty,
                        self.llbc,
                        self.gc_struct_ids,
                        self.immutable_fields,
                    );
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let base = if let Some(root) = narrow_root {
                        let narrowed = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(narrowed.clone()),
                            kind: crate::model::cast_instance_call(root, base),
                        });
                        narrowed
                    } else {
                        base
                    };
                    let base = if enum_payload_class {
                        let narrowed = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(narrowed.clone()),
                            kind: crate::model::cast_instance_call(owner_root.clone(), base),
                        });
                        narrowed
                    } else {
                        base
                    };
                    // The field's DECLARED ty is the polymorphic decl's
                    // (monomorphize=false): for a generic container
                    // (`ControlFlow<B, C>`, `Result<T, E>`) it is a bare
                    // type variable, which projects to the `Ref`
                    // fallback even when the instantiated payload is an
                    // `i64`.  The place's post-projection type carries
                    // the substituted use-site type, so prefer it; keep
                    // the decl ty for the rare place shapes whose
                    // post-projection type the reader cannot resolve.
                    //
                    // A shell variant's payload takes the shell projection:
                    // the `&P` a slice accessor hands back through
                    // `Option`/`Result`/`ControlFlow` is the primitive, not
                    // a pointer the program stores.  A closure-env field
                    // whose declared type is a shared borrow of a primitive
                    // is the same integer: the body reads the capture
                    // through that borrow as a scalar (`Rvalue::Ref`
                    // aliases the referent).  An ordinary struct's `&P`
                    // field is a pointer the program stores and compares.
                    let ty = crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                        &owner_root,
                        &field_name,
                        adt_field_read_value_type(
                            &place_ty,
                            &field_ty,
                            container_is_enum,
                            owner_is_closure_env,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                    );
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let narrow_instance_class = container_is_enum
                        && field_name == "__pos_0"
                        && self.enum_payload_is_nullable_instance_ptr(&place_ty);
                    // `Constants<C>(Box<[C]>)` is opaque in the interpreter
                    // snapshot, so the field ty is not a visible `Box<[T]>`.
                    // The length is still the fat metadata word.
                    let constants_box = (field_name == "__pos_0" || field_name == "0")
                        && (owner_root == "Constants" || owner_root.ends_with("::Constants"));
                    // `SetStrategyRef::deref` / `DictStrategyRef::deref` is
                    // `self.imp` (`&dyn SetStrategy` / `&dyn DictStrategy`).
                    // The caller reads `method_*` off that returned word as
                    // the vtable. Trait-impl paths render as `<Impl>::deref`,
                    // so the field's owner is what identifies the read. The
                    // data word is the unit strategy; the vtable is next.
                    let dyn_vtable = field_name == "imp"
                        && (owner_root == "SetStrategyRef"
                            || owner_root.ends_with("::SetStrategyRef")
                            || owner_root == "DictStrategyRef"
                            || owner_root.ends_with("::DictStrategyRef"))
                        && (tyref_is_dyn_pointer(&field_ty, self.llbc)
                            || tyref_is_dyn_pointer(&place_ty, self.llbc));
                    let mut field = FieldDescriptor::new(field_name, Some(owner_root))
                        .with_owner_id(owner_id)
                        .with_owner_declared_gc(owner_declared_gc)
                        .with_base_is_deref(base_is_deref)
                        .with_host_index(host_index)
                        .with_inline_vec(
                            field_ty_is_inline_vec(&field_ty, self.llbc)
                                || field_ty_is_inline_vec(&place_ty, self.llbc),
                        );
                    // `RustVecRepr`: a one-word-item `Vec` is the header
                    // address (`rewrite_op_getsubstruct`). `with_vec_part`
                    // later clears this for Buf/Len so `.len()` stays a load.
                    if tyref_rust_vec_item_kind(&field_ty, self.llbc).is_some()
                        || tyref_rust_vec_item_kind(&place_ty, self.llbc).is_some()
                    {
                        field = field.with_taken_by_address(true);
                    }
                    if dyn_vtable {
                        field = field.with_vec_part(crate::model::VecFieldPart::FatVtable);
                    }
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field,
                            ty,
                            pure: false,
                        },
                    });
                    if tyref_is_inline_fat_ptr(&field_ty, self.llbc)
                        || tyref_is_inline_fat_ptr(&place_ty, self.llbc)
                        || tyref_is_u8_window(&field_ty, self.llbc)
                        || tyref_is_u8_window(&place_ty, self.llbc)
                        || constants_box
                    {
                        self.fat_box_vars.push(res.clone());
                    }
                    // The option (or the guard itself) is the reference
                    // `OwnerRootGuard::new` stored. A later `.0 as *mut T`
                    // has to return this read, not an integer cast of it.
                    if owner_root_guard::tyref_is_guard(&place_ty, self.llbc)
                        || owner_root_guard::tyref_is_option_of_guard(&place_ty, self.llbc)
                    {
                        self.guard_word_vars.insert(res.clone());
                    }
                    if narrow_instance_class {
                        let narrowed = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(narrowed.clone()),
                            kind: crate::model::cast_instance_call("PyObject", res),
                        });
                        return Ok(narrowed);
                    }
                    // `W_BytesObject.data` is the `rstr.py` `STR` owner.
                    return Ok(res);
                }
                // `xs[i]` element read — the symmetric counterpart of
                // the `ArrayWrite` Index arm in
                // `emit_projection_write`.  Collapsing to the base
                // (the previous behaviour for every non-Field Tagged
                // projection) aliased the element to the sequence
                // itself, so a method call on the element resolved
                // against the list annotation.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(index_payload) = v.as_object().and_then(|m| m.get("Index"))
                {
                    let idx_var = self.index_offset_var(mir_bb, index_payload)?;
                    let copied_bytes = self.graph.name.ends_with("::copied_byte_vec");
                    let string_byte_view = copied_bytes
                        || self
                            .string_byte_view_locals
                            .iter()
                            .any(|&local| place_references_local(&inner, local));
                    let list_view = self.place_is_list_view(inner.as_ref());
                    let window_index = !string_byte_view
                        && !list_view
                        && self.place_is_window_index_base(inner.as_ref());
                    let string_array_view = self
                        .string_array_view_locals
                        .iter()
                        .any(|(local, _)| place_references_local(&inner, *local));
                    let (value_local, index_spelling) = index_identity_inputs(&inner, self.llbc);
                    let (mut array_type_id, mut nolength) =
                        fixed_array_index_identity(value_local, &index_spelling);
                    if string_array_view {
                        array_type_id = Some(STRING_GCREF_GCARRAY_TYPE_ID.to_string());
                        nolength = crate::front::typestr::nolength_from_array_type_id(
                            array_type_id.as_deref(),
                        );
                    }
                    if window_index {
                        let data = if let Some((ptr, _)) =
                            self.pair_place_parts(mir_bb, inner.as_ref())?
                        {
                            ptr
                        } else {
                            let base = self.resolve_place(mir_bb, *inner)?;
                            let bb_id = self.emit_bb(mir_bb);
                            self.fat_component(bb_id, &base, crate::model::VecFieldPart::FatData)
                                .unwrap_or(base)
                        };
                        let loaded = self.emit_window_raw_load(mir_bb, data, idx_var);
                        return Ok(loaded);
                    }
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let element_is_pointer_word = self.var_tracks_pointer_word_array(&base);
                    if element_is_pointer_word {
                        self.pointer_word_vars.insert(res.clone());
                    }
                    // `Box<[T]>`'s data word has no length header. `[u8]` is
                    // the length-prefixed byte block, so only this base is
                    // remapped onto `Vec<T>`.
                    let fat_data = if string_byte_view || list_view {
                        None
                    } else {
                        self.fat_component(bb_id, &base, crate::model::VecFieldPart::FatData)
                    };
                    let base = if let Some(data) = fat_data {
                        if !string_array_view {
                            let remapped =
                                crate::front::typestr::fat_box_scalar_array_id(&index_spelling)
                                    .or_else(|| {
                                        array_type_id.as_deref().and_then(
                                            crate::front::typestr::fat_box_scalar_array_id,
                                        )
                                    });
                            if let Some(id) = remapped {
                                array_type_id = Some(id.to_string());
                                nolength = crate::front::typestr::nolength_from_array_type_id(
                                    array_type_id.as_deref(),
                                );
                            }
                        }
                        data
                    } else {
                        base
                    };
                    // An object-pointer index is `ll_getitem_fast` on a list
                    // of `PyObject`. That slice arrives as a classdef-less
                    // instance, and `getitem_SomeInstance` rewrites the
                    // index to `getattr("__getitem__")`. `[PyObject]` is
                    // the spelling `project_struct_field_type` projects to
                    // `SomeList` (`rtyper_makerepr`). The ARRAY identity
                    // already names that list; every graph with this type
                    // takes the same path.
                    let base = if !string_byte_view
                        && array_type_id.as_deref() == Some(OBJECT_REF_GCARRAY_TYPE_ID)
                    {
                        self.narrow_value_to_instance_root(
                            bb_id,
                            crate::model::LinkArg::Value(base),
                            "[PyObject]",
                        )
                        .as_variable()
                        .expect("an object-pointer slice base stays a Variable")
                        .clone()
                    } else {
                        base
                    };
                    let kind = if string_byte_view {
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__string_byte_getitem".to_string()],
                                fun_decl_id: None,
                            },
                            args: crate::model::call_args(vec![base, idx_var]),
                            result_ty: if copied_bytes {
                                ValueType::Unsigned
                            } else {
                                ValueType::Int
                            },
                        }
                    } else {
                        OpKind::ArrayRead {
                            base,
                            index: idx_var,
                            item_ty: if string_array_view {
                                ValueType::Str
                            } else if element_is_pointer_word {
                                ValueType::Ref(None)
                            } else {
                                tyref_to_value_type_with(
                                    &place_ty,
                                    self.llbc,
                                    self.tombstoned_leaves,
                                    self.gc_struct_ids,
                                    &self.raw_alloc_flavor_structs,
                                )
                            },
                            array_type_id,
                            nolength,
                            pure: false,
                        }
                    };
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind,
                    });
                    return Ok(res);
                }
                // Positional aggregate `.N` read: the base local was
                // bound by a non-Adt `Rvalue::Aggregate`, so emit the
                // `FieldRead __pos_<N>` that pairs with the
                // construction-side `FieldWrite __pos_<N>` instead of
                // aliasing the base.
                if let ProjectionElem::Tagged(v) = &elem
                    && let PlaceKind::Local(i) = inner.kind
                    && let Some(owner_root) =
                        self.positional_aggregate_locals.get(&(i as usize)).cloned()
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some(idx) = self.positional_field_index(field_payload)
                {
                    let base = self.local_var[i as usize]
                        .as_ref()
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: read of MIR local {i} before any Assign — \
                             uninitialised local, not yet supported"
                            ))
                        })?
                        .one()?;
                    let bb_id = self.emit_bb(mir_bb);
                    // Element type, not `Ref(None)`: `place_ty` is the
                    // post-projection type, the same source the genuine-Ref
                    // tuple read below uses.  A `(FieldlessEnum, bool)`
                    // scrutinee is built as a positional aggregate and its
                    // `.0` read back to feed the `match` switch, so a blanket
                    // `Ref` here types that switch value as a pointer and
                    // `flatten.py assert kind == 'int'` rejects the graph.
                    // An unresolved projected type (a closure's `tupled_args`
                    // typed as the trait's `Args` parameter) falls back to
                    // the tuple's own element generics.
                    let ty = tuple_element_field_value_type(
                        &place_ty,
                        tyref_tuple_element(&inner.ty, idx, self.llbc),
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let elem_ty = tyref_tuple_element(&inner.ty, idx, self.llbc);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(format!("__pos_{idx}"), Some(owner_root))
                                .with_inline_vec(
                                    elem_ty
                                        .as_ref()
                                        .is_some_and(|ty| field_ty_is_inline_vec(ty, self.llbc))
                                        || field_ty_is_inline_vec(&place_ty, self.llbc),
                                ),
                            ty,
                            pure: false,
                        },
                    });
                    return Ok(res);
                }
                // A `Field` projection whose base is a genuine Ref tuple
                // (`inner`'s post-projection type is a non-unit
                // `(A, B, ...)`) extracts element N: emit a typed
                // `FieldRead __pos_<N>` carrying the element type, the
                // same shape the positional-aggregate read above
                // produces.  This covers tuples the lowering does not
                // build inline — function-return tuples, enum-variant
                // payloads read through an `Option`/`Result` downcast —
                // whose base is an opaque Ref rather than a
                // transparent-ctor aggregate, so the base flows through
                // `inner` as a Ref while element N may be an `Int`.
                // Without it, `tuple.1` aliases the whole tuple (Ref) and
                // a later merge with an `Int`-typed sibling value trips
                // the assembler's kind cross-check.
                //
                // The `*Checked (value, bool)` local is field-dependent.
                // It lowered to a scalar `BinOp` (`binop_result_locals`):
                // field `.0` is that scalar, so it collapses to the base
                // Variable below.  Field `.1` is the Rust overflow bit,
                // which the JIT IR does not model — the paired overflow
                // `Assert` is dropped in `lower_statement`, and ovfcheck
                // is carried by separate `int_*_ovf` + guard ops, never a
                // boolean tuple field.  A live read of `.1` therefore has
                // no lowering: fail loud rather than silently alias the
                // overflow bool to the arithmetic value.
                if let ProjectionElem::Tagged(v) = &elem
                    && self.place_is_tuple(&inner)
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some(idx) = self.positional_field_index(field_payload)
                {
                    if self.place_is_binop_scalar(&inner) {
                        if idx != 0 {
                            return Err(LowerError::Unsupported(format!(
                                "bb{mir_bb}: live read of field .{idx} of a \
                                 checked-binop `(value, bool)` local — the \
                                 overflow bit is not modeled (ovfcheck uses \
                                 separate guard ops, not a tuple field)"
                            )));
                        }
                        // idx == 0: fall through to the base-collapse below.
                    } else if !tyref_tuple_suffix(&inner.ty, self.llbc).is_empty() {
                        // Same spelling the construction-side FieldWrite
                        // chain records for builtin tuple aggregates
                        // (`aggregate_ctor_name` id atom + the per-shape
                        // `<…>` suffix rendered from this tuple's element
                        // types), so read and write attrs key under one
                        // owner. An unresolved suffix is the bare `Tuple`
                        // lattice, not a class named `Tuple`.
                        let owner = format!("Tuple{}", tyref_tuple_suffix(&inner.ty, self.llbc));
                        // The element type as the tuple type spells it.  A
                        // closure's `call_once` body reads its `tupled_args`
                        // through a place whose projected type Charon leaves
                        // as the trait's `Args` parameter, which projects to
                        // `Ref(None)`; the tuple's own generics still name the
                        // concrete element (`(char,)`).
                        let element_ty = tyref_tuple_element(&inner.ty, idx, self.llbc);
                        let inline_vec = element_ty
                            .as_ref()
                            .is_some_and(|ty| field_ty_is_inline_vec(ty, self.llbc))
                            || field_ty_is_inline_vec(&place_ty, self.llbc);
                        // A tuple reached through a pointer/reference deref
                        // (a residual call returning `&(A, B)`, a
                        // `*mut (A, B)` field) resolves its base to a
                        // classdef-less `SomeInstance`: pyre's pointer
                        // erasure drops the tuple shape, so the annotator
                        // Blocks the `__pos_<N>` read (dispatched as
                        // `getattr`) on a classdef-less instance.  Narrow the
                        // base to this tuple's `Tuple{suffix}` classdef first
                        // — identity when it already carries it — the same
                        // `__cast_instance_intrinsic` downcast the raw-ptr-deref
                        // Adt field arm applies (see `narrow_root` above).  A
                        // by-value tuple base is not deref-shaped and keeps
                        // its concrete classdef, so it is left untouched.
                        let base_is_deref = matches!(
                            &inner.kind,
                            PlaceKind::Projection(_, ProjectionElem::Atom(s)) if s == "Deref"
                        );
                        let base = self.resolve_place(mir_bb, *inner)?;
                        let bb_id = self.emit_bb(mir_bb);
                        let base = if base_is_deref {
                            let narrowed = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(narrowed.clone()),
                                kind: crate::model::cast_instance_call(owner.clone(), base),
                            });
                            narrowed
                        } else {
                            base
                        };
                        let ty = tuple_element_field_value_type(
                            &place_ty,
                            element_ty,
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
                                field: FieldDescriptor::new(format!("__pos_{idx}"), Some(owner))
                                    .with_inline_vec(inline_vec),
                                ty,
                                pure: false,
                            },
                        });
                        return Ok(res);
                    }
                }
                // `*_r` reads the place recorded when `_r` was bound, the
                // same place a later write updates. The binding itself
                // holds the value from before that write. A cell parameter
                // reads the cell's `value` field instead of collapsing to
                // the cell pointer.
                if let ProjectionElem::Atom(name) = &elem
                    && name == "Deref"
                    && let PlaceKind::Local(local) = &inner.kind
                {
                    let local = *local as usize;
                    if let Some(root) = self.gc_mut_ref_param_root(local) {
                        let base = self.defined_local(mir_bb, local)?;
                        return Ok(self.emit_gc_mut_ref_field_read(mir_bb, base, &root));
                    }
                    if matches!(
                        self.local_var.get(local).and_then(|slot| slot.as_ref()),
                        Some(LocalValue::ItemAddr(_))
                    ) && let Some((kind, item_ty)) = self.item_addr_slot_kind(local)
                    {
                        let addr = self.item_addr_var(mir_bb, local)?;
                        let index = self.emit_const_uint(mir_bb, 0);
                        let helper_ty = Self::pair_helper_item_ty(kind);
                        let item = self.emit_path_call(
                            mir_bb,
                            majit_ir::rvec::slice_helper_path(
                                majit_ir::rvec::SliceOp::GetItem,
                                kind,
                            ),
                            vec![addr, index],
                            helper_ty.clone(),
                        );
                        return Ok(self.retype_word(mir_bb, item, &helper_ty, &item_ty));
                    }
                    if let Some(recorded) = self.atomic_ref_place.get(&local).cloned() {
                        let followed = self.concrete_borrow_place(recorded);
                        // A local is the slot `mem::replace` writes. A projection
                        // stays the borrowed value: re-resolving it would emit a
                        // fresh field or index read for every borrow.
                        if matches!(followed.kind, PlaceKind::Local(_)) {
                            return self.resolve_place(mir_bb, followed);
                        }
                        if let Some(cell) = self.deref_cell_param_local(&followed) {
                            let root = self.gc_mut_ref_param_root(cell).ok_or_else(|| {
                                LowerError::Unsupported(format!(
                                    "bb{mir_bb}: GC reference cell {cell} has no root"
                                ))
                            })?;
                            let base = self.defined_local(mir_bb, cell)?;
                            return Ok(self.emit_gc_mut_ref_field_read(mir_bb, base, &root));
                        }
                    }
                }
                // `*(p as *const i64)` and `*guard` where `guard: &bool`
                // came back from a call (`MutexGuard::deref`) read one
                // primitive at that address. `jtransform.py`
                // `rewrite_op_raw_load` emits `raw_load_<kind>` and
                // `history.py` `getkind` banks the loaded word as `int`
                // (signed), `int` (unsigned / bool) or `float`. Aliasing
                // `p` returns the pointer, which this front end banks as
                // `Ref`, so a function declared to return the word
                // disagrees with its CFG. A reference `Rvalue::Ref` already
                // aliased is that word, and loading through it would treat
                // the word as an address — those locals sit on
                // `atomic_ref_place` and are excluded. A call result is not
                // that alias: it is recorded on `scalar_address_locals`.
                // `&raw const *p` of an address is not recorded as an alias.
                //
                // A `TypedItemsBlock` word is `GcArray(Signed|Float)`
                // `getarrayitem`, the same op [`Self::typed_items_elem_ptr_add`]
                // emits for `.add(idx)`. That arm binds the element on
                // `index_elem_alias`, so the following `*p` is the element
                // already read. A primitive deref is `rewrite_op_raw_load`;
                // `translate_op` carries that op as `raw_load` and `flowin`
                // binds `item_ty`. A deref with no `.add`
                // (`sizehint_state_value`) is that array at index 0.
                //
                // The object-ref twin: `*items_block_items_base(block)` is
                // `l.items[0]` (`rlist.py ll_getitem_fast`). A length-2
                // `tupleobject.py wraptuple` reads that slot then
                // `*base.add(1)`; without this ArrayRead the header alias
                // is the value itself and the second item never becomes a
                // Ref.
                let scalar_address = matches!(
                    inner.kind,
                    PlaceKind::Local(local)
                        if self.scalar_address_locals.contains(&(local as usize))
                );
                let deref_local = match &inner.kind {
                    PlaceKind::Local(local) => Some(*local as usize),
                    _ => None,
                };
                if let ProjectionElem::Atom(name) = &elem
                    && name == "Deref"
                    && let Some(local) = deref_local
                    && !self.index_elem_alias.contains_key(&local)
                    && let Some((item_ty, array_type_id)) =
                        raw_ptr_typed_items_element(&inner.ty, self.llbc)
                    && base_traces_to_typed_items_block_accessor(self.body, local, self.llbc)
                {
                    let ptr = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let base = self
                        .narrow_value_to_instance_root(bb_id, ptr.into(), &array_type_id)
                        .as_variable()
                        .expect("a materialized typed-items base stays a Variable")
                        .clone();
                    let index = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(index.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    let loaded = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(loaded.clone()),
                        kind: OpKind::ArrayRead {
                            base,
                            index,
                            item_ty,
                            array_type_id: Some(array_type_id.clone()),
                            nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            pure: false,
                        },
                    });
                    return Ok(loaded);
                }
                if let ProjectionElem::Atom(name) = &elem
                    && name == "Deref"
                    && let Some(local) = deref_local
                    && !self.index_elem_alias.contains_key(&local)
                    && is_object_ref_items_ptr(&inner.ty, self.llbc)
                    && base_traces_to_items_block_accessor(self.body, local, self.llbc)
                {
                    let ptr = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let index = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(index.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    let loaded = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(loaded.clone()),
                        kind: OpKind::ArrayRead {
                            base: ptr,
                            index,
                            item_ty: ValueType::Ref(None),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            pure: false,
                        },
                    });
                    return Ok(loaded);
                }
                if let ProjectionElem::Atom(name) = &elem
                    && name == "PtrMetadata"
                {
                    // A fat pointer's metadata word (`ClassRepr.getclsfield`'s
                    // vtable pointer, or a slice's length). Dyn metadata is
                    // `FatMeta` (`two_word_fat_struct` item 1); a slice is
                    // `FatLen` (`ListRepr.rtype_len`). `tyref_is_fat_dyn`
                    // covers `Box<dyn T>` as well as one-level `&dyn T`.
                    let is_dyn = tyref_is_fat_dyn(&inner.ty, self.llbc);
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let part = if is_dyn {
                        crate::model::VecFieldPart::FatMeta
                    } else {
                        crate::model::VecFieldPart::FatLen
                    };
                    if let Some(meta) = self.fat_component(bb_id, &base, part) {
                        return Ok(meta);
                    }
                    return Ok(base);
                }
                if let ProjectionElem::Atom(name) = &elem
                    && name == "Deref"
                    && (tyref_is_raw_pointer(&inner.ty, self.llbc) || scalar_address)
                    && !matches!(inner.kind, PlaceKind::Local(local)
                        if self.atomic_ref_place.contains_key(&(local as usize))
                            || self.index_elem_alias.contains_key(&(local as usize)))
                    && tyref_is_primitive_scalar(&place_ty, self.llbc)
                    && let Some((item_ty, itemsize, is_item_signed)) =
                        self.raw_word_descr(&place_ty)
                {
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let offset = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(offset.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    // Integer pointee (`u8` window byte, `CCHARP`) is `CHAR`.
                    // `Unknown` banks as `'r'` and `rewrite_op_raw_load`
                    // wraps the loaded word in `cast_int_to_ptr`.
                    let loaded_ct = if matches!(item_ty, ValueType::Int | ValueType::Unsigned) {
                        crate::model::ConcreteType::Signed
                    } else {
                        crate::model::ConcreteType::Unknown
                    };
                    let loaded = self.graph.alloc_value_var_with_type(loaded_ct);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(loaded.clone()),
                        kind: OpKind::RawLoad {
                            base,
                            offset,
                            item_ty,
                            itemsize,
                            is_item_signed,
                        },
                    });
                    return Ok(loaded);
                }
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && tyref_transparent_inner_value_type(
                        &inner.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    )
                    .is_some()
                {
                    // Same identity as `tyref_transparent_nonzst_field` on
                    // a defining crate: an Opaque wrapper has no field list,
                    // so [`Self::resolve_adt_field`] never reaches that arm.
                    // The non-ZST field is the word; a ZST marker has no
                    // layout word (`opaque_scalar_field` is `None`).
                    if self
                        .opaque_scalar_field(&inner.ty, &place_ty, field_payload)
                        .is_some()
                    {
                        let wrapper_ty = clone_tyref(&inner.ty);
                        let base = self.resolve_place(mir_bb, *inner)?;
                        return Ok(self.cast_off_error_carrier(mir_bb, &wrapper_ty, base));
                    }
                    return Ok(self.emit_unit(self.emit_bb(mir_bb)));
                }
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some((owner, name, ty, word)) =
                        self.opaque_scalar_field(&inner.ty, &place_ty, field_payload)
                {
                    // `(*p).field` of an opaque dependency struct. The
                    // declaration has no field list, and the place type is
                    // the scalar, so the load is that layout word.
                    let base_is_deref = inner.is_deref_projection();
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.emit_bb(mir_bb);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(name, Some(owner))
                                .with_base_is_deref(base_is_deref)
                                .with_scalar_word(word),
                            ty,
                            pure: false,
                        },
                    });
                    return Ok(res);
                }
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(offset) = v.get("Offset")
                {
                    return self.resolve_offset_place(mir_bb, *inner, offset);
                }
                match elem {
                    ProjectionElem::Tagged(_) | ProjectionElem::Atom(_) => {
                        self.resolve_place(mir_bb, *inner)
                    }
                }
            }
            // `Global { id, .. }` — static/const item reference.
            // The production trace supplies host addresses for the
            // object-space singletons pyre reads from statics. Preserve
            // those as constants; a synthetic 0-arg call would invent a
            // callable that neither Rust nor RPython has.
            PlaceKind::Global { id, .. } => {
                // A `NamedConst` (Rust `const`, not `static`) has no
                // address — the value is inlined at every use site.
                // Charon still emits a `Global` read, so fold the
                // trivial literal initializer to a constant rather than
                // calling a non-existent accessor.  Statics keep the
                // address/`FunctionPath` path.
                if let Some(res) = self.emit_named_const_value(mir_bb, id, &place_ty) {
                    return Ok(res);
                }
                // rustc promotes `(a..=b)` to an `AnonConst` whose init is
                // `RangeInclusive::new(const, const)` plus a `&` return.
                // `promoted_init_from_body` requires a 1-BB assign+return,
                // so the 3-BB Call+Ref+Unwind shape is not spliced, and
                // the Global read would residualize a 0-arg Call on the
                // static's path (`…::is_primitive::promoted`). That path
                // is not a function: `symbolic_fnaddr_for_path` hashes it
                // to `SYMBOLIC_FNADDR_BASE` and the walker aborts.
                //
                // RPython constants are live host values at translation
                // time (`const_eval_init_body`); un-promote the range back
                // into the caller as the inline `new(lo, hi)`. PyPy's
                // `isinstance(ct, W_CTypePrimitive)` on a promoted ctype
                // is a constant-folded class check
                // (`ctypeprim.W_CTypePrimitive`); a helper with a graph is
                // `guess_call_kind` → `'regular'` (`call.py`) → `inline_call`.
                if let Some((lo, hi)) = promoted_int_range_inclusive_bounds(self.llbc, id)
                    && tyref_is_int_range_inclusive(&place_ty, self.llbc)
                {
                    return Ok(self.emit_promoted_int_range_inclusive(mir_bb, lo, hi, &place_ty));
                }
                let segments = self.global_segments(mir_bb, id)?;
                // `pyre_object::pyobject::PY_NULL` is the Rust spelling of
                // the `None` stored in PyPy's `locals_cells_stack_w` slots
                // (`pyframe.py`: `[None] * size`, plus the pop/drop clears).
                // Preserve the high-level W_Root boundary: construct the
                // repr-adaptive nullable pointer, then narrow it to the
                // declared `PyObjectRef` pointee class.  Leaving the computed
                // const to `const_eval_global` produces a bare nullable
                // `SomeInstance(None)`; unioning that into the fixed object
                // array erases its `PyObject` external item repr, whereas
                // upstream's `SomeNone ∪ SomeInstance(W_Root)` keeps W_Root
                // and only sets `can_be_None`.
                if fmt_path_ends_with(&segments, &["pyobject", "PY_NULL"])
                    && tyref_class_root_with(&place_ty, self.llbc, self.tombstoned_leaves)
                        .as_deref()
                        == Some("PyObject")
                {
                    let bb_id = self.emit_bb(mir_bb);
                    let raw = self.graph.push_null_mut_ptr(bb_id);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: crate::model::cast_instance_call("PyObject", raw),
                    });
                    return Ok(res);
                }
                // A class-singleton static (`&SLICE_TYPE`, `&CEL_INT_CLASS`):
                // narrow the raw address through
                // `__cast_instance_intrinsic[<root>]` so the read types
                // `SomeInstance(<root>)`, matching the `(*obj).ob_type`
                // field-read.  The bare `ConstInt` address would pair
                // `IntegerRepr` against that field's `InstanceRepr` and
                // block `rtype_is_` on the `ob_type == &TYPE`
                // pointer-identity chain — the same narrow
                // `obj as *const RegisteredStruct` already uses.
                //
                // The root is the static's own declared type, read off the
                // `Global` place, not a fixed name: a prebuilt instance is
                // annotated with the class of the object itself
                // (`rpython/annotator/bookkeeper.py` `immutablevalue`
                // `SomeInstance(self.getuniqueclassdef(x.__class__))`).
                // pyre's whole `pytypes` bucket is declared `PyType`
                // (`pyre-object/src/pyobject.rs`, and the
                // `#[pyre_class]`-minted statics at
                // `pyre-macros/src/lib.rs`), so it derives `"PyType"`
                // exactly as the fixed name did; a host crate with its own
                // class root (`charon-corpus` `CelClass`) derives that root
                // instead of being stamped with pyre's.
                //
                // A `pytypes` row whose static is not a named ADT has no
                // class root to narrow to and is not a class singleton at
                // all — a mis-bucketed key.  Declining here drops it to the
                // ordinary global handling below (`refs` / `int_values` /
                // initialiser fold / residual call), which is what any other
                // static of that shape gets.  Measured empty for pyre: all
                // 7362 `pyre-object` local functions lower with an identical
                // `__cast_instance_intrinsic` root histogram either way.
                //
                // The identity route runs first and the path route is the
                // fallback, not the other way round: a `PyreClassPyTypeOf`
                // associated const has a path, and that path names its
                // siblings too, so letting the key matcher answer first
                // would let the label decide a question the identity can
                // answer exactly.
                if let Some(addr) = self
                    .pytype_addr_by_impl_identity(id)
                    .or_else(|| self.pytype_static_addr(&segments))
                    && let Some(root) =
                        tyref_class_root_with(&place_ty, self.llbc, self.tombstoned_leaves)
                {
                    let bb_id = self.emit_bb(mir_bb);
                    let raw = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(raw.clone()),
                        kind: OpKind::ConstRefAddr(addr),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: crate::model::cast_instance_call(root, raw),
                    });
                    return Ok(res);
                }
                // Bookkeeper.immutablevalue annotates a prebuilt instance
                // with its class regardless of whether that class has fields.
                // Preserve that owner on both unit strategy implementations
                // and field-bearing strategy holders, whose address is stored
                // in W_DictObject.dstrategy. A bare ConstRefAddr would instead
                // annotate the latter as SomePtr and lose its instance repr.
                if let Some(root) = self.refs_static_struct_root(&place_ty)
                    && let Some(op @ OpKind::ConstRefAddr(_)) = self.static_addr_op(&segments)
                {
                    let bb_id = self.emit_bb(mir_bb);
                    let raw = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(raw.clone()),
                        kind: op,
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: crate::model::cast_instance_call(root, raw),
                    });
                    return Ok(res);
                }
                // A prebuilt JitDriver `const` has no registered address. Its
                // value exists in the flow graph only as operand zero of a
                // marker method: `ExtEnterLeaveMarker.specialize_call` makes
                // it a receiver Constant. `rpbc.py SingleFrozenPBCRepr`
                // gives a frozen PBC with no attributes `lltype.Void`; stamp
                // that here so pre-rtyper `split_block` rematerialises
                // `same_as(Constant(None, Void))` (`unsimplify.py`).
                // `Transformer.try_handle_jit_marker` discards the receiver.
                // Keep the producer constant-shaped so `split_block` can
                // rematerialise it, without turning the const path into a
                // nullary residual call. The declared ADT is matched against
                // the same configured receiver roots the codewriter uses; the
                // global's own name is deliberately irrelevant.
                if self.is_jitdriver_named_const(id, &place_ty) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                    let bb_id = self.emit_bb(mir_bb);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ConstRefAddr(JITDRIVER_NAMEDCONST_SENTINEL_ADDR),
                    });
                    return Ok(res);
                }
                let op = self
                    .static_addr_op(&segments)
                    .or_else(|| self.static_int_value_op(&segments))
                    .or_else(|| known_array_layout_const(&segments))
                    .or_else(|| self.const_eval_global(id))
                    .or_else(|| self.fold_size_const_global(id))
                    .or_else(|| self.fold_transparent_int_const_global(id))
                    .or_else(|| self.fold_named_const_int_array_global(id))
                    .or_else(|| self.fold_named_const_str_array_global(id))
                    .or_else(|| self.fold_const_fn_int_array_global(id))
                    .or_else(|| primitive_float_const(&segments))
                    .or_else(|| code_flags_const(&segments))
                    .or_else(|| encoding_name_const(&segments))
                    .or_else(|| gc_ref_null_const(&segments))
                    .or_else(|| bitflags_trait_empty_const(&segments));
                // No lane produced a value for this static.  The synthetic
                // nullary call below targets the static's own path, which is
                // not a function, so the failure only becomes visible two
                // stages later as `not registered in CallRegistry` against
                // that path — indistinguishable from a genuinely unregistered
                // callee.  Record the lane failure here, where the reason
                // still exists, so the census can tell the two apart.
                let op = op.unwrap_or_else(|| {
                    crate::decline::record_named(
                        crate::decline::gate::GLOBAL_PLACE,
                        "no-lane-produced-a-value",
                        &segments.join("::"),
                    );
                    OpKind::Call {
                        target: CallTarget::function_path(segments),
                        args: crate::model::call_args(vec![]),
                        result_ty: tyref_to_value_type_with(
                            &place_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                    }
                });
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let bb_id = self.emit_bb(mir_bb);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: op,
                });
                Ok(res)
            }
            PlaceKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Place::Unknown"
            ))),
        }
    }

    /// Resolve a Charon `Field` projection payload to the
    /// `(owner_root_leaf, field_name, field_ty)` triple suitable for
    /// `OpKind::FieldRead` emission.
    ///
    /// Charon encodes a Field as `[variant_idx, idx]` where
    /// `variant_idx` is `null` for structs and the variant position for
    /// enums; the owning ADT is the base place's type.  Returns `None` when:
    ///
    /// - the base type is not a nominal `Adt` (Tuple etc. — caller falls
    ///   back to the existing collapse-to-base behaviour);
    /// - the `type_id` is missing from the LLBC's type table
    ///   (forward-decl / opaque);
    /// - the resolved TypeDecl is not `Struct(_)` / `Enum(_)`;
    /// - the field index is out of range for the resolved variant.
    ///
    /// Resolve a Charon `AggregateKind::Adt` payload to the
    /// `(owner_path, ctor_leaf, field_rows)` triple suitable for a
    /// transparent-ctor + FieldWrite chain emission.  Each row carries the
    /// field's name and its rendered type, so the emission site can tell a
    /// zero-sized field (which gets no slot and therefore no write) from a
    /// stored one.
    ///
    /// Charon encodes `Aggregate(AggregateKind::Adt(type_id,
    /// variant_idx, ..), operands)` as `{"Adt": [type_id, variant_idx,
    /// ..]}`.  Struct aggregates use `variant_idx = null` and pull
    /// field names and types straight from the
    /// `TypeDeclKind::Struct(fields)` list; enum aggregates use a
    /// non-null `variant_idx` to select
    /// the right `VariantDecl` and emit the qualified ctor leaf
    /// (`Variant`) under the enum's `owner_path` (everything up to but
    /// not including the leaf in the resolved `name_path()`).
    ///
    /// Returns `None` when the kind is not Adt or the LLBC has no
    /// `TypeDecl` for `type_id`; the caller then falls back to the
    /// generic-tag ctor name with positional `__pos_<i>` fields.
    /// The construction-side `owner_root` when `rvalue` is an
    /// [`Rvalue::Aggregate`] that the lowering models as a synthetic
    /// transparent-ctor + positional `__pos_<i>` `FieldWrite` chain —
    /// i.e. a non-Adt aggregate (tuple / array / closure) for which
    /// [`Self::resolve_aggregate_adt`] returns `None`.  The owner is
    /// exactly what the [`Rvalue::Aggregate`] arm uses as
    /// `result_ty_owner` (`aggregate_ctor_name` plus its per-shape suffix,
    /// since `owner_path` is empty for the unresolved branch), so storing it
    /// lets `.N` reads emit a `FieldRead __pos_<N>` with the matching
    /// `owner_root`.
    /// Returns `None` for Adt aggregates: their `.field` reads already
    /// take the typed [`Self::resolve_adt_field`] path and never reach
    /// the collapse fallback.
    pub(in crate::front::mir) fn positional_aggregate_owner(
        &self,
        rvalue: &Rvalue,
        dest_ty: &TyRef,
    ) -> Option<String> {
        match rvalue {
            Rvalue::Aggregate(kind, _) if self.resolve_aggregate_adt(kind, None).is_none() => {
                // Suffix from `dest_ty` (the positional aggregate's
                // `place.ty`, element types present), matching the
                // construction-side `build_rvalue` owner and the projection
                // read; the `AggregateKind` head carries no element `types`.
                Some(format!(
                    "{}{}",
                    aggregate_ctor_name(kind),
                    tyref_positional_aggregate_suffix(dest_ty, self.llbc)
                ))
            }
            _ => None,
        }
    }

    /// True when `place`'s post-projection type is a non-unit tuple
    /// `(A, B, ...)` — a genuine Ref tuple whose `.N` reads extract
    /// element N rather than aliasing the base.
    pub(in crate::front::mir) fn place_is_tuple(&self, place: &Place) -> bool {
        tyref_is_tuple(&place.ty, self.llbc)
    }

    /// True when `place` is a bare local bound by a scalar
    /// [`Rvalue::BinaryOp`] (a `*Checked (value, bool)` result lowered to
    /// a single scalar Variable): its `.0` read must collapse to that
    /// scalar, not extract a tuple element.  See
    /// [`Lowering::binop_result_locals`].
    pub(in crate::front::mir) fn place_is_binop_scalar(&self, place: &Place) -> bool {
        matches!(
            &place.kind,
            PlaceKind::Local(i) if self.binop_result_locals.contains(&(*i as usize))
        )
    }

    /// The field index of a `Field` projection payload `[variant, idx]`.
    /// `None` for a malformed payload.
    pub(in crate::front::mir) fn positional_field_index(
        &self,
        payload: &serde_json::Value,
    ) -> Option<usize> {
        indexed_field_parts(payload).map(|(_, idx)| idx)
    }

    /// The discriminant integer of a fieldless-enum `AggregateKind::Adt`
    /// construction, or `None` for any non-fieldless-enum aggregate
    /// (struct / tuple / array / payload-enum variant).  Charon encodes
    /// `AggregateKind::Adt(type_id, variant_idx, ..)` as `{"Adt":
    /// [type_id, variant_idx, ..]}` (same head shapes `resolve_aggregate_adt`
    /// decodes); a fieldless enum's variant carries zero fields, so the
    /// whole value is its `discriminant_i64()`.
    pub(in crate::front::mir) fn aggregate_fieldless_enum_discriminant(
        &self,
        kind: &serde_json::Value,
    ) -> Option<i64> {
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => type_decl_ref_adt_id(head.as_object()?)?,
        };
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64)? as usize;
        let td = self.llbc.type_by_id(type_id)?;
        match &td.kind {
            TypeDeclKind::Enum(variants) if type_decl_is_fieldless_enum(td, self.llbc) => {
                variants.get(variant_idx)?.discriminant_i64()
            }
            TypeDeclKind::Opaque => opaque_fieldless_enum_tags(self.llbc, td)?
                .get(variant_idx)
                .copied(),
            _ => None,
        }
    }

    /// The trailing `bool` is `true` when the decl is a `TypeDeclKind::Struct`.
    /// A struct's value is its fields; a variant additionally carries a tag
    /// Charon does not spell as an operand, and the two must stay
    /// distinguishable downstream — see `CallTarget::SyntheticTransparentCtor`'s
    /// `is_struct`.
    /// How `Rvalue::Aggregate(kind, ..)` with `n_operands` operands builds
    /// a value of `dest_ty`.  Decided before any operand is resolved, so a
    /// constructor function item (`Result::map(Some)`) builds the same shape
    /// from values it already holds.
    pub(in crate::front::mir) fn aggregate_shape(
        &self,
        kind: &serde_json::Value,
        dest_ty: &TyRef,
        n_operands: usize,
    ) -> Result<AggregateShape, String> {
        // A fieldless (C-like) enum variant carries no payload, so
        // constructing it is just naming its discriminant integer
        // — the by-value representation of the whole enum (RPython
        // has no enum type; a fieldless enum is a named `int`).
        // Fold to `ConstInt(discriminant)` so the constructed value
        // is an int end-to-end, matching the int-modeled param /
        // field read / `Discriminant` sites.  Emitting the
        // transparent ctor (a `Ref` result) here would disagree
        // with the int coloring `tyref_to_value_type` gives the
        // destination and re-introduce the classdef-less base.
        if let Some(tag) = self.aggregate_fieldless_enum_discriminant(kind) {
            return Ok(AggregateShape::Discriminant(tag));
        }
        // A fieldless struct, unit `()`, or a borrow of one is
        // `lltype.Void` (`getkind == 'void'`): no constructor and no
        // register, only the `Constant(None, Void)` definition.
        // Fieldless enums stay discriminant integers and are handled
        // above. A named empty class is still an instance
        // (`Bookkeeper.immutablevalue` / `InstanceRepr.convert_const`);
        // the constructor is interned to that prebuilt constant.
        if tyref_is_void_zst(dest_ty, self.llbc)
            && !tyref_keeps_empty_class_instance(dest_ty, self.llbc, self.tombstoned_leaves)
        {
            return Ok(AggregateShape::Void);
        }
        // `Option<E>` over a dense fieldless enum E is the same
        // scalar plus one reserved `None` value.  Constructing
        // `Some(e)` therefore aliases e; constructing `None` emits
        // the reserved niche.  No aggregate allocation or tag write
        // exists in Rust's physical representation.
        if let Some(niche) = tyref_option_fieldless_niche(dest_ty, self.llbc) {
            return match n_operands {
                0 => Ok(AggregateShape::Discriminant(niche.none_tag)),
                1 => Ok(AggregateShape::Operand {
                    index: 0,
                    transparent: false,
                }),
                n => Err(format!("Option<fieldless-enum> aggregate has {n} operands")),
            };
        }
        // `Option<NonZero<_>>`: `None` is the integer 0 and `Some` is the
        // nonzero word. A transparent ctor would be a Ref, which cannot
        // meet the int bank `tyref_option_nonzero_scalar` assigns.
        if let Some(kind) = option_nonzero_word_kind(dest_ty, self.llbc) {
            return match n_operands {
                0 => Ok(match kind {
                    ValueType::Unsigned => AggregateShape::UnsignedZero,
                    _ => AggregateShape::Discriminant(0),
                }),
                1 => Ok(AggregateShape::Operand {
                    index: 0,
                    transparent: false,
                }),
                n => Err(format!("Option<NonZero> aggregate has {n} operands")),
            };
        }
        // A one-word niche `Option` is represented by its payload
        // pointer, so
        // constructing it names either the null pointer (`None`) or
        // the wrapped non-null pointer (`Some(p)`) directly — not a
        // two-word aggregate with a `__discriminant` tag.  Fold `None`
        // (zero-operand variant) to the `null_mut()` builtin call and
        // `Some(p)` to the identity on its single pointer operand, so a
        // constructed value is a maybe-null pointer end-to-end and both
        // variants unify at the function return merge.  The null is a
        // `null_mut()` call (a classdef-less nullable `SomeInstance`,
        // the same lattice a `*mut T` value carries) rather than a
        // `ConstRefNull` (a GCREF `_ptr`): the return merge unions the
        // `None` value with the pointer-valued `Some` branch, and the
        // two must share a universe — a bare `ConstRefNull` would
        // annotate `SomePtr` and fail to union with the `SomeInstance`
        // pointer.  (RPython has no `Option`; a maybe-null `Ptr` is the
        // pointer itself, null for the absent case.)
        if self.tyref_is_niche_option_ptr(dest_ty) {
            return Ok(if n_operands == 0 {
                AggregateShape::NicheNull
            } else {
                AggregateShape::Operand {
                    index: 0,
                    transparent: false,
                }
            });
        }
        // A `repr(transparent)` aggregate is its one sized field.
        // Zero-sized markers are not stored, so the operand is that
        // field rather than whichever single operand arrived. An
        // opaque newtype, or a wrapper whose only field is zero-sized,
        // still has one operand and stays the identity. Constructing
        // a separate instance would split one machine word into
        // incompatible scalar and reference annotations at later
        // merges. The bank comes from the tombstoned classifier.
        // `UnsafeCell<T>` is `#[repr(transparent)]` but the foreign decl
        // is `Opaque`, so the scalar-transparent table never records
        // it. One operand is the cell word.
        if n_operands == 1 && tyref_is_unsafecell(dest_ty, self.llbc) {
            return Ok(AggregateShape::Operand {
                index: 0,
                transparent: false,
            });
        }
        if tyref_transparent_inner_value_type(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
        .is_some()
        {
            if let Some((index, _)) = tyref_transparent_nonzst_field(dest_ty, self.llbc) {
                if index >= n_operands {
                    return Err(format!("transparent aggregate missing field {index}"));
                }
                return Ok(AggregateShape::Operand {
                    index,
                    transparent: true,
                });
            } else if n_operands == 1 {
                return Ok(AggregateShape::Operand {
                    index: 0,
                    transparent: true,
                });
            }
        }
        // Resolve the user-defined owner + field names and types from the
        // Adt kind's `type_id` when possible.  Charon encodes
        // `AggregateKind::Adt(type_id, variant_idx, ..)` as
        // `{"Adt": [type_id, variant_idx, ..]}`; struct variants
        // use `variant_idx = null`, enum variants index into the
        // `TypeDeclKind::Enum` variant list.
        let resolved = self.resolve_aggregate_adt(kind, Some(dest_ty));
        let (owner_path, ctor_name, field_rows, owner_id, adt_is_struct, enum_variant_tag) =
            match resolved {
                Some((owner_path, ctor_name, field_rows, owner_id, is_struct, tag)) => (
                    owner_path,
                    ctor_name,
                    field_rows,
                    Some(owner_id),
                    is_struct,
                    tag,
                ),
                None => {
                    // Synthetic placeholders for non-Adt aggregates
                    // (`Tuple`, `Array`, `Closure`) — they have no
                    // user-defined class to resolve into.  A non-empty
                    // tuple or array carries its per-shape `<…>` suffix
                    // so its `__pos_N` attrs do not collide with other
                    // shapes on one global class; the suffix matches
                    // `positional_aggregate_owner` (Site-A reads) and
                    // `tyref_positional_aggregate_suffix` (Site-B reads).
                    // The per-shape `<…>` suffix is rendered from the
                    // destination place type, not the `AggregateKind` head:
                    // Charon's `AggregateKind::Adt(Tuple, …)` carries no
                    // element `types`, so keying off it would spell a bare
                    // `Tuple` on the write while the `.N` projection reads
                    // (which do see `place.ty`'s element types) spell the
                    // suffixed owner — a write/read owner split.  `dest_ty`
                    // is that same `place.ty`, so both sides agree.
                    let leaf = format!(
                        "{}{}",
                        aggregate_ctor_name(kind),
                        tyref_positional_aggregate_suffix(dest_ty, self.llbc)
                    );
                    let positional = (0..n_operands)
                        .map(|i| (format!("__pos_{i}"), String::new(), None))
                        .collect();
                    // Not an Adt at all, so there is no struct decl to
                    // stand behind an allocation rewrite.
                    (Vec::new(), leaf, positional, None, false, None)
                }
            };
        let result_owner = if owner_path.is_empty() {
            ctor_name.clone()
        } else {
            format!("{}::{}", owner_path.join("::"), ctor_name)
        };
        // Emit the transparent ctor with empty args so the
        // annotator's `ClassDesc::pycall` `args.fixedunpack(0)`
        // check (`classdesc.rs`, mirroring upstream
        // `classdesc.py`) succeeds for classes whose
        // `__init__` is not registered with the bookkeeper —
        // the operand values flow through the FieldWrite chain
        // instead.  A named struct's constructor is the
        // malloc marker; [`scalar_replace_named_struct_aggregates`]
        // replaces it with per-field SSA after boxing fusion.
        let target = if owner_path.is_empty() {
            CallTarget::synthetic_transparent_ctor(ctor_name.clone())
        } else if adt_is_struct {
            // Resolved to a `TypeDeclKind::Struct`, so the value is
            // its field writes and nothing else; `jtransform` may
            // lower the constructor to a bare allocation.
            CallTarget::synthetic_transparent_struct_ctor(owner_path.clone(), ctor_name.clone())
        } else if let Some(tag) = enum_variant_tag {
            // A resolved enum variant records its declaration
            // index so a payload-less variant folded to a
            // prebuilt singleton keeps its discriminant.
            CallTarget::synthetic_transparent_enum_variant_ctor(
                owner_path.clone(),
                ctor_name.clone(),
                tag,
            )
        } else {
            CallTarget::synthetic_transparent_ctor_with_owner(owner_path.clone(), ctor_name.clone())
        };
        // Every operand is surfaced through a separate FieldWrite so
        // the field-to-value binding survives into the
        // codewriter / annotator.  Field names default to
        // `__pos_<i>` when the resolver could not project a real
        // schema entry (tuple aggregates, deduplicated types
        // not in the LLBC's local table).
        let fields = (0..n_operands)
            .map(|i| {
                let (name, field_ty) = field_rows
                    .get(i)
                    .map(|(name, field_ty, _)| (name.clone(), field_ty.clone()))
                    .unwrap_or_else(|| (format!("__pos_{i}"), String::new()));
                let declared_ty = field_rows.get(i).and_then(|(_, _, ty)| ty.as_ref());
                AggregateField {
                    name,
                    // `_names_without_voids()` (`lltype.py`) /
                    // `heaptracker.py` `get_fielddescr_index_in`: a
                    // zero-sized field occupies no storage and gets no slot,
                    // so no write is generated for it.
                    void: crate::codewriter::call::get_type_flag(
                        &field_ty,
                        &self.raw_alloc_flavor_structs,
                    )
                    .1 == majit_ir::value::Type::Void
                        || declared_ty.is_some_and(|ty| tyref_is_void_zst(ty, self.llbc)),
                    narrow_root: self.typed_ref_field_narrow_root(declared_ty),
                    window: declared_ty.is_some_and(|ty| tyref_is_u8_window(ty, self.llbc))
                        || majit_ir::descr::is_shared_u8_slice_spelling(&field_ty)
                        || majit_ir::descr::is_option_of_u8_window_spelling(&field_ty),
                }
            })
            .collect();
        Ok(AggregateShape::Ctor(AggregateCtor {
            target,
            result_owner,
            owner_id,
            owner_path,
            ctor_name,
            fields,
        }))
    }

    /// The shape a call of the constructor function item `fd` builds, for a
    /// call whose result type is `dest_ty` (the instantiated ADT).  `None`
    /// when `fd` is not a constructor Charon synthesised, or its ADT or
    /// variant does not resolve.
    pub(in crate::front::mir) fn constructor_fn_item_shape(
        &self,
        fd: &FunDecl,
        dest_ty: &TyRef,
    ) -> Option<AggregateShape> {
        if !fd.is_adt_constructor() {
            return None;
        }
        let dest = self
            .tyref_peel_ref_to_pointee(dest_ty)
            .unwrap_or_else(|| dest_ty.clone());
        let node = strip_ty_indirections(tyref_node(&dest, self.llbc)?, self.llbc)?;
        let head = node.as_object()?.get("Adt")?.clone();
        let td = self
            .llbc
            .type_by_id(type_decl_ref_adt_id(head.as_object()?)?)?;
        // Copied fields are for projection. Constructing the import still
        // builds the opaque aggregate this crate lowered before the copy.
        if td.published_struct_copy {
            return None;
        }
        // Charon names a tuple-variant constructor `<Enum>::<Variant>` and a
        // tuple-struct constructor `<Struct>`.
        let variant_idx = match &td.kind {
            TypeDeclKind::Enum(variants) => {
                let leaf = fd.item_meta.name_path();
                let leaf = leaf.rsplit("::").next()?;
                Some(variants.iter().position(|v| v.name == leaf)?)
            }
            TypeDeclKind::Struct(_) => None,
            _ => return None,
        };
        let kind = serde_json::json!({ "Adt": [head, variant_idx, null] });
        self.aggregate_shape(&kind, &dest, fd.signature.inputs.len())
            .ok()
    }

    pub(in crate::front::mir) fn resolve_aggregate_adt(
        &self,
        kind: &serde_json::Value,
        dest_ty: Option<&TyRef>,
    ) -> Option<(
        Vec<String>,
        String,
        Vec<(String, String, Option<TyRef>)>,
        majit_ir::descr::StructId,
        bool,
        Option<i64>,
    )> {
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        // `AggregateKind::Adt` head: either a bare `type_id` u64 or a
        // full `TypeDeclRef` object `{"generics": …, "id": // <type_id> | "Tuple" | {"Builtin": …}}` (the object shape is
        // what Charon emits for generic-instantiated types such as
        // `Result<T, E>`).  The bare-u64 read alone made every generic
        // aggregate fall through to the `"Adt"` ctor-name fallback,
        // collapsing all such constructors onto one identity (variant +
        // owner lost).  A `"Tuple"` / builtin atom `id` has no
        // user-defined class and falls through to `None` (the
        // Tuple/Array placeholder).
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => type_decl_ref_adt_id(head.as_object()?)?,
        };
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64);
        let td = self.llbc.type_by_id(type_id)?;
        if td.published_struct_copy {
            return None;
        }
        let name_path = td.item_meta.name_path();
        let head_adt = head.as_object();
        let mut segments: Vec<String> = name_path.split("::").map(str::to_string).collect();
        let type_leaf = segments.pop().unwrap_or_default();
        let owner_path = segments;
        match (&td.kind, variant_idx) {
            (TypeDeclKind::Struct(fields), None) | (TypeDeclKind::Struct(fields), Some(_)) => {
                // Substitute dest_ty / head generics so a field whose
                // declaration is a type variable (`RDict.hash_builder: S`)
                // sees `BuildHasherDefault<H>` and `_names_without_voids`
                // can drop the ZST (`rordereddict.py` has no hasher field).
                let dest_adt = dest_ty.and_then(|ty| tyref_adt_map(ty, self.llbc));
                let inst_adt = match head_adt {
                    Some(h) if !render_adt_type_args(h, self.llbc, 0).is_empty() => Some(h),
                    _ => dest_adt.or(head_adt),
                };
                let inst_args = inst_adt
                    .map(|h| render_adt_type_args(h, self.llbc, 0))
                    .unwrap_or_default();
                let field_rows: Vec<(String, String, Option<TyRef>)> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let field_ty = if inst_args.is_empty() {
                            tyref_to_ast_string(&f.ty, self.llbc)
                        } else {
                            substitute_field_type(
                                &f.ty,
                                &inst_args,
                                self.llbc,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            )
                        };
                        (
                            f.name.clone().unwrap_or_else(|| {
                                majit_charon_reader::ullbc::positional_field_name(i)
                            }),
                            field_ty,
                            Some(clone_tyref(&f.ty)),
                        )
                    })
                    .collect();
                let template = majit_ir::descr::StructId::from_canonical(&decl_path_for_tombstone(
                    &name_path,
                    self.tombstoned_leaves,
                ));
                // A generic struct in a harvested foreign-body crate is
                // one class per instantiation (`bookkeeper.py`
                // `getuniqueclassdef`); `Entry<K, V>` is the local-crate
                // split. An unsuffixed template unions scalar fields
                // across instantiations.
                //
                // The aggregate head is often a bare type_id (same as the
                // enum arm below). Prefer the destination place's
                // TypeDeclRef when the head has no type arguments.
                let dest_adt = dest_ty.and_then(|ty| tyref_adt_map(ty, self.llbc));
                let inst_adt = concrete_instantiation_adt(head_adt, dest_adt, self.llbc);
                let type_leaf = inst_adt
                    .and_then(|h| struct_instantiation_suffix(&name_path, h, self.llbc))
                    .map(|suffix| majit_ir::descr::with_instantiation_suffix(&type_leaf, &suffix))
                    .unwrap_or(type_leaf);
                Some((
                    owner_path,
                    type_leaf,
                    field_rows,
                    concrete_adt_struct_id(template, inst_adt, self.llbc),
                    true,
                    None,
                ))
            }
            (TypeDeclKind::Enum(variants), Some(idx)) => {
                let v = variants.get(idx as usize)?;
                let mut variant_owner = owner_path;
                // A reference-payload workspace enum instantiation
                // constructs into its per-instantiation variant class
                // (`Result<Tuple>::Ok`), the same spelling the receiver
                // type projects and the discriminant narrowing mints, so
                // the constructor's `setattr` and the narrowing land on
                // one classdef per instantiation (no cross-instantiation
                // payload union).
                //
                // The aggregate head is often a bare type_id, or a
                // TypeDeclRef whose `generics` were dropped. The
                // destination place type still carries the instantiated
                // ADT (TyRef generics / `item_meta.instantiation`). Prefer
                // that TypeDeclRef when the head has no type arguments,
                // rather than intern the bare template (`Option::Some`)
                // and union every payload onto one `__pos_0`.
                let dest_adt = dest_ty.and_then(|ty| tyref_adt_map(ty, self.llbc));
                let inst_adt = concrete_instantiation_adt(head_adt, dest_adt, self.llbc);
                let suffix = inst_adt.and_then(|h| adt_head_instantiation_suffix(h, self.llbc));
                let leaf = match suffix {
                    Some(suffix) => majit_ir::descr::with_instantiation_suffix(&type_leaf, &suffix),
                    None => type_leaf,
                };
                variant_owner.push(leaf);
                // Same dest/head substitution as the Struct arm: a field
                // whose declaration is a type variable (`Option<T>::Some`
                // `__pos_0: T`) has to see `()` / `i64` / … so
                // `_names_without_voids` can drop a ZST payload
                // (`widen_unit_return_to_void`) and a niche payload can
                // intern as the pointer word rather than the template
                // `??TypeVar`.
                let inst_args = inst_adt
                    .map(|h| render_adt_type_args(h, self.llbc, 0))
                    .unwrap_or_default();
                let field_rows: Vec<(String, String, Option<TyRef>)> = v
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let field_ty = if inst_args.is_empty() {
                            tyref_to_ast_string(&f.ty, self.llbc)
                        } else {
                            substitute_field_type(
                                &f.ty,
                                &inst_args,
                                self.llbc,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            )
                        };
                        (
                            f.name.clone().unwrap_or_else(|| {
                                majit_charon_reader::ullbc::positional_field_name(i)
                            }),
                            field_ty,
                            Some(clone_tyref(&f.ty)),
                        )
                    })
                    .collect();
                let template = majit_ir::descr::StructId::from_canonical(&format!(
                    "{}::{}",
                    decl_path_for_tombstone(&name_path, self.tombstoned_leaves),
                    v.name
                ));
                Some((
                    variant_owner,
                    v.name.clone(),
                    field_rows,
                    concrete_adt_struct_id(template, inst_adt, self.llbc),
                    false,
                    Some(idx as i64),
                ))
            }
            _ => None,
        }
    }

    /// Integer width/sign class of the single type argument of `head` when
    /// `head` names `type_path` (`core::ops::range::Range` / `RangeInclusive`)
    /// over an integer scalar.  `Some((narrow, unsigned))`: `narrow` is true
    /// when the element is narrower than 64 bits; `unsigned` is true for an
    /// unsigned integer element. Threaded to `RangeSiteKind::Inclusive` next
    /// to `narrow`; do not re-derive signedness from path names later.
    pub(in crate::front::mir) fn int_range_head_narrow(
        &self,
        head: &serde_json::Map<String, serde_json::Value>,
        type_path: &str,
    ) -> Option<(bool, bool)> {
        let is_match = type_decl_ref_adt_id(head)
            .and_then(|id| self.llbc.type_by_id(id))
            .is_some_and(|td| td.item_meta.name_path() == type_path);
        if !is_match {
            return None;
        }
        let lit = type_decl_ref_generics(head, self.llbc)
            .and_then(|g| g.as_object())
            .and_then(|g| g.get("types"))
            .and_then(|t| t.as_array())
            .and_then(|t| t.first())
            .and_then(|n| strip_ty_wrappers(n, self.llbc))
            .and_then(|e| e.as_object())
            .and_then(|m| m.get("Scalar"))
            .and_then(|l| l.as_object())?;
        let int = lit.get("Integer").and_then(serde_json::Value::as_object)?;
        match (
            int.get("Signed").and_then(serde_json::Value::as_str),
            int.get("Unsigned").and_then(serde_json::Value::as_str),
        ) {
            (Some("I8" | "I16" | "I32"), _) => Some((true, false)),
            (_, Some("U8" | "U16" | "U32")) => Some((true, true)),
            (Some("I64" | "Isize"), _) => Some((false, false)),
            (_, Some("U64" | "Usize")) => Some((false, true)),
            _ => None,
        }
    }

    /// True when the `AggregateKind::Adt` head constructs a
    /// `core::ops::range::Range` whose element type is an integer that
    /// fits a machine word (anything but `i128` / `u128`) — the int-range
    /// shape `front::range_iter` diverts to the orthodox `range()` builtin.
    /// The element is read from the head's `generics.types[0]` (present even
    /// when the bounds are literal consts); a non-int element, a non-object
    /// head (bare `type_id`, no generics), or a non-`Range` owner declines,
    /// leaving the ordinary ADT ctor path.
    pub(in crate::front::mir) fn aggregate_head_is_int_range(
        &self,
        kind: &serde_json::Value,
    ) -> bool {
        kind.as_object()
            .and_then(|m| m.get("Adt"))
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_object())
            .and_then(|head| self.int_range_head_narrow(head, "core::ops::range::Range"))
            .is_some()
    }

    /// The `TypeDeclRef` (`{"id", "generics", "builtin"}`) of a field
    /// projection's base type, behind the dedup / hash-cons indirections.
    pub(in crate::front::mir) fn field_base_adt<'s>(
        &'s self,
        base_ty: &'s TyRef,
    ) -> Option<&'s serde_json::Map<String, serde_json::Value>> {
        strip_ty_wrappers(tyref_node(base_ty, self.llbc)?, self.llbc)?
            .as_object()?
            .get("Adt")?
            .as_object()
    }

    /// Resolve a `Field` projection `[variant, index]` off the place whose
    /// type is `base_ty`.  The projection names only the variant and the
    /// field index; the owning ADT, with the generic arguments of this use,
    /// is the base place's type.
    ///
    /// For a struct the owner_root is the LLBC TypeDecl's leaf name
    /// (`PyFrame` from `pyre_interpreter::pyframe::PyFrame`) so the
    /// downstream `struct_fields` registry resolves with the same leaf
    /// key.  For an enum the read is a variant downcast, so the
    /// owner_root is the variant-qualified `{enum_leaf}::{variant}` key
    /// (the variant subclass carries its own fields at the exact
    /// per-variant offset).
    pub(in crate::front::mir) fn resolve_adt_field(
        &self,
        base_ty: &TyRef,
        payload: &serde_json::Value,
    ) -> Option<(
        String,
        String,
        TyRef,
        Option<majit_ir::descr::StructId>,
        Option<(u32, u32)>,
    )> {
        let (variant_idx, field_idx) = indexed_field_parts(payload)?;
        let host_index = host_field_index(variant_idx, field_idx);
        let head = self.field_base_adt(base_ty)?;
        let head_adt = Some(head);
        let type_id = type_decl_ref_adt_id(head)?;
        let td = self.llbc.type_by_id(type_id)?;
        // The full `name_path()` is in hand here, so mint the owning
        // type's object-identity token from the crate-stripped qualified
        // path (`module::Type` / `module::Enum::Variant`) — the same key
        // the layout layer is registered under.  `owner_root` stays the
        // bare leaf its non-layout consumers expect; `owner_id` carries
        // the collision-free identity for the layout layer.
        let name_path = td.item_meta.name_path();
        // `owner_root` is the annotation-side classdef key: a
        // reference-payload workspace enum instantiation reads its field
        // off the per-instantiation variant class (`Result<Tuple>::Ok`),
        // matching the receiver / constructor projection. `owner_id` is the
        // independent layout-side identity and preserves every concrete type
        // argument, including primitive payloads for which the annotator does
        // not split classdefs.
        let owner_leaf = name_path.rsplit("::").next().unwrap_or("").to_string();
        // Duplicate-leaf hardening drops the shared `closure` /
        // `closure#N` alias from the field registry.  Field reads of a
        // capture must key the full crate-stripped path the registry
        // kept — the same spelling [`tyref_input_class_root`] uses for
        // the env parameter.
        // A tombstoned leaf uses the same spelling as
        // [`adt_node_class_root_with`]: the crate-stripped path, or the
        // full declaration path when stripping yields the leaf itself.
        let owner_root = if majit_charon_reader::ullbc::is_closure_leaf(&owner_leaf) {
            strip_crate_prefix(&name_path)
        } else {
            let owner_base = if self.tombstoned_leaves.contains(&owner_leaf) {
                tombstoned_owner_spelling(&name_path, &owner_leaf, self.tombstoned_leaves)
            } else {
                owner_leaf
            };
            match adt_head_instantiation_suffix(head, self.llbc) {
                Some(suffix) => majit_ir::descr::with_instantiation_suffix(&owner_base, &suffix),
                None => match struct_instantiation_suffix(&name_path, head, self.llbc) {
                    Some(suffix) => {
                        majit_ir::descr::with_instantiation_suffix(&owner_base, &suffix)
                    }
                    None => owner_base,
                },
            }
        };
        if variant_idx.is_none()
            && let Some(fields) = type_decl_struct_fields(td, self.llbc)
        {
            // `type Alias = Struct<…>` is an Alias whose fields live on
            // the generic struct. Keep the alias as owner so
            // `exact_layouts` uses the monomorph's rustc offsets.
            let f = fields.get(field_idx)?;
            let name = f
                .name
                .clone()
                .unwrap_or_else(|| majit_charon_reader::ullbc::positional_field_name(field_idx));
            let ty = clone_tyref(&f.ty);
            let template = majit_ir::descr::StructId::from_canonical(&decl_path_for_tombstone(
                &name_path,
                self.tombstoned_leaves,
            ));
            let owner_id = Some(concrete_adt_struct_id(template, head_adt, self.llbc));
            return Some((owner_root, name, ty, owner_id, host_index));
        }
        match (&td.kind, variant_idx) {
            (TypeDeclKind::Enum(variants), Some(vidx)) => {
                let variant = variants.get(vidx as usize)?;
                let f = variant.fields.get(field_idx)?;
                let name = f.name.clone().unwrap_or_else(|| {
                    majit_charon_reader::ullbc::positional_field_name(field_idx)
                });
                let ty = clone_tyref(&f.ty);
                // owner_root = the variant subclass `{enum_leaf}::{variant}`
                // — the read resolves the variant's own field at its exact
                // enum-relative offset (`variant_layouts[vidx].field_offsets`,
                // registered under this key).  The downcast statically fixes
                // the variant.
                let variant_owner = format!("{owner_root}::{}", variant.name);
                let template = majit_ir::descr::StructId::from_canonical(&format!(
                    "{}::{}",
                    decl_path_for_tombstone(&name_path, self.tombstoned_leaves),
                    variant.name
                ));
                let owner_id = Some(concrete_adt_struct_id(template, head_adt, self.llbc));
                Some((variant_owner, name, ty, owner_id, host_index))
            }
            _ => None,
        }
    }

    /// Scalar field of an opaque dependency struct.
    ///
    /// The declaration keeps a layout and drops the field list, so
    /// [`Self::resolve_adt_field`] returns `None`. A copy scalar of one,
    /// two, four, or eight bytes still has a byte offset. Pointer fields
    /// stay on the alias: their place type is a reference, not a scalar.
    /// `i128` / `u128` / `f32` have no word-sized field load.
    pub(in crate::front::mir) fn opaque_scalar_field(
        &self,
        base_ty: &TyRef,
        place_ty: &TyRef,
        payload: &serde_json::Value,
    ) -> Option<(String, String, ValueType, crate::model::ScalarFieldWord)> {
        let (variant_idx, field_idx) = indexed_field_parts(payload)?;
        if variant_idx.is_some() {
            return None;
        }
        let head = self.field_base_adt(base_ty)?;
        let type_id = type_decl_ref_adt_id(head)?;
        let td = self.llbc.type_by_id(type_id)?;
        if !matches!(td.kind, TypeDeclKind::Opaque) {
            return None;
        }
        let target = std::env::var("TARGET").unwrap_or_default();
        let layout = td.layout_for_target(self.llbc, &target)?;
        if layout.variant_layouts.len() != 1 || layout.discriminant_offset().is_some() {
            return None;
        }
        let offset = usize::try_from(layout.struct_field_offset(field_idx)?).ok()?;
        let place = strip_ty_indirections(tyref_node(place_ty, self.llbc)?, self.llbc)?;
        // A fieldless (C-like) enum is the discriminant integer — RPython
        // has no enum type — including when the defining crate is Opaque
        // here.  Leaving the projection to collapse onto the parent struct
        // makes `eq` compare that instance with a `ConstInt` tag
        // (`pair(rtype_eq) not implemented for (InstanceRepr, IntegerRepr)`).
        let (ty, size, signed) = if json_ty_is_copy_scalar(place, self.llbc) {
            json_ty_raw_store_descr(place, self.llbc)?
        } else {
            fieldless_enum_raw_store_descr(place_ty, self.llbc)?
        };
        let width = match (&ty, signed, size) {
            (ValueType::Int, true, 1 | 2 | 4 | 8) => crate::model::ScalarFieldWidth::Signed,
            (ValueType::Int | ValueType::Unsigned, false, 1 | 2 | 4 | 8) => {
                crate::model::ScalarFieldWidth::Unsigned
            }
            (ValueType::Float, _, 8) => crate::model::ScalarFieldWidth::Float,
            _ => return None,
        };
        let owner = td.item_meta.name_path();
        if owner.is_empty() {
            return None;
        }
        let name = self.struct_fields.map_or_else(
            || format!("__pos_{field_idx}"),
            |reg| opaque_struct_field_name(&owner, field_idx, reg),
        );
        Some((
            owner,
            name,
            ty,
            crate::model::ScalarFieldWord {
                offset,
                size,
                width,
            },
        ))
    }

    /// Decode an inline-Field `dyn Trait` call operand into the
    /// `(trait_root, method_name)` pair carried as the `family_key` of the
    /// [`OpKind::IndirectCall`] this lowers to.  `rpbc::lower_indirect_calls`
    /// consumes that key to attach the same `c_graphs` PBC family
    /// `FunctionReprBase.call` appends, then clears it.
    ///
    /// The inline dispatch reads the method fn-ptr straight out of the
    /// receiver's vtable — the operand's terminal projection is
    /// `Field[Adt(vtable_id), slot]` off `(*recv).vtable`, where the
    /// vtable type is the `<Trait>::{vtable}` struct and the slot field
    /// is named `method_<name>`.  Returns `None` for any other operand
    /// shape (the one-hop `Option<fn-ptr>` deref and `call_once` closure
    /// shims are bare locals / non-`Field[Adt]` projections), which keeps
    /// those on the synthetic `__dyn_call` path.
    pub(in crate::front::mir) fn dyn_indirect_target(
        &self,
        dyn_operand: &Operand,
    ) -> Option<(String, String)> {
        let place = match dyn_operand {
            Operand::Copy(p) | Operand::Move(p) => p,
            Operand::Const(_) => return None,
        };
        let PlaceKind::Projection(base, elem) = &place.kind else {
            return None;
        };
        let ProjectionElem::Tagged(v) = elem else {
            return None;
        };
        let field_payload = v.as_object()?.get("Field")?;
        // The terminal container must be an `Adt` (the vtable struct).
        let vtable_id = type_decl_ref_adt_id(self.field_base_adt(&base.ty)?)?;
        // `resolve_adt_field` names the slot `method_<name>`; strip the
        // prefix for the trait method leaf.
        let (_owner, field_name, _ty, _id, _) = self.resolve_adt_field(&base.ty, field_payload)?;
        let method_name = field_name.strip_prefix("method_")?.to_string();
        // The vtable type is spelled `<path>::<Trait>::{vtable}`; the
        // trait_root the family lookup keys on is the bare trait leaf
        // (matching `register_trait_method`, which registers under
        // `SemanticFunction.trait_root` = the trait's leaf name).
        let vtable_path = self.llbc.type_by_id(vtable_id)?.item_meta.name_path();
        let trait_path = vtable_path
            .strip_suffix("::{vtable}")
            .unwrap_or(vtable_path.as_str());
        let trait_root = trait_path.rsplit("::").next()?.to_string();
        Some((trait_root, method_name))
    }

    /// The unwrapped ADT body `Value` a [`TyRef`] resolves to, following
    /// `Deduplicated` / `Value` wrapper layers.  The returned
    /// object carries the `{"Adt": {"id": …, "generics": …}}` shape both
    /// [`inline_adt_def_id`] and [`adt_head_instantiation_suffix`] decode.
    /// `None` for non-ADT shapes.
    pub(in crate::front::mir) fn tyref_adt_body<'t>(
        &self,
        ty: &'t TyRef,
    ) -> Option<&'t serde_json::Value>
    where
        'a: 't,
    {
        let mut v = self.tyref_body(ty)?;
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
            break;
        }
        Some(v)
    }

    /// The canonical owner of a prebuilt ref whose declared type is a known
    /// struct. Mirrors Bookkeeper.immutablevalue's SomeInstance classification
    /// and InstanceRepr.convert_const_exact's per-instance representation.
    /// Field-bearing instances need the same owner as zero-field ones.
    ///
    /// Opaque declarations do not prove the pointed-to layout. In particular,
    /// the OnceLock globals used to initialize Python singletons are opaque in
    /// LLBC, and their registered refs identify the initialized Python objects,
    /// not the OnceLock containers. Keep those on their existing raw-ref path.
    ///
    /// The name is crate-stripped ([`strip_crate_prefix`]) so it matches
    /// the canonical `module::Leaf` spelling every other classdef path keys
    /// on (`STRUCT_ORIGIN_REGISTRY` / `canonical_struct_name`, the struct
    /// field registry, and the `register_trait_family` impl subclasses).
    /// The raw `name_path()` carries the crate segment (`pyre_object::…`),
    /// which `canonical_struct_name` passes through verbatim, minting a
    /// SEPARATE `ClassDef` from the trait-family subclass — so the singleton
    /// instance would have no `basedef` and two sibling strategy singletons
    /// (`EmptyDictStrategy` / `EmptyKwargsDictStrategy`) `commonbase` to
    /// nothing at a `mergeinputargs` join (`UnionError: no common base`).
    pub(in crate::front::mir) fn refs_static_struct_root(&self, ty: &TyRef) -> Option<String> {
        let v = self.tyref_adt_body(ty)?;
        let def_id = inline_adt_def_id(v)?;
        let td = self.llbc.type_by_id(def_id)?;
        if td.published_struct_copy {
            return None;
        }
        match &td.kind {
            TypeDeclKind::Struct(_) => Some(strip_crate_prefix(&td.item_meta.name_path())),
            _ => None,
        }
    }

    /// The per-instantiation classdef root of the ADT a [`TyRef`]
    /// resolves to: the fully-qualified `name_path()` plus the `<…>`
    /// instantiation suffix ([`adt_head_instantiation_suffix`]) when the
    /// ADT is a reference-payload enum whose args all split.  This is the
    /// same spelling the receiver-type ([`adt_node_class_root_with`]),
    /// constructor ([`resolve_aggregate_adt`]) and field-read
    /// ([`resolve_adt_field`]) sites project, so a `Discriminant` read
    /// keys the SAME per-instantiation base class the constructor wrote
    /// (`Option<Result<…>>` rather than the bare `Option`) and the
    /// discriminant narrowing mints matching variant subclasses.  Bare
    /// leaf (no suffix) for non-enum / primitive-payload heads.  `None`
    /// for non-ADT shapes.
    pub(in crate::front::mir) fn tyref_adt_class_root(&self, ty: &TyRef) -> Option<String> {
        let v = self.tyref_adt_body(ty)?;
        let def_id = inline_adt_def_id(v)?;
        let name_path = self.llbc.type_by_id(def_id)?.item_meta.name_path();
        let adt = v.as_object()?.get("Adt")?.as_object()?;
        match adt_head_instantiation_suffix(adt, self.llbc) {
            Some(suffix) => Some(majit_ir::descr::with_instantiation_suffix(
                &name_path, &suffix,
            )),
            None => match struct_instantiation_suffix(&name_path, adt, self.llbc) {
                Some(suffix) => Some(majit_ir::descr::with_instantiation_suffix(
                    &name_path, &suffix,
                )),
                None => Some(name_path),
            },
        }
    }

    /// Concrete physical-layout identity of an ADT type. Unlike
    /// [`Self::tyref_adt_class_root`], this always preserves generic arguments:
    /// annotator class splitting is selective, while Rust monomorphization is
    /// not.
    pub(in crate::front::mir) fn tyref_adt_layout_id(
        &self,
        ty: &TyRef,
    ) -> Option<majit_ir::descr::StructId> {
        let value = self.tyref_adt_body(ty)?;
        let def_id = inline_adt_def_id(value)?;
        let name_path = self.llbc.type_by_id(def_id)?.item_meta.name_path();
        let template = majit_ir::descr::StructId::from_canonical(&decl_path_for_tombstone(
            &name_path,
            self.tombstoned_leaves,
        ));
        let adt = value.as_object()?.get("Adt")?.as_object();
        Some(concrete_adt_struct_id(template, adt, self.llbc))
    }

    /// `true` when `ty` resolves to a FIELDLESS enum whose discriminant
    /// tag sits at the value's base (byte 0).  Mirrors the
    /// [`Lowering::tyref_adt_name_path`] resolution (dedup / hash-consed
    /// bodies), then checks the enum layout's tag position.  The
    /// `Discriminant` lowering folds the tag read of such an inline enum
    /// field directly into the container at the field offset: a
    /// fieldless enum's whole value IS the tag, so reading the field's
    /// bytes (sized to the enum at `disc_ty` above) yields the
    /// discriminant.  A data-carrying enum or a non-zero tag offset is
    /// rejected so the read never picks up variant payload bytes.
    pub(in crate::front::mir) fn inline_fieldless_enum_field_tag0(&self, ty: &TyRef) -> bool {
        let Some(mut v) = self.tyref_body(ty) else {
            return false;
        };
        loop {
            let Some(obj) = v.as_object() else {
                return false;
            };
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                let Some(next) = self.llbc.dedup_body(id) else {
                    return false;
                };
                v = next;
                continue;
            }
            if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            break;
        }
        let Some(def_id) = inline_adt_def_id(v) else {
            return false;
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return false;
        };
        let target = std::env::var("TARGET").unwrap_or_default();
        type_decl_is_fieldless_enum(td, self.llbc)
            && td
                .layout_for_target(self.llbc, &target)
                .and_then(|l| l.discriminant_offset())
                == Some(0)
    }

    /// The resolved JSON body of a [`TyRef`], following the dedup
    /// table.  Two `TyRef`s denote the same type iff their bodies are
    /// structurally equal (the dedup table only dedupes identical
    /// bodies, so mixed inline/dedup spellings still compare equal).
    pub(in crate::front::mir) fn tyref_body<'t>(
        &self,
        ty: &'t TyRef,
    ) -> Option<&'t serde_json::Value>
    where
        'a: 't,
    {
        match ty {
            TyRef::Inline { value: (_, v) } => Some(v),
            TyRef::Other(v) => Some(v),
            TyRef::Dedup { id } => self.llbc.dedup_body(*id),
        }
    }

    /// Bit width and unsignedness of a literal integer `TyRef`.
    /// `I128` / `U128` stay `None` so the 128-bit carrier is left alone.
    ///
    /// `Isize` / `Usize` are the target word, not the i64 box. A 4-byte
    /// word is 32 bits so `as usize` truncates (`jtransform.py`
    /// `_int_to_int_cast` when `size2 < sizeof(Signed)`).
    pub(in crate::front::mir) fn literal_int_width(&self, ty: &TyRef) -> Option<(u32, bool)> {
        let word_bytes = crate::layout::target_word_size();
        if let Some(atom) = self.tyref_literal_uint_atom(ty) {
            let bits = crate::front::checked_arith_uint::literal_int_atom_bits(atom, word_bytes)?;
            return Some((bits, true));
        }
        if let Some(atom) = self.tyref_literal_int_atom(ty) {
            let bits = crate::front::checked_arith_uint::literal_int_atom_bits(atom, word_bytes)?;
            return Some((bits, false));
        }
        None
    }

    /// `jtransform.py` `_int_to_int_cast` / `rewrite_op_cast_primitive`.
    /// A word-sized target is an identity retype (`r_uint` / `intmask`).
    /// A narrower unsigned target is `int_and` with the low-byte mask; a
    /// narrower signed target is `int_signext(v, nbytes)`.
    pub(in crate::front::mir) fn materialize_int_cast(
        &mut self,
        mir_bb: usize,
        src: (u32, bool),
        dst: (u32, bool),
        arg: Variable,
    ) -> Option<(Option<OpKind>, Variable)> {
        let action = int_cast_action(src.0, src.1, dst.0, dst.1)?;
        let bb_id = self.emit_bb(mir_bb);
        Some(match action {
            IntCastAction::Alias => (None, arg),
            IntCastAction::RUint => {
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                (
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                                .into_iter()
                                .map(str::to_string)
                                .collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![arg]),
                        result_ty: ValueType::Unsigned,
                    }),
                    res,
                )
            }
            IntCastAction::IntMask => {
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                (
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: ["rpython", "rlib", "rarithmetic", "intmask"]
                                .into_iter()
                                .map(str::to_string)
                                .collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![arg]),
                        result_ty: ValueType::Int,
                    }),
                    res,
                )
            }
            IntCastAction::And(mask) => {
                let mask_var = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(mask_var.clone()),
                    kind: OpKind::ConstUInt(mask),
                });
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                (
                    Some(OpKind::BinOp {
                        op: "and".to_string(),
                        lhs: arg,
                        rhs: mask_var,
                        result_ty: ValueType::Unsigned,
                    }),
                    res,
                )
            }
            IntCastAction::SignExt(nbytes) => {
                let width = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(width.clone()),
                    kind: OpKind::ConstInt(nbytes),
                });
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                (
                    Some(OpKind::BinOp {
                        op: "signext".to_string(),
                        lhs: arg,
                        rhs: width,
                        result_ty: ValueType::Int,
                    }),
                    res,
                )
            }
        })
    }

    /// The `UInt` width atom (`"U8"` / `"U32"` / `"Usize"` …) of a
    /// scalar-typed [`TyRef`], `None` for any non-`UInt` shape.
    pub(in crate::front::mir) fn tyref_literal_uint_atom<'t>(
        &self,
        ty: &'t TyRef,
    ) -> Option<&'t str>
    where
        'a: 't,
    {
        let value = match ty {
            TyRef::Inline { value: (_, v) } => v,
            TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        value
            .as_object()?
            .get("Scalar")?
            .as_object()?
            .get("Integer")?
            .as_object()?
            .get("Unsigned")?
            .as_str()
    }

    /// The variant-qualified owner key `{owner}::{variant}` for the
    /// SUCCESS variant a tagged-pair `__pos_0` payload belongs to
    /// (`Option::Some` at disc 1, `Result::Ok` at disc 0).
    ///
    /// Resolution prefers the variant carrying the declared discriminant.
    /// When that is absent, the variant at `disc` as a source index is
    /// next: `Result::Ok` / `Option::None` occupy 0, `Result::Err` /
    /// `Option::Some` occupy 1. A unit Ok (`Result<(), E>`) has no
    /// payload field, so jumping to the sole payload-carrying sibling
    /// would attribute Err's `__pos_0` to Ok and merge Tuple ∪ OsError
    /// (`result_exc` `unit_ok`, `exceptiontransform.py`).
    ///
    /// For a niche-optimised layout where even the index is out of
    /// range, it falls back to the SOLE payload-carrying variant — the
    /// `__pos_0` payload belongs to it by definition, never to the empty
    /// `None`/`Err` sibling or the enum root (whose only field after the
    /// base/variant split is `__discriminant`).  Resolving to the variant
    /// keeps the write owner in agreement with the variant-qualified
    /// read owner (`resolve_adt_field`).  `None` only when `td` is not
    /// an enum or no single payload variant can be identified.
    pub(in crate::front::mir) fn tagged_pair_payload_owner(
        td: &TypeDecl,
        owner: &str,
        disc: i64,
    ) -> Option<String> {
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return None;
        };
        let v = variants
            .iter()
            .find(|v| v.discriminant_i64() == Some(disc))
            .or_else(|| {
                let idx = usize::try_from(disc).ok()?;
                variants.get(idx)
            })
            .or_else(|| {
                let mut payload_variants = variants.iter().filter(|v| !v.fields.is_empty());
                match (payload_variants.next(), payload_variants.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            })?;
        Some(format!("{owner}::{}", v.name))
    }

    /// The `Int` width atom (`"I8"` / `"I32"` / `"Isize"` …) of a
    /// signed-integer literal type, mirroring [`tyref_literal_uint_atom`]
    /// for the `{"Scalar": {"Integer": {"Signed": "Isize"}}}` shell.  `None` for any
    /// non-signed-literal type.
    pub(in crate::front::mir) fn tyref_literal_int_atom<'t>(&self, ty: &'t TyRef) -> Option<&'t str>
    where
        'a: 't,
    {
        self.tyref_literal_atom(ty, "Int")
    }

    /// The `Float` width atom (`"F32"` / `"F64"`) of a float literal type.
    /// `None` for any non-float-literal type.
    pub(in crate::front::mir) fn tyref_literal_float_atom<'t>(
        &self,
        ty: &'t TyRef,
    ) -> Option<&'t str>
    where
        'a: 't,
    {
        self.tyref_literal_atom(ty, "Float")
    }

    pub(in crate::front::mir) fn tyref_literal_atom<'t>(
        &self,
        ty: &'t TyRef,
        kind: &str,
    ) -> Option<&'t str>
    where
        'a: 't,
    {
        let value = match ty {
            TyRef::Inline { value: (_, v) } => v,
            TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        let scalar = value.as_object()?.get("Scalar")?.as_object()?;
        if kind == "Float" {
            return scalar.get("Float").and_then(serde_json::Value::as_str);
        }
        let sign = if kind == "Int" { "Signed" } else { "Unsigned" };
        scalar.get("Integer")?.as_object()?.get(sign)?.as_str()
    }

    /// Shared tail for the decomposed checked-arithmetic /
    /// infallible-conversion call lowerings: write `disc` and
    /// `payload` into the destination `Option`/`Result` local as a
    /// synthetic aggregate — the same transparent-ctor + `FieldWrite`
    /// chain [`Rvalue::Aggregate`] emits — then bind the destination
    /// local and close the block toward the call's success target.
    ///
    /// This is temporary MIR input for TaggedPairAggregateSite. `finish`
    /// replaces the entire tail with concrete variant construction before
    /// simplification/annotation; a runtime tag requires a branch, not an
    /// enum-base instance carrying subclass fields. Deferring the split until
    /// then preserves lower_framestate's original MIR-successor mapping.
    ///
    /// **Precondition on the caller.** Every caller fails with no live payload —
    /// `Option::None` carries none, and `i32::try_from`'s
    /// `TryFromIntError(())` is the unit. A caller with live failure data must
    /// provide a constructor for that data rather than reuse this tail.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn emit_tagged_pair_aggregate(
        &mut self,
        mir_bb: usize,
        owner: &str,
        payload_owner: &str,
        disc: Variable,
        payload: Variable,
        dest_local: usize,
        target: usize,
    ) -> Result<(), LowerError> {
        self.emit_tagged_pair_aggregate_typed(
            mir_bb,
            owner,
            payload_owner,
            disc,
            payload,
            ValueType::Int,
            dest_local,
            target,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn emit_tagged_pair_aggregate_typed(
        &mut self,
        mir_bb: usize,
        owner: &str,
        payload_owner: &str,
        disc: Variable,
        payload: Variable,
        payload_ty: ValueType,
        dest_local: usize,
        target: usize,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let operation_index = self.graph.block(bb_id).operations.len();
        let mut owner_path = crate::model::split_qualified_path(owner);
        let ctor_name = owner_path.pop().unwrap_or_default();
        let ctor_target = if owner_path.is_empty() {
            CallTarget::synthetic_transparent_ctor(ctor_name.clone())
        } else {
            CallTarget::synthetic_transparent_ctor_with_owner(owner_path, ctor_name)
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.tagged_pair_aggregate_sites
            .push(TaggedPairAggregateSite {
                block: bb_id,
                operation_index,
                result: res.clone(),
                owner: owner.to_string(),
                payload_owner: payload_owner.to_string(),
                disc: disc.clone(),
                payload: payload.clone(),
                payload_ty: payload_ty.clone(),
            });
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: ctor_target,
                args: Vec::new(),
                // Temporary native carrier only. `finish` consumes this
                // recorded tail before any annotator or codewriter sees it;
                // each replacement constructor has its actual variant type.
                result_ty: ValueType::Ref(Some(payload_owner.to_string())),
            },
        });
        // The `__discriminant` tag is always an `i64` (matching the
        // `Rvalue::Discriminant` `FieldRead` and field registration).
        // The success payload keeps the producer's actual low-level type:
        // integer for checked arithmetic / widening, string for the
        // `Wtf8::as_str` validity decomposition.  `__discriminant` keys the
        // root (tag offset 0); `__pos_0` keys the success variant so its exact
        // offset and type match the read.
        for (name, value, field_owner, ty) in [
            ("__discriminant", disc, owner, ValueType::Int),
            ("__pos_0", payload, payload_owner, payload_ty),
        ] {
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: res.clone(),
                    field: crate::model::FieldDescriptor {
                        name: name.to_string(),
                        owner_root: Some(field_owner.to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: LinkArg::Value(value),
                    ty,
                },
            });
        }
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }
}

/// Canonicalise a Charon `BinaryOp` tag (PascalCase + JSON-tagged
/// variants) to the RPython-style snake_case label the codewriter
/// expects.  After this the assembler's `op_kind_to_opname` reaches the
/// already-wired `int_{label}` / `ptr_{label}` keys instead of inventing
/// `int_AddChecked` / `int_BitAnd` shapes that have no blackhole handler.
///
/// Mapping reflects RPython's `jtransform` / `rint` /  `rptr` rewrites:
///   - `Add` / `Sub` / `Mul` plain → `add`/`sub`/`mul` (wrapping arith).
///   - `*Checked` variants → `*_ovf` (overflow-guarded arith, paired
///     with `guard_no_overflow` downstream).
///   - Shift `*Wrap` / `*Checked` collapse onto the canonical
///     `lshift`/`rshift` (RPython treats them identically because
///     shifts cannot overflow into a different repr).
///   - `BitAnd` / `BitOr` / `BitXor` → `and`/`or`/`xor` to match
///     `blackhole.py` canonical bitwise opnames.
///   - Comparisons `Eq` / `Ne` / `Lt` / `Le` / `Gt` / `Ge` pass through
///     as lowercase; the assembler later branches on operand kind
///     (`ii` → `int_eq`, `rr` → `ptr_eq`, …).
pub(in crate::front::mir) fn binop_label(v: &serde_json::Value) -> Result<String, LowerError> {
    if let Some(s) = v.as_str() {
        return Ok(canonical_binop_label(s, None));
    }
    if let Some(obj) = v.as_object()
        && let Some((k, payload)) = obj.iter().next()
    {
        let suffix = match payload {
            serde_json::Value::String(s) => Some(s.as_str()),
            _ => None,
        };
        return Ok(canonical_binop_label(k, suffix));
    }
    Err(LowerError::Schema(format!(
        "BinaryOp op label has unexpected shape: {v}"
    )))
}

/// Charon's `UnaryOp` tag → canonical RPython unary opname.  Plain
/// atoms (`"Neg"`, `"Not"`) share `binop_label`'s mapping; tagged
/// `{"Cast": {...}}` payloads encode the dest-only CastKind and
/// project onto `cast_int_to_float` / `cast_float_to_int` /
/// `cast_int_to_ptr` / `cast_ptr_to_int` per `blackhole.py bhimpl_cast_ptr_to_int`.
/// Cast shapes the JIT models as identity (RawPtr→RawPtr,
/// Scalar Int↔UInt of the same width, Unsize) collapse to `same_as`
/// so the assembler emits the per-kind copy op instead of an unwired
/// `int_unary.*` shape.
/// `true` when a `Rvalue::UnaryOp` op payload is a `Cast` (the JSON
/// object `{"Cast": ..}`) rather than `Neg` / `Not`. Casts alias the
/// operand instead of emitting an `OpKind::UnaryOp`.
pub(in crate::front::mir) fn unary_op_is_cast(v: &serde_json::Value) -> bool {
    v.as_object()
        .and_then(|o| o.keys().next())
        .is_some_and(|k| k == "Cast")
}

pub(in crate::front::mir) fn unary_op_label(v: &serde_json::Value) -> Result<String, LowerError> {
    if let Some(s) = v.as_str() {
        return Ok(canonical_binop_label(s, None));
    }
    let Some(obj) = v.as_object() else {
        return Err(LowerError::Schema(format!(
            "UnaryOp op label has unexpected shape: {v}"
        )));
    };
    let Some((tag, payload)) = obj.iter().next() else {
        return Err(LowerError::Schema("UnaryOp object is empty".into()));
    };
    match tag.as_str() {
        "Cast" => Ok(cast_label_from_payload(payload)),
        _ => {
            let suffix = payload.as_str();
            Ok(canonical_binop_label(tag, suffix))
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum IntCastAction {
    Alias,
    RUint,
    IntMask,
    And(u64),
    SignExt(i64),
}

pub(in crate::front::mir) fn integer_bounds(bits: u32, unsigned: bool) -> (i128, i128) {
    if unsigned {
        (0, (1i128 << bits) - 1)
    } else {
        let half = 1i128 << (bits - 1);
        (-half, half - 1)
    }
}

/// `jtransform.py` `_int_to_int_cast`. `None` for a width the word-sized
/// JIT carrier does not narrow (128-bit).
///
/// The comparison word is the Int box (`Value::Int(i64)`), 64 bits on every
/// target, not the pointer width: on wasm32 an `i64 -> u32` still has to
/// mask and a `u32 -> i32` still has to sign-extend inside that box.
pub(in crate::front::mir) fn int_cast_action(
    src_bits: u32,
    src_unsigned: bool,
    dst_bits: u32,
    dst_unsigned: bool,
) -> Option<IntCastAction> {
    let word_bits = i64::BITS;
    if src_bits == 0 || dst_bits == 0 || src_bits > word_bits || dst_bits > word_bits {
        return None;
    }
    if dst_bits == word_bits {
        return Some(if src_unsigned == dst_unsigned {
            IntCastAction::Alias
        } else if dst_unsigned {
            IntCastAction::RUint
        } else {
            IntCastAction::IntMask
        });
    }
    let (min1, max1) = integer_bounds(src_bits, src_unsigned);
    let (min2, max2) = integer_bounds(dst_bits, dst_unsigned);
    if min2 <= min1 && max1 <= max2 {
        return Some(if src_unsigned == dst_unsigned {
            IntCastAction::Alias
        } else if dst_unsigned {
            IntCastAction::RUint
        } else {
            IntCastAction::IntMask
        });
    }
    if dst_unsigned {
        let mask = if dst_bits == 64 {
            u64::MAX
        } else {
            (1u64 << dst_bits) - 1
        };
        Some(IntCastAction::And(mask))
    } else {
        Some(IntCastAction::SignExt(i64::from(dst_bits / 8)))
    }
}

/// Translate a Charon `CastKind` JSON payload into a canonical RPython
/// cast opname.  Dest-only `Scalar(Float)` drives `cast_int_to_float`;
/// the float-to-int reverse is recovered from the assign destination in
/// the UnaryOp Cast arm.  Ptr casts go through
/// `bhimpl_cast_{int_to_ptr,ptr_to_int}`.  Same-repr casts (RawPtr→RawPtr,
/// same-width Int↔UInt, Unsize) are JIT-no-ops → `same_as` (the
/// assembler's per-kind copy fallback).  Variants the JIT does not model
/// remain identifiable in the unwired diagnostic via the lower-cased
/// default.
pub(in crate::front::mir) fn cast_label_from_payload(payload: &serde_json::Value) -> String {
    let Some(obj) = payload.as_object() else {
        return "same_as".into();
    };
    let Some((kind, inner)) = obj.iter().next() else {
        return "same_as".into();
    };
    match kind.as_str() {
        // Dest-only `CastKind::Scalar(ScalarTy)` — int↔float crossings
        // surface as the canonical RPython cast opnames; int↔uint of
        // any width is a JIT-no-op (`same_as` copies the i64 carrier).
        "Scalar" => {
            // Dest-only `CastKind::Scalar(ScalarTy)`. Int↔float uses the
            // destination: a float dest is `cast_int_to_float`; a float
            // source with an integer dest is recovered from the assign
            // destination in the UnaryOp Cast arm.
            if scalar_is_float(inner) {
                "cast_int_to_float".into()
            } else {
                "same_as".into()
            }
        }
        // `RawPtr(Ty)` — pointer-to-pointer reinterpret; same i64
        // machine repr, so the JIT copies through `same_as`.
        "RawPtr" => "same_as".into(),
        // `Unsize` produces a fat pointer at the source level; the JIT
        // models the array head as a single Ref so this is a no-op.
        "Unsize" => "same_as".into(),
        // `FnPtr` / `Transmute` / `VTable*` etc. — preserve a stable
        // identifier so the unwired diagnostic surfaces the shape.
        _ => kind.to_lowercase(),
    }
}

/// `true` for the Charon `CastKind::RawPtr` payload (atom or
/// single-key object form) — the pointer-to-pointer reinterpret
/// `expr as *const T` / `as *mut T`.
pub(in crate::front::mir) fn cast_kind_is_raw_ptr(kind: &serde_json::Value) -> bool {
    kind.as_str() == Some("RawPtr") || kind.as_object().is_some_and(|o| o.contains_key("RawPtr"))
}

pub(in crate::front::mir) fn scalar_is_float(v: &serde_json::Value) -> bool {
    if let Some(s) = v.as_str() {
        return matches!(s, "F32" | "F64");
    }
    if let Some(obj) = v.as_object()
        && obj.contains_key("Float")
    {
        return true;
    }
    false
}

pub(in crate::front::mir) fn canonical_binop_label(tag: &str, subkind: Option<&str>) -> String {
    // Charon emits `*Checked` (Rust debug-mode trap-on-overflow) and
    // `*Wrap` (release-mode wrapping) variants either as single
    // PascalCase atoms (`"AddChecked"`, `"ShrWrap"`) or as tagged
    // objects (`{"Add": "Checked"}`); both forms collapse onto the
    // plain RPython opname because the JIT does not model Rust's
    // debug-trap semantics — overflow guarding belongs to the
    // optimizer / `guard_no_overflow` level (`pure.rs:int_add_ovf`)
    // and is not emitted from MIR rvalues.
    match (tag, subkind) {
        // Arithmetic (atomic + tagged).
        ("Add" | "AddChecked" | "AddWrap", _) => "add".into(),
        ("Sub" | "SubChecked" | "SubWrap", _) => "sub".into(),
        ("Mul" | "MulChecked" | "MulWrap", _) => "mul".into(),
        ("Div", _) => "floordiv".into(),
        ("Rem", _) => "mod".into(),
        // Bitwise.  The canonical pyre labels carry the `bit` prefix so
        // `codewriter::jtransform` (`bitand`/`bitor`/`bitxor` arm) and
        // the rtyper adapter `normalize_binop_name` (`bitand`->`and_`,
        // `bitor`->`or_`, `bitxor`->`xor`) recognise them.  Bare `and`/`or`
        // are reserved for short-circuit control flow, which never reaches
        // here: rustc lowers `&&`/`||` to branches before charon.
        ("BitAnd", _) => "bitand".into(),
        ("BitOr", _) => "bitor".into(),
        ("BitXor", _) => "bitxor".into(),
        // Shifts.
        ("Shl" | "ShlChecked" | "ShlWrap", _) => "lshift".into(),
        ("Shr" | "ShrChecked" | "ShrWrap", _) => "rshift".into(),
        // Comparisons.
        ("Eq", _) => "eq".into(),
        ("Ne", _) => "ne".into(),
        ("Lt", _) => "lt".into(),
        ("Le", _) => "le".into(),
        ("Gt", _) => "gt".into(),
        ("Ge", _) => "ge".into(),
        // Unary tags surface here through `Rvalue::UnaryOp`.
        ("Neg", _) => "neg".into(),
        ("Not", _) => "invert".into(),
        // Default: lower-case the tag + subkind so unknown shapes
        // remain identifiable in `unwired` diagnostics.
        _ => match subkind {
            Some(s) => format!("{}_{}", tag.to_lowercase(), s.to_lowercase()),
            None => tag.to_lowercase(),
        },
    }
}

/// Best-effort name for an [`Rvalue::Aggregate`]'s constructor, used as
/// the [`CallTarget::SyntheticTransparentCtor::name`] string.  Shape is
/// either an enum-tag object (`{"Adt": {...}}`, `{"Tuple": null}`,
/// `{"Array": null}`) or a bare string.  We project a stable label per
/// kind so debug output is readable; the codewriter does not yet route
/// on these names.
pub(in crate::front::mir) fn aggregate_ctor_name(kind: &serde_json::Value) -> String {
    if let Some(s) = kind.as_str() {
        return s.to_string();
    }
    if let Some(obj) = kind.as_object() {
        // `{"Adt": [{"id": N, "builtin": "Tuple"}, ...]}` — an aggregate
        // of a builtin container rather than a nominal ADT.  Name the
        // placeholder after the builtin tag so every tuple site shares one
        // spelling; the wrapper key "Adt" would mint a fresh same-named
        // class per site (the adapter's bare-leaf arm does not intern), and
        // two `()` values meeting at a join then fail to union ("RPython
        // cannot unify instances with no common base class").
        if let Some(builtin) = obj
            .get("Adt")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(type_decl_ref_builtin)
        {
            return builtin.to_string();
        }
        if let Some(k) = obj.keys().next() {
            return k.clone();
        }
    }
    "ctor".to_string()
}

/// Project a `Value` body to the underlying ADT
/// `def_id` when the body has shape `{"Adt": {"id": <def_id>}}`.
/// Mirrors the reader's private helper used to build
/// `Llbc::dedup_to_adt_def_id`; reproduced here because the inline
/// arm of [`Lowering::resolve_tyexpr_to_adt_def_id`] decodes the
/// same body shape without going through the dedup cache.
pub(in crate::front::mir) fn inline_adt_def_id(body: &serde_json::Value) -> Option<u64> {
    adt_node_def_id(body)
}

/// An inline `Vec<T, A>` value, not `Box<Vec<_>>` and not `&Vec<_>`.
///
/// A one-word-item `Vec` local or field stays the address of its raw
/// `{ptr, len, cap}` header (`RustVecRepr`, kind `int` /
/// `ll_vec_length` / `gcarray_from_pyobject_vec`). Marking that field
/// `inline_vec` makes getattr of it a `RustVec` and unions Integer with
/// that object (`ll_vec_alloc_and_set_*`). `Vec::len` of such a field
/// still retargets through `inline_vec_field_part` (`VecFieldPart::Len`,
/// `rlist.py` `ll_length` = parent `{name}.len` Signed word) from the
/// Int FieldRead producer, without flipping `inline_vec`.
///
/// The ADT is the type decl whose name segments are `alloc::vec::Vec`
/// (that decl's identity). The use site's type-argument count is the
/// decl's generic arity: one (`T`) or two (`T` and the allocator).
pub(in crate::front::mir) fn field_ty_is_inline_vec(ty: &TyRef, llbc: &Llbc) -> bool {
    let body = match tyref_node(ty, llbc) {
        Some(body) => body,
        None => return false,
    };
    let Some(obj) = body.as_object() else {
        return false;
    };
    if obj.contains_key("Ref") || obj.contains_key("RawPtr") {
        return false;
    }
    let Some(id) = inline_adt_def_id(body) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(id) else {
        return false;
    };
    if td.item_meta.template_name().count() != 3 {
        return false;
    }
    let idents: Vec<&str> = td
        .item_meta
        .template_name()
        .filter_map(|seg| match seg {
            NameSeg::Ident { ident: (s, _) } => Some(s.as_str()),
            NameSeg::Other(_) => None,
        })
        .collect();
    if idents != ["alloc", "vec", "Vec"] {
        return false;
    }
    let arity = obj
        .get("Adt")
        .and_then(|adt| adt.as_object())
        .and_then(|adt| type_decl_ref_generics(adt, llbc))
        .and_then(|generics| generics.get("types"))
        .and_then(|types| types.as_array())
        .map(|types| types.len());
    if !matches!(arity, Some(1 | 2)) {
        return false;
    }
    tyref_rust_vec_item_kind(ty, llbc).is_none()
}

/// Project a Charon [`TyRef`] to the JIT-visible [`ValueType`].
///
/// Numeric scalars → `Int` / `Unsigned` / `Float`, bool → `Bool`, unit → `Void`,
/// everything else (structs, pointers, references) → `Ref`.  The
/// TyRef's serialized form is the source of truth —
/// `TyRef::label()` produces a compact short form
/// (`"ty#170"`, `"ty<Adt>"`) for opaque IDs, while the underlying
/// JSON carries the primitive name for literal types.
///
/// For `TyRef::Deduplicated{id}`, the projection consults
/// `llbc.dedup_body(id)` to recover the inline body shape and runs
/// the same primitive-pattern match.  A nested `{"Deduplicated": id}`
/// inside `TyRef::Other` (a Ref pointee of `&i64`) is followed the
/// same way.  Required so FunDecl return types serialized as
/// `Deduplicated` (≈92% in `pyre-interpreter.ullbc`) resolve to
/// `Int` / `Bool` / `Float` instead of falling back to `Ref`.
/// The JIT register bank a [`ValueType`] occupies, mirroring
/// `flatten.py getkind`: the integer family (`Int` / `Unsigned` /
/// `Bool`) shares the `'int'` bank, `Ref` the `'ref'` bank, `Float` the
/// `'float'` bank.  Non-value kinds (`Void` / `State` / `Unknown`) get a
/// distinct discriminant so they never compare equal to a real bank.
pub(crate) fn value_type_bank(ty: &ValueType) -> u8 {
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => 0,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
        ValueType::Int128 => 6,
        ValueType::UInt128 => 7,
        // Its own bank, not the integer one `getkind` would give it: a
        // same-bank pair is aliased without an op, and an `f32` that
        // aliased an `i64` is what let Rust's native `f32` arithmetic
        // lower to integer arithmetic over the float's bit pattern.
        ValueType::SingleFloat => 8,
    }
}

/// Whether `transmute::<A, B>` is an ordinary copy in the lowered graph.
///
/// The predicate is deliberately the conjunction of two independent facts:
///
/// * `A` and `B` occupy the same real JIT register bank (`Int`/`Unsigned`/
///   `Bool`, `Ref`/`Str`/`StringBuilder`, or `Float`); and
/// * the exact sizes recovered from Charon's serialized scalar type or the
///   target-selected `TypeDecl` layout are both known and equal.
///
/// Unknown layouts decline.  In particular, matching `ValueType`s alone is
/// insufficient because that projection intentionally erases scalar widths
/// and aggregate sizes.
pub(in crate::front::mir) fn transmute_is_same_layout_bank(
    src: &TyRef,
    dst: &TyRef,
    llbc: &Llbc,
) -> bool {
    let src_bank = value_type_bank(&tyref_to_value_type(src, llbc));
    let dst_bank = value_type_bank(&tyref_to_value_type(dst, llbc));
    matches!((src_bank, dst_bank), (0, 0) | (1, 1) | (2, 2))
        && matches!(
            (tyref_exact_layout_size(src, llbc), tyref_exact_layout_size(dst, llbc)),
            (Some(src_size), Some(dst_size)) if src_size == dst_size
        )
}

/// Exact byte size encoded by a Charon [`TyRef`].
///
/// Named ADTs use the target-selected LLBC layout.  Primitive widths are part
/// of the serialized LLBC type itself; references/raw pointers/function
/// pointers have the target word size when their representation is known to
/// be thin.  Every other shape returns `None`, keeping the transmute residual.
pub(in crate::front::mir) fn tyref_exact_layout_size(ty: &TyRef, llbc: &Llbc) -> Option<u64> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let obj = node.as_object()?;

    if let Some(scalar) = obj.get("Scalar") {
        if let Some(atom) = scalar.as_str() {
            return match atom {
                "Bool" => Some(1),
                "Char" => Some(4),
                _ => None,
            };
        }
        let scalar = scalar.as_object()?;
        let scalar_size = |kind: &str| match kind {
            "I8" | "U8" => Some(1),
            "I16" | "U16" | "F16" => Some(2),
            "I32" | "U32" | "F32" => Some(4),
            "I64" | "U64" | "F64" => Some(8),
            "I128" | "U128" | "F128" => Some(16),
            "Isize" | "Usize" => Some(crate::layout::target_word_size() as u64),
            _ => None,
        };
        if let Some(int) = scalar.get("Integer").and_then(serde_json::Value::as_object)
            && let Some(kind) = int
                .get("Signed")
                .or_else(|| int.get("Unsigned"))
                .and_then(serde_json::Value::as_str)
        {
            return scalar_size(kind);
        }
        if let Some(kind) = scalar.get("Float").and_then(serde_json::Value::as_str) {
            return scalar_size(kind);
        }
        return None;
    }

    if let Some(def_id) = adt_node_def_id(node) {
        let target = std::env::var("TARGET").unwrap_or_default();
        return llbc
            .type_by_id(def_id)?
            .layout_for_target(llbc, &target)?
            .size;
    }

    let pointee = obj
        .get("Ref")
        .and_then(serde_json::Value::as_array)
        .and_then(|parts| parts.get(1))
        .or_else(|| {
            obj.get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|parts| parts.first())
        });
    if pointee.is_some_and(|pointee| json_ty_is_statically_sized(pointee, llbc))
        || obj.contains_key("FnPtr")
    {
        return Some(crate::layout::target_word_size() as u64);
    }

    None
}

/// The fully-qualified host-callable path for a bank-crossing cast, or
/// `None` when `src` and `dst` share a bank (a JIT no-op aliased to the
/// operand) or the bank pair has no host cast callable.  A bank crossing
/// lowers to `simple_call(<host_callable>, v)`, never an `OpKind::UnaryOp`:
/// the rtyper retired every typed cast opname from the unary-op path
/// (`flowspace_adapter::normalize_unary_op_name` accepts only
/// `neg` / `bool` / `invert` / `same_as`), so the only surface that reaches
/// `rtype_cast_int_to_ptr` / `rtype_cast_ptr_to_int` (rbuiltin.py)
/// and `rtype_builtin_float` / `rtype_builtin_int` (rbuiltin.py,
/// which delegate to `rtype_float` / `rtype_int` and emit the low-level
/// `cast_int_to_float` / `cast_float_to_int`) is a `simple_call`.  `int →
/// ptr` / `ptr → int` resolve the `lltype.cast_*` module attr
/// (`["rpython", "rtyper", "lltypesystem", "lltype", …]` per
/// `flowspace_adapter` Branch 3b); `int → float` / `float → int` resolve
/// the `__builtin__.float` / `__builtin__.int` objects.  Keeping the builtin
/// owner in the path preserves the callable identity that keys
/// `rbuiltin.py::BUILTIN_TYPER`; a bare leaf can collide with an unrelated
/// Rust function named `float` or `int` in the flat call registry.
/// `rffi.size_and_sign(T)` of a Rust integer literal type: its byte size and
/// whether it is unsigned. `None` for anything but a
/// `{"Integer": {"Signed" | "Unsigned": _}}` scalar (`bool` and `char` included, which do not reach an
/// integer `and_` without their own conversion first).
pub(in crate::front::mir) fn int_cast_size_and_sign(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(u64, bool)> {
    let int = tyref_node(ty, llbc)?
        .as_object()?
        .get("Scalar")?
        .as_object()?
        .get("Integer")?
        .as_object()?;
    let (atom, unsigned) =
        if let Some(atom) = int.get("Unsigned").and_then(serde_json::Value::as_str) {
            (atom, true)
        } else {
            (
                int.get("Signed").and_then(serde_json::Value::as_str)?,
                false,
            )
        };
    let size = match atom {
        "I8" | "U8" => 1,
        "I16" | "U16" => 2,
        "I32" | "U32" => 4,
        "I64" | "U64" => 8,
        "I128" | "U128" => 16,
        "Isize" | "Usize" => crate::layout::target_word_size() as u64,
        _ => return None,
    };
    Some((size, unsigned))
}

/// The operation `jtransform.py _int_to_int_cast` rewrites an integer
/// `force_cast` into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::front::mir) enum IntToIntCast {
    /// The destination holds every source value: no operation.
    Noop,
    /// `int_and(v, mask)` — narrowing to an unsigned type.
    And(u64),
    /// `int_signext(v, numbytes)` — narrowing to a signed type.
    Signext(u64),
}

/// `jtransform.py _int_to_int_cast` for a cast between two integer types of
/// at most `word` bytes, given as `(size, unsigned)` pairs
/// (`rffi.size_and_sign`). `None` for a longlong operand, whose
/// `truncate_longlong_to_int` / `cast_*_to_*longlong` legs this does not
/// cover.
pub(in crate::front::mir) fn int_to_int_cast(
    src: (u64, bool),
    dst: (u64, bool),
    word: u64,
) -> Option<IntToIntCast> {
    let (size1, unsigned1) = src;
    let (size2, unsigned2) = dst;
    if size1 > word || size2 > word {
        return None;
    }
    // the target type is LONG or ULONG
    if size2 == word {
        return Some(IntToIntCast::Noop);
    }
    // `rarithmetic.integer_bounds`
    let integer_bounds = |size: u64, unsigned: bool| -> (i128, i128) {
        let bits = 8 * size as u32;
        if unsigned {
            (0, (1i128 << bits) - 1)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        }
    };
    let (min1, max1) = integer_bounds(size1, unsigned1);
    let (min2, max2) = integer_bounds(size2, unsigned2);
    // the target type includes the source range
    if min2 <= min1 && max1 <= max2 {
        return Some(IntToIntCast::Noop);
    }
    Some(if min2 != 0 {
        IntToIntCast::Signext(size2)
    } else {
        IntToIntCast::And(((1u128 << (8 * size2)) - 1) as u64)
    })
}

/// `lltype.cast_int_to_ptr` as the host call the rtyper accepts.
/// `normalize_unary_op_name` rejects a unary op of this name.
pub(crate) fn cast_int_to_ptr_call(operand: crate::flowspace::model::Variable) -> OpKind {
    OpKind::Call {
        target: CallTarget::function_path([
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_int_to_ptr",
        ]),
        args: crate::model::call_args(vec![operand]),
        result_ty: ValueType::Ref(None),
    }
}

/// `lltype.cast_ptr_to_int` as the host call the rtyper accepts.
/// `normalize_unary_op_name` rejects a unary op of this name.
pub(crate) fn cast_ptr_to_int_call(operand: crate::flowspace::model::Variable) -> OpKind {
    OpKind::Call {
        target: CallTarget::function_path([
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_ptr_to_int",
        ]),
        args: crate::model::call_args(vec![operand]),
        result_ty: ValueType::Int,
    }
}

pub(in crate::front::mir) fn cast_call_segments(
    src: &ValueType,
    dst: &ValueType,
) -> Option<Vec<String>> {
    let (s, d) = (value_type_bank(src), value_type_bank(dst));
    let lltype = |name: &str| -> Vec<String> {
        ["rpython", "rtyper", "lltypesystem", "lltype", name]
            .into_iter()
            .map(str::to_string)
            .collect()
    };
    match (s, d) {
        _ if s == d => None,
        // int → ptr / ptr → int — `lltype.cast_int_to_ptr` /
        // `lltype.cast_ptr_to_int`.
        (0, 1) => Some(lltype("cast_int_to_ptr")),
        (1, 0) => Some(lltype("cast_ptr_to_int")),
        // int → float / float → int — `float(v)` / `int(v)`, whose
        // rtyper delegates to `rtype_float` / `rtype_int`.  The explicit
        // `__builtin__` owner is the flowspace Constant's host identity, not
        // a new namespace: `HOST_ENV.import_module("__builtin__")` returns
        // the same singleton objects that key `BUILTIN_TYPER`.
        (0, 2) => Some(vec!["__builtin__".to_string(), "float".to_string()]),
        (2, 0) => Some(vec!["__builtin__".to_string(), "int".to_string()]),
        // No host callable for the remaining pairs (e.g. ref↔float, or any
        // pair touching Void/State/Unknown): alias the operand.
        _ => None,
    }
}

/// `true` when an integer binop has a pointer operand and no pointer
/// form of that op exists.
///
/// `rptr.py` registers only `rtype_eq` / `rtype_ne`. There is no
/// `ptr_lt` / `ptr_mod` / `ptr_floordiv`. The RPython spelling is
/// `lltype.cast_ptr_to_int` then the integer op (`rbuiltin.py
/// rtype_cast_ptr_to_int`). Equality is excluded on purpose.
pub(in crate::front::mir) fn int_binop_needs_ptr_to_int(
    op: &str,
    lhs: Option<&ValueType>,
    rhs: Option<&ValueType>,
) -> bool {
    matches!(op, "lt" | "le" | "gt" | "ge" | "mod" | "floordiv" | "div")
        && (matches!(lhs, Some(ValueType::Ref(_))) || matches!(rhs, Some(ValueType::Ref(_))))
}

/// Overlay field lltype. A raw pointer to a GC struct is GCREF
/// (`pointer_field_lltype` of a GC pointee). A raw pointer to a non-GC
/// ADT (`MapRef` / `*const MapNode`) is `_NONGCREF` (`llmemory.py`):
/// `SomePtr.getattr` then returns `SomePtr`, so a later dest-overlay
/// recast can `rtype_cast_ptr_to_int` (`rbuiltin.py` requires `PtrRepr`).
/// `tyref_to_external_lltype` maps every raw pointer to Address (the
/// residual-call ABI); Address annotates as `SomeAddress` and that
/// recast Skips. Scalars keep `tyref_to_external_lltype`.
pub(in crate::front::mir) fn overlay_field_lltype(
    ty: &TyRef,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{GCREF, LowLevelType, NONGCREF};
    let Some(node) = tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| type_node_peel_aliases(n, llbc))
    else {
        return tyref_to_external_lltype(ty, llbc);
    };
    let pointee = node
        .as_object()
        .and_then(|obj| {
            obj.get("RawPtr")
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())
                .or_else(|| {
                    obj.get("Ref")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|arr| arr.get(1))
                })
        })
        .and_then(|p| strip_ty_wrappers(p, llbc))
        .and_then(|p| type_node_peel_aliases(p, llbc));
    if let Some(pointee) = pointee
        && let Some(id) = adt_node_def_id(pointee)
        && let Some(td) = llbc.type_by_id(id)
    {
        if struct_id_is_declared_gc(
            gc_struct_ids,
            struct_id_for_type_decl(td),
            &strip_crate_prefix(&td.item_meta.name_path()),
        ) {
            return (*GCREF).clone();
        }
        return LowLevelType::Ptr(Box::new(NONGCREF.clone()));
    }
    tyref_to_external_lltype(ty, llbc)
}

/// `lltype.cast_int_to_ptr(PTRTYPE, int)` (`lltype.py` `ann_cast_int_to_ptr`).
/// Same 2-arg shape as [`emit_rffi_cast_to_carray_ptr`]; dest is the overlay
/// `Ptr`. Raw `Ptr` `getkind` is `'int'` (`history.py`).
pub(in crate::front::mir) fn emit_dest_typed_cast_int_to_ptr(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    addr: Variable,
    ptr_ty: crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Variable {
    use crate::flowspace::model::ConstValue;
    let raw = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    raw.set_concretetype(Some(ptr_ty.clone()));
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(raw.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([
                "rpython",
                "rtyper",
                "lltypesystem",
                "lltype",
                "cast_int_to_ptr",
            ]),
            args: vec![
                crate::model::LinkArg::from(ConstValue::LowLevelType(Box::new(ptr_ty))),
                crate::model::LinkArg::from(addr),
            ],
            result_ty: ValueType::Int,
        },
    });
    raw
}

/// `rffi.cast(CArrayPtr(T), p)` (`rffi.py` `cast`): `cast_ptr_to_int` then
/// `cast_int_to_ptr(PTRTYPE, int)` (`lltype.py` `ann_cast_int_to_ptr`).
/// `__cast_address_intrinsic` (`p as *mut u8`) and a classdef-less
/// `*mut u8` input annotate as `SomeInstance`; `ann_direct_ptradd`
/// needs `SomePtr`. The 2-arg form supplies `PtrT.const` so the
/// analyzer returns a raw `CArrayPtr` (`getkind` Signed), not the
/// 1-arg Gc-Opaque placeholder (GcRef).
pub(in crate::front::mir) fn emit_rffi_cast_to_carray_ptr(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    ptr: Variable,
    item: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Variable {
    let addr = push_cast_ptr_to_int(graph, bb_id, ptr);
    emit_cast_int_to_carray_ptr(graph, bb_id, addr, item)
}

/// 2-arg `lltype.cast_int_to_ptr(PtrT, int)` (`lltype.py`
/// `ann_cast_int_to_ptr`). The result is a raw `CArrayPtr` / `CCHARP`.
pub(in crate::front::mir) fn emit_cast_int_to_carray_ptr(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    addr: Variable,
    item: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Variable {
    use crate::flowspace::model::ConstValue;
    let ptr_ty = lltype_for_direct_ptradd_pointer(None, item);
    let raw = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    raw.set_concretetype(Some(ptr_ty.clone()));
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(raw.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([
                "rpython",
                "rtyper",
                "lltypesystem",
                "lltype",
                "cast_int_to_ptr",
            ]),
            args: vec![
                crate::model::LinkArg::from(ConstValue::LowLevelType(Box::new(ptr_ty))),
                crate::model::LinkArg::from(addr),
            ],
            result_ty: ValueType::Int,
        },
    });
    raw
}

/// `cast_int_to_ptr(CCHARP, int)` so a window data word is `CCHARP` rather
/// than a bare `Integer`.
pub(in crate::front::mir) fn emit_cast_int_to_ccharp(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    addr: Variable,
) -> Variable {
    emit_cast_int_to_carray_ptr(
        graph,
        bb_id,
        addr,
        &crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Char,
    )
}

/// Emit `simple_call(lltype.direct_ptradd, p, n)` and return the raw
/// pointer.  `ann_direct_ptradd` returns `s_p`.  `n` is an unscaled item
/// count.  `item` is `TO.OF`: a `Char` builds `CCHARP` (byte shift);
/// anything else is what `rewrite_op_direct_ptradd` multiplies by.
/// The result kind is int.
pub(in crate::front::mir) fn push_direct_ptradd(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    ptr: Variable,
    count: Variable,
    item: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Variable {
    // A Rust `*const T` / `*mut T` is `rffi.CArrayPtr(T)` (`Ptr(Array)` with
    // `nolength`), and a byte pointer is `CCHARP`. An earlier cast may have
    // stamped `Signed` or a GC pointer; `rewrite_op_direct_ptradd` scales
    // from `TO.OF`, so the operand has to carry the array pointer.
    // Stamp the array pointer on an int-kind value. A GC / classdef-less
    // Instance is not `SomePtr`: peel `__cast_address_intrinsic` and
    // `rffi.cast` onto `CArrayPtr` (`lltype.py` `ann_direct_ptradd`,
    // `rffi.py` `ptradd`). `cast_ptr_to_int` alone would make the operand
    // `Integer`; `cast_int_to_ptr` after it is the pointer.
    let already_int = ptr
        .concretetype()
        .as_ref()
        .is_some_and(|ty| crate::model::getkind(ty) == crate::model::ConcreteType::Signed);
    let ptr = if already_int {
        ptr
    } else {
        let peeled = direct_ptradd_pointer_operand(graph, ptr);
        let peeled_int = peeled
            .concretetype()
            .as_ref()
            .is_some_and(|ty| crate::model::getkind(ty) == crate::model::ConcreteType::Signed);
        if peeled_int {
            peeled
        } else {
            emit_rffi_cast_to_carray_ptr(graph, bb_id, peeled, item)
        }
    };
    let ptr_ty = lltype_for_direct_ptradd_pointer(ptr.concretetype(), item);
    let ptr = if ptr.concretetype().as_ref() == Some(&ptr_ty) {
        ptr
    } else {
        // `clone` shares the concretetype cell. A fresh `same_as` result
        // carries `TO.OF` without rewriting the caller's variable.
        let fresh = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        fresh.set_concretetype(Some(ptr_ty.clone()));
        graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(fresh.clone()),
            kind: OpKind::UnaryOp {
                op: "same_as".to_string(),
                operand: ptr,
                result_ty: ValueType::Int,
            },
        });
        fresh
    };
    // An unstamped count reads as ref (`Unknown` → `'r'`), and
    // `rewrite_op_direct_ptradd` then refuses the scale. The count is
    // an item index: `lltype.Signed`.
    if FunctionGraph::concretetype_of(&count) == crate::model::ConcreteType::Unknown {
        FunctionGraph::set_concretetype_of_inline(&count, crate::model::ConcreteType::Signed);
    }
    let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    result.set_concretetype(Some(ptr_ty));
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: [
                    "rpython",
                    "rtyper",
                    "lltypesystem",
                    "lltype",
                    "direct_ptradd",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![ptr, count]),
            result_ty: ValueType::Int,
        },
    });
    result
}

/// `rffi.cast(PTRTYPE, v)` (`rffi.py` `cast` = `ll2ctypes.force_cast`).
/// `ForceCastEntry` (`ll2ctypes.py`) → `genop('force_cast', [v_arg], RESTYPE)`.
pub(in crate::front::mir) fn emit_rffi_cast(
    graph: &mut FunctionGraph,
    bb: BlockId,
    v: Variable,
    ptr_ty: crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Variable {
    use crate::flowspace::model::ConstValue;
    let raw = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    raw.set_concretetype(Some(ptr_ty.clone()));
    graph.block_mut(bb).operations.push(SpaceOperation {
        result: Some(raw.clone()),
        kind: OpKind::Call {
            target: CallTarget::function_path([
                "rpython",
                "rtyper",
                "lltypesystem",
                "rffi",
                "cast",
            ]),
            args: vec![
                crate::model::LinkArg::from(ConstValue::LowLevelType(Box::new(ptr_ty))),
                crate::model::LinkArg::from(v),
            ],
            result_ty: ValueType::Int,
        },
    });
    raw
}

/// Restore the raw entry pointer type after a byte `direct_ptradd`.
/// `ann_direct_ptradd` returns `s_p`; the byte offset runs on CCHARP, so
/// the result is reinterpreted back to the raw entry type with `rffi.cast`
/// (`ll2ctypes.py ForceCastEntry` → `force_cast`). A checked
/// `lltype.cast_pointer` would reject two unrelated array pointer types
/// (`lltype.py castable`).
pub(in crate::front::mir) fn restore_offset_pointer_type(
    graph: &mut FunctionGraph,
    bb: BlockId,
    entry_ty: Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType>,
    result: Variable,
) -> Variable {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    if let Some(t) = entry_ty
        && matches!(&t, LowLevelType::Ptr(p) if !p._needsgc())
        && result.concretetype().as_ref() != Some(&t)
    {
        return emit_rffi_cast(graph, bb, result, t);
    }
    // Int-kind (`getkind` Signed) or None: no pointer annotation to restore.
    // A GC pointer plus a byte offset is an interior address, which stays CCHARP.
    result
}

/// Operand `direct_ptradd` annotates. `__cast_address_intrinsic` drops the
/// pointee class and answers `SomeInstance`; peel it so `rffi.cast` sees
/// the pointer that was erased. A GC header that remains an Instance is
/// then `rffi.cast` onto `CArrayPtr` in [`push_direct_ptradd`].
pub(in crate::front::mir) fn direct_ptradd_pointer_operand(
    graph: &FunctionGraph,
    ptr: Variable,
) -> Variable {
    let Some((block_id, idx)) = resolve_to_producer_op(graph, &ptr) else {
        return ptr;
    };
    let Some(op) = graph
        .blocks
        .iter()
        .find(|block| block.id == block_id)
        .and_then(|block| block.operations.get(idx))
    else {
        return ptr;
    };
    let OpKind::Call { target, args, .. } = &op.kind else {
        return ptr;
    };
    let CallTarget::FunctionPath { segments, .. } = target else {
        return ptr;
    };
    if segments.last().map(String::as_str) != Some(crate::runtime_names::shims::CAST_ADDRESS) {
        return ptr;
    }
    args.first()
        .and_then(crate::model::LinkArg::as_variable)
        .cloned()
        .unwrap_or(ptr)
}

pub(in crate::front::mir) fn is_raw_array_ptr(
    ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> bool {
    raw_array_ptr_item(ty).is_some()
}

pub(in crate::front::mir) fn lltype_is_ccharp(
    ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> bool {
    raw_array_ptr_item(ty).is_some_and(|of| {
        *of == crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Char
    })
}

pub(in crate::front::mir) fn raw_array_ptr_item(
    ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> Option<&crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    let LowLevelType::Ptr(ptr) = ty else {
        return None;
    };
    let PtrTarget::Array(arr) = &ptr.TO else {
        return None;
    };
    Some(&arr.OF)
}

/// `CArrayPtr(item)`, or `CCHARP` when the item is a byte (`lltype.Char`).
/// An existing `Ptr(Array)` is already that shape and is kept.
pub(crate) fn lltype_for_direct_ptradd_pointer(
    existing: Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType>,
    item: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::lltypesystem::rffi::{CArrayPtr, CCHARP};
    if let Some(existing) = existing.as_ref()
        && is_raw_array_ptr(existing)
        && raw_array_ptr_item(existing).is_some_and(|of| of == item)
    {
        return existing.clone();
    }
    if *item == LowLevelType::Char {
        return (*CCHARP).clone();
    }
    CArrayPtr(item.clone())
}

/// Item type whose `llmemory.sizeof` is `n` bytes, so
/// `rewrite_op_direct_ptradd` can read `TO.OF`.  One byte is `Char`
/// (`CCHARP` skips the multiply).
pub(in crate::front::mir) fn item_lltype_for_size(
    n: i64,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{FixedSizeArray, LowLevelType};
    let word = crate::layout::target_word_size() as i64;
    if n == 1 {
        LowLevelType::Char
    } else if n == word {
        LowLevelType::Signed
    } else if n == 4 {
        LowLevelType::UniChar
    } else if n == 16 {
        LowLevelType::SignedLongLongLong
    } else {
        LowLevelType::FixedSizeArray(Box::new(FixedSizeArray::new(
            LowLevelType::Char,
            n as usize,
        )))
    }
}

/// Emit `simple_call(lltype.cast_ptr_to_int, p)` and return the Signed
/// address integer. Shared by `p as usize` (`push_ptr_to_unsigned_cast`)
/// and integer binops over a Ref. The result is stamped Signed —
/// `rbuiltin.py rtype_cast_ptr_to_int` uses `resulttype=lltype.Signed`.
/// Leaving it `Unknown` makes `get_value_kind_var` report `'r'` and
/// jtransform re-inserts a second cast.
pub(in crate::front::mir) fn push_cast_ptr_to_int(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    arg: Variable,
) -> Variable {
    let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: [
                    "rpython",
                    "rtyper",
                    "lltypesystem",
                    "lltype",
                    "cast_ptr_to_int",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![arg]),
            result_ty: ValueType::Int,
        },
    });
    result
}

pub(in crate::front::mir) fn push_r_uint(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    arg: Variable,
) -> Variable {
    let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: rarithmetic_mask("r_uint", arg, ValueType::Unsigned),
    });
    result
}

pub(in crate::front::mir) fn rarithmetic_mask(
    name: &str,
    arg: Variable,
    result_ty: ValueType,
) -> OpKind {
    OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: ["rpython", "rlib", "rarithmetic", name]
                .into_iter()
                .map(str::to_string)
                .collect(),
            fun_decl_id: None,
        },
        args: crate::model::call_args(vec![arg]),
        result_ty,
    }
}

/// Emit RPython's pointer-to-Unsigned primitive cast sequence.
///
/// `rbuiltin.py`'s `gen_cast` first produces a Signed address integer with
/// `cast_ptr_to_int`, then applies `gen_cast(..., Unsigned)`.  The caller
/// appends the returned `r_uint` operation after the Signed producer pushed
/// here, preserving that ordering while fitting [`Lowering::build_rvalue`]'s
/// one-returned-operation interface.
pub(in crate::front::mir) fn push_ptr_to_unsigned_cast(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    arg: Variable,
) -> (OpKind, Variable) {
    let signed = push_cast_ptr_to_int(graph, bb_id, arg);
    // `r_uint` is still the integer bank (`getkind(Unsigned) == 'int'`).
    let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    let retype = OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            fun_decl_id: None,
        },
        args: crate::model::call_args(vec![signed]),
        result_ty: ValueType::Unsigned,
    };
    (retype, result)
}

#[cfg(test)]
mod tests;
