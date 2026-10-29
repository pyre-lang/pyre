//! Low-level list/str views (pair/window/str-view/list-view; `rlist.py` / `rstr.py` reprs).

use super::*;

impl<'a> Lowering<'a> {
    /// Resolve an [`Operand`] to the Variable the IR should reference.
    /// The item kind of `ty` when it is a pair slice of an enabled kind.
    pub(in crate::front::mir) fn pair_slice_kind(
        &self,
        ty: &TyRef,
    ) -> Option<majit_ir::rvec::VecItemKind> {
        tyref_pair_slice_item_kind(ty, self.llbc)
            .filter(|kind| PAIR_SLICE_ITEM_KINDS.contains(kind))
    }

    /// The item kind of `ty` when it is a slice iterator of an enabled kind.
    pub(in crate::front::mir) fn pair_iter_kind(
        &self,
        ty: &TyRef,
    ) -> Option<majit_ir::rvec::VecItemKind> {
        tyref_pair_slice_iter_item_kind(ty, self.llbc)
            .filter(|kind| PAIR_SLICE_ITEM_KINDS.contains(kind))
    }

    /// The item kind of pair-slice MIR local `local`, or of pair-slice
    /// iterator local `local`.
    pub(in crate::front::mir) fn pair_local_kind(
        &self,
        local: usize,
    ) -> Option<majit_ir::rvec::VecItemKind> {
        let ty = &self.body.locals.locals.get(local)?.ty;
        self.pair_slice_kind(ty).or_else(|| self.pair_iter_kind(ty))
    }

    /// Whether MIR local `local` is a pair-slice iterator.
    pub(in crate::front::mir) fn pair_local_is_iter(&self, local: usize) -> bool {
        self.body
            .locals
            .locals
            .get(local)
            .is_some_and(|loc| self.pair_iter_kind(&loc.ty).is_some())
    }

    /// The pair-slice iterator `place` names: `it` itself, or `*r` when `r`
    /// holds that iterator.
    pub(in crate::front::mir) fn pair_iter_referent(&self, place: &Place) -> Option<Place> {
        let local = match &place.kind {
            PlaceKind::Local(local) => *local as usize,
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                let PlaceKind::Local(local) = inner.kind else {
                    return None;
                };
                local as usize
            }
            _ => return None,
        };
        if !self.pair_local_is_iter(local)
            && !matches!(
                self.local_var.get(local).and_then(|slot| slot.as_ref()),
                Some(LocalValue::SliceIter { .. })
            )
        {
            return None;
        }
        Some(Place {
            kind: PlaceKind::Local(local as u64),
            ty: clone_tyref(&self.body.locals.locals[local].ty),
        })
    }

    /// `core::slice::iter::<Impl>::next` on a slice `Iter` / `IterMut`
    /// receiver, or any `next` whose receiver MIR type is that iterator.
    /// The suffix alone is not enough.
    pub(in crate::front::mir) fn slice_iter_next_value(
        &self,
        call: &CallPayload,
    ) -> Option<LocalValue> {
        let CallFunc::Regular(reg) = &call.func else {
            return None;
        };
        let path = regular_call_name_path(reg, self.llbc)?;
        let operand = call.args.first()?;
        let (Operand::Copy(place) | Operand::Move(place)) = operand else {
            return None;
        };
        let local = match &place.kind {
            PlaceKind::Local(local) => *local as usize,
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                let PlaceKind::Local(local) = inner.kind else {
                    return None;
                };
                local as usize
            }
            _ => return None,
        };
        let value = self.local_var.get(local)?.clone()?;
        if !matches!(value, LocalValue::SliceIter { .. }) {
            return None;
        }
        let recv_ty = operand_tyref(operand).unwrap_or(&self.body.locals.locals.get(local)?.ty);
        let recv_is_iter = self.pair_iter_kind(recv_ty).is_some() || self.pair_local_is_iter(local);
        let path_ok =
            path == "core::slice::iter::<Impl>::next" || (recv_is_iter && path.ends_with("::next"));
        path_ok.then_some(value)
    }

    /// The `(ptr, len)` words a pair-slice local holds, or the `(index, items)`
    /// words of a slice iterator.
    pub(in crate::front::mir) fn pair_local_parts(
        &self,
        mir_bb: usize,
        local: usize,
    ) -> Result<(Variable, Variable), LowerError> {
        let uninit = || {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: read of pair slice local {local} before any Assign"
            ))
        };
        self.local_var
            .get(local)
            .and_then(|slot| slot.as_ref())
            .ok_or_else(uninit)?
            .pair()
            .map_err(|_| uninit())
    }

    pub(in crate::front::mir) fn bind_pair_local(
        &mut self,
        local: usize,
        ptr: Variable,
        len: Variable,
        owner: Option<Variable>,
    ) {
        let len_slot = self.len_shadow[local].expect("pair slice local has a length slot");
        self.local_var[local] = Some(LocalValue::Pair(ptr, len.clone(), owner.clone()));
        self.local_var[len_slot] = Some(LocalValue::One(len));
        if let Some(owner_slot) = self.owner_shadow.get(local).copied().flatten() {
            self.local_var[owner_slot] = owner.map(LocalValue::One);
        }
        self.positional_aggregate_locals.remove(&local);
        self.string_byte_view_locals.retain(|&l| l != local);
        self.list_view_locals.retain(|&l| l != local);
    }

    /// GC owner of a pair window whose data word is `ptr`, if this frame
    /// still holds one (`rffi.get_raw_address_of_string` / list header).
    pub(in crate::front::mir) fn pair_owner_of_ptr(&self, ptr: &Variable) -> Option<Variable> {
        self.local_var.iter().find_map(|slot| match slot.as_ref() {
            Some(LocalValue::Pair(data, _, owner)) if data == ptr => owner.clone(),
            _ => None,
        })
    }

    pub(in crate::front::mir) fn bind_window_owner_shadow(
        &mut self,
        local: usize,
        owner: Option<Variable>,
    ) {
        if let Some(LocalValue::Pair(_, _, slot)) =
            self.local_var.get_mut(local).and_then(Option::as_mut)
        {
            *slot = owner.clone();
        }
        if let Some(owner_slot) = self.owner_shadow.get(local).copied().flatten() {
            self.local_var[owner_slot] = owner.map(LocalValue::One);
        }
    }

    /// GC owner of a pair window local `local`, if any.
    /// A one-word STR/list view is the header itself (`ll_stritem_nonneg`)
    /// and has no owner (`keepalive_until_here` protects a raw `char *`).
    pub(in crate::front::mir) fn window_owner_of_local(&self, local: usize) -> Option<Variable> {
        match self.local_var.get(local).and_then(|slot| slot.as_ref()) {
            Some(LocalValue::Pair(_, _, Some(owner))) => Some(owner.clone()),
            _ => self
                .owner_shadow
                .get(local)
                .copied()
                .flatten()
                .and_then(|slot| {
                    self.local_var
                        .get(slot)
                        .and_then(|entry| entry.as_ref())
                        .and_then(|value| value.one().ok())
                }),
        }
    }

    /// Owner of a window source word: the Pair's third field, a STR/list
    /// header, or the owner_shadow of the local that holds `source`.
    pub(in crate::front::mir) fn owner_for_window_source(
        &self,
        source: &Variable,
    ) -> Option<Variable> {
        if let Some(owner) = self.pair_owner_of_ptr(source) {
            return Some(owner);
        }
        for (i, slot) in self.local_var.iter().enumerate() {
            if let Some(LocalValue::One(v)) = slot.as_ref()
                && v == source
                && let Some(owner) = self.window_owner_of_local(i)
            {
                return Some(owner);
            }
        }
        if self.var_is_str_repr(source) || self.list_view_local_of_var(source).is_some() {
            Some(source.clone())
        } else {
            None
        }
    }

    /// Stamp `owner` onto every pair window whose data word is `source`.
    pub(in crate::front::mir) fn capture_window_owner_of_var(
        &mut self,
        source: &Variable,
        owner: Option<Variable>,
    ) {
        let locals: Vec<usize> = self
            .local_var
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot.as_ref() {
                Some(LocalValue::Pair(data, _, _)) if data == source => Some(i),
                _ => None,
            })
            .collect();
        for local in locals {
            self.bind_window_owner_shadow(local, owner.clone());
        }
    }

    /// One-word `Ptr(STR)` view (`space.bytes_w`). No window, no
    /// `emit_str_to_window` at the definition.
    pub(in crate::front::mir) fn bind_str_view_local(
        &mut self,
        dest_local: usize,
        value: Variable,
    ) {
        stamp_as_rstr_str(&value);
        self.alias_dest_to_arg0(dest_local, value, true);
    }

    /// A W1 STR source into a window dest (`len_shadow`) borrows
    /// `{chars, length}` here. A STR-view dest stays one `Ptr(STR)` word.
    pub(in crate::front::mir) fn bind_str_source_to_dest(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        header: Variable,
    ) {
        let window_dest = self.len_shadow.get(dest_local).copied().flatten().is_some()
            && self
                .body
                .locals
                .locals
                .get(dest_local)
                .is_some_and(|loc| tyref_is_u8_window(&loc.ty, self.llbc));
        if window_dest {
            let (ptr, len, owner) = self.emit_str_to_window(mir_bb, header);
            let ptr = self.ensure_window_ccharp(mir_bb, ptr);
            self.bind_pair_local(dest_local, ptr, len, Some(owner));
        } else {
            self.bind_str_view_local(dest_local, header);
        }
    }

    pub(in crate::front::mir) fn local_holds_str_view(&self, local: usize) -> bool {
        self.string_byte_view_locals.contains(&local)
            && matches!(
                self.local_var.get(local).and_then(Option::as_ref),
                Some(LocalValue::One(_))
            )
    }

    /// One-word list view of a W2 owner. No window, no
    /// `emit_list_to_window` at the definition.
    pub(in crate::front::mir) fn bind_list_view_local(
        &mut self,
        dest_local: usize,
        list: Variable,
    ) {
        self.string_byte_view_locals
            .retain(|&local| local != dest_local);
        if !self.list_view_locals.contains(&dest_local) {
            self.list_view_locals.push(dest_local);
        }
        self.local_var[dest_local] = Some(LocalValue::One(list));
    }

    pub(in crate::front::mir) fn local_holds_list_view(&self, local: usize) -> bool {
        self.list_view_locals.contains(&local)
            && matches!(
                self.local_var.get(local).and_then(Option::as_ref),
                Some(LocalValue::One(_))
            )
    }

    pub(in crate::front::mir) fn operand_is_list_view(&self, op: &Operand) -> bool {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self
                .list_view_locals
                .iter()
                .any(|&local| place_references_local(place, local)),
            Operand::Const(_) => false,
        }
    }

    pub(in crate::front::mir) fn place_is_list_view(&self, place: &Place) -> bool {
        self.list_view_locals
            .iter()
            .any(|&local| place_references_local(place, local))
    }

    pub(in crate::front::mir) fn list_view_local_of_var(&self, var: &Variable) -> Option<usize> {
        self.list_view_locals.iter().copied().find(|&local| {
            self.local_var
                .get(local)
                .and_then(Option::as_ref)
                .and_then(|value| value.one().ok())
                .is_some_and(|current| current == *var)
        })
    }

    /// Window-boundary conversion of a list view (`rlist.py` `ll_length` /
    /// `ll_getitem_nonneg` keep the list until this point). A fixed-size
    /// owner has no `nonmoving_raw_ptr_for_resizable_list`
    /// (`rgc._check_resizable_list_of_chars`).
    pub(in crate::front::mir) fn emit_list_view_to_window(
        &mut self,
        mir_bb: usize,
        list: Variable,
    ) -> Result<(Variable, Variable), LowerError> {
        let fixed = self
            .list_view_local_of_var(&list)
            .is_some_and(|local| self.list_view_fixed.get(local).copied().unwrap_or(false));
        if fixed {
            return Err(LowerError::Unsupported(
                "fixed-size list has no nonmoving raw ptr (rgc._check_resizable_list_of_chars)"
                    .to_string(),
            ));
        }
        self.emit_list_to_window(mir_bb, list, None)
    }

    /// A one-word owner (STR or list) assigned into a window dest becomes
    /// `{CCHARP, len}` here.
    pub(in crate::front::mir) fn convert_one_word_owner_to_window_dest(
        &mut self,
        mir_bb: usize,
        dest: usize,
        var: Variable,
        src_ty: Option<&TyRef>,
    ) -> Result<(), LowerError> {
        if self.local_holds_str_view(dest) {
            self.bind_str_source_to_dest(mir_bb, dest, var);
            return Ok(());
        }
        let owner = self.owner_for_window_source(&var);
        let (ptr, len) = self.owner_to_window(mir_bb, var, src_ty)?;
        let ptr = self.ensure_window_ccharp(mir_bb, ptr);
        self.bind_pair_local(dest, ptr, len, owner);
        Ok(())
    }

    pub(in crate::front::mir) fn var_is_str_repr(&self, var: &Variable) -> bool {
        if var.concretetype().as_ref()
            == Some(&*crate::translator::rtyper::lltypesystem::rstr::STRPTR)
        {
            return true;
        }
        self.string_byte_view_locals.iter().any(|&local| {
            self.local_var
                .get(local)
                .and_then(Option::as_ref)
                .and_then(|value| value.one().ok())
                .is_some_and(|current| current == *var)
        })
    }

    /// Producer `ValueType::Str` or a `STRPTR` concretetype. Used when the
    /// place's static type is a window but the bound value is already a
    /// string (`rstr.py` `STR`).
    pub(in crate::front::mir) fn var_has_str_value_type(&self, var: &Variable) -> bool {
        if var.concretetype().as_ref()
            == Some(&*crate::translator::rtyper::lltypesystem::rstr::STRPTR)
        {
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
                ty: ValueType::Str,
                ..
            } | OpKind::Call {
                result_ty: ValueType::Str,
                ..
            } | OpKind::Input {
                ty: ValueType::Str,
                ..
            }
        )
    }

    pub(in crate::front::mir) fn rvalue_is_str_view_source(&self, rvalue: &Rvalue) -> bool {
        match rvalue {
            Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                self.operand_is_string_byte_view(op)
                    || operand_tyref(op).is_some_and(|ty| tyref_is_w1_str(ty, self.llbc))
            }
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                self.string_byte_view_locals
                    .iter()
                    .any(|&local| place_references_local(place, local))
                    || tyref_is_w1_str(&place.ty, self.llbc)
            }
            _ => false,
        }
    }

    pub(in crate::front::mir) fn bind_slice_iter(
        &mut self,
        local: usize,
        index: Variable,
        items: Variable,
    ) {
        let len_slot = self.len_shadow[local].expect("slice iterator local has a shadow slot");
        self.local_var[local] = Some(LocalValue::SliceIter {
            items: items.clone(),
            index,
        });
        self.local_var[len_slot] = Some(LocalValue::One(items));
        self.positional_aggregate_locals.remove(&local);
    }

    /// The pair-slice local a `*s` place dereferences.
    pub(in crate::front::mir) fn pair_deref_local(&self, place: &Place) -> Option<usize> {
        let PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) = &place.kind else {
            return None;
        };
        let PlaceKind::Local(local) = inner.kind else {
            return None;
        };
        let local = local as usize;
        (atom == "Deref"
            && self.len_shadow[local].is_some()
            && self
                .pair_slice_kind(&self.body.locals.locals[local].ty)
                .is_some())
        .then_some(local)
    }

    /// `(ptr, len, kind)` of an operand whose type is a pair slice, or
    /// `None` for any other operand.  A pair slice held anywhere but a
    /// local, or spelled as a constant, has no lowering yet.
    pub(in crate::front::mir) fn pair_operand(
        &mut self,
        mir_bb: usize,
        op: &Operand,
    ) -> Result<Option<(Variable, Variable, majit_ir::rvec::VecItemKind)>, LowerError> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Const(_) => {
                if const_operand_tyref(op, self.llbc)
                    .is_some_and(|ty| self.pair_slice_kind(&ty).is_some())
                {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: pair slice constant"
                    )));
                }
                return Ok(None);
            }
        };
        let Some(kind) = self
            .pair_slice_kind(&place.ty)
            .or_else(|| self.pair_iter_kind(&place.ty))
        else {
            return Ok(None);
        };
        match self.pair_place_parts(mir_bb, place)? {
            Some((ptr, len)) => Ok(Some((ptr, len, kind))),
            // A pair-shaped type with no length slot in this frame is the
            // one-word helper ABI (an object-pointer `&mut [T]` parameter).
            None if matches!(place.kind, PlaceKind::Local(_)) => Ok(None),
            None => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice read through a projection"
            ))),
        }
    }

    /// `(ptr, len)` of a pair-slice or window operand.
    pub(in crate::front::mir) fn pair_or_window_operand(
        &mut self,
        mir_bb: usize,
        op: &Operand,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        if let Some((ptr, len, _)) = self.pair_operand(mir_bb, op)? {
            return Ok(Some((ptr, len)));
        }
        self.window_operand(mir_bb, op)
    }

    /// `(ptr, len)` of an aggregate operand or assigned value that holds a
    /// pair slice: a `&[T]`, or a reference to one, which aliases its
    /// referent.  `None` for any other operand.
    pub(in crate::front::mir) fn pair_value_operand(
        &mut self,
        mir_bb: usize,
        op: &Operand,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Const(_) => {
                return Ok(self
                    .pair_operand(mir_bb, op)?
                    .map(|(ptr, len, _)| (ptr, len)));
            }
        };
        if tyref_pair_field_kind(&place.ty, self.llbc).is_none() {
            return Ok(None);
        }
        match self.pair_place_parts(mir_bb, place)? {
            Some(parts) => Ok(Some(parts)),
            None => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice read through a projection"
            ))),
        }
    }

    /// `(ptr, len)` of a place holding a pair slice: a pair local, a
    /// dereference of a reference to one, or an aggregate field that stores
    /// the two words (its pointer under the field's name, its length under
    /// [`pair_len_field_name`]).  `None` for any other place.
    pub(in crate::front::mir) fn pair_place_parts(
        &mut self,
        mir_bb: usize,
        place: &Place,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        match &place.kind {
            PlaceKind::Local(local) if self.len_shadow[*local as usize].is_some() => {
                match self
                    .local_var
                    .get(*local as usize)
                    .and_then(|slot| slot.as_ref())
                {
                    Some(LocalValue::One(_))
                        if self.local_holds_str_view(*local as usize)
                            || self.local_holds_list_view(*local as usize) =>
                    {
                        Ok(None)
                    }
                    _ => {
                        let (ptr, len) = self.pair_local_parts(mir_bb, *local as usize)?;
                        let data = if tyref_is_u8_window(&place.ty, self.llbc) {
                            self.ensure_window_ccharp(mir_bb, ptr)
                        } else {
                            ptr
                        };
                        Ok(Some((data, len)))
                    }
                }
            }
            PlaceKind::Local(local) if self.local_holds_str_view(*local as usize) => Ok(None),
            PlaceKind::Local(local) if self.local_holds_list_view(*local as usize) => Ok(None),
            // `*s` of a window / pair slice is the fat value `s` already
            // holds: the referent is unsized, so the borrow is the pair.
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                if atom == "Deref"
                    && (tyref_is_u8_window(&inner.ty, self.llbc)
                        || self.pair_slice_kind(&inner.ty).is_some()) =>
            {
                self.pair_place_parts(mir_bb, inner)
            }
            // `*r` for a reference `r` to a pair is the pair `r` aliases.
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                if atom == "Deref"
                    && tyref_pair_slice_item_kind(&inner.ty, self.llbc).is_none()
                    && tyref_pair_field_kind(&inner.ty, self.llbc).is_some() =>
            {
                self.pair_place_parts(mir_bb, inner)
            }
            // `Some(w)` of a niche u8-window Option is `w`: the payload
            // `Field` (and a `Downcast` onto `Some`) is the Option's
            // `{data, len}`, the same pair `Discriminant` reads. A
            // tagged-instance `__pos_0` would rebuild a list.
            PlaceKind::Projection(inner, ProjectionElem::Tagged(v))
                if (v.get("Field").is_some() || v.get("Downcast").is_some())
                    && tyref_is_u8_window(&inner.ty, self.llbc) =>
            {
                self.pair_place_parts(mir_bb, inner)
            }
            PlaceKind::Projection(_, ProjectionElem::Tagged(v))
                if v.get("Field").is_some()
                    && tyref_pair_field_kind(&place.ty, self.llbc).is_some() =>
            {
                self.pair_field_parts(mir_bb, place)
            }
            _ if tyref_is_u8_window(&place.ty, self.llbc)
                && !matches!(place.kind, PlaceKind::Local(_)) =>
            {
                let var = self.resolve_place(mir_bb, place.clone())?;
                Ok(Some(self.split_window_value(mir_bb, var)?))
            }
            _ => Ok(None),
        }
    }

    /// The two words of the pair slice aggregate field `place`: the field
    /// read the projection lowers to, retyped to the pointer word, and the
    /// same read of the length word.
    pub(in crate::front::mir) fn pair_field_parts(
        &mut self,
        mir_bb: usize,
        place: &Place,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let ptr = self.resolve_place_as(mir_bb, place.clone(), true)?;
        let block = self.graph.block_mut(bb_id);
        let Some(SpaceOperation {
            result: Some(result),
            kind: OpKind::FieldRead {
                base, field, pure, ..
            },
        }) = block.operations.last_mut()
        else {
            return Ok(None);
        };
        if *result != ptr {
            return Ok(None);
        }
        let len_read = OpKind::FieldRead {
            base: base.clone(),
            field: FieldDescriptor {
                name: pair_len_field_name(&field.name),
                ..field.clone()
            },
            ty: ValueType::Unsigned,
            pure: *pure,
        };
        if let Some(SpaceOperation {
            kind: OpKind::FieldRead { ty, .. },
            ..
        }) = block.operations.last_mut()
        {
            *ty = ValueType::Unsigned;
        }
        let len = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(len.clone()),
            kind: len_read,
        });
        Ok(Some((ptr, len)))
    }

    /// Bind `dest_local` to the object array copied from a `Vec<PyObjectRef>`.
    pub(in crate::front::mir) fn emit_object_vec_gcarray(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        header: Variable,
        target: usize,
    ) -> Result<(), LowerError> {
        let view = self.emit_path_call(
            mir_bb,
            "pyre_object::gc_roots::gcarray_from_pyobject_vec",
            vec![header],
            ValueType::Ref(None),
        );
        // The residual copies the vec into a length-prefixed object
        // GcArray (`ll_fixed_items`). A list that is never resized is
        // `FixedSizeListRepr`; its annotation is the position ListDef
        // (`Bookkeeper.getlistdef`). Narrow through the destination
        // slice's `[Class]` root so `cast_instance_intrinsic` reuses
        // that ListDef (`Bookkeeper.newlist`). The helper's stub body
        // returns `null_mut`, which would otherwise bind
        // `SomeInstance(classdef=None)` and fail `union` with List.
        let bb_id = self.emit_bb(mir_bb);
        let dest_ty = {
            let local = self.body.locals.locals.get(dest_local).ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: object vec gcarray dest local {dest_local} is missing"
                ))
            })?;
            clone_tyref(&local.ty)
        };
        let list_root = object_pointer_slice_list_root(&dest_ty, self.llbc).ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: object vec gcarray dest is not an object-pointer slice"
            ))
        })?;
        let list = self.narrow_value_to_instance_root(bb_id, LinkArg::Value(view), &list_root);
        self.local_var[dest_local] = Some(LocalValue::One(
            list.as_variable()
                .expect("cast_instance of a Variable stays a Variable")
                .clone(),
        ));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `&vec`, `&mut vec`, `&vec[..]`, `&mut vec[..]` of a `Vec<PyObjectRef>`
    /// are the same `SomeList` value. RPython has no borrow wrapper.
    ///
    /// Shared views copy the items into one length-prefixed object array
    /// (`gcarray_from_pyobject_vec`). Mutable views keep the header address
    /// so a write, or `shadow_stack_copy_range`, updates that vec.
    pub(in crate::front::mir) fn try_lower_object_vec_whole_list_view(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
        reg: &RegularCall,
    ) -> Result<bool, LowerError> {
        let Some(view) = object_vec_whole_list_view(reg, call.args.len(), self.llbc) else {
            return Ok(false);
        };
        let Some(first_ty) = call.args.first().and_then(operand_tyref) else {
            return Ok(false);
        };
        if !tyref_is_object_pointer_rust_vec(first_ty, self.llbc) {
            return Ok(false);
        }
        let header = self.resolve_operand(mir_bb, call.args[0].clone())?;
        let inherit = operand_local(call.args.first())
            .is_some_and(|local| self.string_byte_view_locals.contains(&local));
        self.bind_object_vec_whole_list_view(mir_bb, dest_local, header, target, view, inherit)?;
        Ok(true)
    }

    /// The shared object-vec whole-list arm. [`Self::try_lower_object_vec_whole_list_view`]
    /// is the only caller; Deref and RangeFull both reach it.
    pub(in crate::front::mir) fn bind_object_vec_whole_list_view(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        header: Variable,
        target: usize,
        view: ObjectVecWholeListView,
        inherit_byte_view: bool,
    ) -> Result<(), LowerError> {
        if view == ObjectVecWholeListView::Shared
            && !rust_vec_deref_feeds_only_reverse(self.body, self.llbc, dest_local)
        {
            return self.emit_object_vec_gcarray(mir_bb, dest_local, header, target);
        }
        if view == ObjectVecWholeListView::Mut {
            self.note_object_vec_header(&header);
        }
        self.alias_dest_to_arg0(dest_local, header, inherit_byte_view);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph
            .set_goto(self.block_id[mir_bb], target_bb, link_args);
        Ok(())
    }

    /// Emit a call of the helper at `path` and return its result.
    pub(in crate::front::mir) fn emit_path_call(
        &mut self,
        mir_bb: usize,
        path: &str,
        args: Vec<Variable>,
        result_ty: ValueType,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let result = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: path.split("::").map(str::to_string).collect(),
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(args),
                result_ty,
            },
        });
        result
    }

    /// Re-type a word between `Signed` and `Unsigned`: `intmask` one way,
    /// `r_uint` the other, the identity on the machine word.
    pub(in crate::front::mir) fn retype_word(
        &mut self,
        mir_bb: usize,
        value: Variable,
        from: &ValueType,
        to: &ValueType,
    ) -> Variable {
        let path = match (from, to) {
            (ValueType::Unsigned, ValueType::Int) => "rpython::rlib::rarithmetic::intmask",
            (ValueType::Int, ValueType::Unsigned) => "rpython::rlib::rarithmetic::r_uint",
            // A reference item enters a `_r` helper as `GCREF` and leaves it
            // narrowed back to the item's instance root, the `cast_opaque_ptr`
            // pair `GcArray(GCREF)` items take (`rgcref.py`).
            (ValueType::Ref(_), ValueType::Ref(None)) => {
                return self.cast_ref_word(mir_bb, value, "GCREF");
            }
            (ValueType::Ref(None), ValueType::Ref(Some(root))) => {
                let root = root.clone();
                return self.cast_ref_word(mir_bb, value, &root);
            }
            _ => return value,
        };
        self.emit_path_call(mir_bb, path, vec![value], to.clone())
    }

    /// `item_ty` carrying the class root of `ty` (an item, or a reference
    /// to one) when it is a reference, the root a `_r` helper's result is
    /// narrowed back to.
    pub(in crate::front::mir) fn with_item_class_root(
        &self,
        item_ty: ValueType,
        ty: &TyRef,
    ) -> ValueType {
        match item_ty {
            ValueType::Ref(None) => {
                ValueType::Ref(tyref_class_root_with(ty, self.llbc, self.tombstoned_leaves))
            }
            other => other,
        }
    }

    /// `__cast_instance_intrinsic(value, root)` in the current block.
    pub(in crate::front::mir) fn cast_ref_word(
        &mut self,
        mir_bb: usize,
        value: Variable,
        root: &str,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let cast = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(cast.clone()),
            kind: if root == "GCREF" {
                crate::model::cast_instance_call_result(root, value, ValueType::Ref(None))
            } else {
                crate::model::cast_instance_call(root, value)
            },
        });
        cast
    }

    /// The address of item 0 of the items block whose header `block` holds:
    /// `cast_ptr_to_int(block) + <items offset>`, as an Unsigned pair
    /// pointer.  `offset_path` names the block's items-offset constant.
    pub(in crate::front::mir) fn items_block_item_address(
        &mut self,
        mir_bb: usize,
        block: Variable,
        offset_path: &[&str],
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let header = push_cast_ptr_to_int(&mut self.graph, bb_id, block);
        let offset_path: Vec<String> = offset_path.iter().map(|s| s.to_string()).collect();
        let offset_op =
            known_array_layout_const(&offset_path).expect("items block offset is known");
        let offset = self
            .graph
            .push_op_var(bb_id, offset_op, true)
            .expect("offset constant");
        let addr = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(addr.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: header,
                rhs: offset,
                result_ty: ValueType::Int,
            },
        });
        self.retype_word(mir_bb, addr, &ValueType::Int, &ValueType::Unsigned)
    }

    /// Item address for `ptr::copy_nonoverlapping`. A
    /// `typed_items_block_items_base` / `items_block_items_base` result is
    /// the array header (`l.items`); `raw_memcopy` wants `items[0]`, so
    /// add `itemoffsetof`. A pair-slice `as_ptr` is already that address.
    pub(in crate::front::mir) fn copy_nonoverlapping_item_addr(
        &mut self,
        mir_bb: usize,
        value: Variable,
        local: Option<usize>,
        ty: Option<&TyRef>,
    ) -> Option<Variable> {
        if local.is_some_and(|base| {
            base_traces_to_typed_items_block_accessor(self.body, base, self.llbc)
        }) {
            return Some(self.items_block_item_address(
                mir_bb,
                value,
                &[
                    "majit_rlib",
                    "lltypesystem",
                    "rlist",
                    "TYPED_ITEMS_BLOCK_ITEMS_OFFSET",
                ],
            ));
        }
        if local.is_some_and(|base| base_traces_to_items_block_accessor(self.body, base, self.llbc))
        {
            return Some(self.items_block_item_address(
                mir_bb,
                value,
                &["pyre_object", "object_array", "ITEMS_BLOCK_ITEMS_OFFSET"],
            ));
        }
        let ty = ty?;
        raw_ptr_one_word_item_kind(ty, self.llbc)?;
        let from = tyref_to_value_type_with(
            ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        Some(self.retype_word(mir_bb, value, &from, &ValueType::Unsigned))
    }

    /// The pair's item word type for a slice helper: the helpers take and
    /// return `usize` for an int item.
    pub(in crate::front::mir) fn pair_helper_item_ty(
        kind: majit_ir::rvec::VecItemKind,
    ) -> ValueType {
        match kind {
            majit_ir::rvec::VecItemKind::Int => ValueType::Unsigned,
            majit_ir::rvec::VecItemKind::Ref => ValueType::Ref(None),
            majit_ir::rvec::VecItemKind::Float => ValueType::Float,
        }
    }

    /// The item sequence `place` denotes: `*s` over a pair slice local, or
    /// a raw-buffer array spelled `a` or `*b`.
    pub(in crate::front::mir) fn item_seq_of_place(&self, place: &Place) -> Option<ItemSeq> {
        if let Some(local) = self.pair_deref_local(place) {
            return Some(ItemSeq::Pair(local));
        }
        match &place.kind {
            PlaceKind::Local(a) => match self.raw_array[*a as usize] {
                Some(RawArrayRole::Storage { .. }) => Some(ItemSeq::RawArray(*a as usize)),
                _ => None,
            },
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                match inner.kind {
                    PlaceKind::Local(b) => match self.raw_array[b as usize] {
                        Some(RawArrayRole::Borrow(a)) => Some(ItemSeq::RawArray(a)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The item pointer and item kind of `seq`.
    pub(in crate::front::mir) fn item_seq_ptr(
        &self,
        mir_bb: usize,
        seq: ItemSeq,
    ) -> Result<(Variable, majit_ir::rvec::VecItemKind), LowerError> {
        match seq {
            ItemSeq::Pair(local) => {
                let (ptr, _) = self.pair_local_parts(mir_bb, local)?;
                Ok((ptr, self.pair_local_kind(local).expect("pair slice local")))
            }
            ItemSeq::RawArray(storage) => {
                let Some(RawArrayRole::Storage { kind, .. }) = self.raw_array[storage] else {
                    unreachable!("raw array storage local");
                };
                let ptr = self.local_var[storage]
                    .as_ref()
                    .and_then(|value| value.one().ok())
                    .expect("a raw array buffer is bound in every block");
                Ok((ptr, kind))
            }
        }
    }

    /// The length word of `seq`.
    pub(in crate::front::mir) fn item_seq_len(
        &mut self,
        mir_bb: usize,
        seq: ItemSeq,
    ) -> Result<Variable, LowerError> {
        match seq {
            ItemSeq::Pair(local) => Ok(self.pair_local_parts(mir_bb, local)?.1),
            ItemSeq::RawArray(storage) => {
                let Some(RawArrayRole::Storage { len, .. }) = self.raw_array[storage] else {
                    unreachable!("raw array storage local");
                };
                Ok(self.emit_const_uint(mir_bb, len))
            }
        }
    }

    /// Field literals of `rvalue` when it copies a multi-field integer
    /// `NamedConst`, or a local already bound to one.
    pub(in crate::front::mir) fn take_const_struct_fields(
        &self,
        rvalue: &Rvalue,
    ) -> Option<Vec<u64>> {
        let Rvalue::Use(Operand::Copy(place) | Operand::Move(place), _) = rvalue else {
            return None;
        };
        match &place.kind {
            PlaceKind::Global { id, .. } => const_struct_field_lits(self.llbc, *id),
            PlaceKind::Local(src) => self
                .const_struct_fields
                .get(*src as usize)
                .and_then(|fields| fields.clone()),
            _ => None,
        }
    }

    /// `Const*` of one integer field of a const-struct local or global.
    /// `Ok(None)` leaves the projection to the ordinary field read.
    pub(in crate::front::mir) fn folded_const_struct_field(
        &mut self,
        mir_bb: usize,
        inner: &Place,
        elem: &ProjectionElem,
        place_ty: &TyRef,
    ) -> Result<Option<Variable>, LowerError> {
        let ProjectionElem::Tagged(tagged) = elem else {
            return Ok(None);
        };
        let Some(payload) = tagged.as_object().and_then(|map| map.get("Field")) else {
            return Ok(None);
        };
        let Some((variant, index)) = indexed_field_parts(payload) else {
            return Ok(None);
        };
        if variant.is_some() {
            return Ok(None);
        }
        let fields = match &inner.kind {
            PlaceKind::Local(local) => self
                .const_struct_fields
                .get(*local as usize)
                .and_then(|fields| fields.clone()),
            PlaceKind::Global { id, .. } => const_struct_field_lits(self.llbc, *id),
            _ => None,
        };
        let Some(fields) = fields else {
            return Ok(None);
        };
        let Some(bits) = fields.get(index).copied() else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: const struct field {index} out of range"
            )));
        };
        let ty = tyref_to_value_type_with(
            place_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        let kind = match ty {
            ValueType::Unsigned => OpKind::ConstUInt(bits),
            ValueType::Int => {
                let Some(value) = i64::try_from(bits).ok() else {
                    return Ok(None);
                };
                OpKind::ConstInt(value)
            }
            ValueType::Bool => OpKind::ConstBool(bits != 0),
            _ => return Ok(None),
        };
        let bb_id = self.block_id[mir_bb];
        let var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(var.clone()),
            kind,
        });
        Ok(Some(var))
    }

    /// A `ConstUInt(value)` in `mir_bb`.
    pub(in crate::front::mir) fn emit_const_uint(&mut self, mir_bb: usize, value: u64) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(var.clone()),
            kind: OpKind::ConstUInt(value),
        });
        var
    }

    /// Unsigned `lhs op rhs` in `mir_bb`.
    pub(in crate::front::mir) fn emit_uint_binop(
        &mut self,
        mir_bb: usize,
        op: &str,
        lhs: Variable,
        rhs: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: op.to_string(),
                lhs,
                rhs,
                result_ty: ValueType::Unsigned,
            },
        });
        res
    }

    /// `cond ? then_word : else_word` for a 0/1 unsigned `cond`, as
    /// `cond * then + (1 - cond) * else`.
    pub(in crate::front::mir) fn emit_uint_select(
        &mut self,
        mir_bb: usize,
        cond: Variable,
        then_word: Variable,
        else_word: Variable,
    ) -> Variable {
        let one = self.emit_const_uint(mir_bb, 1);
        let not_cond = self.emit_uint_binop(mir_bb, "sub", one, cond.clone());
        let then_part = self.emit_uint_binop(mir_bb, "mul", cond, then_word);
        let else_part = self.emit_uint_binop(mir_bb, "mul", not_cond, else_word);
        self.emit_uint_binop(mir_bb, "add", then_part, else_part)
    }

    /// The index an `Index` projection over `seq` selects: `offset`, or
    /// `len - offset` when it counts from the end.
    pub(in crate::front::mir) fn item_seq_index(
        &mut self,
        mir_bb: usize,
        seq: ItemSeq,
        index_payload: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        let offset = self.index_offset_var(mir_bb, index_payload)?;
        if index_payload
            .get("from_end")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Ok(offset);
        }
        let len = self.item_seq_len(mir_bb, seq)?;
        Ok(self.emit_index_from_end(mir_bb, len, offset))
    }

    /// `len - offset`, the index a `from_end` projection names.
    pub(in crate::front::mir) fn emit_index_from_end(
        &mut self,
        mir_bb: usize,
        len: Variable,
        offset: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let index = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(index.clone()),
            kind: OpKind::BinOp {
                op: "sub".to_string(),
                lhs: len,
                rhs: offset,
                result_ty: ValueType::Unsigned,
            },
        });
        index
    }

    /// Graph block that currently receives operations for `mir_bb`.
    pub(in crate::front::mir) fn emit_bb(&self, mir_bb: usize) -> BlockId {
        self.emit_block[mir_bb]
    }

    /// Copy variable-keyed provenance onto the in-bounds tail's fresh
    /// inputargs. Locals are remapped separately through [`LocalValue::remap_words`].
    pub(in crate::front::mir) fn remap_forwarded_provenance(
        &mut self,
        remap: &std::collections::HashMap<u64, Variable>,
    ) {
        if remap.is_empty() {
            return;
        }
        let rewrite = |var: &Variable| remap.get(&var.id()).cloned().unwrap_or_else(|| var.clone());
        let ptr_src = std::mem::take(&mut self.cast_ptr_to_int_src);
        self.cast_ptr_to_int_src = ptr_src
            .into_iter()
            .map(|(k, v)| (rewrite(&k), rewrite(&v)))
            .collect();
        let raw_src = std::mem::take(&mut self.raw_address_uint_src);
        self.raw_address_uint_src = raw_src
            .into_iter()
            .map(|(k, v)| (rewrite(&k), rewrite(&v)))
            .collect();
        let pointer_word = std::mem::take(&mut self.pointer_word_vars);
        self.pointer_word_vars = pointer_word.into_iter().map(|v| rewrite(&v)).collect();
        let guard_word = std::mem::take(&mut self.guard_word_vars);
        self.guard_word_vars = guard_word.into_iter().map(|v| rewrite(&v)).collect();
        let pointer_arrays = std::mem::take(&mut self.pointer_word_arrays);
        self.pointer_word_arrays = pointer_arrays.into_iter().map(|v| rewrite(&v)).collect();
        let niche = std::mem::take(&mut self.niche_disc_vars);
        self.niche_disc_vars = niche
            .into_iter()
            .map(|id| remap.get(&id).map(Variable::id).unwrap_or(id))
            .collect();
        for (_, var) in &mut self.string_array_view_locals {
            if let Some(copy) = remap.get(&var.id()) {
                *var = copy.clone();
            }
        }
        for (_, var) in &mut self.numeric_array_view_locals {
            if let Some(copy) = remap.get(&var.id()) {
                *var = copy.clone();
            }
        }
        for (orig, copy) in remap {
            self.input_copied_from.insert(copy.id(), *orig);
            if self.object_vec_headers.contains(orig) {
                self.object_vec_headers.insert(copy.id());
            }
        }
    }

    /// Split the current block on `cond`. The true arm is in bounds and
    /// becomes the emit cursor for `mir_bb`; the false arm calls
    /// `ll_slice_bounds_panic`. Incoming edges still target [`Self::block_id`].
    /// Locals defined in the head, and every `carry` word, are forwarded
    /// onto the true arm. Returns `carry` as those forwarded copies.
    pub(in crate::front::mir) fn fork_in_bounds(
        &mut self,
        mir_bb: usize,
        cond: Variable,
        carry: &[Variable],
    ) -> Result<Vec<Variable>, LowerError> {
        let head = self.emit_bb(mir_bb);
        for word in carry {
            if !self.graph.variable_defined_in_block(head, word) {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice bounds guard carries a word defined outside the block"
                )));
            }
        }
        let mut forwarded: Vec<Variable> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for slot in &self.local_var {
            let Some(value) = slot else {
                continue;
            };
            for word in value.words() {
                if self.graph.variable_defined_in_block(head, &word) && seen.insert(word.id()) {
                    forwarded.push(word);
                }
            }
        }
        for word in carry {
            if seen.insert(word.id()) {
                forwarded.push(word.clone());
            }
        }
        let copies: Vec<Variable> = forwarded.iter().map(Variable::copy).collect();
        let tail = self.graph.create_block();
        for copy in &copies {
            self.graph.push_inputarg_var(tail, copy.clone());
        }
        let remap: std::collections::HashMap<u64, Variable> = forwarded
            .iter()
            .zip(copies.iter())
            .map(|(orig, copy)| (orig.id(), copy.clone()))
            .collect();
        for value in self.local_var.iter_mut().flatten() {
            value.remap_words(&remap);
        }
        self.remap_forwarded_provenance(&remap);
        let fail = self.graph.create_block();
        let panic = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(fail).operations.push(SpaceOperation {
            result: Some(panic),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: majit_ir::rvec::SLICE_BOUNDS_PANIC
                        .split("::")
                        .map(str::to_string)
                        .collect(),
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Void,
            },
        });
        self.graph.set_raise_implicit(fail, "IndexError");
        self.graph
            .set_branch(head, cond, tail, forwarded, fail, Vec::new());
        self.emit_block.set(mir_bb, tail);
        Ok(carry
            .iter()
            .map(|word| {
                remap
                    .get(&word.id())
                    .cloned()
                    .expect("carried word was forwarded")
            })
            .collect())
    }

    /// `index < len`, then the in-bounds copies of `ptr` and `index`.
    pub(in crate::front::mir) fn guard_item_index(
        &mut self,
        mir_bb: usize,
        len: Variable,
        ptr: Variable,
        index: Variable,
    ) -> Result<(Variable, Variable), LowerError> {
        let in_bounds = self.emit_uint_binop(mir_bb, "lt", index.clone(), len);
        let carried = self.fork_in_bounds(mir_bb, in_bounds, &[ptr, index])?;
        Ok((carried[0].clone(), carried[1].clone()))
    }

    /// `seq[i]`: `ll_slice_getitem_fast` after `index < len`.
    pub(in crate::front::mir) fn pair_getitem(
        &mut self,
        mir_bb: usize,
        seq: ItemSeq,
        index_payload: &serde_json::Value,
        item_ty: &ValueType,
    ) -> Result<Variable, LowerError> {
        let (ptr, kind) = self.item_seq_ptr(mir_bb, seq)?;
        let index = self.item_seq_index(mir_bb, seq, index_payload)?;
        let len = self.item_seq_len(mir_bb, seq)?;
        let (ptr, index) = self.guard_item_index(mir_bb, len, ptr, index)?;
        let helper_ty = Self::pair_helper_item_ty(kind);
        let item = self.emit_path_call(
            mir_bb,
            majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::GetItem, kind),
            vec![ptr, index],
            helper_ty.clone(),
        );
        Ok(self.retype_word(mir_bb, item, &helper_ty, item_ty))
    }

    /// The item `next` over pair-slice iterator `iter` yielded into `opt`:
    /// `ll_slice_getitem_fast(ptr, index)`.
    pub(in crate::front::mir) fn pair_iter_item(
        &mut self,
        mir_bb: usize,
        payload: &Place,
        opt: usize,
        payload_ty: &TyRef,
    ) -> Result<Variable, LowerError> {
        let iter = self
            .slice_next_iter
            .get(opt)
            .copied()
            .flatten()
            .ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice iterator step local {opt} has no iterator"
                ))
            })?;
        let ptr = match self.local_var.get(iter).and_then(|slot| slot.as_ref()) {
            Some(LocalValue::SliceIter { items, .. }) => items.clone(),
            Some(value) => {
                value
                    .pair()
                    .map_err(|_| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: slice iterator local {iter} has no item pointer"
                        ))
                    })?
                    .1
            }
            None => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice iterator local {iter} has no item pointer"
                )));
            }
        };
        let kind = self.pair_local_kind(iter).ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: slice iterator payload is not Option<&T>"
            ))
        })?;
        // The payload read itself is the index: the `Some` payload the range
        // iterator's `next` yields. Hide the step so this field read is the
        // option word, not another item load.
        let saved = self.local_var[opt].clone();
        if let Some(LocalValue::SliceIter { index: cursor, .. }) = &saved {
            self.local_var[opt] = Some(LocalValue::One(cursor.clone()));
        }
        let saved_iter = self.slice_next_iter[opt];
        self.slice_next_iter[opt] = None;
        let index = self.resolve_place(mir_bb, payload.clone());
        self.slice_next_iter[opt] = saved_iter;
        self.local_var[opt] = saved;
        let index = index?;
        let index = self.retype_word(mir_bb, index, &ValueType::Int, &ValueType::Unsigned);
        // `IterMut` yields `&mut T`: the item's address in the slice buffer,
        // so a later `*slot = v` writes through `ptr`. Shared `Iter` yields
        // the loaded item, the list-iteration value.
        if ref_kind_is_mut(payload_ty, self.llbc) {
            let (size, _) = kind.size_align(crate::layout::target_word_size());
            let size = self.emit_const_uint(mir_bb, size as u64);
            let offset = self.emit_uint_binop(mir_bb, "mul", index, size);
            return Ok(self.emit_uint_binop(mir_bb, "add", ptr, offset));
        }
        let item_ty = tyref_ref_pointee_value_type(
            payload_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
        .ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: slice iterator payload is not a reference"
            ))
        })?;
        let item_ty = self.with_item_class_root(item_ty, payload_ty);
        let helper_ty = Self::pair_helper_item_ty(kind);
        let item = self.emit_path_call(
            mir_bb,
            majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::GetItem, kind),
            vec![ptr, index],
            helper_ty.clone(),
        );
        Ok(self.retype_word(mir_bb, item, &helper_ty, &item_ty))
    }

    /// `seq[i] = value`: `ll_slice_setitem_fast`.
    pub(in crate::front::mir) fn pair_setitem(
        &mut self,
        mir_bb: usize,
        seq: ItemSeq,
        index_payload: &serde_json::Value,
        value: Variable,
        value_ty: &ValueType,
    ) -> Result<(), LowerError> {
        let (ptr, kind) = self.item_seq_ptr(mir_bb, seq)?;
        let index = self.item_seq_index(mir_bb, seq, index_payload)?;
        self.emit_slice_setitem(mir_bb, ptr, index, kind, value, value_ty);
        Ok(())
    }

    /// `ll_slice_setitem_fast(ptr, index, value)`.
    pub(in crate::front::mir) fn emit_slice_setitem(
        &mut self,
        mir_bb: usize,
        ptr: Variable,
        index: Variable,
        kind: majit_ir::rvec::VecItemKind,
        value: Variable,
        value_ty: &ValueType,
    ) {
        let helper_ty = Self::pair_helper_item_ty(kind);
        let value = self.retype_word(mir_bb, value, value_ty, &helper_ty);
        self.emit_path_call(
            mir_bb,
            majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::SetItem, kind),
            vec![ptr, index, value],
            ValueType::Void,
        );
    }

    /// `a = [x, y, ..]` / `a = [x; N]` into raw-buffer array `a`: item stores
    /// into its buffer.
    pub(in crate::front::mir) fn lower_raw_array_assign(
        &mut self,
        mir_bb: usize,
        storage: usize,
        rvalue: Rvalue,
    ) -> Result<(), LowerError> {
        let Some(RawArrayRole::Storage { kind, len }) = self.raw_array[storage] else {
            unreachable!("raw array storage local");
        };
        let item_ty = tyref_raw_array_parts(&self.body.locals.locals[storage].ty, self.llbc)
            .map(|(item, _)| {
                tyref_to_value_type_with(
                    &TyRef::Other(item.clone()),
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            })
            .expect("raw array local has an array type");
        let ptr = self.local_var[storage]
            .as_ref()
            .and_then(|value| value.one().ok())
            .expect("a raw array buffer is bound in every block");
        match rvalue {
            Rvalue::Aggregate(_, ops) => {
                let values = ops
                    .into_iter()
                    .map(|op| self.resolve_operand(mir_bb, op))
                    .collect::<Result<Vec<_>, _>>()?;
                for (at, value) in values.into_iter().enumerate() {
                    let index = self.emit_const_uint(mir_bb, at as u64);
                    self.emit_slice_setitem(mir_bb, ptr.clone(), index, kind, value, &item_ty);
                }
            }
            Rvalue::Repeat(op, ..) => {
                let value = self.resolve_operand(mir_bb, op)?;
                let helper_ty = Self::pair_helper_item_ty(kind);
                let value = self.retype_word(mir_bb, value, &item_ty, &helper_ty);
                let length = self.emit_const_uint(mir_bb, len);
                self.emit_path_call(
                    mir_bb,
                    majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::BufferFill, kind),
                    vec![ptr, length, value],
                    ValueType::Void,
                );
            }
            other => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: raw array local {storage} from {}",
                    rvalue_variant_name(&other)
                )));
            }
        }
        Ok(())
    }

    /// `dest = rvalue` into pair slice local `dest`. A copy or reborrow of
    /// another pair takes both words. An `Unsize` of `&[T; N]` binds the
    /// array address and `N`.
    pub(in crate::front::mir) fn lower_pair_assign(
        &mut self,
        mir_bb: usize,
        dest: usize,
        rvalue: Rvalue,
    ) -> Result<(), LowerError> {
        let dest_ty = clone_tyref(&self.body.locals.locals[dest].ty);
        if self.rvalue_is_str_view_source(&rvalue) && tyref_is_u8_window(&dest_ty, self.llbc) {
            // Dest has a length shadow: it is a window. Convert on this
            // write (`rffi.get_nonmovingbuffer` / `ll_chars`), not at the
            // STR-view definition.
            let header = match &rvalue {
                Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                    self.resolve_operand(mir_bb, op.clone())?
                }
                Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                    self.resolve_place(mir_bb, place.clone())?
                }
                _ => {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: STR view local {dest} from {}",
                        rvalue_variant_name(&rvalue)
                    )));
                }
            };
            let (ptr, len, owner) = self.emit_str_to_window(mir_bb, header);
            let ptr = self.ensure_window_ccharp(mir_bb, ptr);
            self.bind_pair_local(dest, ptr, len, Some(owner));
            return Ok(());
        }
        // A reference to a pair slice aliases the pair it borrows.
        let dest_is_pair_ref = tyref_pair_slice_item_kind(&dest_ty, self.llbc).is_none()
            && self.pair_iter_kind(&dest_ty).is_none();
        let source = match &rvalue {
            Rvalue::Use(op, _) => match self.pair_value_operand(mir_bb, op)? {
                Some(pair) => Some(pair),
                None => self
                    .pair_operand(mir_bb, op)?
                    .map(|(ptr, len, _)| (ptr, len)),
            },
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } if dest_is_pair_ref => {
                self.pair_place_parts(mir_bb, place)?
            }
            // `&*s` reborrows the pair / window `s` holds.
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => match &place.kind {
                PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                    if atom == "Deref"
                        && (self.pair_slice_kind(&inner.ty).is_some()
                            || tyref_is_u8_window(&inner.ty, self.llbc)) =>
                {
                    self.pair_place_parts(mir_bb, inner)?
                }
                _ => None,
            },
            Rvalue::UnaryOp(cast, Operand::Copy(src) | Operand::Move(src))
                if cast_kind_is_unsize(cast) =>
            {
                self.pair_unsize_source(mir_bb, src)
                    .map(|(ptr, len, _)| (ptr, len))
            }
            Rvalue::Cast(cast, op, _) => match op {
                Operand::Copy(src) | Operand::Move(src) if cast_kind_is_unsize(cast) => {
                    match self.pair_unsize_source(mir_bb, src) {
                        Some(pair) => Ok(Some(pair)),
                        None => self.pair_operand(mir_bb, op),
                    }?
                }
                _ => self.pair_operand(mir_bb, op)?,
            }
            .map(|(ptr, len, _)| (ptr, len)),
            _ => None,
        };
        let source = match source {
            Some(pair) => Some(pair),
            None => self.window_rvalue_parts(mir_bb, &rvalue)?,
        };
        let Some((ptr, len)) = source else {
            let detail = match &rvalue {
                Rvalue::Cast(kind, ..) => format!("Cast {kind}"),
                other => rvalue_variant_name(other).to_string(),
            };
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice local {dest} from {detail}"
            )));
        };
        let ptr = if tyref_is_u8_window(&dest_ty, self.llbc) {
            self.ensure_window_ccharp(mir_bb, ptr)
        } else {
            ptr
        };
        let owner = self
            .owner_for_window_source(&ptr)
            .or_else(|| match &rvalue {
                Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                    operand_local(Some(op)).and_then(|l| self.window_owner_of_local(l))
                }
                Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => match &place.kind {
                    PlaceKind::Local(l) => self.window_owner_of_local(*l as usize),
                    PlaceKind::Projection(inner, _) => match &inner.kind {
                        PlaceKind::Local(l) => self.window_owner_of_local(*l as usize),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            });
        self.bind_pair_local(dest, ptr, len, owner);
        Ok(())
    }

    /// Item kind of MIR local `local` when it is `Option<&T>` or `Option<T>`
    /// and `T` is a one-word slice item. `by_ref` is the `&` in `Option<&T>`.
    pub(in crate::front::mir) fn option_pair_item(
        &self,
        local: usize,
    ) -> Option<(majit_ir::rvec::VecItemKind, bool)> {
        let ty = &self.body.locals.locals.get(local)?.ty;
        let payload = crate::front::result_exc::tyref_option_payload(ty, self.llbc)?;
        let (item, by_ref) = match ref_pointee_ty(&payload, self.llbc) {
            Some(inner) => (inner, true),
            None => (payload, false),
        };
        let spelling = tyref_to_ast_string(&item, self.llbc);
        let kind = majit_ir::rvec::rust_slice_item_kind_for_spelling(
            &format!("[{spelling}]"),
            crate::layout::target_word_size(),
        )?;
        PAIR_SLICE_ITEM_KINDS
            .contains(&kind)
            .then_some((kind, by_ref))
    }

    /// Item kind and pointee bank of a `&mut T` / `&T` local that holds a
    /// pair-slice item address. `None` for an `Option<&T>` address (the
    /// `first` / `get` result), which is not a store slot.
    pub(in crate::front::mir) fn item_addr_slot_kind(
        &self,
        local: usize,
    ) -> Option<(majit_ir::rvec::VecItemKind, ValueType)> {
        let ty = &self.body.locals.locals.get(local)?.ty;
        if crate::front::result_exc::tyref_is_option(ty, self.llbc) {
            return None;
        }
        let pointee = ref_pointee_ty(ty, self.llbc)?;
        let spelling = tyref_to_ast_string(&pointee, self.llbc);
        let kind = majit_ir::rvec::rust_slice_item_kind_for_spelling(
            &format!("[{spelling}]"),
            crate::layout::target_word_size(),
        )?;
        PAIR_SLICE_ITEM_KINDS.contains(&kind).then(|| {
            (
                kind,
                self.with_item_class_root(
                    tyref_to_value_type_with(
                        &pointee,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ),
                    &pointee,
                ),
            )
        })
    }

    /// The address word when `local` holds a slice-item option.
    pub(in crate::front::mir) fn item_addr_var(
        &self,
        mir_bb: usize,
        local: usize,
    ) -> Result<Variable, LowerError> {
        match self.local_var.get(local).and_then(|slot| slot.as_ref()) {
            Some(LocalValue::ItemAddr(addr)) => Ok(addr.clone()),
            _ => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: read of slice item reference local {local} before any Assign"
            ))),
        }
    }

    /// A rustc promoted-const / NamedConst accessor is a `FunDecl` whose
    /// `src` is `GlobalInitializer` (`FunDecl::is_global_initializer`).
    /// The body copies an existing `Place::Global` and returns a reference;
    /// `Place::Global` already folds that inner item (`Constant` for a
    /// `NamedConst`, `ConstRefAddr` for a prebuilt JitDriver). A body that
    /// is pure constant construction (a fieldless-enum `Aggregate`, a
    /// literal) is the same translation-time `flowspace` `Constant`. Emitting
    /// the accessor as `residual_call` would keep a `symbolic_fnaddr_for_path`
    /// that `getfunctionptr` never binds (`jtransform.py`
    /// `Transformer.handle_residual_call`).
    pub(in crate::front::mir) fn lower_global_initializer_call(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let gid = fd.is_global_initializer();
        if gid.is_none() && !fundecl_is_promoted_const(fd) {
            return Ok(false);
        }
        if !call.args.is_empty() {
            return Ok(false);
        }
        let Some(body) = fd.unstructured() else {
            return Ok(false);
        };
        if let Some(inner) = promoted_const_copied_global(&body)
            && gid != Some(inner_global_id(&inner))
        {
            let value = self.resolve_place(mir_bb, inner)?;
            self.bind_initializer_result(mir_bb, dest_local, target, value)?;
            return Ok(true);
        }
        // `_1 = Aggregate(E::V); _0 = &_1; return` — rustc's promoted
        // form of a fieldless-enum constant. The by-value model is the
        // discriminant integer (`tyref_to_value_type` Int / `ConstInt`).
        if let Some(op) = const_eval_init_body(self.llbc, &body) {
            let bb_id = self.block_id[mir_bb];
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind: op,
            });
            self.bind_initializer_result(mir_bb, dest_local, target, res)?;
            return Ok(true);
        }
        // `_1 = const C; _0 = &_1; return` — rustc's promoted form of a
        // `&str` / integer / fn-item literal. The by-value model is the
        // same op `build_rvalue` emits for `C` (`__str_const`, `ConstInt`,
        // `fn_const`).
        if let Some(value) = promoted_const_copied_const(&body)
            && let Ok(decoded) = decode_constant(self.llbc, &value)
        {
            let bb_id = self.block_id[mir_bb];
            let res = self.emit_decoded_const(bb_id, decoded);
            self.bind_initializer_result(mir_bb, dest_local, target, res)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(in crate::front::mir) fn bind_initializer_result(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        value: Variable,
    ) -> Result<(), LowerError> {
        self.local_var[dest_local] = Some(LocalValue::One(value));
        let bb_id = self.block_id[mir_bb];
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// An `Option` method over a slice item reference local: `copied` /
    /// `cloned` keep the address, `unwrap_or` loads the item or takes the
    /// default, `is_some` / `is_none` test the address. The item kind comes
    /// from the receiver local's MIR type. Answers whether the call was one
    /// of these.
    pub(in crate::front::mir) fn lower_item_addr_call(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let Some(Operand::Copy(opt) | Operand::Move(opt)) = call.args.first() else {
            return Ok(false);
        };
        let PlaceKind::Local(opt) = opt.kind else {
            return Ok(false);
        };
        let opt = opt as usize;
        if !matches!(
            self.local_var.get(opt).and_then(|slot| slot.as_ref()),
            Some(LocalValue::ItemAddr(_))
        ) {
            return Ok(false);
        }
        let Some((kind, by_ref)) = self.option_pair_item(opt) else {
            return Ok(false);
        };
        let name = regular_call_name_path(reg, self.llbc).unwrap_or_default();
        let addr = self.item_addr_var(mir_bb, opt)?;
        let dest_ty = self.with_item_class_root(
            tyref_to_value_type_with(
                &call.dest.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ),
            &call.dest.ty,
        );
        let (result, keep_addr) = match name.as_str() {
            "core::option::<Impl>::copied" | "core::option::<Impl>::cloned" if by_ref => {
                (addr, true)
            }
            "core::option::<Impl>::unwrap_or" if !by_ref && call.args.len() == 2 => {
                let default = self.resolve_operand(mir_bb, call.args[1].clone())?;
                let helper_ty = Self::pair_helper_item_ty(kind);
                let default = self.retype_word(mir_bb, default, &dest_ty, &helper_ty);
                let item = self.emit_path_call(
                    mir_bb,
                    majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::LoadOr, kind),
                    vec![addr, default],
                    helper_ty.clone(),
                );
                (self.retype_word(mir_bb, item, &helper_ty, &dest_ty), false)
            }
            "core::option::<Impl>::is_some" | "core::option::<Impl>::is_none" => {
                let zero = self.emit_const_uint(mir_bb, 0);
                let op = if name.ends_with("is_some") {
                    "ne"
                } else {
                    "eq"
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let bb_id = self.emit_bb(mir_bb);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: OpKind::BinOp {
                        op: op.to_string(),
                        lhs: addr,
                        rhs: zero,
                        result_ty: dest_ty,
                    },
                });
                (res, false)
            }
            _ => return Ok(false),
        };
        self.local_var[dest_local] = Some(if keep_addr {
            LocalValue::ItemAddr(result)
        } else {
            LocalValue::One(result)
        });
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `(ptr, N, kind)` of `&a as &[T]` over raw-buffer array `a` borrowed
    /// by `src`.
    pub(in crate::front::mir) fn raw_array_unsize_source(
        &mut self,
        mir_bb: usize,
        src: &Place,
    ) -> Option<(Variable, Variable, majit_ir::rvec::VecItemKind)> {
        let PlaceKind::Local(b) = src.kind else {
            return None;
        };
        let Some(RawArrayRole::Borrow(storage)) = self.raw_array[b as usize] else {
            return None;
        };
        let Some(RawArrayRole::Storage { kind, len }) = self.raw_array[storage] else {
            return None;
        };
        let ptr = self.local_var[storage]
            .as_ref()
            .and_then(|value| value.one().ok())?;
        let len = self.emit_const_uint(mir_bb, len);
        Some((ptr, len, kind))
    }

    /// `Unsize` of an object-pointer array literal to a read-only
    /// object slice (`&[T]` / `*const [T]`) is `newlist` of those items.
    ///
    /// RPython `[a, b, c]` is `newlist` (`rtype_newlist`); a list that is
    /// never resized is `FixedSizeListRepr`, whose `LIST` becomes the
    /// `GcArray` itself (`ll_fixed_items`). The slice parameter is that
    /// same one-word `Ptr(GcArray(Ptr(PyObject)))`, so the unsize must
    /// produce the list, not the `Array<T;N>` synthetic struct. Indexed
    /// reads of the array itself keep the positional aggregate: only the
    /// slice value is the list.
    ///
    /// A `&mut [T]` / `*mut [T]` destination is not rewritten: a callee
    /// may write the slice, and those stores must land in the same
    /// aggregate later indexed reads of the array observe.
    pub(in crate::front::mir) fn object_array_unsize_newlist(
        &mut self,
        dest_ty: &TyRef,
        array_var: &Variable,
    ) -> Option<(OpKind, Variable)> {
        match tyref_object_pointer_slice_mutability(dest_ty, self.llbc) {
            Some("Shared" | "Const") => {}
            _ => return None,
        }
        let items = read_array_literal_elements(&self.graph, array_var)?;
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        // `do_fixed_newlist` reads `ARRAY = op.result.concretetype.TO`.
        // The slice is `Ptr(GcArray(Ptr(PyObject)))`.
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, GCREF, LowLevelType, Ptr, PtrTarget,
        };
        res.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(GCREF.clone())),
        }))));
        Some((OpKind::NewList { args: items }, res))
    }

    /// `(ptr, N, kind)` of `&a as &[T]`. A raw-buffer array local is
    /// [`Self::raw_array_unsize_source`]. Otherwise `src` is a reference to
    /// a fixed `[T; N]` of an enabled pair-item kind. The pointer word is
    /// that array's unsigned address: `cast_ptr_to_int` then `r_uint`, the
    /// same word a `&[T]` parameter arrives in.
    pub(in crate::front::mir) fn pair_unsize_source(
        &mut self,
        mir_bb: usize,
        src: &Place,
    ) -> Option<(Variable, Variable, majit_ir::rvec::VecItemKind)> {
        if let Some(pair) = self.raw_array_unsize_source(mir_bb, src) {
            return Some(pair);
        }
        let PlaceKind::Local(local) = src.kind else {
            return None;
        };
        let array_ty = ref_pointee_ty(&src.ty, self.llbc)?;
        let (kind, len) = tyref_raw_array_item_kind(&array_ty, self.llbc)?;
        let ptr = self
            .local_var
            .get(local as usize)
            .and_then(|slot| slot.as_ref())
            .and_then(|value| value.one().ok())?;
        let bb_id = self.emit_bb(mir_bb);
        let (retype, ptr) = push_ptr_to_unsigned_cast(&mut self.graph, bb_id, ptr);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(ptr.clone()),
            kind: retype,
        });
        let len = self.emit_const_uint(mir_bb, len);
        Some((ptr, len, kind))
    }

    /// Whether a call is one [`Self::lower_pair_call`] lowers: it defines a
    /// pair slice, or it is a slice method that reads one.
    pub(in crate::front::mir) fn is_pair_call(
        &self,
        call: &CallPayload,
        dest_local: usize,
    ) -> bool {
        let dest_is_window_slot = self.len_shadow.get(dest_local).copied().flatten().is_some()
            && self
                .body
                .locals
                .locals
                .get(dest_local)
                .is_some_and(|loc| tyref_is_u8_window(&loc.ty, self.llbc));
        // A W1 STR view stays one `Ptr(STR)` word. Index / `len` / `eq`
        // take the HEAD STR-view lowering (`string_byte_view_locals`).
        // A mixed STR/list dest has a length shadow and converts here.
        if !dest_is_window_slot {
            if call_dest_is_w1_str_source(call, self.llbc) {
                return false;
            }
            if operand_local(call.args.first())
                .is_some_and(|local| self.string_byte_view_locals.contains(&local))
            {
                return false;
            }
            if self.local_holds_list_view(dest_local)
                || operand_local(call.args.first())
                    .is_some_and(|local| self.local_holds_list_view(local))
            {
                return false;
            }
        }
        // A length slot means this dest is a pair in this frame only when
        // its MIR type is a pair slice, an option of one, a slice
        // iterator, or a u8 window. `IterMut::next` writes `Option<&mut T>`,
        // which is an item address, not a slice.
        if self.len_shadow[dest_local].is_some()
            && self.body.locals.locals.get(dest_local).is_some_and(|loc| {
                if tyref_is_u8_window(&loc.ty, self.llbc) {
                    return true;
                }
                let pair_dest = self.pair_slice_kind(&loc.ty).is_some()
                    || self.pair_iter_kind(&loc.ty).is_some()
                    || self.option_pair_slice_kind(&loc.ty).is_some();
                if !pair_dest {
                    return false;
                }
                // An object-pointer `&mut [T]` dest is the aliasing view
                // only when the receiver is a `Vec` (`pair_of_vec`). The
                // same type from a length-prefixed object array stays one
                // word. `IterMut` of those items is the iterator of that
                // view and keeps the dest-shadow path.
                if slice_ty_item_is_object_pointer(&loc.ty, self.llbc)
                    && (ref_kind_is_mut(&loc.ty, self.llbc)
                        || raw_ptr_kind_is_mut(&loc.ty, self.llbc))
                {
                    call.args
                        .first()
                        .and_then(operand_tyref)
                        .is_some_and(|ty| tyref_rust_vec_item_kind(ty, self.llbc).is_some())
                } else {
                    true
                }
            })
        {
            return true;
        }
        let CallFunc::Regular(reg) = &call.func else {
            return false;
        };
        // The receiver must be a pair in this frame (a length slot), not
        // merely a pair-shaped type. An object-pointer `&mut [T]` parameter
        // keeps the one-word helper ABI; only a body local of that type is
        // the `{ptr, len}` view. A whole-list `&mut vec` aliases the vec
        // header ([`Self::try_lower_object_vec_whole_list_view`]) and has
        // no length slot; slice methods on that view still take this arm,
        // which splits the header with [`Self::pair_of_vec`].
        // `<[T]>::reverse` of the header stays `ll_vec_reverse` in the
        // regular call arm (`rlist.py` `ll_reverse`).
        operand_local(call.args.first()).is_some_and(|local| {
            self.len_shadow[local].is_some()
                || (self.local_is_object_vec_header(local)
                    && regular_call_name_path(reg, self.llbc).as_deref()
                        != Some("core::slice::<Impl>::reverse"))
        }) && call.args.first().and_then(operand_tyref).is_some_and(|ty| {
            self.pair_slice_kind(ty).is_some() || tyref_is_u8_window(ty, self.llbc)
        }) && regular_call_name_path(reg, self.llbc).is_some_and(|path| {
            let window_recv = call
                .args
                .first()
                .and_then(operand_tyref)
                .is_some_and(|ty| tyref_is_u8_window(ty, self.llbc));
            match path.as_str() {
                "core::slice::<Impl>::len"
                | "core::slice::<Impl>::is_empty"
                | "core::slice::<Impl>::as_ptr"
                | "core::slice::<Impl>::as_mut_ptr" => true,
                "core::slice::index::<Impl>::index"
                | "core::slice::index::<Impl>::index_mut"
                | "core::ops::index::<Impl>::index"
                | "core::ops::index::<Impl>::index_mut" => window_recv,
                "core::slice::<Impl>::reverse"
                | "core::slice::<Impl>::rotate_left"
                | "core::slice::<Impl>::rotate_right"
                | "core::slice::<Impl>::copy_from_slice"
                | "core::slice::<Impl>::contains"
                | "core::slice::<Impl>::first"
                | "core::slice::<Impl>::last"
                | "alloc::slice::<Impl>::to_vec" => !window_recv,
                // Only the scalar `SliceIndex` instantiation yields one item.
                "core::slice::<Impl>::get" => {
                    !window_recv
                        && call
                            .args
                            .get(1)
                            .and_then(|op| {
                                operand_tyref(op)
                                    .map(clone_tyref)
                                    .or_else(|| const_operand_tyref(op, self.llbc))
                            })
                            .is_some_and(|ty| tyref_to_ast_string(&ty, self.llbc) == "usize")
                }
                _ => false,
            }
        })
    }

    /// A call that defines a pair slice, or a slice method over one.
    ///
    /// - `<[T]>::len` is the length word, `is_empty` compares it with 0 and
    ///   `reverse` is `ll_slice_reverse`. `rotate_left` / `rotate_right` are
    ///   `ll_slice_rotate_*` of the pair plus the count. `copy_from_slice` is
    ///   `ll_slice_arraycopy` of the destination length. `as_ptr` and
    ///   `as_mut_ptr` are the pointer word.
    /// - `Vec::deref` / `deref_mut` / `as_slice` / `as_mut_slice` and
    ///   `Vec::index(RangeFull)` of a one-word item vec are
    ///   `(ll_vec_items(v), ll_vec_length(v))`. `Vec<PyObjectRef>`
    ///   whole-list views are one `SomeList` word
    ///   ([`Self::try_lower_object_vec_whole_list_view`]); a slice method
    ///   on a mut view splits that header with [`Self::pair_of_vec`] and
    ///   uses the same `ll_slice_*` helpers.
    /// - `slice::from_raw_parts(p, n)` is `(p, n)`.
    /// - `slice::from_ref(p)` / `from_mut(p)` is `(p, 1)`.
    /// - Any other callee returning a pair slice takes the address of a
    ///   one-word length slot as a trailing argument, stores the length
    ///   there and returns the pointer.
    pub(in crate::front::mir) fn lower_pair_call(
        &mut self,
        mir_bb: usize,
        call: CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<(), LowerError> {
        let CallFunc::Regular(reg) = &call.func else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice through a dynamic call"
            )));
        };
        let name = regular_call_name_path(reg, self.llbc).unwrap_or_default();
        if self.try_lower_pair_unwrap_or(mir_bb, &call, &name, dest_local, target)? {
            return Ok(());
        }
        let arg_tys: Vec<Option<TyRef>> = call
            .args
            .iter()
            .map(|op| operand_tyref(op).map(clone_tyref))
            .collect();
        let dest_ty = clone_tyref(&call.dest.ty);
        let mut args: Vec<Variable> = Vec::with_capacity(call.args.len() + 1);
        let mut pair_lens: Vec<(usize, Variable)> = Vec::new();
        let host_dict = is_rdict_host_rewritten_call(reg, self.llbc);
        let input_flags = regular_call_fun_decl_id(&reg.kind)
            .and_then(|id| self.llbc.fn_by_id(id))
            .map(|fd| self.pair_or_window_fun_inputs(fd, &reg.generics).0)
            .unwrap_or_default();
        for op in &call.args {
            let index = args.len();
            match self.pair_or_window_operand(mir_bb, op)? {
                Some((ptr, len)) => {
                    if host_dict && index != 0 {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: window argument of {name}"
                        )));
                    }
                    pair_lens.push((index, len));
                    args.push(ptr);
                }
                None => {
                    if !host_dict
                        && input_flags.get(index).copied().unwrap_or(false)
                        && self.operand_is_string_byte_view(op)
                        && !self.is_wtf8_from_bytes_identity(reg, &dest_ty)
                        && !tyref_is_u8_window(&dest_ty, self.llbc)
                        && !tyref_is_string_value(&dest_ty, self.llbc)
                    {
                        let var = self.resolve_operand(mir_bb, op.clone())?;
                        let (ptr, len) = self.owner_to_window(mir_bb, var, operand_tyref(op))?;
                        pair_lens.push((index, len));
                        args.push(self.ensure_window_ccharp(mir_bb, ptr));
                    } else if !host_dict
                        && input_flags.get(index).copied().unwrap_or(false)
                        && self.operand_is_list_view(op)
                    {
                        let var = self.resolve_operand(mir_bb, op.clone())?;
                        let (ptr, len) = self.emit_list_view_to_window(mir_bb, var)?;
                        pair_lens.push((index, len));
                        args.push(self.ensure_window_ccharp(mir_bb, ptr));
                    } else {
                        args.push(self.resolve_operand(mir_bb, op.clone())?);
                    }
                }
            }
        }
        // Whole-list `&mut vec` / `&mut vec[..]` of `Vec<PyObjectRef>` is
        // the header word, so it has no length slot. Slice methods on that
        // view (`rotate_left` / `rotate_right`, `copy_from_slice`, ...) and
        // `iter_mut` still walk the item buffer: split the header the way
        // `pair_of_vec` splits a `Vec` (`ll_vec_items` / `ll_vec_length`).
        if !pair_lens.iter().any(|(at, _)| *at == 0)
            && args
                .first()
                .is_some_and(|v| self.var_is_object_vec_header(v))
        {
            let kind = arg_tys
                .first()
                .and_then(Option::as_ref)
                .and_then(|ty| self.pair_slice_kind(ty))
                .unwrap_or(majit_ir::rvec::VecItemKind::Ref);
            let (ptr, len) = self.pair_of_vec(mir_bb, kind, args[0].clone());
            args[0] = ptr;
            pair_lens.insert(0, (0, len));
        }
        let vec_kind = arg_tys
            .first()
            .and_then(Option::as_ref)
            .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc));
        let arg_is_range_full = arg_tys
            .get(1)
            .and_then(Option::as_ref)
            .is_some_and(|ty| tyref_to_ast_string(ty, self.llbc) == "RangeFull");
        let arg_is_pair_iter = arg_tys
            .first()
            .and_then(Option::as_ref)
            .is_some_and(|ty| self.pair_iter_kind(ty).is_some());
        if self.pair_local_is_iter(dest_local)
            && arg_is_pair_iter
            && name == "core::iter::traits::collect::<Impl>::into_iter"
            && let Some((0, item_ptr)) = pair_lens.first().cloned()
        {
            // The reflexive `into_iter` of an iterator is the iterator.
            self.bind_slice_iter(dest_local, args[0].clone(), item_ptr);
        } else if self.pair_local_is_iter(dest_local) {
            // `for x in s` / `s.iter()`: a range iterator over `0..len`
            // with the item pointer beside it.  `for x in &v` over a `Vec`
            // iterates the pair the `Vec` derefs to.
            let vec_pair = match (name.as_str(), vec_kind) {
                ("alloc::vec::<Impl>::into_iter", Some(kind)) if args.len() == 1 => {
                    Some(self.pair_of_vec(mir_bb, kind, args[0].clone()))
                }
                _ => None,
            };
            let from_vec = vec_pair.is_some();
            if let Some((ptr, len)) = vec_pair {
                args[0] = ptr;
                pair_lens = vec![(0, len)];
            }
            let Some((0, len)) = pair_lens.first().cloned() else {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice iterator from {name}"
                )));
            };
            if !from_vec
                && !matches!(
                    name.as_str(),
                    "core::slice::iter::<Impl>::into_iter"
                        | "core::slice::<Impl>::iter"
                        | "core::slice::<Impl>::iter_mut"
                )
            {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: slice iterator from {name}"
                )));
            }
            let bb_id = self.emit_bb(mir_bb);
            let stop = self.retype_word(mir_bb, len.clone(), &ValueType::Unsigned, &ValueType::Int);
            let start = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(start.clone()),
                kind: OpKind::ConstInt(0),
            });
            let range = self.graph.alloc_value_var();
            let iter = self.graph.alloc_value_var();
            let ops = &mut self.graph.block_mut(bb_id).operations;
            ops.push(crate::front::range_iter::range_builtin_call(
                range.clone(),
                start,
                stop,
            ));
            ops.push(crate::front::range_iter::slice_iter_call(
                iter.clone(),
                range,
            ));
            self.bind_slice_iter(dest_local, iter, args[0].clone());
        } else if self.len_shadow[dest_local].is_some() && tyref_is_u8_window(&dest_ty, self.llbc) {
            // A builder byte view whose dest keeps a length shadow is still
            // the built STR (`AbstractStringBuilderRepr.rtype_method_build` /
            // `ll_build`).
            let arg_locals: Vec<Option<usize>> =
                call.args.iter().map(|op| operand_local(Some(op))).collect();
            let built = if call.args.len() == 1 {
                self.builder_identity_string(mir_bb, &arg_locals)?
            } else {
                None
            };
            if let Some(built) = built {
                self.bind_str_source_to_dest(mir_bb, dest_local, built);
            } else if call_dest_is_w1_str_source(&call, self.llbc)
                || call_dest_is_format_str_view(&call, self.llbc, &self.accum)
            {
                let header = if self.is_w_bytes_data(reg) || self.is_w_bytes_block(reg) {
                    let bb_id = self.emit_bb(mir_bb);
                    self.emit_w_bytes_data_str_read(bb_id, args[0].clone())
                } else if self.is_w_str_storage(reg) {
                    let bb_id = self.emit_bb(mir_bb);
                    self.emit_w_str_utf8_read(bb_id, args[0].clone())
                } else {
                    args[0].clone()
                };
                self.bind_str_source_to_dest(mir_bb, dest_local, header);
            } else {
                let (ptr, len) = self.lower_window_dest_call(
                    mir_bb,
                    &name,
                    &args,
                    &arg_tys,
                    &pair_lens,
                    reg,
                    &dest_ty,
                    arg_is_range_full,
                )?;
                let ptr = self.ensure_window_ccharp(mir_bb, ptr);
                let owner = self
                    .owner_for_window_source(&ptr)
                    .or_else(|| args.first().and_then(|v| self.owner_for_window_source(v)));
                self.bind_pair_local(dest_local, ptr, len, owner);
            }
        } else if self.len_shadow[dest_local].is_some() {
            // `<[T; N]>::index(&a, range)` over a raw-buffer array.
            let array_receiver = match (name.as_str(), call.args.first()) {
                (
                    "core::array::<Impl>::index" | "core::array::<Impl>::index_mut",
                    Some(Operand::Copy(place) | Operand::Move(place)),
                ) if args.len() == 2 => self.raw_array_unsize_source(mir_bb, place),
                _ => None,
            };
            let src_owner = args.first().and_then(|v| self.owner_for_window_source(v));
            let (ptr, len) = match (name.as_str(), vec_kind) {
                _ if let Some((ptr, len, kind)) = array_receiver => {
                    self.pair_subslice(mir_bb, kind, ptr, len, &args[1], arg_is_range_full, true)?
                }
                (
                    "alloc::vec::<Impl>::deref"
                    | "alloc::vec::<Impl>::deref_mut"
                    | "alloc::vec::<Impl>::as_slice"
                    | "alloc::vec::<Impl>::as_mut_slice",
                    Some(kind),
                ) if args.len() == 1 => self.pair_of_vec(mir_bb, kind, args[0].clone()),
                ("alloc::vec::<Impl>::index" | "alloc::vec::<Impl>::index_mut", Some(kind))
                    if args.len() == 2 =>
                {
                    let (ptr, len) = self.pair_of_vec(mir_bb, kind, args[0].clone());
                    self.pair_subslice(mir_bb, kind, ptr, len, &args[1], arg_is_range_full, true)?
                }
                ("core::slice::<Impl>::get", _)
                    if args.len() == 2 && matches!(pair_lens.as_slice(), [(0, _)]) =>
                {
                    let recv_kind = self
                        .pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"))
                        .expect("pair receiver");
                    let dest_kind = self.option_pair_slice_kind(&dest_ty).ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: pair slice passed to or returned from {name}"
                        ))
                    })?;
                    if dest_kind != recv_kind {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: pair slice passed to or returned from {name}"
                        )));
                    }
                    let len = pair_lens[0].1.clone();
                    let index_ty = arg_tys
                        .get(1)
                        .and_then(Option::as_ref)
                        .cloned()
                        .or_else(|| {
                            const_operand_tyref(call.args.get(1).expect("range"), self.llbc)
                        });
                    self.pair_get_option(
                        mir_bb,
                        recv_kind,
                        args[0].clone(),
                        len,
                        &args[1],
                        index_ty.as_ref(),
                        arg_is_range_full,
                    )?
                }
                (
                    "core::slice::index::<Impl>::index" | "core::slice::index::<Impl>::index_mut",
                    _,
                ) if args.len() == 2 && matches!(pair_lens.as_slice(), [(0, _)]) => {
                    let kind = self
                        .pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"))
                        .expect("pair receiver");
                    let len = pair_lens[0].1.clone();
                    self.pair_subslice(
                        mir_bb,
                        kind,
                        args[0].clone(),
                        len,
                        &args[1],
                        arg_is_range_full,
                        true,
                    )?
                }
                _ if is_slice_from_ref_name(&name) && args.len() == 1 => {
                    let ptr_ty = arg_tys[0].as_ref().map(|ty| {
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
                    (ptr, len)
                }
                (
                    "core::slice::raw::from_raw_parts" | "core::slice::raw::from_raw_parts_mut",
                    _,
                ) if args.len() == 2 => {
                    let ptr_ty = arg_tys[0].as_ref().map(|ty| {
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
                        // `typed_items_block_items_base(block)` lowers to the
                        // block header (the array descr re-adds `base_size`),
                        // so the item address is
                        // `cast_ptr_to_adr(block) + itemoffsetof(ITEMS, 0)`.
                        Some(ValueType::Ref(_))
                            if operand_local(call.args.first()).is_some_and(|base| {
                                base_traces_to_typed_items_block_accessor(
                                    self.body, base, self.llbc,
                                )
                            }) =>
                        {
                            self.items_block_item_address(
                                mir_bb,
                                args[0].clone(),
                                &[
                                    "majit_rlib",
                                    "lltypesystem",
                                    "rlist",
                                    "TYPED_ITEMS_BLOCK_ITEMS_OFFSET",
                                ],
                            )
                        }
                        // `items_block_items_base(block)` likewise lowers to
                        // the `ItemsBlock` header.
                        Some(ValueType::Ref(_))
                            if operand_local(call.args.first()).is_some_and(|base| {
                                base_traces_to_items_block_accessor(self.body, base, self.llbc)
                            }) =>
                        {
                            self.items_block_item_address(
                                mir_bb,
                                args[0].clone(),
                                &["pyre_object", "object_array", "ITEMS_BLOCK_ITEMS_OFFSET"],
                            )
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
                        other => {
                            return Err(LowerError::Unsupported(format!(
                                "bb{mir_bb}: from_raw_parts over a {other:?} pointer"
                            )));
                        }
                    };
                    // W3 is only a `u8` window (`tyref_is_u8_window`).
                    // `from_raw_parts` of `*mut i64` / `*mut PyObjectRef`
                    // keeps the window-commit pair `(ptr, n)`.
                    let data = if tyref_is_u8_window(&dest_ty, self.llbc) {
                        self.emit_window_from_raw_parts_ptr(
                            mir_bb,
                            ptr,
                            arg_tys.first().and_then(Option::as_ref),
                        )
                    } else {
                        ptr
                    };
                    (data, args[1].clone())
                }
                _ => {
                    let fd = self.pair_callee_decl(mir_bb, reg, &name, &pair_lens, true)?;
                    if self.dont_look_inside.contains(&fd.item_meta.name_path()) {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: residual {name} returns a pair slice"
                        )));
                    }
                    let slot = self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::SLICE_LEN_SLOT_NEW,
                        vec![],
                        ValueType::Unsigned,
                    );
                    let target_op = self.pair_call_target(mir_bb, reg, &args, &pair_lens)?;
                    let mut call_args = splice_pair_lens(args, pair_lens);
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
                    (ptr, len)
                }
            };
            let owner = self.owner_for_window_source(&ptr).or(src_owner);
            self.bind_pair_local(dest_local, ptr, len, owner);
        } else {
            let Some((0, len)) = pair_lens.first().cloned() else {
                return Err(LowerError::Schema(format!(
                    "bb{mir_bb}: slice method {name} without a pair receiver"
                )));
            };
            let result = match name.as_str() {
                "core::slice::<Impl>::len" => len,
                "core::slice::<Impl>::is_empty" => {
                    let bb_id = self.emit_bb(mir_bb);
                    let zero = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(zero.clone()),
                        kind: OpKind::ConstUInt(0),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: len,
                            rhs: zero,
                            result_ty: tyref_to_value_type_with(
                                &dest_ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            ),
                        },
                    });
                    res
                }
                "core::slice::<Impl>::reverse" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::Reverse, kind),
                        vec![args[0].clone(), len],
                        ValueType::Void,
                    )
                }
                "core::slice::<Impl>::rotate_left" | "core::slice::<Impl>::rotate_right" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    let Some(count) = args.get(1).cloned() else {
                        return Err(LowerError::Schema(format!(
                            "bb{mir_bb}: {name} without a count"
                        )));
                    };
                    let op = if name == "core::slice::<Impl>::rotate_left" {
                        majit_ir::rvec::SliceOp::RotateLeft
                    } else {
                        majit_ir::rvec::SliceOp::RotateRight
                    };
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(op, kind),
                        vec![args[0].clone(), len, count],
                        ValueType::Void,
                    )
                }
                "core::slice::<Impl>::copy_from_slice" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    let Some(src_ptr) = args.get(1).cloned() else {
                        return Err(LowerError::Schema(format!(
                            "bb{mir_bb}: copy_from_slice without a source slice"
                        )));
                    };
                    let Some(src_len) = pair_lens
                        .iter()
                        .find(|(at, _)| *at == 1)
                        .map(|(_, src_len)| src_len.clone())
                    else {
                        return Err(LowerError::Schema(format!(
                            "bb{mir_bb}: copy_from_slice source is not a pair slice"
                        )));
                    };
                    let same_len = self.emit_uint_binop(mir_bb, "eq", src_len, len.clone());
                    let carried =
                        self.fork_in_bounds(mir_bb, same_len, &[src_ptr, args[0].clone(), len])?;
                    let src_ptr = carried[0].clone();
                    let dest_ptr = carried[1].clone();
                    let dst_len = carried[2].clone();
                    let zero = self.emit_const_uint(mir_bb, 0);
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::ArrayCopy, kind),
                        vec![src_ptr, dest_ptr, zero.clone(), zero, dst_len],
                        ValueType::Void,
                    )
                }
                "core::slice::<Impl>::as_ptr" | "core::slice::<Impl>::as_mut_ptr" => {
                    let data = args[0].clone();
                    self.stamp_pair_window_as_ptr_ccharp(&dest_ty, &data);
                    data
                }
                "core::slice::index::<Impl>::index"
                | "core::slice::index::<Impl>::index_mut"
                | "core::ops::index::<Impl>::index"
                | "core::ops::index::<Impl>::index_mut"
                    if args.len() == 2 =>
                {
                    // Scalar `s[i]` of a window is `&u8` at `data + i`.
                    // The following Deref is the 1-byte `raw_load`.
                    self.window_ptr_add(mir_bb, args[0].clone(), args[1].clone())
                }
                "core::slice::<Impl>::contains" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    let item_ty = arg_tys[1]
                        .as_ref()
                        .and_then(|ty| {
                            tyref_ref_pointee_value_type(
                                ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            )
                        })
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: slice contains of a non-reference item"
                            ))
                        })?;
                    let helper_ty = Self::pair_helper_item_ty(kind);
                    let item = self.retype_word(mir_bb, args[1].clone(), &item_ty, &helper_ty);
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::Contains, kind),
                        vec![args[0].clone(), len, item],
                        tyref_to_value_type_with(
                            &dest_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                    )
                }
                "alloc::slice::<Impl>::to_vec" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::ToVec, kind),
                        vec![args[0].clone(), len],
                        tyref_to_value_type_with(
                            &dest_ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                    )
                }
                "core::slice::<Impl>::first"
                | "core::slice::<Impl>::last"
                | "core::slice::<Impl>::get" => {
                    let kind = self.pair_slice_kind(arg_tys[0].as_ref().expect("pair receiver"));
                    let kind = kind.expect("pair receiver");
                    let index = match name.as_str() {
                        "core::slice::<Impl>::first" => self.emit_const_uint(mir_bb, 0),
                        "core::slice::<Impl>::last" => {
                            // `len - 1` wraps to the largest index for an
                            // empty slice, which is out of bounds.
                            let one = self.emit_const_uint(mir_bb, 1);
                            let index = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            let bb_id = self.emit_bb(mir_bb);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(index.clone()),
                                kind: OpKind::BinOp {
                                    op: "sub".to_string(),
                                    lhs: len.clone(),
                                    rhs: one,
                                    result_ty: ValueType::Unsigned,
                                },
                            });
                            index
                        }
                        _ => args[1].clone(),
                    };
                    let addr = self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::GetAddr, kind),
                        vec![args[0].clone(), len, index],
                        ValueType::Unsigned,
                    );
                    self.local_var[dest_local] = Some(LocalValue::ItemAddr(addr.clone()));
                    addr
                }
                _ => {
                    return Err(LowerError::Schema(format!(
                        "bb{mir_bb}: {name} is not a pair slice method"
                    )));
                }
            };
            if !matches!(
                self.local_var
                    .get(dest_local)
                    .and_then(|slot| slot.as_ref()),
                Some(LocalValue::ItemAddr(_))
            ) {
                self.local_var[dest_local] = Some(LocalValue::One(result));
            }
        }
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// `&s[range]` over pair `(ptr, len)`: `(ptr + start, end - start)`.
    ///
    /// `checked` panics when the range is outside the slice. `RangeFull` is
    /// the whole slice. `RangeTo` panics when `end > len` and does not move
    /// the pointer. `RangeFrom` panics when `start > len`. `Range` panics
    /// when `start > end` or `end > len`. `start == end == len` is in bounds.
    /// `slice.get` passes `checked = false` and selects a null pointer.
    pub(in crate::front::mir) fn pair_subslice(
        &mut self,
        mir_bb: usize,
        kind: majit_ir::rvec::VecItemKind,
        ptr: Variable,
        len: Variable,
        range: &Variable,
        range_is_full: bool,
        checked: bool,
    ) -> Result<(Variable, Variable), LowerError> {
        if range_is_full {
            return Ok((ptr, len));
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
                "bb{mir_bb}: pair subslice by a range this body did not build"
            )));
        };
        let Some(start) = start else {
            if !checked {
                return Ok((ptr, end));
            }
            let in_bounds = self.emit_uint_binop(mir_bb, "le", end.clone(), len);
            let carried = self.fork_in_bounds(mir_bb, in_bounds, &[ptr, end])?;
            return Ok((carried[0].clone(), carried[1].clone()));
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
        let ptr = self.emit_path_call(
            mir_bb,
            majit_ir::rvec::slice_helper_path(majit_ir::rvec::SliceOp::Offset, kind),
            vec![ptr, start.clone()],
            ValueType::Unsigned,
        );
        let len = self.emit_uint_binop(mir_bb, "sub", end, start);
        Ok((ptr, len))
    }

    /// Item kind of `Option<&[T]>` / `Option<&mut [T]>` when `&[T]` is a pair.
    pub(in crate::front::mir) fn option_pair_slice_kind(
        &self,
        ty: &TyRef,
    ) -> Option<majit_ir::rvec::VecItemKind> {
        tyref_option_pair_slice_item_kind(ty, self.llbc)
    }

    /// `(ptr, len)` of an `Option<&[T]>` operand carried as a pair, `None`
    /// when `op` is not that option.
    pub(in crate::front::mir) fn option_pair_operand(
        &mut self,
        mir_bb: usize,
        op: &Operand,
    ) -> Result<Option<(Variable, Variable)>, LowerError> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Const(_) => return Ok(None),
        };
        if self.option_pair_slice_kind(&place.ty).is_none() {
            return Ok(None);
        }
        match self.pair_place_parts(mir_bb, place)? {
            Some(parts) => Ok(Some(parts)),
            None => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice read through a projection"
            ))),
        }
    }

    /// `slice.get(range)` into `Option<&[T]>`: `None` is a null item pointer.
    /// The in-bounds pair is [`Self::pair_subslice`]; both words are then
    /// kept only when `start <= end && end <= len`.
    pub(in crate::front::mir) fn pair_get_option(
        &mut self,
        mir_bb: usize,
        kind: majit_ir::rvec::VecItemKind,
        ptr: Variable,
        len: Variable,
        range: &Variable,
        index_ty: Option<&TyRef>,
        range_is_full: bool,
    ) -> Result<(Variable, Variable), LowerError> {
        let spelling = index_ty
            .map(|ty| tyref_to_ast_string(ty, self.llbc))
            .unwrap_or_default();
        let zero = self.emit_const_uint(mir_bb, 0);
        let (start, end) = if range_is_full || spelling == "RangeFull" {
            (zero.clone(), len.clone())
        } else if spelling == "RangeFrom<usize>" {
            let start = self
                .slice_index_rangefrom_sites
                .iter()
                .find(|site| site.range_result == *range)
                .map(|site| site.start.clone())
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: pair subslice by a range this body did not build"
                    ))
                })?;
            (start, len.clone())
        } else if spelling == "RangeTo<usize>" {
            let end = self
                .slice_index_rangeto_sites
                .iter()
                .find(|site| site.range_result == *range)
                .map(|site| site.end.clone())
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: pair subslice by a range this body did not build"
                    ))
                })?;
            (zero.clone(), end)
        } else if spelling == "Range<usize>" {
            let site = self
                .slice_index_range_sites
                .iter()
                .find(|site| site.range_result == *range)
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: pair subslice by a range this body did not build"
                    ))
                })?;
            (site.start.clone(), site.end.clone())
        } else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice passed to or returned from core::slice::<Impl>::get"
            )));
        };
        let start_le_end = self.emit_uint_binop(mir_bb, "le", start, end.clone());
        let end_le_len = self.emit_uint_binop(mir_bb, "le", end, len.clone());
        let in_bounds = self.emit_uint_binop(mir_bb, "bitand", start_le_end, end_le_len);
        let full = range_is_full || spelling == "RangeFull";
        let (sub_ptr, sub_len) = self.pair_subslice(mir_bb, kind, ptr, len, range, full, false)?;
        let ptr = self.emit_uint_select(mir_bb, in_bounds.clone(), sub_ptr, zero.clone());
        let len = self.emit_uint_select(mir_bb, in_bounds, sub_len, zero);
        Ok((ptr, len))
    }

    /// `unwrap_or` on an `Option<&[T]>` pair: both words of the option when
    /// its item pointer is non-null, otherwise both words of the default.
    pub(in crate::front::mir) fn try_lower_pair_unwrap_or(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        name: &str,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if name != "core::option::<Impl>::unwrap_or"
            || call.args.len() != 2
            || self.len_shadow[dest_local].is_none()
        {
            return Ok(false);
        }
        let Some(dest_kind) = self.pair_slice_kind(&call.dest.ty) else {
            return Ok(false);
        };
        let opt_ty = operand_tyref(&call.args[0]).map(clone_tyref);
        let Some(opt_kind) = opt_ty
            .as_ref()
            .and_then(|ty| self.option_pair_slice_kind(ty))
        else {
            return Ok(false);
        };
        if opt_kind != dest_kind {
            return Ok(false);
        }
        let (opt_ptr, opt_len) = self
            .option_pair_operand(mir_bb, &call.args[0])?
            .ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: pair slice passed to or returned from {name}"
                ))
            })?;
        let (def_ptr, def_len, def_kind) =
            self.pair_operand(mir_bb, &call.args[1])?.ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: pair slice passed to or returned from {name}"
                ))
            })?;
        if def_kind != dest_kind {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice passed to or returned from {name}"
            )));
        }
        let zero = self.emit_const_uint(mir_bb, 0);
        let is_some = self.emit_uint_binop(mir_bb, "ne", opt_ptr.clone(), zero);
        let owner = self
            .owner_for_window_source(&opt_ptr)
            .or_else(|| self.owner_for_window_source(&def_ptr));
        let ptr = self.emit_uint_select(mir_bb, is_some.clone(), opt_ptr, def_ptr);
        let len = self.emit_uint_select(mir_bb, is_some, opt_len, def_len);
        self.bind_pair_local(dest_local, ptr, len, owner);
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `(ll_vec_items(v), ll_vec_length(v))`: the pair a `Vec` derefs to.
    pub(in crate::front::mir) fn pair_of_vec(
        &mut self,
        mir_bb: usize,
        kind: majit_ir::rvec::VecItemKind,
        vec: Variable,
    ) -> (Variable, Variable) {
        let ptr = self.emit_path_call(
            mir_bb,
            majit_ir::rvec::vec_helper_path(majit_ir::rvec::VecOp::Items, kind),
            vec![vec.clone()],
            ValueType::Unsigned,
        );
        let len = self.emit_path_call(
            mir_bb,
            majit_ir::rvec::vec_helper_path(majit_ir::rvec::VecOp::Length, kind),
            vec![vec],
            ValueType::Int,
        );
        let len = self.retype_word(mir_bb, len, &ValueType::Int, &ValueType::Unsigned);
        (ptr, len)
    }

    /// FunDecl slots that take the two-word pair/window ABI.
    ///
    /// A generic translation subject (`hash_of<Q>` with `Q = [u8]`)
    /// declares `&Q`; the specialized body sees `&[u8]` and gets a
    /// length slot (`pair_len_shadows_of`). The call site substitutes
    /// the same `generics.types` so the argument list matches that
    /// body. An opaque std helper (`core::slice::<Impl>::get`) has no
    /// body to specialize: its unsubstituted `&[T]` stays one word and
    /// the length stays in the caller frame.
    pub(in crate::front::mir) fn pair_or_window_fun_inputs(
        &self,
        fd: &FunDecl,
        generics: &serde_json::Value,
    ) -> (Vec<bool>, bool) {
        let substituted = crate::front::clause_spec::decl_has_unstructured_body(fd)
            .then(|| crate::front::clause_spec::concrete_type_args(generics, self.llbc))
            .flatten()
            .map(|(types, consts)| {
                crate::front::clause_spec::substituted_signature(
                    &fd.signature,
                    self.llbc,
                    &types,
                    &consts,
                )
            });
        let inputs = substituted
            .as_ref()
            .map(|sig| sig.inputs.as_slice())
            .unwrap_or(fd.signature.inputs.as_slice());
        let output = substituted
            .as_ref()
            .map(|sig| &sig.output)
            .unwrap_or(&fd.signature.output);
        let is_pair_or_window =
            |ty: &TyRef| self.pair_slice_kind(ty).is_some() || tyref_is_u8_window(ty, self.llbc);
        (
            inputs.iter().map(is_pair_or_window).collect(),
            is_pair_or_window(output),
        )
    }

    /// The declaration of a callee that takes or returns pair slices: its
    /// declared parameters must be pair slices exactly where the call passes
    /// one, and its result one when `returns_pair`.  A generic callee, whose
    /// `&[T]` is one word for every `T`, fails this.
    pub(in crate::front::mir) fn pair_callee_decl(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
        name: &str,
        pair_lens: &[(usize, Variable)],
        returns_pair: bool,
    ) -> Result<&'a FunDecl, LowerError> {
        let mismatch = || {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: pair slice passed to or returned from {name}"
            ))
        };
        let fd = regular_call_fun_decl_id(&reg.kind)
            .and_then(|id| self.llbc.fn_by_id(id))
            .ok_or_else(mismatch)?;
        let (input_flags, output_is_pair) = self.pair_or_window_fun_inputs(fd, &reg.generics);
        let params_match = input_flags
            .iter()
            .enumerate()
            .all(|(index, wants)| *wants == pair_lens.iter().any(|(at, _)| *at == index));
        let result_matches = output_is_pair == returns_pair;
        if !params_match || !result_matches || abstract_trait_call_target(reg, self.llbc).is_some()
        {
            return Err(mismatch());
        }
        Ok(fd)
    }

    /// The call target of a monomorphic callee taking pair slices.  A
    /// method target keeps its receiver in `args[0]`, so a pair receiver
    /// names the function by path instead.  A pair or window argument
    /// that is not the receiver is the same: the host method rewrite
    /// (`getattr` + `simple_call`) keeps the unspliced arity
    /// (`RDict::get` is receiver + key; `OrderedDictRepr.rtype_method_get`
    /// → `ll_dict_get`), while the specialized body takes the two-word ABI.
    pub(in crate::front::mir) fn pair_call_target(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
        args: &[Variable],
        pair_lens: &[(usize, Variable)],
    ) -> Result<CallTarget, LowerError> {
        let (segments, method_hint) = self.call_target_segments(mir_bb, reg, None)?;
        let fun_decl_id = regular_call_fun_decl_id(&reg.kind);
        let receiver_is_pair = fun_decl_id
            .and_then(|id| self.llbc.fn_by_id(id))
            .and_then(|fd| fd.signature.inputs.first())
            .is_some_and(|ty| {
                self.pair_slice_kind(ty).is_some() || tyref_is_u8_window(ty, self.llbc)
            })
            || pair_lens.iter().any(|(at, _)| *at == 0);
        let non_receiver_pair = pair_lens.iter().any(|(at, _)| *at != 0);
        Ok(match method_hint {
            Some((owner_root, leaf, _))
                if !args.is_empty() && !receiver_is_pair && !non_receiver_pair =>
            {
                let mut target = CallTarget::method(leaf, Some(owner_root));
                if let Some(id) = fun_decl_id {
                    target = target.with_fun_decl_id(id);
                    if let Some(fd) = self.llbc.fn_by_id(id) {
                        target =
                            target.with_resolved_path(registered_path_for_fun_decl(self.llbc, fd));
                    }
                }
                target
            }
            _ => {
                let mut target = CallTarget::FunctionPath {
                    segments,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                };
                if let Some(id) = fun_decl_id {
                    target = target.with_fun_decl_id(id);
                }
                target
            }
        })
    }
}

/// The field-read variable of a declared virtualizable array that `var`
/// was loaded from, if any. That variable is the one `vable_array_vars`
/// keys, so the array op has to name it rather than a later deref.
pub(in crate::front::mir) fn declared_vable_array_var(
    graph: &FunctionGraph,
    var: &Variable,
) -> Option<Variable> {
    let mut current = var.clone();
    for _ in 0..6 {
        let mut base = None;
        let mut declared = false;
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() != Some(&current) {
                    continue;
                }
                match &op.kind {
                    OpKind::FieldRead {
                        base: field_base,
                        field,
                        ..
                    } => {
                        if crate::virtualizable_decl::is_declared_array_field(
                            field.owner_root.as_deref(),
                            &field.name,
                        ) {
                            declared = true;
                        }
                        base = Some(field_base.clone());
                    }
                    OpKind::UnaryOp { operand, .. } => {
                        base = Some(operand.clone());
                    }
                    _ => {}
                }
            }
        }
        if declared {
            return Some(current);
        }
        {
            let next = base?;
            current = next
        }
    }
    None
}

pub(in crate::front::mir) fn reaches_declared_vable_array(
    graph: &FunctionGraph,
    var: &Variable,
) -> bool {
    declared_vable_array_var(graph, var).is_some()
}

/// Fat-pointer DATA word of `recv`, or `None` when `recv` is not a fat field.
pub(in crate::front::mir) fn emit_dyn_receiver_data_word(
    graph: &mut FunctionGraph,
    bb_id: crate::model::BlockId,
    recv: &Variable,
) -> Option<Variable> {
    let (base, mut field, pure) = fat_field_producer_of(graph, recv)?;
    field.vec_part = Some(crate::model::VecFieldPart::FatData);
    field.taken_by_address = false;
    field.inline_vec = false;
    // Same typing as every other FatData read (`fat_part_value_type`):
    // a GC host's data word is Ref (the strategy instance);
    // a raw host's data word is Int (`llmemory.Address`).
    let ty = fat_part_value_type(crate::model::VecFieldPart::FatData, field.owner_declared_gc);
    let concretetype = fat_part_concretetype(&ty);
    // `try_ensure` mints a `Variable.copy()` and `Block.renamevariables`
    // rewrites existing uses (`ssa.py` `SSA_to_SSI`). The new FatData
    // read has to name that local; binding the producer would leave a
    // base with no definition in this block.
    let base = graph
        .try_ensure_variable_at_block_local(bb_id, &base)
        .unwrap_or(base);
    let res = graph.alloc_value_var_with_type(concretetype);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        },
    });
    Some(res)
}

/// `(base, field, pure)` of a fat-pointer FieldRead that produced
/// `recv`, walking `same_as` / `cast_int_to_ptr`. The vec_part tag
/// (`FatVtable` / `FatData` / `FatMeta` / `FatLen`) identifies the
/// field; a name test is not used.
pub(in crate::front::mir) fn fat_field_producer_of(
    graph: &FunctionGraph,
    recv: &Variable,
) -> Option<(Variable, crate::model::FieldDescriptor, bool)> {
    let mut current = recv.clone();
    for _ in 0..8 {
        let mut next = None;
        for block in &graph.blocks {
            for op in &block.operations {
                if op.result.as_ref() != Some(&current) {
                    continue;
                }
                match &op.kind {
                    OpKind::FieldRead {
                        base, field, pure, ..
                    } if matches!(
                        field.vec_part,
                        Some(
                            crate::model::VecFieldPart::FatVtable
                                | crate::model::VecFieldPart::FatData
                                | crate::model::VecFieldPart::FatMeta
                                | crate::model::VecFieldPart::FatLen
                        )
                    ) =>
                    {
                        return Some((base.clone(), field.clone(), *pure));
                    }
                    OpKind::UnaryOp { operand, .. } => {
                        next = Some(operand.clone());
                    }
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        args,
                        ..
                    } if segments.last().map(|s| s.as_str()) == Some("cast_int_to_ptr")
                        || segments.last().map(|s| s.as_str()) == Some("same_as") =>
                    {
                        next = args
                            .first()
                            .and_then(crate::model::LinkArg::as_variable)
                            .cloned();
                    }
                    _ => {}
                }
            }
        }
        current = next?;
    }
    None
}

/// `(base, field, pure)` of an inline `Vec` field read that produced `vec_var`.
///
/// `admit_plain_int_len` is the look-inside-length arm: a one-word Int
/// field that is not stamped `inline_vec` is still the header `len`
/// (`rlist.py` `ll_length`).
pub(in crate::front::mir) fn inline_vec_field_producer(
    graph: &FunctionGraph,
    vec_var: &Variable,
    admit_plain_int_len: bool,
) -> Option<(Variable, FieldDescriptor, bool)> {
    graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| {
            if op.result.as_ref() != Some(vec_var) {
                return None;
            }
            match &op.kind {
                OpKind::FieldRead {
                    base,
                    field,
                    pure,
                    ty,
                    ..
                } if field.vec_part.is_none()
                    && (field.inline_vec
                        || (admit_plain_int_len
                            && matches!(ty, ValueType::Int | ValueType::Unsigned))) =>
                {
                    Some((base.clone(), field.clone(), *pure))
                }
                _ => None,
            }
        })
    })
}

/// `l.length = n` (`rlist.py` `ll_set_length`). An inline `Vec` field keeps
/// the containing STRUCT as owner (`vec_part` Len). A standalone receiver
/// is not a LIST header: leave `Vec::set_len` for the generic call path.
pub(in crate::front::mir) fn vec_set_len_write(
    graph: &FunctionGraph,
    vec_var: &Variable,
    new_len: Variable,
) -> Option<OpKind> {
    let (base, mut field, _) = inline_vec_field_producer(graph, vec_var, true)?;
    field.vec_part = Some(crate::model::VecFieldPart::Len);
    field.taken_by_address = false;
    Some(OpKind::FieldWrite {
        base,
        field,
        value: new_len.into(),
        ty: ValueType::Int,
    })
}

/// The `(base, field)` read of one header word of an inline `Vec` field,
/// when `vec_var` is that field's read; `None` otherwise.  `buf_ty` is the
/// buffer word's type: the `Vec` of one-word items is a raw buffer (`Int`).
///
/// One inline field read can feed both `.len()` and an index. Mutating
/// that shared read into the first component makes the second use load a
/// Vec word from the component. Keep the field read and emit a distinct
/// component read for this use.
pub(in crate::front::mir) fn inline_vec_field_part(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    vec_var: &Variable,
    part: crate::model::VecFieldPart,
    buf_ty: ValueType,
) -> Option<Variable> {
    let (base, mut field, pure) =
        inline_vec_field_producer(graph, vec_var, part == crate::model::VecFieldPart::Len)?;
    field.vec_part = Some(part);
    field.taken_by_address = false;
    // Do not stamp `inline_vec` on a one-word-item Len. getattr of that
    // field as a list (`flowspace_adapter` getattr+len) unions Integer
    // with RustVec at `ll_vec_alloc_and_set_*`. `rlist.py` `ll_length`
    // is `return l.length`, a Signed word of the parent; the assembler
    // already suffixes `.len`. Keep the original flag so wide-item
    // fields still take getattr+len and one-word fields take the
    // Signed `{name}.len` getattr.
    // `try_ensure` mints a `Variable.copy()` and `Block.renamevariables`
    // rewrites existing uses (`ssa.py` `SSA_to_SSI`). The new component
    // read has to name that local; binding the producer would leave a
    // base with no definition in this block.
    let base = graph
        .try_ensure_variable_at_block_local(bb_id, &base)
        .unwrap_or(base);
    let ty = match part {
        crate::model::VecFieldPart::Buf => buf_ty,
        crate::model::VecFieldPart::FatData => fat_part_value_type(part, field.owner_declared_gc),
        crate::model::VecFieldPart::FatVtable => ValueType::Ref(None),
        crate::model::VecFieldPart::Len
        | crate::model::VecFieldPart::FatLen
        | crate::model::VecFieldPart::FatMeta => ValueType::Int,
    };
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        },
    });
    Some(res)
}

pub(in crate::front::mir) fn retarget_vec_operand(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    vec_var: &Variable,
    part: crate::model::VecFieldPart,
) -> Variable {
    if reaches_declared_vable_array(graph, vec_var) {
        return vec_var.clone();
    }
    let buf_ty = ValueType::Ref(None);
    if let Some(res) = inline_vec_field_part(graph, bb_id, vec_var, part, buf_ty) {
        return res;
    }
    let name = match part {
        crate::model::VecFieldPart::Buf => "buf",
        crate::model::VecFieldPart::Len => "len",
        // A fat box is not a `Vec`. Loading `len` off the already-loaded
        // word is the `arraylen_gc` fault this path exists to avoid.
        crate::model::VecFieldPart::FatData
        | crate::model::VecFieldPart::FatLen
        | crate::model::VecFieldPart::FatMeta
        | crate::model::VecFieldPart::FatVtable => {
            return vec_var.clone();
        }
    };
    let ty = match part {
        crate::model::VecFieldPart::Buf
        | crate::model::VecFieldPart::FatData
        | crate::model::VecFieldPart::FatVtable => ValueType::Ref(None),
        crate::model::VecFieldPart::Len
        | crate::model::VecFieldPart::FatLen
        | crate::model::VecFieldPart::FatMeta => ValueType::Int,
    };
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::FieldRead {
            base: vec_var.clone(),
            field: FieldDescriptor::new(name, Some("alloc::vec::Vec".to_string()))
                .with_vec_part(part),
            ty,
            pure: false,
        },
    });
    res
}

/// Field read of an inline fat pointer (`Box<[T]>`), if `var` is that
/// read or a single phi of it. The component read is a new op: one field
/// feeds both `.len()` and an index, so the shared read stays whole.
pub(in crate::front::mir) fn retarget_fat_operand(
    graph: &mut FunctionGraph,
    fat_box_vars: &[Variable],
    bb_id: BlockId,
    var: &Variable,
    part: crate::model::VecFieldPart,
) -> Option<Variable> {
    if fat_box_vars.is_empty()
        || !matches!(
            part,
            crate::model::VecFieldPart::FatData
                | crate::model::VecFieldPart::FatLen
                | crate::model::VecFieldPart::FatMeta
                | crate::model::VecFieldPart::FatVtable
        )
    {
        return None;
    }
    let (base, mut field, pure) = fat_field_producer(graph, fat_box_vars, var)?;
    field.vec_part = Some(part);
    field.taken_by_address = false;
    field.inline_vec = false;
    let ty = match part {
        crate::model::VecFieldPart::FatLen | crate::model::VecFieldPart::FatMeta => ValueType::Int,
        crate::model::VecFieldPart::FatData => fat_part_value_type(part, field.owner_declared_gc),
        crate::model::VecFieldPart::FatVtable => ValueType::Ref(None),
        _ => return None,
    };
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        },
    });
    Some(res)
}

/// Window pair `{data, len}` split of an inline fat box. Data is `CCHARP`
/// (`Address` / int-kind), even when the owner is a GC `Box<[u8]>`: the
/// pair ABI is two integers, not a GC fat-ref plus a length.
pub(in crate::front::mir) fn retarget_window_fat_operand(
    graph: &mut FunctionGraph,
    fat_box_vars: &[Variable],
    bb_id: BlockId,
    var: &Variable,
    part: crate::model::VecFieldPart,
) -> Option<Variable> {
    if fat_box_vars.is_empty()
        || !matches!(
            part,
            crate::model::VecFieldPart::FatData | crate::model::VecFieldPart::FatLen
        )
    {
        return None;
    }
    let (base, mut field, pure) = fat_field_producer(graph, fat_box_vars, var)?;
    field.vec_part = Some(part);
    field.taken_by_address = false;
    field.inline_vec = false;
    field.owner_declared_gc = Some(false);
    let ty = match part {
        crate::model::VecFieldPart::FatLen => ValueType::Int,
        crate::model::VecFieldPart::FatData => ValueType::Unsigned,
        _ => return None,
    };
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Signed);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        },
    });
    Some(res)
}

/// A raw owner's fat data word is an address (`Int`). A GC owner's
/// data word stays a reference. Length is always an int.
pub(crate) fn fat_part_value_type(
    part: crate::model::VecFieldPart,
    owner_declared_gc: Option<bool>,
) -> ValueType {
    match part {
        crate::model::VecFieldPart::FatData if owner_declared_gc == Some(false) => ValueType::Int,
        crate::model::VecFieldPart::Buf
        | crate::model::VecFieldPart::FatData
        | crate::model::VecFieldPart::FatVtable => ValueType::Ref(None),
        crate::model::VecFieldPart::Len
        | crate::model::VecFieldPart::FatLen
        | crate::model::VecFieldPart::FatMeta => ValueType::Int,
    }
}

pub(crate) fn fat_part_concretetype(ty: &ValueType) -> crate::model::ConcreteType {
    match ty {
        ValueType::Int | ValueType::Unsigned => crate::model::ConcreteType::Signed,
        _ => crate::model::ConcreteType::GcRef,
    }
}

pub(in crate::front::mir) fn peel_copy_var(graph: &FunctionGraph, var: &Variable) -> Variable {
    let mut current = var.clone();
    for _ in 0..6 {
        let Some(op) = graph.blocks.iter().find_map(|block| {
            block
                .operations
                .iter()
                .find(|op| op.result.as_ref() == Some(&current))
        }) else {
            break;
        };
        match &op.kind {
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => current = operand.clone(),
            _ => break,
        }
    }
    current
}

pub(in crate::front::mir) fn constants_field_read(
    graph: &FunctionGraph,
    var: &Variable,
) -> Option<(Variable, FieldDescriptor, bool)> {
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() != Some(var) {
                continue;
            }
            if let OpKind::FieldRead {
                base, field, pure, ..
            } = &op.kind
                && field.name == "constants"
                && field.vec_part.is_none()
            {
                return Some((base.clone(), field.clone(), *pure));
            }
        }
    }
    None
}

pub(in crate::front::mir) fn direct_fat_field_read(
    graph: &FunctionGraph,
    fat_box_vars: &[Variable],
    var: &Variable,
) -> Option<(Variable, FieldDescriptor, bool)> {
    if !fat_box_vars.iter().any(|marked| marked == var) {
        return None;
    }
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() != Some(var) {
                continue;
            }
            if let OpKind::FieldRead {
                base, field, pure, ..
            } = &op.kind
                && field.vec_part.is_none()
                && !field.inline_vec
            {
                return Some((base.clone(), field.clone(), *pure));
            }
        }
    }
    None
}

/// `var` is one hop away from a marked fat-box field: a call argument
/// that produced the struct the field was read from, or a further field
/// read of the data word. The owner itself is not that hop —
/// `code.varnames` does not make `code` a `Box<[T]>`. The length stays
/// on the box.
pub(in crate::front::mir) fn fat_len_through_data_word(
    graph: &FunctionGraph,
    fat_box_vars: &[Variable],
    var: &Variable,
) -> Option<(Variable, FieldDescriptor, bool)> {
    for marked in fat_box_vars {
        let Some((base, field, pure)) =
            direct_fat_field_read(graph, std::slice::from_ref(marked), marked)
        else {
            continue;
        };
        if call_first_arg_is(graph, &base, var) || field_read_base_is(graph, var, marked) {
            return Some((base, field, pure));
        }
    }
    None
}

pub(in crate::front::mir) fn field_read_base_is(
    graph: &FunctionGraph,
    var: &Variable,
    base: &Variable,
) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op.result.as_ref() == Some(var)
                && matches!(&op.kind, OpKind::FieldRead { base: got, .. } if got == base)
        })
    })
}

pub(in crate::front::mir) fn call_first_arg_is(
    graph: &FunctionGraph,
    result: &Variable,
    arg: &Variable,
) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            op.result.as_ref() == Some(result)
                && matches!(
                    &op.kind,
                    OpKind::Call { args, .. } if args.first().and_then(crate::model::LinkArg::as_variable) == Some(arg)
                )
        })
    })
}

pub(in crate::front::mir) fn fat_field_producer(
    graph: &FunctionGraph,
    fat_box_vars: &[Variable],
    var: &Variable,
) -> Option<(Variable, FieldDescriptor, bool)> {
    let var = peel_copy_var(graph, var);
    if let Some(found) = direct_fat_field_read(graph, fat_box_vars, &var) {
        return Some(found);
    }
    // `CodeObject.constants` is `Constants<C>(Box<[C]>)`. Its length is the
    // fat metadata word even when the deref was not pushed into `fat_box_vars`.
    if let Some(found) = constants_field_read(graph, &var) {
        return Some(found);
    }
    // `<[T]>::len` often sees the data word (or the address the word was
    // loaded from), not the fat field read itself. The length is still
    // the box's metadata word.
    if let Some(found) = fat_len_through_data_word(graph, fat_box_vars, &var) {
        return Some(found);
    }
    let phis: Vec<(BlockId, usize)> = graph
        .blocks
        .iter()
        .filter_map(|block| {
            block
                .inputargs
                .iter()
                .position(|arg| arg == &var)
                .map(|index| (block.id, index))
        })
        .collect();
    if phis.is_empty() {
        return None;
    }
    let mut agreed: Option<(Variable, FieldDescriptor, bool)> = None;
    for (block_id, index) in phis {
        let preds = graph.predecessors(block_id);
        if preds.is_empty() {
            return None;
        }
        for pred_id in preds {
            let sources: Vec<Option<Variable>> = graph
                .block(pred_id)
                .exits
                .iter()
                .filter(|link| link.target == block_id)
                .map(|link| {
                    link.args
                        .get(index)
                        .and_then(crate::model::LinkArg::as_variable)
                        .cloned()
                })
                .collect();
            if sources.is_empty() {
                return None;
            }
            for src in sources {
                let Some(src) = src else {
                    return None;
                };
                let Some(prod) = direct_fat_field_read(graph, fat_box_vars, &src) else {
                    return None;
                };
                if let Some((base, field, _)) = &agreed {
                    if base != &prod.0
                        || field.name != prod.1.name
                        || field.owner_root != prod.1.owner_root
                    {
                        return None;
                    }
                } else {
                    agreed = Some(prod);
                }
            }
        }
    }
    agreed
}

/// `<[T]>::to_vec` — Rust MIR `alloc::slice::<Impl>::to_vec`, a slice→owned-Vec
/// copy. Retargeted to the `list` builtin (`rtype_bltn_list` -> `ll_copy`).
pub(in crate::front::mir) fn is_slice_to_vec(segments: &[String]) -> bool {
    matches!(segments, [a, b, c, d]
        if a == "alloc" && b == "slice" && c == "<Impl>" && d == "to_vec")
}

pub(in crate::front::mir) fn regular_call_is_slice_to_vec(reg: &RegularCall, llbc: &Llbc) -> bool {
    regular_call_name_path(reg, llbc)
        .as_deref()
        .is_some_and(|name| {
            let segs: Vec<String> = name
                .split("::")
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            is_slice_to_vec(&segs)
        })
}

/// `core::slice::from_ref` / `from_mut` and the `raw` / `<Impl>`
/// spellings. rustc `slice::from_ref` is `from_raw_parts(s, 1)`.
pub(in crate::front::mir) fn is_slice_from_ref_path(segments: &[String]) -> bool {
    let Some(leaf) = segments.last() else {
        return false;
    };
    if leaf != "from_ref" && leaf != "from_mut" {
        return false;
    }
    let Some(first) = segments.first() else {
        return false;
    };
    (first == "core" || first == "std") && segments.iter().any(|s| s == "slice")
}

pub(in crate::front::mir) fn is_slice_from_ref_name(name: &str) -> bool {
    let leaf = name.rsplit("::").next().unwrap_or(name);
    (leaf == "from_ref" || leaf == "from_mut")
        && name.contains("::slice::")
        && (name.starts_with("core::") || name.starts_with("std::"))
}

pub(in crate::front::mir) fn is_vec_extend_segments(segments: &[String]) -> bool {
    segments == ["vec", "Vec", "extend"]
}

/// Whether a call operand is a concrete `Vec<T>` value (possibly borrowed).
/// A `Vec::IntoIter<T>` is deliberately excluded: it is an iterator repr, may
/// already be partially consumed, and therefore cannot be treated as the
/// complete source list merely because it originated from one.  An arbitrary
/// `IntoIterator<Item=T>` has no list-representation proof either.
pub(in crate::front::mir) fn tyref_is_vec_value(ty: &TyRef, llbc: &Llbc) -> bool {
    adt_path_of_tyref(ty, llbc).as_deref() == Some("alloc::vec::Vec")
}

/// `Vec<T>` whose `Clone` is `rlist.py` `ll_copy` (`ll_arraycopy`).
/// `ll_copy` copies element words and never calls `T::clone`, so this
/// matches `Vec<T>::clone` only when T's clone is the identity copy: T
/// is Copy, or T is a GC reference whose Clone is identity
/// (`json_ty_is_copy_or_zst_clone_identity`). Nested `Vec` and `String`
/// stay residual.
pub(in crate::front::mir) fn tyref_vec_clone_is_ll_copy(ty: &TyRef, llbc: &Llbc) -> bool {
    if !tyref_is_vec_value(ty, llbc) {
        return false;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc)) else {
        return false;
    };
    let Some(elem) = node
        .pointer("/Adt/generics/types")
        .and_then(serde_json::Value::as_array)
        .and_then(|types| types.first())
    else {
        return false;
    };
    json_ty_is_copy_or_zst_clone_identity(elem, llbc)
}

/// One `&` / `&mut` / `*const` / `*mut` whose pointee is `alloc::vec::Vec`.
/// The value is the header address (`RustVecRepr`); `as usize` is a
/// pointer-to-integer cast, not an integer retype. A `Vec` by value is the
/// header itself and is not this shape.
pub(in crate::front::mir) fn tyref_ptr_pointee_is_vec(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(pointee) =
        tyref_peel_one_ref_node(ty, llbc).or_else(|| tyref_peel_one_raw_ptr_node(ty, llbc))
    else {
        return false;
    };
    adt_node_def_id(pointee)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path() == "alloc::vec::Vec")
}

/// `Ptr(Struct(raw) "RustVec")` for `kind` (`rrustvec.rs rust_vec_lltype`).
///
/// An integer item's lltype follows the Rust spelling: `usize` / `u64` are
/// `Unsigned` (`rint.py` `unsigned_repr`), `isize` / `i64` stay `Signed`.
/// The register kind stays `Int` either way. No spelling keeps `Signed`.
pub(in crate::front::mir) fn rust_vec_header_lltype(
    kind: majit_ir::rvec::VecItemKind,
    item_spelling: Option<&str>,
) -> crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::rrustvec::rust_vec_lltype;
    let item = match kind {
        majit_ir::rvec::VecItemKind::Int => match item_spelling {
            Some("usize" | "u64") => LowLevelType::Unsigned,
            _ => LowLevelType::Signed,
        },
        majit_ir::rvec::VecItemKind::Float => LowLevelType::Float,
        majit_ir::rvec::VecItemKind::Ref => {
            crate::translator::rtyper::lltypesystem::lltype::GCREF.clone()
        }
    };
    rust_vec_lltype(&item)
}

/// Stamp `rust_vec_lltype` on `var` when `ty` is a one-word `Vec`.
///
/// The lowering's `setconcretetype`: `tyref_rust_vec_item_kind` reads the
/// LLBC `Vec` and the lltype is `RustVecRepr.lowleveltype`. The rtyper
/// overwrites it from the annotation when the graph is specialized. A
/// graph that skips specialization keeps this stamp.
pub(in crate::front::mir) fn stamp_rust_vec_header(
    var: &crate::flowspace::model::Variable,
    ty: &TyRef,
    llbc: &Llbc,
) {
    let Some(kind) = tyref_rust_vec_item_kind(ty, llbc) else {
        return;
    };
    if var.concretetype().is_some() {
        return;
    }
    let spelling = tyref_rust_vec_item_spelling(ty, llbc);
    var.set_concretetype(Some(rust_vec_header_lltype(kind, spelling.as_deref())));
}

/// AST spelling of a one-word `alloc::vec::Vec<T>`, by value, behind a
/// borrow, or behind a raw pointer. `None` when the items are not one word.
pub(in crate::front::mir) fn tyref_one_word_vec_ast(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let mut node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    if let Some(pointee) = node
        .get("RawPtr")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    {
        node = strip_ty_wrappers(pointee, llbc)?;
    }
    let id = adt_node_def_id(node)?;
    if llbc.type_by_id(id)?.item_meta.name_path() != "alloc::vec::Vec" {
        return None;
    }
    let ast = charon_type_value_to_ast_string(node, llbc, 0);
    majit_ir::rvec::rust_vec_item_kind_for_spelling(&ast, crate::layout::target_word_size())?;
    Some(ast)
}

/// Item kind of an `alloc::vec::Vec<T>` whose items are one word — by value,
/// behind a borrow, or behind a raw pointer, all of which the translator
/// represents as the address of the raw `{ptr, len, cap}` header
/// (`RustVecRepr`).  `None` for every other type.
pub(in crate::front::mir) fn tyref_rust_vec_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    let ast = tyref_one_word_vec_ast(ty, llbc)?;
    majit_ir::rvec::rust_vec_item_kind_for_spelling(&ast, crate::layout::target_word_size())
}

/// Item spelling of [`tyref_rust_vec_item_kind`]'s `Vec` (`usize`, `i64`, …).
pub(in crate::front::mir) fn tyref_rust_vec_item_spelling(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<String> {
    let ast = tyref_one_word_vec_ast(ty, llbc)?;
    majit_ir::rvec::vec_item_spelling(&ast).map(|item| item.trim().to_string())
}

/// `Vec<PyObjectRef>` / `Vec<*mut PyObject>`, by value or behind one
/// borrow or raw pointer. The slice view of this vec is one object array.
pub(in crate::front::mir) fn tyref_is_object_pointer_rust_vec(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(mut node) = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc)) else {
        return false;
    };
    if let Some(pointee) = node
        .get("RawPtr")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    {
        let Some(peeled) = strip_ty_wrappers(pointee, llbc) else {
            return false;
        };
        node = peeled;
    }
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    if llbc
        .type_by_id(id)
        .is_none_or(|td| td.item_meta.name_path() != "alloc::vec::Vec")
    {
        return false;
    }
    let Some(item) = node
        .pointer("/Adt/generics/types")
        .and_then(serde_json::Value::as_array)
        .and_then(|types| types.first())
    else {
        return false;
    };
    json_ty_is_objectptr(item, llbc)
        || object_pointer_item_spelling(&charon_type_value_to_ast_string(item, llbc, 0))
}

/// Call arguments with each pair slice's length right after its pointer.
pub(in crate::front::mir) fn splice_pair_lens(
    args: Vec<Variable>,
    pair_lens: Vec<(usize, Variable)>,
) -> Vec<Variable> {
    let mut lens = pair_lens.into_iter().peekable();
    let mut out = Vec::with_capacity(args.len() + lens.len());
    for (index, arg) in args.into_iter().enumerate() {
        out.push(arg);
        if let Some((_, len)) = lens.next_if(|(at, _)| *at == index) {
            out.push(len);
        }
    }
    out
}

/// Item kinds whose `&[T]` / `&mut [T]` locals the front end carries as the
/// `(ptr, len)` pair: the pointer in the local itself and the length in its
/// shadow slot (see [`pair_len_shadows`]).
///
/// A shared object-pointer slice (`&[PyObjectRef]`) is the length-prefixed
/// object array, one `Ref` word (`arraylen_gc` / `getarrayitem_gc`). A
/// mutable object-pointer slice (`&mut [PyObjectRef]`) is the pair: the
/// view that aliases a `Vec`'s item buffer (`pair_of_vec`).
pub(in crate::front::mir) const PAIR_SLICE_ITEM_KINDS: &[majit_ir::rvec::VecItemKind] = &[
    majit_ir::rvec::VecItemKind::Int,
    majit_ir::rvec::VecItemKind::Ref,
];

/// The attribute holding the length word of the pair slice whose pointer
/// word an aggregate stores under `field`.
pub(crate) fn pair_len_field_name(field: &str) -> String {
    format!("{field}.len")
}

/// The enabled item kind of an aggregate field of type `ty` that holds a
/// pair slice: `&[T]` itself, or a reference to one, which aliases its
/// referent.
pub(in crate::front::mir) fn tyref_pair_field_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    tyref_pair_slice_item_kind(ty, llbc)
        .or_else(|| {
            let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
            let referent = node.get("Ref")?.as_array()?.get(1)?;
            tyref_pair_slice_item_kind(&TyRef::Other(referent.clone()), llbc)
        })
        .filter(|kind| PAIR_SLICE_ITEM_KINDS.contains(kind))
        // A struct field of `&mut [PyObjectRef]` stays the one-word object
        // array. Body locals of that type are the `{ptr, len}` view.
        .filter(|_| !tyref_is_object_pointer_mut_slice_or_iter(ty, llbc))
}

/// [`tyref_pair_field_kind`] over a rendered field spelling, where a
/// reference renders as its referent: `[T]` is a borrowed slice, since no
/// field holds an unsized slice by value.
pub(in crate::front::mir) fn spelling_pair_field_kind(
    spelling: &str,
) -> Option<majit_ir::rvec::VecItemKind> {
    let spelling = spelling.trim();
    let spelling = spelling
        .strip_prefix("*const ")
        .or_else(|| spelling.strip_prefix("*mut "))
        .unwrap_or(spelling);
    if !(spelling.starts_with('[') && spelling.ends_with(']'))
        || shaped_array_parts(spelling).is_some()
    {
        return None;
    }
    let item = spelling[1..spelling.len() - 1].trim();
    if object_pointer_item_spelling(item) {
        return None;
    }
    majit_ir::rvec::rust_slice_item_kind_for_spelling(spelling, crate::layout::target_word_size())
        .filter(|kind| PAIR_SLICE_ITEM_KINDS.contains(kind))
}

/// `PyObjectRef` / `*mut PyObject` / `*const PyObject`: the item of the
/// length-prefixed object array, not of a `(ptr, len)` pair.
pub(in crate::front::mir) fn object_pointer_item_spelling(item: &str) -> bool {
    let item = item.trim();
    if item == "PyObjectRef" || item.ends_with("::PyObjectRef") {
        return true;
    }
    let Some(pointee) = item
        .strip_prefix("*mut ")
        .or_else(|| item.strip_prefix("*const "))
    else {
        return false;
    };
    let pointee = pointee.trim();
    pointee == "PyObject" || pointee.ends_with("::PyObject")
}

/// `GCREF` / `majit_gc::GCREF`: a GC box, not a host pointer. A `[GCREF; N]`
/// is a `GcArray` or a `push_roots` livevar list (`shadowstack.py`), not a
/// raw `lltype.malloc(Array(T), N, flavor='raw')` buffer.
pub(in crate::front::mir) fn gcref_item_spelling(item: &str) -> bool {
    let item = item.trim();
    item == "GCREF" || item.ends_with("::GCREF")
}

/// The item kind of `Option<&[T]>` / `Option<&mut [T]>` when the payload
/// slice is a pair. `None` is a null item pointer; both words are the pair.
pub(in crate::front::mir) fn tyref_option_pair_slice_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    let payload = crate::front::result_exc::tyref_option_payload(ty, llbc)?;
    tyref_pair_slice_item_kind(&payload, llbc).filter(|kind| PAIR_SLICE_ITEM_KINDS.contains(kind))
}

/// `&[PyObjectRef]` / `&[*mut PyObject]` (and the mut / raw-pointer
/// spellings): the length-prefixed object array, one GcArray word.
pub(in crate::front::mir) fn tyref_is_object_pointer_slice(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_object_pointer_slice_mutability(ty, llbc).is_some()
}

/// Shared / const object-pointer slice: one GcArray word. Mut slices of
/// the same items are the `{ptr, len}` pair and stay out.
pub(in crate::front::mir) fn tyref_is_object_pointer_gcarray_slice(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    matches!(
        tyref_object_pointer_slice_mutability(ty, llbc),
        Some("Shared" | "Const")
    )
}

/// `&T` whose pointee is a `PyObject` pointer — the argument of
/// `slice::from_ref` that becomes `newlist(x)`.
pub(in crate::front::mir) fn tyref_is_shared_ref_to_object_pointer(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return false;
    };
    let Some(arr) = node.get("Ref").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let Some(kind) = arr.get(2).and_then(serde_json::Value::as_str) else {
        return false;
    };
    if kind != "Shared" && kind != "Const" {
        return false;
    }
    let Some(pointee) = arr.get(1) else {
        return false;
    };
    let pointee = strip_ty_indirections(pointee, llbc).unwrap_or(pointee);
    json_ty_is_objectptr(pointee, llbc)
        || object_pointer_item_spelling(&charon_type_value_to_ast_string(pointee, llbc, 0))
}

/// Mutability of an object-pointer slice: `"Shared"` / `"Const"` for
/// `&[T]` / `*const [T]`, `"Mut"` for `&mut [T]` / `*mut [T]`.
pub(in crate::front::mir) fn tyref_object_pointer_slice_mutability<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<&'l str> {
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    let (pointee, kind) = match (node.get("Ref"), node.get("RawPtr")) {
        (Some(reference), _) => {
            let arr = reference.as_array()?;
            (arr.get(1)?, arr.get(2).and_then(serde_json::Value::as_str)?)
        }
        (None, Some(raw)) => {
            let arr = raw.as_array()?;
            (
                arr.first()?,
                arr.get(1).and_then(serde_json::Value::as_str)?,
            )
        }
        (None, None) => return None,
    };
    let pointee = strip_ty_indirections(pointee, llbc)?;
    let item = pointee
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())?;
    (json_ty_is_objectptr(item, llbc)
        || object_pointer_item_spelling(&charon_type_value_to_ast_string(item, llbc, 0)))
    .then_some(kind)
}

/// The item kind of a `&[T]` / `&mut [T]` / `*const [T]` / `*mut [T]` whose
/// items are one word: a fat pointer, one level over a slice.  `&[u8]`, `&str` and a borrow of
/// a fixed-length array answer `None`.
pub(in crate::front::mir) fn tyref_pair_slice_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    // `&[T]` spells `{"Ref": [region, pointee, kind]}`, `*const [T]`
    // `{"RawPtr": [pointee, kind]}`.
    let pointee = match (node.get("Ref"), node.get("RawPtr")) {
        (Some(reference), _) => reference.as_array()?.get(1)?,
        (None, Some(raw)) => raw.as_array()?.first()?,
        (None, None) => return None,
    };
    let pointee = strip_ty_indirections(pointee, llbc)?;
    // `[T]` spells `{"Slice": [elem, trait_ref]}`.
    let item = pointee.get("Slice")?.as_array()?.first()?;
    if let Some(kind) = pair_item_kind_of_node(item, llbc) {
        return Some(kind);
    }
    // Object-pointer `&mut [T]` / `*mut [T]`: the `{ptr, len}` view that
    // aliases a `Vec` item buffer. A shared `&[T]` of those items is the
    // length-prefixed object array (`gcarray_from_pyobject_vec`).
    if slice_item_node_is_object_pointer(item, llbc)
        && (ref_kind_is_mut(ty, llbc) || raw_ptr_kind_is_mut(ty, llbc))
    {
        return Some(majit_ir::rvec::VecItemKind::Ref);
    }
    None
}

pub(in crate::front::mir) fn slice_item_node_is_object_pointer(
    item: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    json_ty_is_objectptr(item, llbc)
        || object_pointer_item_spelling(&charon_type_value_to_ast_string(item, llbc, 0))
}

/// The item kind of a one-word slice item type node.  A reference item `&T`
/// is an address in the slice but the referent's value everywhere else in
/// the value model, so it is never an item of the pair.  An object pointer
/// is the length-prefixed object array ([`object_pointer_item_spelling`]).
pub(in crate::front::mir) fn pair_item_kind_of_node(
    item: &serde_json::Value,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    if strip_ty_indirections(item, llbc)?.get("Ref").is_some() {
        return None;
    }
    if json_ty_is_objectptr(item, llbc) {
        return None;
    }
    let spelling = charon_type_value_to_ast_string(item, llbc, 0);
    if object_pointer_item_spelling(&spelling) {
        return None;
    }
    majit_ir::rvec::rust_slice_item_kind_for_spelling(
        &format!("[{spelling}]"),
        crate::layout::target_word_size(),
    )
}

/// One-word item kind of `*const T` / `*mut T` for `copy_nonoverlapping`.
pub(in crate::front::mir) fn raw_ptr_one_word_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    let pointee = type_node_raw_ptr_pointee(tyref_node(ty, llbc)?, llbc)?;
    pair_item_kind_of_node(pointee, llbc)
}

/// The item kind of a `core::slice::iter::Iter<T>` / `IterMut<T>` over one-word
/// items.  Such an iterator local is a range iterator over the indices of the
/// slice, with the slice's item pointer in its second slot.
pub(in crate::front::mir) fn tyref_pair_slice_iter_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<majit_ir::rvec::VecItemKind> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let path = llbc
        .type_by_id(adt_node_def_id(node)?)?
        .item_meta
        .name_path();
    if path != "core::slice::iter::Iter" && path != "core::slice::iter::IterMut" {
        return None;
    }
    let item = node
        .get("Adt")?
        .get("generics")?
        .get("types")?
        .as_array()?
        .first()?;
    if let Some(kind) = pair_item_kind_of_node(item, llbc) {
        return Some(kind);
    }
    // `IterMut` over an object pointer: the iterator of the `{ptr, len}`
    // view. Shared `Iter` of those items stays the object-array walk.
    if path == "core::slice::iter::IterMut" && slice_item_node_is_object_pointer(item, llbc) {
        return Some(majit_ir::rvec::VecItemKind::Ref);
    }
    None
}

/// An object-pointer `&mut [T]` / `IterMut<T>` whose parameter form keeps
/// the one-word helper ABI. Body locals of the same type are the pair view.
pub(in crate::front::mir) fn tyref_is_object_pointer_mut_slice_or_iter(
    ty: &TyRef,
    llbc: &Llbc,
) -> bool {
    if let Some(payload) = crate::front::result_exc::tyref_option_payload(ty, llbc) {
        return tyref_is_object_pointer_mut_slice_or_iter(&payload, llbc);
    }
    slice_ty_item_is_object_pointer(ty, llbc)
        && (ref_kind_is_mut(ty, llbc) || raw_ptr_kind_is_mut(ty, llbc))
        || iter_mut_item_is_object_pointer(ty, llbc)
}

pub(in crate::front::mir) fn slice_ty_item_is_object_pointer(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc)) else {
        return false;
    };
    let Some(pointee) = (match (node.get("Ref"), node.get("RawPtr")) {
        (Some(reference), _) => reference.as_array().and_then(|a| a.get(1)),
        (None, Some(raw)) => raw.as_array().and_then(|a| a.first()),
        (None, None) => None,
    }) else {
        return false;
    };
    let Some(pointee) = strip_ty_indirections(pointee, llbc) else {
        return false;
    };
    pointee
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|a| a.first())
        .is_some_and(|item| slice_item_node_is_object_pointer(item, llbc))
}

pub(in crate::front::mir) fn iter_mut_item_is_object_pointer(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc)) else {
        return false;
    };
    let Some(path) = adt_node_def_id(node)
        .and_then(|id| llbc.type_by_id(id))
        .map(|td| td.item_meta.name_path())
    else {
        return false;
    };
    if path != "core::slice::iter::IterMut" {
        return false;
    }
    node.get("Adt")
        .and_then(|adt| adt.get("generics"))
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)
        .and_then(|types| types.first())
        .is_some_and(|item| slice_item_node_is_object_pointer(item, llbc))
}

/// The value type of the pointee of a `&T` / `&mut T`.
/// The pointee of `&T` / `&mut T`, if `ty` is a reference.
pub(in crate::front::mir) fn ref_pointee_ty(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let pointee = node.get("Ref")?.as_array()?.get(1)?;
    Some(TyRef::Other(pointee.clone()))
}

pub(in crate::front::mir) fn tyref_ref_pointee_value_type(
    ty: &TyRef,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<ValueType> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let pointee = node.get("Ref")?.as_array()?.get(1)?;
    Some(tyref_to_value_type_with(
        &TyRef::Other(pointee.clone()),
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    ))
}

/// The length slot of each MIR local that holds a pair slice of an enabled
/// item kind.  Slot `n_locals + k` belongs to the `k`-th such local, so the
/// frame rows, liveness sets and link arguments carry the length beside the
/// pointer without a second value space.
///
/// A body-local reference to a pair slice aliases that pair and holds it
/// too.  The return value and the parameters keep their signature's
/// words, so only a `&[T]` there is a pair.
/// Locals defined by `get` / `first` / `last` on a pair slice, plus
/// `copied` / `cloned` of such a local. The address word is what the local
/// holds; the item kind is read from the local's MIR type at the use.
/// Destination local of `next` on a pair-slice iterator, mapped to the
/// iterator local the call reads. One-shot, from the body.
pub(in crate::front::mir) fn slice_next_iters(
    body: &Unstructured,
    llbc: &Llbc,
) -> Vec<Option<usize>> {
    let n = body.locals.locals.len();
    let is_iter_local = |local: usize| {
        body.locals.locals.get(local).is_some_and(|loc| {
            tyref_pair_slice_iter_item_kind(&loc.ty, llbc)
                .is_some_and(|kind| PAIR_SLICE_ITEM_KINDS.contains(&kind))
        })
    };
    // `&mut it` stores the iterator in the borrow local. `next` reads that
    // borrow; the items stay on `it`, which is what stays live across the
    // loop.
    let mut referent = vec![None; n];
    let mut changed = true;
    let forward_blocks = forward_reachable_mask(llbc, body);
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(dest) = place.kind else {
                    continue;
                };
                let (Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. }) = rvalue
                else {
                    continue;
                };
                let src_local = match &src.kind {
                    PlaceKind::Local(local) => *local as usize,
                    PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                        let PlaceKind::Local(local) = inner.kind else {
                            continue;
                        };
                        local as usize
                    }
                    _ => continue,
                };
                let iter = if is_iter_local(src_local) {
                    Some(src_local)
                } else {
                    referent.get(src_local).copied().flatten()
                };
                if let Some(iter) = iter
                    && referent[dest as usize] != Some(iter)
                {
                    referent[dest as usize] = Some(iter);
                    changed = true;
                }
            }
        }
    }
    let mut out = vec![None; n];
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let Some(path) = regular_call_name_path(reg, llbc) else {
            continue;
        };
        let Some(operand) = call.args.first() else {
            continue;
        };
        let (Operand::Copy(place) | Operand::Move(place)) = operand else {
            continue;
        };
        let local = match &place.kind {
            PlaceKind::Local(local) => *local as usize,
            PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                let PlaceKind::Local(local) = inner.kind else {
                    continue;
                };
                local as usize
            }
            _ => continue,
        };
        let iter = referent.get(local).copied().flatten().unwrap_or(local);
        let is_iter = tyref_pair_slice_iter_item_kind(&place.ty, llbc)
            .is_some_and(|kind| PAIR_SLICE_ITEM_KINDS.contains(&kind))
            || is_iter_local(local)
            || is_iter_local(iter);
        // The receiver's MIR type is the slice `Iter` / `IterMut`, or the
        // borrow resolves to one. The path leaf is not enough on its own.
        let path_ok =
            is_iter && (path == "core::slice::iter::<Impl>::next" || path.ends_with("::next"));
        if path_ok && (dest as usize) < n {
            out[dest as usize] = Some(iter);
        }
    }
    out
}

pub(in crate::front::mir) fn item_addr_locals(
    body: &Unstructured,
    llbc: &Llbc,
    slice_next_iter: &[Option<usize>],
) -> Vec<bool> {
    let n = body.locals.locals.len();
    let mut addr = vec![false; n];
    let mut changed = true;
    let forward_blocks = forward_reachable_mask(llbc, body);
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(dest) = place.kind else {
                    continue;
                };
                let dest = dest as usize;
                if dest >= n || addr[dest] {
                    continue;
                }
                let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = rvalue else {
                    continue;
                };
                let from_addr = match &src.kind {
                    PlaceKind::Local(src_local) => {
                        addr.get(*src_local as usize).copied().unwrap_or(false)
                    }
                    PlaceKind::Projection(inner, ProjectionElem::Tagged(v))
                        if v.get("Field").is_some() =>
                    {
                        let PlaceKind::Local(opt) = inner.kind else {
                            continue;
                        };
                        slice_next_iter
                            .get(opt as usize)
                            .copied()
                            .flatten()
                            .is_some()
                            && ref_kind_is_mut(&place.ty, llbc)
                    }
                    _ => false,
                };
                if from_addr {
                    addr[dest] = true;
                    changed = true;
                }
            }
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let PlaceKind::Local(dest) = call.dest.kind else {
                continue;
            };
            let dest = dest as usize;
            if dest >= n || addr[dest] {
                continue;
            }
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let Some(name) = regular_call_name_path(reg, llbc) else {
                continue;
            };
            let recv_place = call.args.first().and_then(|op| match op {
                Operand::Copy(place) | Operand::Move(place) => Some(place),
                Operand::Const(_) => None,
            });
            let recv = recv_place.and_then(|place| match place.kind {
                PlaceKind::Local(local) => Some(local as usize),
                _ => None,
            });
            let pair_recv = recv_place.is_some_and(|place| {
                tyref_pair_slice_item_kind(&place.ty, llbc)
                    .is_some_and(|kind| PAIR_SLICE_ITEM_KINDS.contains(&kind))
                    || tyref_pair_field_kind(&place.ty, llbc).is_some()
            });
            let mark = match name.as_str() {
                "core::slice::<Impl>::first"
                | "core::slice::<Impl>::last"
                | "core::slice::<Impl>::get" => pair_recv,
                "core::option::<Impl>::copied" | "core::option::<Impl>::cloned" => {
                    recv.is_some_and(|local| addr.get(local).copied().unwrap_or(false))
                }
                _ => false,
            };
            if mark {
                addr[dest] = true;
                changed = true;
            }
        }
    }
    addr
}

pub(in crate::front::mir) fn pair_len_shadows(
    body: &Unstructured,
    llbc: &Llbc,
    graph_name: &str,
    accum: &AccumulatorFacts,
) -> (Vec<Option<usize>>, Vec<bool>, Vec<bool>, Vec<bool>) {
    let views = objectptr_pair_view_locals(body, llbc);
    let (mut str_views, list_views, list_view_fixed) = body_owner_view_locals(body, llbc, accum);
    // `RDict` / `HashMap` key parameters are `space.bytes_w` / `str`
    // (`BytesDictStrategy.unwrap`), one `Ptr(STR)` word. A window ABI on
    // those formals splices the key and harvests `equivalent` (`PartialEq::eq`).
    mark_dict_method_str_view_keys(graph_name, body, llbc, &mut str_views);
    (
        pair_len_shadows_of(&body.locals, llbc, &views, &str_views, &list_views),
        str_views,
        list_views,
        list_view_fixed,
    )
}

/// Whether `ty` is a W1 STR owner (`rstr.py` `STR` / `space.bytes_w`).
pub(in crate::front::mir) fn tyref_is_w1_str(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_string_value(ty, llbc) || tyref_raw_ptr_pointee_is_string_value(ty, llbc)
}

/// W1 STR receiver: a W1 STR value (not a Rust-heap `String`, which is a
/// `{cap, ptr, len}` instance), or a `#[repr(transparent)]` newtype whose
/// single non-ZST field is a W1 STR.
pub(in crate::front::mir) fn tyref_is_w1_str_receiver(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_is_w1_str(ty, llbc) {
        return true;
    }
    let Some((index, _)) = tyref_transparent_nonzst_field(ty, llbc) else {
        return false;
    };
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(decl) = adt_node_def_id(node).and_then(|id| llbc.type_by_id(id)) else {
        return false;
    };
    let TypeDeclKind::Struct(fields) = &decl.kind else {
        return false;
    };
    fields
        .get(index)
        .is_some_and(|field| tyref_is_w1_str(&field.ty, llbc))
}

/// Shared `&[u8]` / `*const [u8]` spelled as Charon ADT `builtin: "Slice"`
/// of `u8`. Distinct from the top-level `{"Slice": [u8, …]}` node that
/// [`tyref_is_shared_u8_slice`] matches; both are a W1 view dest.
pub(in crate::front::mir) fn tyref_is_adt_shared_u8_slice(ty: &TyRef, llbc: &Llbc) -> bool {
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
    let Some(adt) = pointee.get("Adt") else {
        return false;
    };
    if type_decl_ref_builtin(adt) != Some("Slice") {
        return false;
    }
    let Some(item) = adt
        .get("generics")
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
    else {
        return false;
    };
    json_ty_scalar_element_spelling(item, llbc).as_deref() == Some("u8")
}

/// Shared `&[u8]` / `&str`: a W1 view dest of a STR owner.
/// Covers both Charon spellings of shared `&[u8]`: top-level `Slice` and
/// ADT `builtin: "Slice"`.
pub(in crate::front::mir) fn tyref_is_w1_str_view_dest(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_is_shared_u8_slice(ty, llbc)
        || tyref_strips_to_str(ty, llbc)
        || tyref_is_adt_shared_u8_slice(ty, llbc)
}

pub(in crate::front::mir) fn place_is_w1_str_or_marked(
    place: &Place,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    if tyref_is_w1_str(&place.ty, llbc) {
        return true;
    }
    match &place.kind {
        PlaceKind::Local(i) => mark.get(*i as usize).copied().unwrap_or(false),
        PlaceKind::Projection(inner, _) => place_is_w1_str_or_marked(inner, mark, llbc),
        PlaceKind::Global { .. } | PlaceKind::Unknown => false,
    }
}

pub(in crate::front::mir) fn operand_is_w1_str_or_marked(
    op: &Operand,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    match op {
        Operand::Copy(place) | Operand::Move(place) => place_is_w1_str_or_marked(place, mark, llbc),
        Operand::Const(_) => false,
    }
}

pub(in crate::front::mir) fn rvalue_copies_str_view(
    rvalue: &Rvalue,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    match rvalue {
        Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => operand_is_w1_str_or_marked(op, mark, llbc),
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_is_w1_str_or_marked(place, mark, llbc)
        }
        _ => false,
    }
}

pub(in crate::front::mir) fn place_is_w2_list_or_marked(
    place: &Place,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    if tyref_is_u8_list(&place.ty, llbc) {
        return true;
    }
    match &place.kind {
        PlaceKind::Local(i) => mark.get(*i as usize).copied().unwrap_or(false),
        PlaceKind::Projection(inner, _) => place_is_w2_list_or_marked(inner, mark, llbc),
        PlaceKind::Global { .. } | PlaceKind::Unknown => false,
    }
}

pub(in crate::front::mir) fn operand_is_w2_list_or_marked(
    op: &Operand,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    match op {
        Operand::Copy(place) | Operand::Move(place) => {
            place_is_w2_list_or_marked(place, mark, llbc)
        }
        Operand::Const(_) => false,
    }
}

pub(in crate::front::mir) fn rvalue_copies_list_view(
    rvalue: &Rvalue,
    mark: &[bool],
    llbc: &Llbc,
) -> bool {
    match rvalue {
        Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => operand_is_w2_list_or_marked(op, mark, llbc),
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_is_w2_list_or_marked(place, mark, llbc)
        }
        _ => false,
    }
}

pub(in crate::front::mir) fn rvalue_list_view_is_fixed(rvalue: &Rvalue, llbc: &Llbc) -> bool {
    match rvalue {
        Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
            operand_tyref(op).is_some_and(|ty| tyref_is_fixed_u8_list(ty, llbc))
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            tyref_is_fixed_u8_list(&place.ty, llbc)
        }
        _ => false,
    }
}

/// A 1-arg call whose dest is a shared `&[u8]` and whose first argument
/// is a W2 list owner (`Vec<u8>` / `&mut [u8]` / `Box<[u8]>`).
pub(in crate::front::mir) fn call_dest_is_w2_list_source(call: &CallPayload, llbc: &Llbc) -> bool {
    if !tyref_is_u8_window(&call.dest.ty, llbc) || call.args.len() != 1 {
        return false;
    }
    call.args
        .first()
        .and_then(operand_tyref)
        .is_some_and(|ty| tyref_is_u8_list(ty, llbc))
}

/// Callee leaf from the resolved FunDecl `name_path`, or the registered
/// path when that leaf is empty. A missing FunDecl (other-crate callee
/// not in this LLBC) is `None`, not an error.
pub(in crate::front::mir) fn callee_leaf_name(reg: &RegularCall, llbc: &Llbc) -> Option<String> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let path = fd.item_meta.name_path();
    if let Some(leaf) = path.rsplit("::").next().filter(|s| !s.is_empty()) {
        return Some(leaf.to_string());
    }
    registered_path_for_fun_decl(llbc, fd)
        .segments
        .last()
        .cloned()
}

/// `as_*` view no-op: the callee leaf starts with `as_` and the receiver
/// shares a representation class with the dest.
///
/// W1: receiver is a STR (`tyref_is_string_family` / `tyref_is_w1_str`)
/// or a transparent newtype over a W1 STR source pointer; dest is a
/// shared `&[u8]` / `&str`. The dest binds to the receiver's STR word
/// (the newtype's single field is that word).
/// W2: receiver is a list and dest is a list view (`as_slice`). Already
/// lowered by [`Lowering::is_container_slice_identity`]; kept here so
/// `as_ptr` (list → CCHARP) is not a no-op.
pub(in crate::front::mir) fn is_as_view_noop(
    llbc: &Llbc,
    reg: &RegularCall,
    first_arg_ty: Option<&TyRef>,
    dest_ty: &TyRef,
) -> bool {
    let Some(leaf) = callee_leaf_name(reg, llbc) else {
        return false;
    };
    if !leaf.starts_with("as_") {
        return false;
    }
    let Some(first) = first_arg_ty else {
        return false;
    };
    if tyref_is_w1_str_receiver(first, llbc) && tyref_is_w1_str_view_dest(dest_ty, llbc) {
        return true;
    }
    tyref_is_u8_list(first, llbc)
        && (tyref_is_shared_u8_slice(dest_ty, llbc) || tyref_is_adt_shared_u8_slice(dest_ty, llbc))
}

/// A 1-arg call whose dest is a shared `&[u8]` / `*const [u8]` and whose
/// defining value is a W1 STR source (`BytesDictStrategy.unwrap` /
/// `space.bytes_w`). Matches [`Lowering::lower_window_dest_call`]'s
/// `str_src` predicate.
pub(in crate::front::mir) fn call_dest_is_w1_str_source(call: &CallPayload, llbc: &Llbc) -> bool {
    if !tyref_is_u8_window(&call.dest.ty, llbc) {
        return false;
    }
    let CallFunc::Regular(reg) = &call.func else {
        return false;
    };
    if call.args.len() != 1 {
        return false;
    }
    let first_ty = call.args.first().and_then(operand_tyref);
    let w1_as_view = is_as_view_noop(llbc, reg, first_ty, &call.dest.ty)
        && first_ty.is_some_and(|ty| tyref_is_w1_str_receiver(ty, llbc));
    let Some(path) = regular_call_name_path(reg, llbc) else {
        return w1_as_view;
    };
    let leaf = path.rsplit("::").next().unwrap_or("");
    w1_as_view
        || (leaf == "bytes"
            && path.contains("str")
            && first_ty.is_some_and(|ty| tyref_is_w1_str_receiver(ty, llbc)))
        || leaf == "utf8_payload_bytes"
        || leaf == "bytes_block_chars"
        || leaf == "w_bytes_data"
        || leaf == "w_bytes_block"
        || leaf == "w_str_storage"
}

/// A 1-arg `as_*` / `str::bytes`-style identity call whose argument is the
/// identity temp of a builder-mode accumulator: `ll_build`
/// (`AbstractStringBuilderRepr.rtype_method_build`) makes the receiver a
/// `Ptr(STR)`, so the window dest is a W1 STR view of the built string.
pub(in crate::front::mir) fn call_dest_is_built_str_view(
    call: &CallPayload,
    accum: &AccumulatorFacts,
) -> bool {
    if !accum.has_builder || call.args.len() != 1 {
        return false;
    }
    let Some(rt) = operand_local(call.args.first()) else {
        return false;
    };
    let Some(&Some(c)) = accum.identity_accumulator.get(rt) else {
        return false;
    };
    accum.builder_mode.get(c).copied() == Some(true)
}

/// A string identity call ([`is_string_identity_call`]) whose argument is a
/// [`AccumulatorFacts::format_str`] local: the dest is a W1 STR view of the
/// formatted string.
pub(in crate::front::mir) fn call_dest_is_format_str_view(
    call: &CallPayload,
    llbc: &Llbc,
    accum: &AccumulatorFacts,
) -> bool {
    is_string_identity_call(llbc, call)
        && operand_local(call.args.first()).is_some_and(|l| accum.format_str.get(l) == Some(&true))
}

/// Body locals whose every window-typed def is a W1 STR source or a W2
/// list view (or a Use/Ref/Cast/reborrow of one). Parameters keep the
/// two-word window ABI, except `RDict` / `HashMap` key formals
/// (`mark_dict_method_str_view_keys`). A mixed STR/list/window local is
/// excluded so it keeps a length shadow.
pub(in crate::front::mir) fn body_owner_view_locals(
    body: &Unstructured,
    llbc: &Llbc,
    accum: &AccumulatorFacts,
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let n = body.locals.locals.len();
    let arg_count = body.locals.arg_count as usize;
    let mut str_view = vec![false; n];
    let mut list_view = vec![false; n];
    let mut list_view_fixed = vec![false; n];
    let mut window_def = vec![false; n];
    let mut pending_window = vec![false; n];
    let mut w1_call_def = vec![false; n];
    let mut list_call_def = vec![false; n];
    let is_body_local = |dest: usize| dest < n && !(dest >= 1 && dest <= arg_count);
    for bb in &body.body {
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && let PlaceKind::Local(dest) = call.dest.kind
        {
            let dest = dest as usize;
            if is_body_local(dest) && tyref_is_u8_window(&call.dest.ty, llbc) {
                if call_dest_is_w1_str_source(&call, llbc)
                    || call_dest_is_built_str_view(&call, accum)
                    || call_dest_is_format_str_view(&call, llbc, accum)
                {
                    str_view[dest] = true;
                    w1_call_def[dest] = true;
                } else if call_dest_is_w2_list_source(&call, llbc) {
                    list_view[dest] = true;
                    list_call_def[dest] = true;
                    if call
                        .args
                        .first()
                        .and_then(operand_tyref)
                        .is_some_and(|ty| tyref_is_fixed_u8_list(ty, llbc))
                    {
                        list_view_fixed[dest] = true;
                    }
                } else {
                    pending_window[dest] = true;
                }
            }
        }
        for st in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind_ref() else {
                continue;
            };
            let PlaceKind::Local(dest) = place.kind else {
                continue;
            };
            let dest = dest as usize;
            if !is_body_local(dest) || !tyref_is_u8_window(&place.ty, llbc) {
                continue;
            }
            match rvalue {
                Rvalue::Use(_, _)
                | Rvalue::Cast(_, _, _)
                | Rvalue::Ref { .. }
                | Rvalue::RawPtr { .. } => {}
                _ => window_def[dest] = true,
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for bb in &body.body {
            if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
                && let PlaceKind::Local(dest) = call.dest.kind
            {
                let dest = dest as usize;
                if is_body_local(dest) && tyref_is_u8_window(&call.dest.ty, llbc) {
                    if !str_view[dest]
                        && call
                            .args
                            .first()
                            .is_some_and(|op| operand_is_w1_str_or_marked(op, &str_view, llbc))
                    {
                        str_view[dest] = true;
                        changed = true;
                    }
                    if !list_view[dest]
                        && call
                            .args
                            .first()
                            .is_some_and(|op| operand_is_w2_list_or_marked(op, &list_view, llbc))
                    {
                        list_view[dest] = true;
                        if call
                            .args
                            .first()
                            .and_then(operand_tyref)
                            .is_some_and(|ty| tyref_is_fixed_u8_list(ty, llbc))
                        {
                            list_view_fixed[dest] = true;
                        }
                        changed = true;
                    }
                }
            }
            for st in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(dest) = place.kind else {
                    continue;
                };
                let dest = dest as usize;
                if !is_body_local(dest) || !tyref_is_u8_window(&place.ty, llbc) {
                    continue;
                }
                if !str_view[dest] && rvalue_copies_str_view(rvalue, &str_view, llbc) {
                    str_view[dest] = true;
                    changed = true;
                }
                if !list_view[dest] && rvalue_copies_list_view(rvalue, &list_view, llbc) {
                    list_view[dest] = true;
                    if rvalue_list_view_is_fixed(rvalue, llbc) {
                        list_view_fixed[dest] = true;
                    }
                    changed = true;
                }
            }
        }
    }
    for i in 0..n {
        // A local defined by both a W1-source call and a window-returning
        // call keeps a length shadow so the window arm uses the two-word
        // return ABI. A slice of a STR view is pending_window plus
        // copy-propagated str_view with no W1 call def, and stays a view.
        // A STR view on one path and a list view on another is a window.
        if str_view[i] && list_view[i] {
            window_def[i] = true;
        }
        if (w1_call_def[i] || list_call_def[i]) && pending_window[i] {
            window_def[i] = true;
        }
        if pending_window[i] && !str_view[i] && !list_view[i] {
            window_def[i] = true;
        }
    }
    for i in 0..n {
        if str_view[i] && window_def[i] {
            str_view[i] = false;
        }
        if list_view[i] && window_def[i] {
            list_view[i] = false;
            list_view_fixed[i] = false;
        }
    }
    (str_view, list_view, list_view_fixed)
}

/// Trailing length-slot parameter of a body that returns a pair or a
/// window, including a STR-view local 0 that has no length shadow.
pub(in crate::front::mir) fn ret_len_slot_index(
    locals: &majit_charon_reader::ullbc::Locals,
    llbc: &Llbc,
    len_shadow: &[Option<usize>],
    n_slots: usize,
) -> Option<usize> {
    if len_shadow.first().copied().flatten().is_some() {
        return Some(n_slots);
    }
    locals
        .locals
        .first()
        .is_some_and(|local| tyref_is_u8_window(&local.ty, llbc))
        .then_some(n_slots)
}

/// Graph name of an `RDict` method whose key formal is `&Q`
/// (`rordereddict.py` `ll_dict_get` / `ll_dict_lookup`). Body formals of a
/// descended method stay STR views (`space.bytes_w`).
pub(in crate::front::mir) fn is_rdict_borrowed_key_method(name: &str) -> bool {
    matches!(
        rdict_method_leaf_if_owner(name),
        Some("get" | "get_mut" | "contains_key" | "remove")
    )
}

/// Owner segment of `name` must satisfy `rdict_type_leaf`; otherwise the leaf
/// is ignored. `unspecialized_leaf` sees through a monomorphized copy.
pub(in crate::front::mir) fn rdict_method_leaf_if_owner(name: &str) -> Option<&str> {
    let mut segs = name.rsplit("::");
    let leaf = segs.next()?;
    let leaf = crate::front::clause_spec::unspecialized_leaf(leaf);
    let owner = segs.next().unwrap_or("");
    crate::translator::rtyper::flowspace_adapter::rdict_type_leaf(owner).then_some(leaf)
}

/// Key formals of [`is_rdict_borrowed_key_method`] that are a shared
/// `&[u8]` stay `Ptr(STR)` (`space.bytes_w`). Local 0 is the return; local 1
/// is the receiver.
pub(in crate::front::mir) fn mark_dict_method_str_view_keys(
    graph_name: &str,
    body: &Unstructured,
    llbc: &Llbc,
    str_views: &mut [bool],
) {
    if !is_rdict_borrowed_key_method(graph_name) {
        return;
    }
    let arg_count = body.locals.arg_count as usize;
    for i in 2..=arg_count {
        if i < str_views.len()
            && body
                .locals
                .locals
                .get(i)
                .is_some_and(|local| tyref_is_u8_window(&local.ty, llbc))
        {
            str_views[i] = true;
        }
    }
}

/// `RDict::insert` / `RDict::get` (`rordereddict.py`
/// `ll_dict_setitem_with_hash` / `ll_dict_get` / `ll_dict_lookup`). The key
/// is one `str` word (`space.bytes_w`); never a spliced window.
pub(in crate::front::mir) fn is_rdict_host_rewritten_call(reg: &RegularCall, llbc: &Llbc) -> bool {
    regular_call_name_path(reg, llbc)
        .is_some_and(|path| matches!(rdict_method_leaf_if_owner(&path), Some("insert" | "get")))
}

/// Locals that hold the `{ptr, len}` view of an object-pointer `Vec`
/// buffer: dest of a `Vec` method that yields `&mut [T]`, and dest of
/// `iter_mut` over one of those slices.
pub(in crate::front::mir) fn objectptr_pair_view_locals(
    body: &Unstructured,
    llbc: &Llbc,
) -> Vec<bool> {
    let n = body.locals.locals.len();
    let mut view = vec![false; n];
    // Whole-list `&mut vec` / `&mut vec[..]` is the header word, not a
    // `{ptr, len}` pair. `iter_mut` of that dest still walks the buffer.
    let mut whole_list_mut_dest = vec![false; n];
    for bb in &body.body {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        let dest = dest as usize;
        if dest >= n {
            continue;
        }
        let Some(first_ty) = call.args.first().and_then(operand_tyref) else {
            continue;
        };
        if tyref_rust_vec_item_kind(first_ty, llbc).is_some()
            && slice_ty_item_is_object_pointer(&call.dest.ty, llbc)
            && (ref_kind_is_mut(&call.dest.ty, llbc) || raw_ptr_kind_is_mut(&call.dest.ty, llbc))
        {
            // A Range / RangeFrom / RangeTo subslice stays the aliasing view.
            let whole_list_mut = matches!(&call.func, CallFunc::Regular(reg)
                if object_vec_whole_list_view(reg, call.args.len(), llbc)
                    == Some(ObjectVecWholeListView::Mut));
            if whole_list_mut {
                whole_list_mut_dest[dest] = true;
            } else {
                view[dest] = true;
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for bb in &body.body {
            let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
                continue;
            };
            let PlaceKind::Local(dest) = call.dest.kind else {
                continue;
            };
            let dest = dest as usize;
            if dest >= n || view[dest] {
                continue;
            }
            if !iter_mut_item_is_object_pointer(&call.dest.ty, llbc) {
                continue;
            }
            let recv = match call.args.first() {
                Some(Operand::Copy(place) | Operand::Move(place)) => match &place.kind {
                    PlaceKind::Local(local) => *local as usize,
                    PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                        let PlaceKind::Local(local) = &inner.kind else {
                            continue;
                        };
                        *local as usize
                    }
                    _ => continue,
                },
                _ => continue,
            };
            if view.get(recv).copied().unwrap_or(false)
                || whole_list_mut_dest.get(recv).copied().unwrap_or(false)
            {
                view[dest] = true;
                changed = true;
            }
        }
    }
    view
}

pub(in crate::front::mir) fn pair_len_shadows_of(
    locals: &majit_charon_reader::ullbc::Locals,
    llbc: &Llbc,
    view_locals: &[bool],
    str_view_locals: &[bool],
    list_view_locals: &[bool],
) -> Vec<Option<usize>> {
    let n_locals = locals.locals.len();
    let arg_count = locals.arg_count as usize;
    let mut next = n_locals;
    locals
        .locals
        .iter()
        .enumerate()
        .map(|(index, local)| {
            // A shared `&[u8]` / `*const [u8]` / `Option<&[u8]>` is the
            // two-word window `{data, len}`, independent of
            // `PAIR_SLICE_ITEM_KINDS` (`u8` is not a word item). A W1 STR
            // view (`space.bytes_w`) stays one `Ptr(STR)` word. A W2 list
            // view of a `Vec<u8>` / `Box<[u8]>` owner stays the list.
            if tyref_is_u8_window(&local.ty, llbc) {
                if str_view_locals.get(index).copied().unwrap_or(false)
                    || list_view_locals.get(index).copied().unwrap_or(false)
                {
                    return None;
                }
                next += 1;
                return Some(next - 1);
            }
            let kind = if index <= arg_count {
                tyref_pair_slice_item_kind(&local.ty, llbc)
            } else {
                tyref_pair_field_kind(&local.ty, llbc)
                    .or_else(|| tyref_pair_slice_item_kind(&local.ty, llbc))
            }
            .or_else(|| tyref_pair_slice_iter_item_kind(&local.ty, llbc))
            .or_else(|| tyref_option_pair_slice_item_kind(&local.ty, llbc))?;
            // Parameters keep the one-word helper ABI. Body locals of an
            // object-pointer `&mut [T]` / `IterMut<T>` are the `{ptr, len}`
            // view only when they alias a `Vec` buffer (`pair_of_vec`).
            if tyref_is_object_pointer_mut_slice_or_iter(&local.ty, llbc)
                && (index <= arg_count || !view_locals.get(index).copied().unwrap_or(false))
            {
                return None;
            }
            PAIR_SLICE_ITEM_KINDS.contains(&kind).then(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

/// Extra SSA slot for a pair window's GC owner. Same locals as
/// `pair_len_shadows_of` (a length shadow): a one-word STR/list view is
/// the header itself (`ll_stritem_nonneg`) and needs no owner.
pub(in crate::front::mir) fn pair_owner_shadows_of(
    len_shadow: &[Option<usize>],
    start_slot: usize,
) -> Vec<Option<usize>> {
    let mut next = start_slot;
    len_shadow
        .iter()
        .map(|len| {
            len.is_some().then(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

/// The role of a MIR local in a `[T; N]` array the lowering keeps in a raw
/// item buffer, `lltype.malloc(Array(T), N, flavor='raw')`, because the
/// array is borrowed as a pair slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum RawArrayRole {
    /// The array itself; its Variable is the buffer's item pointer.
    Storage {
        kind: majit_ir::rvec::VecItemKind,
        len: u64,
    },
    /// `&a` or `&*b` of the storage local named here; its Variable is the
    /// same item pointer.
    Borrow(usize),
}

/// A place whose items a subscript reaches through a raw item pointer.
#[derive(Clone, Copy, Debug)]
pub(in crate::front::mir) enum ItemSeq {
    /// `*s`, the slice pair slice local `s` borrows.
    Pair(usize),
    /// Raw-buffer array storage local `a`, spelled `a` or `*b`.
    RawArray(usize),
}

/// The item kind and length of a `[T; N]` whose items are one word of an
/// enabled pair-slice kind.
pub(in crate::front::mir) fn tyref_raw_array_item_kind(
    ty: &TyRef,
    llbc: &Llbc,
) -> Option<(majit_ir::rvec::VecItemKind, u64)> {
    let (item, len) = tyref_raw_array_parts(ty, llbc)?;
    let spelling = charon_type_value_to_ast_string(item, llbc, 0);
    if gcref_item_spelling(&spelling) {
        return None;
    }
    let kind = pair_item_kind_of_node(item, llbc)?;
    PAIR_SLICE_ITEM_KINDS.contains(&kind).then_some((kind, len))
}

/// The item type node and the literal length of a `[T; N]` type,
/// `{"Array": [item, len, trait_ref]}`.
pub(in crate::front::mir) fn tyref_raw_array_parts<'l>(
    ty: &'l TyRef,
    llbc: &'l Llbc,
) -> Option<(&'l serde_json::Value, u64)> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let array = node
        .get("Array")?
        .as_array()
        .filter(|array| array.len() >= 2)?;
    let len = charon_array_len_to_string(&array[1], llbc)
        .parse::<u64>()
        .ok()?;
    Some((&array[0], len))
}

/// Whether the JSON `place` is `a` or `*b` for a raw-array storage local `a`
/// or a borrow `b` of one.
pub(in crate::front::mir) fn json_place_is_raw_array_items(
    place: &serde_json::Value,
    roles: &[Option<RawArrayRole>],
) -> bool {
    let Some(kind) = place.get("kind") else {
        return false;
    };
    let role = |local: &serde_json::Value| {
        local
            .as_u64()
            .and_then(|local| roles.get(local as usize).copied().flatten())
    };
    if let Some(local) = kind.get("Local") {
        return matches!(role(local), Some(RawArrayRole::Storage { .. }));
    }
    let Some([inner, elem]) = kind
        .get("Projection")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
    else {
        return false;
    };
    elem.as_str() == Some("Deref")
        && inner
            .get("kind")
            .and_then(|kind| kind.get("Local"))
            .is_some_and(|local| matches!(role(local), Some(RawArrayRole::Borrow(_))))
}

/// Whether `v` names a local of `wanted` anywhere but as the base of an item
/// subscript `a[i]` / `(*b)[i]` over a raw-buffer array, or in an assertion's
/// panic description.
pub(in crate::front::mir) fn mentions_local_outside_raw_array_items(
    v: &serde_json::Value,
    wanted: &bit_set::BitSet,
    roles: &[Option<RawArrayRole>],
) -> bool {
    match v {
        serde_json::Value::Object(map) => {
            if let Some([base, elem]) = map
                .get("kind")
                .and_then(|kind| kind.get("Projection"))
                .and_then(serde_json::Value::as_array)
                .map(Vec::as_slice)
                && let Some(index) = elem.get("Index")
                && json_place_is_raw_array_items(base, roles)
            {
                return mentions_local_outside_raw_array_items(index, wanted, roles);
            }
            if let Some(local) = map.get("Local").and_then(serde_json::Value::as_u64)
                && wanted.contains(local as usize)
            {
                return true;
            }
            // An assertion's `check_kind` only describes the panic message;
            // the lowering strips the assertion and never reads it.
            map.iter()
                .filter(|(key, _)| key.as_str() != "check_kind")
                .any(|(_, nested)| mentions_local_outside_raw_array_items(nested, wanted, roles))
        }
        serde_json::Value::Array(items) => items
            .iter()
            .any(|nested| mentions_local_outside_raw_array_items(nested, wanted, roles)),
        _ => false,
    }
}

/// Whether operand `op` reads any local in `wanted`, through any projection.
pub(in crate::front::mir) fn operand_mentions_any_local(
    op: &Operand,
    wanted: &bit_set::BitSet,
) -> bool {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return false;
    };
    let mut place = place;
    loop {
        match &place.kind {
            PlaceKind::Local(local) => return wanted.contains(*local as usize),
            PlaceKind::Projection(inner, elem) => {
                if let ProjectionElem::Tagged(tagged) = elem
                    && json_mentions_any_local(tagged, wanted)
                {
                    return true;
                }
                place = inner;
            }
            _ => return false,
        }
    }
}

/// Whether serialized MIR `v` names any local in `wanted`.
pub(in crate::front::mir) fn json_mentions_any_local(
    v: &serde_json::Value,
    wanted: &bit_set::BitSet,
) -> bool {
    match v {
        serde_json::Value::Object(map) => {
            map.get("Local")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|local| wanted.contains(local as usize))
                || map
                    .values()
                    .any(|nested| json_mentions_any_local(nested, wanted))
        }
        serde_json::Value::Array(items) => items
            .iter()
            .any(|nested| json_mentions_any_local(nested, wanted)),
        _ => false,
    }
}

/// Whether `op` is an `Unsize` cast payload.
pub(in crate::front::mir) fn cast_kind_is_unsize(op: &serde_json::Value) -> bool {
    op.get("Cast").and_then(|cast| cast.get("Unsize")).is_some() || op.get("Unsize").is_some()
}

/// The raw-buffer array roles of `body`'s locals.
///
/// A non-parameter `[T; N]` local of an enabled one-word item kind is kept in
/// a raw item buffer when it is borrowed as a pair slice and every other use
/// reaches its items: its only writes are array literals `[x, y]` /
/// `[x; N]` and item stores `a[i] = v`, its borrows `&a` / `&*b` feed only
/// further such borrows and `Unsize` casts into pair slices, and every other
/// mention is an item subscript.  An array read or moved as one value keeps
/// the positional aggregate lowering.
pub(in crate::front::mir) fn raw_array_roles(
    body: &Unstructured,
    llbc: &Llbc,
) -> Vec<Option<RawArrayRole>> {
    let n_locals = body.locals.locals.len();
    let arg_count = body.locals.arg_count as usize;
    let mut roles: Vec<Option<RawArrayRole>> = body
        .locals
        .locals
        .iter()
        .enumerate()
        .map(|(local, decl)| {
            if local <= arg_count {
                return None;
            }
            let (kind, len) = tyref_raw_array_item_kind(&decl.ty, llbc)?;
            Some(RawArrayRole::Storage { kind, len })
        })
        .collect();
    if roles.iter().all(Option::is_none) {
        return roles;
    }
    let forward_blocks = forward_reachable_mask(llbc, body);
    let assigns = || {
        body.body
            .iter()
            .enumerate()
            .filter(|(bb_idx, _)| forward_blocks[*bb_idx])
            .flat_map(|(_, bb)| {
                bb.statements
                    .iter()
                    .filter_map(|stmt| match stmt.stmt_kind_ref() {
                        Ok(StmtKind::Assign(place, rvalue)) => match place.kind {
                            PlaceKind::Local(dest) => Some((dest as usize, place, rvalue)),
                            _ => None,
                        },
                        _ => None,
                    })
            })
    };
    let mut poisoned = bit_set::BitSet::with_capacity(n_locals);
    // Borrows, to a fixpoint over `&*b` chains and moves of a borrow.
    loop {
        let mut changed = false;
        for (dest, _, rvalue) in assigns() {
            let place = match &rvalue {
                Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => place,
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => match src.kind {
                    PlaceKind::Local(b)
                        if matches!(roles[b as usize], Some(RawArrayRole::Borrow(_))) =>
                    {
                        src
                    }
                    _ => continue,
                },
                _ => continue,
            };
            let storage = match &place.kind {
                PlaceKind::Local(b) if matches!(&rvalue, Rvalue::Use(..)) => {
                    match roles[*b as usize] {
                        Some(RawArrayRole::Borrow(a)) => Some(a),
                        _ => None,
                    }
                }
                PlaceKind::Local(a) => match roles[*a as usize] {
                    Some(RawArrayRole::Storage { .. }) => Some(*a as usize),
                    _ => None,
                },
                PlaceKind::Projection(inner, ProjectionElem::Atom(atom)) if atom == "Deref" => {
                    match inner.kind {
                        PlaceKind::Local(b) => match roles[b as usize] {
                            Some(RawArrayRole::Borrow(a)) => Some(a),
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some(storage) = storage else {
                continue;
            };
            match roles[dest] {
                None if dest > arg_count => {
                    roles[dest] = Some(RawArrayRole::Borrow(storage));
                    changed = true;
                }
                Some(RawArrayRole::Borrow(other)) if other == storage => {}
                Some(RawArrayRole::Borrow(other)) => {
                    poisoned.insert(other);
                    poisoned.insert(storage);
                }
                _ => {
                    poisoned.insert(storage);
                }
            }
        }
        if !changed {
            break;
        }
    }
    let storage_of = |roles: &[Option<RawArrayRole>], local: usize| match roles[local] {
        Some(RawArrayRole::Storage { .. }) => Some(local),
        Some(RawArrayRole::Borrow(a)) => Some(a),
        None => None,
    };
    // Every mention has to be one the lowering handles.
    let mut borrowed_as_slice = bit_set::BitSet::with_capacity(n_locals);
    let group_locals = |roles: &[Option<RawArrayRole>], storage: usize| {
        let mut wanted = bit_set::BitSet::with_capacity(n_locals);
        for local in 0..n_locals {
            if storage_of(roles, local) == Some(storage) {
                wanted.insert(local);
            }
        }
        wanted
    };
    let storages: Vec<usize> = (0..n_locals)
        .filter(|&local| matches!(roles[local], Some(RawArrayRole::Storage { .. })))
        .collect();
    for &storage in &storages {
        if poisoned.contains(storage) {
            continue;
        }
        let Some(RawArrayRole::Storage { kind, .. }) = roles[storage] else {
            continue;
        };
        let wanted = group_locals(&roles, storage);
        let mut ok = true;
        'blocks: for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let rvalue_json = stmt
                    .kind_value()
                    .get("Assign")
                    .and_then(|assign| assign.get(1));
                let allowed = match stmt.stmt_kind_ref() {
                    Ok(
                        StmtKind::StorageLive(_)
                        | StmtKind::StorageDead(_)
                        | StmtKind::PlaceMention(_)
                        | StmtKind::Borrowck(_),
                    ) => true,
                    Ok(StmtKind::Assign(place, rvalue)) => match (&place.kind, rvalue) {
                        (PlaceKind::Local(dest), Rvalue::Aggregate(agg, _))
                            if *dest as usize == storage && aggregate_ctor_name(agg) == "Array" =>
                        {
                            !rvalue_json.is_some_and(|rv| {
                                mentions_local_outside_raw_array_items(rv, &wanted, &roles)
                            })
                        }
                        (PlaceKind::Local(dest), Rvalue::Repeat(..))
                            if *dest as usize == storage =>
                        {
                            !rvalue_json.is_some_and(|rv| {
                                mentions_local_outside_raw_array_items(rv, &wanted, &roles)
                            })
                        }
                        (PlaceKind::Local(dest), Rvalue::Ref { .. } | Rvalue::RawPtr { .. })
                            if roles[*dest as usize] == Some(RawArrayRole::Borrow(storage)) =>
                        {
                            true
                        }
                        (
                            PlaceKind::Local(dest),
                            Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _),
                        ) if roles[*dest as usize] == Some(RawArrayRole::Borrow(storage))
                            && matches!(src.kind, PlaceKind::Local(b) if roles[b as usize] == Some(RawArrayRole::Borrow(storage))) =>
                        {
                            true
                        }
                        (
                            PlaceKind::Local(dest),
                            Rvalue::UnaryOp(op, Operand::Copy(src) | Operand::Move(src)),
                        ) if cast_kind_is_unsize(op)
                            && matches!(src.kind, PlaceKind::Local(b) if roles[b as usize] == Some(RawArrayRole::Borrow(storage)))
                            && tyref_pair_slice_item_kind(
                                &body.locals.locals[*dest as usize].ty,
                                llbc,
                            ) == Some(kind) =>
                        {
                            borrowed_as_slice.insert(storage);
                            true
                        }
                        (
                            PlaceKind::Local(dest),
                            Rvalue::Cast(op, Operand::Copy(src) | Operand::Move(src), _),
                        ) if (cast_kind_is_unsize(op) || op.get("Unsize").is_some())
                            && matches!(src.kind, PlaceKind::Local(b) if roles[b as usize] == Some(RawArrayRole::Borrow(storage)))
                            && tyref_pair_slice_item_kind(
                                &body.locals.locals[*dest as usize].ty,
                                llbc,
                            ) == Some(kind) =>
                        {
                            borrowed_as_slice.insert(storage);
                            true
                        }
                        _ => !mentions_local_outside_raw_array_items(
                            stmt.kind_value(),
                            &wanted,
                            &roles,
                        ),
                    },
                    _ => {
                        !mentions_local_outside_raw_array_items(stmt.kind_value(), &wanted, &roles)
                    }
                };
                if !allowed {
                    ok = false;
                    break 'blocks;
                }
            }
            // `<[T; N]>::index(&a, range)` is a subslice of the buffer.
            if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
                && let CallFunc::Regular(reg) = &call.func
                && regular_call_name_path(reg, llbc).is_some_and(|path| {
                    matches!(
                        path.as_str(),
                        "core::array::<Impl>::index" | "core::array::<Impl>::index_mut"
                    )
                })
                && let [Operand::Copy(src) | Operand::Move(src), range] = call.args.as_slice()
                && matches!(src.kind, PlaceKind::Local(b) if roles[b as usize] == Some(RawArrayRole::Borrow(storage)))
                && !matches!(call.dest.kind, PlaceKind::Local(d) if wanted.contains(d as usize))
                && tyref_pair_slice_item_kind(&call.dest.ty, llbc) == Some(kind)
                && !operand_mentions_any_local(range, &wanted)
            {
                borrowed_as_slice.insert(storage);
                continue;
            }
            if mentions_local_outside_raw_array_items(bb.terminator.kind_value(), &wanted, &roles) {
                ok = false;
                break;
            }
        }
        if !ok {
            poisoned.insert(storage);
        }
    }
    // `shadowstack.py push_roots` emits `gc_push_roots` on the livevars
    // themselves. A `[T; N]` that exists only to feed `pin_roots` /
    // `publish_roots` is that livevar list, not a data array: leave it
    // off the entry-malloc buffer so the pin pass writes each GCREF.
    let mut push_roots_only = bit_set::BitSet::with_capacity(n_locals);
    for &storage in &storages {
        if poisoned.contains(storage) || !borrowed_as_slice.contains(storage) {
            continue;
        }
        if raw_array_slice_is_push_roots_only(body, llbc, &roles, storage) {
            push_roots_only.insert(storage);
        }
    }
    for local in 0..n_locals {
        if let Some(storage) = storage_of(&roles, local)
            && (poisoned.contains(storage)
                || !borrowed_as_slice.contains(storage)
                || push_roots_only.contains(storage))
        {
            roles[local] = None;
        }
    }
    roles
}

/// Whether every slice of `storage` is the argument of `pin_roots` /
/// `publish_roots` (`gc_roots`). `install_range_state(&[i64; N])` and
/// other data slices stay on the raw-buffer path.
pub(in crate::front::mir) fn raw_array_slice_is_push_roots_only(
    body: &Unstructured,
    llbc: &Llbc,
    roles: &[Option<RawArrayRole>],
    storage: usize,
) -> bool {
    let mut slices = bit_set::BitSet::with_capacity(body.locals.locals.len());
    for (dest, _, rvalue) in body.body.iter().flat_map(|bb| {
        bb.statements
            .iter()
            .filter_map(|stmt| match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => match place.kind {
                    PlaceKind::Local(d) => Some((d as usize, place, rvalue)),
                    _ => None,
                },
                _ => None,
            })
    }) {
        match &rvalue {
            Rvalue::UnaryOp(op, Operand::Copy(src) | Operand::Move(src))
            | Rvalue::Cast(op, Operand::Copy(src) | Operand::Move(src), _)
                if (cast_kind_is_unsize(op) || op.get("Unsize").is_some())
                    && matches!(
                        src.kind,
                        PlaceKind::Local(b)
                            if roles[b as usize] == Some(RawArrayRole::Borrow(storage))
                    ) =>
            {
                slices.insert(dest);
            }
            _ => {}
        }
    }
    if slices.is_empty() {
        return false;
    }
    loop {
        let mut changed = false;
        for bb in &body.body {
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Use(op, _))) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(dest) = place.kind else {
                    continue;
                };
                if slices.contains(dest as usize) {
                    continue;
                }
                if operand_mentions_any_local(&op, &slices) {
                    slices.insert(dest as usize);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut seen_push = false;
    for bb in &body.body {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            if mentions_local_outside_raw_array_items(bb.terminator.kind_value(), &slices, roles) {
                return false;
            }
            continue;
        };
        let mentions_slice = call
            .args
            .iter()
            .any(|op| operand_mentions_any_local(op, &slices));
        if !mentions_slice {
            continue;
        }
        let CallFunc::Regular(reg) = &call.func else {
            return false;
        };
        let Some(path) = regular_call_name_path(reg, llbc) else {
            return false;
        };
        let leaf = path.rsplit("::").next();
        if matches!(leaf, Some("pin_roots") | Some("publish_roots"))
            && path.split("::").any(|s| s == ROOT_SCOPE_MODULE)
        {
            seen_push = true;
            continue;
        }
        return false;
    }
    seen_push
}

/// The method name of an `alloc::vec::Vec` inherent-method callee path:
/// `alloc::vec::<Impl>::<leaf>`, or the short `vec::Vec::<leaf>` spelling
/// `call_target_segments` gives an impl method it resolved by its owner type.
pub(in crate::front::mir) fn rust_vec_std_leaf(segments: &[String]) -> Option<&str> {
    match segments {
        [a, b, c, leaf] if a == "alloc" && b == "vec" && (c == "<Impl>" || c == "Vec") => {
            Some(leaf)
        }
        [a, b, leaf] if a == "vec" && b == "Vec" => Some(leaf),
        _ => None,
    }
}

/// `alloc::vec::from_elem`, the call `vec![item; count]` lowers to.
pub(in crate::front::mir) fn is_alloc_vec_from_elem(segments: &[String]) -> bool {
    matches!(segments, [a, b, c] if a == "alloc" && b == "vec" && c == "from_elem")
}

/// The `Vec` operation and item kind of a `Vec::new()` /
/// `Vec::with_capacity(n)` / `v.push(x)` / `v.as_ptr()` / `v.as_mut_ptr()`
/// call on a `Vec` of one-word items: the constructors read the item from
/// the result type, `push` / `as_ptr` / `as_mut_ptr` from the receiver.
/// `as_ptr` / `as_mut_ptr` are `ll_items` (`rlist.py`: `return l.items`).
/// `Vec::set_len` is a `FieldWrite` of the header `len` word (`l.length = n`),
/// not a `VecOp` helper.
pub(in crate::front::mir) fn rust_vec_std_call(
    segments: &[String],
    nargs: usize,
    dest_ty: &TyRef,
    first_arg_ty: Option<&TyRef>,
    llbc: &Llbc,
) -> Option<(majit_ir::rvec::VecOp, majit_ir::rvec::VecItemKind)> {
    use majit_ir::rvec::VecOp;
    if nargs == 2 && is_alloc_vec_from_elem(segments) {
        return Some((VecOp::AllocAndSet, tyref_rust_vec_item_kind(dest_ty, llbc)?));
    }
    match (rust_vec_std_leaf(segments)?, nargs) {
        ("new", 0) => Some((VecOp::NewEmpty, tyref_rust_vec_item_kind(dest_ty, llbc)?)),
        ("with_capacity", 1) => Some((VecOp::NewHint, tyref_rust_vec_item_kind(dest_ty, llbc)?)),
        ("push", 2) => Some((
            VecOp::Append,
            tyref_rust_vec_item_kind(first_arg_ty?, llbc)?,
        )),
        ("as_ptr" | "as_mut_ptr", 1) => {
            Some((VecOp::Items, tyref_rust_vec_item_kind(first_arg_ty?, llbc)?))
        }
        _ => None,
    }
}

/// `core::slice::<Impl>::reverse`.
pub(in crate::front::mir) fn is_core_slice_reverse(segments: &[String]) -> bool {
    matches!(segments, [a, b, c, d] if a == "core" && b == "slice" && c == "<Impl>" && d == "reverse")
}

/// Item kind of the `Vec` of one-word items whose `deref` / `deref_mut`
/// call defines MIR `local`.
pub(in crate::front::mir) fn local_is_rust_vec_deref(
    body: &Unstructured,
    llbc: &Llbc,
    local: usize,
) -> Option<majit_ir::rvec::VecItemKind> {
    let forward_blocks = forward_reachable_mask(llbc, body);
    body.body.iter().enumerate().find_map(|(bb_idx, bb)| {
        if !forward_blocks[bb_idx] {
            return None;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            return None;
        };
        if !matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == local) {
            return None;
        }
        let CallFunc::Regular(reg) = &call.func else {
            return None;
        };
        if !is_deref_call(reg, llbc) {
            return None;
        }
        tyref_rust_vec_item_kind(operand_tyref(call.args.first()?)?, llbc)
    })
}

/// Whether every use of MIR `local`, the slice a `Vec` deref defines, is
/// the receiver of a `<[T]>::reverse` call.
pub(in crate::front::mir) fn rust_vec_deref_feeds_only_reverse(
    body: &Unstructured,
    llbc: &Llbc,
    local: usize,
) -> bool {
    let mut watched = bit_set::BitSet::new();
    watched.insert(local);
    let mut reverses = 0;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        if bb.statements.iter().any(|stmt| {
            !matches!(
                stmt.stmt_kind_ref(),
                Ok(StmtKind::StorageLive(_)
                    | StmtKind::StorageDead(_)
                    | StmtKind::PlaceMention(_)
                    | StmtKind::Borrowck(_))
            ) && mentions_local(stmt.kind_value(), &watched)
        }) {
            return false;
        }
        if !mentions_local(bb.terminator.kind_value(), &watched) {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            return false;
        };
        let CallFunc::Regular(reg) = &call.func else {
            return false;
        };
        if matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == local) {
            if is_deref_call(reg, llbc) {
                continue;
            }
            return false;
        }
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = llbc.fn_by_id(*id) else {
            return false;
        };
        let segments: Vec<String> = fd
            .item_meta
            .name_path()
            .split("::")
            .map(str::to_string)
            .collect();
        if !is_core_slice_reverse(&segments)
            || call.args.len() != 1
            || operand_local(call.args.first()) != Some(local)
        {
            return false;
        }
        reverses += 1;
    }
    reverses > 0
}

/// The `ll_vec_*` helper path for `op` on a `Vec` of `kind` items, as call
/// segments.
pub(in crate::front::mir) fn rust_vec_helper_segments(
    op: majit_ir::rvec::VecOp,
    kind: majit_ir::rvec::VecItemKind,
) -> Vec<String> {
    majit_ir::rvec::vec_helper_path(op, kind)
        .split("::")
        .map(str::to_string)
        .collect()
}

/// Kind-independent `ll_items`: the header's buffer pointer word, for a
/// `Vec<T>` whose item is not one word (`ll_vec_as_ptr`). The call site
/// converts the Vec with `cast_ptr_to_int` then `r_uint` before this
/// helper (`pair_rustvec_integer_convert_from_to`). Byte items wrap the
/// integer result with `cast_int_to_ptr(CCHARP)` (`rlist.py` `ll_items`,
/// `rgc.py` `ll_nonmovable_raw_ptr_for_resizable_list`).
pub(in crate::front::mir) fn rust_vec_as_ptr_helper_segments() -> Vec<String> {
    majit_ir::rvec::RVEC_MODULE
        .split("::")
        .chain(std::iter::once("ll_vec_as_ptr"))
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests;
