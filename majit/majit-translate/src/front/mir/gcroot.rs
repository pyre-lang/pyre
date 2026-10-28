//! Root scopes, root brackets, and drops (`rpython/memory/gctransform/shadowstack.py`).

use super::*;

impl<'a> Lowering<'a> {
    /// Close a root bracket at the guard's `Drop`, by calling the same rewind
    /// the destructor body calls. A non-guard drop is a no-op.
    ///
    /// The call takes the guard by reference rather than the two words the
    /// rewind needs, so this arm reads no field of it. A crate that only
    /// imports the guard sees an opaque stub with no fields, and a
    /// field-keyed close would silently confine itself to the crate that
    /// defines one.
    pub(in crate::front::mir) fn emit_root_scope_close(&mut self, mir_bb: usize, place: &Place) {
        let Some(class_root) = self.tyref_adt_class_root(&place.ty) else {
            return;
        };
        if !gc_root_scope_type_path(&class_root) {
            return;
        }
        // Charon stamps a drop `Conditional` when the destructor runs only if
        // the place still holds a value.  The flag that decides it is not in
        // the artefact, so a guard the body moves out of keeps its bracket
        // open: closing one the moved-to owner still holds would rewind a
        // shadow stack that owner is using.
        if let PlaceKind::Local(local) = place.kind
            && self.root_scope_moved_locals.contains(local as usize)
        {
            self.decline_root_scope_close("guard-moved");
            return;
        }
        // An erased bracket published nothing, so there is nothing to rewind.
        if let PlaceKind::Local(local) = place.kind
            && self.root_bracket.is_erased_scope(local as usize)
        {
            return;
        }
        // A drop can still name a local no block lowered so far has assigned.
        // Leaving that bracket open is what every drop did before this arm
        // existed, so the graph stays lowerable and the gap is counted rather
        // than turned into a refused function.
        let Ok(base) = self.resolve_place(mir_bb, clone_place(place)) else {
            self.decline_root_scope_close("guard-place-unresolved");
            return;
        };
        // The helper is a sibling of the guard's own type, so its path is the
        // guard's with the leaf replaced. Nothing here names an owning crate.
        let mut segments: Vec<String> = class_root.split("::").map(str::to_string).collect();
        segments.pop();
        segments.push(ROOT_SCOPE_CLOSE.to_string());
        let bb_id = self.emit_bb(mir_bb);
        let void = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(void),
            kind: OpKind::Call {
                target: CallTarget::function_path(segments),
                args: crate::model::call_args(vec![base]),
                result_ty: ValueType::Void,
            },
        });
    }

    /// Count a bracket this pass left open, against the graph that holds it.
    /// The rewind is what the guard's destructor does, so a skipped one is a
    /// retention the census has to be able to name.
    pub(in crate::front::mir) fn decline_root_scope_close(&self, reason: &'static str) {
        crate::decline::record_named("root_bracket_close", reason, &self.graph.name);
    }

    /// Emit the destructor of `place` into the current block, the same
    /// operations a [`TermKind::Drop`] of it emits.
    ///
    /// `glue` is Charon's drop-glue `fn_ptr`. `None` is an explicit
    /// `core::mem::drop` and decides the root-scope case from the place
    /// type. `Ok(true)` means this front end accounted for the destructor
    /// (including a type with no drop glue, which emits nothing). `Ok(false)`
    /// means the caller must keep the residual call.
    pub(in crate::front::mir) fn emit_place_drop(
        &mut self,
        mir_bb: usize,
        place: &Place,
        glue: Option<&RegularCall>,
    ) -> Result<bool, LowerError> {
        if drop_place_is_frame_anchor(place, self.body, self.llbc) {
            self.emit_one_word_guard_release(
                mir_bb,
                place,
                ["pyre_interpreter", "eval", "frame_anchor_release"],
            )?;
            return Ok(true);
        }
        if drop_place_is_list_guard(place, self.llbc) {
            self.emit_one_word_guard_release(mir_bb, place, LIST_LOCK_RELEASE_PATH)?;
            return Ok(true);
        }
        if drop_place_is_set_guard(place, self.llbc) {
            self.emit_one_word_guard_release(mir_bb, place, SET_LOCK_RELEASE_PATH)?;
            return Ok(true);
        }
        let root_scope = match glue {
            Some(fn_ptr) => drop_lowers_as_glue_call(place, fn_ptr, self.llbc),
            None => self
                .tyref_adt_class_root(&place.ty)
                .is_some_and(|root| gc_root_scope_type_path(&root)),
        };
        if root_scope {
            self.emit_root_scope_close(mir_bb, place);
            return Ok(true);
        }
        // The guard's slot does not exist in a jitcode. Dropping it, or the
        // `Option` that holds it, releases nothing.
        if owner_root_guard::tyref_is_erased_drop(&place.ty, self.llbc) {
            return Ok(true);
        }
        if tyref_is_owned_vec(&place.ty, self.llbc) {
            if let PlaceKind::Local(local) = place.kind {
                match vec_init_at_drop(self.body, self.llbc, mir_bb, local as usize) {
                    VecInit::Maybe => {
                        crate::decline::record_named(
                            crate::decline::gate::RUST_VEC_DROP,
                            "maybe-initialised",
                            &self.graph.name,
                        );
                        return Ok(true);
                    }
                    VecInit::Uninit => return Ok(true),
                    VecInit::Init => {
                        return Ok(self.lower_rust_vec_drop(self.emit_bb(mir_bb), place));
                    }
                }
            }
            return Ok(self.lower_rust_vec_drop(self.emit_bb(mir_bb), place));
        }
        if tyref_rust_vec_item_kind(&place.ty, self.llbc).is_some() {
            return Ok(self.lower_rust_vec_drop(self.emit_bb(mir_bb), place));
        }
        if glue.is_none() && self.tyref_is_trivially_dropless(&place.ty, 0) {
            return Ok(true);
        }
        Ok(false)
    }

    /// Drop of a `Vec` of one-word items: `ll_vec_free` of the header and
    /// its buffer, as `rewrite_op_free` frees a `flavor='raw'` block.  The
    /// items carry no drop glue.  The item buffer is raw memory for every
    /// item kind, managed references included: the interpreter keeps `Vec`
    /// contents untraced and roots them where it allocates, and the lowered
    /// code keeps that contract, so freeing it releases no GC object.
    ///
    /// `rewrite_op_free` (`jtransform.py`) emits `raw_free` wherever the
    /// graph frees; the pointer is threaded through Links like any other
    /// variable. Ownership is a header this graph allocated through
    /// `ll_vec_newemptylist` / `ll_vec_newlist_hint` / `ll_vec_alloc_and_set`,
    /// following SSA copies through block-input Links and phi merges. A
    /// call result or function input is not this graph's allocation.
    /// Returns false so the caller keeps the drop glue. `true` means
    /// `ll_vec_free` was queued.
    pub(in crate::front::mir) fn lower_rust_vec_drop(
        &mut self,
        bb_id: BlockId,
        place: &Place,
    ) -> bool {
        let PlaceKind::Local(local) = place.kind else {
            return false;
        };
        let Some(kind) = tyref_rust_vec_item_kind(&place.ty, self.llbc) else {
            if tyref_is_vec_value(&place.ty, self.llbc) {
                crate::decline::record_named(
                    crate::decline::gate::RUST_VEC_DROP,
                    "vec-drop-unknown-item-kind",
                    &self.graph.name,
                );
            }
            return false;
        };
        let Some(header) = self.local_var[local as usize]
            .as_ref()
            .and_then(|value| value.one().ok())
        else {
            crate::decline::record_named(
                crate::decline::gate::RUST_VEC_DROP,
                "header-not-live-at-drop",
                &self.graph.name,
            );
            return false;
        };
        // Prove ownership after links match inputargs (framestate Pass 2).
        self.pending_vec_frees.push((bb_id, header, kind));
        true
    }

    pub(in crate::front::mir) fn emit_pending_vec_frees(&mut self) {
        let pending = std::mem::take(&mut self.pending_vec_frees);
        for (bb_id, header, kind) in pending {
            emit_rust_vec_drop_if_allocated(&mut self.graph, bb_id, &header, kind);
        }
    }

    /// Drop a one-word guard local by passing its word to the bound release
    /// residual `release`. The caller continues to the drop's successor.
    ///
    /// A [`FrameAnchor`] is one such guard. `front::mir` aliases that local
    /// to its depth `Int`. `FrameAnchor::drop` takes `&mut self` (`Ref`), so
    /// emitting that path as the residual made the containing graph fail to
    /// look-inside (`opcode_compare_op` residualized as unbound `ri`).
    /// `frame_anchor_release` is already `dont_look_inside` and takes the
    /// depth as `usize`.
    pub(in crate::front::mir) fn emit_one_word_guard_release(
        &mut self,
        mir_bb: usize,
        place: &Place,
        release: [&str; 3],
    ) -> Result<(), LowerError> {
        let PlaceKind::Local(local) = place.kind else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: guard Drop over a projection place ({release:?})"
            )));
        };
        let bb_id = self.emit_bb(mir_bb);
        if let Some(arg) = self.local_var[local as usize]
            .as_ref()
            .and_then(|value| value.one().ok())
        {
            // The residual ABI is the depth word (`frame_anchor_release`
            // / lock-release). `FUNC.ARGS` is Int; stamp the cell so a
            // later Skip-arm `s_None` projection cannot Void it before
            // `getcalldescr` (`call.py`).
            crate::model::FunctionGraph::set_concretetype_of_inline(
                &arg,
                crate::model::ConcreteType::Signed,
            );
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::function_path(release),
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Void,
                },
            });
        }
        Ok(())
    }

    /// Lower a local `Drop` as the glue call named by its MIR terminator.
    pub(in crate::front::mir) fn lower_drop_as_glue_call(
        &mut self,
        mir_bb: usize,
        place: Place,
        fn_ptr: RegularCall,
        target: usize,
        on_unwind: usize,
    ) -> Result<(), LowerError> {
        let PlaceKind::Local(dest_local) = &place.kind else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Drop-as-call over a projection place"
            )));
        };
        let dest_local = *dest_local;
        let call = CallPayload {
            func: CallFunc::Regular(fn_ptr),
            args: vec![Operand::Move(place)],
            dest: Place {
                kind: PlaceKind::Local(dest_local),
                ty: unit_tyref(),
            },
        };
        self.lower_call(mir_bb, call, target, on_unwind)
    }

    /// Lower one call of a bracket [`RootBracketPlan`] erased, and report
    /// whether it did.
    ///
    /// `push_roots` binds nothing: the guard has no reader left.  `base()`
    /// binds nothing either -- its result only ever indexed the read-backs
    /// below, and so does `pin_roots`, whose result names the first slot of
    /// that run; the run it publishes is answered element by element.
    /// `shadow_stack_len` is the same depth read when the body names slots as
    /// `len + k` instead of `base + k`.  `pin_root` is the identity on the
    /// value it publishes, the same statement `try_gc_current_object_address`
    /// makes about the read half: the translated graph already carries that
    /// reference in a slot the backend root map rewrites when the object
    /// moves.  `get(base + k)` / `get(len + k)` answers with the value the
    /// `k`-th pin published, for the same reason. A [`RootSlotGetter`]
    /// closure is that read with the callee as the closure `call`:
    /// `get_sites` names the pin, and the capture temps bind no Variable.
    pub(in crate::front::mir) fn lower_erased_root_bracket_call(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if !self.root_bracket.enabled {
            return Ok(false);
        }
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let Some(path) = regular_call_name_path(reg, self.llbc) else {
            return Ok(false);
        };
        if !path.split("::").any(|s| s == "gc_roots") {
            // A RootSlotGetter closure: `analyze_root_brackets` records the
            // call in `get_sites` and the capture temps bind no Variable.
            // The callee is the closure `call`, so the gc_roots leaf match
            // below would miss it and the skipped receiver would read as an
            // uninitialised local.
            let Some(value) = self.root_bracket.get_answer(mir_bb) else {
                return Ok(false);
            };
            let bound = self.local_var[value]
                .as_ref()
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased root read answers with unbound MIR local {value}"
                    ))
                })?
                .one()?;
            self.local_var[dest_local] = Some(LocalValue::One(bound));
            let bb_id = self.emit_bb(mir_bb);
            let target_bb = self.block_id[target];
            let link_args = self.edge_args(mir_bb, target)?;
            self.graph.set_goto(bb_id, target_bb, link_args);
            return Ok(true);
        }
        let leaf = path.rsplit("::").next().unwrap_or("");
        let receiver_scope = operand_local(call.args.first())
            .and_then(|local| self.root_bracket.scope_of_receiver(local))
            .or_else(|| self.root_bracket.free_site_scope(mir_bb));
        let bound = match leaf {
            _ if gc_root_scope_open_path(&path) => {
                if !self.root_bracket.is_erased_scope(dest_local) {
                    return Ok(false);
                }
                None
            }
            "base" | "pin_roots" | "shadow_stack_len" if receiver_scope.is_some() => None,
            "shadow_stack_copy_range" | "shadow_stack_copy_range_into_vec"
                if receiver_scope.is_some() =>
            {
                let Some((slice, k)) = self.root_bracket.slice_copy_answer(mir_bb) else {
                    return Ok(false);
                };
                let src = self.local_var[slice]
                    .as_ref()
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: erased copy_range reads unbound slice local {slice}"
                        ))
                    })?
                    .one()?;
                let dest_op = call.args.get(1).cloned().ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased copy_range without a destination operand"
                    ))
                })?;
                let dest = self.resolve_operand(mir_bb, dest_op)?;
                let start = {
                    let bb_id = self.block_id[mir_bb];
                    let var = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(var.clone()),
                        kind: OpKind::ConstInt(k as i64),
                    });
                    var
                };
                self.emit_path_call(
                    mir_bb,
                    "pyre_object::gc_roots::copy_object_slice_range_into_vec",
                    vec![src, start, dest],
                    ValueType::Void,
                );
                None
            }
            "pin_root" if receiver_scope.is_some() => {
                // Free `gc_roots::pin_root(value)` has one argument. The
                // method form is `scope.pin_root(value)`.
                let value_op = match call.args.len() {
                    1 => call.args.first(),
                    2 => call.args.get(1),
                    _ => None,
                };
                let value = operand_local(value_op).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased pin_root without a plain-local value operand"
                    ))
                })?;
                Some(
                    self.local_var[value]
                        .as_ref()
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: erased pin_root reads unbound MIR local {value}"
                            ))
                        })?
                        .one()?,
                )
            }
            "get" | "shadow_stack_get" => {
                let Some(scope) = receiver_scope else {
                    return Ok(false);
                };
                if let Some((slice, k)) = self.root_bracket.slice_get_answer(mir_bb) {
                    let src = self.local_var[slice]
                        .as_ref()
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: erased slice get reads unbound local {slice}"
                            ))
                        })?
                        .one()?;
                    let bb_id = self.block_id[mir_bb];
                    let index = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(index.clone()),
                        kind: OpKind::ConstInt(k as i64),
                    });
                    let loaded = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(loaded.clone()),
                        kind: OpKind::ArrayRead {
                            base: src,
                            index,
                            item_ty: ValueType::Ref(None),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            pure: false,
                        },
                    });
                    Some(loaded)
                } else {
                    let value = self.root_bracket.get_answer(mir_bb).ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: erased root read has no pinned value for guard {scope}"
                        ))
                    })?;
                    Some(
                        self.local_var[value]
                            .as_ref()
                            .ok_or_else(|| {
                                LowerError::Unsupported(format!(
                                    "bb{mir_bb}: erased root read answers with unbound MIR local {value}"
                                ))
                            })?
                            .one()?,
                    )
                }
            }
            _ => return Ok(false),
        };
        self.local_var[dest_local] = bound.map(LocalValue::One);
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// The handle operations of an [`OwnerRootPlan`] body: `new(v)` binds the
    /// handle to `v` and `deref` answers the handle's value. A producer call
    /// stays an ordinary call whose graph returns the value, and the handle's
    /// `Drop` already lowers to plain control flow.
    pub(in crate::front::mir) fn lower_erased_owner_root_call(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if self.owner_root.roots.is_empty() {
            return Ok(false);
        }
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let source = match owner_root_callee(reg, self.llbc) {
            OwnerRootCallee::New if self.owner_root.erases(dest_local) => {
                operand_local(call.args.first())
            }
            OwnerRootCallee::Deref => operand_local(call.args.first())
                .and_then(|borrow| self.owner_root.handle_of_borrow(borrow)),
            _ => return Ok(false),
        };
        let Some(source) = source else {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: erased owner-root call without a plain-local operand"
            )));
        };
        let value = self.local_var[source]
            .as_ref()
            .ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: erased owner-root call reads unbound MIR local {source}"
                ))
            })?
            .one()?;
        self.local_var[dest_local] = Some(LocalValue::One(value));
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// [`owner_root_guard`]: `new` and `get` alias the guard's word, `set`
    /// writes that word back, and `Drop` / `Option::take` do not call.
    pub(in crate::front::mir) fn lower_owner_root_guard_call(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let Some(op) = owner_root_guard::classify(call, self.llbc) else {
            return Ok(false);
        };
        let bb_id = self.emit_bb(mir_bb);
        match op {
            owner_root_guard::Op::New | owner_root_guard::Op::Get => {
                let source = operand_local(call.args.first()).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased OwnerRootGuard {op:?} without a plain-local operand"
                    ))
                })?;
                let value = self.local_var[source]
                    .as_ref()
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: erased OwnerRootGuard {op:?} reads unbound MIR local {source}"
                        ))
                    })?
                    .one()?;
                self.local_var[dest_local] = Some(LocalValue::One(value));
            }
            owner_root_guard::Op::Set => {
                let borrow = operand_local(call.args.first()).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased OwnerRootGuard::set without a plain-local receiver"
                    ))
                })?;
                let slot = self.mem_slot(Some(borrow)).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased OwnerRootGuard::set has no place for MIR local {borrow}"
                    ))
                })?;
                let MemSlot::Place(place) = slot else {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased OwnerRootGuard::set through an index element"
                    )));
                };
                let place = owner_root_guard::guard_sink_place(&place, self.llbc);
                let value_local = operand_local(call.args.get(1)).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased OwnerRootGuard::set without a plain-local value"
                    ))
                })?;
                let value = self.local_var[value_local]
                    .as_ref()
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: erased OwnerRootGuard::set reads unbound MIR local {value_local}"
                        ))
                    })?
                    .one()?;
                self.write_mem_slot(mir_bb, MemSlot::Place(place), value)?;
                let unit = self.emit_unit(bb_id);
                self.local_var[dest_local] = Some(LocalValue::One(unit));
            }
            owner_root_guard::Op::Drop => {
                let unit = self.emit_unit(bb_id);
                self.local_var[dest_local] = Some(LocalValue::One(unit));
            }
            owner_root_guard::Op::OptionTake => {
                let borrow = operand_local(call.args.first()).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased Option<OwnerRootGuard>::take without a plain-local receiver"
                    ))
                })?;
                let slot = self.mem_slot(Some(borrow)).ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased Option<OwnerRootGuard>::take has no place for MIR local {borrow}"
                    ))
                })?;
                let MemSlot::Place(place) = slot else {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased Option<OwnerRootGuard>::take through an index element"
                    )));
                };
                let place = owner_root_guard::guard_sink_place(&place, self.llbc);
                if !owner_root_guard::tyref_is_option_of_guard(&place.ty, self.llbc) {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: erased Option<OwnerRootGuard>::take place is not the option"
                    )));
                }
                let option_ty = clone_tyref(&place.ty);
                let old = self.read_mem_slot(mir_bb, &MemSlot::Place(clone_place(&place)))?;
                let null = self.push_niche_null_ptr(mir_bb, &option_ty);
                self.write_mem_slot(mir_bb, MemSlot::Place(place), null)?;
                self.local_var[dest_local] = Some(LocalValue::One(old));
            }
        }
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// A residual callee runs the real Rust layout. A pointer to the guard,
    /// or to an aggregate that contains one, would show it the erased word.
    pub(in crate::front::mir) fn refuse_residual_owner_root_guard(
        &self,
        mir_bb: usize,
        call: &CallPayload,
    ) -> Result<(), LowerError> {
        let Some(callee) =
            owner_root_guard::residual_escape(call, self.llbc, self.dont_look_inside)
        else {
            return Ok(());
        };
        crate::decline::record_named(
            crate::decline::gate::OWNER_ROOT_GUARD_ESCAPE,
            "pointer-to-residual",
            &self.graph.name,
        );
        Err(LowerError::Unsupported(format!(
            "bb{mir_bb}: `{}` passes a pointer to OwnerRootGuard into residual `{callee}`",
            self.graph.name
        )))
    }

    /// The `RBigIntGcRoot::new(value)` call that makes a producer's value the
    /// handle a kept body expects.
    ///
    /// A producer whose body erases its handle returns the rbigint that
    /// handle roots ([`OwnerRootPlan`]). A body whose handles stay real reads
    /// the call's destination as a handle, so it gets `h = new(producer(..))`,
    /// the same value the unerased producer body returned. The real `new`
    /// still enters `acquire_owner_root`, so the kept body keeps its decline.
    /// A producer whose own handle stays real already returns the handle.
    pub(in crate::front::mir) fn owner_root_reroot_call(
        &self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
    ) -> Result<Option<CallPayload>, LowerError> {
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(None);
        };
        if self.owner_root.erases(dest_local)
            || owner_root_callee(reg, self.llbc) != OwnerRootCallee::Producer
        {
            return Ok(None);
        }
        match owner_root_producer_returns_value(reg, self.llbc) {
            Some(true) => {}
            Some(false) => return Ok(None),
            None => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: kept owner-root handle takes the result of a producer \
                     whose body is not in this crate"
                )));
            }
        }
        let new = owner_root_new_decl(self.llbc).ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: kept owner-root handle takes a producer value but \
                 `RBigIntGcRoot::new` is not declared"
            ))
        })?;
        let value_ty = new.signature.inputs.first().ok_or_else(|| {
            LowerError::Unsupported(format!(
                "bb{mir_bb}: `RBigIntGcRoot::new` declares no value parameter"
            ))
        })?;
        Ok(Some(CallPayload {
            func: CallFunc::Regular(RegularCall {
                kind: CallKind::Fun(FunId::Regular { id: new.def_id }),
                generics: serde_json::json!({
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }),
            }),
            args: vec![Operand::Move(Place {
                kind: PlaceKind::Local(dest_local as u64),
                ty: clone_tyref(value_ty),
            })],
            dest: call.dest.clone(),
        }))
    }

    /// A residual or `dont_look_inside` callee expects a real address for
    /// `&T`. A `repr(transparent)` scalar wrapper's borrow is the word, so
    /// passing it would hand that word to the callee as a pointer.
    ///
    /// Charon leaves the declaration generic (`monomorphize:false`), so
    /// `fn f<T>(x: &T)` is a `TypeVar` and does not name the wrapper. The
    /// operand's place type is the type the call actually passes.
    pub(in crate::front::mir) fn refuse_borrowed_transparent_scalar_residual(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
        arg_tys: &[Option<TyRef>],
    ) -> Result<(), LowerError> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return Ok(());
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(());
        };
        if !fun_decl_is_residual(fd, self.dont_look_inside) {
            return Ok(());
        }
        for (index, declared) in fd.signature.inputs.iter().enumerate() {
            let passed = arg_tys.get(index).and_then(Option::as_ref);
            if std::iter::once(declared).chain(passed).any(|ty| {
                tyref_is_borrowed_transparent_scalar(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                )
            }) {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: residual `{}` argument {index} borrows a \
                     repr(transparent) scalar; the word is not an address",
                    fd.item_meta.name_path()
                )));
            }
        }
        Ok(())
    }
}

/// True for the root-stack publication helper.  Both the free function and
/// `RootScope` method spell the last two semantic components `gc_roots` and
/// `pin_root`, with an optional `<Impl>` segment between them.
pub(in crate::front::mir) fn gc_root_pin_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    segments.last() == Some(&"pin_root") && segments.contains(&"gc_roots")
}

/// `gc_hook::gc_identity_hash` (`framework.py` `gct_gc_identityhash`).
/// The unique graph's `usize` slot is `r_uint`; callers that pass Signed
/// (`cast_ptr_to_int`) must retype before `mergeinputargs`.
pub(in crate::front::mir) fn gc_identity_hash_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    segments.last() == Some(&"gc_identity_hash") && segments.contains(&"gc_hook")
}

/// The root-bracket guard type and the call that closes one.
/// Every spelling here is a leaf or module segment: the owning crate comes
/// from the dropped value's own resolved path, so nothing below names one.
pub(in crate::front::mir) const ROOT_SCOPE_MODULE: &str = "gc_roots";
pub(in crate::front::mir) const ROOT_SCOPE_TYPE: &str = "RootScope";
/// The shadow-stack rewind the guard's destructor performs, spelled as a call
/// that takes the guard by reference.
pub(crate) const ROOT_SCOPE_CLOSE: &str = "root_scope_close";

/// True for the root-bracket guard's own type path.
pub(in crate::front::mir) fn gc_root_scope_type_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    segments.last() == Some(&ROOT_SCOPE_TYPE) && segments.contains(&ROOT_SCOPE_MODULE)
}

/// Match Charon's `gc_roots::RootScope::<Impl>::drop_in_place` path.
pub(in crate::front::mir) fn gc_root_scope_drop_glue_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    segments.last() == Some(&"drop_in_place")
        && segments.contains(&"gc_roots")
        && segments.contains(&"RootScope")
}

/// `Drop` terminator JSON for `RootScope`, recovered from a `FunDecl`
/// when the body's own cleanup `Drop` was stripped.
pub(in crate::front::mir) fn root_scope_drop_glue_template(
    llbc: &Llbc,
) -> Option<serde_json::Value> {
    let fd = llbc
        .iter_fun_decls()
        .find(|fd| gc_root_scope_drop_glue_path(&fd.item_meta.name_path()))?;
    Some(serde_json::json!({
        "Drop": {
            "place": {"kind": {"Local": 0}, "ty": {"Deduplicated": 0}},
            "fn_ptr": {
                "kind": {"Fun": fd.def_id},
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            },
            "target": 0,
            "on_unwind": 0
        }
    }))
}

/// Match Charon's `eval::FrameAnchor::<Impl>::drop_in_place` path.
///
/// `frame_anchor_new_jit_abi` publishes the slot and forgets `Drop`;
/// the matching `drop_in_place` must become a residual
/// `frame_anchor_release` or compiled loops leak every iteration.
pub(in crate::front::mir) fn frame_anchor_drop_glue_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    let last = segments.last().copied();
    // Charon names the Drop impl `FrameAnchor::drop`; the MIR
    // terminator glue is `drop_in_place`. Either must close the
    // `frame_anchor_new_jit_abi` slot.
    (last == Some("drop_in_place") || last == Some("drop")) && segments.contains(&"FrameAnchor")
}

pub(in crate::front::mir) fn tyref_is_frame_anchor(ty: &TyRef, llbc: &Llbc) -> bool {
    if tyref_class_root(ty, llbc).is_some_and(|root| root.contains("FrameAnchor")) {
        return true;
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    if let Some(id) = adt_id_flexible(node)
        && llbc
            .type_by_id(id)
            .is_some_and(|td| td.item_meta.name_path().contains("FrameAnchor"))
    {
        return true;
    }
    false
}

pub(in crate::front::mir) fn adt_id_flexible(node: &serde_json::Value) -> Option<u64> {
    let adt = node.get("Adt")?;
    if let Some(id) = adt.as_u64() {
        return Some(id);
    }
    let id = adt.get("id")?;
    if let Some(n) = id.as_u64() {
        return Some(n);
    }
    id.get("Adt").and_then(serde_json::Value::as_u64)
}

/// Dedup `place.ty` often has no Adt name. Opcode handlers name the
/// one-word local `anchor` / `frame_anchor`; those locals are
/// [`FrameAnchor`] (every `let anchor` in the interpreter is one).
pub(in crate::front::mir) fn drop_place_is_frame_anchor(
    place: &Place,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    if tyref_is_frame_anchor(&place.ty, llbc) {
        return true;
    }
    let PlaceKind::Local(local) = place.kind else {
        return false;
    };
    let Some(decl) = body.locals.locals.get(local as usize) else {
        return false;
    };
    matches!(decl.name.as_deref(), Some("anchor") | Some("frame_anchor"))
        || tyref_is_frame_anchor(&decl.ty, llbc)
}

/// The release residual a `ListGuard` drop lowers to (`rthread.py`
/// `Lock.release`).
pub(in crate::front::mir) const LIST_LOCK_RELEASE_PATH: [&str; 3] =
    ["pyre_object", "listobject", "w_list_lock_release"];

/// The release residual a `SetGuard` drop lowers to. Same `Lock.release`
/// boundary as [`LIST_LOCK_RELEASE_PATH`], on the set stripe.
pub(in crate::front::mir) const SET_LOCK_RELEASE_PATH: [&str; 3] =
    ["pyre_object", "setobject", "w_set_lock_release"];

/// A drop of the list lock's one-word `ListGuard` (`listobject::ListGuard`).
/// Its `Drop` releases the lock word; without this arm the drop fell through
/// to a plain goto and the jitcode never released an acquisition.
pub(in crate::front::mir) fn drop_place_is_list_guard(place: &Place, llbc: &Llbc) -> bool {
    if !matches!(place.kind, PlaceKind::Local(_)) {
        return false;
    }
    let Some(node) = tyref_node(&place.ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    adt_id_flexible(node)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            let path = td.item_meta.name_path();
            let segments: Vec<&str> = path.split("::").collect();
            segments.last() == Some(&"ListGuard") && segments.contains(&"listobject")
        })
}

/// Match the bound `w_list_lock_release` residual a `ListGuard` drop emits.
pub(crate) fn is_list_lock_release_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.last().map(String::as_str) == Some(LIST_LOCK_RELEASE_PATH[2])
}

/// A drop of the set lock's one-word `SetGuard` (`setobject::SetGuard`).
pub(in crate::front::mir) fn drop_place_is_set_guard(place: &Place, llbc: &Llbc) -> bool {
    if !matches!(place.kind, PlaceKind::Local(_)) {
        return false;
    }
    let Some(node) = tyref_node(&place.ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    adt_id_flexible(node)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            let path = td.item_meta.name_path();
            let segments: Vec<&str> = path.split("::").collect();
            segments.last() == Some(&"SetGuard") && segments.contains(&"setobject")
        })
}

/// Match the bound `w_set_lock_release` residual a `SetGuard` drop emits.
pub(crate) fn is_set_lock_release_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.last().map(String::as_str) == Some(SET_LOCK_RELEASE_PATH[2])
}

/// Drops supported by both lowering and its liveness analysis.
pub(in crate::front::mir) fn drop_lowers_as_glue_call(
    place: &Place,
    fn_ptr: &RegularCall,
    llbc: &Llbc,
) -> bool {
    if !matches!(place.kind, PlaceKind::Local(_)) {
        return false;
    }
    if tyref_is_frame_anchor(&place.ty, llbc) {
        return true;
    }
    regular_call_name_path(fn_ptr, llbc)
        .as_deref()
        .is_some_and(|name| gc_root_scope_drop_glue_path(name) || frame_anchor_drop_glue_path(name))
}

/// Match the lowered RootScope close used by result/exception rewrites.
///
/// #1689 hoists Charon's `drop_in_place` glue; this branch emits the named
/// `root_scope_close` residual.  Both rewind the same stack, so a rewrite
/// that walks a close has to see either spelling.
pub(crate) fn is_root_scope_drop_glue_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    let leaf = segments.last().map(String::as_str);
    let in_gc_roots = segments.iter().any(|s| s == "gc_roots");
    (leaf == Some("drop_in_place") && in_gc_roots && segments.iter().any(|s| s == "RootScope"))
        || (leaf == Some(ROOT_SCOPE_CLOSE) && in_gc_roots)
}

/// Match the bound [`frame_anchor_release`] residual emitted for a
/// `FrameAnchor` Drop. Result/exception rewrites must keep it on the
/// Err edge the same way they keep a RootScope close: `set_raise`
/// bypasses the forwarding block that held the Drop.
pub(crate) fn is_frame_anchor_release_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.last().map(String::as_str) == Some("frame_anchor_release")
}

/// Guard releases that an exceptional Result rewrite must replay on the
/// raise edge: shadow-stack closes, the list lock release and the free of
/// a raw array buffer that ends with the frame.
pub(crate) fn is_shadow_stack_bracket_close(kind: &OpKind) -> bool {
    is_root_scope_drop_glue_call(kind)
        || is_frame_anchor_release_call(kind)
        || is_list_lock_release_call(kind)
        || is_set_lock_release_call(kind)
        || is_slice_buffer_free_call(kind)
        || is_rust_vec_free_call(kind)
}

/// Match the `ll_slice_buffer_free` a raw array's frame exit emits.
pub(crate) fn is_slice_buffer_free_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments.join("::") == majit_ir::rvec::SLICE_BUFFER_FREE
}

/// Match `ll_vec_free_*` of a one-word `Vec` header, the `rewrite_op_free`
/// residual a Drop of that vec emits. Result/exception rewrites replay it
/// the same way they replay `ll_slice_buffer_free`.
pub(crate) fn is_rust_vec_free_call(kind: &OpKind) -> bool {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        ..
    } = kind
    else {
        return false;
    };
    segments
        .last()
        .is_some_and(|leaf| leaf.starts_with("ll_vec_free_"))
}

/// `core::mem::drop(move local)`: the local that call drops.
pub(in crate::front::mir) fn mem_drop_moved_place<'a>(
    call: &'a CallPayload,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<(u64, &'a Place)> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    if name_of(reg).as_deref() != Some("core::mem::drop") || call.args.len() != 1 {
        return None;
    }
    let Operand::Move(place) = &call.args[0] else {
        return None;
    };
    match place.kind {
        PlaceKind::Local(local) => Some((local, place)),
        _ => None,
    }
}

/// The local a drop terminator or an explicit `core::mem::drop(move local)` drops.
pub(in crate::front::mir) fn term_dropped_local(
    term: &TermKind,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<u64> {
    match term {
        TermKind::Drop { place, .. } => match place.kind {
            PlaceKind::Local(local) => Some(local),
            _ => None,
        },
        TermKind::Call { call, .. } => mem_drop_moved_place(call, name_of).map(|(local, _)| local),
        _ => None,
    }
}

/// Guard local an explicit `drop` closes.
///
/// Charon spells `drop(guard)` as `_t = move _g; core::mem::drop(move _t)`
/// in the same block. The operand is the temporary; the guard is the source
/// of that move chain. A direct `drop(move _g)` returns `_g`.
pub(in crate::front::mir) fn explicit_drop_guard_local(
    bb: &majit_charon_reader::ullbc::BasicBlock,
    llbc: &Llbc,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<usize> {
    let TermKind::Call { call, .. } = bb.term_ref(llbc).ok()? else {
        return None;
    };
    let (cur, _) = mem_drop_moved_place(call, name_of)?;
    let mut cur = cur as usize;
    for stmt in bb.statements.iter().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if !matches!(place.kind, PlaceKind::Local(d) if d as usize == cur) {
            continue;
        }
        let Rvalue::Use(Operand::Move(src), _) = rvalue else {
            break;
        };
        let PlaceKind::Local(src) = src.kind else {
            break;
        };
        cur = src as usize;
    }
    Some(cur)
}

/// The guard's only move-outs are same-block chains into `core::mem::drop`.
///
/// `moved_out_locals` counts `_t = move _g` and then drops the `drop(move _t)`
/// operand, so a guard closed that way looks moved-out. It still owns the
/// bracket: the temporary never escapes the drop.
pub(in crate::front::mir) fn explicit_drop_is_the_only_move(
    body: &Unstructured,
    llbc: &Llbc,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    guard: usize,
) -> bool {
    let closes: Vec<&majit_charon_reader::ullbc::BasicBlock> = body
        .body
        .iter()
        .filter(|bb| explicit_drop_guard_local(bb, llbc, name_of) == Some(guard))
        .collect();
    if closes.is_empty() {
        return false;
    }
    let counts = local_move_counts(body, llbc);
    if counts.get(&guard).copied().unwrap_or(0) != closes.len() {
        return false;
    }
    for bb in closes {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            return false;
        };
        let Some((operand, _)) = mem_drop_moved_place(call, name_of) else {
            return false;
        };
        let mut cur = operand as usize;
        let mut steps = bb.statements.len();
        while cur != guard {
            if steps == 0 {
                return false;
            }
            steps -= 1;
            if counts.get(&cur).copied().unwrap_or(0) != 0 {
                return false;
            }
            let Some(src) = bb.statements.iter().rev().find_map(|stmt| {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    return None;
                };
                if !matches!(place.kind, PlaceKind::Local(d) if d as usize == cur) {
                    return None;
                }
                let Rvalue::Use(Operand::Move(src), _) = rvalue else {
                    return None;
                };
                match src.kind {
                    PlaceKind::Local(src) => Some(src as usize),
                    _ => None,
                }
            }) else {
                return false;
            };
            cur = src;
        }
    }
    true
}

/// `local` is a temporary on an explicit-drop chain of a guard in `opener`.
pub(in crate::front::mir) fn explicit_drop_temp_of_owned_guard(
    body: &Unstructured,
    llbc: &Llbc,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    local: usize,
    opener: &std::collections::HashMap<usize, usize>,
) -> bool {
    let forward_blocks = forward_reachable_mask(llbc, body);
    body.body.iter().enumerate().any(|(bb_idx, bb)| {
        if !forward_blocks[bb_idx] {
            return false;
        }
        let Some(guard) = explicit_drop_guard_local(bb, llbc, name_of) else {
            return false;
        };
        if !opener.contains_key(&guard) || guard == local {
            return false;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            return false;
        };
        let Some((operand, _)) = mem_drop_moved_place(call, name_of) else {
            return false;
        };
        let mut cur = operand as usize;
        let mut steps = bb.statements.len();
        while cur != guard {
            if cur == local {
                return true;
            }
            if steps == 0 {
                return false;
            }
            steps -= 1;
            let Some(src) = bb.statements.iter().rev().find_map(|stmt| {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    return None;
                };
                if !matches!(place.kind, PlaceKind::Local(d) if d as usize == cur) {
                    return None;
                }
                let Rvalue::Use(Operand::Move(src), _) = rvalue else {
                    return None;
                };
                match src.kind {
                    PlaceKind::Local(src) => Some(src as usize),
                    _ => None,
                }
            }) else {
                return false;
            };
            cur = src;
        }
        false
    })
}

pub(in crate::front::mir) fn place_ty_is_root_scope(place: &Place, llbc: &Llbc) -> bool {
    output_adt_def_id_free(&place.ty, llbc)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .is_some_and(|decl| gc_root_scope_type_path(&decl.item_meta.name_path()))
}

pub(in crate::front::mir) fn glue_call_drop_blocks(
    body: &Unstructured,
    llbc: &Llbc,
) -> bit_set::BitSet {
    let mut out = bit_set::BitSet::with_capacity(body.body.len());
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(term) = bb.term_ref(llbc) else {
            continue;
        };
        let (place, glue) = match term {
            TermKind::Drop { place, fn_ptr, .. } => (place, Some(fn_ptr)),
            TermKind::Call { call, .. } => match mem_drop_moved_place(call, &name_of) {
                Some((_, place)) => (place, None),
                None => continue,
            },
            _ => continue,
        };
        // A later block's Drop reads the guard or Vec header word to emit
        // the release or glue call. An unmarked word is absent there, and
        // the destructor is skipped.
        let lowers = drop_place_is_frame_anchor(place, body, llbc)
            || drop_place_is_list_guard(place, llbc)
            || drop_place_is_set_guard(place, llbc)
            || match glue {
                Some(fn_ptr) => drop_lowers_as_glue_call(place, fn_ptr, llbc),
                None => place_ty_is_root_scope(place, llbc),
            };
        if lowers {
            out.insert(bb_idx);
        }
    }
    // A definitely-initialised `Drop _v` of a `Vec<T>` local is a use:
    // the destructor reads the header. `elaborate_vec_drops` has already
    // rewritten a definitely-moved-out Drop to `Goto`, so those never
    // ask liveness for `_v`. A maybe-initialised Drop is left unmarked.
    out.union_with(&definitely_init_vec_drop_blocks(body, llbc));
    out
}

/// A root bracket this pass takes out of the jitcode altogether.
///
/// Upstream never puts one there.  `gc_push_roots` / `gc_pop_roots` are
/// genop'd by the shadow-stack transformer (`memory/gctransform/
/// shadowstack.py push_roots`, reached from `framework.py`), and that
/// transformer runs out of the C backend's database (`translator/c/
/// database.py:64`, per graph from `translator/c/funcgen.py make_funcgen`), long  allow-line-citation
/// after `warmspot.py make_jitcodes` has already read the graphs.  A
/// jitcode therefore never carries a bracket, and `rg 'gc_push_roots'
/// rpython/jit/` finds nothing.
///
/// pyre spells the same bracket in interpreter source, so its jitcodes read
/// it back in.  Each of the three consumers of a jitcode already has its own
/// root set for the references the bracket republishes:
///
/// * the compiled trace — the jitframe gcmap the backend writes around every
///   call whose effect info says it can collect (`regalloc.rs get_gcmap`,
///   read back by `majit-gc/src/shadow_stack.rs walk_jf_roots`);
/// * the blackhole — `push_bh_regs` roots the whole `registers_r` bank of the
///   running frame and every pending caller for the length of `run()`;
/// * the recording walk — every value it computes is stamped onto the op
///   recorded for it, and `walk_active_trace_refs` forwards those cells, so
///   the recorder is the root set (`fbw_state.rs FBW_FINISH_CONCRETE`).
///
/// The native Rust build has none of those, and keeps executing the source
/// bracket unchanged: this pass rewrites what a jitcode says, never what
/// rustc compiled.
///
/// The erasure is confined to the shape whose read-backs can be answered
/// without the shadow stack: pins that run once each, in one order, and every
/// `get` indexed by that scope's own `base()` plus a constant, or by a
/// `shadow_stack_len()` read of the same scope plus a constant.  A later
/// `pin_roots` dest is the first pin of that run, not the guard's base.  A
/// free `pin_root(x)` inside that scope is the `k`-th pin exactly as
/// `pin_roots` is, and the scope's close rewinds it, so the close is erased
/// with the rest.  Then `get(base + k)` / `get(len + k)` is the value of the
/// `k`-th pin, and nothing else observes the guard.  `pin_roots` of an
/// incoming `&[PyObjectRef]` parameter is the same shape with the list itself
/// as the source: `get(base + k)` is `getarrayitem` of that list, and
/// `shadow_stack_copy_range(base + k, dest)` copies the live slice
/// (`W_CData.call` / `cdataobj.py descr_call` passes `args_w` through).
/// Every other call inside the bracket must leave the root stack as it found
/// it ([`RootStackAnalyzer`]), which is the per-graph balance the upstream
/// transformer guarantees.  Spanning a collecting call does not by itself
/// keep the bracket: the three jitcode consumers already root the
/// republished refs.  A bracket that pins in a loop, reads a slot this pass
/// cannot name, or spans a call that can change the stack, keeps every op
/// it has.
#[derive(Default)]
pub(in crate::front::mir) struct RootBracketPlan {
    /// Whether the pass runs at all (`MAJIT_ROOT_BRACKET_ERASE`).
    pub(in crate::front::mir) enabled: bool,
    /// Guard locals whose whole bracket is erased.
    pub(in crate::front::mir) scopes: bit_set::BitSet,
    /// `_a = &_scope` temporaries, mapped to the guard they borrow.
    pub(in crate::front::mir) aliases: std::collections::HashMap<usize, usize>,
    /// Guard local -> the locals its pins publish, in slot order.
    pub(in crate::front::mir) pins: std::collections::HashMap<usize, Vec<usize>>,
    /// `RootScope::base` result locals, mapped to their guard.
    pub(in crate::front::mir) base_results: std::collections::HashMap<usize, usize>,
    /// `base + k` temporaries and `shadow_stack_len` results, mapped to their
    /// guard.  Both only ever indexed a read-back, so they bind nothing.
    pub(in crate::front::mir) slot_temps: std::collections::HashMap<usize, usize>,
    /// `(block, pinned local)` for every erased `get`, so the pinned value
    /// stays live up to the read it now answers.
    pub(in crate::front::mir) get_sites: Vec<(usize, usize)>,
    /// The temporaries a `pin_roots(&[..])` builds its argument through --
    /// the array, its borrows and the unsizing cast -- mapped to their guard.
    /// They only ever fed the erased pin, so they bind nothing.
    pub(in crate::front::mir) pin_temps: std::collections::HashMap<usize, usize>,
    /// `get(base + k)` of a `pin_roots` of an incoming `&[PyObjectRef]`.
    /// `(block, slice local, k)`, answered with `getarrayitem`.
    pub(in crate::front::mir) slice_gets: Vec<(usize, usize, u64)>,
    /// `shadow_stack_copy_range(base + k, dest)` of that same pin.
    /// `(block, slice local, k)`, rewritten as a copy from the live list.
    pub(in crate::front::mir) slice_copies: Vec<(usize, usize, u64)>,
    /// Temporaries that only build a closure capture of an erased index: the
    /// borrow of the index, the closure aggregate, the receiver borrow and the
    /// argument tuple.  The closure call is answered by the pinned value, so
    /// these bind nothing — the index they borrow may itself bind nothing.
    pub(in crate::front::mir) capture_temps: std::collections::HashMap<usize, usize>,
    /// Blocks whose free `gc_roots::pin_roots`, `pin_root`, `shadow_stack_len`
    /// or `shadow_stack_get` call is a pin, a depth read or a read-back of
    /// the guard mapped to.  The free call names no guard and acts on the
    /// innermost bracket open where it runs. A free `pin_root` whose slot
    /// that scope's `shadow_stack_len` names is the `k`-th pin; a free pin
    /// this pass cannot name still owes its close.
    pub(in crate::front::mir) free_sites: std::collections::HashMap<usize, usize>,
}

impl RootBracketPlan {
    pub(in crate::front::mir) fn is_erased_scope(&self, local: usize) -> bool {
        self.enabled && self.scopes.contains(local)
    }

    /// The pinned local an erased `get` in `mir_bb` answers with.
    pub(in crate::front::mir) fn get_answer(&self, mir_bb: usize) -> Option<usize> {
        self.get_sites
            .iter()
            .find(|(bb, _)| *bb == mir_bb)
            .map(|(_, value)| *value)
    }

    /// The incoming list and index an erased `get` in `mir_bb` answers with.
    pub(in crate::front::mir) fn slice_get_answer(&self, mir_bb: usize) -> Option<(usize, u64)> {
        self.slice_gets
            .iter()
            .find(|(bb, _, _)| *bb == mir_bb)
            .map(|(_, slice, k)| (*slice, *k))
    }

    /// The incoming list and start an erased `shadow_stack_copy_range` in
    /// `mir_bb` copies from.
    pub(in crate::front::mir) fn slice_copy_answer(&self, mir_bb: usize) -> Option<(usize, u64)> {
        self.slice_copies
            .iter()
            .find(|(bb, _, _)| *bb == mir_bb)
            .map(|(_, slice, k)| (*slice, *k))
    }

    /// The erased guard a free pin or read-back in `mir_bb` acts on.
    pub(in crate::front::mir) fn free_site_scope(&self, mir_bb: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let scope = self.free_sites.get(&mir_bb).copied()?;
        self.scopes.contains(scope).then_some(scope)
    }

    /// The guard an erased-bracket call's receiver operand names.
    pub(in crate::front::mir) fn scope_of_receiver(&self, local: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let scope = self.aliases.get(&local).copied()?;
        self.scopes.contains(scope).then_some(scope)
    }
}

/// `MAJIT_ROOT_BRACKET_ERASE` — on by default; `0` / `false` keeps every
/// root-bracket op in the jitcode, which is the shape that shipped before
/// [`RootBracketPlan`] existed.
pub(in crate::front::mir) fn root_bracket_erase_enabled() -> bool {
    !matches!(
        std::env::var("MAJIT_ROOT_BRACKET_ERASE").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// `graphanalyze.py BoolGraphAnalyzer` over the shadow stack.  The top result
/// means a call can leave the root stack deeper or shallower than it found it,
/// or read or write a slot it did not publish itself.
///
/// Upstream needs no such analysis.  `ShadowStackFrameworkGCTransformer`
/// (`memory/gctransform/shadowstack.py`, driven per graph from
/// `framework.py`) saves and restores the live variables of a graph inside
/// that same graph, so every call leaves the stack as it found it, and a
/// bracket the JIT never sees owes nothing to its callees.  pyre spells the
/// bracket in interpreter source, where a helper may hand its pins to its
/// caller's close.  Erasing that caller's bracket would leak one slot per
/// call, so the per-graph balance is proved here instead of assumed.
///
/// A body that opens its own bracket and closes it again is balanced whatever
/// runs inside, because the close truncates to the length the opener saw.
/// That is the "covered" test in [`Self::analyze_body`].
pub(crate) struct RootStackAnalyzer<'a> {
    pub(in crate::front::mir) llbc: &'a Llbc,
    pub(in crate::front::mir) state: &'a RootStackState,
}

/// The per-artefact caches a [`RootStackAnalyzer`] fills.
pub(crate) struct RootStackState {
    /// Whether this artefact defines the root-stack API.  A dependency cannot
    /// name a crate built on top of it, so a foreign body here reaches the
    /// root stack only through a callback it is handed, and that callback is
    /// analysed at the call site.  In a crate that only imports the API a
    /// foreign body may be one of its helpers, so nothing is known about it.
    pub(in crate::front::mir) root_api_is_local: bool,
    /// Whether this artefact names the root-stack API at all, locally or as
    /// an import.  One that does not sits below the API's crate, as do the
    /// crates it calls into, so none of its bodies can reach the stack
    /// except through a callback analysed at the call site.
    pub(in crate::front::mir) root_api_is_visible: bool,
    /// `GraphAnalyzer._analyzed_calls`, keyed by `FunDecl` id.
    pub(in crate::front::mir) analyzed_calls: std::cell::RefCell<
        crate::tool::algo::unionfind::UnionFind<
            u64,
            crate::translator::backendopt::graphanalyze::Dependency<bool>,
        >,
    >,
    /// Every impl body a trait method call may reach, built on first use.
    pub(in crate::front::mir) trait_methods: std::cell::OnceCell<TraitMethodBodies>,
    /// `FunDecl` id -> whether that body opens a bracket and returns its
    /// guard inside the result, as `DictOperationGuard::new` does.
    pub(in crate::front::mir) scope_constructors:
        std::cell::RefCell<std::collections::HashMap<u64, bool>>,
}

/// The impl bodies behind each trait method of one artefact.
#[derive(Default)]
pub(in crate::front::mir) struct TraitMethodBodies {
    /// `(trait decl id, method index)` -> every impl's body for it.
    pub(in crate::front::mir) by_trait: std::collections::HashMap<(u64, u64), Vec<u64>>,
    /// `(trait impl id, method index)` -> that impl's body.
    pub(in crate::front::mir) by_impl: std::collections::HashMap<(u64, u64), u64>,
}

pub(in crate::front::mir) type RootStackTracker =
    crate::translator::backendopt::graphanalyze::DependencyTracker<bool, u64>;

impl RootStackState {
    pub(crate) fn new(llbc: &Llbc) -> Self {
        let root_api_is_local = llbc.iter_type_decls().any(|decl| {
            decl.item_meta.is_local && gc_root_scope_type_path(&decl.item_meta.name_path())
        });
        let root_api_is_visible = root_api_is_local
            || llbc
                .iter_type_decls()
                .any(|decl| gc_root_scope_type_path(&decl.item_meta.name_path()))
            || llbc.iter_fun_decls().any(|fd| {
                fd.item_meta
                    .name_path()
                    .split("::")
                    .any(|s| s == ROOT_SCOPE_MODULE)
            });
        Self {
            root_api_is_local,
            root_api_is_visible,
            analyzed_calls: std::cell::RefCell::new(crate::tool::algo::unionfind::UnionFind::new(
                |_: &u64| crate::translator::backendopt::graphanalyze::Dependency::new(false),
            )),
            trait_methods: std::cell::OnceCell::new(),
            scope_constructors: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }
}

impl<'a> RootStackAnalyzer<'a> {
    pub(crate) fn new(llbc: &'a Llbc, state: &'a RootStackState) -> Self {
        Self { llbc, state }
    }

    /// Whether this call returns a value owning a bracket the callee opened.
    /// The caller's local holding the result is then that bracket's guard:
    /// the call opens it and the local's drop closes it.
    pub(in crate::front::mir) fn call_returns_owned_scope(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.fn_returns_owned_scope(*id)
    }

    /// Whether `id`'s body opens a bracket and returns its guard inside the
    /// result. A declaration with no body answers from the flag the link
    /// step set on that `FunDecl` when its identity matched a defining crate.
    pub(in crate::front::mir) fn fn_returns_owned_scope(&self, id: u64) -> bool {
        if let Some(&known) = self.state.scope_constructors.borrow().get(&id) {
            return known;
        }
        // A cycle does not return a scope through itself. A direct
        // `push_roots` in this body is still seen below.
        self.state.scope_constructors.borrow_mut().insert(id, false);
        let answer = match self.llbc.fn_by_id(id) {
            Some(fd) => match fd.unstructured() {
                Some(body) => body_returns_owned_scope(
                    self.llbc,
                    &body,
                    &|reg| regular_call_name_path(reg, self.llbc),
                    &|reg| self.call_returns_owned_scope(reg),
                ),
                None => fd
                    .returns_owned_scope
                    .load(std::sync::atomic::Ordering::Relaxed),
            },
            None => false,
        };
        self.state
            .scope_constructors
            .borrow_mut()
            .insert(id, answer);
        answer
    }

    /// Whether a statically resolved call can change the root stack.
    pub(crate) fn regular_call_touches_root_stack(&self, reg: &RegularCall) -> bool {
        let mut seen = RootStackTracker::new();
        self.analyze_regular_call(reg, &mut seen)
    }

    pub(in crate::front::mir) fn analyze_regular_call(
        &self,
        reg: &RegularCall,
        seen: &mut RootStackTracker,
    ) -> bool {
        match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => self.analyze_direct_call(*id, seen),
            // Builtins: arithmetic, `Box::new`, slice indexing.
            CallKind::Fun(FunId::Other(_)) => false,
            CallKind::Trait(v) => self.analyze_trait_call(v, seen),
            // A body reached through a pointer cannot know which caller's
            // close would rewind a pin it left behind, so it closes its own:
            // the per-graph balance `ShadowStackFrameworkGCTransformer` gives
            // every graph.  The helpers that hand pins to their caller are
            // called directly, and the direct arm above reaches them.
            CallKind::Ptr(_) => false,
            CallKind::Unknown => true,
        }
    }

    /// `analyze_direct_call`, with the `DependencyTracker` that merges a
    /// recursive cycle into one result.
    pub(in crate::front::mir) fn analyze_direct_call(
        &self,
        id: u64,
        seen: &mut RootStackTracker,
    ) -> bool {
        let Some(fd) = self.llbc.fn_by_id(id) else {
            return true;
        };
        let path = fd.item_meta.name_path();
        if path.split("::").any(|s| s == ROOT_SCOPE_MODULE) {
            return !root_stack_api_is_neutral(&path);
        }
        // Charon's drop glue without a body this reader can walk, for a type
        // outside the root-stack API: the generic `Drop` reasoning in
        // `analyze_trait_call` applies to it unchanged -- a value can close a
        // bracket only by holding a guard it was handed, and the body that
        // handed it over is charged.
        if path.rsplit("::").next() == Some("drop_in_place") && fd.unstructured().is_none() {
            return false;
        }
        if fd.body.is_none() {
            return self.analyze_external_call(fd);
        }
        if !seen.enter(id, &mut self.state.analyzed_calls.borrow_mut()) {
            return seen.get_cached_result(id, &mut self.state.analyzed_calls.borrow_mut());
        }
        let result = match fd.unstructured() {
            Some(mut body) => {
                elaborate_explicit_root_closes(self.llbc, &mut body, &|reg| {
                    regular_call_name_path(reg, self.llbc)
                });
                elaborate_vec_drops(self.llbc, &mut body);
                self.analyze_body(&body, seen)
            }
            None => self.analyze_external_call(fd),
        };
        seen.leave_with(id, result, &mut self.state.analyzed_calls.borrow_mut());
        result
    }

    /// `analyze_external_call`: a declaration with no body in this artefact.
    ///
    /// A crate that is not part of the linked translation input cannot name
    /// the root-stack API, which lives in one that is; like a dependency of
    /// the API's own crate, it reaches the stack only through a callback,
    /// and a callback is analysed at the call site.  A body in a linked crate
    /// analysed earlier answers with that artefact's published result.
    ///
    /// A local declaration whose body Charon left `Opaque` is unknown, except
    /// in an artefact that does not name the API at all: nothing it contains
    /// can reach the stack but a callback.
    pub(in crate::front::mir) fn analyze_external_call(&self, fd: &FunDecl) -> bool {
        if fd.item_meta.is_local {
            return self.state.root_api_is_visible;
        }
        if self.state.root_api_is_local {
            return false;
        }
        if !self.llbc.has_root_stack_effects() {
            return self.state.root_api_is_visible;
        }
        let path = fd.item_meta.name_path();
        let krate = path.split("::").next().unwrap_or_default();
        match self.llbc.root_stack_effect(krate, &path) {
            Some(touches) => touches,
            None => crate::local_crates::is_local_crate_root(krate),
        }
    }

    /// Whether the body with this `FunDecl` id can change the root stack.
    pub(in crate::front::mir) fn fn_touches_root_stack(&self, id: u64) -> bool {
        let mut seen = RootStackTracker::new();
        self.analyze_direct_call(id, &mut seen)
    }

    /// Every body a trait method call may reach: the impl the call selected,
    /// or every impl of the trait when the call is still generic, plus the
    /// trait's default body.  No body at all is an unknown callee.
    pub(in crate::front::mir) fn analyze_trait_call(
        &self,
        v: &serde_json::Value,
        seen: &mut RootStackTracker,
    ) -> bool {
        let Some(arr) = v.as_array() else {
            return true;
        };
        let (Some(traitref), Some(method)) =
            (arr.first(), arr.get(1).and_then(serde_json::Value::as_u64))
        else {
            return true;
        };
        let bodies = self
            .state
            .trait_methods
            .get_or_init(|| trait_method_bodies(self.llbc));
        let mut targets: Vec<u64> = Vec::new();
        if let Some(impl_id) = traitref_impl_id(traitref, self.llbc, 0) {
            targets.extend(bodies.by_impl.get(&(impl_id, method)).copied());
        } else if let Some(trait_id) = traitref_decl_id(traitref, self.llbc, 0) {
            // A still-generic `Drop::drop` (or the `Destruct` drop glue that
            // reaches it) destroys a value whose type the
            // caller chose.  The only destructors that change the root stack
            // are guards closing their own bracket, and a guard reaches a
            // generic drop only by being passed in by value: the body that
            // handed it over is the one charged, because its pins then run
            // outside any close it owns.  The close itself truncates to the
            // guard's save point, which drops no pin its opener did not see
            // made.
            if self.llbc.trait_by_id(trait_id).is_some_and(|decl| {
                matches!(
                    decl.item_meta.name_path().as_str(),
                    "core::ops::drop::Drop" | "core::marker::Destruct"
                )
            }) {
                return false;
            }
            targets.extend(
                bodies
                    .by_trait
                    .get(&(trait_id, method))
                    .into_iter()
                    .flatten()
                    .copied(),
            );
        }
        // Only a provided default is a body; a required method's declaration
        // carries none to analyse, and every impl is already listed.
        if let Some(decl) = traitref_decl_id(traitref, self.llbc, 0)
            .and_then(|trait_id| trait_default_method_id(self.llbc, trait_id, method))
            && self
                .llbc
                .fn_by_id(decl)
                .is_some_and(|fd| fd.unstructured().is_some())
        {
            targets.push(decl);
        }
        // No body in this artefact implements the method.  An artefact that
        // does not name the root-stack API reaches it only through an impl a
        // crate built on top of it supplies, which is a callback, as in
        // `analyze_external_call`.
        if targets.is_empty() {
            return self.state.root_api_is_visible;
        }
        targets
            .into_iter()
            .any(|target| self.analyze_direct_call(target, seen))
    }

    /// The function items a block names as values rather than calls, such as
    /// the callback `jit.conditional_call` receives.  A value that is called
    /// later is called from wherever it flows, so it is charged here.
    pub(in crate::front::mir) fn analyze_fn_values(
        &self,
        v: &serde_json::Value,
        seen: &mut RootStackTracker,
    ) -> bool {
        let mut found = false;
        for_each_fn_def_const(self.llbc, v, &mut |reg| {
            found = found || self.analyze_regular_call(reg, seen);
        });
        found
    }

    /// [`Self::analyze_fn_values`] for one call operand.  An `Operand::Const`
    /// carries the constant itself, without the `Const` key a statement's
    /// operand is spelled under.
    pub(in crate::front::mir) fn analyze_fn_operand(
        &self,
        op: &Operand,
        seen: &mut RootStackTracker,
    ) -> bool {
        let Operand::Const(c) = op else {
            return false;
        };
        fn_def_of_const(self.llbc, c).is_some_and(|reg| self.analyze_regular_call(&reg, seen))
            || self.analyze_fn_values(c, seen)
    }

    /// The body's own result: some call it makes outside every bracket it
    /// opens and closes itself can change the root stack.
    ///
    /// A call inside such a bracket is not visited at all.  Its answer would
    /// not reach this body's result, and `DependencyTracker` merges every
    /// cycle it walks on the premise that each edge joins the callee's result
    /// into the caller's: a cycle closed through a covered call would then
    /// share one answer with graphs that cannot observe it.
    pub(in crate::front::mir) fn analyze_body(
        &self,
        body: &Unstructured,
        seen: &mut RootStackTracker,
    ) -> bool {
        let name_of = |reg: &RegularCall| regular_call_name_path(reg, self.llbc);
        let owned = owned_root_scopes(self.llbc, body, &name_of, &|reg| {
            self.call_returns_owned_scope(reg)
        });
        let covered = owned.covered_blocks(self.llbc, body, &name_of);
        let forward_blocks = forward_reachable_mask(self.llbc, body);
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            if covered.contains(bb_idx) {
                continue;
            }
            let mut touches = bb
                .statements
                .iter()
                .any(|stmt| self.analyze_fn_values(stmt.kind_value(), seen));
            // An opener's own call is what its close truncates: a guard
            // constructor's pins land in the bracket this body now holds.
            let opens = owned.opener.values().any(|&open_bb| open_bb == bb_idx);
            touches = touches
                || match bb.term_ref(self.llbc) {
                    Ok(TermKind::Call { .. }) if opens => false,
                    Ok(TermKind::Call { call, .. }) => {
                        // An explicit `drop` of a bracket this body opened is
                        // its close, the same balance a `Drop` terminator is.
                        if explicit_drop_guard_local(bb, self.llbc, &name_of)
                            .is_some_and(|local| owned.opener.contains_key(&local))
                        {
                            false
                        } else {
                            match &call.func {
                                CallFunc::Regular(reg) => {
                                    self.analyze_regular_call(reg, seen)
                                        || call
                                            .args
                                            .iter()
                                            .any(|op| self.analyze_fn_operand(op, seen))
                                }
                                // `dyn Trait` dispatch: the pointer arm's reasoning.
                                CallFunc::Dynamic(..) => false,
                                CallFunc::Unknown => true,
                            }
                        }
                    }
                    // The close of a bracket this body opened is its balance,
                    // not an effect. The temporary Charon moves the guard
                    // through on the way to `mem::drop` is the same close.
                    Ok(TermKind::Drop { place, fn_ptr, .. }) => {
                        let owned_local = matches!(place.kind, PlaceKind::Local(l) if {
                            let local = l as usize;
                            owned.opener.contains_key(&local)
                                || explicit_drop_temp_of_owned_guard(
                                    body,
                                    self.llbc,
                                    &name_of,
                                    local,
                                    &owned.opener,
                                )
                        });
                        !owned_local && self.analyze_regular_call(fn_ptr, seen)
                    }
                    Ok(_) => false,
                    Err(_) => true,
                };
            if touches {
                return true;
            }
        }
        false
    }
}

/// Root-stack API calls that leave the stack's depth alone and name no slot
/// they did not open themselves.
///
/// Opening a bracket saves the current length, and a scope's `base` is its
/// own save point.  A read or write by explicit index is neutral too: the
/// index comes from the body that opened the slot, and
/// [`analyze_root_brackets_with`] step (3) keeps any bracket whose slot index
/// reaches a call or a captured value, so no callee ever indexes a slot an
/// erased bracket failed to push.  What stays charged is a depth change, and
/// a position read -- `shadow_stack_len` or the top slot -- made outside any
/// bracket the body opened itself, which observes whatever its caller pushed.
///
/// The static roots (`RootedOnceRef`, the prebuilt dirty bit) share the
/// module but are not stack slots.
pub(in crate::front::mir) fn root_stack_api_is_neutral(path: &str) -> bool {
    gc_root_scope_open_path(path)
        || gc_root_scope_base_path(path)
        || matches!(
            path.rsplit("::").next(),
            Some(
                "root_stack_depth"
                    | "increase_root_stack_depth"
                    | "shadow_stack_get"
                    | "shadow_stack_set"
                    | "shadow_stack_copy_range"
                    | "get"
                    | "set"
                    | "take"
                    | "get_or_init"
                    | "normalize"
                    | "normalize_moved"
                    | "normalize_roots"
                    | "gcarray_ref_items"
                    | "mark_prebuilt_roots_dirty"
                    | "prebuilt_roots_dirty"
            )
        )
}

/// `(trait decl id, method index)` and `(impl id, method index)` -> the impl
/// method body, from the artefact's `trait_impls` rows.
pub(in crate::front::mir) fn trait_method_bodies(llbc: &Llbc) -> TraitMethodBodies {
    let mut out = TraitMethodBodies::default();
    for row in llbc.trait_impls_raw() {
        let Some(impl_id) = row.get("def_id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let Some(methods) = row.get("methods").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for method in methods {
            let Some(key) = method
                .get("kind")
                .and_then(|kind| kind.get("TraitMethod"))
                .and_then(serde_json::Value::as_array)
            else {
                continue;
            };
            let (Some(trait_id), Some(index), Some(body)) = (
                key.first().and_then(serde_json::Value::as_u64),
                key.get(1).and_then(serde_json::Value::as_u64),
                method
                    .get("skip_binder")
                    .and_then(|b| b.get("id"))
                    .and_then(serde_json::Value::as_u64),
            ) else {
                continue;
            };
            out.by_trait
                .entry((trait_id, index))
                .or_default()
                .push(body);
            out.by_impl.insert((impl_id, index), body);
        }
    }
    out
}

/// The trait's default body for its `method`-th method
/// (`methods[method].skip_binder.default.id`), when it has one.
pub(in crate::front::mir) fn trait_default_method_id(
    llbc: &Llbc,
    trait_id: u64,
    method: u64,
) -> Option<u64> {
    llbc.trait_by_id(trait_id)?
        .methods
        .get(method as usize)?
        .pointer("/skip_binder/default/id")?
        .as_u64()
}

/// The function item a `FnDef` node names, spelled as the `RegularCall`
/// a direct call to it would carry — Fun id and the instantiation
/// `generics`. A `RegionBinder` unwraps to `skip_binder`.
pub(in crate::front::mir) fn regular_call_from_fn_def(
    fn_def: &serde_json::Value,
) -> Option<RegularCall> {
    serde_json::from_value(fn_def.get("skip_binder").unwrap_or(fn_def).clone()).ok()
}

/// The function item a constant names, spelled as the `RegularCall` a direct
/// call to it would carry.
pub(in crate::front::mir) fn fn_def_of_const(
    llbc: &Llbc,
    c: &serde_json::Value,
) -> Option<RegularCall> {
    let kind = llbc.const_expr_kind(c)?;
    regular_call_from_fn_def(kind.get("FnDef")?)
}

/// Visit every `FnDef` constant under a statement or operand.
pub(in crate::front::mir) fn for_each_fn_def_const(
    llbc: &Llbc,
    v: &serde_json::Value,
    visit: &mut impl FnMut(&RegularCall),
) {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(reg) = map.get("Const").and_then(|c| fn_def_of_const(llbc, c)) {
                visit(&reg);
            }
            for nested in map.values() {
                for_each_fn_def_const(llbc, nested, visit);
            }
        }
        serde_json::Value::Array(items) => {
            for nested in items {
                for_each_fn_def_const(llbc, nested, visit);
            }
        }
        _ => {}
    }
}

/// The brackets a body opens on a guard it keeps, and the blocks that run
/// while one of them is open.
pub(in crate::front::mir) struct OwnedRootScopes {
    /// Guard local -> the block whose call opens it.
    pub(in crate::front::mir) opener: std::collections::HashMap<usize, usize>,
}

pub(in crate::front::mir) fn owned_root_scopes(
    llbc: &Llbc,
    body: &Unstructured,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    returns_owned_scope: &impl Fn(&RegularCall) -> bool,
) -> OwnedRootScopes {
    let moved = MovedOutLocals::new(body, llbc);
    let mut opener = std::collections::HashMap::new();
    let mut twice = bit_set::BitSet::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        if !name_of(reg).is_some_and(|path| gc_root_scope_open_path(&path))
            && !returns_owned_scope(reg)
        {
            continue;
        }
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        let dest = dest as usize;
        if moved.contains(dest) && !explicit_drop_is_the_only_move(body, llbc, name_of, dest) {
            continue;
        }
        if opener.insert(dest, bb_idx).is_some() {
            twice.insert(dest);
        }
    }
    for dest in twice.iter() {
        opener.remove(&dest);
    }
    OwnedRootScopes { opener }
}

/// Whether `body` opens a bracket and hands its guard back inside the value
/// it returns: the guard, through any plain moves, enters an aggregate that
/// becomes `_0`, directly or through one moved local.  Such a constructor's pins belong to
/// the bracket its caller now holds.
pub(in crate::front::mir) fn body_returns_owned_scope(
    llbc: &Llbc,
    body: &Unstructured,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    returns_owned: &impl Fn(&RegularCall) -> bool,
) -> bool {
    let guards: bit_set::BitSet = body
        .body
        .iter()
        .filter_map(|bb| match bb.term_ref(llbc) {
            Ok(TermKind::Call { call, .. }) => match (&call.func, &call.dest.kind) {
                (CallFunc::Regular(reg), PlaceKind::Local(dest))
                    if name_of(reg).is_some_and(|path| gc_root_scope_open_path(&path))
                        || returns_owned(reg) =>
                {
                    Some(*dest as usize)
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    if guards.is_empty() {
        return false;
    }
    // Follow plain moves of a guard (`_21 = move _3`) to the local that
    // enters the aggregate.
    let mut guards = guards;
    let forward_blocks = forward_reachable_mask(llbc, body);
    loop {
        let before = guards.count();
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                if let Ok(StmtKind::Assign(place, Rvalue::Use(Operand::Move(src), _))) =
                    stmt.stmt_kind_ref()
                    && let (PlaceKind::Local(dest), PlaceKind::Local(from)) =
                        (&place.kind, &src.kind)
                    && guards.contains(*from as usize)
                {
                    guards.insert(*dest as usize);
                }
            }
        }
        if guards.count() == before {
            break;
        }
    }
    let mut holders = bit_set::BitSet::new();
    let mut moved_into_result = bit_set::BitSet::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, value)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let PlaceKind::Local(dest) = place.kind else {
                continue;
            };
            match value {
                Rvalue::Aggregate(_, operands)
                    if operands.iter().any(|op| {
                        matches!(op, Operand::Move(src)
                            if matches!(src.kind, PlaceKind::Local(g) if guards.contains(g as usize)))
                    }) =>
                {
                    holders.insert(dest as usize);
                }
                Rvalue::Use(Operand::Move(src), _) if dest == 0 => {
                    if let PlaceKind::Local(h) = src.kind {
                        moved_into_result.insert(h as usize);
                    }
                }
                _ => {}
            }
        }
    }
    holders.contains(0) || holders.iter().any(|h| moved_into_result.contains(h))
}

impl OwnedRootScopes {
    /// Blocks every path to which opens one of these brackets and has not
    /// closed it since: dominated by the opener, and not reachable from a
    /// close without passing the opener again.  The close truncates whatever
    /// such a block pushes.
    pub(in crate::front::mir) fn covered_blocks(
        &self,
        llbc: &Llbc,
        body: &Unstructured,
        name_of: &impl Fn(&RegularCall) -> Option<String>,
    ) -> bit_set::BitSet {
        let mut covered = bit_set::BitSet::new();
        if self.opener.is_empty() {
            return covered;
        }
        let dom = block_dominators(llbc, body);
        let forward_blocks = forward_reachable_mask(llbc, body);
        for (&scope, &open_bb) in &self.opener {
            let drops: Vec<usize> = body
                .body
                .iter()
                .enumerate()
                .filter(|(bb_idx, bb)| {
                    forward_blocks[*bb_idx]
                        && (bb
                            .term_ref(llbc)
                            .ok()
                            .and_then(|term| term_dropped_local(term, name_of))
                            == Some(scope as u64)
                            || explicit_drop_guard_local(bb, llbc, name_of) == Some(scope))
                })
                .map(|(i, _)| i)
                .collect();
            let mut after_close = bit_set::BitSet::new();
            let mut work: Vec<usize> = drops
                .iter()
                .flat_map(|&d| block_successors(llbc, body, d))
                .collect();
            while let Some(bb) = work.pop() {
                if bb == open_bb || !after_close.insert(bb) {
                    continue;
                }
                work.extend(block_successors(llbc, body, bb));
            }
            for bb in 0..body.body.len() {
                if !forward_blocks[bb] {
                    continue;
                }
                if bb != open_bb
                    && dom[bb].contains(open_bb)
                    && !after_close.contains(bb)
                    && !drops.contains(&bb)
                {
                    covered.insert(bb);
                }
            }
        }
        covered
    }
}

/// Successors of a block, unwind edges included: a wider predecessor set can
/// only shrink the dominator sets this feeds, so an unwind edge that is not
/// really taken costs an erasure rather than licensing one.
pub(in crate::front::mir) fn block_successors(
    llbc: &Llbc,
    body: &Unstructured,
    bb: usize,
) -> Vec<usize> {
    let n = body.body.len();
    let mut out = Vec::new();
    let mut push = |target: u64| {
        let target = target as usize;
        if target < n {
            out.push(target);
        }
    };
    match body.body[bb].term_ref(llbc) {
        Ok(TermKind::Goto { target }) => push(*target),
        Ok(TermKind::Switch { targets, .. }) => match targets {
            SwitchTargets::If(a, b) => {
                push(*a);
                push(*b);
            }
            SwitchTargets::SwitchInt(_, arms, default) => {
                for (_, target) in arms {
                    push(*target);
                }
                push(*default);
            }
        },
        Ok(TermKind::Call {
            target, on_unwind, ..
        })
        | Ok(TermKind::Assert {
            target, on_unwind, ..
        })
        | Ok(TermKind::Drop {
            target, on_unwind, ..
        }) => {
            push(*target);
            push(*on_unwind);
        }
        _ => {}
    }
    out
}

/// Per-block dominator sets, by the textbook iterative intersection.
pub(in crate::front::mir) fn block_dominators(
    llbc: &Llbc,
    body: &Unstructured,
) -> Vec<bit_set::BitSet> {
    let n = body.body.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for bb in 0..n {
        for succ in block_successors(llbc, body, bb) {
            preds[succ].push(bb);
        }
    }
    let all: bit_set::BitSet = (0..n).collect();
    let mut dom: Vec<bit_set::BitSet> = vec![all; n];
    if n > 0 {
        dom[0] = std::iter::once(0).collect();
    }
    let mut changed = true;
    while changed {
        changed = false;
        for bb in 1..n {
            let mut next: Option<bit_set::BitSet> = None;
            for &pred in &preds[bb] {
                next = Some(match next {
                    None => dom[pred].clone(),
                    Some(mut acc) => {
                        acc.intersect_with(&dom[pred]);
                        acc
                    }
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(bb);
            if next != dom[bb] {
                dom[bb] = next;
                changed = true;
            }
        }
    }
    dom
}

/// Prove that a pin is executed at most once per opening of its scope.
/// `ShadowStackFrameworkGCTransformer` `push_roots` and
/// `shadowcolor.expand_one_pop_roots` pair a particular live value with a
/// particular slot; a pin a backedge reaches again without reopening the
/// scope appends another slot under the same base, which that pairing
/// cannot name.  What else runs before the pin is
/// [`root_bracket_stack_effects_are_known`]'s to prove.
pub(in crate::front::mir) fn root_pin_runs_once_per_opening(
    llbc: &Llbc,
    body: &Unstructured,
    opener: usize,
    pin: usize,
) -> bool {
    let mut seen = bit_set::BitSet::new();
    let mut work = block_successors(llbc, body, pin);
    while let Some(bb) = work.pop() {
        if bb == opener || !seen.insert(bb) {
            continue;
        }
        if bb == pin {
            return false;
        }
        work.extend(block_successors(llbc, body, bb));
    }
    true
}

/// The block a bracket opens into: the opener call's return edge.  Its unwind
/// edge is taken when the opener itself fails, before the guard holds a
/// scope, so no block reached only that way runs inside the bracket.
pub(in crate::front::mir) fn root_bracket_entry(
    llbc: &Llbc,
    body: &Unstructured,
    opener: usize,
) -> Vec<usize> {
    match body.body[opener].term_ref(llbc) {
        Ok(TermKind::Call { target, .. }) if (*target as usize) < body.body.len() => {
            vec![*target as usize]
        }
        _ => block_successors(llbc, body, opener),
    }
}

/// The blocks a bracket's scope is open in: reachable from its opener without
/// passing a drop of the guard or the opener again.  The drop blocks
/// themselves are included.
pub(in crate::front::mir) fn root_bracket_region(
    llbc: &Llbc,
    body: &Unstructured,
    opener: usize,
    scope: usize,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> bit_set::BitSet {
    let mut region = bit_set::BitSet::new();
    let mut work = root_bracket_entry(llbc, body, opener);
    while let Some(bb) = work.pop() {
        if bb == opener || !region.insert(bb) {
            continue;
        }
        if body.body[bb]
            .term_ref(llbc)
            .ok()
            .and_then(|term| term_dropped_local(term, name_of))
            == Some(scope as u64)
        {
            continue;
        }
        work.extend(block_successors(llbc, body, bb));
    }
    region
}

/// Dominator sets over one bracket's region, rooted at its opener.  A block
/// the region reaches only around a backedge through the opener belongs to the
/// next opening, so the edges into the opener are not part of this graph.
pub(in crate::front::mir) fn root_bracket_region_dominators(
    llbc: &Llbc,
    body: &Unstructured,
    opener: usize,
    region: &bit_set::BitSet,
) -> std::collections::HashMap<usize, bit_set::BitSet> {
    let mut nodes: bit_set::BitSet = region.clone();
    nodes.insert(opener);
    let mut preds: std::collections::HashMap<usize, Vec<usize>> =
        nodes.iter().map(|bb| (bb, Vec::new())).collect();
    for bb in nodes.iter() {
        if bb != opener && !region.contains(bb) {
            continue;
        }
        for succ in block_successors(llbc, body, bb) {
            if succ != opener && region.contains(succ) {
                preds.get_mut(&succ).expect("region node").push(bb);
            }
        }
    }
    let mut dom: std::collections::HashMap<usize, bit_set::BitSet> = nodes
        .iter()
        .map(|bb| {
            let set = if bb == opener {
                std::iter::once(opener).collect()
            } else {
                nodes.clone()
            };
            (bb, set)
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for bb in region.iter() {
            if bb == opener {
                continue;
            }
            let mut next: Option<bit_set::BitSet> = None;
            for pred in &preds[&bb] {
                next = Some(match next {
                    None => dom[pred].clone(),
                    Some(mut acc) => {
                        acc.intersect_with(&dom[pred]);
                        acc
                    }
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(bb);
            if next != dom[&bb] {
                dom.insert(bb, next);
                changed = true;
            }
        }
    }
    dom
}

/// Whether a MIR local denotes one immutable value throughout this body.
/// Arguments have an entry definition; call destinations count as writes too.
/// Taking the address of the local could write it indirectly, so that shape
/// cannot stand in for the saved SSA value in `expand_one_pop_roots`.
pub(in crate::front::mir) fn root_pin_value_is_stable(
    llbc: &Llbc,
    body: &Unstructured,
    local: usize,
) -> bool {
    let mut definitions = usize::from(local > 0 && local <= body.locals.arg_count as usize);
    let watched: bit_set::BitSet = std::iter::once(local).collect();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, value)) = stmt.stmt_kind_ref() {
                if matches!(place.kind, PlaceKind::Local(dest) if dest as usize == local) {
                    definitions += 1;
                }
                if matches!(value, Rvalue::Ref { .. } | Rvalue::RawPtr { .. })
                    && mentions_local(stmt.kind_value(), &watched)
                {
                    return false;
                }
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && matches!(call.dest.kind, PlaceKind::Local(dest) if dest as usize == local)
        {
            definitions += 1;
        }
    }
    definitions == 1
}

/// Whether every read of the pinned `local` inside one bracket sees the value
/// its pin published.  This is the SSA view gctransform has: `pop_roots`
/// reloads a livevar into the same variable, so a `mut` local that the
/// bracket's own `get`s write back (the `with_roots!` restore) still denotes
/// one value across the region.  Inside the region the only definitions may
/// be the destinations of this scope's `get`s, directly or through one plain
/// move; outside it the local may be redefined freely, because a path from
/// the pin to a `get` never leaves the region.  Taking the address of the
/// local anywhere still disqualifies it.
pub(in crate::front::mir) fn root_pin_value_is_stable_in_bracket(
    llbc: &Llbc,
    body: &Unstructured,
    local: usize,
    region: &bit_set::BitSet,
    own_gets: &[usize],
) -> bool {
    if root_pin_value_is_stable(llbc, body, local) {
        return true;
    }
    let watched: bit_set::BitSet = std::iter::once(local).collect();
    let get_dest = |bb: usize| match body.body[bb].term_ref(llbc) {
        Ok(TermKind::Call { call, .. }) => match call.dest.kind {
            PlaceKind::Local(dest) => Some(dest as usize),
            _ => None,
        },
        _ => None,
    };
    let own_dests: bit_set::BitSet = own_gets.iter().filter_map(|&bb| get_dest(bb)).collect();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let in_region = region.contains(bb_idx);
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, value)) = stmt.stmt_kind_ref() {
                if matches!(value, Rvalue::Ref { .. } | Rvalue::RawPtr { .. })
                    && mentions_local(stmt.kind_value(), &watched)
                {
                    return false;
                }
                if in_region
                    && matches!(place.kind, PlaceKind::Local(dest) if dest as usize == local)
                {
                    let from_own_get = match value {
                        Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                            matches!(src.kind, PlaceKind::Local(t) if own_dests.contains(t as usize))
                        }
                        _ => false,
                    };
                    if !from_own_get {
                        return false;
                    }
                }
            }
        }
        if in_region && get_dest(bb_idx) == Some(local) && !own_gets.contains(&bb_idx) {
            return false;
        }
    }
    true
}

/// Prove the whole bracket's stack effects, not just uses of its guard.
/// `ShadowStackFrameworkGCTransformer.push_roots/pop_roots` owns every saved
/// slot. A free `pin_root` or a callee can also change our source stack without
/// mentioning the guard, so absence from the receiver map does not prove an
/// empty bracket.  Inside the bracket only the guard's own `base`, `pin_root`
/// and `get` may touch the root stack; every other call, and every function
/// item the bracket hands out as a value, must be one `touches` proves leaves
/// the stack as it found it.  Otherwise erasing the close leaks their pins or
/// changes a later get.
///
/// `pins_are_incoming_slice` is the `W_CData.call` / `cdataobj.py descr_call`
/// shape, and the `W_CTypeFunc._call` `n == 1` shape (`pin_root(args_w[0])`):
/// the pins are the caller's argument array, already live, and the rewritten
/// reads are `getarrayitem` of that array. A callee that opens and closes
/// its own bracket cannot shift those reads, so its `touches` answer does
/// not keep this bracket in the jitcode.
pub(in crate::front::mir) fn root_bracket_stack_effects_are_known(
    llbc: &Llbc,
    body: &Unstructured,
    opener: usize,
    scope: usize,
    aliases: &std::collections::HashMap<usize, usize>,
    free_sites: &std::collections::HashMap<usize, usize>,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    touches: &impl Fn(&RegularCall) -> bool,
    pins_are_incoming_slice: bool,
) -> bool {
    let fn_values_touch = |v: &serde_json::Value| {
        let mut found = false;
        for_each_fn_def_const(llbc, v, &mut |reg| found = found || touches(reg));
        found
    };
    let mut seen = bit_set::BitSet::new();
    let mut work = root_bracket_entry(llbc, body, opener);
    while let Some(bb) = work.pop() {
        if !seen.insert(bb) {
            continue;
        }
        for stmt in &body.body[bb].statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_)) => {
                }
                Ok(StmtKind::Assign(place, _)) if matches!(place.kind, PlaceKind::Local(_)) => {
                    if fn_values_touch(stmt.kind_value()) {
                        return false;
                    }
                }
                // An indirect write could overwrite a saved slot too.
                _ => return false,
            }
        }
        match body.body[bb].term_ref(llbc) {
            Ok(term) if term_dropped_local(term, name_of) == Some(scope as u64) => {
                continue;
            }
            Ok(TermKind::Drop { fn_ptr, .. }) => {
                if touches(fn_ptr) {
                    return false;
                }
            }
            Ok(TermKind::Call { call, .. }) => {
                let reg = match &call.func {
                    CallFunc::Regular(reg) => reg,
                    // `dyn Trait` dispatch: see `RootStackAnalyzer`'s
                    // pointer arm.
                    CallFunc::Dynamic(..) => {
                        work.extend(block_successors(llbc, body, bb));
                        continue;
                    }
                    CallFunc::Unknown => return false,
                };
                let path = name_of(reg);
                let root_api = path
                    .as_deref()
                    .is_some_and(|path| path.split("::").any(|part| part == ROOT_SCOPE_MODULE));
                if root_api {
                    // Another bracket's pin, a free pin, or a nested opener
                    // all move this bracket's slots under it. A free
                    // `pin_root` or `shadow_stack_len` is this guard's own
                    // leaf only when `free_sites` recorded that call for it.
                    let own_leaf = matches!(
                        path.as_deref().and_then(|path| path.rsplit("::").next()),
                        Some("base" | "pin_root" | "pin_roots" | "get")
                    ) && operand_local(call.args.first())
                        .and_then(|receiver| aliases.get(&receiver).copied())
                        == Some(scope)
                        || free_sites.get(&bb) == Some(&scope);
                    if !own_leaf {
                        return false;
                    }
                } else if !pins_are_incoming_slice
                    && (touches(reg)
                        || call.args.iter().any(|op| match op {
                            Operand::Const(c) => {
                                fn_def_of_const(llbc, c).is_some_and(|reg| touches(&reg))
                                    || fn_values_touch(c)
                            }
                            _ => false,
                        }))
                {
                    return false;
                }
            }
            // An overflow or bounds check has nothing to do with the stack.
            // `UnwindResume` is the terminator `FunDecl::unstructured` leaves
            // after dropping rustc cleanup blocks: every `on_unwind` edge
            // names that one block, and it does not itself change the stack.
            // `UndefinedBehavior` is the never-returning sink Charon used to
            // spell `Abort("UndefinedBehavior")`.
            // Return closes the region; 10.04 cleanup stripping still
            // reaches it after the appended UnwindResume is walked
            // (`ShadowStackFrameworkGCTransformer` stack-neutral closer).
            Ok(
                TermKind::Goto { .. }
                | TermKind::Switch { .. }
                | TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::Assert { .. }
                | TermKind::Return
                | TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::UndefinedBehavior,
            ) => {}
            _ => return false,
        }
        work.extend(block_successors(llbc, body, bb));
    }
    true
}

/// Every MIR local a statement or terminator names, however deeply.
pub(in crate::front::mir) fn mentions_local(
    v: &serde_json::Value,
    wanted: &bit_set::BitSet,
) -> bool {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(local) = map.get("Local").and_then(serde_json::Value::as_u64)
                && wanted.contains(local as usize)
            {
                return true;
            }
            map.values().any(|nested| mentions_local(nested, wanted))
        }
        serde_json::Value::Array(items) => {
            items.iter().any(|nested| mentions_local(nested, wanted))
        }
        _ => false,
    }
}

/// The locals `scope.pin_roots(&[a, b, ..])` publishes, in slot order, and
/// the temporaries its slice argument was built through.
///
/// MIR spells the argument as `_arr = [move _a, move _b]`, `_r = &_arr`, a
/// reborrow `_r2 = &(*_r)` and `_s = _r2 as &[_] (Unsize)`.  Each temporary
/// has to be assigned exactly once, and each element a plain local, or the
/// pin is not one this pass can answer a read-back for.
pub(in crate::front::mir) fn pin_roots_slice_values(
    body: &Unstructured,
    slice: usize,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(Vec<usize>, Vec<usize>)> {
    let definition = |local: usize| -> Option<Rvalue> {
        if assigned.get(&local) != Some(&1) {
            return None;
        }
        body.body
            .iter()
            .enumerate()
            .filter(|(bb_idx, _)| forward_blocks[*bb_idx])
            .flat_map(|(_, bb)| bb.statements.iter())
            .find_map(|stmt| {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, value))
                    if matches!(place.kind, PlaceKind::Local(d) if d as usize == local) =>
                {
                    Some(value.clone())
                }
                _ => None,
            }
        })
    };
    let mut temps = Vec::new();
    let mut local = slice;
    loop {
        temps.push(local);
        if temps.len() > 8 {
            return None;
        }
        match definition(local)? {
            Rvalue::UnaryOp(kind, operand)
                if kind
                    .get("Cast")
                    .is_some_and(|cast| cast.get("Unsize").is_some()) =>
            {
                local = operand_local(Some(&operand))?;
            }
            Rvalue::Ref { place, .. } => match place.kind {
                PlaceKind::Local(src) => local = src as usize,
                PlaceKind::Projection(base, ProjectionElem::Atom(elem)) if elem == "Deref" => {
                    match base.kind {
                        PlaceKind::Local(src) => local = src as usize,
                        _ => return None,
                    }
                }
                _ => return None,
            },
            Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                let values = operands
                    .iter()
                    .map(|op| operand_local(Some(op)))
                    .collect::<Option<Vec<_>>>()?;
                return (!values.is_empty()).then_some((values, temps));
            }
            _ => return None,
        }
    }
}

/// `scope.pin_roots(args)` when `args` is an incoming `&[PyObjectRef]`
/// parameter, not a `[move _a, move _b]` aggregate.
///
/// The list is already a live argument the jitcode consumers root, so
/// `get(base + k)` is `getarrayitem` of that list (`W_CData.call` /
/// `cdataobj.py descr_call` passes `args_w` through).
pub(in crate::front::mir) fn pin_roots_incoming_slice(
    body: &Unstructured,
    slice: usize,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, Vec<usize>)> {
    let arg_count = body.locals.arg_count as usize;
    // `assigned` counts forward blocks only. A cleanup assign must not be
    // read as that one definition.
    let definition = |local: usize| -> Option<Rvalue> {
        if assigned.get(&local) != Some(&1) {
            return None;
        }
        body.body
            .iter()
            .enumerate()
            .filter(|(bb_idx, _)| forward_blocks[*bb_idx])
            .flat_map(|(_, bb)| bb.statements.iter())
            .find_map(|stmt| {
                match stmt.stmt_kind_ref() {
                    Ok(StmtKind::Assign(place, value))
                        if matches!(place.kind, PlaceKind::Local(d) if d as usize == local) =>
                    {
                        Some(value.clone())
                    }
                    _ => None,
                }
            })
    };
    let mut temps = Vec::new();
    let mut local = slice;
    loop {
        if temps.len() > 8 {
            return None;
        }
        let assigns = assigned.get(&local).copied().unwrap_or(0);
        if assigns == 0 {
            return (1..=arg_count).contains(&local).then_some((local, temps));
        }
        if assigns != 1 {
            return None;
        }
        temps.push(local);
        match definition(local)? {
            Rvalue::Use(operand, _) => local = operand_local(Some(&operand))?,
            Rvalue::UnaryOp(kind, operand)
                if kind
                    .get("Cast")
                    .is_some_and(|cast| cast.get("Unsize").is_some()) =>
            {
                local = operand_local(Some(&operand))?;
            }
            Rvalue::Ref { place, .. } => match place.kind {
                PlaceKind::Local(src) => local = src as usize,
                PlaceKind::Projection(base, ProjectionElem::Atom(elem)) if elem == "Deref" => {
                    match base.kind {
                        PlaceKind::Local(src) => local = src as usize,
                        _ => return None,
                    }
                }
                _ => return None,
            },
            _ => return None,
        }
    }
}

/// The unique `Assign` of `local`, when it has exactly one.
///
/// The count in `assigned` is forward-reachable only, so the scan is too:
/// an `on_unwind` cleanup block is still in the body.
pub(in crate::front::mir) fn unique_assigned_rvalue(
    body: &Unstructured,
    local: usize,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<Rvalue> {
    if assigned.get(&local) != Some(&1) {
        return None;
    }
    body.body
        .iter()
        .enumerate()
        .filter(|(bb_idx, _)| forward_blocks[*bb_idx])
        .flat_map(|(_, bb)| bb.statements.iter())
        .find_map(|stmt| match stmt.stmt_kind_ref() {
            Ok(StmtKind::Assign(place, value))
                if matches!(place.kind, PlaceKind::Local(d) if d as usize == local) =>
            {
                Some(value.clone())
            }
            _ => None,
        })
}

/// A `usize` constant held in `local`, walking `Use` copies.
pub(in crate::front::mir) fn local_usize_literal(
    llbc: &Llbc,
    body: &Unstructured,
    local: usize,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<u64> {
    let mut local = local;
    for _ in 0..8 {
        match unique_assigned_rvalue(body, local, assigned, forward_blocks)? {
            Rvalue::Use(operand, _) => {
                if let Some(k) = operand_usize_literal(llbc, &operand) {
                    return Some(k);
                }
                local = operand_local(Some(&operand))?;
            }
            _ => return None,
        }
    }
    None
}

/// `offset` of an `Index` projection: a `usize` literal, or a local that
/// holds one.
pub(in crate::front::mir) fn index_offset_usize(
    llbc: &Llbc,
    body: &Unstructured,
    offset: &serde_json::Value,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<u64> {
    let op: Operand = serde_json::from_value(offset.clone()).ok()?;
    if let Some(k) = operand_usize_literal(llbc, &op) {
        return Some(k);
    }
    local_usize_literal(
        llbc,
        body,
        operand_local(Some(&op))?,
        assigned,
        forward_blocks,
    )
}

/// `(*args_w)[k]` / `args_w[k]` when `args_w` is an incoming parameter and
/// `k` is a `usize` constant.
pub(in crate::front::mir) fn place_incoming_slice_elem(
    llbc: &Llbc,
    body: &Unstructured,
    place: &Place,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, u64)> {
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(v)) = &place.kind else {
        return None;
    };
    let index = v.get("Index")?;
    if index.get("from_end").and_then(serde_json::Value::as_bool) == Some(true) {
        return None;
    }
    let k = index_offset_usize(llbc, body, index.get("offset")?, assigned, forward_blocks)?;
    let slice_place = match &inner.kind {
        PlaceKind::Projection(base, ProjectionElem::Atom(atom)) if atom == "Deref" => base.as_ref(),
        _ => inner.as_ref(),
    };
    let PlaceKind::Local(slice) = slice_place.kind else {
        return None;
    };
    pin_roots_incoming_slice(body, slice as usize, assigned, forward_blocks)
        .map(|(slice, _)| (slice, k))
}

/// `pin_root(args_w[k])` / `pin_root(w_arg)` when `w_arg` was loaded from an
/// incoming `&[PyObjectRef]` at a constant index. `W_CTypeFunc._call`
/// (`ctypefunc.py` `_call`) pins one argument that way for `n == 1`; the
/// value is already live in the caller's list, so `get(base)` is
/// `getarrayitem` of that list at `k`. The pin's own local is
/// `StorageDead` before the get (`do_call` discards `pin_root`'s result),
/// so the answer has to be the list element, not that temporary.
pub(in crate::front::mir) fn pin_value_incoming_slice_elem(
    llbc: &Llbc,
    body: &Unstructured,
    value: usize,
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, u64)> {
    let mut local = value;
    for _ in 0..8 {
        {
            let rvalue = unique_assigned_rvalue(body, local, assigned, forward_blocks)?;
            match rvalue {
                Rvalue::Use(operand, _) => {
                    if let Some(place) = operand_place(&operand)
                        && let Some(elem) =
                            place_incoming_slice_elem(llbc, body, place, assigned, forward_blocks)
                    {
                        return Some(elem);
                    }
                    local = operand_local(Some(&operand))?;
                }
                Rvalue::Ref { place, .. } => {
                    if let Some(elem) =
                        place_incoming_slice_elem(llbc, body, &place, assigned, forward_blocks)
                    {
                        return Some(elem);
                    }
                    match place.kind {
                        PlaceKind::Local(src) => local = src as usize,
                        PlaceKind::Projection(base, ProjectionElem::Atom(elem))
                            if elem == "Deref" =>
                        {
                            match base.kind {
                                PlaceKind::Local(src) => local = src as usize,
                                _ => return None,
                            }
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }
    None
}

/// Pins of one guard that are all constant-index loads of the same
/// incoming `&[PyObjectRef]`, in slot order. `None` unless every pin is
/// one such load.
pub(in crate::front::mir) fn incoming_elem_pins(
    llbc: &Llbc,
    body: &Unstructured,
    scope_pins: &[(usize, usize)],
    assigned: &std::collections::HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, Vec<u64>)> {
    if scope_pins.is_empty() {
        return None;
    }
    let mut elems = Vec::with_capacity(scope_pins.len());
    for &(_, value) in scope_pins {
        elems.push(pin_value_incoming_slice_elem(
            llbc,
            body,
            value,
            assigned,
            forward_blocks,
        )?);
    }
    let slice = elems[0].0;
    if elems.iter().any(|&(s, _)| s != slice) {
        return None;
    }
    Some((slice, elems.into_iter().map(|(_, k)| k).collect()))
}

/// Slot `k` for a `base` or `base + k` index, without `shadow_stack_len`.
pub(in crate::front::mir) fn root_read_const_slot(
    index_local: usize,
    scope: usize,
    bases: &std::collections::HashMap<usize, usize>,
    offsets: &std::collections::HashMap<usize, RootSlotSum>,
) -> Option<usize> {
    if bases.get(&index_local) == Some(&scope) {
        return Some(0);
    }
    if let Some(slot) = offsets.get(&index_local)
        && slot.scope == scope
        && slot.from_len.is_none()
    {
        return usize::try_from(slot.k).ok();
    }
    None
}

/// How a closure body reads one shadow-stack slot.
///
/// `Capture` is `shadow_stack_get` of the captured index alone.
/// `CapturePlusArg` is that index plus the closure's `usize` argument.
/// A sum of the capture with itself is neither: both leaves are the same
/// local, and answering that as slot zero would name the wrong pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum RootSlotGetter {
    Capture,
    CapturePlusArg,
}

/// One walk of the index expression inside a getter body.
///
/// `env` and `arg` count leaves. A bit set would treat `capture + capture`
/// as "the capture", which is slot zero of a different sum.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::front::mir) struct RootSlotTrace {
    pub(in crate::front::mir) env: u32,
    pub(in crate::front::mir) arg: u32,
    pub(in crate::front::mir) adds: u32,
    pub(in crate::front::mir) opaque: bool,
}

impl RootSlotTrace {
    pub(in crate::front::mir) fn env_leaf() -> Self {
        Self {
            env: 1,
            ..Self::default()
        }
    }

    pub(in crate::front::mir) fn arg_leaf() -> Self {
        Self {
            arg: 1,
            ..Self::default()
        }
    }

    pub(in crate::front::mir) fn opaque() -> Self {
        Self {
            opaque: true,
            ..Self::default()
        }
    }

    pub(in crate::front::mir) fn join_add(self, other: Self) -> Self {
        Self {
            env: self.env.saturating_add(other.env),
            arg: self.arg.saturating_add(other.arg),
            adds: self.adds.saturating_add(other.adds).saturating_add(1),
            opaque: self.opaque || other.opaque,
        }
    }

    pub(in crate::front::mir) fn classify(self) -> Option<RootSlotGetter> {
        if self.opaque {
            return None;
        }
        match (self.env, self.arg, self.adds) {
            (1, 0, 0) => Some(RootSlotGetter::Capture),
            (1, 1, 1) => Some(RootSlotGetter::CapturePlusArg),
            _ => None,
        }
    }
}

/// Whether `fun_id`'s body is only `shadow_stack_get` of its captured index,
/// or of that index plus its `usize` argument, and that call is the value
/// every normal return yields.
///
/// Charon's closure `call` takes the capture as local 1 and the argument
/// tuple as local 2; local 0 is the return slot. The `shadow_stack_get`
/// writes local 0, and every `Return` reachable from the entry is on that
/// call's normal edge. A second call, a drop, a branch or an rvalue other
/// than a copy or an add is not a read this pass can answer. A body above
/// 16 KiB, or of more than eight blocks, is skipped before that walk.
pub(in crate::front::mir) fn function_is_root_slot_getter(
    llbc: &Llbc,
    fun_id: u64,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<RootSlotGetter> {
    let fd = llbc.fn_by_id(fun_id)?;
    let raw = fd.body.as_ref()?;
    if raw.get().len() > 16 * 1024 {
        return None;
    }
    let body = fd.unstructured()?;
    if body.body.len() > 8 || body.locals.arg_count != 2 {
        return None;
    }
    classify_root_slot_getter_body(llbc, &body, name_of)
}

pub(in crate::front::mir) fn classify_root_slot_getter_body(
    llbc: &Llbc,
    body: &Unstructured,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<RootSlotGetter> {
    let mut defs: std::collections::HashMap<usize, Rvalue> = std::collections::HashMap::new();
    let mut getter_arg: Option<Operand> = None;
    let mut getter_call: Option<(usize, usize)> = None;
    let mut calls = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref().ok()? {
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::Borrowck(_)
                | StmtKind::PlaceMention(_)
                | StmtKind::Assert(_) => {}
                StmtKind::Assign(place, value) => {
                    let PlaceKind::Local(dest) = place.kind else {
                        return None;
                    };
                    let dest = dest as usize;
                    // Locals 0..=2 are the return slot and the two arguments.
                    if dest <= 2 || defs.insert(dest, value.clone()).is_some() {
                        return None;
                    }
                    if !root_slot_rvalue_is_traceable(&value) {
                        return None;
                    }
                }
                StmtKind::Unknown => return None,
            }
        }
        // `BasicBlock::term` asks for an `Llbc` only to decode `Switch`
        // payloads. A getter has none — a `Switch` is rejected below — so
        // the terminator parses straight from its text.
        let term = serde_json::from_str::<TermKind>(bb.terminator.kind.get()).ok()?;
        match term {
            TermKind::Assert { .. }
            | TermKind::Goto { .. }
            | TermKind::Return
            | TermKind::UnwindResume
            | TermKind::UnwindTerminate => {}
            TermKind::Call { call, target, .. } => {
                calls += 1;
                if calls > 1 {
                    return None;
                }
                let CallFunc::Regular(reg) = &call.func else {
                    return None;
                };
                let path = name_of(reg)?;
                let leaf = path.rsplit("::").next().unwrap_or("");
                if leaf != "shadow_stack_get"
                    || !path.split("::").any(|segment| segment == ROOT_SCOPE_MODULE)
                    || call.args.len() != 1
                    || !matches!(call.dest.kind, PlaceKind::Local(0))
                {
                    return None;
                }
                getter_call = Some((bb_idx, target as usize));
                getter_arg = Some(call.args.into_iter().next()?);
            }
            TermKind::Drop { .. }
            | TermKind::Switch { .. }
            | TermKind::Abort(_)
            | TermKind::Panic { .. }
            | TermKind::UndefinedBehavior
            | TermKind::Unknown => return None,
        }
    }
    let (getter_bb, normal_target) = getter_call?;
    if !getter_returns_follow_call(llbc, body, getter_bb, normal_target) {
        return None;
    }
    root_slot_trace_operand(&defs, &getter_arg?, &mut bit_set::BitSet::new()).classify()
}

/// The getter wrote return local 0, and every `Return` the entry reaches is
/// on that call's normal edge. A `Return` only on the unwind edge, or one
/// that skips the call, is not that value.
pub(in crate::front::mir) fn getter_returns_follow_call(
    llbc: &Llbc,
    body: &Unstructured,
    getter_bb: usize,
    normal_target: usize,
) -> bool {
    let n = body.body.len();
    if normal_target >= n {
        return false;
    }
    let dom = block_dominators(llbc, body);
    let mut from_entry = bit_set::BitSet::new();
    let mut work = vec![0usize];
    while let Some(bb) = work.pop() {
        if bb >= n || !from_entry.insert(bb) {
            continue;
        }
        work.extend(block_successors(llbc, body, bb));
    }
    let mut after_normal = bit_set::BitSet::new();
    let mut work = vec![normal_target];
    while let Some(bb) = work.pop() {
        if bb >= n || bb == getter_bb || !after_normal.insert(bb) {
            continue;
        }
        work.extend(
            block_successors(llbc, body, bb)
                .into_iter()
                .filter(|succ| *succ != getter_bb),
        );
    }
    let mut saw_return = false;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if !matches!(block.term_ref(llbc), Ok(TermKind::Return)) {
            continue;
        }
        if !from_entry.contains(bb) {
            continue;
        }
        saw_return = true;
        if !dom[bb].contains(getter_bb) || !after_normal.contains(bb) {
            return false;
        }
    }
    saw_return
}

pub(in crate::front::mir) fn root_slot_rvalue_is_traceable(value: &Rvalue) -> bool {
    match value {
        Rvalue::Use(_, _) => true,
        Rvalue::BinaryOp(op, _, _) => root_slot_sum_is_checked(op).is_some(),
        _ => false,
    }
}

pub(in crate::front::mir) fn projection_is_deref_or_field(elem: &ProjectionElem) -> bool {
    match elem {
        ProjectionElem::Atom(name) => name == "Deref",
        ProjectionElem::Tagged(value) => value.get("Field").is_some(),
    }
}

pub(in crate::front::mir) fn place_root_through_deref_or_field(place: &Place) -> Option<usize> {
    let mut kind = &place.kind;
    loop {
        match kind {
            PlaceKind::Local(local) => return Some(*local as usize),
            PlaceKind::Projection(base, elem) if projection_is_deref_or_field(elem) => {
                kind = &base.kind;
            }
            _ => return None,
        }
    }
}

pub(in crate::front::mir) fn root_slot_trace_operand(
    defs: &std::collections::HashMap<usize, Rvalue>,
    op: &Operand,
    visiting: &mut bit_set::BitSet,
) -> RootSlotTrace {
    match op {
        Operand::Copy(place) | Operand::Move(place) => root_slot_trace_place(defs, place, visiting),
        Operand::Const(_) => RootSlotTrace::opaque(),
    }
}

pub(in crate::front::mir) fn root_slot_trace_place(
    defs: &std::collections::HashMap<usize, Rvalue>,
    place: &Place,
    visiting: &mut bit_set::BitSet,
) -> RootSlotTrace {
    let Some(root) = place_root_through_deref_or_field(place) else {
        return RootSlotTrace::opaque();
    };
    root_slot_trace_local(defs, root, visiting)
}

pub(in crate::front::mir) fn root_slot_trace_local(
    defs: &std::collections::HashMap<usize, Rvalue>,
    local: usize,
    visiting: &mut bit_set::BitSet,
) -> RootSlotTrace {
    if local == 1 {
        return RootSlotTrace::env_leaf();
    }
    if local == 2 {
        return RootSlotTrace::arg_leaf();
    }
    if !visiting.insert(local) {
        return RootSlotTrace::opaque();
    }
    let Some(value) = defs.get(&local) else {
        visiting.remove(local);
        return RootSlotTrace::opaque();
    };
    let trace = match value {
        Rvalue::Use(op, _) => root_slot_trace_operand(defs, op, visiting),
        Rvalue::BinaryOp(op, lhs, rhs) if root_slot_sum_is_checked(op).is_some() => {
            root_slot_trace_operand(defs, lhs, visiting)
                .join_add(root_slot_trace_operand(defs, rhs, visiting))
        }
        _ => RootSlotTrace::opaque(),
    };
    visiting.remove(local);
    trace
}

/// The slot a `shadow_stack_len` names: the pin that runs after every pin
/// already executed.
///
/// `dom[bb]` contains `bb` and the blocks that dominate it, so
/// `dom[len].contains(pin)` means that pin has already run. A pin in the
/// len's own block counts as already executed — both are terminators, so
/// that pin is not after the len — and the read is rejected when no later
/// pin follows. A pin on a side path is rejected too.
pub(in crate::front::mir) fn len_names_next_pin(
    len_bb: usize,
    ordered: &[(usize, usize)],
    dom: &std::collections::HashMap<usize, bit_set::BitSet>,
) -> Option<usize> {
    let len_dom = dom.get(&len_bb)?;
    let mut n = 0usize;
    for (pin_bb, _) in ordered {
        let pin_dom = dom.get(pin_bb)?;
        let pin_dominates_len = len_dom.contains(*pin_bb);
        let len_dominates_pin = pin_dom.contains(len_bb);
        if !pin_dominates_len && !len_dominates_pin {
            return None;
        }
        if pin_dominates_len {
            n += 1;
        }
    }
    let (next_bb, _) = ordered.get(n)?;
    if *next_bb == len_bb || !dom.get(next_bb)?.contains(len_bb) {
        return None;
    }
    Some(n)
}

/// A `base + k`, `len + k`, or `pin_roots_dest + k` index.
///
/// `from_len` is the `shadow_stack_len` block the sum was taken from when it
/// is the latter. `from_run` is the `pin_roots` block whose dest the sum was
/// taken from: that dest names the first pin of that run, not the guard's
/// base, once a later `pin_roots` has run (`zip_two_tuple_next`).
#[derive(Clone, Copy)]
pub(in crate::front::mir) struct RootSlotSum {
    pub(in crate::front::mir) scope: usize,
    pub(in crate::front::mir) k: u64,
    pub(in crate::front::mir) from_len: Option<usize>,
    pub(in crate::front::mir) from_run: Option<usize>,
}

/// The pin index of a `pin_roots` dest (or a copy of one) in `ordered`.
///
/// `pin_roots` answers with the slot its first value took. The first run in
/// a bracket is also the guard's base; a later run is `base + k` for that
/// first value, so a get of its dest must not answer slot 0.
pub(in crate::front::mir) fn root_pin_run_slot(
    index_local: usize,
    scope: usize,
    run_dests: &std::collections::HashMap<usize, (usize, usize)>,
    ordered: &[(usize, usize)],
) -> Option<usize> {
    let &(run_scope, pin_bb) = run_dests.get(&index_local)?;
    if run_scope != scope {
        return None;
    }
    ordered.iter().position(|(bb, _)| *bb == pin_bb)
}

/// The pin index `index_local` names for `scope`, before a closure's constant
/// addend. A base is slot 0, a `base + k` temporary is slot `k`, a
/// `shadow_stack_len` result is the next pin ([`len_names_next_pin`]), and
/// `len + k` is that next pin plus `k`. A later `pin_roots` dest is the first
/// pin of that run, and `that_dest + k` is that pin plus `k`.
pub(in crate::front::mir) fn root_read_base_slot(
    index_local: usize,
    scope: usize,
    bases: &std::collections::HashMap<usize, usize>,
    offsets: &std::collections::HashMap<usize, RootSlotSum>,
    len_index: &std::collections::HashMap<usize, (usize, usize)>,
    run_dests: &std::collections::HashMap<usize, (usize, usize)>,
    ordered: &[(usize, usize)],
    dom: &std::collections::HashMap<usize, bit_set::BitSet>,
) -> Option<usize> {
    if let Some(slot) = root_pin_run_slot(index_local, scope, run_dests, ordered) {
        return Some(slot);
    }
    if bases.get(&index_local) == Some(&scope) {
        return Some(0);
    }
    if let Some(slot) = offsets.get(&index_local)
        && slot.scope == scope
    {
        let extra = usize::try_from(slot.k).ok()?;
        if let Some(len_bb) = slot.from_len {
            return len_names_next_pin(len_bb, ordered, dom)?.checked_add(extra);
        }
        if let Some(run_bb) = slot.from_run {
            return ordered
                .iter()
                .position(|(bb, _)| *bb == run_bb)?
                .checked_add(extra);
        }
        return Some(extra);
    }
    if let Some((len_scope, len_bb)) = len_index.get(&index_local)
        && *len_scope == scope
    {
        return len_names_next_pin(*len_bb, ordered, dom);
    }
    None
}

/// Reads answered from a closure whose body is a [`RootSlotGetter`], and the
/// temporaries that only build that closure.
pub(in crate::front::mir) struct RootGetterReads {
    pub(in crate::front::mir) gets: Vec<(usize, usize, usize, u64)>,
    pub(in crate::front::mir) capture_assigns: bit_set::BitSet,
    pub(in crate::front::mir) capture_temps: std::collections::HashMap<usize, usize>,
    /// Watched index locals a whitelisted capture statement may name.
    pub(in crate::front::mir) admitted_indexes: bit_set::BitSet,
}

pub(in crate::front::mir) fn no_root_getter_reads() -> RootGetterReads {
    RootGetterReads {
        gets: Vec::new(),
        capture_assigns: bit_set::BitSet::new(),
        capture_temps: std::collections::HashMap::new(),
        admitted_indexes: bit_set::BitSet::new(),
    }
}

#[derive(Clone, Copy)]
pub(in crate::front::mir) struct RootGetterIndex<'a> {
    pub(in crate::front::mir) assigned: &'a std::collections::HashMap<usize, usize>,
    pub(in crate::front::mir) bases: &'a std::collections::HashMap<usize, usize>,
    pub(in crate::front::mir) offsets: &'a std::collections::HashMap<usize, RootSlotSum>,
    pub(in crate::front::mir) len_index: &'a std::collections::HashMap<usize, (usize, usize)>,
    pub(in crate::front::mir) candidates: &'a bit_set::BitSet,
    pub(in crate::front::mir) aliases: &'a std::collections::HashMap<usize, usize>,
    pub(in crate::front::mir) call_dests: &'a bit_set::BitSet,
    pub(in crate::front::mir) forward_blocks: &'a [bool],
}

pub(in crate::front::mir) fn single_statement_rvalue(
    body: &Unstructured,
    assigned: &std::collections::HashMap<usize, usize>,
    local: usize,
    forward_blocks: &[bool],
) -> Option<Rvalue> {
    if assigned.get(&local) != Some(&1) {
        return None;
    }
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, value)) = stmt.stmt_kind_ref()
                && matches!(place.kind, PlaceKind::Local(dest) if dest as usize == local)
            {
                return Some(value.clone());
            }
        }
    }
    None
}

pub(in crate::front::mir) fn root_index_scope(
    index: &RootGetterIndex<'_>,
    local: usize,
) -> Option<usize> {
    if let Some(scope) = index.bases.get(&local) {
        return Some(*scope);
    }
    if let Some(slot) = index.offsets.get(&local) {
        return Some(slot.scope);
    }
    if let Some((scope, _)) = index.len_index.get(&local) {
        return Some(*scope);
    }
    None
}

pub(in crate::front::mir) struct PeeledCapture {
    pub(in crate::front::mir) closure: usize,
    pub(in crate::front::mir) carriers: Vec<usize>,
    pub(in crate::front::mir) capture: Operand,
}

pub(in crate::front::mir) enum CapturePeel {
    Ready(PeeledCapture),
    /// Found the closure aggregate, and it does not capture exactly one index.
    BadClosure(usize),
    /// No closure local could be named.
    Unknown,
}

pub(in crate::front::mir) fn peel_closure_capture(
    _llbc: &Llbc,
    body: &Unstructured,
    index: &RootGetterIndex<'_>,
    receiver: &Operand,
) -> CapturePeel {
    let Some(mut local) = operand_local(Some(receiver)) else {
        return CapturePeel::Unknown;
    };
    let mut carriers = Vec::new();
    for _ in 0..4 {
        if index.call_dests.contains(local) || index.assigned.get(&local) != Some(&1) {
            return CapturePeel::Unknown;
        }
        let Some(value) =
            single_statement_rvalue(body, index.assigned, local, index.forward_blocks)
        else {
            return CapturePeel::Unknown;
        };
        carriers.push(local);
        match value {
            Rvalue::Use(inner, _) => {
                let Some(next) = operand_local(Some(&inner)) else {
                    return CapturePeel::Unknown;
                };
                local = next;
            }
            Rvalue::Ref { place, .. } => {
                let PlaceKind::Local(src) = place.kind else {
                    return CapturePeel::Unknown;
                };
                local = src as usize;
            }
            Rvalue::Aggregate(_, mut operands) => {
                if operands.len() == 1
                    && let Some(capture) = operands.pop()
                {
                    return CapturePeel::Ready(PeeledCapture {
                        closure: local,
                        carriers,
                        capture,
                    });
                }
                return CapturePeel::BadClosure(local);
            }
            _ => return CapturePeel::Unknown,
        }
    }
    CapturePeel::Unknown
}

/// `(index local, scope, temporaries walked, not including the index)`.
pub(in crate::front::mir) fn peel_captured_index(
    _llbc: &Llbc,
    body: &Unstructured,
    index: &RootGetterIndex<'_>,
    capture: &Operand,
) -> Option<(usize, usize, Vec<usize>)> {
    let Some(mut local) = operand_local(Some(capture)) else {
        return None;
    };
    let mut carriers = Vec::new();
    for _ in 0..4 {
        if let Some(scope) = root_index_scope(index, local) {
            let has_stmt = index.assigned.get(&local).copied().unwrap_or(0) > 0;
            // A local that is both a statement and a call destination does
            // not name one value.
            if has_stmt && index.call_dests.contains(local) {
                return None;
            }
            return Some((local, scope, carriers));
        }
        if index.call_dests.contains(local) || index.assigned.get(&local) != Some(&1) {
            return None;
        }
        let value = single_statement_rvalue(body, index.assigned, local, index.forward_blocks)?;
        carriers.push(local);
        match value {
            Rvalue::Use(inner, _) => {
                local = operand_local(Some(&inner))?;
            }
            Rvalue::Ref { place, .. } => {
                let PlaceKind::Local(src) = place.kind else {
                    return None;
                };
                local = src as usize;
            }
            _ => return None,
        }
    }
    None
}

pub(in crate::front::mir) fn aggregate_is_empty_unit(
    kind: &serde_json::Value,
    operands: &[Operand],
) -> bool {
    if !operands.is_empty() {
        return false;
    }
    let Some(head) = kind
        .get("Adt")
        .and_then(|adt| adt.as_array())
        .and_then(|arr| arr.first())
    else {
        return false;
    };
    let Some(id) = head
        .as_u64()
        .or_else(|| head.get("id").and_then(serde_json::Value::as_u64))
    else {
        return false;
    };
    id == 0
}

pub(in crate::front::mir) struct PeeledAddend {
    pub(in crate::front::mir) addend: u64,
    pub(in crate::front::mir) carriers: Vec<usize>,
}

pub(in crate::front::mir) fn peel_getter_addend(
    llbc: &Llbc,
    body: &Unstructured,
    index: &RootGetterIndex<'_>,
    kind: RootSlotGetter,
    args: &[Operand],
) -> Option<PeeledAddend> {
    match kind {
        RootSlotGetter::Capture => match args.len() {
            1 => Some(PeeledAddend {
                addend: 0,
                carriers: Vec::new(),
            }),
            2 => Some(PeeledAddend {
                addend: 0,
                carriers: peel_empty_unit(llbc, body, index, &args[1])?,
            }),
            _ => None,
        },
        RootSlotGetter::CapturePlusArg => {
            if args.len() != 2 {
                return None;
            }
            peel_usize_addend(llbc, body, index, &args[1], 0)
        }
    }
}

pub(in crate::front::mir) fn peel_empty_unit(
    _llbc: &Llbc,
    body: &Unstructured,
    index: &RootGetterIndex<'_>,
    op: &Operand,
) -> Option<Vec<usize>> {
    let mut local = operand_local(Some(op))?;
    let mut carriers = Vec::new();
    for _ in 0..4 {
        if index.call_dests.contains(local) || index.assigned.get(&local) != Some(&1) {
            return None;
        }
        let value = single_statement_rvalue(body, index.assigned, local, index.forward_blocks)?;
        carriers.push(local);
        match value {
            Rvalue::Use(inner, _) => {
                local = operand_local(Some(&inner))?;
            }
            Rvalue::Aggregate(kind, operands) if aggregate_is_empty_unit(&kind, &operands) => {
                return Some(carriers);
            }
            _ => return None,
        }
    }
    None
}

pub(in crate::front::mir) fn peel_usize_addend(
    llbc: &Llbc,
    body: &Unstructured,
    index: &RootGetterIndex<'_>,
    op: &Operand,
    depth: u32,
) -> Option<PeeledAddend> {
    if depth > 4 {
        return None;
    }
    if let Some(addend) = operand_usize_literal(llbc, op) {
        return Some(PeeledAddend {
            addend,
            carriers: Vec::new(),
        });
    }
    let local = operand_local(Some(op))?;
    if index.call_dests.contains(local) || index.assigned.get(&local) != Some(&1) {
        return None;
    }
    let value = single_statement_rvalue(body, index.assigned, local, index.forward_blocks)?;
    let mut peeled = match value {
        Rvalue::Use(inner, _) => peel_usize_addend(llbc, body, index, &inner, depth + 1)?,
        Rvalue::Aggregate(_, operands) => {
            let mut operands = operands;
            if operands.len() != 1 {
                return None;
            }
            let inner = operands.pop()?;
            peel_usize_addend(llbc, body, index, &inner, depth + 1)?
        }
        _ => return None,
    };
    peeled.carriers.push(local);
    Some(peeled)
}

pub(in crate::front::mir) fn local_is_root_index_or_guard(
    index: &RootGetterIndex<'_>,
    local: usize,
) -> bool {
    index.candidates.contains(local)
        || index.aliases.contains_key(&local)
        || root_index_scope(index, local).is_some()
}

pub(in crate::front::mir) fn capture_carriers_stay_inside(
    body: &Unstructured,
    carriers: &bit_set::BitSet,
    getter_bbs: &bit_set::BitSet,
    forward_blocks: &[bool],
) -> bool {
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_)) => {
                }
                Ok(StmtKind::Assign(place, _)) if matches!(place.kind, PlaceKind::Local(dest) if carriers.contains(dest as usize)) =>
                    {}
                _ => {
                    if mentions_local(stmt.kind_value(), carriers) {
                        return false;
                    }
                }
            }
        }
        let term = bb.terminator.kind_value();
        let drop_carrier = matches!(
            serde_json::from_str::<TermKind>(bb.terminator.kind.get()),
            Ok(TermKind::Drop { place, .. })
                if matches!(place.kind, PlaceKind::Local(local) if carriers.contains(local as usize))
        );
        let getter_call = getter_bbs.contains(bb_idx)
            && matches!(
                serde_json::from_str::<TermKind>(bb.terminator.kind.get()),
                Ok(TermKind::Call { .. })
            );
        if drop_carrier || getter_call {
            continue;
        }
        if mentions_local(term, carriers) {
            return false;
        }
    }
    true
}

pub(in crate::front::mir) struct ResolvedGetter {
    pub(in crate::front::mir) bb: usize,
    pub(in crate::front::mir) closure: usize,
    pub(in crate::front::mir) scope: usize,
    pub(in crate::front::mir) index: usize,
    pub(in crate::front::mir) addend: u64,
    pub(in crate::front::mir) carriers: Vec<usize>,
}

/// Closure calls whose body is only `shadow_stack_get` of a captured index.
///
/// Every use of one closure must name the same index. The temporaries that
/// build the capture — the borrow, the aggregate, the receiver and the
/// argument tuple — must not be mentioned anywhere else, or the real
/// `shadow_stack_get` would still run and the bracket stays. A call that is
/// a getter but whose closure local cannot be named rejects every closure
/// in the body: an unresolved use might be the one that escapes.
pub(in crate::front::mir) fn record_root_slot_getter_reads(
    llbc: &Llbc,
    body: &Unstructured,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
    index: &RootGetterIndex<'_>,
) -> RootGetterReads {
    let mut call_dests = bit_set::BitSet::new();
    let mut calls: Vec<(usize, CallPayload)> = Vec::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !index.forward_blocks[bb_idx] {
            continue;
        }
        let Ok(term) = bb.term(llbc) else {
            continue;
        };
        let TermKind::Call { call, .. } = term else {
            continue;
        };
        if let PlaceKind::Local(dest) = call.dest.kind {
            call_dests.insert(dest as usize);
        }
        calls.push((bb_idx, call));
    }
    let index = RootGetterIndex {
        call_dests: &call_dests,
        ..*index
    };
    // The caller's `call_dests` slot is empty; the set above is the one the
    // peels consult. Rebuild so the borrow lives for the rest of this call.
    let mut getter_cache: std::collections::HashMap<u64, Option<RootSlotGetter>> =
        std::collections::HashMap::new();
    let mut resolved: Vec<ResolvedGetter> = Vec::new();
    let mut poisoned = bit_set::BitSet::new();
    for (bb_idx, call) in &calls {
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let Some(fun_id) = regular_call_fun_decl_id(&reg.kind) else {
            continue;
        };
        let kind = *getter_cache
            .entry(fun_id)
            .or_insert_with(|| function_is_root_slot_getter(llbc, fun_id, name_of));
        let Some(kind) = kind else {
            continue;
        };
        let Some(receiver) = call.args.first() else {
            return no_root_getter_reads();
        };
        let peeled = match peel_closure_capture(llbc, body, &index, receiver) {
            CapturePeel::Unknown => return no_root_getter_reads(),
            CapturePeel::BadClosure(closure) => {
                poisoned.insert(closure);
                continue;
            }
            CapturePeel::Ready(peeled) => peeled,
        };
        let Some((index_local, scope, index_carriers)) =
            peel_captured_index(llbc, body, &index, &peeled.capture)
        else {
            poisoned.insert(peeled.closure);
            continue;
        };
        if !index.candidates.contains(scope) {
            poisoned.insert(peeled.closure);
            continue;
        }
        let Some(addend) = peel_getter_addend(llbc, body, &index, kind, &call.args) else {
            poisoned.insert(peeled.closure);
            continue;
        };
        let mut carriers = peeled.carriers;
        carriers.extend(index_carriers);
        carriers.extend(addend.carriers);
        resolved.push(ResolvedGetter {
            bb: *bb_idx,
            closure: peeled.closure,
            scope,
            index: index_local,
            addend: addend.addend,
            carriers,
        });
    }

    let mut by_closure: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (ord, use_) in resolved.iter().enumerate() {
        by_closure.entry(use_.closure).or_default().push(ord);
    }
    let mut reads = no_root_getter_reads();
    for (closure, idxs) in by_closure {
        if poisoned.contains(closure) {
            continue;
        }
        let first = &resolved[idxs[0]];
        if idxs
            .iter()
            .any(|&ord| resolved[ord].scope != first.scope || resolved[ord].index != first.index)
        {
            continue;
        }
        let mut carriers = bit_set::BitSet::new();
        let mut getter_bbs = bit_set::BitSet::new();
        let mut bad_carrier = false;
        for &ord in &idxs {
            getter_bbs.insert(resolved[ord].bb);
            for local in &resolved[ord].carriers {
                if *local == first.index || local_is_root_index_or_guard(&index, *local) {
                    bad_carrier = true;
                    break;
                }
                carriers.insert(*local);
            }
        }
        if bad_carrier
            || !capture_carriers_stay_inside(body, &carriers, &getter_bbs, index.forward_blocks)
        {
            continue;
        }
        for local in carriers.iter() {
            if let Some(previous) = reads.capture_temps.insert(local, first.scope)
                && previous != first.scope
            {
                return no_root_getter_reads();
            }
            reads.capture_assigns.insert(local);
        }
        if !index.candidates.contains(first.index) {
            reads.admitted_indexes.insert(first.index);
        }
        for &ord in &idxs {
            let use_ = &resolved[ord];
            reads
                .gets
                .push((use_.bb, use_.scope, use_.index, use_.addend));
        }
    }
    reads
}

/// Every local body of `llbc` that can change the root stack, by path.
///
/// Published to the artefacts linked after this one, so a call into this
/// crate from there is answered by its body instead of being assumed to
/// touch the stack.  Paths that name several bodies (`<Impl>::new`) are
/// listed if any of them touches.
/// One struct body copied onto an importing crate's opaque `TypeDecl`.
///
/// `ty` is a type node whose ADT ids are `{"Path": "<name_path>"}` and
/// whose dedup indirections have been expanded against the defining crate.
/// [`apply_published_structs`] remaps those paths onto the destination's
/// def ids and installs the fields as `TypeDeclKind::Struct`.
#[derive(Clone)]
pub struct PublishedStructBody {
    pub path: String,
    pub fields: Vec<PublishedStructField>,
}

/// One field of [`PublishedStructBody`].
#[derive(Clone)]
pub struct PublishedStructField {
    pub name: Option<String>,
    pub ty: serde_json::Value,
}

/// Struct bodies of this artefact, keyed by the instantiated type
/// identity: `name_path` plus `ItemMeta::instantiation` arguments.
/// `name_path` drops `Instantiated` segments; field types live on the
/// concrete type (`lltype.Struct.__init__`). An importing crate's
/// `Opaque` view of the same identity receives these fields so
/// [`Lowering::resolve_adt_field`]'s struct arm names them.
pub fn harvest_published_structs(llbc: &Llbc) -> Vec<PublishedStructBody> {
    let mut rows = Vec::new();
    for td in llbc.iter_type_decls() {
        let TypeDeclKind::Struct(fields) = &td.kind else {
            continue;
        };
        if fields.is_empty() {
            continue;
        }
        let portable: Vec<PublishedStructField> = fields
            .iter()
            .map(|field| PublishedStructField {
                name: field.name.clone(),
                ty: expand_portable_tyref(llbc, &field.ty),
            })
            .collect();
        rows.push(PublishedStructBody {
            path: type_decl_identity(llbc, td),
            fields: portable,
        });
    }
    rows
}

/// Copy `rows` onto every `Opaque` or `Unknown` type decl whose
/// instantiated identity matches. A crate-stripped identity is used only
/// when it names exactly one published row; an ambiguous suffix stays
/// opaque. Already-unprefixed identities count in that uniqueness
/// bucket (`LowLevelType.__eq__` compares the whole type). Crate
/// stripping applies to the declaration path, not to `::` inside
/// generic arguments. An existing `Struct` is left alone. A path that does not
/// resolve to a destination decl stays a `{"Path": ...}` id.
pub fn apply_published_structs(llbc: &mut Llbc, rows: &[PublishedStructBody]) {
    if rows.is_empty() {
        return;
    }
    let dest_identities: Vec<(u64, String)> = llbc
        .iter_type_decls()
        .map(|td| (td.def_id, type_decl_identity(llbc, td)))
        .collect();
    let mut exact: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut stripped_ids: std::collections::HashMap<String, Vec<u64>> =
        std::collections::HashMap::new();
    for (id, key) in &dest_identities {
        exact.entry(key.clone()).or_insert(*id);
        stripped_ids
            .entry(strip_identity_crate_prefix(key))
            .or_default()
            .push(*id);
    }
    let stripped_unique = unique_stripped_values(stripped_ids);
    let mut by_identity: std::collections::HashMap<&str, &PublishedStructBody> =
        std::collections::HashMap::new();
    for row in rows {
        by_identity.entry(row.path.as_str()).or_insert(row);
    }
    let mut stripped_rows: std::collections::HashMap<String, Vec<&PublishedStructBody>> =
        std::collections::HashMap::new();
    for row in rows {
        stripped_rows
            .entry(strip_identity_crate_prefix(&row.path))
            .or_default()
            .push(row);
    }
    let stripped_unique_rows = unique_stripped_values(stripped_rows);
    let dest_key_by_id: std::collections::HashMap<u64, String> =
        dest_identities.into_iter().collect();
    let mut updates: std::collections::HashMap<u64, Vec<majit_charon_reader::ullbc::FieldDecl>> =
        std::collections::HashMap::new();
    for td in llbc.iter_type_decls() {
        if !matches!(td.kind, TypeDeclKind::Opaque | TypeDeclKind::Unknown) {
            continue;
        }
        let Some(key) = dest_key_by_id.get(&td.def_id) else {
            continue;
        };
        let row = by_identity.get(key.as_str()).copied().or_else(|| {
            let stripped = strip_identity_crate_prefix(key);
            (stripped != *key)
                .then(|| stripped_unique_rows.get(&stripped).copied())
                .flatten()
        });
        let Some(row) = row else {
            continue;
        };
        let fields = row
            .fields
            .iter()
            .map(|field| {
                let mut ty = field.ty.clone();
                remap_portable_type_ids(&mut ty, llbc, &exact, &stripped_unique);
                majit_charon_reader::ullbc::FieldDecl {
                    name: field.name.clone(),
                    ty: TyRef::Other(ty),
                    attr_info: None,
                }
            })
            .collect();
        updates.insert(td.def_id, fields);
    }
    for td in llbc.type_decls_mut() {
        let Some(fields) = updates.remove(&td.def_id) else {
            continue;
        };
        td.published_struct_copy = true;
        td.kind = TypeDeclKind::Struct(fields);
    }
}

pub(in crate::front::mir) fn unique_stripped_values<T>(
    grouped: std::collections::HashMap<String, Vec<T>>,
) -> std::collections::HashMap<String, T> {
    grouped
        .into_iter()
        .filter_map(|(key, mut values)| {
            if values.len() == 1 {
                Some((key, values.pop().expect("len == 1")))
            } else {
                None
            }
        })
        .collect()
}

pub(in crate::front::mir) fn type_decl_identity(llbc: &Llbc, td: &TypeDecl) -> String {
    type_path_identity(
        llbc,
        td.item_meta.name_path_str(),
        td.item_meta.instantiation(),
    )
}

pub(in crate::front::mir) fn type_path_identity(
    llbc: &Llbc,
    path: &str,
    instantiation: Option<&serde_json::Value>,
) -> String {
    let (types, consts) = identity_instantiation(llbc, instantiation);
    format_instantiated_identity(path, &types, &consts)
}

pub(in crate::front::mir) fn format_instantiated_identity(
    name: &str,
    types: &str,
    consts: &str,
) -> String {
    if types.is_empty() && consts.is_empty() {
        name.to_string()
    } else if consts.is_empty() {
        format!("{name}<{types}>")
    } else {
        format!("{name}<{types}|{consts}>")
    }
}

pub(in crate::front::mir) fn lookup_published_type_path(
    exact: &std::collections::HashMap<String, u64>,
    stripped_unique: &std::collections::HashMap<String, u64>,
    key: &str,
) -> Option<u64> {
    if let Some(id) = exact.get(key) {
        return Some(*id);
    }
    let stripped = strip_identity_crate_prefix(key);
    if stripped != key {
        return stripped_unique.get(&stripped).copied();
    }
    None
}

/// Drop a leading crate ident from an instantiated identity.
///
/// `strip_crate_prefix` is for `name_path`. Instantiated identity is
/// `name<types>` / `name<types|consts>` (`format_instantiated_identity`);
/// a `::` inside those arguments is part of an argument, not a crate
/// segment. `LowLevelType.__eq__` compares the type class and attributes.
pub(in crate::front::mir) fn strip_identity_crate_prefix(identity: &str) -> String {
    match identity.split_once("::") {
        Some((head, rest)) if !head.contains('<') => rest.to_string(),
        _ => identity.to_string(),
    }
}

/// Identities of local bodies that open a bracket and return its guard,
/// keeping only an identity that this crate publishes once.
///
/// `<Impl>::new` is not a unique path. The rendered name substitutes the
/// impl's `Self` ADT, and the signature, instantiation, and span separate
/// two constructors that still share that ADT and leaf.
pub fn harvest_scope_owning_identities(llbc: &Llbc) -> Vec<String> {
    let root_state = RootStackState::new(llbc);
    let analyzer = RootStackAnalyzer::new(llbc, &root_state);
    let mut identities: Vec<String> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some() && analyzer.fn_returns_owned_scope(fd.def_id))
        .map(|fd| scope_owner_identity(llbc, fd))
        .collect();
    identities.sort();
    let mut unique = Vec::new();
    let mut index = 0;
    while index < identities.len() {
        let mut next = index + 1;
        while next < identities.len() && identities[next] == identities[index] {
            next += 1;
        }
        if next == index + 1 {
            unique.push(std::mem::take(&mut identities[index]));
        }
        index = next;
    }
    unique
}

/// Set [`FunDecl::returns_owned_scope`] on every body-less foreign
/// declaration whose [`scope_owner_identity`] is in `identities`.
/// `identities` is sorted. A defining-crate duplicate was already dropped,
/// so every foreign copy of one published constructor is the guard.
pub fn mark_foreign_scope_owners(llbc: &Llbc, identities: &[String]) {
    if identities.is_empty() {
        return;
    }
    for fd in llbc.iter_fun_decls() {
        if fd.item_meta.is_local || fd.unstructured().is_some() {
            continue;
        }
        let identity = scope_owner_identity(llbc, fd);
        if identities.binary_search(&identity).is_ok() {
            fd.returns_owned_scope
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// `{name}|{inputs}|{output}|{span}|{type args}|{const args}`.
///
/// Region ids and trait-ref ids are crate-local and are not part of the
/// string. An inherent or trait impl's `Self` is that ADT's `name_path`.
pub fn scope_owner_identity(llbc: &Llbc, fd: &FunDecl) -> String {
    let name = scope_owner_rendered_name(llbc, fd);
    let inputs = fd
        .signature
        .inputs
        .iter()
        .map(|ty| identity_ty(llbc, ty))
        .collect::<Vec<_>>()
        .join(",");
    let output = identity_ty(llbc, &fd.signature.output);
    let span = identity_span(llbc, fd);
    let (type_args, const_args) = identity_instantiation(llbc, fd.item_meta.instantiation());
    format!("{name}|{inputs}|{output}|{span}|{type_args}|{const_args}")
}

pub(in crate::front::mir) fn scope_owner_rendered_name(llbc: &Llbc, fd: &FunDecl) -> String {
    let owner = impl_owner_path(llbc, fd);
    let mut out = String::new();
    for seg in fd.item_meta.template_name() {
        if !out.is_empty() {
            out.push_str("::");
        }
        match seg {
            NameSeg::Ident {
                ident: (name, disambiguator),
            } => {
                out.push_str(name);
                if name == "closure" && *disambiguator > 0 {
                    out.push('#');
                    out.push_str(&disambiguator.to_string());
                }
            }
            NameSeg::Other(value) => {
                if value.as_object().and_then(|map| map.get("Impl")).is_some() {
                    match owner.as_deref() {
                        Some(path) => out.push_str(path),
                        None => out.push_str("<Impl>"),
                    }
                    continue;
                }
                if let Some(label) = majit_charon_reader::ullbc::builtin_path_label(value) {
                    out.push_str(&label);
                    continue;
                }
                let label = value
                    .as_object()
                    .and_then(|map| map.keys().next().cloned())
                    .unwrap_or_else(|| "?".into());
                out.push('<');
                out.push_str(&label);
                out.push('>');
            }
        }
    }
    out
}

pub(in crate::front::mir) fn identity_span(llbc: &Llbc, fd: &FunDecl) -> String {
    let Some(data) = llbc.span_data(&fd.item_meta.span) else {
        return "?:?:?".to_string();
    };
    let file = llbc.file_path(data.file_id).unwrap_or("?");
    format!("{file}:{}:{}", data.beg.line, data.beg.col)
}

pub(in crate::front::mir) fn identity_instantiation(
    llbc: &Llbc,
    instantiation: Option<&serde_json::Value>,
) -> (String, String) {
    let Some(instantiation) = instantiation else {
        return (String::new(), String::new());
    };
    let types = instantiation
        .get("types")
        .and_then(serde_json::Value::as_array)
        .map(|args| {
            args.iter()
                .map(|ty| identity_type_value(llbc, ty, 0))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let consts = instantiation
        .get("const_generics")
        .and_then(serde_json::Value::as_array)
        .map(|args| {
            args.iter()
                .map(|value| identity_const(llbc, value))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    (types, consts)
}

pub(in crate::front::mir) fn identity_ty(llbc: &Llbc, ty: &TyRef) -> String {
    match tyref_node(ty, llbc) {
        Some(value) => identity_type_value(llbc, value, 0),
        None => "?".to_string(),
    }
}

pub(in crate::front::mir) fn identity_type_value(
    llbc: &Llbc,
    value: &serde_json::Value,
    depth: usize,
) -> String {
    if depth > 32 {
        return "?".to_string();
    }
    let Some(obj) = value.as_object() else {
        return value.as_str().unwrap_or("?").to_string();
    };
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return match llbc.dedup_body(id) {
            Some(body) => identity_type_value(llbc, body, depth + 1),
            None => "?".to_string(),
        };
    }
    if let Some(pair) = obj.get("Value").and_then(serde_json::Value::as_array)
        && pair.len() == 2
    {
        return identity_type_value(llbc, &pair[1], depth + 1);
    }
    if let Some(literal) = obj.get("Scalar") {
        return charon_literal_to_ast_string(literal);
    }
    if let Some(reference) = obj.get("Ref").and_then(serde_json::Value::as_array)
        && reference.len() >= 2
    {
        let inner = identity_type_value(llbc, &reference[1], depth + 1);
        let mutable = reference
            .get(2)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind.eq_ignore_ascii_case("Mut"));
        return if mutable {
            format!("&mut {inner}")
        } else {
            format!("&{inner}")
        };
    }
    if let Some(pointer) = obj.get("RawPtr").and_then(serde_json::Value::as_array)
        && !pointer.is_empty()
    {
        let inner = identity_type_value(llbc, &pointer[0], depth + 1);
        let mutable = pointer
            .get(1)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| kind.eq_ignore_ascii_case("Mut"));
        return if mutable {
            format!("*mut {inner}")
        } else {
            format!("*const {inner}")
        };
    }
    if let Some(adt) = obj.get("Adt").and_then(serde_json::Value::as_object) {
        return identity_adt(llbc, adt, depth);
    }
    if let Some(type_var) = obj.get("TypeVar") {
        if let Some(index) = type_var.get("Free").and_then(serde_json::Value::as_u64) {
            return format!("TypeVar#{index}");
        }
        if let Some(bound) = type_var.get("Bound").and_then(serde_json::Value::as_array) {
            let index = bound
                .get(1)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            return format!("TypeVar#{index}");
        }
        return "TypeVar".to_string();
    }
    // `lltype.FuncType` stores ARGS and RESULT; `LowLevelType.__eq__`
    // compares that dict, so two function-pointer instantiations stay
    // distinct types.
    if let Some(fnptr) = obj.get("FnPtr") {
        return identity_fn_ptr(llbc, fnptr, depth);
    }
    if let Some(array) = obj.get("Array").and_then(serde_json::Value::as_array)
        && !array.is_empty()
    {
        let elem = identity_type_value(llbc, &array[0], depth + 1);
        let len = array
            .get(1)
            .map(|len| identity_const(llbc, len))
            .unwrap_or_else(|| "?".to_string());
        return format!("[{elem};{len}]");
    }
    if let Some(elem) = obj
        .get("Slice")
        .and_then(serde_json::Value::as_array)
        .and_then(|array| array.first())
    {
        return format!("[{}]", identity_type_value(llbc, elem, depth + 1));
    }
    obj.keys().next().cloned().unwrap_or_else(|| "?".into())
}

pub(in crate::front::mir) fn identity_adt(
    llbc: &Llbc,
    adt: &serde_json::Map<String, serde_json::Value>,
    depth: usize,
) -> String {
    let name = if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Tuple") {
        "Tuple".to_string()
    } else if let Some(builtin) = adt.get("builtin").filter(|builtin| !builtin.is_null()) {
        builtin
            .as_str()
            .or_else(|| {
                builtin
                    .as_object()
                    .and_then(|map| map.keys().next().map(String::as_str))
            })
            .unwrap_or("?")
            .to_string()
    } else if let Some(id) = adt.get("id").and_then(serde_json::Value::as_u64) {
        llbc.type_by_id(id)
            .map(|td| td.item_meta.name_path())
            .unwrap_or_else(|| "?".to_string())
    } else if let Some(path) = adt
        .get("id")
        .and_then(|id| id.get("Path"))
        .and_then(serde_json::Value::as_str)
    {
        path.to_string()
    } else {
        "?".to_string()
    };
    // Use-site generics, else the TypeDecl Instantiated segment
    // (`type_decl_ref_generics`). Const args belong on the concrete type
    // the same way `types` do (`lltype.Struct.__init__`).
    let generics = type_decl_ref_generics(adt, llbc);
    let args = generics
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        .map(|args| {
            args.iter()
                .map(|ty| identity_type_value(llbc, ty, depth + 1))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let consts = generics
        .and_then(|generics| generics.get("const_generics"))
        .and_then(serde_json::Value::as_array)
        .map(|args| {
            args.iter()
                .map(|value| identity_const(llbc, value))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    format_instantiated_identity(&name, &args.join(","), &consts.join(","))
}

pub(in crate::front::mir) fn identity_fn_ptr(
    llbc: &Llbc,
    fnptr: &serde_json::Value,
    depth: usize,
) -> String {
    let Some(sig) = fnptr.get("skip_binder").unwrap_or(fnptr).as_object() else {
        return "fn".to_string();
    };
    let inputs = sig
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .map(|args| {
            args.iter()
                .map(|ty| identity_type_value(llbc, ty, depth + 1))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let output = sig
        .get("output")
        .map(|ty| identity_type_value(llbc, ty, depth + 1))
        .unwrap_or_else(|| "()".to_string());
    if sig.get("is_unsafe").and_then(serde_json::Value::as_bool) == Some(true) {
        format!("unsafe fn({inputs}) -> {output}")
    } else {
        format!("fn({inputs}) -> {output}")
    }
}

pub(in crate::front::mir) fn identity_const(llbc: &Llbc, value: &serde_json::Value) -> String {
    if let Some(id) = value
        .get("Deduplicated")
        .and_then(serde_json::Value::as_u64)
    {
        return match llbc.dedup_const_body(id) {
            Some(body) => const_literal_text(body),
            None => "?".to_string(),
        };
    }
    if let Some(pair) = value.get("Value").and_then(serde_json::Value::as_array)
        && pair.len() == 2
    {
        return const_literal_text(&pair[1]);
    }
    if value.as_array().is_some() {
        return const_literal_text(value);
    }
    "?".to_string()
}

pub(in crate::front::mir) fn const_literal_text(body: &serde_json::Value) -> String {
    let Some(literal) = body.as_array().and_then(|pair| pair.first()) else {
        return "?".to_string();
    };
    if let Some(boolean) = literal.get("Bool").and_then(serde_json::Value::as_bool) {
        return boolean.to_string();
    }
    // A `char` const generic is a `ConstantExpr` literal
    // (`Llbc::const_expr_literal`). `lltype.Char` / `lltype.UniChar` are
    // distinct primitives; rendering Charon's `Char` as `?` collapsed
    // every such instantiation onto one identity.
    if let Some(ch) = literal.get("Char") {
        return match ch {
            serde_json::Value::String(s) => format!("char:{s}"),
            serde_json::Value::Number(n) => format!("char:{n}"),
            _ => "char:?".to_string(),
        };
    }
    literal
        .get("Integer")
        .and_then(serde_json::Value::as_object)
        .and_then(|int| int.values().next())
        .and_then(serde_json::Value::as_array)
        .and_then(|parts| parts.get(1))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?")
        .to_string()
}

pub(in crate::front::mir) fn expand_portable_tyref(llbc: &Llbc, ty: &TyRef) -> serde_json::Value {
    match ty {
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(body) => expand_portable_type(llbc, body, 0),
            None => serde_json::json!({"Unresolved": true}),
        },
        TyRef::Inline { value: (_, value) } | TyRef::Other(value) => {
            expand_portable_type(llbc, value, 0)
        }
    }
}

pub(in crate::front::mir) fn expand_portable_type(
    llbc: &Llbc,
    value: &serde_json::Value,
    depth: usize,
) -> serde_json::Value {
    if depth > 32 {
        return serde_json::json!({"Unresolved": true});
    }
    let Some(obj) = value.as_object() else {
        return value.clone();
    };
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return match llbc.dedup_body(id) {
            Some(body) => expand_portable_type(llbc, body, depth + 1),
            None => serde_json::json!({"Unresolved": true}),
        };
    }
    if let Some(pair) = obj.get("Value").and_then(serde_json::Value::as_array)
        && pair.len() == 2
    {
        return expand_portable_type(llbc, &pair[1], depth + 1);
    }
    let mut out = value.clone();
    let Some(map) = out.as_object_mut() else {
        return out;
    };
    if map.contains_key("Adt") {
        let adt = map.get_mut("Adt").expect("Adt");
        expand_portable_adt(llbc, adt, depth);
    }
    if let Some(array) = map
        .get_mut("Array")
        .and_then(serde_json::Value::as_array_mut)
        && array.len() >= 2
    {
        array[0] = expand_portable_type(llbc, &array[0], depth + 1);
        array[1] = expand_portable_const(llbc, &array[1], depth + 1);
        if array.len() >= 3 {
            array[2] = serde_json::Value::Null;
        }
    }
    if let Some(slice) = map
        .get_mut("Slice")
        .and_then(serde_json::Value::as_array_mut)
        && !slice.is_empty()
    {
        slice[0] = expand_portable_type(llbc, &slice[0], depth + 1);
        if slice.len() >= 2 {
            slice[1] = serde_json::Value::Null;
        }
    }
    if let Some(reference) = map.get_mut("Ref").and_then(serde_json::Value::as_array_mut)
        && reference.len() >= 2
    {
        reference[0] = serde_json::json!({"Erased": true});
        reference[1] = expand_portable_type(llbc, &reference[1], depth + 1);
    }
    if let Some(pointer) = map
        .get_mut("RawPtr")
        .and_then(serde_json::Value::as_array_mut)
        && !pointer.is_empty()
    {
        pointer[0] = expand_portable_type(llbc, &pointer[0], depth + 1);
    }
    out
}

pub(in crate::front::mir) fn expand_portable_adt(
    llbc: &Llbc,
    adt: &mut serde_json::Value,
    depth: usize,
) {
    let Some(obj) = adt.as_object_mut() else {
        return;
    };
    let decl_types = obj
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| {
            let td = llbc.type_by_id(id)?;
            let path = td.item_meta.name_path();
            let instantiation = td.item_meta.instantiation().cloned();
            Some((path, instantiation))
        });
    if let Some(id) = obj.get("id").and_then(serde_json::Value::as_u64) {
        match &decl_types {
            Some((path, _)) => {
                obj.insert("id".into(), serde_json::json!({"Path": path}));
            }
            None => {
                let _ = id;
                obj.insert("id".into(), serde_json::json!({"Unresolved": true}));
            }
        }
    }
    if let Some((_, instantiation)) = decl_types {
        fill_adt_generics_from_instantiation(obj, instantiation.as_ref());
    }
    let Some(generics) = obj
        .get_mut("generics")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    if let Some(types) = generics
        .get_mut("types")
        .and_then(serde_json::Value::as_array_mut)
    {
        for ty in types.iter_mut() {
            *ty = expand_portable_type(llbc, ty, depth + 1);
        }
    }
    if let Some(consts) = generics
        .get_mut("const_generics")
        .and_then(serde_json::Value::as_array_mut)
    {
        for value in consts.iter_mut() {
            *value = expand_portable_const(llbc, value, depth + 1);
        }
    }
    if let Some(regions) = generics
        .get_mut("regions")
        .and_then(serde_json::Value::as_array_mut)
    {
        for region in regions.iter_mut() {
            *region = serde_json::json!({"Erased": true});
        }
    }
    if let Some(trait_refs) = generics
        .get_mut("trait_refs")
        .and_then(serde_json::Value::as_array_mut)
    {
        for trait_ref in trait_refs.iter_mut() {
            *trait_ref = serde_json::Value::Null;
        }
    }
}

pub(in crate::front::mir) fn fill_adt_generics_from_instantiation(
    adt: &mut serde_json::Map<String, serde_json::Value>,
    instantiation: Option<&serde_json::Value>,
) {
    let Some(instantiation) = instantiation else {
        return;
    };
    let generics = adt
        .entry("generics")
        .or_insert_with(|| serde_json::json!({}));
    let Some(generics) = generics.as_object_mut() else {
        return;
    };
    let types_empty = generics
        .get("types")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|types| types.is_empty());
    if types_empty
        && let Some(types) = instantiation
            .get("types")
            .filter(|types| types.as_array().is_some_and(|types| !types.is_empty()))
    {
        generics.insert("types".into(), types.clone());
    }
    let consts_empty = generics
        .get("const_generics")
        .and_then(serde_json::Value::as_array)
        .is_none_or(|consts| consts.is_empty());
    if consts_empty
        && let Some(consts) = instantiation
            .get("const_generics")
            .filter(|consts| consts.as_array().is_some_and(|consts| !consts.is_empty()))
    {
        generics.insert("const_generics".into(), consts.clone());
    }
}

pub(in crate::front::mir) fn expand_portable_const(
    llbc: &Llbc,
    value: &serde_json::Value,
    depth: usize,
) -> serde_json::Value {
    if depth > 32 {
        return serde_json::json!({"Unresolved": true});
    }
    if let Some(id) = value
        .get("Deduplicated")
        .and_then(serde_json::Value::as_u64)
    {
        return match llbc.dedup_const_body(id) {
            Some(body) => {
                serde_json::json!({"Value": [0, expand_const_body(llbc, body, depth + 1)]})
            }
            None => serde_json::json!({"Unresolved": true}),
        };
    }
    if let Some(pair) = value.get("Value").and_then(serde_json::Value::as_array)
        && pair.len() == 2
    {
        return serde_json::json!({"Value": [0, expand_const_body(llbc, &pair[1], depth + 1)]});
    }
    if value.as_array().is_some_and(|pair| pair.len() == 2) {
        return serde_json::json!({"Value": [0, expand_const_body(llbc, value, depth + 1)]});
    }
    value.clone()
}

pub(in crate::front::mir) fn expand_const_body(
    llbc: &Llbc,
    body: &serde_json::Value,
    depth: usize,
) -> serde_json::Value {
    let Some(pair) = body.as_array().filter(|pair| pair.len() == 2) else {
        return body.clone();
    };
    serde_json::json!([
        pair[0].clone(),
        expand_portable_type(llbc, &pair[1], depth + 1)
    ])
}

pub(in crate::front::mir) fn remap_portable_type_ids(
    value: &mut serde_json::Value,
    llbc: &Llbc,
    exact: &std::collections::HashMap<String, u64>,
    stripped_unique: &std::collections::HashMap<String, u64>,
) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                remap_portable_type_ids(item, llbc, exact, stripped_unique);
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(adt) = map
                .get_mut("Adt")
                .and_then(serde_json::Value::as_object_mut)
            {
                let path = adt
                    .get("id")
                    .and_then(|id| id.get("Path"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                let generics = adt.get("generics").cloned();
                if let Some(path) = path {
                    // Instantiated identity first (`lltype.Struct.__init__`
                    // names the concrete type). A `monomorphize:false`
                    // template is keyed by bare `name_path`
                    // (`ItemMeta::name_path` drops Instantiated); the
                    // use-site TypeDeclRef still carries generics
                    // (`type_decl_ref_generics`). `type_decl_ref_adt_id`
                    // reads only a numeric id.
                    let key = type_path_identity(llbc, &path, generics.as_ref());
                    let mut def_id = lookup_published_type_path(exact, stripped_unique, &key);
                    if def_id.is_none() && key != path {
                        def_id = lookup_published_type_path(exact, stripped_unique, &path);
                    }
                    if let Some(def_id) = def_id {
                        adt.insert("id".into(), serde_json::json!(def_id));
                    }
                }
            }
            for child in map.values_mut() {
                remap_portable_type_ids(child, llbc, exact, stripped_unique);
            }
        }
        _ => {}
    }
}

/// Whether the body at `id` returns a root-bracket guard, including a
/// body-less declaration answered from [`FunDecl::returns_owned_scope`].
pub fn fn_returns_owned_scope(llbc: &Llbc, id: u64) -> bool {
    let root_state = RootStackState::new(llbc);
    let analyzer = RootStackAnalyzer::new(llbc, &root_state);
    analyzer.fn_returns_owned_scope(id)
}

pub fn harvest_root_stack_touching_paths(llbc: &Llbc) -> Vec<String> {
    let root_state = RootStackState::new(llbc);
    let analyzer = RootStackAnalyzer::new(llbc, &root_state);
    let mut touching: Vec<String> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some() && analyzer.fn_touches_root_stack(fd.def_id))
        .map(|fd| fd.item_meta.name_path())
        .collect();
    touching.sort();
    touching.dedup();
    touching
}

/// Build the erasure plan for one body.  Everything it does not recognise
/// leaves the bracket alone, so a body this returns nothing for lowers
/// exactly as it did before.
pub(in crate::front::mir) fn analyze_root_brackets(
    body: &Unstructured,
    llbc: &Llbc,
    moved: &MovedOutLocals<'_>,
    root_stack: &RootStackAnalyzer<'_>,
) -> RootBracketPlan {
    analyze_root_brackets_with(
        llbc,
        body,
        moved,
        |reg| regular_call_name_path(reg, llbc),
        |reg| root_stack.regular_call_touches_root_stack(reg),
    )
}

/// The root-bracket guards [`lower_fun_decl`] takes out of this body, as MIR
/// locals.
///
/// An erased bracket publishes nothing, so it owes no rewind and lowers no
/// close.  A test that counts closes needs this to tell that case from a close
/// the lowering dropped on the floor.
pub fn erased_root_bracket_guards(llbc: &Llbc, fd: &FunDecl, body: &Unstructured) -> Vec<usize> {
    // A body the scalar replacement rewrote keeps no bracket at all.
    if let Ok(Some(_)) = shadow_stack_erase::erase_shadow_stack(fd, body, llbc) {
        return (0..body.locals.locals.len())
            .filter(|local| shadow_stack_erase::is_root_scope_local(body, llbc, *local))
            .collect();
    }
    let moved = MovedOutLocals::new(body, llbc);
    let root_stack_state = RootStackState::new(llbc);
    let root_stack = RootStackAnalyzer::new(llbc, &root_stack_state);
    analyze_root_brackets(body, llbc, &moved, &root_stack)
        .scopes
        .iter()
        .collect()
}

/// [`analyze_root_brackets`] with the callee-name lookup and the root-stack
/// analysis supplied, so the accept/reject decisions can be exercised without
/// an `Llbc`.
pub(in crate::front::mir) fn analyze_root_brackets_with(
    llbc: &Llbc,
    body: &Unstructured,
    moved: &MovedOutLocals<'_>,
    name_of: impl Fn(&RegularCall) -> Option<String>,
    touches: impl Fn(&RegularCall) -> bool,
) -> RootBracketPlan {
    let mut plan = RootBracketPlan {
        enabled: root_bracket_erase_enabled(),
        ..RootBracketPlan::default()
    };
    if !plan.enabled {
        return plan;
    }
    // (1) Candidate guards: the destination of a bracket opener.
    let mut candidates = bit_set::BitSet::new();
    let mut opener_block: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let Some(path) = name_of(reg) else {
            continue;
        };
        if !gc_root_scope_open_path(&path) {
            continue;
        }
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        if moved.contains(dest as usize) || opener_block.contains_key(&(dest as usize)) {
            // A guard opened twice in one body, or moved out of it, is not a
            // shape this pass models.
            candidates.remove(dest as usize);
            continue;
        }
        candidates.insert(dest as usize);
        opener_block.insert(dest as usize, bb_idx);
    }
    if candidates.is_empty() {
        return plan;
    }
    // (2) Borrow temporaries, so a receiver operand names its guard.
    let mut aliases: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, Rvalue::Ref { place: src, .. })) = stmt.stmt_kind_ref()
            else {
                continue;
            };
            let (PlaceKind::Local(dest), PlaceKind::Local(scope)) = (&place.kind, &src.kind) else {
                continue;
            };
            if !candidates.contains(*scope as usize) {
                continue;
            }
            if let Some(previous) = aliases.insert(*dest as usize, *scope as usize) {
                // One temporary borrowing two guards: give up on both.
                candidates.remove(previous);
                candidates.remove(*scope as usize);
            }
        }
    }
    // (2.5) `base()` results, and the copies a call argument is passed
    //     through.  MIR moves an argument into a fresh temporary rather than
    //     naming the callee's own destination, so `roots.get(ob_slot)` reads
    //     `_t` where `_t = copy _ob_slot`; without the closure below every
    //     read-back looks like an index this pass cannot name.
    let mut assigned: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && let PlaceKind::Local(dest) = place.kind
            {
                *assigned.entry(dest as usize).or_default() += 1;
            }
        }
    }
    // The innermost candidate bracket open at a block: the one whose region
    // holds the block and whose opener every other such region holds.
    let regions: std::collections::HashMap<usize, bit_set::BitSet> = candidates
        .iter()
        .map(|scope| {
            (
                scope,
                root_bracket_region(llbc, body, opener_block[&scope], scope, &name_of),
            )
        })
        .collect();
    let open_scope_at = |bb: usize| -> Option<usize> {
        let open: Vec<usize> = regions
            .iter()
            .filter(|(_, region)| region.contains(bb))
            .map(|(scope, _)| *scope)
            .collect();
        open.iter().copied().find(|scope| {
            open.iter()
                .all(|other| other == scope || regions[other].contains(opener_block[scope]))
        })
    };
    let mut free_sites: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut bases: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    // Block of each `pin_roots` call -> (guard, published locals, argument
    // temporaries).
    let mut pin_runs: std::collections::HashMap<usize, (usize, Vec<usize>, Vec<usize>)> =
        std::collections::HashMap::new();
    // `pin_roots` dest (and copies) -> (guard, pin block). The dest names the
    // first slot of that run, which is the guard's base only for the first
    // run (`zip_two_tuple_next` pins each cursor with its own `pin_roots`).
    let mut run_dests: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();
    // Block of each `pin_roots` of an incoming `&[PyObjectRef]` parameter
    // -> (guard, slice local, argument temporaries).
    let mut slice_runs: std::collections::HashMap<usize, (usize, usize, Vec<usize>)> =
        std::collections::HashMap::new();
    // `shadow_stack_copy_range(base + k, dest)` -> (guard, index local).
    let mut range_copies: Vec<(usize, usize, usize)> = Vec::new();
    // Free `pin_root(value)` -> (guard, published local). Its result is the
    // object it forwarded, not the slot base `pin_roots` returns.
    let mut single_pins: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();
    // `shadow_stack_len()` destination -> (guard, block of the call).
    let mut len_sites: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();
    // A destination two `shadow_stack_len` calls write names no one depth.
    let mut len_poison = bit_set::BitSet::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let Some(path) = name_of(reg) else {
            continue;
        };
        // `gc_roots::pin_roots(&[..])` pins onto the bracket open where it
        // runs, and answers with the slot its first value took. A later run
        // in the same bracket is not the guard's base; `run_dests` names it.
        if path.rsplit("::").next() == Some("pin_roots")
            && call.args.len() == 1
            && path.split("::").any(|s| s == ROOT_SCOPE_MODULE)
        {
            let (Some(slice), PlaceKind::Local(dest)) =
                (operand_local(call.args.first()), &call.dest.kind)
            else {
                continue;
            };
            let Some(scope) = open_scope_at(bb_idx) else {
                continue;
            };
            if let Some(run) = pin_roots_slice_values(body, slice, &assigned, &forward_blocks) {
                pin_runs.insert(bb_idx, (scope, run.0, run.1));
            } else if let Some(run) =
                pin_roots_incoming_slice(body, slice, &assigned, &forward_blocks)
            {
                slice_runs.insert(bb_idx, (scope, run.0, run.1));
            } else {
                continue;
            }
            free_sites.insert(bb_idx, scope);
            bases.insert(*dest as usize, scope);
            run_dests.insert(*dest as usize, (scope, bb_idx));
            continue;
        }
        // `pin_root(value)` pins onto the bracket open where it runs. The
        // result is `value` again, so it is not a base and not a `pin_runs`
        // entry. A bracket whose only publication is this pin still owes its
        // close: step 4 keeps it unless a read answers the slot.
        if path.rsplit("::").next() == Some("pin_root")
            && call.args.len() == 1
            && path.split("::").any(|segment| segment == ROOT_SCOPE_MODULE)
        {
            if let (Some(value), PlaceKind::Local(_)) =
                (operand_local(call.args.first()), &call.dest.kind)
                && let Some(scope) = open_scope_at(bb_idx)
            {
                single_pins.insert(bb_idx, (scope, value));
                free_sites.insert(bb_idx, scope);
            }
            continue;
        }
        // `shadow_stack_len()` reads the depth of that same bracket. A
        // statement that also writes the destination, or a second call, leaves
        // the depth unnamed — and the call stays out of `free_sites`, so the
        // bracket is not erased.
        if path.rsplit("::").next() == Some("shadow_stack_len")
            && call.args.is_empty()
            && path.split("::").any(|segment| segment == ROOT_SCOPE_MODULE)
        {
            if let PlaceKind::Local(dest) = call.dest.kind {
                let dest = dest as usize;
                if !len_poison.contains(dest) {
                    if let Some((prev_scope, prev_bb)) = len_sites.remove(&dest) {
                        if free_sites.get(&prev_bb) == Some(&prev_scope) {
                            free_sites.remove(&prev_bb);
                        }
                        len_poison.insert(dest);
                    } else if assigned.get(&dest).copied().unwrap_or(0) == 0
                        && !candidates.contains(dest)
                        && let Some(scope) = open_scope_at(bb_idx)
                    {
                        len_sites.insert(dest, (scope, bb_idx));
                        free_sites.insert(bb_idx, scope);
                    }
                }
            }
            continue;
        }
        let pins_many = path.rsplit("::").next() == Some("pin_roots") && call.args.len() == 2;
        if !(gc_root_scope_base_path(&path) && call.args.len() == 1) && !pins_many {
            continue;
        }
        let (Some(receiver), PlaceKind::Local(dest)) =
            (operand_local(call.args.first()), &call.dest.kind)
        else {
            continue;
        };
        let Some(scope) = aliases.get(&receiver).copied() else {
            continue;
        };
        if pins_many {
            // `scope.pin_roots(&[..])` answers with the slot its first value
            // took. The first run is the scope's base; a later run is not.
            let Some(slice) = operand_local(call.args.get(1)) else {
                continue;
            };
            if let Some(run) = pin_roots_slice_values(body, slice, &assigned, &forward_blocks) {
                pin_runs.insert(bb_idx, (scope, run.0, run.1));
            } else if let Some(run) =
                pin_roots_incoming_slice(body, slice, &assigned, &forward_blocks)
            {
                slice_runs.insert(bb_idx, (scope, run.0, run.1));
            } else {
                continue;
            }
            run_dests.insert(*dest as usize, (scope, bb_idx));
        }
        bases.insert(*dest as usize, scope);
    }
    // A free `pin_root` is the pin a `shadow_stack_len` names. Without that
    // len it can append a slot the close still has to rewind, so it is not a
    // known site. The pin may run before the len, so this drops it after both
    // have been seen.
    let len_scopes: bit_set::BitSet = len_sites.values().map(|(scope, _)| *scope).collect();
    single_pins.retain(|bb, (scope, _)| {
        let keep = len_scopes.contains(*scope);
        if !keep {
            free_sites.remove(bb);
        }
        keep
    });
    // A copy of a `base()` result answers for the same slot, provided nothing
    // else ever writes the temporary that holds it.
    let mut copies: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Use(operand, _))) = stmt.stmt_kind_ref()
                else {
                    continue;
                };
                let (PlaceKind::Local(dest), Some(src)) = (
                    &place.kind,
                    use_operand_slot_src(body, &assigned, operand, &forward_blocks),
                ) else {
                    continue;
                };
                let dest = *dest as usize;
                if bases.contains_key(&dest) || !bases.contains_key(&src) {
                    continue;
                }
                if assigned.get(&dest) != Some(&1)
                    || candidates.contains(dest)
                    || aliases.contains_key(&dest)
                {
                    continue;
                }
                bases.insert(dest, bases[&src]);
                copies.insert(dest, src);
                if let Some(&run) = run_dests.get(&src) {
                    run_dests.insert(dest, run);
                }
                changed = true;
            }
        }
    }
    // A copy of a `shadow_stack_len` result names the same depth. It is not a
    // base: inserting it there would make `get` answer slot 0.
    let mut len_index = len_sites.clone();
    changed = true;
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Use(operand, _))) = stmt.stmt_kind_ref()
                else {
                    continue;
                };
                let (PlaceKind::Local(dest), Some(src)) = (
                    &place.kind,
                    use_operand_slot_src(body, &assigned, operand, &forward_blocks),
                ) else {
                    continue;
                };
                let dest = *dest as usize;
                if len_index.contains_key(&dest) || !len_index.contains_key(&src) {
                    continue;
                }
                if assigned.get(&dest) != Some(&1)
                    || candidates.contains(dest)
                    || aliases.contains_key(&dest)
                    || bases.contains_key(&dest)
                {
                    continue;
                }
                let (scope, len_bb) = len_index[&src];
                len_index.insert(dest, (scope, len_bb));
                copies.insert(dest, src);
                changed = true;
            }
        }
    }
    // (2.6) `base + k` and `len + k`.  `roots.get(base + 1)` reaches `get` as
    //     `_s = AddChecked(copy _b, const 1)`, an overflow `Assert` on `_s.1`
    //     and `_i = move _s.0`; an unchecked build spells the sum as the index
    //     directly.  `shadow_stack_len` names the next pin, so the same sum
    //     off a len result names that pin plus `k`.  Either way the index
    //     names a slot this pass can answer.
    let fresh = |dest: usize| {
        assigned.get(&dest) == Some(&1)
            && !candidates.contains(dest)
            && !aliases.contains_key(&dest)
            && !bases.contains_key(&dest)
            && !len_index.contains_key(&dest)
    };
    // Checked-sum tuples, and the slot indices read out of them or summed
    // directly, each with its guard and offset.
    let mut sums: std::collections::HashMap<usize, RootSlotSum> = std::collections::HashMap::new();
    let mut offsets: std::collections::HashMap<usize, RootSlotSum> =
        std::collections::HashMap::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, Rvalue::BinaryOp(op, lhs, rhs))) = stmt.stmt_kind_ref()
            else {
                continue;
            };
            let PlaceKind::Local(dest) = place.kind else {
                continue;
            };
            let (Some(src), Some(k)) = (operand_local(Some(lhs)), operand_usize_literal(llbc, rhs))
            else {
                continue;
            };
            let (scope, from_len, from_run) = if let Some(&(scope, run_bb)) = run_dests.get(&src) {
                (scope, None, Some(run_bb))
            } else if let Some(&scope) = bases.get(&src) {
                (scope, None, None)
            } else if let Some(&(scope, len_bb)) = len_index.get(&src) {
                (scope, Some(len_bb), None)
            } else {
                continue;
            };
            if !fresh(dest as usize) {
                continue;
            }
            let slot = RootSlotSum {
                scope,
                k,
                from_len,
                from_run,
            };
            match root_slot_sum_is_checked(op) {
                Some(true) => {
                    sums.insert(dest as usize, slot);
                }
                Some(false) => {
                    offsets.insert(dest as usize, slot);
                }
                None => continue,
            };
        }
    }
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, Rvalue::Use(operand, _))) = stmt.stmt_kind_ref() else {
                continue;
            };
            let PlaceKind::Local(dest) = place.kind else {
                continue;
            };
            let (Operand::Copy(src) | Operand::Move(src)) = &operand else {
                continue;
            };
            let Some(sum) = checked_sum_field_local(src, 0) else {
                continue;
            };
            if let Some(slot) = sums.get(&sum).copied()
                && fresh(dest as usize)
                && !sums.contains_key(&(dest as usize))
            {
                offsets.insert(dest as usize, slot);
            }
        }
    }
    // A copy of a `base + k` / `len + k` index names the same slot. MIR
    // moves a call argument into a fresh temporary, the same spelling
    // (2.5) already follows for `base()` and `shadow_stack_len`.
    let mut changed = true;
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Use(operand, _))) = stmt.stmt_kind_ref()
                else {
                    continue;
                };
                let (PlaceKind::Local(dest), Some(src)) = (
                    &place.kind,
                    use_operand_slot_src(body, &assigned, operand, &forward_blocks),
                ) else {
                    continue;
                };
                let dest = *dest as usize;
                if offsets.contains_key(&dest) || !offsets.contains_key(&src) {
                    continue;
                }
                if assigned.get(&dest) != Some(&1)
                    || candidates.contains(dest)
                    || aliases.contains_key(&dest)
                    || bases.contains_key(&dest)
                    || len_index.contains_key(&dest)
                    || sums.contains_key(&dest)
                {
                    continue;
                }
                offsets.insert(dest, offsets[&src]);
                copies.insert(dest, src);
                changed = true;
            }
        }
    }
    // (3) Classify every mention of a guard, one of its borrows, or one of its
    //     slot indices.  Anything this loop does not account for retires the
    //     guard.
    let guards = {
        let mut set = candidates.clone();
        for local in aliases.keys() {
            set.insert(*local);
        }
        set
    };
    let mut watched = guards.clone();
    for local in bases
        .keys()
        .chain(sums.keys())
        .chain(offsets.keys())
        .chain(len_index.keys())
    {
        watched.insert(*local);
    }
    // Every watched local that is not a guard names the guard it belongs to,
    // so a mention this pass does not model retires the right bracket.
    let mut owner = aliases.clone();
    owner.extend(bases.iter().map(|(local, scope)| (*local, *scope)));
    owner.extend(
        sums.iter()
            .chain(offsets.iter())
            .map(|(local, slot)| (*local, slot.scope)),
    );
    owner.extend(len_index.iter().map(|(local, (scope, _))| (*local, *scope)));
    let call_dests = bit_set::BitSet::new();
    let getter_index = RootGetterIndex {
        assigned: &assigned,
        bases: &bases,
        offsets: &offsets,
        len_index: &len_index,
        candidates: &candidates,
        aliases: &aliases,
        call_dests: &call_dests,
        forward_blocks: &forward_blocks,
    };
    // Closure reads are collected before step 3 retires a guard: the `&index`
    // that builds the capture is a mention the whitelist below keeps.
    let reads = record_root_slot_getter_reads(llbc, body, &name_of, &getter_index);
    let mut pins: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    let mut gets: Vec<(usize, usize, usize, u64)> = reads.gets;
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_))
                | Ok(StmtKind::StorageDead(_))
                | Ok(StmtKind::Borrowck(_)) => continue,
                Ok(StmtKind::Assign(place, Rvalue::Ref { place: src, .. }))
                    if matches!(place.kind, PlaceKind::Local(d) if aliases.contains_key(&(d as usize)))
                        && matches!(src.kind, PlaceKind::Local(s) if candidates.contains(s as usize)) =>
                {
                    continue;
                }
                // `_t = copy _slot`, the argument temporary (2.5) recorded,
                // and a field of a tuple aggregate that names the same slot.
                Ok(StmtKind::Assign(place, Rvalue::Use(operand, _))) if matches!(place.kind, PlaceKind::Local(d) if copies.get(&(d as usize)).copied() == use_operand_slot_src(body, &assigned, operand, &forward_blocks)) =>
                {
                    continue;
                }
                // `_t = (slot0, slot1, …)`: the carrier whose fields (2.6)
                // copies follow. Naming those slots here is that alias.
                Ok(StmtKind::Assign(place, Rvalue::Aggregate(kind, operands)))
                    if aggregate_kind_is_tuple(kind)
                        && matches!(place.kind, PlaceKind::Local(d) if assigned.get(&(d as usize)) == Some(&1))
                        && operands.iter().all(|op| {
                            operand_local(Some(op)).is_none_or(|l| {
                                !watched.contains(l)
                                    || bases.contains_key(&l)
                                    || offsets.contains_key(&l)
                                    || len_index.contains_key(&l)
                                    || sums.contains_key(&l)
                            })
                        }) =>
                {
                    continue;
                }
                // The `base + k` sum and the index read out of it, (2.6).
                // Each is assigned once, so this is the statement (2.6)
                // matched.
                Ok(StmtKind::Assign(place, Rvalue::BinaryOp(..))) if matches!(place.kind, PlaceKind::Local(d) if sums.contains_key(&(d as usize)) || offsets.contains_key(&(d as usize))) =>
                {
                    continue;
                }
                Ok(StmtKind::Assign(
                    place,
                    Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _),
                )) if matches!(place.kind, PlaceKind::Local(d) if offsets.contains_key(&(d as usize)))
                    && checked_sum_field_local(src, 0)
                        .is_some_and(|sum| sums.contains_key(&sum)) =>
                {
                    continue;
                }
                _ => {}
            }
            // The borrow, aggregate and argument tuple that build a closure
            // over an index. The index itself may be named; any other watched
            // local in the statement still retires its bracket.
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && let PlaceKind::Local(dest) = place.kind
                && reads.capture_assigns.contains(dest as usize)
            {
                if mentions_local(stmt.kind_value(), &watched) {
                    retire_mentioned_except(
                        stmt.kind_value(),
                        &watched,
                        &owner,
                        &mut candidates,
                        &reads.admitted_indexes,
                    );
                }
                continue;
            }
            if mentions_local(stmt.kind_value(), &watched) {
                retire_mentioned(stmt.kind_value(), &watched, &owner, &mut candidates);
            }
        }
        let term_kind = body.body[bb_idx].terminator.kind_value();
        match bb.term_ref(llbc) {
            Ok(term)
                if term_dropped_local(term, &name_of)
                    .is_some_and(|local| candidates.contains(local as usize)) =>
            {
                continue;
            }
            // The overflow check on a `base + k` sum.  It fails only past
            // `usize::MAX`, and the lowering strips it like every assert.
            Ok(TermKind::Assert { assert, .. })
                if matches!(&assert.cond, Operand::Copy(src) | Operand::Move(src)
                    if checked_sum_field_local(src, 1).is_some_and(|sum| sums.contains_key(&sum))) =>
            {
                continue;
            }
            Ok(TermKind::Call { call, .. }) => {
                // The opener names its guard as the call destination, which is
                // the one mention of a guard that is not a use of one.
                let opens_candidate = matches!(call.dest.kind, PlaceKind::Local(d) if opener_block.get(&(d as usize)) == Some(&bb_idx))
                    && !call
                        .args
                        .iter()
                        .any(|op| operand_local(Some(op)).is_some_and(|l| guards.contains(l)));
                if opens_candidate {
                    continue;
                }
                // The free pin (2.5) recorded, and a free read-back of a slot
                // one of the guard's own bases names.
                let free_leaf = match &call.func {
                    CallFunc::Regular(reg) => name_of(reg)
                        .filter(|path| path.split("::").any(|s| s == ROOT_SCOPE_MODULE))
                        .map(|path| path.rsplit("::").next().unwrap_or("").to_string()),
                    _ => None,
                };
                let dest_is_guard =
                    matches!(call.dest.kind, PlaceKind::Local(d) if guards.contains(d as usize));
                // `shadow_stack_len` takes no argument. Its destination is
                // watched, so failing to continue here retires the bracket.
                if call.args.is_empty()
                    && free_leaf.as_deref() == Some("shadow_stack_len")
                    && let PlaceKind::Local(dest) = call.dest.kind
                    && let Some(&(scope, len_bb)) = len_sites.get(&(dest as usize))
                    && len_bb == bb_idx
                    && free_sites.get(&bb_idx) == Some(&scope)
                    && candidates.contains(scope)
                {
                    continue;
                }
                if call.args.len() == 2
                    && free_leaf.as_deref() == Some("shadow_stack_copy_range")
                    && let Some(index) = operand_local(call.args.first())
                    && let Some(scope) = bases
                        .get(&index)
                        .copied()
                        .or_else(|| offsets.get(&index).map(|slot| slot.scope))
                    && candidates.contains(scope)
                {
                    range_copies.push((bb_idx, scope, index));
                    free_sites.insert(bb_idx, scope);
                    continue;
                }
                if call.args.len() == 1 && !dest_is_guard {
                    match free_leaf.as_deref() {
                        Some("pin_roots") => {
                            if let Some((scope, values, _)) = pin_runs.get(&bb_idx)
                                && free_sites.get(&bb_idx) == Some(scope)
                                && candidates.contains(*scope)
                            {
                                let entry = pins.entry(*scope).or_default();
                                entry.extend(values.iter().map(|v| (bb_idx, *v)));
                                continue;
                            }
                            if let Some((scope, _, _)) = slice_runs.get(&bb_idx)
                                && free_sites.get(&bb_idx) == Some(scope)
                                && candidates.contains(*scope)
                            {
                                continue;
                            }
                        }
                        Some("pin_root") => {
                            if let Some(&(scope, value)) = single_pins.get(&bb_idx)
                                && free_sites.get(&bb_idx) == Some(&scope)
                                && candidates.contains(scope)
                            {
                                pins.entry(scope).or_default().push((bb_idx, value));
                                continue;
                            }
                        }
                        Some("shadow_stack_get") => {
                            if let Some(index) = operand_local(call.args.first())
                                && let Some(scope) = bases
                                    .get(&index)
                                    .copied()
                                    .or_else(|| offsets.get(&index).map(|slot| slot.scope))
                                    .or_else(|| len_index.get(&index).map(|(scope, _)| *scope))
                                && candidates.contains(scope)
                            {
                                gets.push((bb_idx, scope, index, 0));
                                free_sites.insert(bb_idx, scope);
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
                let leaf_scope = match &call.func {
                    CallFunc::Regular(reg) => name_of(reg)
                        .filter(|path| path.split("::").any(|s| s == "gc_roots"))
                        .and_then(|path| {
                            let leaf = path.rsplit("::").next().unwrap_or("").to_string();
                            let receiver = operand_local(call.args.first())?;
                            let scope = aliases.get(&receiver).copied()?;
                            Some((leaf, scope))
                        }),
                    _ => None,
                };
                let handled = match &leaf_scope {
                    Some((leaf, scope)) if candidates.contains(*scope) => {
                        let dest = match call.dest.kind {
                            PlaceKind::Local(d) => Some(d as usize),
                            _ => None,
                        };
                        let names_guard_elsewhere = dest.is_some_and(|d| guards.contains(d))
                            || call.args.iter().skip(1).any(|op| {
                                operand_local(Some(op)).is_some_and(|l| guards.contains(l))
                            });
                        if names_guard_elsewhere {
                            false
                        } else {
                            match (leaf.as_str(), dest) {
                                ("pin_root", Some(_)) => match operand_local(call.args.get(1)) {
                                    Some(value) if call.args.len() == 2 => {
                                        pins.entry(*scope).or_default().push((bb_idx, value));
                                        true
                                    }
                                    _ => false,
                                },
                                // (2.5) already recorded the slot this answers.
                                ("base", Some(dest)) if call.args.len() == 1 => {
                                    bases.get(&dest) == Some(scope)
                                }
                                ("pin_roots", Some(dest)) if call.args.len() == 2 => {
                                    match pin_runs.get(&bb_idx) {
                                        Some((run_scope, values, _))
                                            if run_scope == scope
                                                && bases.get(&dest) == Some(scope) =>
                                        {
                                            let entry = pins.entry(*scope).or_default();
                                            entry.extend(values.iter().map(|v| (bb_idx, *v)));
                                            true
                                        }
                                        _ => match slice_runs.get(&bb_idx) {
                                            Some((run_scope, _, _))
                                                if run_scope == scope
                                                    && bases.get(&dest) == Some(scope) =>
                                            {
                                                true
                                            }
                                            _ => false,
                                        },
                                    }
                                }
                                ("get", Some(_)) if call.args.len() == 2 => {
                                    match operand_local(call.args.get(1)) {
                                        Some(index) => {
                                            gets.push((bb_idx, *scope, index, 0));
                                            true
                                        }
                                        None => false,
                                    }
                                }
                                _ => false,
                            }
                        }
                    }
                    _ => false,
                };
                if !handled && mentions_local(term_kind, &watched) {
                    retire_mentioned(term_kind, &watched, &owner, &mut candidates);
                }
            }
            _ => {
                if mentions_local(term_kind, &watched) {
                    retire_mentioned(term_kind, &watched, &owner, &mut candidates);
                }
            }
        }
    }
    // (4) Keep only the guards whose reads this pass can answer.
    let mut base_results: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut slot_temps: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut pinned: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    let mut get_sites: Vec<(usize, usize)> = Vec::new();
    let mut pin_temps: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let surviving: Vec<usize> = candidates.iter().collect();
    for scope in surviving {
        let opener = opener_block[&scope];
        let incoming_elem = pins.get(&scope).and_then(|scope_pins| {
            incoming_elem_pins(llbc, body, scope_pins, &assigned, &forward_blocks)
        });
        let pins_are_incoming_slice = slice_runs
            .values()
            .any(|(run_scope, _, _)| *run_scope == scope)
            || incoming_elem.is_some();
        if !root_bracket_stack_effects_are_known(
            llbc,
            body,
            opener,
            scope,
            &aliases,
            &free_sites,
            &name_of,
            &touches,
            pins_are_incoming_slice,
        ) {
            continue;
        }
        if let Some((pin_bb, slice_local, temps)) =
            slice_runs.iter().find_map(|(bb, (run_scope, loc, temps))| {
                (*run_scope == scope).then_some((*bb, *loc, temps.clone()))
            })
        {
            if !root_pin_runs_once_per_opening(llbc, body, opener, pin_bb) {
                continue;
            }
            let region = root_bracket_region(llbc, body, opener, scope, &name_of);
            let own_get_bbs: Vec<usize> = gets
                .iter()
                .filter(|(_, get_scope, _, _)| *get_scope == scope)
                .map(|(get_bb, _, _, _)| *get_bb)
                .collect();
            // `scope.pin_roots(args)` reborrows the incoming `&[PyObjectRef]`
            // (`_t = &*args`). That Ref is the pin's argument, not a write of
            // the fat pointer; `root_pin_value_is_stable` treats any Ref of a
            // local as aliasing. A parameter with no Assign is still the same
            // array `W_CData.call` received.
            let slice_is_unassigned_param = assigned.get(&slice_local).copied().unwrap_or(0) == 0
                && (1..=body.locals.arg_count as usize).contains(&slice_local);
            if (!slice_is_unassigned_param
                && !root_pin_value_is_stable_in_bracket(
                    llbc,
                    body,
                    slice_local,
                    &region,
                    &own_get_bbs,
                ))
                || !region.contains(pin_bb)
            {
                continue;
            }
            let mut ok = true;
            let mut sites = Vec::new();
            for (get_bb, _, index_local, addend) in gets
                .iter()
                .filter(|(_, get_scope, _, _)| *get_scope == scope)
            {
                let slot =
                    root_read_const_slot(*index_local, scope, &bases, &offsets).and_then(|base| {
                        usize::try_from(*addend)
                            .ok()
                            .and_then(|extra| base.checked_add(extra))
                    });
                let Some(k) = slot else {
                    ok = false;
                    break;
                };
                if !region.contains(*get_bb) {
                    ok = false;
                    break;
                }
                sites.push((*get_bb, slice_local, k as u64));
            }
            let mut copies = Vec::new();
            for (copy_bb, copy_scope, index_local) in &range_copies {
                if *copy_scope != scope {
                    continue;
                }
                let Some(k) = root_read_const_slot(*index_local, scope, &bases, &offsets) else {
                    ok = false;
                    break;
                };
                if !region.contains(*copy_bb) {
                    ok = false;
                    break;
                }
                copies.push((*copy_bb, slice_local, k as u64));
            }
            if !ok {
                continue;
            }
            plan.scopes.insert(scope);
            plan.slice_gets.extend(sites);
            plan.slice_copies.extend(copies);
            for (base_local, base_scope) in &bases {
                if *base_scope == scope {
                    base_results.insert(*base_local, scope);
                }
            }
            for (temp, slot) in sums.iter().chain(offsets.iter()) {
                if slot.scope == scope {
                    slot_temps.insert(*temp, scope);
                }
            }
            for (temp, (temp_scope, _)) in &len_index {
                if *temp_scope == scope {
                    slot_temps.insert(*temp, scope);
                }
            }
            pin_temps.extend(temps.iter().map(|temp| (*temp, scope)));
            continue;
        }
        if let Some((slice_local, elem_ks)) = incoming_elem {
            let scope_pins: &[(usize, usize)] = pins.get(&scope).map(Vec::as_slice).unwrap_or(&[]);
            if scope_pins
                .iter()
                .any(|&(pin_bb, _)| !root_pin_runs_once_per_opening(llbc, body, opener, pin_bb))
            {
                continue;
            }
            let region = root_bracket_region(llbc, body, opener, scope, &name_of);
            if scope_pins
                .iter()
                .any(|(pin_bb, _)| !region.contains(*pin_bb))
            {
                continue;
            }
            let mut ok = true;
            let mut sites = Vec::new();
            for (get_bb, _, index_local, addend) in gets
                .iter()
                .filter(|(_, get_scope, _, _)| *get_scope == scope)
            {
                let slot =
                    root_read_const_slot(*index_local, scope, &bases, &offsets).and_then(|base| {
                        usize::try_from(*addend)
                            .ok()
                            .and_then(|extra| base.checked_add(extra))
                    });
                let Some(k) = slot.and_then(|i| elem_ks.get(i).copied()) else {
                    ok = false;
                    break;
                };
                if !region.contains(*get_bb) {
                    ok = false;
                    break;
                }
                sites.push((*get_bb, slice_local, k));
            }
            if !ok {
                continue;
            }
            plan.scopes.insert(scope);
            plan.slice_gets.extend(sites);
            for (base_local, base_scope) in &bases {
                if *base_scope == scope {
                    base_results.insert(*base_local, scope);
                }
            }
            for (temp, slot) in sums.iter().chain(offsets.iter()) {
                if slot.scope == scope {
                    slot_temps.insert(*temp, scope);
                }
            }
            continue;
        }
        let scope_pins: &[(usize, usize)] = pins.get(&scope).map(Vec::as_slice).unwrap_or(&[]);
        let scope_gets: Vec<(usize, usize, u64)> = gets
            .iter()
            .filter(|(_, get_scope, _, _)| *get_scope == scope)
            .map(|(get_bb, _, index_local, addend)| (*get_bb, *index_local, *addend))
            .collect();
        if scope_pins.is_empty() {
            // No receiver pin, and the whole-bracket proof excluded free pins
            // and unknown callees too: nothing was published.
            if !scope_gets.is_empty() {
                continue;
            }
        } else if scope_gets.is_empty()
            && single_pins
                .values()
                .any(|(pin_scope, _)| *pin_scope == scope)
        {
            // A free `pin_root` published a slot nothing read back. The close
            // still rewinds that slot, so the bracket stays.
            continue;
        } else {
            let region = root_bracket_region(llbc, body, opener, scope, &name_of);
            let own_get_bbs: Vec<usize> = scope_gets.iter().map(|(get_bb, _, _)| *get_bb).collect();
            if scope_pins.iter().any(|&(pin_bb, value_local)| {
                !root_pin_value_is_stable_in_bracket(llbc, body, value_local, &region, &own_get_bbs)
                    || !root_pin_runs_once_per_opening(llbc, body, opener, pin_bb)
            }) {
                continue;
            }
            if scope_pins
                .iter()
                .any(|(pin_bb, _)| !region.contains(*pin_bb))
            {
                continue;
            }
            let dom = root_bracket_region_dominators(llbc, body, opener, &region);
            // Slots are numbered by execution order.  Every other call in the
            // bracket leaves the stack as it found it, so the `k`-th pin fills
            // `base + k` provided each pin runs after all the earlier ones on
            // every path: the pins form one dominator chain.
            let mut ordered = scope_pins.to_vec();
            ordered.sort_by_key(|(pin_bb, _)| {
                scope_pins
                    .iter()
                    .filter(|(other, _)| dom[pin_bb].contains(*other))
                    .count()
            });
            // One `pin_roots` block publishes its whole run in array order,
            // so consecutive pins may share it; two separate pins may not.
            // A later `pin_roots` is another pin of the same chain: its dest
            // names that run's first slot (`root_read_base_slot`), not the
            // guard's base.  A run off the chain still fails the windows
            // check below, which is the unbalanced case this used to reject
            // by refusing every non-first run.
            if ordered.windows(2).any(|w| {
                if w[0].0 == w[1].0 {
                    !pin_runs.contains_key(&w[0].0)
                } else {
                    !dom[&w[1].0].contains(w[0].0)
                }
            }) {
                continue;
            }
            let mut sites = Vec::new();
            let mut ok = true;
            for (get_bb, index_local, addend) in scope_gets {
                // The index has to be this guard's own base, a `base + k`, a
                // later `pin_roots` dest, or the next pin a `shadow_stack_len`
                // names, plus the constant a closure adds. The pin that filled
                // that slot has to have run on every path reaching the read.
                let slot = root_read_base_slot(
                    index_local,
                    scope,
                    &bases,
                    &offsets,
                    &len_index,
                    &run_dests,
                    &ordered,
                    &dom,
                )
                .and_then(|base| {
                    usize::try_from(addend)
                        .ok()
                        .and_then(|extra| base.checked_add(extra))
                });
                let Some(&(pin_bb, value_local)) = slot.and_then(|k| ordered.get(k)) else {
                    ok = false;
                    break;
                };
                if !region.contains(get_bb)
                    || !dom.get(&get_bb).is_some_and(|set| set.contains(pin_bb))
                {
                    ok = false;
                    break;
                }
                sites.push((get_bb, value_local));
            }
            if !ok {
                continue;
            }
            pinned.insert(scope, ordered.iter().map(|(_, value)| *value).collect());
            get_sites.extend(sites);
        }
        plan.scopes.insert(scope);
        for (base_local, base_scope) in &bases {
            if *base_scope == scope {
                base_results.insert(*base_local, scope);
            }
        }
        for (temp, slot) in sums.iter().chain(offsets.iter()) {
            if slot.scope == scope {
                slot_temps.insert(*temp, scope);
            }
        }
        // The len result and its copies only ever indexed a read-back.
        for (temp, (temp_scope, _)) in &len_index {
            if *temp_scope == scope {
                slot_temps.insert(*temp, scope);
            }
        }
        for (run_scope, _, temps) in pin_runs.values() {
            if *run_scope == scope {
                pin_temps.extend(temps.iter().map(|temp| (*temp, scope)));
            }
        }
    }
    plan.aliases = aliases
        .into_iter()
        .filter(|(_, scope)| plan.scopes.contains(*scope))
        .collect();
    plan.base_results = base_results;
    plan.slot_temps = slot_temps;
    plan.pins = pinned;
    plan.get_sites = get_sites;
    plan.pin_temps = pin_temps;
    plan.capture_temps = reads
        .capture_temps
        .into_iter()
        .filter(|(_, scope)| plan.scopes.contains(*scope))
        .collect();
    plan.free_sites = free_sites
        .into_iter()
        .filter(|(_, scope)| plan.scopes.contains(*scope))
        .collect();
    plan
}

/// `k` for a `usize` literal operand.
pub(in crate::front::mir) fn operand_usize_literal(llbc: &Llbc, op: &Operand) -> Option<u64> {
    let Operand::Const(c) = op else {
        return None;
    };
    let literal = llbc.const_expr_literal(c)?;
    let unsigned = literal.get("Scalar")?.get("Unsigned")?.as_array()?;
    if unsigned.first()?.as_str()? != "Usize" {
        return None;
    }
    unsigned.get(1)?.as_str()?.parse().ok()
}

/// Whether a `base + k` sum is the overflow-checked spelling (a `(sum,
/// overflowed)` tuple) or the plain one.  `None` for any other operator.
pub(in crate::front::mir) fn root_slot_sum_is_checked(op: &serde_json::Value) -> Option<bool> {
    match op {
        serde_json::Value::String(name) if name == "AddChecked" => Some(true),
        serde_json::Value::String(name) if name == "Add" => Some(false),
        serde_json::Value::Object(map) if map.contains_key("Add") => Some(false),
        _ => None,
    }
}

/// Index into the host layout's per-variant field offsets.
/// A struct projection (`[null, i]`) is variant 0.
pub(in crate::front::mir) fn host_field_index(
    variant_idx: Option<u64>,
    field_idx: usize,
) -> Option<(u32, u32)> {
    Some((
        u32::try_from(variant_idx.unwrap_or(0)).ok()?,
        u32::try_from(field_idx).ok()?,
    ))
}

/// `[null, i]` or `[variant, i]`. `None` when the payload is a named
/// `{"Adt": ...}` container.
pub(in crate::front::mir) fn indexed_field_parts(
    payload: &serde_json::Value,
) -> Option<(Option<u64>, usize)> {
    let arr = payload.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let idx = arr.get(1)?.as_u64()? as usize;
    if arr[0].is_null() {
        return Some((None, idx));
    }
    arr[0].as_u64().map(|variant| (Some(variant), idx))
}

/// Operand `N` of a single-assignment tuple aggregate, when `operand` is
/// `Copy`/`Move` of that aggregate's field `N`.
pub(in crate::front::mir) fn tuple_aggregate_field_src(
    body: &Unstructured,
    assigned: &std::collections::HashMap<usize, usize>,
    operand: &Operand,
    forward_blocks: &[bool],
) -> Option<usize> {
    let (Operand::Copy(place) | Operand::Move(place)) = operand else {
        return None;
    };
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
        return None;
    };
    let parts = elem.get("Field")?.as_array()?;
    let idx = if parts.first()?.is_null() || parts.first()?.as_u64().is_some() {
        parts.get(1)?.as_u64()?
    } else {
        return None;
    };
    let PlaceKind::Local(tmp) = inner.kind else {
        return None;
    };
    let Rvalue::Aggregate(kind, operands) =
        single_statement_rvalue(body, assigned, tmp as usize, forward_blocks)?
    else {
        return None;
    };
    if !aggregate_kind_is_tuple(&kind) {
        return None;
    }
    operand_local(operands.get(idx as usize))
}

pub(in crate::front::mir) fn aggregate_kind_is_tuple(kind: &serde_json::Value) -> bool {
    if aggregate_ctor_name(kind) == "Tuple" {
        return true;
    }
    kind.get("Adt")
        .and_then(serde_json::Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(serde_json::Value::as_str)
        == Some("Tuple")
}

/// The local a Use names: a plain copy, or field `N` of a single-assignment
/// tuple aggregate whose operand `N` is that local.
pub(in crate::front::mir) fn use_operand_slot_src(
    body: &Unstructured,
    assigned: &std::collections::HashMap<usize, usize>,
    operand: &Operand,
    forward_blocks: &[bool],
) -> Option<usize> {
    operand_local(Some(operand))
        .or_else(|| tuple_aggregate_field_src(body, assigned, operand, forward_blocks))
}

/// The local `_s` of a `_s.<field>` read of a tuple, such as the sum or the
/// overflow flag of a checked add.
pub(in crate::front::mir) fn checked_sum_field_local(place: &Place, field: u64) -> Option<usize> {
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
        return None;
    };
    let parts = elem.get("Field")?.as_array()?;
    let idx = if parts.first()?.is_null() || parts.first()?.as_u64().is_some() {
        parts.get(1)?.as_u64()?
    } else {
        return None;
    };
    if idx != field {
        return None;
    }
    match inner.kind {
        PlaceKind::Local(local) => Some(local as usize),
        _ => None,
    }
}

/// Retire every guard a statement or terminator names in a way
/// [`analyze_root_brackets`] does not model.  `owner` maps a borrow or a slot
/// index back to the guard it belongs to, so naming one of those retires the
/// bracket it would have read.
pub(in crate::front::mir) fn retire_mentioned(
    kind: &serde_json::Value,
    watched: &bit_set::BitSet,
    owner: &std::collections::HashMap<usize, usize>,
    candidates: &mut bit_set::BitSet,
) {
    let mut named = bit_set::BitSet::new();
    collect_locals(kind, watched, &mut named);
    for local in named.iter() {
        candidates.remove(local);
        if let Some(scope) = owner.get(&local) {
            candidates.remove(*scope);
        }
    }
}

/// [`retire_mentioned`], except locals in `except` — an index a closure
/// captures. Naming that index does not retire its bracket. Any other watched
/// local in the same statement still does. An index that is itself a guard is
/// not in `except`.
pub(in crate::front::mir) fn retire_mentioned_except(
    kind: &serde_json::Value,
    watched: &bit_set::BitSet,
    owner: &std::collections::HashMap<usize, usize>,
    candidates: &mut bit_set::BitSet,
    except: &bit_set::BitSet,
) {
    let mut named = bit_set::BitSet::new();
    collect_locals(kind, watched, &mut named);
    for local in named.iter() {
        if except.contains(local) {
            continue;
        }
        candidates.remove(local);
        if let Some(scope) = owner.get(&local) {
            candidates.remove(*scope);
        }
    }
}

pub(in crate::front::mir) fn collect_locals(
    v: &serde_json::Value,
    wanted: &bit_set::BitSet,
    out: &mut bit_set::BitSet,
) {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(local) = map.get("Local").and_then(serde_json::Value::as_u64)
                && wanted.contains(local as usize)
            {
                out.insert(local as usize);
            }
            for nested in map.values() {
                collect_locals(nested, wanted, out);
            }
        }
        serde_json::Value::Array(items) => {
            for nested in items {
                collect_locals(nested, wanted, out);
            }
        }
        _ => {}
    }
}

/// How a call relates to the rbigint owner-root handle, from the callee's
/// declaration alone. The declaration is all a crate's LLBC has for a
/// foreign callee, so every crate classifies one call the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum OwnerRootCallee {
    /// `RBigIntGcRoot::new(value)`: acquires a slot for `value`.
    New,
    /// Any other function returning a fresh handle by value
    /// (`live_rbigint`, `live_long_num`).
    Producer,
    /// `<RBigIntGcRoot as Deref>::deref` / `DerefMut::deref_mut`.
    Deref,
    /// Anything else.
    Other,
}

/// The rbigint owner-root handles [`lower_fun_decl`] takes out of a body.
///
/// `RBigIntGcRoot` is the hand-written stand-in for the fixed frame slot that
/// `gc_save_root` / `gc_restore_root` give a local live across collecting
/// calls. `ShadowStackFrameworkGCTransformer` inserts those operations into
/// the final graphs long after `warmspot.py` `make_jitcodes` read them, so no
/// jitcode upstream holds one: the local is the rbigint itself. The three
/// consumers of a jitcode root the value by their own means (see
/// [`RootBracketPlan`]), so the handle lowers as the value it roots:
/// `new(v)` and `deref` are `v`, and the drop that releases the slot is
/// nothing.
///
/// That rewrite is sound only while every use of the handle is one of those
/// operations. A body where the handle is also passed on, stored, or reached
/// through `set` keeps the real handle.
#[derive(Default)]
pub(in crate::front::mir) struct OwnerRootPlan {
    /// Handle locals that lower as the rooted `RBigInt`.
    pub(in crate::front::mir) roots: bit_set::BitSet,
    /// For each borrow of an erased handle, the handle it borrows.
    pub(in crate::front::mir) borrow_of: Vec<Option<usize>>,
}

impl OwnerRootPlan {
    pub(in crate::front::mir) fn erases(&self, local: usize) -> bool {
        self.roots.contains(local)
    }

    pub(in crate::front::mir) fn handle_of_borrow(&self, local: usize) -> Option<usize> {
        self.borrow_of.get(local).copied().flatten()
    }
}

/// Whether a type node, behind dedup and hash-cons indirections only, is the
/// `RBigIntGcRoot` ADT itself.
pub(in crate::front::mir) fn type_node_is_owner_root(
    node: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    strip_ty_indirections(node, llbc)
        .and_then(adt_node_def_id)
        .is_some_and(|id| type_id_is_owner_root(id, llbc))
}

pub(in crate::front::mir) fn type_id_is_owner_root(id: u64, llbc: &Llbc) -> bool {
    llbc.type_by_id(id).is_some_and(|td| {
        // The leaf compare keeps the full path off every other ADT local.
        matches!(td.item_meta.template_name().last(), Some(NameSeg::Ident { ident: (leaf, _) }) if leaf == "RBigIntGcRoot")
            && owner_root_type_path(&td.item_meta.name_path())
    })
}

/// The handle's declaration path, with or without its crate prefix.
pub(in crate::front::mir) fn owner_root_type_path(path: &str) -> bool {
    path == "rbigint::gc::RBigIntGcRoot" || path.ends_with("::rbigint::gc::RBigIntGcRoot")
}

/// Whether `RBigIntGcRoot` occurs anywhere in a type node: `&RBigIntGcRoot`,
/// `Option<RBigIntGcRoot>`, a tuple holding one.
pub(in crate::front::mir) fn type_node_mentions_owner_root(
    node: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> bool {
    if depth > 32 {
        // Too deep to tell: treat it as naming the handle.
        return true;
    }
    match node {
        serde_json::Value::Object(map) => {
            if let Some(id) = map.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                return llbc
                    .dedup_body(id)
                    .is_some_and(|body| type_node_mentions_owner_root(body, llbc, depth + 1));
            }
            if adt_node_def_id(node).is_some_and(|id| type_id_is_owner_root(id, llbc)) {
                return true;
            }
            map.values()
                .any(|nested| type_node_mentions_owner_root(nested, llbc, depth + 1))
        }
        serde_json::Value::Array(items) => items
            .iter()
            .any(|nested| type_node_mentions_owner_root(nested, llbc, depth + 1)),
        _ => false,
    }
}

pub(in crate::front::mir) fn tyref_is_owner_root(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc).is_some_and(|node| type_node_is_owner_root(node, llbc))
}

/// [`analyze_owner_roots_with`] over a real body.
pub(in crate::front::mir) fn analyze_owner_roots(
    body: &Unstructured,
    llbc: &Llbc,
) -> OwnerRootPlan {
    let ty_of = |local: usize| &body.locals.locals[local].ty;
    analyze_owner_roots_with(
        llbc,
        body,
        |local| tyref_is_owner_root(ty_of(local), llbc),
        |local| tyref_mentions_owner_root(ty_of(local), llbc),
        |reg| owner_root_callee(reg, llbc),
    )
}

pub(in crate::front::mir) fn tyref_mentions_owner_root(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc).is_some_and(|node| type_node_mentions_owner_root(node, llbc, 0))
}

pub(in crate::front::mir) fn owner_root_callee(reg: &RegularCall, llbc: &Llbc) -> OwnerRootCallee {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return OwnerRootCallee::Other;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return OwnerRootCallee::Other;
    };
    owner_root_callee_of_decl(fd, llbc)
}

pub(in crate::front::mir) fn owner_root_callee_of_decl(
    fd: &FunDecl,
    llbc: &Llbc,
) -> OwnerRootCallee {
    let sig = &fd.signature;
    let path = fd.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or("");
    let first_input_is_root_borrow = sig.inputs.first().is_some_and(|ty| {
        tyref_node(ty, llbc)
            .and_then(|node| strip_ty_wrappers(node, llbc))
            .is_some_and(|node| type_node_is_owner_root(node, llbc))
    });
    if matches!(leaf, "deref" | "deref_mut") && sig.inputs.len() == 1 && first_input_is_root_borrow
    {
        return OwnerRootCallee::Deref;
    }
    if !tyref_is_owner_root(&sig.output, llbc)
        || sig
            .inputs
            .iter()
            .any(|ty| tyref_mentions_owner_root(ty, llbc))
    {
        return OwnerRootCallee::Other;
    }
    let is_new = leaf == "new"
        && impl_method_owner_for_fundecl(llbc, fd)
            .is_some_and(|(owner, _)| owner_root_type_path(&owner));
    if is_new {
        OwnerRootCallee::New
    } else {
        OwnerRootCallee::Producer
    }
}

/// The `RBigIntGcRoot::new` declaration.
pub(in crate::front::mir) fn owner_root_new_decl(llbc: &Llbc) -> Option<&FunDecl> {
    llbc.iter_local_fns()
        .find(|fd| owner_root_callee_of_decl(fd, llbc) == OwnerRootCallee::New)
}

/// Whether a producer's graph returns the value its handle roots: its body
/// erases the return place's handle. `None` when this crate's LLBC has no
/// body for the callee.
pub(in crate::front::mir) fn owner_root_producer_returns_value(
    reg: &RegularCall,
    llbc: &Llbc,
) -> Option<bool> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let body = llbc.fn_by_id(*id)?.unstructured()?;
    Some(analyze_owner_roots(&body, llbc).erases(0))
}

/// Build the owner-root erasure plan for one body. `is_root_local` says
/// which locals have the handle type exactly, `mentions_root_local` which
/// locals name it anywhere in their type.
///
/// Every handle local is erased or none is. A handle is erasable when it is
/// not a parameter, every definition is a `New` or `Producer` call with no
/// handle among the arguments or a move from another handle local, and every
/// other mention is a `Drop`, a move into another handle local (the return
/// place included), or a borrow. Each borrow `_t = &h` / `&mut h`, or its
/// reborrow `_u = &(*_t)`, is read exactly once: by the reborrow, or as the
/// sole argument of a `Deref` call.
pub(in crate::front::mir) fn analyze_owner_roots_with(
    llbc: &Llbc,
    body: &Unstructured,
    is_root_local: impl Fn(usize) -> bool,
    mentions_root_local: impl Fn(usize) -> bool,
    callee_of: impl Fn(&RegularCall) -> OwnerRootCallee,
) -> OwnerRootPlan {
    let n_locals = body.locals.locals.len();
    let arg_count = body.locals.arg_count as usize;
    let roots: bit_set::BitSet = (0..n_locals).filter(|&l| is_root_local(l)).collect();
    if roots.is_empty() {
        return OwnerRootPlan::default();
    }
    let refuse = OwnerRootPlan::default;
    if roots.iter().any(|l| (1..=arg_count).contains(&l)) {
        return refuse();
    }
    // Borrows of a handle, each defined once: `_t = &h` / `&mut h`, and the
    // reborrow `_u = &(*_t)` MIR inserts for an auto-deref receiver.
    let mut borrow_of: Vec<Option<usize>> = vec![None; n_locals];
    let mut changed = true;
    let forward_blocks = forward_reachable_mask(llbc, body);
    while changed {
        changed = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, Rvalue::Ref { place: src, .. })) =
                    stmt.stmt_kind_ref()
                else {
                    continue;
                };
                let PlaceKind::Local(dest) = place.kind else {
                    continue;
                };
                let dest = dest as usize;
                let Some(root) = owner_root_borrow_source(src, &roots, &borrow_of) else {
                    continue;
                };
                match borrow_of.get(dest).copied() {
                    Some(Some(known)) if known == root => {}
                    Some(None) if !roots.contains(dest) => {
                        borrow_of[dest] = Some(root);
                        changed = true;
                    }
                    _ => return refuse(),
                }
            }
        }
    }
    let mut watched = roots.clone();
    for (local, root) in borrow_of.iter().enumerate() {
        if root.is_some() {
            watched.insert(local);
        }
    }
    // Any other local whose type names the handle holds one in a way this
    // pass does not model (`Option<RBigIntGcRoot>`, a tuple of handles).
    if (0..n_locals).any(|l| !watched.contains(l) && mentions_root_local(l)) {
        return refuse();
    }
    // Each borrow is read exactly once: by `Deref` or by a reborrow.
    let mut borrow_uses = vec![0usize; n_locals];
    let names_watched = |op: &Operand| operand_local(Some(op)).is_some_and(|l| watched.contains(l));
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_)) => {
                    continue;
                }
                Ok(StmtKind::Assign(place, Rvalue::Ref { place: src, .. })) if matches!(place.kind, PlaceKind::Local(d) if borrow_of.get(d as usize).copied().flatten().is_some()) =>
                {
                    if let PlaceKind::Projection(inner, _) = &src.kind
                        && let PlaceKind::Local(from) = inner.kind
                    {
                        borrow_uses[from as usize] += 1;
                    }
                    continue;
                }
                // A handle moved into another handle local, the return place
                // included.
                Ok(StmtKind::Assign(place, Rvalue::Use(Operand::Move(src), _)))
                    if matches!(place.kind, PlaceKind::Local(d) if roots.contains(d as usize))
                        && matches!(src.kind, PlaceKind::Local(s) if roots.contains(s as usize)) =>
                {
                    continue;
                }
                _ => {}
            }
            if mentions_local(assert_cond_or_whole(stmt.kind_value(), false), &watched) {
                return refuse();
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Drop {
                place:
                    Place {
                        kind: PlaceKind::Local(local),
                        ..
                    },
                ..
            }) if roots.contains(*local as usize) => {}
            Ok(TermKind::Call { call, .. }) => {
                let callee = match &call.func {
                    CallFunc::Regular(reg) => callee_of(reg),
                    _ => OwnerRootCallee::Other,
                };
                let dest = match call.dest.kind {
                    PlaceKind::Local(d) => Some(d as usize),
                    _ => None,
                };
                let accepted = match callee {
                    OwnerRootCallee::New | OwnerRootCallee::Producer => {
                        dest.is_some_and(|d| roots.contains(d))
                            && !call.args.iter().any(names_watched)
                    }
                    OwnerRootCallee::Deref => match operand_local(call.args.first()) {
                        Some(borrow)
                            if call.args.len() == 1
                                && borrow_of.get(borrow).copied().flatten().is_some()
                                && !dest.is_some_and(|d| watched.contains(d)) =>
                        {
                            borrow_uses[borrow] += 1;
                            true
                        }
                        _ => false,
                    },
                    OwnerRootCallee::Other => false,
                };
                if !accepted && mentions_local(bb.terminator.kind_value(), &watched) {
                    return refuse();
                }
            }
            _ => {
                if mentions_local(
                    assert_cond_or_whole(bb.terminator.kind_value(), true),
                    &watched,
                ) {
                    return refuse();
                }
            }
        }
    }
    if borrow_of
        .iter()
        .enumerate()
        .any(|(local, root)| root.is_some() && borrow_uses[local] != 1)
    {
        return refuse();
    }
    OwnerRootPlan { roots, borrow_of }
}

/// An `Assert`'s condition, or the whole statement or terminator otherwise.
/// The overflow `check_kind` operands are a diagnostic copy: the lowering reads
/// only `cond`, and Charon's copy may name a local the condition does not.
pub(in crate::front::mir) fn assert_cond_or_whole(
    kind: &serde_json::Value,
    terminator: bool,
) -> &serde_json::Value {
    let assert = kind.get("Assert");
    let assert = if terminator {
        assert.and_then(|a| a.get("assert"))
    } else {
        assert
    };
    assert.and_then(|a| a.get("cond")).unwrap_or(kind)
}

/// The handle a `&place` borrows: a handle local itself, or `*t` for a
/// borrow `t` of one.
pub(in crate::front::mir) fn owner_root_borrow_source(
    src: &Place,
    roots: &bit_set::BitSet,
    borrow_of: &[Option<usize>],
) -> Option<usize> {
    match &src.kind {
        PlaceKind::Local(local) => roots.contains(*local as usize).then_some(*local as usize),
        PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) if elem == "Deref" => {
            match inner.kind {
                PlaceKind::Local(borrow) => borrow_of.get(borrow as usize).copied().flatten(),
                _ => None,
            }
        }
        _ => None,
    }
}

/// True for the two spellings that open a root bracket: the free function and
/// the guard's own constructor.
pub(in crate::front::mir) fn gc_root_scope_open_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    if !segments.contains(&"gc_roots") {
        return false;
    }
    matches!(segments.last(), Some(&"push_roots"))
        || (matches!(segments.last(), Some(&"new"))
            && segments.iter().any(|s| s.contains(ROOT_SCOPE_TYPE)))
}

/// True for the read that answers a bracket's own first slot index.
pub(in crate::front::mir) fn gc_root_scope_base_path(name: &str) -> bool {
    let segments: Vec<&str> = name.split("::").collect();
    matches!(segments.last(), Some(&"base")) && segments.contains(&ROOT_SCOPE_MODULE)
}

/// CFG successors the lowering keeps: `Goto` / `Call` / `Assert` / `Drop`
/// / `Switch` targets, leaving out `on_unwind`.
///
/// [`elaborate_explicit_root_closes`] and [`elaborate_vec_drops`] share
/// this edge set so a drop-elaboration fact is the same one the flow
/// graph will see.
pub(in crate::front::mir) fn model_successors(
    body: &Unstructured,
    llbc: &Llbc,
    bb: usize,
) -> Vec<usize> {
    let Some(block) = body.body.get(bb) else {
        return Vec::new();
    };
    let targets: Vec<u64> = match block.term_ref(llbc) {
        Ok(TermKind::Goto { target }) => vec![*target],
        Ok(
            TermKind::Call { target, .. }
            | TermKind::Assert { target, .. }
            | TermKind::Drop { target, .. },
        ) => vec![*target],
        Ok(TermKind::Switch { targets, .. }) => match targets {
            SwitchTargets::If(a, b) => vec![*a, *b],
            SwitchTargets::SwitchInt(_, arms, default) => arms
                .iter()
                .map(|(_, bb)| *bb)
                .chain(std::iter::once(*default))
                .collect(),
        },
        _ => Vec::new(),
    };
    targets
        .into_iter()
        .map(|t| t as usize)
        .filter(|&t| t < body.body.len())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum VecInit {
    Init,
    Uninit,
    Maybe,
}

pub(in crate::front::mir) const VEC_INIT: u8 = 1;
pub(in crate::front::mir) const VEC_UNINIT: u8 = 2;

pub(in crate::front::mir) fn vec_init_from_bits(bits: u8) -> VecInit {
    match bits {
        VEC_INIT => VecInit::Init,
        VEC_UNINIT => VecInit::Uninit,
        _ => VecInit::Maybe,
    }
}

/// An owned `alloc::vec::Vec<T>` value, not `&Vec<_>` and not `*mut Vec<_>`.
pub(in crate::front::mir) fn tyref_is_owned_vec(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_indirections(n, llbc)) else {
        return false;
    };
    if node
        .as_object()
        .is_some_and(|obj| obj.contains_key("Ref") || obj.contains_key("RawPtr"))
    {
        return false;
    }
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    llbc.type_by_id(id)
        .is_some_and(|td| td.item_meta.name_path() == "alloc::vec::Vec")
}

pub(in crate::front::mir) fn operand_moves_local(op: &Operand, local: usize) -> bool {
    matches!(
        op,
        Operand::Move(place) if matches!(place.kind, PlaceKind::Local(l) if l as usize == local)
    )
}

pub(in crate::front::mir) fn rvalue_moves_local(rvalue: &Rvalue, local: usize) -> bool {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => operand_moves_local(op, local),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            operand_moves_local(lhs, local) || operand_moves_local(rhs, local)
        }
        Rvalue::Aggregate(_, ops) => ops.iter().any(|op| operand_moves_local(op, local)),
        Rvalue::Ref { .. }
        | Rvalue::RawPtr { .. }
        | Rvalue::Len(_)
        | Rvalue::Discriminant(_)
        | Rvalue::NullaryOp(_, _)
        | Rvalue::Unknown => false,
    }
}

pub(in crate::front::mir) fn place_is_local(place: &Place, local: usize) -> bool {
    matches!(place.kind, PlaceKind::Local(l) if l as usize == local)
}

pub(in crate::front::mir) fn vec_transfer_stmt(
    s: u8,
    stmt: &majit_charon_reader::ullbc::Statement,
    local: usize,
) -> u8 {
    match stmt.stmt_kind_ref() {
        Ok(StmtKind::Assign(place, rvalue)) => {
            let mut s = s;
            if rvalue_moves_local(rvalue, local) {
                s = VEC_UNINIT;
            }
            if place_is_local(place, local) {
                s = VEC_INIT;
            }
            s
        }
        Ok(StmtKind::Assert(assert)) => {
            if operand_moves_local(&assert.cond, local) {
                VEC_UNINIT
            } else {
                s
            }
        }
        _ => s,
    }
}

pub(in crate::front::mir) fn vec_transfer_term(
    s: u8,
    term: &TermKind,
    local: usize,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> u8 {
    if term_dropped_local(term, name_of) == Some(local as u64) {
        return VEC_UNINIT;
    }
    match term {
        TermKind::Call { call, .. } => {
            let mut s = s;
            if call.args.iter().any(|op| operand_moves_local(op, local)) {
                s = VEC_UNINIT;
            }
            if let CallFunc::Dynamic(op, _) = &call.func
                && operand_moves_local(op, local)
            {
                s = VEC_UNINIT;
            }
            if place_is_local(&call.dest, local) {
                s = VEC_INIT;
            }
            s
        }
        TermKind::Switch { discr, .. } => {
            if operand_moves_local(discr, local) {
                VEC_UNINIT
            } else {
                s
            }
        }
        TermKind::Assert { assert, .. } => {
            if operand_moves_local(&assert.cond, local) {
                VEC_UNINIT
            } else {
                s
            }
        }
        _ => s,
    }
}

pub(in crate::front::mir) fn vec_local_entry_bits(
    body: &Unstructured,
    llbc: &Llbc,
    local: usize,
) -> Vec<u8> {
    let n = body.body.len();
    let mut state = vec![0u8; n];
    if n == 0 {
        return state;
    }
    let arg_count = body.locals.arg_count as usize;
    state[0] = if (1..=arg_count).contains(&local) {
        VEC_INIT
    } else {
        VEC_UNINIT
    };
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let mut work = vec![0usize];
    while let Some(bb) = work.pop() {
        let mut s = state[bb];
        for stmt in &body.body[bb].statements {
            s = vec_transfer_stmt(s, stmt, local);
        }
        if let Ok(term) = body.body[bb].term_ref(llbc) {
            s = vec_transfer_term(s, term, local, &name_of);
        }
        for succ in model_successors(body, llbc, bb) {
            if state[succ] | s != state[succ] {
                state[succ] |= s;
                work.push(succ);
            }
        }
    }
    state
}

pub(in crate::front::mir) fn vec_init_bits_at_term(
    body: &Unstructured,
    bb: usize,
    local: usize,
    entry: u8,
) -> u8 {
    let mut s = entry;
    for stmt in &body.body[bb].statements {
        s = vec_transfer_stmt(s, stmt, local);
    }
    s
}

pub(in crate::front::mir) fn vec_init_at_drop(
    body: &Unstructured,
    llbc: &Llbc,
    bb: usize,
    local: usize,
) -> VecInit {
    let entry = vec_local_entry_bits(body, llbc, local);
    let bits = vec_init_bits_at_term(body, bb, local, entry.get(bb).copied().unwrap_or(0));
    vec_init_from_bits(bits)
}

/// Charon keeps the scope-end `Drop _v` that rustc's drop elaboration
/// (`elaborate_drops`) removes for a local moved out on every path to it.
/// The same rule [`elaborate_explicit_root_closes`] applies to a root
/// guard: a `Vec<T>` local definitely moved out at its `Drop` becomes a
/// `Goto`, a definitely-initialised `Drop` stays (it USES `_v` and
/// lowers to `ll_vec_free_*`), and a mixed (maybe-initialised) `Drop`
/// is left for the lowering to decline rather than invent a drop flag.
pub(in crate::front::mir) fn elaborate_vec_drops(llbc: &Llbc, body: &mut Unstructured) {
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let vec_locals: Vec<usize> = body
        .locals
        .locals
        .iter()
        .filter(|local| tyref_is_owned_vec(&local.ty, llbc))
        .map(|local| local.index as usize)
        .collect();
    if vec_locals.is_empty() {
        return;
    }
    // `(block, dropped local, Drop/Call target)` for every block whose terminator drops a local.
    let drop_sites: Vec<(usize, u64, Option<u64>)> = (0..body.body.len())
        .filter_map(|bb_idx| {
            let term = body.body[bb_idx].term_ref(llbc).ok()?;
            let dropped = term_dropped_local(term, &name_of)?;
            let target = match term {
                TermKind::Drop { target, .. } | TermKind::Call { target, .. } => Some(*target),
                _ => None,
            };
            Some((bb_idx, dropped, target))
        })
        .collect();
    let mut rewrites: Vec<(usize, serde_json::Value)> = Vec::new();
    for local in vec_locals {
        let mut sites = drop_sites
            .iter()
            .filter(|(_, dropped, _)| *dropped == local as u64)
            .peekable();
        if sites.peek().is_none() {
            continue;
        }
        let entry = vec_local_entry_bits(body, llbc, local);
        for &(bb_idx, _, target) in sites {
            let at_term = vec_init_bits_at_term(body, bb_idx, local, entry[bb_idx]);
            if vec_init_from_bits(at_term) != VecInit::Uninit {
                continue;
            }
            let Some(target) = target else {
                continue;
            };
            rewrites.push((bb_idx, serde_json::json!({"Goto": {"target": target}})));
        }
    }
    for (bb_idx, kind) in rewrites {
        body.body[bb_idx].set_terminator_kind(kind);
    }
}

pub(in crate::front::mir) fn definitely_init_vec_drop_blocks(
    body: &Unstructured,
    llbc: &Llbc,
) -> bit_set::BitSet {
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let mut out = bit_set::BitSet::with_capacity(body.body.len());
    let vec_locals: Vec<usize> = body
        .locals
        .locals
        .iter()
        .filter(|local| tyref_is_owned_vec(&local.ty, llbc))
        .map(|local| local.index as usize)
        .collect();
    let drop_sites: Vec<(usize, u64)> = (0..body.body.len())
        .filter_map(|bb_idx| {
            let term = body.body[bb_idx].term_ref(llbc).ok()?;
            Some((bb_idx, term_dropped_local(term, &name_of)?))
        })
        .collect();
    for local in vec_locals {
        let mut sites = drop_sites
            .iter()
            .filter(|(_, dropped)| *dropped == local as u64)
            .peekable();
        if sites.peek().is_none() {
            continue;
        }
        let entry = vec_local_entry_bits(body, llbc, local);
        for &(bb_idx, _) in sites {
            let at_term = vec_init_bits_at_term(body, bb_idx, local, entry[bb_idx]);
            if vec_init_from_bits(at_term) == VecInit::Init {
                out.insert(bb_idx);
            }
        }
    }
    out
}

/// `drop(guard)`: `_t = move _g; _ = core::mem::drop(move _t)`.
///
/// The guard's destructor runs inside that call, so the call is the close of
/// the bracket `_g` opened.  Charon keeps the scope-end `Drop _g` that rustc's
/// drop elaboration (`elaborate_drops`) removes for a local moved out on every
/// path to it, so this performs both halves of that elaboration for a
/// root-bracket guard: the call becomes `Drop _g` with the guard's own drop
/// glue, and a `Drop` of the guard that only a path through such a close
/// reaches becomes a `Goto`.  `Drop _t` is always one, as `_t` never holds the
/// guard across a block boundary.
///
/// The initialisation state follows the edges the lowering keeps, which leave
/// out `on_unwind`.  A guard moved anywhere but into such a call, closed where
/// it may already be closed, or dropped where it may be either, is left as it
/// is, and [`moved_out_locals`] then keeps its bracket open.
pub(in crate::front::mir) fn elaborate_explicit_root_closes(
    llbc: &Llbc,
    body: &mut Unstructured,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) {
    struct Close {
        bb: usize,
        guard: usize,
        /// `_t` and every other local the guard passed through, closest to
        /// the call first.
        temps: Vec<usize>,
        /// Statement indices of the moves into `temps`.
        moves: Vec<usize>,
        /// The `move _g` operand's place, type and all.
        guard_place: serde_json::Value,
        /// The call's `()` destination.
        dest: usize,
    }
    let mut closes: Vec<Close> = Vec::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        if name_of(reg).as_deref() != Some("core::mem::drop") || call.args.len() != 1 {
            continue;
        }
        let (Some(Operand::Move(arg)), PlaceKind::Local(dest)) =
            (call.args.first(), &call.dest.kind)
        else {
            continue;
        };
        let PlaceKind::Local(arg_local) = arg.kind else {
            continue;
        };
        let mut guard_place = bb.terminator.kind_value()["Call"]["call"]["args"][0]["Move"].clone();
        let mut cur = arg_local as usize;
        let mut temps = Vec::new();
        let mut moves = Vec::new();
        for (i, stmt) in bb.statements.iter().enumerate().rev() {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(d) if d as usize == cur) {
                continue;
            }
            let Rvalue::Use(Operand::Move(src), _) = rvalue else {
                break;
            };
            let PlaceKind::Local(src) = src.kind else {
                break;
            };
            temps.push(cur);
            moves.push(i);
            guard_place = stmt.kind_value()["Assign"][1]["Use"][0]["Move"].clone();
            cur = src as usize;
        }
        closes.push(Close {
            bb: bb_idx,
            guard: cur,
            temps,
            moves,
            guard_place,
            dest: *dest as usize,
        });
    }
    if closes.is_empty() {
        return;
    }
    // Opener blocks per guard local.
    let mut openers: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && let CallFunc::Regular(reg) = &call.func
            && name_of(reg).is_some_and(|path| gc_root_scope_open_path(&path))
            && let PlaceKind::Local(dest) = call.dest.kind
        {
            openers.entry(dest as usize).or_default().push(bb_idx);
        }
    }
    let move_counts = local_move_counts(body, llbc);
    let drop_of = |bb: &majit_charon_reader::ullbc::BasicBlock| -> Option<usize> {
        match bb.term_ref(llbc) {
            Ok(TermKind::Drop { place, .. }) => match place.kind {
                PlaceKind::Local(l) => Some(l as usize),
                _ => None,
            },
            _ => None,
        }
    };
    let mut guards: Vec<usize> = closes.iter().map(|c| c.guard).collect();
    guards.sort_unstable();
    guards.dedup();
    let mut rewrites: Vec<(usize, serde_json::Value)> = Vec::new();
    let mut removed_moves: Vec<(usize, Vec<usize>)> = Vec::new();
    for guard in guards {
        let Some(opener_bbs) = openers.get(&guard) else {
            continue;
        };
        let mine: Vec<&Close> = closes.iter().filter(|c| c.guard == guard).collect();
        let close_bbs: bit_set::BitSet = mine.iter().map(|c| c.bb).collect();
        // Every move of the guard and of its temporaries is a link of one of
        // these chains.
        if move_counts.get(&guard).copied().unwrap_or(0) != mine.len() {
            continue;
        }
        let temps: bit_set::BitSet = mine.iter().flat_map(|c| c.temps.iter().copied()).collect();
        if temps.contains(guard)
            || temps
                .iter()
                .any(|t| move_counts.get(&t).copied().unwrap_or(0) != 1)
        {
            continue;
        }
        // The temporaries and the call's `()` result are named nowhere else.
        let dests: bit_set::BitSet = mine.iter().map(|c| c.dest).collect();
        let mut private = temps.clone();
        private.union_with(&dests);
        let mentioned_elsewhere = body.body.iter().enumerate().any(|(bb_idx, bb)| {
            if !forward_blocks[bb_idx] {
                return false;
            }
            let chain = mine.iter().find(|c| c.bb == bb_idx);
            let stmt_hit = bb.statements.iter().enumerate().any(|(i, stmt)| {
                !chain.is_some_and(|c| c.moves.contains(&i))
                    && !matches!(
                        stmt.stmt_kind_ref(),
                        Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_))
                    )
                    && mentions_local(stmt.kind_value(), &private)
            });
            let term_hit = chain.is_none()
                && !drop_of(bb).is_some_and(|l| temps.contains(l))
                && mentions_local(bb.terminator.kind_value(), &private);
            stmt_hit || term_hit
        });
        if mentioned_elsewhere {
            continue;
        }
        // The guard's own drop glue, from a `Drop` Charon already wrote for it.
        // 10.04 puts that `Drop` in an `is_cleanup` block. The forward
        // scan does not follow `on_unwind`, so when that `Drop` is not
        // in the forward set, recover the glue from the artefact's
        // `RootScope` `drop_in_place` declaration.
        let Some(template) = body
            .body
            .iter()
            .enumerate()
            .find_map(|(bb_idx, bb)| {
                if !forward_blocks[bb_idx] {
                    return None;
                }
                match bb.term_ref(llbc) {
                    Ok(TermKind::Drop { place, fn_ptr, .. })
                        if matches!(place.kind, PlaceKind::Local(l) if l as usize == guard || temps.contains(l as usize))
                            && name_of(&fn_ptr)
                                .as_deref()
                                .is_some_and(gc_root_scope_drop_glue_path) =>
                    {
                        Some(bb.terminator.kind_value().clone())
                    }
                    _ => None,
                }
            })
            .or_else(|| root_scope_drop_glue_template(llbc))
        else {
            continue;
        };
        // Maybe-initialised / maybe-uninitialised at each block entry.
        const INIT: u8 = 1;
        const UNINIT: u8 = 2;
        let n = body.body.len();
        let mut state = vec![0u8; n];
        state[0] = UNINIT;
        let mut work = vec![0usize];
        while let Some(bb) = work.pop() {
            let out = if opener_bbs.contains(&bb) {
                INIT
            } else if close_bbs.contains(bb) || drop_of(&body.body[bb]) == Some(guard) {
                UNINIT
            } else {
                state[bb]
            };
            for succ in model_successors(body, llbc, bb) {
                if state[succ] | out != state[succ] {
                    state[succ] |= out;
                    work.push(succ);
                }
            }
        }
        let mut dead: Vec<usize> = Vec::new();
        let mut sound = mine.iter().all(|c| state[c.bb] == INIT);
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            if drop_of(bb) != Some(guard) {
                continue;
            }
            match state[bb_idx] {
                UNINIT => dead.push(bb_idx),
                0 | INIT => {}
                _ => sound = false,
            }
        }
        if !sound {
            continue;
        }
        for c in &mine {
            let Ok(TermKind::Call {
                target, on_unwind, ..
            }) = body.body[c.bb].term_ref(llbc)
            else {
                continue;
            };
            let mut close = template.clone();
            close["Drop"]["place"] = c.guard_place.clone();
            close["Drop"]["target"] = serde_json::json!(target);
            close["Drop"]["on_unwind"] = serde_json::json!(on_unwind);
            rewrites.push((c.bb, close));
            removed_moves.push((c.bb, c.moves.clone()));
        }
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            let is_dead = dead.contains(&bb_idx) || drop_of(bb).is_some_and(|l| temps.contains(l));
            if is_dead && let Ok(TermKind::Drop { target, .. }) = bb.term_ref(llbc) {
                rewrites.push((bb_idx, serde_json::json!({"Goto": {"target": target}})));
            }
        }
    }
    for (bb_idx, mut moves) in removed_moves {
        moves.sort_unstable_by(|a, b| b.cmp(a));
        for i in moves {
            body.body[bb_idx].statements.remove(i);
        }
    }
    for (bb_idx, kind) in rewrites {
        body.body[bb_idx].set_terminator_kind(kind);
    }
}

/// The MIR locals this body moves out of.
///
/// Charon stamps almost every `Drop` `Conditional`: the destructor runs only
/// if the place still holds a value, and the initialisation flag that decides
/// it is not in the artefact.  For a local nothing moves, that flag is always
/// set and the drop is unconditional in fact; for a local something moves, it
/// is not, and a bracket close emitted there would rewind a shadow stack the
/// moved-to owner is still using.
///
/// A guard is never partially moved — it owns no droppable field and its own
/// fields are private — so a direct `Local` operand is the whole move set.
///
/// An operand is a move of the local only if it carries the local's declared
/// type.  Charon can spell an operand with a local index that is not the one
/// it declares: the `len` of a bounds-check `Assert` has been seen spelled as
/// a `usize` under the index of a local declared as a `RootScope`.
/// Counting that as a move would retire the guard.
///
/// A move into `core::mem::drop` is the drop of that local, not a move-out
/// to another owner.
pub(in crate::front::mir) fn moved_out_locals(body: &Unstructured, llbc: &Llbc) -> bit_set::BitSet {
    local_move_counts(body, llbc).into_keys().collect()
}

/// How many times each MIR local is moved out of, by the rule
/// [`moved_out_locals`] documents.
pub(in crate::front::mir) fn local_move_counts(
    body: &Unstructured,
    llbc: &Llbc,
) -> std::collections::HashMap<usize, usize> {
    fn ty_ref_id(v: &serde_json::Value) -> Option<u64> {
        v.get("Deduplicated")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| {
                v.get("Value")
                    .and_then(|h| h.get(0))
                    .and_then(serde_json::Value::as_u64)
            })
    }
    fn scan(
        v: &serde_json::Value,
        declared: &std::collections::HashMap<u64, u64>,
        out: &mut std::collections::HashMap<usize, usize>,
    ) {
        match v {
            serde_json::Value::Object(map) => {
                if let Some(place) = map.get("Move")
                    && let Some(local) = place
                        .get("kind")
                        .and_then(|kind| kind.get("Local"))
                        .and_then(serde_json::Value::as_u64)
                {
                    let spelled = place.get("ty").and_then(ty_ref_id);
                    let mismatched = matches!(
                        (spelled, declared.get(&local)),
                        (Some(a), Some(b)) if a != *b
                    );
                    if !mismatched {
                        *out.entry(local as usize).or_default() += 1;
                    }
                }
                for nested in map.values() {
                    scan(nested, declared, out);
                }
            }
            serde_json::Value::Array(items) => {
                for nested in items {
                    scan(nested, declared, out);
                }
            }
            _ => {}
        }
    }
    let declared: std::collections::HashMap<u64, u64> = body
        .locals
        .locals
        .iter()
        .filter_map(|l| {
            let id = match &l.ty {
                majit_charon_reader::ullbc::TyRef::Dedup { id } => *id,
                majit_charon_reader::ullbc::TyRef::Inline { value: (id, _) } => *id,
                _ => return None,
            };
            Some((l.index, id))
        })
        .collect();
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let mut out = std::collections::HashMap::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            scan(stmt.kind_value(), &declared, &mut out);
        }
        let term = bb.term_ref(llbc).ok();
        let dropped = term
            .and_then(|term| term_dropped_local(term, &name_of))
            .filter(|_| matches!(term, Some(TermKind::Call { .. })));
        let before = dropped.map(|local| out.get(&(local as usize)).copied().unwrap_or(0));
        scan(bb.terminator.kind_value(), &declared, &mut out);
        if let (Some(local), Some(before)) = (dropped, before) {
            let now = out.get(&(local as usize)).copied().unwrap_or(0);
            if now > before {
                let kept = now - 1;
                if kept == 0 {
                    out.remove(&(local as usize));
                } else {
                    out.insert(local as usize, kept);
                }
            }
        }
    }
    out
}

/// [`moved_out_locals`] of one body, computed on the first query.  Only a
/// body with a root-bracket guard asks, so the others never walk their
/// operands for it.
pub(in crate::front::mir) struct MovedOutLocals<'b> {
    pub(in crate::front::mir) body: &'b Unstructured,
    pub(in crate::front::mir) llbc: Option<&'b Llbc>,
    pub(in crate::front::mir) set: std::cell::OnceCell<bit_set::BitSet>,
}

impl<'b> MovedOutLocals<'b> {
    pub(in crate::front::mir) fn new(body: &'b Unstructured, llbc: &'b Llbc) -> Self {
        Self {
            body,
            llbc: Some(llbc),
            set: std::cell::OnceCell::new(),
        }
    }

    pub(in crate::front::mir) fn contains(&self, local: usize) -> bool {
        self.set
            .get_or_init(|| {
                moved_out_locals(
                    self.body,
                    self.llbc.expect("move set computed without an llbc"),
                )
            })
            .contains(local)
    }
}

/// `(block, local)` for every drop of a root-bracket guard whose close this
/// pass can emit — the guard's local, in the block that drops it.
///
/// A guard the body moves out of is left out: see [`moved_out_locals`].
pub(in crate::front::mir) fn root_scope_drop_sites(
    body: &Unstructured,
    llbc: &Llbc,
    moved: &MovedOutLocals<'_>,
) -> Vec<(usize, usize)> {
    let mut sites = Vec::new();
    let name_of = |reg: &RegularCall| regular_call_name_path(reg, llbc);
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(term) = bb.term_ref(llbc) else {
            continue;
        };
        let place = match term {
            TermKind::Drop { place, .. } => place,
            TermKind::Call { call, .. } => match mem_drop_moved_place(call, &name_of) {
                Some((_, place)) => place,
                None => continue,
            },
            _ => continue,
        };
        let PlaceKind::Local(local) = place.kind else {
            continue;
        };
        let Some(def_id) = output_adt_def_id_free(&place.ty, llbc) else {
            continue;
        };
        let Some(decl) = llbc.type_by_id(def_id) else {
            continue;
        };
        if gc_root_scope_type_path(&decl.item_meta.name_path()) && !moved.contains(local as usize) {
            sites.push((bb_idx, local as usize));
        }
    }
    sites
}

#[cfg(test)]
mod tests;
