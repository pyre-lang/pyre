//! Address-taking of locals: raw scalar spills, `&mut` GC-ref cells, and address escape.

use super::*;

/// Spill allocation for one borrowed primitive passed to a raw
/// scalar pointer. `Tuple<i64>` is one signed field: size 8, align 8,
/// raw gckind (`StructLayout::from_type_strings`).
pub(in crate::front::mir) const RAW_SCALAR_SPILL_OWNER: &str = "Tuple<i64>";
pub(in crate::front::mir) const RAW_SCALAR_SPILL_BYTES: usize = 8;

pub(in crate::front::mir) struct RawScalarBorrowSpill {
    pub(in crate::front::mir) fun_id: u64,
    pub(in crate::front::mir) index: usize,
    pub(in crate::front::mir) copy_out: bool,
    pub(in crate::front::mir) item_ty: ValueType,
    pub(in crate::front::mir) itemsize: usize,
    pub(in crate::front::mir) is_item_signed: bool,
}

pub(in crate::front::mir) struct RawScalarReload {
    pub(in crate::front::mir) ptr: Variable,
    pub(in crate::front::mir) copy_out: bool,
    /// Borrowed place, after peeling `&mut *p` reborrows. `None` when
    /// the borrow temporary has more than one referent. Aliases of one
    /// place share `ptr`.
    pub(in crate::front::mir) place: Option<Place>,
    pub(in crate::front::mir) item_ty: ValueType,
    pub(in crate::front::mir) itemsize: usize,
    pub(in crate::front::mir) is_item_signed: bool,
}

/// One raw address covering every argument that borrows the same place.
pub(in crate::front::mir) struct RawScalarAddressGroup {
    pub(in crate::front::mir) indices: Vec<usize>,
    pub(in crate::front::mir) word: Variable,
    pub(in crate::front::mir) place: Option<Place>,
    pub(in crate::front::mir) copy_out: bool,
    pub(in crate::front::mir) item_ty: ValueType,
    pub(in crate::front::mir) itemsize: usize,
    pub(in crate::front::mir) is_item_signed: bool,
}

/// Where an address-sized integer's bits came from.
pub(in crate::front::mir) enum SpillAddressSource {
    /// `const_expr_literal` integer. The bits are not a spill address.
    Constant,
    /// The bits name this place.
    Place(Place),
    /// A parameter, a call result, arithmetic, or a projection whose
    /// stored operand is not one place.
    Unknown,
}

impl<'a> Lowering<'a> {
    /// A GC reference behind `&mut` lives in a one-field GC cell for the
    /// duration of the borrow. Copy-in/copy-out is exact because the borrow
    /// is exclusive and does not outlive the call. Raw memory cannot hold a
    /// GC reference.
    ///
    /// The pointee has to lower to `ValueType::Ref`, and it has to be a thin
    /// pointer (raw, borrowed, or a transparent newtype of one) whose target
    /// also lowers to `Ref`.
    pub(in crate::front::mir) fn mut_ref_cell_root(&self, pointee: &TyRef) -> Option<String> {
        mut_ref_cell_root_of(
            pointee,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
    }

    pub(in crate::front::mir) fn gc_mut_ref_param_root(&self, local: usize) -> Option<String> {
        if !self.gc_mut_ref_params.contains(&local) {
            return None;
        }
        let ty = &self.body.locals.locals.get(local)?.ty;
        tyref_gc_mut_ref_cell_root(
            ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        )
    }

    pub(in crate::front::mir) fn deref_cell_param_local(&self, place: &Place) -> Option<usize> {
        let PlaceKind::Projection(inner, elem) = &place.kind else {
            return None;
        };
        if !matches!(elem, ProjectionElem::Atom(name) if name == "Deref") {
            return None;
        }
        let PlaceKind::Local(local) = &inner.kind else {
            return None;
        };
        let local = *local as usize;
        self.gc_mut_ref_params.contains(&local).then_some(local)
    }

    /// `&mut *p` where `p` is a cell, or a temp whose borrow peels to one.
    pub(in crate::front::mir) fn reborrow_cell_param(&self, place: &Place) -> Option<usize> {
        if let Some(cell) = self.deref_cell_param_local(place) {
            return Some(cell);
        }
        let PlaceKind::Projection(inner, elem) = &place.kind else {
            return None;
        };
        if !matches!(elem, ProjectionElem::Atom(name) if name == "Deref") {
            return None;
        }
        let PlaceKind::Local(local) = &inner.kind else {
            return None;
        };
        let recorded = self.atomic_ref_place.get(&(*local as usize))?;
        let followed = self.concrete_borrow_place(recorded.clone());
        self.deref_cell_param_local(&followed)
    }

    pub(in crate::front::mir) fn defined_local(
        &self,
        mir_bb: usize,
        local: usize,
    ) -> Result<Variable, LowerError> {
        self.local_var
            .get(local)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: read of MIR local {local} before any Assign — \
                     uninitialised local, not yet supported"
                ))
            })?
            .one()
    }

    /// A GC reference behind `&mut` lives in a one-field GC cell for the
    /// duration of the borrow. Copy-in/copy-out is exact because the borrow
    /// is exclusive and does not outlive the call. Raw memory cannot hold a
    /// GC reference, so `*p` reads the cell's `value` field.
    pub(in crate::front::mir) fn emit_gc_mut_ref_field_read(
        &mut self,
        mir_bb: usize,
        base: Variable,
        root: &str,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let loaded = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(loaded.clone()),
            kind: OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("value", Some(root.to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        loaded
    }

    /// A GC reference behind `&mut` lives in a one-field GC cell for the
    /// duration of the borrow. Copy-in/copy-out is exact because the borrow
    /// is exclusive and does not outlive the call. Raw memory cannot hold a
    /// GC reference, so the caller writes the local into the cell and passes
    /// the cell.
    pub(in crate::front::mir) fn emit_gc_mut_ref_cell(
        &mut self,
        mir_bb: usize,
        root: &str,
        value: Variable,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let cell = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let root_owned = root.to_string();
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(cell.clone()),
            kind: OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    Vec::new(),
                    root_owned.clone(),
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(root_owned.clone())),
            },
        });
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: cell.clone(),
                field: FieldDescriptor::new("value", Some(root_owned)),
                value: LinkArg::Value(value),
                ty: ValueType::Ref(None),
            },
        });
        cell
    }

    pub(in crate::front::mir) fn gc_mut_ref_signature(
        &self,
        func: &CallFunc,
        call_arg_tys: &[Option<TyRef>],
    ) -> Option<(Option<u64>, Vec<(usize, String)>)> {
        match func {
            CallFunc::Regular(reg) => {
                let id = regular_call_fun_decl_id(&reg.kind)?;
                let fd = self.llbc.fn_by_id(id)?;
                // A residual callee is entered through its Rust ABI, which
                // takes the address of the word. The cell exists only inside
                // a traced body.
                if fun_decl_is_residual(fd, self.dont_look_inside) {
                    return None;
                }
                let mut cells = Vec::new();
                for (index, ty) in fd.signature.inputs.iter().enumerate() {
                    let root = tyref_gc_mut_ref_cell_root(
                        ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    )
                    .or_else(|| {
                        if !tyref_is_type_var(ty, self.llbc) {
                            return None;
                        }
                        call_arg_tys
                            .get(index)
                            .and_then(Option::as_ref)
                            .and_then(|arg_ty| {
                                tyref_gc_mut_ref_cell_root(
                                    arg_ty,
                                    self.llbc,
                                    self.tombstoned_leaves,
                                    self.gc_struct_ids,
                                    &self.raw_alloc_flavor_structs,
                                )
                            })
                    });
                    let Some(root) = root else {
                        continue;
                    };
                    cells.push((index, root));
                }
                if cells.is_empty() {
                    None
                } else {
                    Some((Some(id), cells))
                }
            }
            // A `dyn Trait` / unknown callee has no fun-decl inputs. Wrapping
            // from operand types alone also wraps `AtomicPtr::from_ptr` of
            // `PyCode.w_globals`, which must stay a `FieldRead` so Acquire
            // load folds to `record_quasiimmut_field`.
            CallFunc::Dynamic(..) | CallFunc::Unknown => None,
        }
    }

    pub(in crate::front::mir) fn call_target_is_fun(
        &self,
        target: &CallTarget,
        fun_id: u64,
    ) -> bool {
        if target.fun_decl_id() == Some(fun_id) {
            return true;
        }
        if target.fun_decl_id().is_some() {
            return false;
        }
        let Some(decl_path) = self
            .llbc
            .fn_by_id(fun_id)
            .map(|fd| fd.item_meta.name_path())
        else {
            return false;
        };
        match target {
            CallTarget::FunctionPath { segments, .. } => path_names_decl(&decl_path, segments),
            CallTarget::Method {
                resolved_path: Some(path),
                ..
            } => path_names_decl(&decl_path, &path.segments),
            _ => false,
        }
    }

    pub(in crate::front::mir) fn classify_gc_mut_ref_arg(
        &self,
        arg_local: Option<usize>,
        operand: Option<&Operand>,
    ) -> Result<GcMutRefArg, LowerError> {
        if let Some(local) = arg_local {
            if self.gc_mut_ref_params.contains(&local) {
                return Ok(GcMutRefArg::CellParam(local));
            }
            let Some(recorded) = self.atomic_ref_place.get(&local).cloned() else {
                return Err(gc_mut_ref_not_whole_local());
            };
            let followed = self.concrete_borrow_place(recorded);
            if let PlaceKind::Local(slot) = &followed.kind {
                let slot = *slot as usize;
                if self.gc_mut_ref_params.contains(&slot) {
                    return Ok(GcMutRefArg::CellParam(slot));
                }
                return Ok(GcMutRefArg::WholeLocal(slot));
            }
            if let Some(cell) = self.deref_cell_param_local(&followed) {
                return Ok(GcMutRefArg::CellParam(cell));
            }
            if self.mut_ref_cell_root(&followed.ty).is_some() {
                return Ok(GcMutRefArg::Place(followed));
            }
            return Err(gc_mut_ref_not_whole_local());
        }
        // A tuple-field projection of `&mut T` has no bare local. Ordinary
        // call lowering of `f(args.0)` still wraps that place as a cell.
        if let Some(Operand::Copy(place) | Operand::Move(place)) = operand {
            let followed = self.concrete_borrow_place(clone_place(place));
            if let Some(cell) = self.deref_cell_param_local(&followed) {
                return Ok(GcMutRefArg::CellParam(cell));
            }
            if tyref_gc_mut_ref_cell_root(
                &followed.ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
            .is_some()
                || self.mut_ref_cell_root(&followed.ty).is_some()
            {
                return Ok(GcMutRefArg::Place(followed));
            }
        }
        Err(gc_mut_ref_not_whole_local())
    }

    pub(in crate::front::mir) fn install_gc_mut_ref_call(
        &mut self,
        mir_bb: usize,
        op_kind: &mut OpKind,
        sig: &Option<(Option<u64>, Vec<(usize, String)>)>,
        arg_locals: &[Option<usize>],
        arg_operands: &[Operand],
        resolved_args: &[Variable],
    ) -> Result<Vec<GcMutRefCopy>, LowerError> {
        let plan = {
            let Some((fun_id, cells)) = sig else {
                return self.unwrap_residual_gc_mut_ref(mir_bb, op_kind, arg_locals, resolved_args);
            };
            let OpKind::Call { target, args, .. } = &*op_kind else {
                return Ok(Vec::new());
            };
            if let Some(fun_id) = fun_id
                && !self.call_target_is_fun(target, *fun_id)
            {
                return Ok(Vec::new());
            }
            if args.len() != resolved_args.len() {
                return Ok(Vec::new());
            }
            let mut plan = Vec::new();
            for (index, root) in cells {
                let Some(arg) = args.get(*index).and_then(LinkArg::as_variable) else {
                    continue;
                };
                if arg != &resolved_args[*index] {
                    continue;
                }
                let arg_local = arg_locals.get(*index).copied().flatten();
                plan.push((
                    *index,
                    root.clone(),
                    self.classify_gc_mut_ref_arg(arg_local, arg_operands.get(*index))?,
                ));
            }
            plan
        };
        let mut copies = Vec::new();
        let mut replacements = Vec::new();
        for (index, root, class) in plan {
            match class {
                GcMutRefArg::WholeLocal(local) => {
                    let Some(current) = self.local_var.get(local).and_then(Clone::clone) else {
                        return Err(gc_mut_ref_not_whole_local());
                    };
                    let cell = self.emit_gc_mut_ref_cell(mir_bb, &root, current.one()?);
                    copies.push(GcMutRefCopy::Local {
                        cell: cell.clone(),
                        root,
                        local,
                    });
                    replacements.push((index, cell));
                }
                GcMutRefArg::CellParam(local) => {
                    let Some(current) = self.local_var.get(local).and_then(Clone::clone) else {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: GC reference cell {local} before its definition"
                        )));
                    };
                    replacements.push((index, current.one()?));
                }
                GcMutRefArg::Place(place) => {
                    let current = self.resolve_place(mir_bb, clone_place(&place))?;
                    let cell = self.emit_gc_mut_ref_cell(mir_bb, &root, current);
                    copies.push(GcMutRefCopy::Place {
                        cell: cell.clone(),
                        root,
                        place,
                    });
                    replacements.push((index, cell));
                }
            }
        }
        let OpKind::Call { args, .. } = op_kind else {
            return Ok(copies);
        };
        for (index, cell) in replacements {
            args[index] = LinkArg::Value(cell);
        }
        Ok(copies)
    }

    /// Residual / `dont_look_inside` callees take the pointer word. A caller
    /// that already holds a cell unwraps `value` and passes that pointer.
    /// The callee writes through the pointer; it does not assign the word.
    pub(in crate::front::mir) fn unwrap_residual_gc_mut_ref(
        &mut self,
        mir_bb: usize,
        op_kind: &mut OpKind,
        arg_locals: &[Option<usize>],
        resolved_args: &[Variable],
    ) -> Result<Vec<GcMutRefCopy>, LowerError> {
        let OpKind::Call { args, .. } = &*op_kind else {
            return Ok(Vec::new());
        };
        if args.len() != resolved_args.len() {
            return Ok(Vec::new());
        }
        let mut plan = Vec::new();
        for (index, arg) in args.iter().enumerate() {
            let Some(arg) = arg.as_variable() else {
                continue;
            };
            if arg != &resolved_args[index] {
                continue;
            }
            let arg_local = arg_locals.get(index).copied().flatten();
            if let Ok(GcMutRefArg::CellParam(local)) = self.classify_gc_mut_ref_arg(arg_local, None)
            {
                let Some(root) = self.gc_mut_ref_param_root(local) else {
                    continue;
                };
                plan.push((index, root, local));
            }
        }
        let copies = Vec::new();
        let mut replacements = Vec::new();
        for (index, root, local) in plan {
            let Some(cell) = self.local_var.get(local).and_then(Clone::clone) else {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: GC reference cell {local} before its definition"
                )));
            };
            let word = self.emit_gc_mut_ref_field_read(mir_bb, cell.one()?, &root);
            replacements.push((index, word));
        }
        let OpKind::Call { args, .. } = op_kind else {
            return Ok(copies);
        };
        for (index, word) in replacements {
            args[index] = LinkArg::Value(word);
        }
        Ok(copies)
    }

    pub(in crate::front::mir) fn copy_out_gc_mut_ref(
        &mut self,
        mir_bb: usize,
        copies: &[GcMutRefCopy],
    ) -> Result<(), LowerError> {
        for copy in copies {
            match copy {
                GcMutRefCopy::Local { cell, root, local } => {
                    let loaded = self.emit_gc_mut_ref_field_read(mir_bb, cell.clone(), root);
                    self.local_var[*local] = Some(LocalValue::One(loaded));
                }
                GcMutRefCopy::Place { cell, root, place } => {
                    let loaded = self.emit_gc_mut_ref_field_read(mir_bb, cell.clone(), root);
                    let PlaceKind::Projection(inner, elem) = &place.kind else {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: GC reference copy-out place is not a projection"
                        )));
                    };
                    self.emit_projection_write(
                        mir_bb,
                        clone_place(inner),
                        elem.clone(),
                        LinkArg::Value(loaded),
                        &place.ty,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Parameters where the callee wants a raw address and the operand
    /// is a borrow of the pointee word.
    ///
    /// `history.py` `getkind` banks every raw `Ptr` as `int`, and
    /// [`pygraph_initial_block`] already records that parameter as
    /// `Int`. `Rvalue::Ref` of a primitive aliases the scalar, so the
    /// call would pass the word where the callee dereferences an
    /// address (`*hash_out = digest`). rustc's `&mut x` → `*mut T`
    /// coercion is `_r = &mut x; _p = &raw mut (*_r)`: the call
    /// operand is already a raw pointer whose value is that word, and
    /// that pointer is spilled too. A raw pointer that is already an
    /// address (a parameter, or `&raw mut (*p)` of a real pointer) is
    /// not. [`attach_aliasing_raw_arguments`] passes the spill address
    /// when another raw argument names the spilled place. An `&mut i64`
    /// parameter is the word on both sides and is left alone. A byte
    /// pointer stays `Ref`.
    pub(in crate::front::mir) fn raw_scalar_borrow_spills(
        &self,
        func: &CallFunc,
        arg_tys: &[Option<TyRef>],
        arg_locals: &[Option<usize>],
    ) -> Vec<RawScalarBorrowSpill> {
        let CallFunc::Regular(reg) = func else {
            return Vec::new();
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return Vec::new();
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Vec::new();
        };
        let mut spills = Vec::new();
        for (index, declared) in fd.signature.inputs.iter().enumerate() {
            if raw_scalar_address_value_type(declared, self.llbc).is_none() {
                continue;
            }
            let Some(passed) = arg_tys.get(index).and_then(Option::as_ref) else {
                continue;
            };
            let Some(param_node) = tyref_peel_one_raw_ptr_node(declared, self.llbc) else {
                continue;
            };
            let (arg_node, copy_out) = if tyref_is_ref_of_primitive(passed, self.llbc) {
                let Some(arg_node) = tyref_peel_one_ref_node(passed, self.llbc) else {
                    continue;
                };
                (
                    arg_node,
                    raw_ptr_kind_is_mut(declared, self.llbc) && ref_kind_is_mut(passed, self.llbc),
                )
            } else if raw_scalar_address_value_type(passed, self.llbc).is_some()
                && self
                    .raw_scalar_arg_aliases_primitive_word(arg_locals.get(index).copied().flatten())
            {
                let Some(arg_node) = tyref_peel_one_raw_ptr_node(passed, self.llbc) else {
                    continue;
                };
                (
                    arg_node,
                    raw_ptr_kind_is_mut(declared, self.llbc)
                        && raw_ptr_kind_is_mut(passed, self.llbc),
                )
            } else {
                continue;
            };
            let param_pointee = TyRef::Other(param_node.clone());
            let arg_pointee = TyRef::Other(arg_node.clone());
            let Some((item_ty, itemsize, is_item_signed)) = self.raw_word_descr(&param_pointee)
            else {
                continue;
            };
            let Some((arg_ty, arg_size, _)) = self.raw_word_descr(&arg_pointee) else {
                continue;
            };
            // `Tuple<i64>` is the spill allocation: one signed field,
            // 8 bytes, align 8, raw gckind. `CallControl::struct_layout_for`
            // synthesizes that layout, and `jtransform.py`
            // `_rewrite_raw_malloc` residualizes `ll_raw_malloc_fixedsize`.
            // A wider pointee (128-bit) never reaches
            // [`raw_scalar_address_value_type`].
            if itemsize == 0
                || itemsize > RAW_SCALAR_SPILL_BYTES
                || arg_size != itemsize
                || arg_ty != item_ty
            {
                continue;
            }
            spills.push(RawScalarBorrowSpill {
                fun_id: *id,
                index,
                copy_out,
                item_ty,
                itemsize,
                is_item_signed,
            });
        }
        spills
    }

    /// Store the aliased word into a fresh raw address and pass that
    /// address. Borrows of one place share the address. A copied local
    /// is a different place (`let y = x` still aliases the word, and
    /// `&x` with `&y` stay two addresses). The malloc is
    /// `OpKind::RawMalloc` so `_rewrite_raw_malloc` emits the fixed-size
    /// residual; a direct call of `ll_raw_malloc_fixedsize` would look
    /// inside the helper. A pointer or reference in the result can be this
    /// address: a payload, a field, or an opaque body whose layout is not
    /// zero-sized. The call is left unlowered and the spill is not freed.
    /// A generic argument counts when a field stores it.
    pub(in crate::front::mir) fn install_raw_scalar_address_call(
        &mut self,
        mir_bb: usize,
        op_kind: &mut OpKind,
        spills: &[RawScalarBorrowSpill],
        arg_locals: &[Option<usize>],
        arg_operands: &[Operand],
        resolved_args: &[Variable],
        dest_ty: &TyRef,
    ) -> Result<Vec<RawScalarReload>, LowerError> {
        if spills.is_empty() {
            return Ok(Vec::new());
        }
        let fun_id = spills[0].fun_id;
        let groups = {
            let OpKind::Call { target, args, .. } = &*op_kind else {
                return Ok(Vec::new());
            };
            if !self.call_target_is_fun(target, fun_id) || args.len() != resolved_args.len() {
                return Ok(Vec::new());
            }
            let mut groups: Vec<RawScalarAddressGroup> = Vec::new();
            for spill in spills {
                if spill.fun_id != fun_id {
                    continue;
                }
                let Some(arg) = args.get(spill.index).and_then(LinkArg::as_variable) else {
                    continue;
                };
                if arg != &resolved_args[spill.index] {
                    continue;
                }
                let place =
                    self.resolved_borrow_place(arg_locals.get(spill.index).copied().flatten());
                if let Some(group) = groups.iter_mut().find(|group| {
                    raw_scalar_groups_share(
                        group,
                        place.as_ref(),
                        &spill.item_ty,
                        spill.itemsize,
                        spill.is_item_signed,
                    )
                }) {
                    group.indices.push(spill.index);
                    group.copy_out |= spill.copy_out;
                    continue;
                }
                groups.push(RawScalarAddressGroup {
                    indices: vec![spill.index],
                    word: resolved_args[spill.index].clone(),
                    place,
                    copy_out: spill.copy_out,
                    item_ty: spill.item_ty.clone(),
                    itemsize: spill.itemsize,
                    is_item_signed: spill.is_item_signed,
                });
            }
            groups
        };
        let mut groups = groups;
        let mut tracked = spills
            .iter()
            .filter(|spill| spill.fun_id == fun_id)
            .map(|spill| RawScalarBorrowSpill {
                fun_id: spill.fun_id,
                index: spill.index,
                copy_out: spill.copy_out,
                item_ty: spill.item_ty.clone(),
                itemsize: spill.itemsize,
                is_item_signed: spill.is_item_signed,
            })
            .collect::<Vec<_>>();
        if !groups.is_empty() {
            let alias_indices = self.attach_aliasing_raw_arguments(
                fun_id,
                &mut groups,
                arg_locals,
                arg_operands,
                mir_bb,
            )?;
            let template = tracked.first().map(|spill| RawScalarBorrowSpill {
                fun_id: spill.fun_id,
                index: spill.index,
                copy_out: false,
                item_ty: spill.item_ty.clone(),
                itemsize: spill.itemsize,
                is_item_signed: spill.is_item_signed,
            });
            if let Some(template) = template {
                for index in alias_indices {
                    tracked.push(RawScalarBorrowSpill {
                        fun_id: template.fun_id,
                        index,
                        copy_out: false,
                        item_ty: template.item_ty.clone(),
                        itemsize: template.itemsize,
                        is_item_signed: template.is_item_signed,
                    });
                }
            } else if !alias_indices.is_empty() {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: raw scalar spill address would escape through an unspilled raw argument"
                )));
            }
        }
        if !groups.is_empty()
            && (self.raw_scalar_dest_holds_thin_spill_address(dest_ty)
                || callee_returns_spill_address(self.llbc, fun_id, &tracked))
        {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: raw scalar spill address would escape through the call result"
            )));
        }
        let bb_id = self.emit_bb(mir_bb);
        let mut reloads = Vec::new();
        let mut replacements = Vec::new();
        for group in groups {
            let ptr = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(ptr.clone()),
                kind: OpKind::raw_malloc(RAW_SCALAR_SPILL_OWNER, false, None),
            });
            let offset = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(offset.clone()),
                kind: OpKind::ConstInt(0),
            });
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::RawStore {
                    base: ptr.clone(),
                    offset,
                    value: group.word,
                    item_ty: group.item_ty.clone(),
                    itemsize: group.itemsize,
                    is_item_signed: group.is_item_signed,
                },
            });
            for index in group.indices {
                replacements.push((index, ptr.clone()));
            }
            reloads.push(RawScalarReload {
                ptr,
                copy_out: group.copy_out,
                place: group.place,
                item_ty: group.item_ty,
                itemsize: group.itemsize,
                is_item_signed: group.is_item_signed,
            });
        }
        let OpKind::Call { args, .. } = op_kind else {
            return Ok(reloads);
        };
        for (index, ptr) in replacements {
            args[index] = LinkArg::Value(ptr);
        }
        Ok(reloads)
    }

    /// The spill is freed on this block. The call result is that address
    /// when a raw pointer or a reference occurs anywhere the caller
    /// receives it: the result itself, an `Option` or `Result` payload,
    /// a tuple or array element, a struct or enum field, or an opaque
    /// body. `identity(&x) -> *const i64` would then deref a freed
    /// allocation. A status word (`hash_out` returns `i64`) is not this
    /// address, at any nesting depth. A type already queued is not walked
    /// again; its first visit already queued every payload.
    ///
    /// `dont_look_inside_return_token` projects `Result<T, E>` through
    /// `T` when `E` is the error carrier. That `Err` rides
    /// `BH_LAST_EXC_VALUE` and is not the call result.
    /// [`tyref_is_zero_sized`] is a body that stores nothing, so its
    /// generic arguments are not this address. A field's `TypeVar` is
    /// replaced through the types nested inside that field.
    /// `PhantomData<T>` has no such field. A `TypeVar` with no argument
    /// stays unclassified. A `TypeVar` inside a generic argument is the
    /// caller's variable and stays unknown. A callee that casts this
    /// address into its return slot returns the same bits as an
    /// integer, and so does a callee with no body, a call whose callee
    /// returns those bits, a store of those bits through a pointer, a
    /// store into a global, a reload through a reference to the pointer
    /// (`q = &p; *q`), and a drop whose glue can publish them. `*p`
    /// loads the pointee. `*q` loads a value selected by the address
    /// when `q` was computed from it. `*p = clean` writes that pointee
    /// when `p` is still the spill pointer. `q = &p; *q = clean`
    /// overwrites that local. A store through a pointer computed from
    /// the address escapes. A callee that returns `p == null` leaves
    /// that condition for the caller's switch.
    /// A comparison of the spill pointer with a null constant, returned
    /// as a status, stays a status. A copy, `&*p`, and a cast whose
    /// destination holds every address bit still name that pointer.
    /// `(p as usize) & MASK == 0` reads its low bits, and `p as usize as
    /// u8` keeps only that byte, so neither comparison does. A comparison with any other value does
    /// not either: the spill address is not the pointer that value still
    /// names. A discriminant of a place that carries the address reads
    /// those bits back: `((p as usize >> k) & 3) as u8` is a four-variant
    /// tag. A discriminant of a comparison stays that comparison.
    /// `Len` of a fixed array is that array's const generic. `Len` of
    /// a slice is the length recorded for the slice
    /// (`rvalue_length_metadata`), so the payload's address stays out
    /// of the length. A slice with no recorded length stays unlowered.
    /// Two or more comparisons in one aggregate can encode the address.
    /// A switch or an
    /// assertion on `p == null` takes the same edge after the spill
    /// moves. `p < 0` does not. Arithmetic that consumes a comparison,
    /// a store of it through a pointer or into a global, and an `Index`
    /// on that store can
    /// rebuild the address. An `Index` offset carries the comparison
    /// onto the selected element too. Separate field or constant-index
    /// stores of those comparisons add up on the place. A store through
    /// an index that is not a constant counts as many slots. Drop glue
    /// receives that count with the pointer. A call result written
    /// through a pointer or into a global publishes that address at any
    /// depth. A whole-local assignment replaces the address that local
    /// held. A field of an aggregate keeps only that field's address.
    /// A field of that field keeps only that leaf.
    /// A copy or move of that aggregate keeps those fields.
    /// A field of a union carries every field's address. An indirect
    /// call through a tainted function pointer leaves the call unlowered.
    /// A statement, rvalue, or place this walk cannot classify does too.
    /// `Nop` does not carry the address. `SetDiscriminant` does not
    /// when its place is independent of the address. `NullaryOp`
    /// names a type.
    pub(in crate::front::mir) fn raw_scalar_spill_result_escapes(&self, ty: &TyRef) -> bool {
        let mut pending = vec![clone_tyref(ty)];
        let mut seen = Vec::new();
        while let Some(ty) = pending.pop() {
            let key = spill_ty_identity(&ty);
            if seen.iter().any(|seen| seen == &key) {
                continue;
            }
            seen.push(key);
            if tyref_is_raw_pointer(&ty, self.llbc) || output_type_is_ref(&ty, self.llbc) {
                return true;
            }
            match spill_address_payloads(&ty, self.llbc, self.static_addrs.error_carrier) {
                SpillPayloads::Unclassified => return true,
                SpillPayloads::Types(types) => pending.extend(types),
            }
        }
        false
    }

    /// Dest type of a spilled call: a thin scalar pointer (`*mut i64`)
    /// in the result can be the spilled address. A GCREF / `*mut
    /// PyObject` (`Option<GCREF>`) is a heap object, not that word.
    fn raw_scalar_dest_holds_thin_spill_address(&self, ty: &TyRef) -> bool {
        let mut pending = vec![clone_tyref(ty)];
        let mut seen = Vec::new();
        while let Some(ty) = pending.pop() {
            let key = spill_ty_identity(&ty);
            if seen.iter().any(|seen| seen == &key) {
                continue;
            }
            seen.push(key);
            if raw_scalar_address_value_type(&ty, self.llbc).is_some()
                || tyref_is_ref_of_primitive(&ty, self.llbc)
            {
                return true;
            }
            if tyref_is_raw_pointer(&ty, self.llbc) || output_type_is_ref(&ty, self.llbc) {
                continue;
            }
            match spill_address_payloads(&ty, self.llbc, self.static_addrs.error_carrier) {
                SpillPayloads::Unclassified => return true,
                SpillPayloads::Types(types) => pending.extend(types),
            }
        }
        false
    }

    /// A raw parameter that is not a spill still names an address.
    /// When that address is the spilled place, pass the spill pointer:
    /// `*p = 1; return *q` stays one word. An address-sized integer
    /// whose [`pointer_place_of_local`] names that place is the same
    /// pointer (`type_holds_full_address`): `q = &mut x as *mut i64 as
    /// usize` is passed the spill pointer, and that integer does not
    /// copy the pointee back. A field projection is that integer when
    /// its stored operand names a place (`constant_field_path`,
    /// `spill_address_source`). An integer literal
    /// (`const_expr_literal`) stays the caller's value. The pointee
    /// word itself (`hash` beside `&mut keyhash`) and a plain integer
    /// that was never a pointer (`hash_known`) stay the caller's
    /// value: they hold the scalar, not its address. A raw pointer
    /// whose declared type is not a thin scalar address (`*mut
    /// PyObject`) cannot be `q` for this word. A `*mut i64` with no
    /// referent still leaves the call unlowered, and so does a call
    /// result, arithmetic, or `pair.0` whose operand is not a single
    /// place. A pointer or integer that covers the spilled place
    /// leaves the call unlowered. Only a thin scalar pointer shares
    /// the copy-out. A pointer or that integer packed into a tuple,
    /// struct, enum, or array is still the caller's place
    /// (`local_hides_spill_alias`): the aggregate is not rewritten, so
    /// the call stays unlowered. A field that names another place
    /// stays, and so does a tuple of integer literals. A tuple of
    /// address-sized integers this walk cannot trace stays unlowered.
    /// A projected store is part of the same value. An aggregate whose
    /// type can hold a pointer and whose operands this walk cannot see
    /// stays unlowered. A pointer to such an aggregate is the same
    /// check on the pointee.
    pub(in crate::front::mir) fn attach_aliasing_raw_arguments(
        &self,
        fun_id: u64,
        groups: &mut [RawScalarAddressGroup],
        arg_locals: &[Option<usize>],
        arg_operands: &[Operand],
        mir_bb: usize,
    ) -> Result<Vec<usize>, LowerError> {
        let Some(fd) = self.llbc.fn_by_id(fun_id) else {
            return Ok(Vec::new());
        };
        let mut attached = Vec::new();
        for (index, declared) in fd.signature.inputs.iter().enumerate() {
            if groups.iter().any(|group| group.indices.contains(&index)) {
                continue;
            }
            let raw = tyref_is_raw_pointer(declared, self.llbc);
            // `usize` / `u64` can be the spilled address with the pointer
            // bits still intact. A narrower integer cannot name it.
            let address_int = !raw && type_holds_full_address(declared, self.llbc);
            let arg_local = arg_locals.get(index).copied().flatten();
            if !raw && !address_int {
                let hides = match arg_local {
                    Some(local) => self.local_hides_spill_alias(local, groups, 0),
                    None => self.raw_scalar_spill_result_escapes(declared),
                };
                if hides {
                    return Err(self.unspilled_raw_argument(mir_bb));
                }
                continue;
            }
            let Some(op) = arg_operands.get(index) else {
                return Err(self.unspilled_raw_argument(mir_bb));
            };
            // A literal is not the spilled address. A place this walk
            // cannot name stays unlowered unless the operand cannot be
            // that address: the pointee word, a plain integer, or a
            // raw pointer that is not a thin scalar address.
            // `spill_address_source` follows a field when its stored
            // operand names one place.
            let place = match self.spill_address_source(op, 0) {
                SpillAddressSource::Constant if !raw => continue,
                SpillAddressSource::Place(place) => place,
                SpillAddressSource::Constant | SpillAddressSource::Unknown => {
                    if self.unplaced_argument_is_independent_of_spill(declared, op, groups, raw) {
                        continue;
                    }
                    return Err(self.unspilled_raw_argument(mir_bb));
                }
            };
            let mut exact = None;
            let mut covered = false;
            for (group_index, group) in groups.iter().enumerate() {
                let Some(group_place) = group.place.as_ref() else {
                    covered = true;
                    continue;
                };
                if places_alias(group_place, &place) {
                    if exact.is_some() {
                        covered = true;
                    }
                    exact = Some(group_index);
                } else if place_extends(group_place, &place) || place_extends(&place, group_place) {
                    covered = true;
                }
            }
            if covered {
                return Err(self.unspilled_raw_argument(mir_bb));
            }
            let Some(group_index) = exact else {
                // `q = &pair` names `pair`, not the spilled word. A field
                // of `pair` can still be that word.
                if self.place_payload_hides_spill_alias(&place, groups, 0) {
                    return Err(self.unspilled_raw_argument(mir_bb));
                }
                continue;
            };
            // `Copy` of the i64 local is that place without being an
            // address. `q = &mut x as usize` still attaches.
            if !raw && self.operand_is_plain_integer_word(op, groups, 0) {
                continue;
            }
            if raw {
                // A slice or a byte pointer carries more than this word.
                if raw_scalar_address_value_type(declared, self.llbc).is_none() {
                    return Err(self.unspilled_raw_argument(mir_bb));
                }
                if raw_ptr_kind_is_mut(declared, self.llbc) {
                    groups[group_index].copy_out = true;
                }
            }
            groups[group_index].indices.push(index);
            attached.push(index);
        }
        Ok(attached)
    }

    pub(in crate::front::mir) fn unspilled_raw_argument(&self, mir_bb: usize) -> LowerError {
        LowerError::Unsupported(format!(
            "bb{mir_bb}: raw scalar spill address would escape through an unspilled raw argument"
        ))
    }

    /// `spill_address_source` could not name this operand. A `*mut i64`
    /// with no referent might still be `q` in `*p = 1; return *q`. A
    /// pointer to an ADT, a byte pointer, and a slice cannot be that
    /// `q` through the declared type. An address-sized integer that is
    /// the pointee or a plain word (`hash`, `hash_known`) cannot
    /// either.
    fn unplaced_argument_is_independent_of_spill(
        &self,
        declared: &TyRef,
        op: &Operand,
        groups: &[RawScalarAddressGroup],
        raw: bool,
    ) -> bool {
        if raw {
            return raw_scalar_address_value_type(declared, self.llbc).is_none();
        }
        self.operand_is_plain_integer_word(op, groups, 0)
    }

    /// Integer bits that were never a pointer: the spilled word, a
    /// parameter, or a `Use` of one of those. A call result, a
    /// projected store, arithmetic, and `q = &mut x as usize` are not.
    fn operand_is_plain_integer_word(
        &self,
        op: &Operand,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        match op {
            Operand::Const(value) => self.const_is_integer_literal(value),
            Operand::Copy(place) | Operand::Move(place) => {
                self.place_is_plain_integer_word(place, groups, depth)
            }
        }
    }

    fn place_is_plain_integer_word(
        &self,
        place: &Place,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        if tyref_is_raw_pointer(&place.ty, self.llbc)
            || !type_holds_full_address(&place.ty, self.llbc)
        {
            return false;
        }
        if groups.iter().any(|group| {
            group
                .place
                .as_ref()
                .is_some_and(|group_place| places_alias(group_place, place))
        }) && self.place_holds_aliased_primitive_word(place)
        {
            return true;
        }
        let PlaceKind::Local(id) = &place.kind else {
            return false;
        };
        let local = *id as usize;
        if self.local_is_call_dest(local) || self.local_has_projected_store(local) {
            return false;
        }
        if self.pointer_place_of_local(local, 0).is_some() {
            return false;
        }
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind() else {
                    continue;
                };
                let PlaceKind::Local(index) = dest.kind else {
                    continue;
                };
                if index as usize != local {
                    continue;
                }
                let Rvalue::Use(op, _) = rvalue else {
                    return false;
                };
                if !self.operand_is_plain_integer_word(&op, groups, depth + 1) {
                    return false;
                }
            }
        }
        true
    }

    /// `pair = (q,)` with `q` naming the spilled place. The aggregate
    /// still holds the caller's address, so the call cannot be spilled.
    /// A field that names another place does not. An integer literal
    /// does not (`const_expr_literal`). An address-sized integer this
    /// walk cannot trace does (`spill_address_source`): a parameter, a
    /// call result, or `pair.0` with no single stored place. Depth
    /// above 8 is the same bound as [`pointer_place_of_local`].
    pub(in crate::front::mir) fn local_hides_spill_alias(
        &self,
        local: usize,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        let Some(ty) = self.mir_local_ty(local) else {
            return true;
        };
        if depth > 8 {
            return self.raw_scalar_spill_result_escapes(&ty);
        }
        if self.projected_store_hides_spill_alias(local, groups, depth + 1) {
            return true;
        }
        if self.local_is_call_dest(local) && self.raw_scalar_spill_result_escapes(&ty) {
            return true;
        }
        let mut saw = false;
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(id) = dest.kind else {
                    continue;
                };
                if id as usize != local {
                    continue;
                }
                saw = true;
                if self.rvalue_hides_spill_alias(&rvalue, &ty, groups, depth + 1) {
                    return true;
                }
            }
        }
        if !saw {
            return self.raw_scalar_spill_result_escapes(&ty);
        }
        false
    }

    pub(in crate::front::mir) fn mir_local_ty(&self, local: usize) -> Option<TyRef> {
        self.body
            .locals
            .locals
            .iter()
            .find(|decl| decl.index as usize == local)
            .map(|decl| clone_tyref(&decl.ty))
    }

    fn call_dests(&self) -> &CallDestLocals {
        self.call_dest_locals.get_or_init(|| {
            let mut dests = bit_set::BitSet::new();
            let mut unique: Vec<Option<(RegularCall, Vec<Operand>)>> = Vec::new();
            for (bb_idx, block) in self.body.body.iter().enumerate() {
                if !self.forward_reached[bb_idx] {
                    continue;
                }
                if let Ok(TermKind::Call { call, .. }) = block.term_ref(self.llbc)
                    && let PlaceKind::Local(id) = call.dest.kind
                {
                    let local = id as usize;
                    if dests.contains(local) {
                        if local < unique.len() {
                            unique[local] = None;
                        }
                    } else if let CallFunc::Regular(reg) = &call.func {
                        if unique.len() <= local {
                            unique.resize_with(local + 1, || None);
                        }
                        unique[local] = Some((reg.clone(), call.args.clone()));
                    }
                    dests.insert(local);
                }
            }
            CallDestLocals { dests, unique }
        })
    }

    fn local_is_call_dest(&self, local: usize) -> bool {
        self.call_dests().dests.contains(local)
    }

    /// Single argument of a unique `is_ptr_identity_cast` that writes
    /// `local`. Every other call destination stays unnamed.
    fn ptr_identity_cast_operand(&self, local: usize) -> Option<Operand> {
        let (reg, args) = self.call_dests().unique.get(local)?.as_ref()?;
        if args.len() == 1 && self.is_ptr_identity_cast(reg) {
            Some(args[0].clone())
        } else {
            None
        }
    }

    /// `pair.0 = q` after `pair` was built. The field write is the value
    /// the call passes.
    pub(in crate::front::mir) fn projected_store_hides_spill_alias(
        &self,
        local: usize,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                if !matches!(dest.kind, PlaceKind::Projection(..)) {
                    continue;
                }
                if place_root_local(&dest) != Some(local) {
                    continue;
                }
                let field_ty = clone_tyref(&dest.ty);
                if self.rvalue_hides_spill_alias(&rvalue, &field_ty, groups, depth) {
                    return true;
                }
            }
        }
        false
    }

    pub(in crate::front::mir) fn rvalue_hides_spill_alias(
        &self,
        rvalue: &Rvalue,
        result_ty: &TyRef,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        match rvalue {
            Rvalue::Aggregate(_, operands) => operands
                .iter()
                .any(|op| self.operand_hides_spill_alias(op, groups, depth)),
            Rvalue::Repeat(operand, _, _, _) => {
                self.operand_hides_spill_alias(operand, groups, depth)
            }
            Rvalue::Use(operand, _) => self.operand_hides_spill_alias(operand, groups, depth),
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                self.referenced_place_hides_spill_alias(place, groups, depth)
            }
            Rvalue::Cast(kind, operand, target_ty)
                if cast_preserves_spill_address(kind, Some(target_ty), self.llbc) =>
            {
                self.operand_hides_spill_alias(operand, groups, depth)
            }
            Rvalue::UnaryOp(op, operand)
                if unary_op_is_cast(op)
                    && cast_preserves_spill_address(op, Some(result_ty), self.llbc) =>
            {
                self.operand_hides_spill_alias(operand, groups, depth)
            }
            _ => self.raw_scalar_spill_result_escapes(result_ty),
        }
    }

    pub(in crate::front::mir) fn operand_hides_spill_alias(
        &self,
        op: &Operand,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        let place = match op {
            Operand::Const(_) => return false,
            Operand::Copy(place) | Operand::Move(place) => place,
        };
        let raw = tyref_is_raw_pointer(&place.ty, self.llbc);
        let address_int = !raw && type_holds_full_address(&place.ty, self.llbc);
        if raw || address_int {
            match self.spill_address_source(op, depth) {
                SpillAddressSource::Constant => return false,
                SpillAddressSource::Unknown => return true,
                SpillAddressSource::Place(named) => {
                    if spill_place_aliases(&named, groups) {
                        return true;
                    }
                    return self.place_payload_hides_spill_alias(&named, groups, depth);
                }
            }
        }
        match &place.kind {
            PlaceKind::Local(id) => self.local_hides_spill_alias(*id as usize, groups, depth),
            _ => self.raw_scalar_spill_result_escapes(&place.ty),
        }
    }

    /// Place named by an address-sized integer or raw pointer operand.
    /// A literal is [`SpillAddressSource::Constant`]. A field is the
    /// place stored in that field (`constant_field_path`). Depth above
    /// 8 is the same bound as [`pointer_place_of_local`].
    pub(in crate::front::mir) fn spill_address_source(
        &self,
        op: &Operand,
        depth: u8,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        match op {
            Operand::Const(value) => {
                if self.const_is_integer_literal(value) || self.const_is_trait_const(value) {
                    SpillAddressSource::Constant
                } else {
                    SpillAddressSource::Unknown
                }
            }
            Operand::Copy(place) | Operand::Move(place) => self.place_address_source(place, depth),
        }
    }

    pub(in crate::front::mir) fn const_is_integer_literal(
        &self,
        value: &serde_json::Value,
    ) -> bool {
        self.llbc
            .const_expr_literal(value)
            .is_some_and(|literal| literal.get("Scalar").is_some())
    }

    /// A `TraitConst` is a type-level value (`T::SIZE`), not the address
    /// of a spilled local. The selected impl's literal is folded later
    /// (`resolved_trait_const_op`); an unresolved `Clause` is the same
    /// bits for this walk.
    fn const_is_trait_const(&self, value: &serde_json::Value) -> bool {
        self.llbc
            .const_expr_kind(value)
            .and_then(|kind| kind.as_object().map(|obj| obj.contains_key("TraitConst")))
            .unwrap_or(false)
    }

    pub(in crate::front::mir) fn place_address_source(
        &self,
        place: &Place,
        depth: u8,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        match &place.kind {
            PlaceKind::Local(id) => {
                let local = *id as usize;
                if self.local_has_projected_store(local) {
                    return SpillAddressSource::Unknown;
                }
                if self.local_is_call_dest(local) {
                    // A pointer cast is the same address (`lltype.cast_pointer`,
                    // `rffi.cast`). The dest names the place its operand names.
                    if let Some(arg) = self.ptr_identity_cast_operand(local) {
                        return self.spill_address_source(&arg, depth + 1);
                    }
                    return SpillAddressSource::Unknown;
                }
                if let Some(named) = self.pointer_place_of_local(local, 0) {
                    return self.named_place(named);
                }
                self.local_address_source(local, depth + 1)
            }
            PlaceKind::Projection(_, _) => {
                let Some((root, path)) = constant_field_path(place, self.llbc) else {
                    return SpillAddressSource::Unknown;
                };
                if path.is_empty() {
                    return self.local_address_source(root as usize, depth + 1);
                }
                self.field_address_source(root as usize, &path, depth + 1)
            }
            PlaceKind::Global { .. } | PlaceKind::Unknown => SpillAddressSource::Unknown,
        }
    }

    pub(in crate::front::mir) fn named_place(&self, place: Place) -> SpillAddressSource {
        SpillAddressSource::Place(self.concrete_borrow_place(place))
    }

    /// Whole-local assigns of `local`. A call result and a projected
    /// store are not a single place.
    pub(in crate::front::mir) fn local_address_source(
        &self,
        local: usize,
        depth: u8,
    ) -> SpillAddressSource {
        if depth > 8 || self.local_is_call_dest(local) || self.local_has_projected_store(local) {
            return SpillAddressSource::Unknown;
        }
        let mut found = None;
        let mut saw = false;
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(id) = dest.kind else {
                    continue;
                };
                if id as usize != local {
                    continue;
                }
                saw = true;
                found = merge_address_source(
                    found,
                    self.rvalue_address_source(&rvalue, depth + 1, Some(&dest.ty)),
                );
                if matches!(found, Some(SpillAddressSource::Unknown)) {
                    return SpillAddressSource::Unknown;
                }
            }
        }
        if !saw {
            return SpillAddressSource::Unknown;
        }
        found.unwrap_or(SpillAddressSource::Unknown)
    }

    pub(in crate::front::mir) fn local_has_projected_store(&self, local: usize) -> bool {
        self.projected_store_roots
            .get_or_init(|| {
                let mut out = bit_set::BitSet::new();
                for (bb_idx, block) in self.body.body.iter().enumerate() {
                    if !self.forward_reached[bb_idx] {
                        continue;
                    }
                    for stmt in &block.statements {
                        if let Ok(StmtKind::Assign(dest, _)) = stmt.stmt_kind_ref()
                            && matches!(dest.kind, PlaceKind::Projection(..))
                            && let Some(root) = place_root_local(dest)
                        {
                            out.insert(root);
                        }
                    }
                }
                out
            })
            .contains(local)
    }

    /// Operand stored at `path` inside `root`. A call that writes `root`
    /// hides every field: the statement assigns are not ordered against
    /// that call. A store of a child of `path` hides the field too.
    pub(in crate::front::mir) fn field_address_source(
        &self,
        root: usize,
        path: &[usize],
        depth: u8,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        if path.is_empty() {
            return self.local_address_source(root, depth);
        }
        if self.local_is_call_dest(root) {
            return SpillAddressSource::Unknown;
        }
        let mut found = None;
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                if let PlaceKind::Local(id) = dest.kind {
                    if id as usize != root {
                        continue;
                    }
                    found = merge_address_source(
                        found,
                        self.rvalue_field_source(&rvalue, path, depth + 1, Some(&dest.ty)),
                    );
                    if matches!(found, Some(SpillAddressSource::Unknown)) {
                        return SpillAddressSource::Unknown;
                    }
                    continue;
                }
                if place_root_local(&dest) != Some(root) {
                    continue;
                }
                let Some((stored_root, stored_path)) = constant_field_path(&dest, self.llbc) else {
                    return SpillAddressSource::Unknown;
                };
                if stored_root as usize != root {
                    continue;
                }
                let stored_path = stored_path.as_slice();
                let next = if stored_path == path {
                    self.rvalue_address_source(&rvalue, depth + 1, Some(&dest.ty))
                } else if path.starts_with(stored_path) {
                    self.rvalue_field_source(
                        &rvalue,
                        &path[stored_path.len()..],
                        depth + 1,
                        Some(&dest.ty),
                    )
                } else if stored_path.starts_with(path) {
                    return SpillAddressSource::Unknown;
                } else {
                    continue;
                };
                found = merge_address_source(found, next);
                if matches!(found, Some(SpillAddressSource::Unknown)) {
                    return SpillAddressSource::Unknown;
                }
            }
        }
        found.unwrap_or(SpillAddressSource::Unknown)
    }

    pub(in crate::front::mir) fn rvalue_address_source(
        &self,
        rvalue: &Rvalue,
        depth: u8,
        dest_ty: Option<&TyRef>,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        match rvalue {
            Rvalue::Use(op, _) => self.spill_address_source(op, depth + 1),
            Rvalue::Cast(kind, op, target_ty)
                if cast_preserves_spill_address(kind, Some(target_ty), self.llbc) =>
            {
                self.spill_address_source(op, depth + 1)
            }
            Rvalue::UnaryOp(op, operand)
                if unary_op_is_cast(op) && cast_preserves_spill_address(op, dest_ty, self.llbc) =>
            {
                self.spill_address_source(operand, depth + 1)
            }
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                self.named_place(place.clone())
            }
            _ => SpillAddressSource::Unknown,
        }
    }

    pub(in crate::front::mir) fn rvalue_field_source(
        &self,
        rvalue: &Rvalue,
        path: &[usize],
        depth: u8,
        dest_ty: Option<&TyRef>,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        if path.is_empty() {
            return self.rvalue_address_source(rvalue, depth, dest_ty);
        }
        match rvalue {
            Rvalue::Aggregate(_, operands) => {
                let Some(op) = operands.get(path[0]) else {
                    return SpillAddressSource::Unknown;
                };
                self.operand_field_source(op, &path[1..], depth + 1)
            }
            Rvalue::Repeat(operand, _, _, _) => {
                self.operand_field_source(operand, &path[1..], depth + 1)
            }
            Rvalue::Use(op, _) => self.operand_field_source(op, path, depth + 1),
            Rvalue::Cast(kind, op, target_ty)
                if cast_preserves_spill_address(kind, Some(target_ty), self.llbc) =>
            {
                self.operand_field_source(op, path, depth + 1)
            }
            Rvalue::UnaryOp(op, operand)
                if unary_op_is_cast(op) && cast_preserves_spill_address(op, dest_ty, self.llbc) =>
            {
                self.operand_field_source(operand, path, depth + 1)
            }
            _ => SpillAddressSource::Unknown,
        }
    }

    pub(in crate::front::mir) fn operand_field_source(
        &self,
        op: &Operand,
        path: &[usize],
        depth: u8,
    ) -> SpillAddressSource {
        if depth > 8 {
            return SpillAddressSource::Unknown;
        }
        if path.is_empty() {
            return self.spill_address_source(op, depth);
        }
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Const(_) => return SpillAddressSource::Unknown,
        };
        let Some((root, mut place_path)) = constant_field_path(place, self.llbc) else {
            return SpillAddressSource::Unknown;
        };
        place_path.extend_from_slice(path);
        self.field_address_source(root as usize, &place_path, depth + 1)
    }

    /// `q = &raw mut x` names `x`. `x` aliasing a spill is the packed
    /// alias. The pointee can itself hold another such pointer.
    pub(in crate::front::mir) fn referenced_place_hides_spill_alias(
        &self,
        place: &Place,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        if spill_place_aliases(place, groups) {
            return true;
        }
        self.place_payload_hides_spill_alias(place, groups, depth)
    }

    pub(in crate::front::mir) fn place_payload_hides_spill_alias(
        &self,
        place: &Place,
        groups: &[RawScalarAddressGroup],
        depth: u8,
    ) -> bool {
        match &place.kind {
            PlaceKind::Local(id) => {
                let local = *id as usize;
                // A whole local that is one pointer or scalar word is the
                // named slot (`&mut rooted_class`). It does not pack the
                // spilled field the way `pair` does in `q = &pair`.
                if let Some(ty) = self.mir_local_ty(local)
                    && (tyref_is_raw_pointer(&ty, self.llbc)
                        || tyref_is_primitive_scalar(&ty, self.llbc))
                {
                    return self.projected_store_hides_spill_alias(local, groups, depth + 1);
                }
                self.local_hides_spill_alias(local, groups, depth)
            }
            _ => self.raw_scalar_spill_result_escapes(&place.ty),
        }
    }

    /// Place a raw or borrowed local names, following one cast or copy.
    /// `None` when the local is a parameter, was assigned twice, or the
    /// assignment is not an address.
    pub(in crate::front::mir) fn pointer_place_of_local(
        &self,
        local: usize,
        depth: u8,
    ) -> Option<Place> {
        if depth > 8 {
            return None;
        }
        if let Some(place) = self.atomic_ref_place.get(&local) {
            return Some(self.concrete_borrow_place(clone_place(place)));
        }
        let mut found = None;
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(index) = dest.kind else {
                    continue;
                };
                if index as usize != local {
                    continue;
                }
                let place = self.rvalue_pointer_place(&rvalue, depth, Some(&dest.ty))?;
                if found.is_some() {
                    return None;
                }
                found = Some(place);
            }
        }
        found.map(|place| self.concrete_borrow_place(place))
    }

    pub(in crate::front::mir) fn rvalue_pointer_place(
        &self,
        rvalue: &Rvalue,
        depth: u8,
        dest_ty: Option<&TyRef>,
    ) -> Option<Place> {
        match rvalue {
            Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => Some(place.clone()),
            Rvalue::Use(op, _) => self.operand_pointer_place(op, depth),
            Rvalue::Cast(kind, op, target_ty)
                if cast_preserves_spill_address(kind, Some(target_ty), self.llbc) =>
            {
                self.operand_pointer_place(op, depth)
            }
            Rvalue::UnaryOp(op, operand)
                if unary_op_is_cast(op) && cast_preserves_spill_address(op, dest_ty, self.llbc) =>
            {
                self.operand_pointer_place(operand, depth)
            }
            _ => None,
        }
    }

    pub(in crate::front::mir) fn operand_pointer_place(
        &self,
        op: &Operand,
        depth: u8,
    ) -> Option<Place> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => {
                let PlaceKind::Local(id) = place.kind else {
                    return None;
                };
                self.pointer_place_of_local(id as usize, depth + 1)
            }
            Operand::Const(_) => None,
        }
    }

    /// Place a borrow temporary names, after peeling `&mut *p`.
    /// `None` when that temporary is missing or was assigned more than once.
    pub(in crate::front::mir) fn resolved_borrow_place(
        &self,
        arg_local: Option<usize>,
    ) -> Option<Place> {
        let local = arg_local?;
        Some(self.concrete_borrow_place(self.argument_borrow_place(local)?))
    }

    pub(in crate::front::mir) fn emit_raw_scalar_word_load(
        &mut self,
        mir_bb: usize,
        reload: &RawScalarReload,
    ) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let offset = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(offset.clone()),
            kind: OpKind::ConstInt(0),
        });
        let loaded = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(loaded.clone()),
            kind: OpKind::RawLoad {
                base: reload.ptr.clone(),
                offset,
                item_ty: reload.item_ty.clone(),
                itemsize: reload.itemsize,
                is_item_signed: reload.is_item_signed,
            },
        });
        loaded
    }

    /// Read a mutable raw out-parameter back into the borrowed place,
    /// then free the spill. A field or element projection is written
    /// with `emit_projection_write`. A borrow temporary with more than
    /// one referent is not written back. A shared borrow or a `*const`
    /// parameter is not written back. Python exceptions ride the success
    /// edge, which is this block; the unwind edge is dropped.
    pub(in crate::front::mir) fn reload_raw_scalar_addresses(
        &mut self,
        mir_bb: usize,
        reloads: &[RawScalarReload],
    ) -> Result<(), LowerError> {
        for reload in reloads {
            if reload.copy_out
                && let Some(place) = reload.place.clone()
            {
                let ty = clone_tyref(&place.ty);
                match place.kind {
                    PlaceKind::Local(index) => {
                        let index = index as usize;
                        if !(self.local_decl_is_primitive(index) && index < self.local_var.len()) {
                            return Err(LowerError::Unsupported(format!(
                                "bb{mir_bb}: raw scalar out-parameter write-back is not a primitive local"
                            )));
                        }
                        let loaded = self.emit_raw_scalar_word_load(mir_bb, reload);
                        self.local_var[index] = Some(LocalValue::One(loaded));
                    }
                    PlaceKind::Projection(inner, elem) => {
                        let loaded = self.emit_raw_scalar_word_load(mir_bb, reload);
                        self.emit_projection_write(
                            mir_bb,
                            *inner,
                            elem,
                            LinkArg::Value(loaded),
                            &ty,
                        )?;
                    }
                    PlaceKind::Global { .. } | PlaceKind::Unknown => {
                        return Err(LowerError::Unsupported(format!(
                            "bb{mir_bb}: raw scalar out-parameter write-back is not a primitive local or a projection"
                        )));
                    }
                }
            }
            let bb_id = self.emit_bb(mir_bb);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::RawFree {
                    ptr: reload.ptr.clone(),
                },
            });
        }
        Ok(())
    }

    pub(in crate::front::mir) fn argument_borrow_place(&self, local: usize) -> Option<Place> {
        if let Some(place) = self.atomic_ref_place.get(&local) {
            return Some(clone_place(place));
        }
        let mut found = None;
        for (bb_idx, block) in self.body.body.iter().enumerate() {
            if !self.forward_reached[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                let Ok(StmtKind::Assign(dest, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(index) = dest.kind else {
                    continue;
                };
                if index as usize != local {
                    continue;
                }
                let Some(place) = borrowed_place_referent(&rvalue) else {
                    continue;
                };
                if found.is_some() {
                    return None;
                }
                found = Some(place);
            }
        }
        found
    }
}

/// Payloads of `ty` the caller receives as this value.
///
/// [`SpillPayloads::Unclassified`] is a wrapper whose payload did not
/// parse, or an opaque body whose layout does not prove it stores
/// nothing, so the address may still be inside. An alias whose target
/// is not a type contributes no payload. `dont_look_inside_return_token`
/// projects `Result<T, E>` through `T` when `E` is the error carrier.
/// A generic argument is a payload when a field's type is that argument.
pub(in crate::front::mir) enum SpillPayloads {
    Types(Vec<TyRef>),
    Unclassified,
}

pub(in crate::front::mir) fn spill_ty_identity(ty: &TyRef) -> String {
    match ty {
        TyRef::Dedup { id } => format!("#{id}"),
        TyRef::Inline { value: (id, _) } => format!("*{id}"),
        TyRef::Other(value) => value.to_string(),
    }
}

pub(in crate::front::mir) fn spill_one_payload(slot: Option<&serde_json::Value>) -> SpillPayloads {
    match slot.and_then(|slot| serde_json::from_value(slot.clone()).ok()) {
        Some(ty) => SpillPayloads::Types(vec![ty]),
        None => SpillPayloads::Unclassified,
    }
}

pub(in crate::front::mir) fn spill_payloads_from_slots(
    slots: &[serde_json::Value],
) -> SpillPayloads {
    let mut types = Vec::with_capacity(slots.len());
    for slot in slots {
        let Ok(ty) = serde_json::from_value(slot.clone()) else {
            return SpillPayloads::Unclassified;
        };
        types.push(ty);
    }
    SpillPayloads::Types(types)
}

pub(in crate::front::mir) fn spill_address_payloads(
    ty: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> SpillPayloads {
    // A `TypeVar` whose argument was not substituted still names a
    // payload. `Option<T>` reaches that `T` after the `Option` peel.
    if tyref_is_type_var(ty, llbc) {
        return SpillPayloads::Unclassified;
    }
    // The carrier's `Err` is not the call result. A nested
    // `Result<T, Carrier>` is the same ABI for that `Err`.
    if crate::front::result_exc::tyref_is_result_of_carrier(ty, llbc, spec) {
        return match crate::front::result_exc::tyref_result_ok(ty, llbc) {
            Some(ok) => SpillPayloads::Types(vec![ok]),
            None => SpillPayloads::Unclassified,
        };
    }
    if crate::front::result_exc::tyref_is_result(ty, llbc)
        && let Some(ok) = crate::front::result_exc::tyref_result_ok(ty, llbc)
        && let Some(err) = crate::front::result_exc::tyref_result_err(ty, llbc)
    {
        return SpillPayloads::Types(vec![ok, err]);
    }
    if crate::front::result_exc::tyref_is_option(ty, llbc)
        && let Some(payload) = crate::front::result_exc::tyref_option_payload(ty, llbc)
    {
        return SpillPayloads::Types(vec![payload]);
    }
    let Some(node) = tyref_node(ty, llbc).and_then(|node| strip_ty_indirections(node, llbc)) else {
        return SpillPayloads::Types(Vec::new());
    };
    if node.get("Array").is_some() || node.get("Slice").is_some() {
        let key = if node.get("Array").is_some() {
            "Array"
        } else {
            "Slice"
        };
        return spill_one_payload(
            node.get(key)
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first()),
        );
    }
    let Some(adt_value) = node.get("Adt") else {
        return SpillPayloads::Types(Vec::new());
    };
    let Some(adt) = adt_value.as_object() else {
        return SpillPayloads::Unclassified;
    };
    // Tuple elements are the generic arguments, and each one is stored.
    if type_decl_ref_builtin(adt_value) == Some("Tuple") {
        let Some(slots) = type_decl_ref_generics(adt, llbc)
            .and_then(|generics| generics.get("types"))
            .and_then(serde_json::Value::as_array)
        else {
            return SpillPayloads::Unclassified;
        };
        return spill_payloads_from_slots(slots);
    }
    // No nominal body: `str` and the other builtins carry no address.
    let Some(id) = type_decl_ref_adt_id(adt) else {
        return SpillPayloads::Types(Vec::new());
    };
    let Some(decl) = llbc.type_by_id(id) else {
        return SpillPayloads::Unclassified;
    };
    // Size 0 stores nothing. `PhantomData<T>` is this, and so is an
    // opaque marker whose layout says the same. `T` is not a field.
    if tyref_is_zero_sized(ty, llbc) {
        return SpillPayloads::Types(Vec::new());
    }
    if decl.published_struct_copy {
        return SpillPayloads::Unclassified;
    }
    match &decl.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            spill_substituted_fields(fields, node, llbc)
        }
        TypeDeclKind::Enum(variants) => {
            let mut types = Vec::new();
            for variant in variants {
                match spill_substituted_fields(&variant.fields, node, llbc) {
                    SpillPayloads::Unclassified => return SpillPayloads::Unclassified,
                    SpillPayloads::Types(fields) => types.extend(fields),
                }
            }
            SpillPayloads::Types(types)
        }
        TypeDeclKind::Alias(body) => {
            let aliased = if let Ok(aliased) = serde_json::from_value(body.clone()) {
                Some(aliased)
            } else if let Some(aliased) = body.get("aliased_ty")
                && let Ok(aliased) = serde_json::from_value(aliased.clone())
            {
                Some(aliased)
            } else {
                None
            };
            match aliased {
                Some(aliased) => match substitute_spill_typevars(&aliased, node, llbc) {
                    Some(ty) => SpillPayloads::Types(vec![ty]),
                    None => SpillPayloads::Unclassified,
                },
                None => SpillPayloads::Types(Vec::new()),
            }
        }
        // The field list is absent. A generic argument is the pointee of
        // `NonNull<i64>`, not proof that the word itself is a scalar.
        TypeDeclKind::Opaque | TypeDeclKind::Unknown => SpillPayloads::Unclassified,
    }
}

/// Field types of `fields`, with every `TypeVar` inside the field
/// replaced by the argument on `owner`. A missing argument leaves the
/// stored type unknown.
pub(in crate::front::mir) fn spill_substituted_fields(
    fields: &[majit_charon_reader::ullbc::FieldDecl],
    owner: &serde_json::Value,
    llbc: &Llbc,
) -> SpillPayloads {
    let mut types = Vec::with_capacity(fields.len());
    for field in fields {
        let Some(ty) = substitute_spill_typevars(&field.ty, owner, llbc) else {
            return SpillPayloads::Unclassified;
        };
        types.push(ty);
    }
    SpillPayloads::Types(types)
}

/// `ty` with each depth-0 `TypeVar` replaced by `owner`'s generic
/// argument. `Option<T>` stores `T` inside the `Option`, so the
/// substitution walks the field. A missing argument, a free variable,
/// or a nested binder is [`None`]: the payload is still unknown.
/// A `TypeVar` inside the looked-up argument is the caller's variable
/// and is not replaced again.
pub(in crate::front::mir) fn substitute_spill_typevars(
    ty: &TyRef,
    owner: &serde_json::Value,
    llbc: &Llbc,
) -> Option<TyRef> {
    let node = tyref_node(ty, llbc)?;
    let value = substitute_spill_value(node, owner, llbc, &mut Vec::new(), 0)?;
    Some(TyRef::Other(value))
}

/// `arg` was copied out of an aggregate's generic arguments, so it is
/// already in the caller's scope. A `TypeVar` there names the caller's
/// binder, not the aggregate's. `Deduplicated` bodies are resolved. A
/// cycle, a missing body, or a walk deeper than 32 stays unknown.
pub(in crate::front::mir) fn spill_argument_carries_typevar(
    node: &serde_json::Value,
    llbc: &Llbc,
    dedup_stack: &mut Vec<u64>,
    depth: u32,
) -> bool {
    if depth > 32 {
        return true;
    }
    if let Some(id) = node.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        if dedup_stack.contains(&id) {
            return true;
        }
        dedup_stack.push(id);
        let found = match llbc.dedup_body(id) {
            Some(body) => spill_argument_carries_typevar(body, llbc, dedup_stack, depth + 1),
            None => true,
        };
        dedup_stack.pop();
        return found;
    }
    if node.get("TypeVar").is_some() {
        return true;
    }
    if let Some(items) = node.as_array() {
        return items
            .iter()
            .any(|item| spill_argument_carries_typevar(item, llbc, dedup_stack, depth + 1));
    }
    if let Some(obj) = node.as_object() {
        return obj
            .values()
            .any(|value| spill_argument_carries_typevar(value, llbc, dedup_stack, depth + 1));
    }
    false
}

pub(in crate::front::mir) fn substitute_spill_value(
    node: &serde_json::Value,
    owner: &serde_json::Value,
    llbc: &Llbc,
    dedup_stack: &mut Vec<u64>,
    depth: u32,
) -> Option<serde_json::Value> {
    if depth > 32 {
        return None;
    }
    if let Some(id) = node.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        if dedup_stack.contains(&id) {
            return None;
        }
        dedup_stack.push(id);
        let body = llbc.dedup_body(id)?;
        let replaced = substitute_spill_value(body, owner, llbc, dedup_stack, depth + 1);
        dedup_stack.pop();
        return replaced;
    }
    if node.get("TypeVar").is_some() {
        let bound = node.get("TypeVar")?.get("Bound")?.as_array()?;
        let binder = bound.first()?.as_u64()?;
        let index = typevar_bound_index(node)?;
        if binder != 0 {
            return None;
        }
        let arg = type_decl_ref_generics(owner.get("Adt")?.as_object()?, llbc)?
            .get("types")?
            .as_array()?
            .get(index as usize)?;
        if spill_argument_carries_typevar(arg, llbc, &mut Vec::new(), 0) {
            return None;
        }
        return Some(arg.clone());
    }
    if let Some(items) = node.as_array() {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            out.push(substitute_spill_value(
                item,
                owner,
                llbc,
                dedup_stack,
                depth + 1,
            )?);
        }
        return Some(serde_json::Value::Array(out));
    }
    if let Some(obj) = node.as_object() {
        let mut out = serde_json::Map::new();
        for (key, value) in obj {
            out.insert(
                key.clone(),
                substitute_spill_value(value, owner, llbc, dedup_stack, depth + 1)?,
            );
        }
        return Some(serde_json::Value::Object(out));
    }
    Some(node.clone())
}

/// The callee returns the spill address as a scalar. `p as usize` is a
/// cast of the pointer parameter into the return slot, and
/// `usize as *const i64` reads those bits back (`getkind` banks a raw
/// `Ptr` as `int`). A comparison of the spill pointer with a null
/// constant, such as `p == null`, is a status when it is returned. A
/// copy, `&*p`, and a cast whose destination holds every address bit
/// still name that pointer. `(p as usize) & MASK == 0` reads its low
/// bits, and `p as usize as u8` keeps only that byte, so neither
/// comparison does. A
/// comparison with any other value does not either: the spill address
/// is not the pointer that value still names. A discriminant of a place
/// that carries the address reads those bits back: `((p as usize >> k) &
/// 3) as u8` is a four-variant tag. A discriminant of a comparison stays
/// that comparison. `Len` of a fixed array is that array's const
/// generic. `Len` of a slice is the length recorded for the slice
/// (`rvalue_length_metadata`), so the payload's address stays out of
/// the length. A slice with no recorded length stays unlowered. Two
/// or more comparisons in one aggregate can
/// encode the address. A switch or an assertion on `p == null` takes
/// the same edge after the spill moves. `p < 0` does not. Arithmetic
/// that consumes a comparison,
/// a store of it through a pointer or into a global, and an `Index` on
/// that store can rebuild the address. An `Index` offset carries the
/// comparison onto the selected element too. Separate field or
/// constant-index stores of those comparisons add up on the place.
/// A store through an index that is not a constant counts as many
/// slots. Drop glue receives that count with the pointer.
/// A call result written through a pointer or into a global publishes
/// that address at any depth. A whole-local assignment replaces the
/// address that local held. A field of an aggregate keeps only that
/// field's address. A field of that field keeps only that leaf.
/// A field that still holds the spill pointer dereferences as that
/// pointer. A constant index selects that element.
/// A copy or move of that aggregate keeps those fields.
/// A copy or move of a field keeps that field's fields.
/// A field of a union carries every field's address.
/// An indirect call through a tainted function pointer leaves the call
/// unlowered.
/// A statement, rvalue, or place this walk cannot classify does too.
/// `Nop` does not carry the address. `SetDiscriminant` does not when
/// its place is independent of the address. `NullaryOp` names a type.
/// A call writes that address when its callee returns it, and a store
/// of those bits through a pointer or into a global does too.
/// `q = &p; *q` reloads the same bits; `*p` loads the pointee.
/// `*q` loads a value selected by the address when `q` was computed
/// from it. `*p = clean` writes that pointee when `p` is still the
/// spill pointer. `q = &p; *q = clean` overwrites that local. A store
/// through a pointer computed from the address escapes. A callee that
/// returns `p == null` leaves that condition for the caller's switch.
/// A callee that returns `p < 0` stays unlowered.
/// A raw argument that names the spilled place is relocated with it,
/// and so is an address-sized integer whose provenance is that place
/// (`attach_aliasing_raw_arguments`, `type_holds_full_address`,
/// `pointer_place_of_local`). A raw argument whose referent is
/// unknown, or a raw argument or integer that covers that place, stays
/// unlowered. An integer literal keeps the caller's value
/// (`const_expr_literal`). An address-sized integer this walk cannot
/// place stays unlowered (`spill_address_source`), including `pair.0`,
/// a parameter, and a call result. A pointer or address-sized integer
/// packed into a tuple,
/// struct, enum, or array still names the caller's place
/// (`local_hides_spill_alias`), so that call stays unlowered. A field
/// that names another place stays.
/// `&*p` rebuilds the address. A local keeps every depth assigned to
/// it, so `q = p as *const *const i64` and `q = &p` both remain and
/// `*q` can reload the address. `q = &bits` while `bits` is still
/// clean keeps that name (`mark_local_address`), so a later store of
/// the address into `bits` is visible through `*q`. `pair = (q,)` keeps
/// that name on the field (`FieldSlot::referent`), so `*pair.0` sees the
/// same store. `q = &pair.0` names that field (`place_referent`), so a
/// later store into the leaf is visible through `*q`. `read(q)` lifts
/// those depths onto the parameter (`address_through_referent`). Drop
/// glue receives a pointer to the dropped place (`drop_in_place`) and
/// sees a field referent lifted from that place (`dropped_address`).
/// A null check in that place is still a null check in the glue. A callee
/// with no unstructured body can return the bits, so that call stays
/// unlowered.
pub(in crate::front::mir) fn callee_returns_spill_address(
    llbc: &Llbc,
    fun_id: u64,
    spills: &[RawScalarBorrowSpill],
) -> bool {
    let entry: Vec<LocalAddress> = spills
        .iter()
        .filter(|spill| spill.fun_id == fun_id)
        .map(|spill| {
            let mut local = plain_local(spill.index as u64 + 1, 1, 0);
            // Pointee stores through this parameter update the spill.
            // A pointer computed from it is a different destination.
            // `attach_aliasing_raw_arguments` lists a raw parameter
            // that names this place, so `*q` reads the relocated word.
            local.direct = true;
            local
        })
        .collect();
    if entry.is_empty() {
        return false;
    }
    let escape = function_address_escape(llbc, fun_id, &entry, &mut Vec::new());
    // `p == null` is the same branch after the spill moves. `p < 0` is
    // not: the sign can see the address bits. Several null checks stay
    // that branch. A comparison that is not one can encode the address.
    escape.return_bits & 1 != 0 || escape.escapes || (escape.condition > 0 && !escape.invariant)
}

pub(in crate::front::mir) struct AddressEscape {
    /// Bit `n` is set when `n` dereferences of the return slot yield the
    /// address. Bit 0 means the slot holds those bits. A local keeps
    /// every depth assigned to it.
    pub(in crate::front::mir) return_bits: u64,
    /// How many address-derived comparisons the return slot carries.
    /// One `==` / `!=` with zero still frees the spill. `p < 0` does
    /// not. Two or more encode the address when one of them is not a
    /// null check.
    pub(in crate::front::mir) condition: u8,
    /// The address was stored through a pointer, or a call that received
    /// it has no body to classify. The caller can observe those bits
    /// without reading the return slot.
    pub(in crate::front::mir) escapes: bool,
    /// `condition` is `==` / `!=` with zero. The caller's switch can
    /// use it. `p < 0` is not. Several null checks stay invariant.
    pub(in crate::front::mir) invariant: bool,
    /// The return slot is still the spill pointer. `returned == null`
    /// is the same branch after the spill moves. An aggregate is not.
    pub(in crate::front::mir) direct: bool,
    /// The return value is an aggregate on every path. The caller reads
    /// one field (`(p as usize, 0).1`) from these slots. Empty when a
    /// path returned a scalar, so the caller keeps the folded bits.
    pub(in crate::front::mir) split: bool,
    pub(in crate::front::mir) slots: Vec<FieldSlot>,
}

/// A place a raw pointer names. An empty `path` is `local`.
/// `[0, 1]` is `local.0.1`.
#[derive(Clone, PartialEq, Eq)]
pub(in crate::front::mir) struct AddressReferent {
    pub(in crate::front::mir) local: u64,
    pub(in crate::front::mir) path: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq)]
pub(in crate::front::mir) struct FieldSlot {
    pub(in crate::front::mir) index: usize,
    pub(in crate::front::mir) bits: u64,
    pub(in crate::front::mir) condition: u8,
    /// This field's comparisons are `==` / `!=` with zero.
    pub(in crate::front::mir) invariant: bool,
    /// Fields of this field. Empty when the field is a scalar.
    pub(in crate::front::mir) slots: Vec<FieldSlot>,
    /// This field is still the spill pointer. A copy of that pointer
    /// keeps it. An offset stored in the field does not, so `*field`
    /// then reads a value selected by the address.
    pub(in crate::front::mir) direct: bool,
    /// `*field` reads this place. An empty path names a local.
    /// `[0, 1]` names `local.0.1`. `pair = (q,)` keeps `q`'s name here
    /// when `q` does not yet hold the address. `q = &pair.0` records
    /// that field, so a later store into the leaf is visible through
    /// `*field`.
    pub(in crate::front::mir) referent: Option<AddressReferent>,
}

#[derive(Clone)]
pub(in crate::front::mir) struct LocalAddress {
    pub(in crate::front::mir) local: u64,
    pub(in crate::front::mir) bits: u64,
    /// `0` none, `1` one comparison, [`SPILL_CONDITION_MANY`] two or more.
    pub(in crate::front::mir) condition: u8,
    /// `condition` is `==` / `!=` with zero. `p < 0` is not: its sign
    /// can see the address bits.
    pub(in crate::front::mir) invariant: bool,
    /// `slots` names each aggregate field. Projections of other locals
    /// still read `bits` and `condition`.
    pub(in crate::front::mir) split: bool,
    pub(in crate::front::mir) slots: Vec<FieldSlot>,
    /// This local is still the spill pointer. A copy, `&*p`, and a cast
    /// whose destination holds every address bit keep it. Arithmetic
    /// and `p as u8` do not. `*local = clean` updates its pointee, and
    /// `local == 0` is a null check.
    pub(in crate::front::mir) direct: bool,
    /// `*this` writes this place. `&p` names `p`. `&pair.0` names that
    /// field. A copy keeps the path. A cast keeps it when the
    /// destination pointee still covers the source. Absent on the spill
    /// pointer itself: `*p` writes the pointee. Present with zero
    /// address bits when the name was taken before the place held the
    /// address. A later store into that place is then visible through
    /// `*this`.
    pub(in crate::front::mir) referent: Option<AddressReferent>,
    /// Length of a slice value or a slice pointer. `Unsize` records a
    /// const length (`rvalue_length_metadata`). A copy keeps it. `Len`
    /// of a fixed array does not read this: that length is the array
    /// type. Absent when this walk did not see a length, and `Len` of
    /// that slice then stays unlowered.
    pub(in crate::front::mir) length: Option<AddressValue>,
}

pub(in crate::front::mir) fn plain_local(local: u64, bits: u64, condition: u8) -> LocalAddress {
    LocalAddress {
        local,
        bits,
        condition,
        invariant: false,
        split: false,
        slots: Vec::new(),
        direct: false,
        referent: None,
        length: None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::front::mir) struct AddressValue {
    pub(in crate::front::mir) bits: u64,
    /// `0` none, `1` one comparison, [`SPILL_CONDITION_MANY`] two or more.
    pub(in crate::front::mir) condition: u8,
    pub(in crate::front::mir) overflows: bool,
    /// `condition` is `==` / `!=` with zero.
    pub(in crate::front::mir) invariant: bool,
}

/// `p == null` takes the same branch after the spill moves. `p < 0`
/// does not: its sign can see the address bits. A value that still
/// holds those bits is not a branch condition.
pub(in crate::front::mir) fn control_depends_on_address(value: &AddressValue) -> bool {
    value.overflows || value.bits != 0 || (value.condition > 0 && !value.invariant)
}

pub(in crate::front::mir) fn unclassified_address_escape() -> AddressEscape {
    AddressEscape {
        return_bits: 1,
        condition: 0,
        escapes: true,
        invariant: false,
        direct: false,
        split: false,
        slots: Vec::new(),
    }
}

pub(in crate::front::mir) fn clean_address_escape() -> AddressEscape {
    AddressEscape {
        return_bits: 0,
        condition: 0,
        escapes: false,
        invariant: false,
        direct: false,
        split: false,
        slots: Vec::new(),
    }
}

/// `q = &p` is one step above `p`. Deeper than this is unclassified.
pub(in crate::front::mir) const SPILL_REF_DEPTH_LIMIT: u8 = 32;

/// Two address-derived comparisons are enough to encode the address.
pub(in crate::front::mir) const SPILL_CONDITION_MANY: u8 = 2;

pub(in crate::front::mir) fn function_address_escape(
    llbc: &Llbc,
    fun_id: u64,
    entry: &[LocalAddress],
    stack: &mut Vec<u64>,
) -> AddressEscape {
    if stack.contains(&fun_id) {
        return unclassified_address_escape();
    }
    let Some(fd) = llbc.fn_by_id(fun_id) else {
        return unclassified_address_escape();
    };
    let Some(body) = fd.unstructured() else {
        return unclassified_address_escape();
    };
    stack.push(fun_id);
    let escape = unstructured_address_escape(llbc, &body, entry, stack);
    stack.pop();
    escape
}

pub(in crate::front::mir) fn unstructured_address_escape(
    llbc: &Llbc,
    body: &Unstructured,
    entry: &[LocalAddress],
    stack: &mut Vec<u64>,
) -> AddressEscape {
    let mut projections = Vec::new();
    let mut escapes = false;
    let n = body.body.len();
    if n == 0 {
        return clean_address_escape();
    }
    let mut incoming = vec![Vec::new(); n];
    incoming[0] = entry.to_vec();
    // Block 0 reaches its successors. A return above the assignment
    // is reached only after that edge is joined.
    let mut reached = vec![false; n];
    reached[0] = true;
    // The state at each `Return` after that block's statements. The
    // summary reads this once the join has settled.
    let mut returned_at = vec![None; n];
    let mut changed = true;
    let forward_blocks = forward_reachable_mask(llbc, body);
    while changed {
        changed = false;
        for (index, block) in body.body.iter().enumerate() {
            if !forward_blocks[index] {
                continue;
            }
            let mut depths = incoming[index].clone();
            for stmt in &block.statements {
                match stmt.stmt_kind_ref() {
                    Ok(StmtKind::Assign(place, rvalue)) => {
                        let value = rvalue_address(&rvalue, &depths, llbc);
                        record_stored_address(
                            llbc,
                            &mut depths,
                            &place,
                            value,
                            Some(&rvalue),
                            &mut projections,
                            &mut escapes,
                        );
                    }
                    Ok(StmtKind::Assert(assert)) => {
                        let value = operand_address(&assert.cond, &depths, llbc);
                        if control_depends_on_address(&value) {
                            escapes = true;
                        }
                    }
                    Ok(StmtKind::StorageLive(_))
                    | Ok(StmtKind::StorageDead(_))
                    | Ok(StmtKind::PlaceMention(_))
                    | Ok(StmtKind::Borrowck(_)) => {}
                    // `SetDiscriminant`'s payload is not a unit, so the
                    // typed kind is `Err`. The raw key still names it.
                    Ok(StmtKind::Unknown) | Err(_)
                        if statement_cannot_carry_address(stmt.kind_value(), &depths, llbc) => {}
                    Ok(StmtKind::Unknown) | Err(_) => escapes = true,
                }
            }
            match block.term_ref(llbc) {
                Ok(TermKind::Call {
                    call,
                    target,
                    on_unwind,
                }) => {
                    let escape = call_address_escape(llbc, &call, &depths, stack);
                    if escape.escapes {
                        escapes = true;
                    }
                    record_stored_address(
                        llbc,
                        &mut depths,
                        &call.dest,
                        AddressValue {
                            bits: escape.return_bits,
                            condition: escape.condition,
                            overflows: false,
                            // The callee's `p == null` is the same branch
                            // here. `p < 0` is not.
                            invariant: escape.invariant,
                        },
                        None,
                        &mut projections,
                        &mut escapes,
                    );
                    // `q = inner(p)` has no rvalue, so the store above
                    // keeps neither `inner`'s fields nor `direct`.
                    // `(p as usize, 0).1` would otherwise inherit bit 0,
                    // and `q == null` would look derived.
                    if let PlaceKind::Local(id) = &call.dest.kind
                        && let Some(slot) = depths.iter_mut().find(|slot| slot.local == *id)
                    {
                        if escape.split {
                            let (bits, condition) = fold_slots(&escape.slots);
                            let invariant = folded_invariant(&escape.slots);
                            slot.invariant = merged_invariant(
                                slot.condition,
                                slot.invariant,
                                condition,
                                invariant,
                            );
                            slot.bits |= bits;
                            slot.condition = slot.condition.max(condition);
                            slot.split = true;
                            slot.slots = escape.slots.clone();
                            slot.direct = false;
                            slot.referent = None;
                            slot.length = None;
                        } else if escape.direct && !slot.split && slot.bits & 1 != 0 {
                            slot.direct = true;
                        }
                    }
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *target,
                        &depths,
                        &mut escapes,
                    );
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *on_unwind,
                        &depths,
                        &mut escapes,
                    );
                }
                Ok(TermKind::Drop {
                    place,
                    fn_ptr,
                    target,
                    on_unwind,
                }) => {
                    let value = dropped_address(&place, &depths, llbc);
                    if value.overflows {
                        escapes = true;
                    }
                    if value.bits != 0 || value.condition > 0 {
                        let escape = drop_address_escape(
                            llbc,
                            &fn_ptr,
                            value.bits,
                            value.condition,
                            value.invariant,
                            stack,
                        );
                        if escape.return_bits & 1 != 0 || escape.escapes || escape.condition > 1 {
                            escapes = true;
                        }
                    }
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *target,
                        &depths,
                        &mut escapes,
                    );
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *on_unwind,
                        &depths,
                        &mut escapes,
                    );
                }
                Ok(TermKind::Switch { discr, targets }) => {
                    let value = operand_address(&discr, &depths, llbc);
                    if control_depends_on_address(&value) {
                        escapes = true;
                    }
                    let successors: Vec<u64> = match targets {
                        SwitchTargets::If(then_bb, else_bb) => vec![*then_bb, *else_bb],
                        SwitchTargets::SwitchInt(_, arms, default) => {
                            let mut successors: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                            successors.push(*default);
                            successors
                        }
                    };
                    for target in successors {
                        changed |= join_reached(
                            &mut reached,
                            index,
                            &mut incoming,
                            target,
                            &depths,
                            &mut escapes,
                        );
                    }
                }
                Ok(TermKind::Goto { target }) => {
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *target,
                        &depths,
                        &mut escapes,
                    );
                }
                Ok(TermKind::Assert {
                    assert,
                    target,
                    on_unwind,
                }) => {
                    let value = operand_address(&assert.cond, &depths, llbc);
                    if control_depends_on_address(&value) {
                        escapes = true;
                    }
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *target,
                        &depths,
                        &mut escapes,
                    );
                    changed |= join_reached(
                        &mut reached,
                        index,
                        &mut incoming,
                        *on_unwind,
                        &depths,
                        &mut escapes,
                    );
                }
                Ok(TermKind::Return) => returned_at[index] = Some(depths),
                Ok(TermKind::UnwindResume)
                | Ok(TermKind::UnwindTerminate)
                | Ok(TermKind::Abort(_))
                | Ok(TermKind::Panic { .. })
                | Ok(TermKind::UndefinedBehavior) => {}
                Ok(TermKind::Unknown) | Err(_) => escapes = true,
            }
        }
    }
    // `all_direct &=` during the walk sticks at false when block 1 is
    // visited before block 2 stores `_0 = p`. Fold the saved states
    // after `incoming` has stopped changing.
    let mut saw_return = false;
    let mut return_bits = 0;
    let mut return_condition = 0;
    let mut return_invariant = false;
    let mut all_split = true;
    let mut all_direct = true;
    let mut have_slots = false;
    let mut return_slots = Vec::new();
    for (index, saved) in returned_at.iter().enumerate() {
        if !reached[index] {
            continue;
        }
        let Some(depths) = saved else {
            continue;
        };
        saw_return = true;
        let returned = depths.iter().find(|slot| slot.local == 0);
        let (bits, condition, invariant, this_direct, lifted) = match returned {
            Some(local) if local.split => {
                // `return (q, 0)` still names `bits` through `q`. Lift
                // that name into the field before the slots leave this
                // function.
                let (lifted, overflows) = lift_referent_slots(&local.slots, depths);
                if overflows {
                    escapes = true;
                }
                let (bits, condition) = fold_slots(&lifted);
                (
                    bits,
                    condition,
                    folded_invariant(&lifted),
                    false,
                    Some(lifted),
                )
            }
            Some(local) => (
                local.bits,
                local.condition,
                local.invariant,
                local.direct && local.bits & 1 != 0,
                None,
            ),
            None => (0, 0, false, false, None),
        };
        return_invariant =
            merged_invariant(return_condition, return_invariant, condition, invariant);
        return_bits |= bits;
        return_condition = return_condition.max(condition);
        all_direct &= this_direct;
        match lifted {
            Some(lifted) if all_split => {
                if !have_slots {
                    return_slots = lifted;
                    have_slots = true;
                } else {
                    merge_nested_slots(&mut return_slots, &lifted);
                }
            }
            _ => {
                all_split = false;
                return_slots.clear();
            }
        }
    }
    AddressEscape {
        return_bits,
        condition: return_condition,
        escapes,
        invariant: return_invariant,
        direct: saw_return && all_direct,
        split: saw_return && all_split && have_slots,
        slots: if saw_return && all_split && have_slots {
            return_slots
        } else {
            Vec::new()
        },
    }
}

pub(in crate::front::mir) fn join_reached(
    reached: &mut [bool],
    source: usize,
    incoming: &mut [Vec<LocalAddress>],
    target: u64,
    depths: &[LocalAddress],
    escapes: &mut bool,
) -> bool {
    // `RPythonAnnotator.addpendingblock` schedules a block the first
    // time it is reached, even when the incoming facts are unchanged.
    // A jump back to a block that already ran this pass must run again,
    // or a later `Return` stays out of the summary.
    let mut newly_reached = false;
    if reached.get(source).copied() == Some(true)
        && let Some(flag) = reached.get_mut(target as usize)
        && !*flag
    {
        *flag = true;
        newly_reached = true;
    }
    let joined = join_incoming(incoming, target, depths, escapes);
    joined || newly_reached
}

pub(in crate::front::mir) fn record_stored_address(
    llbc: &Llbc,
    depths: &mut Vec<LocalAddress>,
    place: &Place,
    value: AddressValue,
    rvalue: Option<&Rvalue>,
    projections: &mut Vec<(u64, String)>,
    escapes: &mut bool,
) {
    let index = place_index_address(place, depths, llbc);
    if value.overflows || index.overflows {
        *escapes = true;
    }
    // `*p = clean` updates the pointee when `p` is still the spill
    // pointer. A pointer computed from that address selects another
    // destination, so the stored constant publishes the bits.
    if store_through_derived_pointer(place, depths, llbc) {
        *escapes = true;
    }
    // `PlaceKind::Unknown` has no local to update. The value is stored
    // where this walk cannot follow it.
    let exported = place_stores_through_pointer(place)
        || place_rooted_at_global(place)
        || spill_place_root_local(place).is_none();
    if exported
        && (value.bits != 0 || value.condition > 0 || index.bits != 0 || index.condition > 0)
    {
        *escapes = true;
    }
    // `q = &p; *q = null` writes `p`. The dereference has no local of
    // its own (`spill_place_root_local`). Read the referent first: the
    // store below can remove `p` from `depths`.
    let referent = dereference_referent(place, depths, llbc);
    // `aliases[i] = q` is not one leaf (`constant_field_path`).
    // `mark_local_address` clears the split slots and returns while
    // `q` still holds no address bits, so the referent is dropped.
    // A later store into the place `q` names is then invisible
    // through `*aliases[i]`.
    if untracked_projection_carries_referent(llbc, depths, place, rvalue) {
        *escapes = true;
    }
    mark_local_address(llbc, depths, place, value, rvalue, projections);
    if let Some(referent) = &referent {
        store_through_referent(
            llbc,
            depths,
            referent,
            value,
            rvalue,
            projections,
            &place.ty,
            escapes,
        );
    }
}

/// `*q = value` writes the place `q` names. An empty path is that
/// local (`mark_local_address`). A field path updates that leaf
/// (`write_field_path`) and leaves its siblings alone. A tainted store
/// into a place this walk no longer tracks publishes the address.
pub(in crate::front::mir) fn store_through_referent(
    llbc: &Llbc,
    depths: &mut Vec<LocalAddress>,
    referent: &AddressReferent,
    value: AddressValue,
    rvalue: Option<&Rvalue>,
    projections: &mut Vec<(u64, String)>,
    place_ty: &TyRef,
    escapes: &mut bool,
) {
    if referent.path.is_empty() {
        let local_place = Place {
            kind: PlaceKind::Local(referent.local),
            ty: place_ty.clone(),
        };
        mark_local_address(llbc, depths, &local_place, value, rvalue, projections);
        return;
    }
    let field_direct = value.bits & 1 != 0
        && rvalue.is_some_and(|rv| rvalue_is_direct_pointer(rv, depths, llbc, Some(place_ty)));
    let field_referent = rvalue.and_then(|rv| rvalue_referent(rv, depths, llbc, Some(place_ty)));
    if depths
        .iter()
        .any(|slot| slot.local == referent.local && slot.split)
    {
        if let Some(slot) = depths.iter_mut().find(|slot| slot.local == referent.local) {
            write_field_path(slot, &referent.path, &value, field_direct, field_referent);
        }
        return;
    }
    if value.bits != 0 || value.condition > 0 {
        *escapes = true;
    }
}

pub(in crate::front::mir) fn join_incoming(
    incoming: &mut [Vec<LocalAddress>],
    target: u64,
    depths: &[LocalAddress],
    escapes: &mut bool,
) -> bool {
    let Some(slot) = incoming.get_mut(target as usize) else {
        *escapes = true;
        return false;
    };
    join_locals(slot, depths)
}

pub(in crate::front::mir) fn join_locals(
    dst: &mut Vec<LocalAddress>,
    src: &[LocalAddress],
) -> bool {
    let mut grew = false;
    for value in src {
        if value.bits == 0
            && value.condition == 0
            && !value.split
            && value.referent.is_none()
            && value.length.is_none()
        {
            continue;
        }
        if let Some(slot) = dst.iter_mut().find(|slot| slot.local == value.local) {
            grew |= merge_local(slot, value);
        } else {
            dst.push(value.clone());
            grew = true;
        }
    }
    // `q = &p` on one path and no `q` on another. A later `*q = null`
    // must not clear `p` on the path that still returns it.
    for slot in dst.iter_mut() {
        let present = src.iter().any(|value| value.local == slot.local);
        if slot.referent.is_some() && !present {
            slot.referent = None;
            grew = true;
        }
        // One path built the slice and the other left the local
        // untouched. The length from the first path is not the length
        // on the second, so `Len` stays unlowered.
        if slot.length.is_some() && !present {
            slot.length = None;
            grew = true;
        }
    }
    grew
}

pub(in crate::front::mir) fn depth_bits(depths: &[LocalAddress], local: u64) -> u64 {
    depths
        .iter()
        .find(|slot| slot.local == local)
        .map(|slot| slot.bits)
        .unwrap_or(0)
}

pub(in crate::front::mir) fn local_condition(depths: &[LocalAddress], local: u64) -> u8 {
    depths
        .iter()
        .find(|slot| slot.local == local)
        .map(|slot| slot.condition)
        .unwrap_or(0)
}

pub(in crate::front::mir) fn local_invariant(depths: &[LocalAddress], local: u64) -> bool {
    depths
        .iter()
        .find(|slot| slot.local == local)
        .is_some_and(|slot| slot.invariant)
}

pub(in crate::front::mir) fn fold_slots(slots: &[FieldSlot]) -> (u64, u8) {
    let mut bits = 0;
    let mut condition = 0u8;
    for slot in slots {
        bits |= slot.bits;
        condition = condition
            .saturating_add(slot.condition)
            .min(SPILL_CONDITION_MANY);
    }
    (bits, condition)
}

/// True when every comparison in `slots` is a null check. No comparison
/// is not a null check.
pub(in crate::front::mir) fn folded_invariant(slots: &[FieldSlot]) -> bool {
    let mut saw = false;
    for slot in slots {
        if slot.condition == 0 {
            continue;
        }
        if !slot.invariant {
            return false;
        }
        saw = true;
    }
    saw
}

pub(in crate::front::mir) fn field_is_tainted(field: &FieldSlot) -> bool {
    field.bits != 0
        || field.condition > 0
        || field.referent.is_some()
        || field.slots.iter().any(field_is_tainted)
}

/// Both comparisons must be null checks. A path that did not compare
/// leaves the other path's flag in place.
pub(in crate::front::mir) fn merged_invariant(
    left_cond: u8,
    left_inv: bool,
    right_cond: u8,
    right_inv: bool,
) -> bool {
    match (left_cond > 0, right_cond > 0) {
        (true, true) => left_inv && right_inv,
        (true, false) => left_inv,
        (false, true) => right_inv,
        (false, false) => false,
    }
}

pub(in crate::front::mir) fn merge_nested_slots(
    dst: &mut Vec<FieldSlot>,
    src: &[FieldSlot],
) -> bool {
    let mut grew = false;
    for field in src {
        if let Some(found) = dst.iter_mut().find(|slot| slot.index == field.index) {
            grew |= merge_field(found, field);
        } else {
            dst.push(field.clone());
            grew = true;
        }
    }
    grew
}

pub(in crate::front::mir) fn merge_field(dst: &mut FieldSlot, src: &FieldSlot) -> bool {
    let bits = dst.bits | src.bits;
    let condition = dst.condition.max(src.condition);
    let invariant = merged_invariant(dst.condition, dst.invariant, src.condition, src.invariant);
    // One side stored a scalar over the field. The leaf split is gone,
    // so a later nested read uses the whole field.
    let slots_changed = if dst.slots.is_empty() || src.slots.is_empty() {
        let cleared = !dst.slots.is_empty();
        dst.slots.clear();
        cleared
    } else {
        merge_nested_slots(&mut dst.slots, &src.slots)
    };
    // A direct pointer joined with an offset pointer is an offset.
    let direct = dst.direct && src.direct;
    let referent = merged_referent(&dst.referent, &src.referent);
    let changed = dst.bits != bits
        || dst.condition != condition
        || dst.invariant != invariant
        || dst.direct != direct
        || dst.referent != referent
        || slots_changed;
    dst.bits = bits;
    dst.condition = condition;
    dst.invariant = invariant;
    dst.direct = direct;
    dst.referent = referent;
    changed
}

pub(in crate::front::mir) fn merge_local(dst: &mut LocalAddress, src: &LocalAddress) -> bool {
    if dst.split && src.split {
        let mut grew = false;
        for field in &src.slots {
            if let Some(found) = dst.slots.iter_mut().find(|slot| slot.index == field.index) {
                grew |= merge_field(found, field);
            } else {
                dst.slots.push(field.clone());
                grew = true;
            }
        }
        let (bits, condition) = fold_slots(&dst.slots);
        let invariant = folded_invariant(&dst.slots);
        // Report the referent once, while it is still set. A split
        // local is not a reference to one place.
        grew |= dst.bits != bits
            || dst.condition != condition
            || dst.invariant != invariant
            || dst.direct
            || dst.referent.is_some()
            || dst.length.is_some();
        dst.bits = bits;
        dst.condition = condition;
        dst.invariant = invariant;
        dst.direct = false;
        dst.referent = None;
        // A split aggregate is not one slice, so its length metadata
        // would describe a value this local no longer is.
        dst.length = None;
        return grew;
    }
    let bits = dst.bits | src.bits;
    let condition = dst.condition.max(src.condition);
    let invariant = merged_invariant(dst.condition, dst.invariant, src.condition, src.invariant);
    // A split source stays split on the next pass. Counting it here
    // reports a change after this local has already collapsed.
    // A derived pointer joined with the spill pointer is derived.
    let direct = dst.direct && src.direct;
    let referent = merged_referent(&dst.referent, &src.referent);
    let length = merged_length(dst.length, src.length);
    let changed = dst.bits != bits
        || dst.condition != condition
        || dst.invariant != invariant
        || dst.split
        || dst.direct != direct
        || dst.referent != referent
        || dst.length != length;
    dst.bits = bits;
    dst.condition = condition;
    dst.invariant = invariant;
    dst.split = false;
    dst.slots.clear();
    dst.direct = direct;
    dst.referent = referent;
    dst.length = length;
    changed
}

/// Both paths recorded a length. A path with no length drops it: `Len`
/// would otherwise treat one path's const length as the other's.
pub(in crate::front::mir) fn merged_length(
    left: Option<AddressValue>,
    right: Option<AddressValue>,
) -> Option<AddressValue> {
    match (left, right) {
        (Some(left), Some(right)) => Some(AddressValue {
            bits: left.bits | right.bits,
            condition: left.condition.max(right.condition),
            overflows: left.overflows || right.overflows,
            invariant: merged_invariant(
                left.condition,
                left.invariant,
                right.condition,
                right.invariant,
            ),
        }),
        _ => None,
    }
}

/// Both paths name the same local and the same field path. A path that
/// names another place, or none, drops the referent: `*q` then no
/// longer has one place to write.
pub(in crate::front::mir) fn merged_referent(
    left: &Option<AddressReferent>,
    right: &Option<AddressReferent>,
) -> Option<AddressReferent> {
    match (left, right) {
        (Some(left), Some(right)) if left == right => Some(left.clone()),
        _ => None,
    }
}

pub(in crate::front::mir) fn decl_is_union(llbc: &Llbc, id: u64) -> bool {
    matches!(
        llbc.type_by_id(id).map(|decl| &decl.kind),
        Some(TypeDeclKind::Union(_))
    )
}

pub(in crate::front::mir) fn tyref_is_union(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(inline_adt_def_id)
        .is_some_and(|id| decl_is_union(llbc, id))
}

pub(in crate::front::mir) fn aggregate_kind_is_union(
    kind: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(head) = kind
        .as_object()
        .and_then(|obj| obj.get("Adt"))
        .and_then(|adt| adt.as_array())
        .and_then(|adt| adt.first())
    else {
        return false;
    };
    let Some(id) = head
        .as_u64()
        .or_else(|| head.as_object().and_then(type_decl_ref_adt_id))
    else {
        return false;
    };
    decl_is_union(llbc, id)
}

pub(in crate::front::mir) fn aggregate_slots(
    rvalue: Option<&Rvalue>,
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<Vec<FieldSlot>> {
    let Rvalue::Aggregate(kind, ops) = rvalue? else {
        return None;
    };
    // Union fields occupy one address. A per-field slot would let the
    // other field read as clean. The operands' referent is folded onto
    // the local (`rvalue_referent`); a field read uses that one name
    // (`union_folded_referent`).
    if tyref_is_union(&place.ty, llbc) || aggregate_kind_is_union(kind, llbc) {
        return None;
    }
    Some(
        ops.iter()
            .enumerate()
            .map(|(index, op)| {
                let value = operand_address(op, depths, llbc);
                FieldSlot {
                    index,
                    bits: value.bits,
                    condition: value.condition,
                    invariant: value.invariant,
                    slots: operand_nested_slots(op, depths, llbc),
                    direct: operand_is_direct_pointer(op, depths, llbc) && value.bits & 1 != 0,
                    // `pair = (q,)` while `q` names `bits` and `bits` is
                    // still clean. The field keeps that name, so `*pair.0`
                    // sees a later store into `bits`.
                    referent: operand_referent(op, depths, llbc),
                }
            })
            .collect(),
    )
}

pub(in crate::front::mir) fn write_field_path(
    slot: &mut LocalAddress,
    path: &[usize],
    value: &AddressValue,
    direct: bool,
    referent: Option<AddressReferent>,
) -> bool {
    if path.is_empty() {
        return false;
    }
    let changed = write_slot_path(&mut slot.slots, path, value, direct, referent);
    let (bits, condition) = fold_slots(&slot.slots);
    let invariant = folded_invariant(&slot.slots);
    let folded = slot.bits != bits || slot.condition != condition || slot.invariant != invariant;
    slot.bits = bits;
    slot.condition = condition;
    slot.invariant = invariant;
    changed || folded
}

pub(in crate::front::mir) fn write_slot_path(
    slots: &mut Vec<FieldSlot>,
    path: &[usize],
    value: &AddressValue,
    direct: bool,
    referent: Option<AddressReferent>,
) -> bool {
    let Some((&index, rest)) = path.split_first() else {
        return false;
    };
    if rest.is_empty() {
        return write_leaf_slot(slots, index, value, direct, referent);
    }
    if let Some(field) = slots.iter_mut().find(|field| field.index == index) {
        let changed = write_slot_path(&mut field.slots, rest, value, direct, referent);
        refold_field(field);
        return changed;
    }
    let mut field = FieldSlot {
        index,
        bits: 0,
        condition: 0,
        invariant: false,
        slots: Vec::new(),
        direct: false,
        referent: None,
    };
    let changed = write_slot_path(&mut field.slots, rest, value, direct, referent);
    refold_field(&mut field);
    slots.push(field);
    changed
}

pub(in crate::front::mir) fn refold_field(field: &mut FieldSlot) {
    if field.slots.is_empty() {
        return;
    }
    let (bits, condition) = fold_slots(&field.slots);
    field.bits = bits;
    field.condition = condition;
    field.invariant = folded_invariant(&field.slots);
    field.direct = false;
    field.referent = None;
}

pub(in crate::front::mir) fn write_leaf_slot(
    slots: &mut Vec<FieldSlot>,
    index: usize,
    value: &AddressValue,
    direct: bool,
    referent: Option<AddressReferent>,
) -> bool {
    if let Some(field) = slots.iter_mut().find(|field| field.index == index) {
        let changed = field.bits != value.bits
            || field.condition != value.condition
            || field.invariant != value.invariant
            || field.direct != direct
            || field.referent != referent
            || !field.slots.is_empty();
        field.bits = value.bits;
        field.condition = value.condition;
        field.invariant = value.invariant;
        field.direct = direct;
        field.referent = referent;
        // A scalar store replaces the aggregate that field held.
        field.slots.clear();
        return changed;
    }
    slots.push(FieldSlot {
        index,
        bits: value.bits,
        condition: value.condition,
        invariant: value.invariant,
        slots: Vec::new(),
        direct,
        referent,
    });
    true
}

pub(in crate::front::mir) fn copied_aggregate_slots(
    rvalue: Option<&Rvalue>,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<Vec<FieldSlot>> {
    let Rvalue::Use(op, _) = rvalue? else {
        return None;
    };
    let place = match op {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Const(_) => return None,
    };
    // `let inner = outer.0` where `outer.0` is `(p as usize, 0)`.
    // The folded bits of that field would taint the clean element.
    let slots = nested_slots_of_place(place, depths, llbc)?;
    if slots.is_empty() { None } else { Some(slots) }
}

pub(in crate::front::mir) fn mark_local_address(
    llbc: &Llbc,
    depths: &mut Vec<LocalAddress>,
    place: &Place,
    value: AddressValue,
    rvalue: Option<&Rvalue>,
    projections: &mut Vec<(u64, String)>,
) -> bool {
    let index = place_index_address(place, depths, llbc);
    let mut value = value;
    if index.bits != 0
        || index.condition > 0
        || (place_index_is_dynamic(place) && value.condition > 0)
    {
        // A dynamic slot can select more than one comparison.
        value.condition = SPILL_CONDITION_MANY;
        value.invariant = false;
    }
    let Some(dest) = spill_place_root_local(place) else {
        return false;
    };
    if matches!(place.kind, PlaceKind::Local(_)) {
        projections.retain(|(id, _)| *id != dest);
        let slots = aggregate_slots(rvalue, place, depths, llbc)
            .or_else(|| copied_aggregate_slots(rvalue, depths, llbc))
            .or_else(|| repeat_slots(rvalue, depths, llbc));
        let split = slots.is_some();
        let slots = slots.unwrap_or_default();
        let invariant = if split {
            folded_invariant(&slots)
        } else {
            value.invariant
        };
        let tainted = value.bits != 0 || value.condition > 0 || slots.iter().any(field_is_tainted);
        let direct = !split
            && value.bits & 1 != 0
            && rvalue.is_some_and(|rv| rvalue_is_direct_pointer(rv, depths, llbc, Some(&place.ty)));
        // `q = &bits` while `bits` is still clean. Dropping `q` before
        // `rvalue_referent` records the name would make a later
        // `bits = p as usize` invisible through `*q`.
        // A tuple is not one referent. `pair = (q,)` records `q`'s name
        // on field 0 (`aggregate_slots`). A clean aggregate stays
        // split so `q = &pair.0` can name that leaf (`place_referent`).
        let referent = if split {
            None
        } else {
            rvalue.and_then(|rv| rvalue_referent(rv, depths, llbc, Some(&place.ty)))
        };
        // `s = &array as &[T]` is clean apart from this length. Dropping
        // `s` would make `Len(*s)` look untracked.
        let length = if split {
            None
        } else {
            rvalue.and_then(|rv| rvalue_length_metadata(rv, depths, llbc))
        };
        if !tainted && referent.is_none() && !split && length.is_none() {
            let removed = depths.iter().any(|slot| slot.local == dest);
            depths.retain(|slot| slot.local != dest);
            return removed;
        }
        if let Some(slot) = depths.iter_mut().find(|slot| slot.local == dest) {
            let changed = slot.bits != value.bits
                || slot.condition != value.condition
                || slot.invariant != invariant
                || slot.split != split
                || slot.slots != slots
                || slot.direct != direct
                || slot.referent != referent
                || slot.length != length;
            slot.bits = value.bits;
            slot.condition = value.condition;
            slot.invariant = invariant;
            slot.split = split;
            slot.slots = slots;
            slot.direct = direct;
            slot.referent = referent;
            slot.length = length;
            return changed;
        }
        depths.push(LocalAddress {
            local: dest,
            bits: value.bits,
            condition: value.condition,
            invariant,
            split,
            slots,
            direct,
            referent,
            length,
        });
        return true;
    }
    let field_direct = value.bits & 1 != 0
        && rvalue.is_some_and(|rv| rvalue_is_direct_pointer(rv, depths, llbc, Some(&place.ty)));
    let field_referent = rvalue.and_then(|rv| rvalue_referent(rv, depths, llbc, Some(&place.ty)));
    let field_path = constant_field_path(place, llbc);
    if depths.iter().any(|slot| slot.local == dest && slot.split) {
        if let Some((root, path)) = &field_path
            && *root == dest
            && !path.is_empty()
            && let Some(slot) = depths.iter_mut().find(|slot| slot.local == dest)
        {
            // `outer.0.0 = 0` updates that leaf and refolds its parents.
            // Collapsing the tree would keep the old folded address.
            return write_field_path(slot, path, &value, field_direct, field_referent);
        }
        if let Some(slot) = depths.iter_mut().find(|slot| slot.local == dest) {
            slot.split = false;
            slot.slots.clear();
            slot.length = None;
        }
    }
    if value.condition > 0 {
        let count = note_projected_condition(projections, dest, projection_key(place));
        value.condition = value.condition.max(count);
    }
    if value.bits == 0 && value.condition == 0 {
        return false;
    }
    if let Some(slot) = depths.iter_mut().find(|slot| slot.local == dest) {
        let added = value.bits & !slot.bits;
        let added_condition = value.condition > slot.condition;
        let invariant = merged_invariant(
            slot.condition,
            slot.invariant,
            value.condition,
            value.invariant,
        );
        let changed_invariant = slot.invariant != invariant;
        let cleared_direct = slot.direct;
        let cleared_referent = slot.referent.is_some();
        let cleared_length = slot.length.is_some();
        slot.bits |= value.bits;
        slot.condition = slot.condition.max(value.condition);
        slot.invariant = invariant;
        slot.split = false;
        slot.slots.clear();
        slot.direct = false;
        slot.referent = None;
        slot.length = None;
        return added != 0
            || added_condition
            || cleared_direct
            || changed_invariant
            || cleared_referent
            || cleared_length;
    }
    let mut local = plain_local(dest, value.bits, value.condition);
    local.invariant = value.invariant;
    depths.push(local);
    true
}

pub(in crate::front::mir) fn indirect_target_address(
    func: &CallFunc,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match func {
        CallFunc::Dynamic(op, _) => operand_address(op, depths, llbc),
        CallFunc::Regular(reg) => match &reg.kind {
            CallKind::Ptr(value) => value_address(value, depths, 0, llbc),
            _ => AddressValue {
                bits: 0,
                condition: 0,
                overflows: false,
                invariant: false,
            },
        },
        CallFunc::Unknown => AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        },
    }
}

pub(in crate::front::mir) fn value_address(
    value: &serde_json::Value,
    depths: &[LocalAddress],
    depth: u8,
    llbc: &Llbc,
) -> AddressValue {
    if depth > 8 {
        return AddressValue {
            bits: 0,
            condition: 0,
            overflows: true,
            invariant: false,
        };
    }
    if let Ok(op) = serde_json::from_value::<Operand>(value.clone()) {
        return operand_address(&op, depths, llbc);
    }
    if let Ok(place) = serde_json::from_value::<Place>(value.clone()) {
        return place_address(&place, depths, llbc);
    }
    let mut bits = 0;
    let mut condition = 0;
    let mut overflows = false;
    let mut saw_condition = false;
    let mut invariant = true;
    let children: Vec<&serde_json::Value> = match value {
        serde_json::Value::Array(items) => items.iter().collect(),
        serde_json::Value::Object(map) => map.values().collect(),
        _ => {
            return AddressValue {
                bits: 0,
                condition: 0,
                overflows: false,
                invariant: false,
            };
        }
    };
    for child in children {
        let value = value_address(child, depths, depth + 1, llbc);
        bits |= value.bits;
        condition = condition.max(value.condition);
        overflows |= value.overflows;
        if value.condition > 0 {
            saw_condition = true;
            invariant &= value.invariant;
        }
    }
    AddressValue {
        bits,
        condition,
        overflows,
        invariant: saw_condition && invariant && !overflows,
    }
}

/// Field slots of an aggregate passed by value. Empty when the operand
/// is a scalar. The callee reads `.1` of `(p as usize, 0)` as zero.
/// A field referent is lifted onto that field and the caller's local
/// id is dropped (`lift_referent_slots`).
pub(in crate::front::mir) fn operand_split_slots(
    op: &Operand,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<(Vec<FieldSlot>, bool)> {
    let place = match op {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Const(_) => return None,
    };
    let slots = nested_slots_of_place(place, depths, llbc)?;
    if slots.is_empty() {
        None
    } else {
        Some(lift_referent_slots(&slots, depths))
    }
}

/// The callee's `*pair.0` shifts these bits. The caller's local id is
/// not a local in the callee.
pub(in crate::front::mir) fn lift_referent_slots(
    slots: &[FieldSlot],
    depths: &[LocalAddress],
) -> (Vec<FieldSlot>, bool) {
    let mut overflows = false;
    let lifted = slots
        .iter()
        .map(|slot| {
            let (field, field_overflows) = lift_referent_field(slot, depths);
            overflows |= field_overflows;
            field
        })
        .collect();
    (lifted, overflows)
}

pub(in crate::front::mir) fn lift_referent_field(
    slot: &FieldSlot,
    depths: &[LocalAddress],
) -> (FieldSlot, bool) {
    let (slots, mut overflows) = lift_referent_slots(&slot.slots, depths);
    let mut bits = slot.bits;
    let mut condition = slot.condition;
    let mut invariant = slot.invariant;
    let mut direct = slot.direct && slot.referent.is_none();
    // `((q,),)` lifts `q` into the inner field. The outer field still
    // has its pre-lift bits (`fold_slots` would then see zero), so
    // refold the lifted children before the caller's fold.
    if !slots.is_empty() {
        let (folded_bits, folded_condition) = fold_slots(&slots);
        bits = folded_bits;
        condition = folded_condition;
        invariant = folded_invariant(&slots);
        direct = false;
    }
    if let Some(referent) = &slot.referent {
        let lifted = lift_referent_value(
            AddressValue {
                bits,
                condition,
                overflows: false,
                invariant,
            },
            referent,
            depths,
        );
        bits = lifted.bits;
        condition = lifted.condition;
        invariant = lifted.invariant;
        overflows |= lifted.overflows;
    }
    (
        FieldSlot {
            index: slot.index,
            bits,
            condition,
            invariant,
            slots,
            direct,
            referent: None,
        },
        overflows,
    )
}

/// `aliases[i] = q` does not name one element. A constant `Field` or
/// `Index` on a split local updates that leaf (`write_field_path`).
/// Anything else clears the slots (`mark_local_address`) and would
/// drop a referent that still holds no address bits.
pub(in crate::front::mir) fn untracked_projection_carries_referent(
    llbc: &Llbc,
    depths: &[LocalAddress],
    place: &Place,
    rvalue: Option<&Rvalue>,
) -> bool {
    if matches!(place.kind, PlaceKind::Local(_)) {
        return false;
    }
    let Some(dest) = spill_place_root_local(place) else {
        return false;
    };
    if let Some((root, path)) = constant_field_path(place, llbc)
        && root == dest
        && !path.is_empty()
        && depths.iter().any(|slot| slot.local == dest && slot.split)
    {
        return false;
    }
    let stored =
        rvalue.is_some_and(|rv| rvalue_carries_referent(rv, depths, llbc, Some(&place.ty)));
    let cleared = depths
        .iter()
        .any(|slot| slot.local == dest && slot.split && slots_have_referent(&slot.slots));
    stored || cleared
}

/// `q` itself, or a value that nests `q`. A cast that keeps the name
/// is already `rvalue_referent`.
pub(in crate::front::mir) fn rvalue_carries_referent(
    rvalue: &Rvalue,
    depths: &[LocalAddress],
    llbc: &Llbc,
    dest_ty: Option<&TyRef>,
) -> bool {
    if rvalue_referent(rvalue, depths, llbc, dest_ty).is_some() {
        return true;
    }
    match rvalue {
        Rvalue::Use(op, _) | Rvalue::Repeat(op, _, _, _) => {
            operand_referent(op, depths, llbc).is_some()
                || slots_have_referent(&operand_nested_slots(op, depths, llbc))
        }
        Rvalue::Aggregate(_, ops) => !matches!(
            fold_operand_referents(ops, depths, llbc),
            FoldedReferent::None
        ),
        _ => false,
    }
}

/// `q` names `bits` or `pair.0` and stores none of its address bits.
/// The callee receives `q`, one dereference above that place
/// (`referent_value`, `lift_address_bits`). The caller's path stays
/// behind.
pub(in crate::front::mir) fn address_through_referent(
    op: &Operand,
    value: AddressValue,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    let Some(referent) = operand_referent(op, depths, llbc) else {
        return value;
    };
    lift_referent_value(value, &referent, depths)
}

/// One dereference above `referent`'s current value. The caller's path
/// is not a local in the callee.
pub(in crate::front::mir) fn lift_referent_value(
    value: AddressValue,
    referent: &AddressReferent,
    depths: &[LocalAddress],
) -> AddressValue {
    let referred = referent_value(depths, referent);
    let (lifted, overflows) = lift_address_bits(referred.bits);
    let condition = if value.condition > 0 && referred.condition > 0 {
        value
            .condition
            .saturating_add(referred.condition)
            .min(SPILL_CONDITION_MANY)
    } else {
        value.condition.max(referred.condition)
    };
    AddressValue {
        bits: value.bits | lifted,
        condition,
        overflows: value.overflows || referred.overflows || overflows,
        invariant: merged_invariant(
            value.condition,
            value.invariant,
            referred.condition,
            referred.invariant,
        ),
    }
}

pub(in crate::front::mir) fn call_address_escape(
    llbc: &Llbc,
    call: &CallPayload,
    depths: &[LocalAddress],
    stack: &mut Vec<u64>,
) -> AddressEscape {
    let mut arg_overflows = false;
    let entry: Vec<LocalAddress> = call
        .args
        .iter()
        .enumerate()
        .filter_map(|(index, op)| {
            // `q = &bits` while `bits` is clean leaves `q` with no
            // address bits. `bits = p as usize; read(q)` still passes
            // that address, one step above `bits`.
            let mut value = operand_address(op, depths, llbc);
            value = address_through_referent(op, value, depths, llbc);
            let split = operand_split_slots(op, depths, llbc);
            if let Some((_, overflows)) = &split {
                arg_overflows |= *overflows;
            }
            arg_overflows |= value.overflows;
            let slot_tainted = split
                .as_ref()
                .is_some_and(|(slots, _)| slots.iter().any(field_is_tainted));
            (value.bits != 0 || value.condition > 0 || slot_tainted).then_some({
                let mut local = plain_local(index as u64 + 1, value.bits, value.condition);
                local.direct = operand_is_direct_pointer(op, depths, llbc) && value.bits & 1 != 0;
                local.invariant = value.invariant;
                // `(p as usize, 0).1` is zero in the callee. The slots
                // describe that value. A referent names a caller local
                // and stays behind; `address_through_referent` and
                // `lift_referent_slots` already lifted its depths.
                if let Some((slots, _)) = split {
                    local.split = true;
                    local.slots = slots;
                    local.direct = false;
                }
                local
            })
        })
        .collect();
    if arg_overflows {
        return unclassified_address_escape();
    }
    // `fn_table[p == null]()` has no tainted argument. The selected
    // pointer still carries the address (`CallFunc::Dynamic`,
    // `CallKind::Ptr`).
    let target = indirect_target_address(&call.func, depths, llbc);
    if target.bits != 0 || target.condition > 0 || target.overflows {
        return unclassified_address_escape();
    }
    if entry.is_empty() {
        return clean_address_escape();
    }
    let CallFunc::Regular(reg) = &call.func else {
        // Target is not the spill address (`indirect_target_address`).
        // A Dynamic callee borrows the pointer for the call and writes
        // through it then (`copy_out`), the `lltype.scoped_alloc` buffer
        // contract of a raw pointer passed to a call.
        return clean_address_escape();
    };
    match &reg.kind {
        CallKind::Fun(FunId::Regular { id }) => function_address_escape(llbc, *id, &entry, stack),
        CallKind::Fun(FunId::Other(_))
        | CallKind::Trait(_)
        | CallKind::Ptr(_)
        | CallKind::Unknown => unclassified_address_escape(),
    }
}

/// `drop(wrapper)` sees a field that names `bits` (`FieldSlot::referent`).
/// `fold_slots` leaves that name out of `wrapper`'s bits, so the gate
/// lifts it (`lift_referent_slots`) before `drop_in_place`. A parent
/// refolds the lifted child (`lift_referent_field`), so `((q,),)`
/// carries the same bits as `(q,)`. The place's own referent is a
/// different pointer and is lifted again (`lift_referent_value`).
pub(in crate::front::mir) fn dropped_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    let mut value = place_address(place, depths, llbc);
    if let Some(slots) = nested_slots_of_place(place, depths, llbc) {
        let (lifted, overflows) = lift_referent_slots(&slots, depths);
        let (bits, condition) = fold_slots(&lifted);
        let invariant = folded_invariant(&lifted);
        let merged = merged_invariant(value.condition, value.invariant, condition, invariant);
        value.bits |= bits;
        value.condition = value.condition.max(condition);
        value.invariant = merged;
        value.overflows |= overflows;
    }
    if let Some(referent) = reference_referent(place, depths, llbc) {
        value = lift_referent_value(value, &referent, depths);
    }
    value
}

/// `drop_in_place` receives a pointer to the dropped place, one step
/// above each depth that place can hold. Comparisons in the place are
/// on that pointer, so glue can publish them.
pub(in crate::front::mir) fn drop_address_escape(
    llbc: &Llbc,
    fn_ptr: &RegularCall,
    bits: u64,
    condition: u8,
    invariant: bool,
    stack: &mut Vec<u64>,
) -> AddressEscape {
    let (entry_bits, overflow) = lift_address_bits(bits);
    if overflow {
        return unclassified_address_escape();
    }
    if entry_bits == 0 && condition == 0 {
        return clean_address_escape();
    }
    match &fn_ptr.kind {
        CallKind::Fun(FunId::Regular { id }) => {
            let mut local = plain_local(1, entry_bits, condition);
            // `p == null` takes the same branch in the glue. `p < 0`
            // does not: `plain_local` would drop that distinction.
            local.invariant = invariant;
            function_address_escape(llbc, *id, &[local], stack)
        }
        CallKind::Fun(FunId::Other(_))
        | CallKind::Trait(_)
        | CallKind::Ptr(_)
        | CallKind::Unknown => unclassified_address_escape(),
    }
}

/// `SAVED = bits` and `SAVED.field = bits` write a global.
pub(in crate::front::mir) fn place_rooted_at_global(place: &Place) -> bool {
    match &place.kind {
        PlaceKind::Global { .. } => true,
        PlaceKind::Projection(base, _) => place_rooted_at_global(base),
        _ => false,
    }
}

/// `*p = x` and `(*p).field = x` write through a pointer.
pub(in crate::front::mir) fn place_stores_through_pointer(place: &Place) -> bool {
    match &place.kind {
        PlaceKind::Projection(_, elem) if projection_is_deref(elem) => true,
        PlaceKind::Projection(base, _) => place_stores_through_pointer(base),
        _ => false,
    }
}

/// `q = p` keeps the spill pointer. A cast keeps it when the
/// destination holds every address bit, and so does `&*p`. Arithmetic
/// and `p as u8` do not: `*q` then reads a value selected by those bits.
pub(in crate::front::mir) fn rvalue_is_direct_pointer(
    rvalue: &Rvalue,
    depths: &[LocalAddress],
    llbc: &Llbc,
    dest_ty: Option<&TyRef>,
) -> bool {
    match rvalue {
        Rvalue::Use(op, _) => operand_is_direct_pointer(op, depths, llbc),
        Rvalue::Cast(kind, op, target_ty) => {
            operand_is_direct_pointer(op, depths, llbc)
                && cast_preserves_spill_address(kind, Some(target_ty), llbc)
        }
        Rvalue::UnaryOp(op, operand) if unary_op_is_cast(op) => {
            operand_is_direct_pointer(operand, depths, llbc)
                && cast_preserves_spill_address(op, dest_ty, llbc)
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            reborrow_of_direct(place, depths, llbc)
        }
        _ => false,
    }
}

/// `p as usize` and `p as *const u8` still name the spill allocation.
/// `p as u8` keeps the low byte. A float, `Unsize`, and a cast this
/// walk cannot classify do not preserve the address.
pub(in crate::front::mir) fn cast_preserves_spill_address(
    kind: &serde_json::Value,
    target_ty: Option<&TyRef>,
    llbc: &Llbc,
) -> bool {
    let kind = kind.get("Cast").unwrap_or(kind);
    let from_pair = cast_pair_dest(kind)
        .map(|ty| json_type_holds_full_address(ty, llbc) || scalar_ty_holds_full_address(ty));
    let from_target = target_ty.map(|ty| type_holds_full_address(ty, llbc));
    match (from_pair, from_target) {
        (Some(pair), Some(target)) => pair && target,
        (Some(proven), None) | (None, Some(proven)) => {
            proven && (from_pair.is_some() || cast_kind_is_address_cast(kind))
        }
        (None, None) => false,
    }
}

/// Dest type of a dest-only `CastKind` (`Remove the target types of casts`).
/// `RawPtr` / `PtrWithExposedProvenance` carry a `Ty`; `Scalar` a `ScalarTy`.
pub(in crate::front::mir) fn cast_pair_dest(
    kind: &serde_json::Value,
) -> Option<&serde_json::Value> {
    let obj = kind.as_object()?;
    obj.get("RawPtr")
        .or_else(|| obj.get("Scalar"))
        .or_else(|| obj.get("PtrWithExposedProvenance"))
}

/// Integer `ScalarTy` at least as wide as the target word.
pub(in crate::front::mir) fn scalar_ty_holds_full_address(ty: &serde_json::Value) -> bool {
    let Some(int) = ty.get("Integer") else {
        return false;
    };
    let Some(name) = int
        .get("Signed")
        .or_else(|| int.get("Unsigned"))
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };
    int_scalar_width_covers_word(name, crate::layout::target_word_size())
}

/// Integer storage at least as wide as the pointer (`rbuiltin.py` `gen_cast`
/// preserves the value through primitive widening).
pub(in crate::front::mir) fn int_scalar_width_covers_word(name: &str, word: usize) -> bool {
    let width = match name {
        "I8" | "U8" => 1,
        "I16" | "U16" => 2,
        "I32" | "U32" => 4,
        "I64" | "U64" => 8,
        "I128" | "U128" => 16,
        "Isize" | "Usize" => word,
        _ => return false,
    };
    width >= word
}

pub(in crate::front::mir) fn cast_kind_is_address_cast(kind: &serde_json::Value) -> bool {
    kind.as_str().is_some_and(|tag| {
        matches!(
            tag,
            "Scalar" | "RawPtr" | "PtrExposeProvenance" | "PtrWithExposedProvenance"
        )
    }) || kind.as_object().is_some_and(|obj| {
        obj.contains_key("Scalar")
            || obj.contains_key("RawPtr")
            || obj.contains_key("PtrWithExposedProvenance")
    })
}

/// A thin pointer is one word. An integer holds the address when it is
/// at least that wide. `target_word_size` is the pointer width.
pub(in crate::front::mir) fn type_holds_full_address(ty: &TyRef, llbc: &Llbc) -> bool {
    let word = crate::layout::target_word_size() as u64;
    let Some(size) = tyref_exact_layout_size(ty, llbc) else {
        return false;
    };
    if size < word {
        return false;
    }
    let Some(obj) = tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    if obj.contains_key("RawPtr") || obj.contains_key("Ref") {
        return size == word;
    }
    obj.get("Scalar")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|scalar| scalar.contains_key("Integer"))
}

pub(in crate::front::mir) fn json_type_holds_full_address(
    ty: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    serde_json::from_value::<TyRef>(ty.clone()).is_ok_and(|ty| type_holds_full_address(&ty, llbc))
}

/// `&*p` names the spill allocation again. `&p` names the local that
/// holds the pointer.
pub(in crate::front::mir) fn reborrow_of_direct(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    let PlaceKind::Projection(base, elem) = &place.kind else {
        return false;
    };
    projection_is_deref(elem) && local_is_direct_pointer(base, depths, llbc)
}

pub(in crate::front::mir) fn operand_is_direct_pointer(
    op: &Operand,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    let place = match op {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Const(_) => return false,
    };
    local_is_direct_pointer(place, depths, llbc)
}

/// A bare local, or a field / constant index whose slot is still the
/// spill pointer. `*pair.0` then loads the pointee. An offset stored in
/// the field does not qualify.
pub(in crate::front::mir) fn local_is_direct_pointer(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    if let PlaceKind::Local(id) = &place.kind {
        return depths
            .iter()
            .any(|slot| slot.local == *id && slot.direct && !slot.split && slot.bits & 1 != 0);
    }
    let Some((root, path)) = constant_field_path(place, llbc) else {
        return false;
    };
    if path.is_empty() {
        return false;
    }
    lookup_field_path(depths, root, &path).is_some_and(|field| field.direct && field.bits & 1 != 0)
}

/// `&p` names `p` even when `p` holds nothing yet. `&pair.0` and
/// `&arr[0]` name that field (`constant_field_path`). `&*q` names the
/// same place `q` does. `&*p` on the spill pointer names no local.
pub(in crate::front::mir) fn place_referent(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressReferent> {
    match &place.kind {
        PlaceKind::Local(id) => Some(AddressReferent {
            local: *id,
            path: Vec::new(),
        }),
        PlaceKind::Projection(base, elem) if projection_is_deref(elem) => {
            reference_referent(base, depths, llbc)
        }
        _ => {
            let (root, path) = constant_field_path(place, llbc)?;
            if path.is_empty() {
                None
            } else {
                Some(AddressReferent { local: root, path })
            }
        }
    }
}

/// The place a bare reference names. `pair.0` names the same place
/// when that field recorded it (`nested_field_address`).
pub(in crate::front::mir) fn reference_referent(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressReferent> {
    match &place.kind {
        PlaceKind::Local(id) => depths
            .iter()
            .find(|slot| slot.local == *id)
            .and_then(|slot| slot.referent.clone()),
        // `pair.0` names the same place `q` did when `pair = (q,)`.
        PlaceKind::Projection(_, _) => {
            if let Some(referent) =
                nested_field_address(place, depths, llbc).and_then(|field| field.referent.clone())
            {
                return Some(referent);
            }
            // `u.q` overlaps every field. The union local holds one
            // folded referent (`rvalue_referent`).
            union_folded_referent(place, depths, llbc)
        }
        _ => None,
    }
}

/// The value `referent` names now. An empty path is that local. A
/// field path is that leaf's stored bits, not one more dereference.
/// A path this walk no longer tracks overflows: a later read would
/// miss a store into the leaf.
pub(in crate::front::mir) fn referent_value(
    depths: &[LocalAddress],
    referent: &AddressReferent,
) -> AddressValue {
    if referent.path.is_empty() {
        return AddressValue {
            bits: depth_bits(depths, referent.local),
            condition: local_condition(depths, referent.local),
            overflows: false,
            invariant: local_invariant(depths, referent.local),
        };
    }
    match lookup_field_path(depths, referent.local, &referent.path) {
        Some(field) => AddressValue {
            bits: field.bits,
            condition: field.condition,
            overflows: false,
            invariant: field.invariant,
        },
        None => AddressValue {
            bits: 0,
            condition: 0,
            overflows: true,
            invariant: false,
        },
    }
}

pub(in crate::front::mir) fn operand_referent(
    op: &Operand,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressReferent> {
    match op {
        Operand::Copy(place) | Operand::Move(place) => reference_referent(place, depths, llbc),
        Operand::Const(_) => None,
    }
}

/// `q = &p` and `r = q` name `p`. `q as *mut u64` keeps that when the
/// pointee still covers `p`. `q as *mut u8` does not: a byte store
/// leaves the rest of `p` holding the address.
pub(in crate::front::mir) fn rvalue_referent(
    rvalue: &Rvalue,
    depths: &[LocalAddress],
    llbc: &Llbc,
    dest_ty: Option<&TyRef>,
) -> Option<AddressReferent> {
    match rvalue {
        Rvalue::Use(op, _) => operand_referent(op, depths, llbc),
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_referent(place, depths, llbc)
        }
        Rvalue::Cast(kind, op, target_ty)
            if cast_preserves_spill_address(kind, Some(target_ty), llbc)
                && cast_covers_referent_pointee(op, Some(target_ty), kind, llbc) =>
        {
            operand_referent(op, depths, llbc)
        }
        Rvalue::UnaryOp(op, operand)
            if unary_op_is_cast(op)
                && cast_preserves_spill_address(op, dest_ty, llbc)
                && cast_covers_referent_pointee(operand, dest_ty, op, llbc) =>
        {
            operand_referent(operand, depths, llbc)
        }
        // `U { q }` keeps `q`'s name on the local. Per-field slots would
        // show the other field as clean (`aggregate_slots`).
        Rvalue::Aggregate(kind, ops) if aggregate_kind_is_union(kind, llbc) => {
            match fold_operand_referents(ops, depths, llbc) {
                FoldedReferent::One(referent) => Some(referent),
                FoldedReferent::None | FoldedReferent::Overflow => None,
            }
        }
        _ => None,
    }
}

/// Past this, `[q; N]` stays unlowered when `q` names a place.
/// `repeat_slots` would otherwise copy that referent onto every index.
pub(in crate::front::mir) const REPEAT_SLOT_LIMIT: usize = 32;

pub(in crate::front::mir) enum FoldedReferent {
    None,
    One(AddressReferent),
    Overflow,
}

/// One shared name, or [`FoldedReferent::Overflow`] when the operands
/// name more than one place or hide a name inside a field.
pub(in crate::front::mir) fn fold_operand_referents(
    ops: &[Operand],
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> FoldedReferent {
    let mut found = None;
    for op in ops {
        if slots_have_referent(&operand_nested_slots(op, depths, llbc)) {
            return FoldedReferent::Overflow;
        }
        let Some(referent) = operand_referent(op, depths, llbc) else {
            continue;
        };
        match &found {
            None => found = Some(referent),
            Some(prev) if prev == &referent => {}
            Some(_) => return FoldedReferent::Overflow,
        }
    }
    match found {
        Some(referent) => FoldedReferent::One(referent),
        None => FoldedReferent::None,
    }
}

pub(in crate::front::mir) fn slots_have_referent(slots: &[FieldSlot]) -> bool {
    slots
        .iter()
        .any(|slot| slot.referent.is_some() || slots_have_referent(&slot.slots))
}

/// `[q; N]` when `N` is a literal up to [`REPEAT_SLOT_LIMIT`].
pub(in crate::front::mir) fn repeat_element_count(
    llbc: &Llbc,
    count: &serde_json::Value,
) -> Option<usize> {
    let decoded = decode_constant(llbc, count).ok()?;
    let n = match const_lit_from_decoded(decoded)? {
        ConstLit::UInt(n) => n,
        ConstLit::Int(n) if n >= 0 => n as u64,
        _ => return None,
    };
    let n = usize::try_from(n).ok()?;
    if n > REPEAT_SLOT_LIMIT {
        return None;
    }
    Some(n)
}

/// Each element of `[q; N]` keeps `q`'s referent (`operand_referent`).
/// `*aliases[i]` then sees a later store into that place.
pub(in crate::front::mir) fn repeat_slots(
    rvalue: Option<&Rvalue>,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<Vec<FieldSlot>> {
    let Rvalue::Repeat(op, _, count, _) = rvalue? else {
        return None;
    };
    let n = repeat_element_count(llbc, count)?;
    let value = operand_address(op, depths, llbc);
    let referent = operand_referent(op, depths, llbc);
    let nested = operand_nested_slots(op, depths, llbc);
    let direct = operand_is_direct_pointer(op, depths, llbc) && value.bits & 1 != 0;
    Some(
        (0..n)
            .map(|index| FieldSlot {
                index,
                bits: value.bits,
                condition: value.condition,
                invariant: value.invariant,
                slots: nested.clone(),
                direct,
                referent: referent.clone(),
            })
            .collect(),
    )
}

pub(in crate::front::mir) fn place_projection_root(place: &Place) -> &Place {
    let mut current = place;
    while let PlaceKind::Projection(base, _) = &current.kind {
        current = base;
    }
    current
}

/// A field of a union local that is not split. Every field shares the
/// referent recorded for the whole value (`rvalue_referent`).
pub(in crate::front::mir) fn union_folded_referent(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressReferent> {
    let root = place_projection_root(place);
    let PlaceKind::Local(id) = &root.kind else {
        return None;
    };
    if !tyref_is_union(&root.ty, llbc) {
        return None;
    }
    let (_, path) = constant_field_path(place, llbc)?;
    if path.is_empty() {
        return None;
    }
    let slot = depths.iter().find(|slot| slot.local == *id)?;
    if slot.split {
        return None;
    }
    slot.referent.clone()
}

/// A store through the cast result replaces the referenced local only
/// when the destination pointee is at least as wide as the source
/// pointee. `*mut u8` does not cover a `u64`. An integer cast has no
/// pointee, so it does not name the local.
pub(in crate::front::mir) fn cast_covers_referent_pointee(
    op: &Operand,
    target_ty: Option<&TyRef>,
    kind: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let src_ty = match op {
        Operand::Copy(place) | Operand::Move(place) => &place.ty,
        Operand::Const(_) => return false,
    };
    let parsed_dst;
    let dst_ty = if let Some(ty) = target_ty {
        ty
    } else {
        // `UnaryOp` carries dest-only `{"Cast": {"RawPtr": dest_ty}}`.
        let kind = kind.get("Cast").unwrap_or(kind);
        let Some(json) = cast_pair_dest(kind) else {
            return false;
        };
        let Ok(ty) = serde_json::from_value::<TyRef>(json.clone()) else {
            return false;
        };
        parsed_dst = ty;
        &parsed_dst
    };
    let Some(src_pointee) = pointer_pointee_ty(src_ty, llbc) else {
        return false;
    };
    let Some(dst_pointee) = pointer_pointee_ty(dst_ty, llbc) else {
        return false;
    };
    let Some(src_size) = tyref_exact_layout_size(&src_pointee, llbc) else {
        return false;
    };
    let Some(dst_size) = tyref_exact_layout_size(&dst_pointee, llbc) else {
        return false;
    };
    dst_size >= src_size
}

/// One `Ref` or `RawPtr` layer. `*mut *mut u8` yields `*mut u8`.
pub(in crate::front::mir) fn pointer_pointee_ty(ty: &TyRef, llbc: &Llbc) -> Option<TyRef> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let obj = node.as_object()?;
    let pointee = if let Some(parts) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
        parts.first()?
    } else {
        let parts = obj.get("Ref").and_then(serde_json::Value::as_array)?;
        parts.get(1)?
    };
    serde_json::from_value(pointee.clone()).ok()
}

/// `*q` writes the place `q` names. `(*q).field` and `**r` do not: the
/// referent is one dereference of a pointer.
pub(in crate::front::mir) fn dereference_referent(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressReferent> {
    let PlaceKind::Projection(base, elem) = &place.kind else {
        return None;
    };
    if !projection_is_deref(elem) {
        return None;
    }
    reference_referent(base, depths, llbc)
}

/// The destination address was computed from the spill address. Bit 0
/// of the pointer is that address. `q = &p` holds it one step down, so
/// `*q = clean` overwrites the local. `*p = clean` on the spill pointer
/// itself is the pointee update.
pub(in crate::front::mir) fn store_through_derived_pointer(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    match &place.kind {
        PlaceKind::Projection(base, elem) if projection_is_deref(elem) => {
            let ptr = place_address(base, depths, llbc);
            // A higher bit is the address after another dereference.
            // `place_address` already shifted the nested dereference.
            let derived = !local_is_direct_pointer(base, depths, llbc)
                && (ptr.bits & 1 != 0 || ptr.condition > 0 || ptr.overflows);
            derived || store_through_derived_pointer(base, depths, llbc)
        }
        PlaceKind::Projection(base, _) => store_through_derived_pointer(base, depths, llbc),
        _ => false,
    }
}

pub(in crate::front::mir) fn clean_address_value() -> AddressValue {
    AddressValue {
        bits: 0,
        condition: 0,
        overflows: false,
        invariant: false,
    }
}

pub(in crate::front::mir) fn overflow_address_value() -> AddressValue {
    AddressValue {
        bits: 0,
        condition: 0,
        overflows: true,
        invariant: false,
    }
}

pub(in crate::front::mir) enum SpillLenKind {
    Array,
    Slice,
}

/// `{"Array": [elem, len, ..]}` or `{"Slice": [elem, ..]}`.
pub(in crate::front::mir) fn spill_len_kind(ty: &TyRef, llbc: &Llbc) -> Option<SpillLenKind> {
    let node = strip_ty_indirections(tyref_node(ty, llbc)?, llbc)?;
    let obj = node.as_object()?;
    if obj.contains_key("Array") {
        Some(SpillLenKind::Array)
    } else if obj.contains_key("Slice") {
        Some(SpillLenKind::Slice)
    } else {
        None
    }
}

/// The place `Len` reads, or one pointer whose pointee is that place.
/// `[T; N]` and `&[T; N]` are the const length. `[T]` and `&[T]` are
/// the slice length.
pub(in crate::front::mir) fn place_spill_len_kind(
    place: &Place,
    llbc: &Llbc,
) -> Option<SpillLenKind> {
    spill_len_kind(&place.ty, llbc).or_else(|| {
        let pointee = pointer_pointee_ty(&place.ty, llbc)?;
        spill_len_kind(&pointee, llbc)
    })
}

/// Fat-pointer length lives on the pointer. `Len(*q)` reads `q`.
/// A field of an aggregate has no length slot, so that `Len` stays
/// unlowered.
pub(in crate::front::mir) fn place_length_metadata(
    place: &Place,
    depths: &[LocalAddress],
) -> Option<AddressValue> {
    let slot_place = match &place.kind {
        PlaceKind::Projection(base, elem) if projection_is_deref(elem) => base.as_ref(),
        _ => place,
    };
    let PlaceKind::Local(id) = &slot_place.kind else {
        return None;
    };
    depths
        .iter()
        .find(|slot| slot.local == *id)
        .and_then(|slot| slot.length)
}

/// `Len` of a fixed array is the const generic, including when an
/// element holds the spill address. `Len` of a slice is the length
/// recorded for that slice (`rvalue_length_metadata`). The payload's
/// address is not the length. Any other place stays unlowered: `Len`
/// is only defined for slices and arrays.
pub(in crate::front::mir) fn len_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match place_spill_len_kind(place, llbc) {
        Some(SpillLenKind::Array) => clean_address_value(),
        Some(SpillLenKind::Slice) => {
            place_length_metadata(place, depths).unwrap_or_else(overflow_address_value)
        }
        None => overflow_address_value(),
    }
}

/// `Unsize` to a slice carries a const length. `&[T; N]` does too. A
/// copy of a slice pointer keeps the length already recorded on it.
/// A cast that is not `Unsize`, and a slice this walk never saw a
/// length for, record nothing.
pub(in crate::front::mir) fn rvalue_length_metadata(
    rvalue: &Rvalue,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressValue> {
    match rvalue {
        Rvalue::UnaryOp(op, _) if unary_op_is_cast(op) && cast_is_unsize(op) => {
            Some(clean_address_value())
        }
        Rvalue::Cast(kind, _, _) if cast_is_unsize(kind) => Some(clean_address_value()),
        Rvalue::Use(op, _) => operand_length_metadata(op, depths),
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            ref_length_metadata(place, depths, llbc)
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn operand_length_metadata(
    op: &Operand,
    depths: &[LocalAddress],
) -> Option<AddressValue> {
    match op {
        Operand::Copy(place) | Operand::Move(place) => place_length_metadata(place, depths),
        Operand::Const(_) => None,
    }
}

pub(in crate::front::mir) fn ref_length_metadata(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<AddressValue> {
    match spill_len_kind(&place.ty, llbc) {
        Some(SpillLenKind::Array) => Some(clean_address_value()),
        Some(SpillLenKind::Slice) => place_length_metadata(place, depths),
        None => None,
    }
}

/// `{"Cast": {"Unsize": [src, dst, length]}}` or a bare `"Unsize"`.
/// The length operand is a const generic.
pub(in crate::front::mir) fn cast_is_unsize(kind: &serde_json::Value) -> bool {
    let kind = kind.get("Cast").unwrap_or(kind);
    kind.as_str().is_some_and(|tag| tag == "Unsize")
        || kind
            .as_object()
            .is_some_and(|obj| obj.contains_key("Unsize"))
}

pub(in crate::front::mir) fn rvalue_address(
    rvalue: &Rvalue,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match rvalue {
        Rvalue::Use(op, _) | Rvalue::UnaryOp(_, op) | Rvalue::Cast(_, op, _) => {
            operand_address(op, depths, llbc)
        }
        Rvalue::BinaryOp(op, lhs, rhs) if binop_is_comparison(op) => {
            let left = operand_address(lhs, depths, llbc);
            let right = operand_address(rhs, depths, llbc);
            let mut overflows = left.overflows || right.overflows;
            // `install_raw_scalar_address_call` rewrites the spilled
            // borrow and leaves an already-raw pointer alone. `p == q`
            // then answers false where the interpreter answered true.
            // A null constant compares equal to neither address.
            // `(p as usize) & MASK == 0` reads the low bits of whichever
            // address this call used.
            if (left.bits != 0 || right.bits != 0)
                && !comparison_with_null(lhs, rhs, &left, &right, depths, llbc)
            {
                overflows = true;
            }
            let condition = u8::from(
                !overflows
                    && (left.bits != 0
                        || right.bits != 0
                        || left.condition > 0
                        || right.condition > 0),
            );
            // Ordering against zero can see the sign of the address.
            // Only `==` / `!=` stay the same branch after the spill moves.
            // A comparison of statuses stays invariant when each status is.
            let invariant = condition > 0
                && if left.bits != 0 || right.bits != 0 {
                    comparison_with_null(lhs, rhs, &left, &right, depths, llbc)
                        && binop_is_equality(op)
                } else {
                    (left.condition == 0 || left.invariant)
                        && (right.condition == 0 || right.invariant)
                };
            AddressValue {
                bits: 0,
                condition,
                overflows,
                invariant,
            }
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            let left = operand_address(lhs, depths, llbc);
            let right = operand_address(rhs, depths, llbc);
            // `p == null` is 0 or 1 on either address. Adding it does
            // not rebuild the spill address. `p < 0` can, so that sum
            // stays tainted.
            let relocates = (left.condition > 0 && !left.invariant)
                || (right.condition > 0 && !right.invariant);
            AddressValue {
                bits: left.bits | right.bits | u64::from(relocates),
                condition: 0,
                overflows: left.overflows || right.overflows,
                invariant: false,
            }
        }
        Rvalue::Aggregate(kind, ops) => {
            let mut bits = 0;
            let mut condition: u8 = 0;
            let mut overflows = false;
            let mut saw_condition = false;
            let mut invariant = true;
            for op in ops {
                let value = operand_address(op, depths, llbc);
                bits |= value.bits;
                condition = condition
                    .saturating_add(value.condition)
                    .min(SPILL_CONDITION_MANY);
                overflows |= value.overflows;
                if value.condition > 0 {
                    saw_condition = true;
                    invariant &= value.invariant;
                }
            }
            // Two union operands that name different places, or a union
            // operand whose fields name a place, have no single referent
            // (`fold_operand_referents`).
            if aggregate_kind_is_union(kind, llbc)
                && matches!(
                    fold_operand_referents(ops, depths, llbc),
                    FoldedReferent::Overflow
                )
            {
                overflows = true;
            }
            AddressValue {
                bits,
                condition,
                overflows,
                invariant: saw_condition && invariant && !overflows,
            }
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            ref_address(place, depths, llbc)
        }
        Rvalue::Repeat(op, _, count, _) => {
            let mut value = operand_address(op, depths, llbc);
            // `[q; N]` copies `q`. A count this walk can name stores
            // `q`'s referent on each element (`repeat_slots`). An
            // unknown count would drop that name.
            let named = repeat_element_count(llbc, count).is_some();
            let hidden = operand_referent(op, depths, llbc).is_some()
                || slots_have_referent(&operand_nested_slots(op, depths, llbc));
            if hidden && !named {
                value.overflows = true;
            }
            value
        }
        Rvalue::ShallowInitBox(op, _) => operand_address(op, depths, llbc),
        Rvalue::Discriminant(place) => {
            let value = place_address(place, depths, llbc);
            // `((p as usize >> k) & 3) as u8` is a four-variant tag. Its
            // discriminant reads those address bits back. A comparison
            // stored in the place is still one bit.
            let status = value.bits == 0;
            AddressValue {
                bits: 0,
                condition: if status { value.condition } else { 0 },
                overflows: value.overflows || value.bits != 0,
                invariant: status && value.invariant,
            }
        }
        Rvalue::Len(place) => len_address(place, depths, llbc),
        Rvalue::NullaryOp(_, _) => AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        },
        Rvalue::Unknown => AddressValue {
            bits: 0,
            condition: 0,
            overflows: true,
            invariant: false,
        },
    }
}

/// `SetDiscriminant` writes a variant index. `Nop` has no operands.
/// The typed kind of `SetDiscriminant` is an error because its payload
/// is not a unit; the raw key is what names it. The place still
/// selects where that index is written. `CopyNonOverlapping` and any
/// other key can copy the address.
pub(in crate::front::mir) fn statement_cannot_carry_address(
    kind: &serde_json::Value,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    if kind.as_str() == Some("Nop") {
        return true;
    }
    let Some(obj) = kind.as_object() else {
        return false;
    };
    if obj.len() != 1 {
        return false;
    }
    match obj.keys().next().map(String::as_str) {
        Some("Nop") => true,
        Some("SetDiscriminant") => {
            set_discriminant_place_is_independent(obj.get("SetDiscriminant"), depths, llbc)
        }
        _ => false,
    }
}

/// The variant index is not the address. The place must not be chosen
/// by it. A local, a global, and `*p` on the spill pointer are fixed.
/// A pointer computed from the address, an index that carries it, or a
/// place this walk cannot name is not.
pub(in crate::front::mir) fn set_discriminant_place_is_independent(
    payload: Option<&serde_json::Value>,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    let Some(place_json) = payload
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
    else {
        return false;
    };
    let Ok(place) = serde_json::from_value::<Place>(place_json.clone()) else {
        return false;
    };
    if matches!(place.kind, PlaceKind::Unknown)
        || store_through_derived_pointer(&place, depths, llbc)
    {
        return false;
    }
    let index = place_index_address(&place, depths, llbc);
    index.bits == 0 && index.condition == 0 && !index.overflows
}

/// `&*p` keeps depth 0 when `p` holds the address. `&place` is one
/// dereference above every depth `place` can hold.
pub(in crate::front::mir) fn ref_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    let mut bits = 0;
    if let PlaceKind::Projection(base, elem) = &place.kind
        && projection_is_deref(elem)
        && place_address(base, depths, llbc).bits & 1 != 0
    {
        bits |= 1;
    }
    let value = place_address(place, depths, llbc);
    let (lifted, overflows) = lift_address_bits(value.bits);
    AddressValue {
        bits: bits | lifted,
        condition: value.condition,
        overflows: overflows || value.overflows,
        invariant: value.invariant,
    }
}

pub(in crate::front::mir) fn lift_address_bits(bits: u64) -> (u64, bool) {
    let overflows = bits & (1 << SPILL_REF_DEPTH_LIMIT) != 0;
    let kept = bits & !(1 << SPILL_REF_DEPTH_LIMIT);
    (kept << 1, overflows)
}

pub(in crate::front::mir) fn binop_is_comparison(op: &serde_json::Value) -> bool {
    matches!(
        binop_label(op).ok().as_deref(),
        Some("eq" | "ne" | "lt" | "le" | "gt" | "ge")
    )
}

/// `==` and `!=` with zero do not read the address bits. Ordering does.
pub(in crate::front::mir) fn binop_is_equality(op: &serde_json::Value) -> bool {
    matches!(binop_label(op).ok().as_deref(), Some("eq" | "ne"))
}

/// One side is still the spill pointer. The other is a constant zero.
/// A copy, `&*p`, and a cast whose destination holds every address bit
/// keep that pointer. `(p as usize) & MASK` and `p as usize as u8` do
/// not: comparing either with zero reads the address bits.
/// `const_expr_literal` resolves a `Deduplicated` body. An unparsed
/// constant is not zero, and neither is a second pointer operand.
pub(in crate::front::mir) fn comparison_with_null(
    lhs: &Operand,
    rhs: &Operand,
    left: &AddressValue,
    right: &AddressValue,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> bool {
    let left_is_pointer = operand_is_direct_pointer(lhs, depths, llbc) && left.condition == 0;
    let right_is_pointer = operand_is_direct_pointer(rhs, depths, llbc) && right.condition == 0;
    match (left_is_pointer, right_is_pointer) {
        (true, false) => {
            right.bits == 0 && right.condition == 0 && const_operand_is_zero(rhs, llbc)
        }
        (false, true) => left.bits == 0 && left.condition == 0 && const_operand_is_zero(lhs, llbc),
        _ => false,
    }
}

pub(in crate::front::mir) fn const_operand_is_zero(op: &Operand, llbc: &Llbc) -> bool {
    let Operand::Const(value) = op else {
        return false;
    };
    llbc.const_expr_literal(value)
        .is_some_and(|literal| scalar_literal_is_zero(&literal))
}

pub(in crate::front::mir) fn scalar_literal_is_zero(literal: &serde_json::Value) -> bool {
    let Some(scalar) = literal.get("Scalar").and_then(serde_json::Value::as_object) else {
        return false;
    };
    scalar.values().any(|payload| {
        payload
            .as_array()
            .and_then(|items| items.get(1))
            .is_some_and(json_number_is_zero)
    })
}

pub(in crate::front::mir) fn json_number_is_zero(value: &serde_json::Value) -> bool {
    if value.as_i64() == Some(0) || value.as_u64() == Some(0) {
        return true;
    }
    value.as_str().and_then(|text| text.parse::<i128>().ok()) == Some(0)
}

pub(in crate::front::mir) fn operand_address(
    op: &Operand,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match op {
        Operand::Copy(place) | Operand::Move(place) => place_address(place, depths, llbc),
        Operand::Const(_) => AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        },
    }
}

pub(in crate::front::mir) fn operand_nested_slots(
    op: &Operand,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Vec<FieldSlot> {
    let place = match op {
        Operand::Copy(place) | Operand::Move(place) => place,
        Operand::Const(_) => return Vec::new(),
    };
    nested_slots_of_place(place, depths, llbc).unwrap_or_default()
}

pub(in crate::front::mir) fn nested_slots_of_place(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<Vec<FieldSlot>> {
    let (root, path) = constant_field_path(place, llbc)?;
    if path.is_empty() {
        let local = depths
            .iter()
            .find(|slot| slot.local == root && slot.split)?;
        return Some(local.slots.clone());
    }
    Some(lookup_field_path(depths, root, &path)?.slots)
}

/// `Local` plus `Field` or a constant `Index`. A dynamic index and
/// `Deref` are not a stable leaf. The path is root to leaf. An empty
/// path is the local.
pub(in crate::front::mir) fn constant_field_path(
    place: &Place,
    llbc: &Llbc,
) -> Option<(u64, Vec<usize>)> {
    fn walk(place: &Place, path: &mut Vec<usize>, llbc: &Llbc) -> Option<u64> {
        match &place.kind {
            PlaceKind::Local(id) => Some(*id),
            PlaceKind::Projection(base, elem) => {
                let index = projection_slot_index(elem, llbc)?;
                let root = walk(base, path, llbc)?;
                path.push(index);
                Some(root)
            }
            _ => None,
        }
    }
    let mut path = Vec::new();
    let root = walk(place, &mut path, llbc)?;
    Some((root, path))
}

pub(in crate::front::mir) fn lookup_field_path(
    depths: &[LocalAddress],
    root: u64,
    path: &[usize],
) -> Option<FieldSlot> {
    let local = depths
        .iter()
        .find(|slot| slot.local == root && slot.split)?;
    let mut indexes = path.iter();
    let first = indexes.next()?;
    let mut current = local.slots.iter().find(|slot| slot.index == *first)?;
    for index in indexes {
        current = current.slots.iter().find(|slot| slot.index == *index)?;
    }
    Some(current.clone())
}

pub(in crate::front::mir) fn nested_field_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> Option<FieldSlot> {
    let (root, path) = constant_field_path(place, llbc)?;
    if path.is_empty() {
        return None;
    }
    lookup_field_path(depths, root, &path)
}

/// A `Field` or a constant `Index` selects one aggregate operand.
/// The index is a `usize` literal (`const_expr_literal`) or the bare
/// number the fixtures spell.
pub(in crate::front::mir) fn projection_slot_index(
    elem: &ProjectionElem,
    llbc: &Llbc,
) -> Option<usize> {
    let ProjectionElem::Tagged(value) = elem else {
        return None;
    };
    let obj = value.as_object()?;
    if let Some(field) = obj.get("Field") {
        if let Some(index) = field.as_u64() {
            return Some(index as usize);
        }
        return field
            .as_array()?
            .last()?
            .as_u64()
            .map(|index| index as usize);
    }
    let index = obj.get("Index")?;
    if index.get("from_end").and_then(serde_json::Value::as_bool) == Some(true) {
        return None;
    }
    let offset = index.get("offset")?;
    let Operand::Const(expr) = serde_json::from_value::<Operand>(offset.clone()).ok()? else {
        return None;
    };
    const_operand_index(llbc, &expr).map(|index| index as usize)
}

/// `{"Const": 1}` in a fixture, or a `ConstantExpr` integer literal.
pub(in crate::front::mir) fn const_operand_index(
    llbc: &Llbc,
    expr: &serde_json::Value,
) -> Option<u64> {
    if let Some(index) = expr.as_u64() {
        return Some(index);
    }
    let literal = llbc.const_expr_literal(expr)?;
    let scalar = literal.get("Scalar")?;
    let pair = scalar
        .get("Unsigned")
        .or_else(|| scalar.get("Signed"))?
        .as_array()?;
    let value: i128 = pair.get(1)?.as_str()?.parse().ok()?;
    u64::try_from(value).ok()
}

/// Bit 0 is the address. A dereference shifts the set down, so depth 0
/// becomes the pointee and depth `n` reloads depth `n - 1`. An `Index`
/// offset is part of the value: its address bits and its comparison
/// join the element's.
pub(in crate::front::mir) fn place_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match &place.kind {
        PlaceKind::Local(id) => AddressValue {
            bits: depth_bits(depths, *id),
            condition: local_condition(depths, *id),
            overflows: false,
            invariant: local_invariant(depths, *id),
        },
        PlaceKind::Projection(base, elem) if projection_is_deref(elem) => {
            // `q = &p` names `p`. `q = &pair.0` names that field. `*q`
            // is that place's current value, so a store through `q` is
            // visible here. A pointer computed from the address has no
            // referent.
            if let Some(referent) = reference_referent(base, depths, llbc) {
                return referent_value(depths, &referent);
            }
            let inner = place_address(base, depths, llbc);
            // `*p` loads the pointee. `*q` loads a value chosen by the
            // address when `q` was computed from it. Shifting bit 0 off
            // would report that load as clean.
            let derived = !local_is_direct_pointer(base, depths, llbc) && inner.bits & 1 != 0;
            AddressValue {
                bits: inner.bits >> 1,
                condition: inner.condition,
                overflows: inner.overflows || derived,
                invariant: inner.invariant,
            }
        }
        PlaceKind::Projection(base, elem) => {
            if let Some(field) = nested_field_address(place, depths, llbc) {
                let index = projection_index_address(elem, depths, llbc);
                return AddressValue {
                    bits: field.bits | index.bits,
                    condition: field.condition.max(index.condition),
                    overflows: index.overflows,
                    invariant: merged_invariant(
                        field.condition,
                        field.invariant,
                        index.condition,
                        index.invariant,
                    ),
                };
            }
            let inner = place_address(base, depths, llbc);
            let index = projection_index_address(elem, depths, llbc);
            AddressValue {
                bits: inner.bits | index.bits,
                condition: inner.condition.max(index.condition),
                overflows: inner.overflows || index.overflows,
                invariant: merged_invariant(
                    inner.condition,
                    inner.invariant,
                    index.condition,
                    index.invariant,
                ),
            }
        }
        PlaceKind::Global { .. } => AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        },
        PlaceKind::Unknown => AddressValue {
            bits: 0,
            condition: 0,
            overflows: true,
            invariant: false,
        },
    }
}

/// `Index { offset, from_end }` reads `offset`. `Offset { bytes, .. }`
/// reads `bytes` the same way. A comparison used as that operand stays
/// a condition on the element. An operand that does not decode is the
/// address.
pub(in crate::front::mir) fn projection_index_address(
    elem: &ProjectionElem,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    let ProjectionElem::Tagged(v) = elem else {
        return AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        };
    };
    let Some(obj) = v.as_object() else {
        return AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        };
    };
    let operand = if let Some(index) = obj.get("Index") {
        index.get("offset")
    } else if let Some(offset) = obj.get("Offset") {
        offset.get("bytes")
    } else {
        return AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        };
    };
    let Some(offset) = operand else {
        return AddressValue {
            bits: 1,
            condition: 0,
            overflows: true,
            invariant: false,
        };
    };
    match serde_json::from_value::<Operand>(offset.clone()) {
        Ok(op) => operand_address(&op, depths, llbc),
        Err(_) => AddressValue {
            bits: 1,
            condition: 0,
            overflows: true,
            invariant: false,
        },
    }
}

/// Every `Index` offset on `place`, ignoring the value stored there.
pub(in crate::front::mir) fn place_index_address(
    place: &Place,
    depths: &[LocalAddress],
    llbc: &Llbc,
) -> AddressValue {
    match &place.kind {
        PlaceKind::Projection(base, elem) => {
            let inner = place_index_address(base, depths, llbc);
            let index = projection_index_address(elem, depths, llbc);
            AddressValue {
                bits: inner.bits | index.bits,
                condition: inner.condition.max(index.condition),
                overflows: inner.overflows || index.overflows,
                invariant: merged_invariant(
                    inner.condition,
                    inner.invariant,
                    index.condition,
                    index.invariant,
                ),
            }
        }
        _ => AddressValue {
            bits: 0,
            condition: 0,
            overflows: false,
            invariant: false,
        },
    }
}

/// An `Index` whose offset is not a constant can select a different
/// slot each time the store runs.
pub(in crate::front::mir) fn place_index_is_dynamic(place: &Place) -> bool {
    match &place.kind {
        PlaceKind::Projection(base, elem) => {
            index_offset_is_dynamic(elem) || place_index_is_dynamic(base)
        }
        _ => false,
    }
}

pub(in crate::front::mir) fn index_offset_is_dynamic(elem: &ProjectionElem) -> bool {
    let ProjectionElem::Tagged(value) = elem else {
        return false;
    };
    let Some(index) = value.as_object().and_then(|obj| obj.get("Index")) else {
        return false;
    };
    let Some(offset) = index.get("offset") else {
        return true;
    };
    !matches!(
        serde_json::from_value::<Operand>(offset.clone()),
        Ok(Operand::Const(_))
    )
}

/// Distinct field and constant-index stores. A repeated walk of the
/// same projection does not count twice.
pub(in crate::front::mir) fn note_projected_condition(
    notes: &mut Vec<(u64, String)>,
    local: u64,
    key: String,
) -> u8 {
    if !notes
        .iter()
        .any(|(seen, seen_key)| *seen == local && seen_key == &key)
    {
        notes.push((local, key));
    }
    notes
        .iter()
        .filter(|(seen, _)| *seen == local)
        .count()
        .min(usize::from(SPILL_CONDITION_MANY)) as u8
}

pub(in crate::front::mir) fn projection_key(place: &Place) -> String {
    match &place.kind {
        PlaceKind::Projection(base, elem) => {
            let here = match elem {
                ProjectionElem::Atom(label) => label.clone(),
                ProjectionElem::Tagged(value) => value.to_string(),
            };
            format!("{}/{}", projection_key(base), here)
        }
        PlaceKind::Local(id) => id.to_string(),
        PlaceKind::Global { id, .. } => format!("g{id}"),
        PlaceKind::Unknown => "unknown".to_string(),
    }
}

/// The local under `place`, stopping at a dereference. `*p` has no
/// local to update: the store writes the pointee.
pub(in crate::front::mir) fn spill_place_root_local(place: &Place) -> Option<u64> {
    match &place.kind {
        PlaceKind::Local(id) => Some(*id),
        PlaceKind::Projection(base, elem) if !projection_is_deref(elem) => {
            spill_place_root_local(base)
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn projection_is_deref(elem: &ProjectionElem) -> bool {
    match elem {
        ProjectionElem::Atom(label) => label == "Deref",
        ProjectionElem::Tagged(value) => {
            value.as_str() == Some("Deref")
                || value
                    .as_object()
                    .is_some_and(|obj| obj.contains_key("Deref"))
        }
    }
}

/// `output`, or the `Ok` payload of a `Result<T, PyError>`, is a raw
/// scalar address. Any other `Result` stays the ADT
/// [`dont_look_inside_return_token`] leaves whole.
pub(in crate::front::mir) fn output_is_raw_scalar_address(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
) -> bool {
    if raw_scalar_address_value_type(output, llbc).is_some() {
        return true;
    }
    if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec)
        && let Some(ok) = crate::front::result_exc::tyref_result_ok(output, llbc)
    {
        return raw_scalar_address_value_type(&ok, llbc).is_some();
    }
    false
}

/// `output`, or the `Ok` payload of a `Result<T, PyError>`, is a raw
/// address word: a primitive scalar pointer, or a declared-Raw ADT
/// pointer (`raw_ptr_node_is_declared_raw` / `declared_raw_adt_node`).
/// Ordinary looked-inside functions skip the `i64` FUNC.RESULT stamp
/// for these.
pub(in crate::front::mir) fn output_is_raw_address_word(
    output: &TyRef,
    llbc: &Llbc,
    spec: crate::ErrorCarrierSpec<'_>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> bool {
    if output_is_raw_scalar_address(output, llbc, spec) {
        return true;
    }
    let payload = if crate::front::result_exc::tyref_is_result_of_carrier(output, llbc, spec) {
        crate::front::result_exc::tyref_result_ok(output, llbc)
    } else {
        None
    };
    let kind_src = payload.as_ref().unwrap_or(output);
    ty_is_raw_address(kind_src, llbc, gc_struct_ids, raw_alloc_flavor_structs)
}

/// `ty` itself is a Charon primitive scalar, not a pointer to one.
pub(in crate::front::mir) fn tyref_is_primitive_scalar(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|node| strip_ty_indirections(node, llbc))
        .is_some_and(|node| json_ty_is_copy_scalar(node, llbc))
}

/// Peel `Deduplicated` / `Value` wrappers off a Charon value.
pub(in crate::front::mir) fn peel_hashcons_value<'a>(
    llbc: &'a Llbc,
    mut value: &'a serde_json::Value,
) -> &'a serde_json::Value {
    loop {
        let Some(obj) = value.as_object() else {
            break;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            match llbc.dedup_body(id) {
                Some(body) => {
                    value = body;
                    continue;
                }
                None => break,
            }
        }
        if let Some(body) = obj
            .get("Value")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| arr.get(1))
        {
            value = body;
            continue;
        }
        break;
    }
    value
}

pub(in crate::front::mir) fn borrowed_place_referent(rvalue: &Rvalue) -> Option<Place> {
    let place = match rvalue {
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => place,
        _ => return None,
    };
    Some(place.clone())
}

/// Charon's `Place` carries a `TyRef`, which is not `Clone` (see
/// [`clone_tyref`]); this is the same by-hand deep copy for a whole place.
pub(in crate::front::mir) fn clone_place(p: &Place) -> Place {
    Place {
        kind: match &p.kind {
            PlaceKind::Local(i) => PlaceKind::Local(*i),
            PlaceKind::Projection(inner, elem) => {
                PlaceKind::Projection(Box::new(clone_place(inner)), clone_projection_elem(elem))
            }
            PlaceKind::Global { generics, id } => PlaceKind::Global {
                generics: generics.clone(),
                id: *id,
            },
            PlaceKind::Unknown => PlaceKind::Unknown,
        },
        ty: clone_tyref(&p.ty),
    }
}

pub(in crate::front::mir) fn clone_projection_elem(e: &ProjectionElem) -> ProjectionElem {
    match e {
        ProjectionElem::Atom(s) => ProjectionElem::Atom(s.clone()),
        ProjectionElem::Tagged(v) => ProjectionElem::Tagged(v.clone()),
    }
}

/// Same borrowed place, ignoring the place's type. A `Use` copy is a
/// different local even when it aliases the word. An unresolved borrow
/// does not alias another unresolved borrow.
pub(in crate::front::mir) fn raw_scalar_groups_share(
    group: &RawScalarAddressGroup,
    place: Option<&Place>,
    item_ty: &ValueType,
    itemsize: usize,
    is_item_signed: bool,
) -> bool {
    let (Some(group_place), Some(place)) = (group.place.as_ref(), place) else {
        return false;
    };
    places_alias(group_place, place)
        && group.item_ty == *item_ty
        && group.itemsize == itemsize
        && group.is_item_signed == is_item_signed
}

pub(in crate::front::mir) fn merge_address_source(
    found: Option<SpillAddressSource>,
    next: SpillAddressSource,
) -> Option<SpillAddressSource> {
    match (found, next) {
        (None, next) => Some(next),
        (Some(SpillAddressSource::Constant), SpillAddressSource::Constant) => {
            Some(SpillAddressSource::Constant)
        }
        (Some(SpillAddressSource::Place(prev)), SpillAddressSource::Place(next)) => {
            if places_alias(&prev, &next) {
                Some(SpillAddressSource::Place(prev))
            } else {
                Some(SpillAddressSource::Unknown)
            }
        }
        _ => Some(SpillAddressSource::Unknown),
    }
}

/// `place` is one spilled place, or a projection of one. A group with
/// no place covers every pointer: its referent was not a single borrow.
pub(in crate::front::mir) fn spill_place_aliases(
    place: &Place,
    groups: &[RawScalarAddressGroup],
) -> bool {
    let mut exact = false;
    let mut covered = false;
    for group in groups {
        let Some(group_place) = group.place.as_ref() else {
            covered = true;
            continue;
        };
        if places_alias(group_place, place) {
            exact = true;
        } else if place_extends(group_place, place) || place_extends(place, group_place) {
            covered = true;
        }
    }
    exact || covered
}

pub(in crate::front::mir) fn places_alias(a: &Place, b: &Place) -> bool {
    match (&a.kind, &b.kind) {
        (PlaceKind::Local(left), PlaceKind::Local(right)) => left == right,
        (
            PlaceKind::Projection(left_inner, left_elem),
            PlaceKind::Projection(right_inner, right_elem),
        ) => places_alias(left_inner, right_inner) && projection_elems_alias(left_elem, right_elem),
        (PlaceKind::Global { id: left, .. }, PlaceKind::Global { id: right, .. }) => left == right,
        _ => false,
    }
}

/// `inner` is `outer` plus at least one field or index.
/// `place_extends` of `pair` and `pair.word` is true. Those pointers
/// have different widths, so they do not share a spill address.
pub(in crate::front::mir) fn place_extends(outer: &Place, inner: &Place) -> bool {
    match &inner.kind {
        PlaceKind::Projection(base, _) => places_alias(outer, base) || place_extends(outer, base),
        _ => false,
    }
}

pub(in crate::front::mir) fn projection_elems_alias(
    a: &ProjectionElem,
    b: &ProjectionElem,
) -> bool {
    match (a, b) {
        (ProjectionElem::Atom(left), ProjectionElem::Atom(right)) => left == right,
        (ProjectionElem::Tagged(left), ProjectionElem::Tagged(right)) => left == right,
        _ => false,
    }
}

pub(in crate::front::mir) fn clone_tyref(ty: &TyRef) -> TyRef {
    match ty {
        TyRef::Dedup { id } => TyRef::Dedup { id: *id },
        TyRef::Inline { value: (id, v) } => TyRef::Inline {
            value: (*id, v.clone()),
        },
        TyRef::Other(v) => TyRef::Other(v.clone()),
    }
}

#[cfg(test)]
mod tests;
