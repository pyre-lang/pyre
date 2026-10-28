//! Hand lowering of integer / WTF-8 numeric callees.

use super::*;

impl<'a> Lowering<'a> {
    pub(in crate::front::mir) fn emit_int_binop(
        &mut self,
        bb_id: BlockId,
        op: &str,
        lhs: Variable,
        rhs: Variable,
    ) -> Variable {
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: op.to_string(),
                lhs,
                rhs,
                result_ty: ValueType::Int,
            },
        });
        res
    }

    /// Normalize Opaque `core::cmp::{min,max}` (free function or
    /// `impls::<Impl>` method) to the `FunctionPath` the adapter already
    /// turns into `simple_call(min)` / `simple_call(max)`
    /// (`rtype_builtin_min`).  Left as a `Method`, the call never meets
    /// that arm and the rich-graph spine residualizes it.
    pub(in crate::front::mir) fn try_lower_cmp_minmax(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if args.len() != 2 {
            return Ok(false);
        }
        let [first, family, ..] = segments else {
            return Ok(false);
        };
        if first != "core" || family != "cmp" {
            return Ok(false);
        }
        let Some(leaf) = segments.last() else {
            return Ok(false);
        };
        if !matches!(leaf.as_str(), "min" | "max") {
            return Ok(false);
        }
        let result_ty = tyref_to_value_type_with(
            dest_ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        if crate::codewriter::minmax::minmax_value_ty(&result_ty).is_none() {
            return Ok(false);
        }
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "cmp".into(), leaf.clone()],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(args.iter().cloned()),
                result_ty,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Opaque `core::cmp::{eq,ne,lt,le,gt,ge}` on a scalar pair is the
    /// like-named flowspace `BinOp` (`nonraising_core_bridge_opname`).
    /// The string-family arm above already takes `&Wtf8` / `&str`; this
    /// one is the integer and float impls that share `impls::<Impl>`
    /// and were left residual so they would not steal that fold.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_cmp_binop(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        first_arg_ty: Option<&TyRef>,
        second_arg_ty: Option<&TyRef>,
        target: usize,
    ) -> Result<bool, LowerError> {
        if args.len() != 2 {
            return Ok(false);
        }
        // A generic `PartialEq::eq` monomorphized to two scalars stays the
        // trait path `["PartialEq", "eq"]`. Same `BinOp` as `core::cmp::eq`.
        let trait_eq = segments.len() == 2 && segments[0] == "PartialEq" && segments[1] == "eq";
        let leaf = if trait_eq {
            "eq"
        } else {
            match crate::codewriter::minmax::cmp_binop_leaf(segments) {
                Some(leaf) => leaf,
                None => return Ok(false),
            }
        };
        let lhs = first_arg_ty.and_then(|ty| self.scalar_cmp_bank(ty));
        let rhs = second_arg_ty.and_then(|ty| self.scalar_cmp_bank(ty));
        // A missing bank is a const only when that operand has no type.
        // A typed str (or any non-scalar) is not an int const. Trait
        // `PartialEq` on floats stays residual (`float` IEEE vs `int_eq`).
        let typed_nonscalar =
            (first_arg_ty.is_some() && lhs.is_none()) || (second_arg_ty.is_some() && rhs.is_none());
        let trait_eq_float = trait_eq
            && (matches!(lhs, Some(ValueType::Float)) || matches!(rhs, Some(ValueType::Float)));
        let scalar_ok = !typed_nonscalar
            && !trait_eq_float
            && crate::codewriter::minmax::scalar_cmp_banks_compatible(
                lhs.as_ref(),
                rhs.as_ref(),
                leaf,
            );
        // `[u8]` / `str` equality is `BinOp("eq")` (`ll_streq`). The generic
        // `PartialEq::eq` path is what a monomorphized `Equivalent` body
        // still emits for those operands.
        let string_eq = trait_eq
            && [first_arg_ty, second_arg_ty]
                .into_iter()
                .all(|ty| ty.is_some_and(|ty| self.tyref_compares_as_string(ty)));
        if !scalar_ok && !string_eq {
            return Ok(false);
        }
        let op = if scalar_ok {
            crate::codewriter::minmax::scalar_cmp_opname(leaf, lhs.as_ref(), rhs.as_ref())
        } else {
            "eq".to_string()
        };
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op,
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

    /// Signed / unsigned / float / bool bank of `ty`, peeling one
    /// reference so `&i64` compares as `Int`.  Strings and ADTs stay
    /// `None` and keep their own folds.
    pub(in crate::front::mir) fn tyref_compares_as_string(&self, ty: &TyRef) -> bool {
        let peeled = self.tyref_peel_ref_to_pointee(ty);
        let ty = peeled.as_ref().unwrap_or(ty);
        if tyref_is_string_value(ty, self.llbc) {
            return true;
        }
        false
    }

    pub(in crate::front::mir) fn scalar_cmp_bank(&self, ty: &TyRef) -> Option<ValueType> {
        // `tyref_peel_ref_to_pointee` also peels `RawPtr`, so `*mut i64`
        // would look like `i64`.  Ordered pointer compares need
        // `cast_ptr_to_int`; there is no `ptr_lt`.
        if tyref_node(ty, self.llbc)
            .and_then(|node| strip_ty_wrappers(node, self.llbc))
            .and_then(|node| node.as_object())
            .is_some_and(|obj| obj.contains_key("RawPtr"))
        {
            return None;
        }
        let peeled = self.tyref_peel_ref_to_pointee(ty);
        let ty = peeled.as_ref().unwrap_or(ty);
        match tyref_to_value_type_with(
            ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ) {
            ty @ (ValueType::Int | ValueType::Unsigned | ValueType::Float | ValueType::Bool) => {
                Some(ty)
            }
            _ => None,
        }
    }

    /// Truncate a box-wide int (`jtransform.py` `_int_to_int_cast`:
    /// `int_and` unsigned, `int_signext` signed). The planner supplies
    /// [`crate::front::checked_arith_uint::WrappingTruncation`].
    pub(in crate::front::mir) fn emit_narrow_truncation(
        &mut self,
        bb_id: crate::model::BlockId,
        value: Variable,
        truncation: crate::front::checked_arith_uint::WrappingTruncation,
        result_ty: ValueType,
    ) -> Variable {
        use crate::front::checked_arith_uint::WrappingTruncation;
        match truncation {
            WrappingTruncation::None => value,
            WrappingTruncation::Mask(bits) => {
                let mask_val = (1u64 << bits) - 1;
                let mask = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(mask.clone()),
                    kind: OpKind::ConstUInt(mask_val),
                });
                let masked = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(masked.clone()),
                    kind: OpKind::BinOp {
                        op: "and".to_string(),
                        lhs: value,
                        rhs: mask,
                        result_ty,
                    },
                });
                masked
            }
            WrappingTruncation::SignExt(nbytes) => {
                // `jtransform.py` `_int_to_int_cast`: signed narrow target is
                // `int_signext(v, nbytes)` with `nbytes = size2` in bytes.
                // The front spells the leaf the way `"and"` becomes `int_and`.
                let width = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(width.clone()),
                    kind: OpKind::ConstInt(i64::from(nbytes)),
                });
                let extended = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(extended.clone()),
                    kind: OpKind::BinOp {
                        op: "signext".to_string(),
                        lhs: value,
                        rhs: width,
                        result_ty,
                    },
                });
                extended
            }
        }
    }

    pub(in crate::front::mir) fn try_lower_wrapping_binop(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core" || module.as_str() != "num" || impl_seg.as_str() != "<Impl>" {
            return Ok(false);
        }
        // `wrapping_div` / `wrapping_rem` are the C-truncating primitives, so
        // they are `llop.int_floordiv` / `llop.int_mod` the same way
        // `wrapping_add` is `llop.int_add`: `rewrite_op_int_floordiv` /
        // `rewrite_op_int_mod` route those two to `support.py`
        // `_ll_2_int_floordiv` / `_ll_2_int_mod`.  Those bodies call
        // `ll_int_py_div` / `ll_int_py_mod` and correct back to truncation;
        // they are not these primitives.  Left as a `Call`, the leaf is
        // Opaque in the LLBC and every division becomes a symbolic host
        // call the caller must guard for an exception it cannot raise.
        //
        // Unsigned does not follow here. The rename table that folds
        // `uint_{add,sub,mul}` back onto `int_*` — one machine op under two
        // rtyper dispatches — has no entry for `uint_floordiv` / `uint_mod`,
        // which really are different instructions, so an unsigned receiver
        // keeps the `Call` form. Width / truncation / shift-count live in
        // `wrapping_plan`.
        let [lhs, rhs] = args else {
            return Ok(false);
        };
        // Same width gate as `try_lower_checked_neg`: the operand is the
        // receiver — `wrapping_add(self, rhs)` — so read `inputs[0]`.
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        let word_bytes = crate::layout::target_word_size();
        let signed_atom = self.tyref_literal_int_atom(src);
        let unsigned_atom = self.tyref_literal_uint_atom(src);
        let atom = match (signed_atom, unsigned_atom) {
            (Some(atom), None) | (None, Some(atom)) => atom,
            _ => return Ok(false),
        };
        use crate::front::checked_arith_uint::WrappingPlan;
        let Some(WrappingPlan::Inline {
            op,
            unsigned_receiver,
            shift_count_mask,
            truncation,
        }) = crate::front::checked_arith_uint::wrapping_plan(atom, leaf, word_bytes)
        else {
            return Ok(false);
        };
        if op == "neg" {
            return Ok(false);
        }
        let bb_id = self.emit_bb(mir_bb);
        let result_ty = if unsigned_receiver {
            ValueType::Unsigned
        } else {
            ValueType::Int
        };
        let shift_rhs = if let Some(count_mask) = shift_count_mask {
            // Rust `wrapping_shl` / `wrapping_shr` mask the count with
            // `bits - 1` of the receiver (`u32` and a 4-byte usize: 31).
            let mask = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(mask.clone()),
                kind: if unsigned_receiver {
                    OpKind::ConstUInt(u64::from(count_mask))
                } else {
                    OpKind::ConstInt(i64::from(count_mask))
                },
            });
            let masked = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(masked.clone()),
                kind: OpKind::BinOp {
                    op: "and".to_string(),
                    lhs: rhs.clone(),
                    rhs: mask,
                    result_ty: result_ty.clone(),
                },
            });
            masked
        } else {
            rhs.clone()
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::BinOp {
                op: op.to_string(),
                lhs: lhs.clone(),
                rhs: shift_rhs,
                result_ty: result_ty.clone(),
            },
        });
        let res = self.emit_narrow_truncation(bb_id, res, truncation, result_ty);
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `core::ptr::null_mut::<T>()` / `null::<T>()` (and the `std::ptr`
    /// re-exports) whose result is a pointer to a Raw ADT.
    ///
    /// Upstream writes that null as `lltype.nullptr(T)` of the raw
    /// `Ptr` (`rbuiltin.py` `rtype_null_raw`, `annotator/builtin.rs`
    /// `raw_ptr_null_constant`). A GC `null_mut` recast through
    /// `__cast_instance_intrinsic` is a `Ptr(GcStruct)` that
    /// `pairtype(GCRefRepr, Repr).convert_from_to` refuses.
    ///
    /// Raw is [`ty_is_raw_address`] / [`raw_address_owner_root`]
    /// (`gc_struct_ids`). An `_alloc_flavor_ = raw` pointee that still
    /// banks as `Ref` (Opaque host fields fail `adt_def_is_raw_storage`)
    /// uses the same `null_raw` spelling; `raw_ptr_null_constant` then
    /// annotates a nullable `SomeInstance` of that classdef
    /// (`classdesc.get_param('_alloc_flavor_')`). `raw_struct_ptr_annotation`
    /// is storage-raw only and does not accept a flavor-raw spelling.
    /// A GC pointee returns `false` and keeps `null_mut`.
    pub(in crate::front::mir) fn try_lower_raw_adt_ptr_null(
        &mut self,
        mir_bb: usize,
        call: &CallPayload,
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        if !call.args.is_empty() {
            return Ok(false);
        }
        let CallFunc::Regular(reg) = &call.func else {
            return Ok(false);
        };
        let Some(path) = regular_call_name_path(reg, self.llbc) else {
            return Ok(false);
        };
        if !is_core_ptr_null_fun_path(&path) {
            return Ok(false);
        }
        let Some(spelling) = self.raw_adt_ptr_null_spelling(&call.dest.ty) else {
            return Ok(false);
        };
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .push_null_raw_ptr_with_spelling(bb_id, spelling.as_deref());
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Class-root argument of `null_raw` (`lltype.nullptr(T)`), or `None`
    /// to keep the GC `null_mut` lowering.
    ///
    /// Storage-raw pointees (`ty_is_raw_address`) use
    /// [`raw_address_owner_root`]. Flavor-raw pointees
    /// (`_alloc_flavor_ = raw`) use the class-root spelling
    /// harvested into `raw_alloc_flavor_structs`. Annotation of that
    /// spelling is `raw_ptr_null_constant`'s flavor-raw `SomeInstance`
    /// arm; `raw_struct_ptr_annotation` does not accept it.
    pub(in crate::front::mir) fn raw_adt_ptr_null_spelling(
        &self,
        dest_ty: &TyRef,
    ) -> Option<Option<String>> {
        if ty_is_raw_address(
            dest_ty,
            self.llbc,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ) {
            return Some(raw_address_owner_root(
                dest_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            ));
        }
        let node = tyref_node(dest_ty, self.llbc).and_then(|n| strip_ty_wrappers(n, self.llbc))?;
        let root = raw_ptr_pointee_class_root_with(node, self.llbc, self.tombstoned_leaves)?;
        crate::front::llbc_hints::alloc_flavor_raw_contains(&self.raw_alloc_flavor_structs, &root)
            .then_some(Some(root))
    }

    /// `<*const T>::read` / `<*mut T>::read` of a one-word pointee is
    /// `raw_load` at offset 0. A wider `T` stays a residual call.
    pub(in crate::front::mir) fn try_lower_ptr_read(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [core, ptr, which, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if core.as_str() != "core"
            || ptr.as_str() != "ptr"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "read"
            || (which.as_str() != "const_ptr" && which.as_str() != "mut_ptr")
        {
            return Ok(false);
        }
        let [base] = args else {
            return Ok(false);
        };
        let Some((item_ty, itemsize, is_item_signed)) = self.raw_word_descr(dest_ty) else {
            return Ok(false);
        };
        let bb_id = self.emit_bb(mir_bb);
        let offset = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(offset.clone()),
            kind: OpKind::ConstInt(0),
        });
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::RawLoad {
                base: base.clone(),
                offset,
                item_ty,
                itemsize,
                is_item_signed,
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `i64`/`u64`/`isize`/`usize::wrapping_neg` is `llop.int_neg`.  The
    /// binary wrapping table cannot take it: the method is unary. A
    /// receiver narrower than the i64 box (4-byte `usize`/`isize`, `i32`/
    /// `u32`) is `int_neg` then the same truncation as wrapping_add.
    pub(in crate::front::mir) fn try_lower_wrapping_unop(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "wrapping_neg"
        {
            return Ok(false);
        }
        let [operand] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        let word_bytes = crate::layout::target_word_size();
        let signed_atom = self.tyref_literal_int_atom(src);
        let unsigned_atom = self.tyref_literal_uint_atom(src);
        let atom = match (signed_atom, unsigned_atom) {
            (Some(atom), None) | (None, Some(atom)) => atom,
            _ => return Ok(false),
        };
        use crate::front::checked_arith_uint::WrappingPlan;
        let Some(WrappingPlan::Inline {
            op,
            unsigned_receiver,
            truncation,
            ..
        }) = crate::front::checked_arith_uint::wrapping_plan(atom, leaf, word_bytes)
        else {
            return Ok(false);
        };
        if op != "neg" {
            return Ok(false);
        }
        let bb_id = self.emit_bb(mir_bb);
        let result_ty = if unsigned_receiver {
            ValueType::Unsigned
        } else {
            ValueType::Int
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::UnaryOp {
                op: op.to_string(),
                operand: operand.clone(),
                result_ty: result_ty.clone(),
            },
        });
        let res = self.emit_narrow_truncation(bb_id, res, truncation, result_ty);
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Lower a constant `u64`/`usize::rotate_left` to the unsigned
    /// shift/or expansion used by PyPy's 64-bit tuple hash
    /// (`tupleobject.py`, `xxrotate`).  Core integer method bodies are
    /// opaque in LLBC; leaving this as a call therefore hides a primitive
    /// RPython operation behind an unregistrable Rust helper.
    ///
    /// Keep the gate deliberately narrow: the receiver must be a word-sized
    /// unsigned integer and the count must be a graph constant.  A signed
    /// receiver would require an explicit unsigned reinterpretation before
    /// the right shift, and a variable Rust count needs the full modulo-word
    /// normalization.  Neither is present in the current RPython-shaped
    /// source, so those forms remain residual instead of being guessed at.
    pub(in crate::front::mir) fn try_lower_word_rotate(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "rotate_left"
        {
            return Ok(false);
        }
        let [value, count] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(receiver_ty) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        // Upstream picks the rotate constants from the word width
        // (`tupleobject.py`: `if sys.maxint == 2 ** 31 - 1` chooses
        // `(x << 13) | (x >> 19)`, the else arm `(x << 31) | (x >> 33)`).
        // The fold below is that 64-bit arm, so `u64` always qualifies and
        // `usize` only where the target word IS eight bytes.
        let width_matches_the_64_bit_arm = match self.tyref_literal_uint_atom(receiver_ty) {
            Some("U64") => true,
            Some("Usize") => crate::layout::target_word_size() == 8,
            _ => false,
        };
        if !width_matches_the_64_bit_arm {
            return Ok(false);
        }
        let Some(raw_count) = self
            .graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find_map(|op| {
                (op.result.as_ref() == Some(count))
                    .then_some(match &op.kind {
                        OpKind::ConstInt(n) if *n >= 0 => Some(*n as u64),
                        OpKind::ConstUInt(n) => Some(*n),
                        _ => None,
                    })
                    .flatten()
            })
        else {
            return Ok(false);
        };
        let count = (raw_count & 63) as i64;
        let complement = (64 - count) & 63;
        let bb_id = self.emit_bb(mir_bb);
        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(result.clone()),
                kind,
            });
            result
        };
        let count = push_op(&mut self.graph, OpKind::ConstInt(count));
        let left = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "lshift".to_string(),
                lhs: value.clone(),
                rhs: count,
                result_ty: ValueType::Unsigned,
            },
        );
        let complement = push_op(&mut self.graph, OpKind::ConstInt(complement));
        let right = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "rshift".to_string(),
                lhs: value.clone(),
                rhs: complement,
                result_ty: ValueType::Unsigned,
            },
        );
        let result = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "or".to_string(),
                lhs: left,
                rhs: right,
                result_ty: ValueType::Unsigned,
            },
        );
        self.local_var[dest_local] = Some(LocalValue::One(result));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Returns `Ok(false)` when the call is not `checked_neg` /
    /// `checked_abs` (or the destination's `Option` decl cannot be
    /// resolved) so the generic `Call` lowering proceeds.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_checked_neg(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core" || module.as_str() != "num" || impl_seg.as_str() != "<Impl>" {
            return Ok(false);
        }
        // `checked_abs` overflows at the same `i64::MIN` as `checked_neg`
        // (`rint.py rtype_abs_ovf` / `ll_int_abs_ovf`).  The Some payload
        // is wrapping `abs`; the None arm never reads it.
        let unop = match leaf.as_str() {
            "checked_neg" => "neg",
            "checked_abs" => "abs",
            _ => return Ok(false),
        };
        let [arg] = args else {
            return Ok(false);
        };
        // The decomposition compares against `i64::MIN`; restrict it to
        // word-sized signed operands so a narrower `checked_neg` (which
        // overflows at its own `MIN`) is not miscompiled.  The operand
        // is the receiver — `checked_neg(self)` — so read `inputs[0]`.
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        if !matches!(self.tyref_literal_int_atom(src), Some("I64" | "Isize")) {
            return Ok(false);
        }
        // Resolve the destination `Option` decl so the FieldWrite owner
        // matches what `resolve_aggregate_adt` would record for a real
        // `Some(..)` construction site of the same type.
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        // Same owner-path / ctor-leaf split `resolve_aggregate_adt`
        // performs, so the ctor target and FieldWrite owner carry the
        // spellings the rest of the aggregate machinery expects.
        // Route the runtime-discriminant tagged-pair ctor to the SAME
        // per-instantiation enum root a static `Some(..)`/`Ok(..)` mints,
        // by suffixing the bare template name with the `<X>` the
        // destination `Option<X>`/`Result<X, E>` local carries.
        // Fail-closed: no splitting generic yields "" → bare owner.
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let arg = arg.clone();
        let bb_id = self.emit_bb(mir_bb);

        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind,
            });
            res
        };
        let payload = push_op(
            &mut self.graph,
            OpKind::UnaryOp {
                op: unop.to_string(),
                operand: arg.clone(),
                result_ty: ValueType::Int,
            },
        );
        let min = push_op(&mut self.graph, OpKind::ConstInt(i64::MIN));
        let disc = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "ne".to_string(),
                lhs: arg,
                rhs: min,
                result_ty: ValueType::Int,
            },
        );
        // Success (`arg != MIN`) is the `Some` variant (discriminant 1).
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 1).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            payload,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// Lower the runtime-fallible `i64::try_from(&RBigInt)`
    /// (a trait shell in the LLBC)
    /// into its decomposed runtime-discriminant `Result` shape: a
    /// `__discriminant` tag driven by `rbigint.fits_int()` and a
    /// `__pos_0` payload holding the narrowed value.  Callers narrow a
    /// list/str/bytes index or a byte value; overflow raises a Python
    /// `IndexError`/`ValueError`, so the fallibility must survive the
    /// lowering — a panic-residual swap would be a bit-exact regression.
    ///
    /// The arg type — not the module path — is the robust discriminator
    /// (same philosophy as the BigInt binary-operator retarget), so this
    /// guards on the exact `RBigInt` ADT identity rather than hard-matching a
    /// dependency path spelling.
    ///
    /// # Eager-payload panic hazard
    /// [`Self::emit_tagged_pair_aggregate`] writes `__pos_0 = payload`
    /// UNCONDITIONALLY in the call's block, before the consumer's
    /// discriminant switch runs in a successor block.  A
    /// [`jit_bigint_to_i64_value`] payload PANICS on overflow, so on a
    /// non-fitting index (`lst[2**100]`) it would panic in the
    /// walker/blackhole graph BEFORE the switch routes to the `Err` arm —
    /// panic instead of `IndexError`.  `__pos_0` is read only on the
    /// fits (`Ok`) path, so its value is dead on the `Err` path; the
    /// payload op only needs to be TOTAL, not correct-on-overflow.  Hence
    /// the non-trapping [`jit_bigint_to_i64_value_or_zero`] variant, which
    /// returns 0 out of range.
    ///
    /// `Ok=0`/`Err=1`: `disc = (fits == 0)` is 0 when it fits (`Ok` tag)
    /// and 1 otherwise (`Err` tag).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_bigint_i64_try_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [.., impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if impl_seg.as_str() != "<Impl>" || leaf.as_str() != "try_from" {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        // Guard on the ARG type (the opaque BigInt ADT), not the module
        // path — the robust discriminator, so a same-named `try_from`
        // impl on any other type is never redirected.  The narrowed
        // operand is the receiver — `try_from(&BigInt)` — so read
        // `inputs[0]`.
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        if !tyref_is_rbigint(src, self.llbc) {
            return Ok(false);
        }
        // The synthesized fits test and payload are specifically the signed
        // machine-word conversion.  Other BigInt TryFrom impls (notably
        // `u32::try_from(&value)` in formatting.rs::char_arg) have different
        // bounds and must keep the ordinary residual call.  Read Result's
        // success payload from `generics.types[0]`, then apply the same
        // literal-width atom gate as the other integer conversion lowerings.
        // RPython keeps signed `toint`/`fits_int` distinct from unsigned
        // `touint` (`rpython/rlib/rbigint.py, 515-518`).
        let Some(success_ty) = self.tyref_adt_type_arg(dest_ty, 0) else {
            return Ok(false);
        };
        if !matches!(
            self.tyref_literal_int_atom(&success_ty),
            Some("I64" | "Isize")
        ) {
            return Ok(false);
        }
        self.emit_fallible_rbigint_to_i64(mir_bb, arg.clone(), dest_local, dest_ty, target)
    }

    /// Lower the exact local `RBigInt::toint() -> Result<i64,
    /// RBigIntError>` method to the same decomposed Result shape as
    /// [`Self::try_lower_bigint_i64_try_from`].
    ///
    /// RPython's `rbigint.toint` is an elidable, overflow-raising scalar
    /// operation (`rpython/rlib/rbigint.py`). Rust carries that
    /// exceptional edge as `Result`; a dependent LLBC cannot translate into
    /// the pyre-object method body and otherwise leaves a classdef-less
    /// residual Result. Its subsequent discriminant/`Ok.__pos_0` reads then
    /// have no enum class to narrow. Preserve the source control flow by
    /// materializing the runtime Result tag and payload, using the shared
    /// non-trapping payload rule below.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_rbigint_toint(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if segments.last().map(String::as_str) != Some("toint") {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        if !fd
            .signature
            .inputs
            .first()
            .is_some_and(|src| tyref_is_rbigint(src, self.llbc))
        {
            return Ok(false);
        }
        let Some(success_ty) = self.tyref_adt_type_arg(dest_ty, 0) else {
            return Ok(false);
        };
        if !matches!(
            self.tyref_literal_int_atom(&success_ty),
            Some("I64" | "Isize")
        ) {
            return Ok(false);
        }
        self.emit_fallible_rbigint_to_i64(mir_bb, arg.clone(), dest_local, dest_ty, target)
    }

    /// Emit the translated Result for a fallible signed-word RBigInt
    /// conversion. Shared by Rust's direct `RBigInt::toint` spelling and the
    /// `i64::try_from(&RBigInt)` trait spelling so their tags, class identity,
    /// eager-payload safety, and residual ABI cannot drift apart.
    pub(in crate::front::mir) fn emit_fallible_rbigint_to_i64(
        &mut self,
        mir_bb: usize,
        arg: Variable,
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        // Resolve the destination `Result` decl so the FieldWrite owner
        // matches what `resolve_aggregate_adt` records for a real `Ok(..)`
        // construction site of the same type.
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        // Route the runtime-discriminant tagged-pair ctor to the SAME
        // per-instantiation enum root a static `Ok(..)`/`Err(..)` mints,
        // by suffixing the bare template name with the `<X>` the
        // destination `Result<X, E>` local carries.
        // Fail-closed: no splitting generic yields "" → bare owner.
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let bb_id = self.emit_bb(mir_bb);

        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind,
            });
            res
        };
        // Non-trapping payload: read only on the fits (`Ok`) path, dead on
        // the `Err` path, but eagerly evaluated in this block — must not
        // panic on overflow (see the eager-payload hazard above).
        let payload = push_op(
            &mut self.graph,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        crate::runtime_names::crates::OBJECT.to_string(),
                        "longobject".to_string(),
                        "jit_bigint_to_i64_value_or_zero".to_string(),
                    ],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![arg.clone()]),
                result_ty: ValueType::Int,
            },
        );
        let fits = push_op(
            &mut self.graph,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        crate::runtime_names::crates::OBJECT.to_string(),
                        "longobject".to_string(),
                        "jit_bigint_to_i64_fits".to_string(),
                    ],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![arg]),
                result_ty: ValueType::Int,
            },
        );
        let zero = push_op(&mut self.graph, OpKind::ConstInt(0));
        // `disc = (fits == 0)`: 0 when it fits (`Ok` tag), 1 otherwise
        // (`Err` tag) — matching the `Ok=0`/`Err=1` convention.
        let disc = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "eq".to_string(),
                lhs: fits,
                rhs: zero,
                result_ty: ValueType::Int,
            },
        );
        // Success is the `Ok` variant (discriminant 0).
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            payload,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// Lower `Wtf8::as_str() -> Result<&str, Utf8Error>` to the same
    /// runtime-tagged aggregate a translated RPython validity test followed
    /// by the string value would produce.
    ///
    /// PyPy's unicode value can carry lone surrogates, while the Rust host
    /// API exposes that distinction through `Result`.  The validity decision
    /// therefore remains a red residual scalar (`wtf8_key_is_utf8`), and only
    /// the `Ok` payload is the identity string value.  Folding the whole call
    /// to the receiver would incorrectly delete the lone-surrogate arm.
    ///
    /// This lowering is restricted to the current interpreter shape, whose
    /// `Wtf8::as_str` callers inspect only the tag and the `Ok` string.  A
    /// caller that consumes `Utf8Error` would require a separate
    /// foreign-boundary port of `valid_up_to` / `error_len`; this adapter must
    /// not be generalized to such a call site by pretending that payload has
    /// an RPython layout.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_wtf8_as_str(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if segments != ["Wtf8", "as_str"] {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let Some(ok_ty) = self.tyref_adt_type_arg(dest_ty, 0) else {
            return Ok(false);
        };
        if !tyref_strips_to_str(&ok_ty, self.llbc) {
            return Ok(false);
        }
        let Some(err_ty) = self.tyref_adt_type_arg(dest_ty, 1) else {
            return Ok(false);
        };
        let err_is_utf8 = self
            .tyref_adt_def_id(&err_ty)
            .and_then(|id| self.llbc.type_by_id(id))
            .is_some_and(|td| td.item_meta.name_path().ends_with("::Utf8Error"));
        if !err_is_utf8 {
            return Ok(false);
        }
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let bb_id = self.emit_bb(mir_bb);
        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind,
            });
            res
        };
        let valid = push_op(
            &mut self.graph,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "pyre_object".to_string(),
                        "dictmultiobject".to_string(),
                        "wtf8_key_is_utf8".to_string(),
                    ],
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![arg.clone()]),
                result_ty: ValueType::Bool,
            },
        );
        // Result convention is Ok=0 / Err=1.  Comparing the validity bool
        // with False therefore has exactly the required integer tag while
        // preserving the runtime arm.  This is the same flowspace shape as
        // the ordinary Rust `!bool` lowering above: RPython has no logical
        // `not` operation, because `UNARY_NOT` becomes a truth test plus a
        // branch in `flowcontext.py`; a scalar negation is `eq(b, False)`.
        let false_var = push_op(&mut self.graph, OpKind::ConstBool(false));
        let disc = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "eq".to_string(),
                lhs: valid,
                rhs: false_var,
                result_ty: ValueType::Int,
            },
        );
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate_typed(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            arg.clone(),
            ValueType::Str,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// `Layout::new::<T>()` → the virtualized `{size, align}` aggregate
    /// [`crate::front::from_size_align::build_layout_aggregate`] already
    /// emits for `from_size_align`.  `new` is `size_of::<T>()` plus
    /// `align_of::<T>()`; both constants fold through
    /// [`Self::size_align_const_from_tyexpr`], the same helper the inline
    /// `size_of`/`align_of` call fold uses.  Fail-closed: a missing type
    /// argument or unresolved layout leaves the residual call.
    pub(in crate::front::mir) fn try_lower_layout_new(
        &mut self,
        mir_bb: usize,
        reg: &RegularCall,
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let segments: Vec<String> = fd
            .item_meta
            .name_path()
            .split("::")
            .map(String::from)
            .collect();
        if !crate::front::from_size_align::is_layout_new(&segments) {
            return Ok(false);
        }
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let layout_owner = td.item_meta.name_path();
        if !crate::front::from_size_align::is_layout_adt_owner(&layout_owner) {
            return Ok(false);
        }
        let Some(ty) = reg
            .generics
            .get("types")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| a.first())
        else {
            return Ok(false);
        };
        let Some(size) = self.size_align_const_from_tyexpr(false, ty) else {
            return Ok(false);
        };
        let Some(align) = self.size_align_const_from_tyexpr(true, ty) else {
            return Ok(false);
        };
        let bb_id = self.block_id[mir_bb];
        let push_const = |graph: &mut FunctionGraph, value: i64| {
            let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::ConstInt(value),
            });
            res
        };
        let size_var = push_const(&mut self.graph, size);
        let align_var = push_const(&mut self.graph, align);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        crate::front::from_size_align::build_layout_aggregate(
            &mut self.graph,
            bb_id,
            &layout_owner,
            res.clone(),
            size_var,
            align_var,
        );
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Lower Rust's checked integer-to-`usize` shell to the same primitive
    /// comparisons and cast operations RPython emits for integer reprs.
    ///
    /// The infallible `usize::try_from(<u8|u16|u32>)`
    /// (`core::convert::num::ptr_try_from_impls::<Impl>::try_from`,
    /// Opaque in the LLBC like every core fn) to its decomposed
    /// always-`Ok` shape: `__discriminant = 0` (`Result`'s `Ok` tag)
    /// and `__pos_0 = arg` (the widening is an identity on the i64
    /// carrier).  Upstream `rarithmetic.py widen` performs
    /// the same smaller-than-word unsigned → Signed widening as a
    /// no-op; pyre spells it `usize::try_from(x).expect(..)`
    /// (`pyopcode.rs` `u32_as_usize` / `raise_kind_as_usize`), whose
    /// `Err` arm is statically dead for the `U8`/`U16`/`U32` sources
    /// filtered on below — narrower than `usize` on every target pyre
    /// builds for, wasm32 included.  A same-width signed source (`isize`, or
    /// `i64` in a 64-bit LLBC corpus) is fallible only for negative values.
    /// Preserve Rust's observable `Result` branch as a runtime tag and cast
    /// its success payload with RPython's `cast_int_to_uint` spelling
    /// (`rint.py __extend__`).  The cast is total and may run eagerly on the Err
    /// arm because that payload is dead there.  Fixed-width `i64` is admitted
    /// only when Charon's target metadata proves an eight-byte pointer; wider
    /// signed sources on a narrower target remain fail-closed.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    /// `rbigint.int_divmod` / `rbigint.divmod` return `(div, mod)`.  Both are
    /// `@jit.elidable`; RPython lowers the call to one residual returning the
    /// `tuple2` GcStruct and reads its two items.  `RBigIntPair` is that
    /// struct, so the call becomes the pair residual plus two pure item reads
    /// packed back into the source-level `(RBigInt, RBigInt)`.  The Rust
    /// `Result` around it is always `Ok` here -- the residual raises instead
    /// -- so the destination binds to the tuple the way
    /// [`Self::try_lower_usize_try_from`] binds an always-`Ok` payload.
    pub(in crate::front::mir) fn try_lower_rbigint_divmod_pair(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        second_arg_ty: Option<&TyRef>,
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [receiver, divisor] = args else {
            return Ok(false);
        };
        if !segments.iter().any(|s| s == "rbigint")
            || !first_arg_ty.is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
        {
            return Ok(false);
        }
        let Some(second_arg_ty) = second_arg_ty else {
            return Ok(false);
        };
        let int_divisor = if self.tyref_literal_int_atom(second_arg_ty) == Some("I64") {
            true
        } else if tyref_is_rbigint(second_arg_ty, self.llbc) {
            false
        } else {
            return Ok(false);
        };
        let Some(residual) = segments.last().and_then(|leaf| {
            crate::front::rbigint_call::divmod_pair_residual_for_method(leaf, int_divisor)
        }) else {
            return Ok(false);
        };
        if !crate::front::result_exc::tyref_is_result(dest_ty, self.llbc) {
            return Ok(false);
        }
        if divmod_result_observed_as_result(self.body, self.llbc, dest_local) {
            return Ok(false);
        }
        let Some(success_ty) = self.tyref_adt_type_arg(dest_ty, 0) else {
            return Ok(false);
        };
        let suffix = tyref_tuple_suffix(&success_ty, self.llbc);
        if suffix.is_empty() {
            return Ok(false);
        }
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let tuple_owner = format!("Tuple{suffix}");
        let bb_id = self.emit_bb(mir_bb);
        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(result.clone()),
                kind,
            });
            result
        };
        let pair = push_op(
            &mut self.graph,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: crate::model::call_args(vec![receiver.clone(), divisor.clone()]),
                result_ty: ValueType::Ref(Some("RBigIntPair".to_string())),
            },
        );
        let pair = push_op(
            &mut self.graph,
            crate::model::cast_instance_call("RBigIntPair", pair),
        );
        let mut items = Vec::with_capacity(2);
        for field in ["item0", "item1"] {
            let item = push_op(
                &mut self.graph,
                OpKind::FieldRead {
                    base: pair.clone(),
                    field: FieldDescriptor::new(field.to_string(), Some("RBigIntPair".to_string())),
                    ty: ValueType::Ref(None),
                    pure: true,
                },
            );
            items.push(push_op(
                &mut self.graph,
                crate::model::cast_instance_call("RBigInt", item),
            ));
        }
        let ctor = AggregateCtor {
            target: CallTarget::synthetic_transparent_ctor(tuple_owner.clone()),
            result_owner: tuple_owner.clone(),
            owner_id: None,
            owner_path: Vec::new(),
            ctor_name: tuple_owner,
            fields: (0..2)
                .map(|i| AggregateField {
                    name: format!("__pos_{i}"),
                    void: false,
                    narrow_root: None,
                    window: false,
                })
                .collect(),
        };
        let payload = emit_aggregate_ctor(&mut self.graph, bb_id, &ctor, &items);
        if self.multi_assigned_locals.contains(&dest_local) {
            let disc = push_op(&mut self.graph, OpKind::ConstInt(0));
            let payload_owner =
                Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
            self.emit_tagged_pair_aggregate(
                mir_bb,
                &owner,
                &payload_owner,
                disc,
                payload,
                dest_local,
                target,
            )?;
            return Ok(true);
        }
        self.const_discriminant_locals.insert(dest_local, 0);
        self.local_var[dest_local] = Some(LocalValue::One(payload));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    pub(in crate::front::mir) fn try_lower_usize_try_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "ptr_try_from_impls"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "try_from"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        let src_is_small_uint = matches!(
            self.tyref_literal_uint_atom(src),
            Some("U8" | "U16" | "U32")
        );
        let src_is_signed_word = match self.tyref_literal_int_atom(src) {
            Some("Isize") => true,
            Some("I64") => self.llbc.target_pointer_size() == Some(8),
            _ => false,
        };
        if !src_is_small_uint && !src_is_signed_word {
            return Ok(false);
        }
        let Some(success_ty) = self.tyref_adt_type_arg(dest_ty, 0) else {
            return Ok(false);
        };
        if self.tyref_literal_uint_atom(&success_ty) != Some("Usize") {
            return Ok(false);
        }
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        // Route the runtime-discriminant tagged-pair ctor to the SAME
        // per-instantiation enum root a static `Some(..)`/`Ok(..)` mints,
        // by suffixing the bare template name with the `<X>` the
        // destination `Option<X>`/`Result<X, E>` local carries.
        // Fail-closed: no splitting generic yields "" → bare owner.
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(dest_ty, self.llbc)
        );
        let arg = arg.clone();
        if src_is_signed_word {
            let bb_id = self.emit_bb(mir_bb);
            let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
                let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(result.clone()),
                    kind,
                });
                result
            };
            let zero = push_op(&mut self.graph, OpKind::ConstInt(0));
            // Result convention: Ok=0, Err=1.  For equal-width signed →
            // unsigned conversion the sign test is the complete overflow
            // predicate.
            let disc = push_op(
                &mut self.graph,
                OpKind::BinOp {
                    op: "lt".to_string(),
                    lhs: arg.clone(),
                    rhs: zero,
                    result_ty: ValueType::Int,
                },
            );
            let payload = push_op(
                &mut self.graph,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Unsigned,
                },
            );
            let payload_owner =
                Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
            self.emit_tagged_pair_aggregate_typed(
                mir_bb,
                &owner,
                &payload_owner,
                disc,
                payload,
                ValueType::Unsigned,
                dest_local,
                target,
            )?;
            return Ok(true);
        }
        if self.multi_assigned_locals.contains(&dest_local) {
            // A re-bindable local may later carry a runtime `Result`,
            // so the constant tag can't be recorded and consumers
            // can't be folded — materialize the aggregate so field
            // reads on the local stay type-consistent.
            let bb_id = self.emit_bb(mir_bb);
            let disc = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(disc.clone()),
                kind: OpKind::ConstInt(0),
            });
            // Always-`Ok`: the payload belongs to the `Ok` variant (tag 0).
            let payload_owner =
                Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
            self.emit_tagged_pair_aggregate(
                mir_bb,
                &owner,
                &payload_owner,
                disc,
                arg,
                dest_local,
                target,
            )?;
            return Ok(true);
        }
        // Identity widening: bind the destination local directly to
        // the operand — no `Result` object is materialized at all,
        // exactly as upstream `widen` leaves no op in the graph.
        // The discriminant switch always sits in a successor block
        // (a MIR call terminates its own block), so record the
        // constant tag per-local: `Rvalue::Discriminant` folds to
        // `ConstInt(0)` and `expect`/`unwrap` aliases the payload
        // (`expect_on_const_ok`) without touching the variable.
        self.const_discriminant_locals.insert(dest_local, 0);
        self.local_var[dest_local] = Some(LocalValue::One(arg));
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Lower a fallible signed `try_from` core shell to the checked
    /// narrowing shape PyPy writes directly in
    /// `ObjSpace.c_int_w` (`baseobjspace.py`): the value succeeds
    /// exactly when `DST_MIN <= value <= DST_MAX`. Rust carries the two
    /// branches in a `Result<T, TryFromIntError>`, so preserve that source
    /// CFG by materializing `Ok=0` / `Err=1` and leaving the caller's match
    /// arms untouched.
    ///
    /// `i32::try_from(i64)` aliases the i64 carrier as the payload: narrow
    /// integer widths do not have a distinct JIT register bank, and it is
    /// read only on the proven-in-range Ok arm. `i64::try_from(i128)`
    /// (`w_range_iter` one-past bound) uses `longlongmask`
    /// (`rbuiltin.py` `rarith_longlongmask`) so the Ok payload is the
    /// 64-bit signed word. Other source/destination pairs stay residual
    /// until their own bounds and signedness rules are ported.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_i64_to_i32_try_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "try_from"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        // `convert::num::<Impl>::try_from` is a generic FunDecl; the
        // instantiation's source width is the call-site operand
        // (`w_range_iter` `i64::try_from(i128)`), not the TypeVar on
        // the template signature.
        let src_atom = match kind {
            CallKind::Fun(FunId::Regular { id }) => self
                .llbc
                .fn_by_id(*id)
                .and_then(|fd| fd.signature.inputs.first())
                .and_then(|src| self.tyref_literal_int_atom(src)),
            _ => None,
        }
        .or_else(|| first_arg_ty.and_then(|ty| self.tyref_literal_int_atom(ty)));
        let Some(src_atom) = src_atom else {
            return Ok(false);
        };
        let success_ty = self.tyref_adt_type_arg(dest_ty, 0).or_else(|| {
            self.body
                .locals
                .locals
                .get(dest_local)
                .and_then(|decl| self.tyref_adt_type_arg(&decl.ty, 0))
        });
        let Some(success_ty) = success_ty else {
            return Ok(false);
        };
        let Some(dst_atom) = self.tyref_literal_int_atom(&success_ty) else {
            return Ok(false);
        };
        let i128_to_i64 = src_atom == "I128" && dst_atom == "I64";
        let i64_to_i32 = matches!(src_atom, "I64" | "Isize") && dst_atom == "I32";
        if !i128_to_i64 && !i64_to_i32 {
            return Ok(false);
        }
        let result_ty = if self.tyref_adt_def_id(dest_ty).is_some() {
            dest_ty
        } else {
            self.body
                .locals
                .locals
                .get(dest_local)
                .map(|decl| &decl.ty)
                .unwrap_or(dest_ty)
        };
        let Some(def_id) = self.tyref_adt_def_id(result_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(result_ty, self.llbc)
        );
        let bb_id = self.emit_bb(mir_bb);
        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let result = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(result.clone()),
                kind,
            });
            result
        };
        // Bounds live in the signed word bank (`history.py` `getkind`
        // refuses `SignedLongLongLong`). `i64::MIN`/`MAX` are that bank;
        // a `ConstInt128` bound would not project.
        let min = push_op(
            &mut self.graph,
            OpKind::ConstInt(if i128_to_i64 {
                i64::MIN
            } else {
                i32::MIN as i64
            }),
        );
        let below = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "lt".to_string(),
                lhs: arg.clone(),
                rhs: min,
                result_ty: ValueType::Int,
            },
        );
        let max = push_op(
            &mut self.graph,
            OpKind::ConstInt(if i128_to_i64 {
                i64::MAX
            } else {
                i32::MAX as i64
            }),
        );
        let above = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "gt".to_string(),
                lhs: arg.clone(),
                rhs: max,
                result_ty: ValueType::Int,
            },
        );
        // `c_int_w` joins the two bound tests with `or`
        // (`baseobjspace.py`), so join them with the registered `or_`
        // operator (`operation.py`) rather than with arithmetic: the
        // result is the Result discriminant, 0 in range (Ok), 1 out of range
        // (Err).
        let disc = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "or".to_string(),
                lhs: below,
                rhs: above,
                result_ty: ValueType::Int,
            },
        );
        // `i64::try_from(i128)` is in range on the Ok arm, so
        // `longlongmask` is the 64-bit signed word (`rbuiltin.py`).
        let payload = if i128_to_i64 {
            push_op(
                &mut self.graph,
                rarithmetic_mask("longlongmask", arg.clone(), ValueType::Int),
            )
        } else {
            arg.clone()
        };
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            payload,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// `slice::from_ref(&x)` of a GC pointer is `newlist(x)`
    /// (`rlist.py` `rtype_newlist`). A one-element `&[PyObjectRef]` is
    /// the length-prefixed object array, the same `Unsize` of `[x; 1]`
    /// already rewrites.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_slice_from_ref(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if !is_slice_from_ref_path(segments) {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let dest_is_object_slice =
            matches!(
                tyref_object_pointer_slice_mutability(dest_ty, self.llbc),
                Some("Shared" | "Const")
            ) || self.body.locals.locals.get(dest_local).is_some_and(|decl| {
                matches!(
                    tyref_object_pointer_slice_mutability(&decl.ty, self.llbc),
                    Some("Shared" | "Const")
                )
            });
        let arg_is_object_ref =
            first_arg_ty.is_some_and(|ty| tyref_is_shared_ref_to_object_pointer(ty, self.llbc));
        if !dest_is_object_slice && !arg_is_object_ref {
            return Ok(false);
        }
        let bb_id = self.emit_bb(mir_bb);
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, GCREF, LowLevelType, Ptr, PtrTarget,
        };
        res.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(GCREF.clone())),
        }))));
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::NewList {
                args: vec![arg.clone()],
            },
        });
        self.local_var[dest_local] = Some(LocalValue::One(res));
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `Vec::try_reserve_exact(n) -> Result<(), TryReserveError>` is the
    /// fallible half of `ll_newlist_hint` (`rlist.py`). `newlist_hint`
    /// (`objectmodel.py`) is an ExtRegistry; `try_pyobject_vec_with_capacity`
    /// already takes that path. This generic body is inlined, so the std
    /// call has to become operations: an always-`Ok` Result shell whose
    /// `map_err(|_| reservation_failed())` arm stays the MemoryError edge
    /// when a later pass gives the discriminant a runtime value.
    pub(in crate::front::mir) fn try_lower_vec_try_reserve_exact(
        &mut self,
        mir_bb: usize,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        if rust_vec_std_leaf(segments) != Some("try_reserve_exact") {
            return Ok(false);
        }
        if args.len() != 2 {
            return Ok(false);
        }
        let success_ty = self.tyref_adt_type_arg(dest_ty, 0).or_else(|| {
            self.body
                .locals
                .locals
                .get(dest_local)
                .and_then(|decl| self.tyref_adt_type_arg(&decl.ty, 0))
        });
        let Some(success_ty) = success_ty else {
            return Ok(false);
        };
        if !is_unit_type(&success_ty, self.llbc) {
            return Ok(false);
        }
        let result_ty = if self.tyref_adt_def_id(dest_ty).is_some() {
            dest_ty
        } else {
            self.body
                .locals
                .locals
                .get(dest_local)
                .map(|decl| &decl.ty)
                .unwrap_or(dest_ty)
        };
        let Some(def_id) = self.tyref_adt_def_id(result_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let owner = format!(
            "{}{}",
            td.item_meta.name_path(),
            tyref_enum_instantiation_suffix(result_ty, self.llbc)
        );
        let bb_id = self.emit_bb(mir_bb);
        let disc = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::ConstInt(0),
        });
        let payload = self.emit_unit(bb_id);
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            payload,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// Lower an infallible numeric widening `<i64 as From<u32>>::from(x)` or
    /// `<usize as From<bool>>::from(flag)`
    /// (`core::convert::num::<Impl>::from`, Opaque in the LLBC) to an
    /// identity bind on the destination local.  `From` is implemented in
    /// core only for value-preserving conversions, and a
    /// smaller-than-word unsigned source widens into the word-sized
    /// carrier with no change of value — the same no-op `rarithmetic.py
    /// widen` performs — so no op is emitted, exactly as the always-`Ok`
    /// `usize::try_from` payload binds directly
    /// ([`Lowering::try_lower_usize_try_from`]).  `pyopcode.rs`
    /// `u32_as_i64` is `i64::from(x: u32)`.  Word-or-wider sources keep
    /// the `Call` form (no smaller-than-word identity to fold).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    pub(in crate::front::mir) fn try_lower_num_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "from"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        // Smaller-than-word unsigned sources widen as a value-preserving
        // identity in the word carrier (the same gate
        // `try_lower_usize_try_from` uses). Bool is the RPython BoolRepr
        // sibling: converting it to Unsigned emits `cast_bool_to_uint`
        // through the ordinary `r_uint` builtin (`rbool.py`).
        let src_is_bool = tyref_to_value_type_with(
            src,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        ) == ValueType::Bool;
        let src_is_small_uint = matches!(
            self.tyref_literal_uint_atom(src),
            Some("U8" | "U16" | "U32")
        );
        if !src_is_small_uint && !src_is_bool {
            return Ok(false);
        }
        // Destination must be a word-sized integer carrier.  A signed
        // word destination needs the same annotation re-type the cast
        // path applies (the `Rvalue::Cast` unsigned→signed arm): aliasing
        // a `uN` source straight into an `iN` carrier preserves the source
        // `r_uint` annotation and trips the SomeInteger signedness
        // `UnionError` (binaryop.py union) when the value later meets a
        // signed operand.  Route signed destinations through
        // `rarithmetic.intmask` (identity on the i64 carrier, re-types
        // Signed); alias unsigned destinations directly, as upstream
        // `widen` leaves no op.
        let dest_signed_word =
            matches!(self.tyref_literal_int_atom(dest_ty), Some("I64" | "Isize"));
        let dest_unsigned_word =
            matches!(self.tyref_literal_uint_atom(dest_ty), Some("U64" | "Usize"));
        if !dest_signed_word && !dest_unsigned_word {
            return Ok(false);
        }
        let arg = arg.clone();
        let bb_id = self.emit_bb(mir_bb);
        if src_is_bool && dest_unsigned_word {
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["rpython", "rlib", "rarithmetic", "r_uint"]
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Unsigned,
                },
            });
            self.local_var[dest_local] = Some(LocalValue::One(res));
        } else if dest_signed_word {
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["rpython", "rlib", "rarithmetic", "intmask"]
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(vec![arg]),
                    result_ty: ValueType::Int,
                },
            });
            self.local_var[dest_local] = Some(LocalValue::One(res));
        } else {
            // Identity widen: bind the destination directly to the operand —
            // no op materialized, exactly as upstream `widen` leaves none.
            self.local_var[dest_local] = Some(LocalValue::One(arg));
        }
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Resolve `Result::expect` / `Result::unwrap` on a receiver local
    /// recorded always-`Ok` (`const_discriminant_locals`, tag 0) to
    /// the receiver variable itself.  Such locals are bound directly
    /// to the `Ok` payload by [`Lowering::try_lower_usize_try_from`]
    /// (no `Result` object is materialized), so the unwrap is an
    /// alias, not an op.  The match these methods perform lives
    /// inside the Opaque core body, so without the intercept the call
    /// keeps its panic-message `&str` argument and the graph walls on
    /// the `__str_const` lowering even though the `Err` arm is
    /// statically dead.  Upstream has no such call: `ovfcheck`-free
    /// widening is a plain identity (`rarithmetic.py widen`),
    /// so the operand *is* the whole operation.  Returns `None` for
    /// any other callee or a receiver without the always-`Ok` record,
    /// keeping the generic `Call` form.
    pub(in crate::front::mir) fn expect_on_const_ok(
        &self,
        segments: &[String],
        args: &[Variable],
        arg_locals: &[Option<usize>],
    ) -> Option<Variable> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return None;
        };
        if first.as_str() != "core"
            || module.as_str() != "result"
            || impl_seg.as_str() != "<Impl>"
            || !matches!(leaf.as_str(), "expect" | "unwrap")
        {
            return None;
        }
        let recv = args.first()?;
        let recv_local = (*arg_locals.first()?)?;
        if *self.const_discriminant_locals.get(&recv_local)? != 0 {
            return None;
        }
        Some(recv.clone())
    }

    /// Resolve the reflexive blanket `Into::into`
    /// (`core::convert::<Impl>::into` where the argument's declared
    /// type equals the destination's) to its operand.  `impl<T>
    /// From<T> for T` makes `x.into()` an identity, but the blanket
    /// fn's body is Opaque in the LLBC so the `Call` form is
    /// permanently unliftable.  Upstream has no counterpart: an
    /// identity conversion never appears as an op in RPython graphs.
    /// Non-reflexive `into` calls (source ≠ destination type) keep
    /// the generic `Call` form.
    pub(in crate::front::mir) fn reflexive_into_alias(
        &self,
        reg: &RegularCall,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let spelled_blanket = matches!(
            segments,
            [first, .., module, impl_seg, leaf]
                if first == "core" && module == "convert" && impl_seg == "<Impl>" && leaf == "into"
        );
        if !spelled_blanket && !self.callee_is_blanket_into(reg) {
            return None;
        }
        self.identity_self_call_alias(args, first_arg_ty, dest_ty)
    }

    /// The callee is the blanket `impl<T, U: From<T>> Into<U> for T`
    /// method.  Read off the declaration, not the rendered call path: a
    /// monomorphized instance's `Self` is a concrete type, so its path
    /// renders under that type's name.
    pub(in crate::front::mir) fn callee_is_blanket_into(&self, reg: &RegularCall) -> bool {
        self.callee_template_is(reg, "core::convert::<Impl>::into")
    }

    /// The callee's declaration is named `path`, read off its template name
    /// so a monomorphized instance matches like the generic item.
    pub(in crate::front::mir) fn callee_template_is(&self, reg: &RegularCall, path: &str) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == path)
    }

    /// Resolve the reflexive blanket `IntoIterator::into_iter`
    /// (`core::iter::traits::collect::<Impl>::into_iter`) to its
    /// operand.  `impl<I: Iterator> IntoIterator for I` sets
    /// `IntoIter = Self` and `into_iter(self) -> Self { self }`, so the
    /// call is an identity whenever the receiver type equals the
    /// destination type.  The blanket fn's body is Opaque in the LLBC,
    /// leaving the `Call` form permanently unliftable; an
    /// already-an-iterator `into_iter` never appears as an op in RPython
    /// graphs.  Concrete `IntoIterator` impls (`&[T]` → `slice::Iter`,
    /// `Vec` → `vec::IntoIter`) have receiver ≠ destination and keep the
    /// generic `Call` form.
    pub(in crate::front::mir) fn reflexive_into_iter_alias(
        &self,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        if !fmt_path_ends_with(
            segments,
            &["iter", "traits", "collect", "<Impl>", "into_iter"],
        ) {
            return None;
        }
        self.identity_self_call_alias(args, first_arg_ty, dest_ty)
    }

    /// Shared resolution for an Opaque blanket-impl call whose receiver
    /// type equals its result type: the call is a no-op, so it resolves
    /// to its sole operand.  Used by the reflexive `Into::into` and
    /// `IntoIterator::into_iter` blanket-impl aliases above.
    pub(in crate::front::mir) fn identity_self_call_alias(
        &self,
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        let src = self.tyref_body(first_arg_ty?)?;
        let dst = self.tyref_body(dest_ty)?;
        (src == dst).then(|| arg.clone())
    }

    /// Resolve the trait-spelled `Into::into` (`["Into", "into"]` — a
    /// generic-parameter receiver, so Charon cannot select the impl at
    /// the call site) to its operand when the destination is a
    /// string-family value (`alloc::string::String` or the `Wtf8` /
    /// `Wtf8Buf` wrappers).  `impl Into<String>` / `impl Into<Wtf8Buf>`
    /// message parameters (the `PyError` constructor family) reach
    /// `msg.into()` inside the generic body; the annotation model maps
    /// `String`, `str`, `Wtf8` and `Wtf8Buf` to the same string value
    /// (`project_struct_field_type` — `s_unicode0`), matching upstream's
    /// single string type (`rstr.py`), so the conversion is an identity
    /// at the annotation level.  Other destination types keep the
    /// generic `Call` form.
    ///
    /// A monomorphized caller names the selected impl instead: the
    /// blanket `core::convert::<Impl>::into` instance, which is the same
    /// conversion.
    pub(in crate::front::mir) fn trait_into_string_alias(
        &self,
        reg: &RegularCall,
        segments: &[String],
        args: &[Variable],
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let spelled_trait =
            matches!(segments, [trait_seg, leaf] if trait_seg == "Into" && leaf == "into");
        if !spelled_trait && !self.callee_is_blanket_into(reg) {
            return None;
        }
        let [arg] = args else {
            return None;
        };
        tyref_is_string_family(dest_ty, self.llbc).then(|| arg.clone())
    }

    /// Resolve the WTF-8 string wrappers `Wtf8::new(&str) -> &Wtf8` and
    /// `Wtf8Buf::from_string(String) -> Wtf8Buf`, and the reverse reinterpret
    /// `wtf8_key_as_str_unchecked(&Wtf8) -> &str`, to their sole string
    /// argument.  Rust's `&str` / `String` / `Wtf8` / `Wtf8Buf` all map
    /// to the single immutable rpy_string value (`project_struct_field_type`
    /// — `s_unicode0`, matching upstream's one string type in `rstr.py`),
    /// so the wrap is an identity at the annotation level; the boxing the
    /// callers want (`box_str_constant`) happens downstream on the bound
    /// value.  The `Wtf8` bodies are Opaque in the LLBC (external
    /// `rustpython_wtf8` crate), leaving the generic `Call` permanently
    /// unliftable; `wtf8_key_as_str_unchecked` is a pyre free fn whose
    /// `from_utf8_unchecked` body is likewise left unlifted by folding the
    /// call away.  It is matched by leaf name because its qualified path
    /// (`pyre_object::dictmultiobject::wtf8_key_as_str_unchecked`) differs
    /// between the defining-crate LLBC and a cross-crate caller.
    pub(in crate::front::mir) fn wtf8_string_identity_alias(
        &self,
        segments: &[String],
        args: &[Variable],
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        let is_wrap = wtf8_string_identity_segments(segments);
        let is_str_reinterpret =
            matches!(segments, [.., leaf] if leaf == "wtf8_key_as_str_unchecked");
        (is_wrap || is_str_reinterpret).then(|| arg.clone())
    }
}
