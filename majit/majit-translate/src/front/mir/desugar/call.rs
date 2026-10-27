//! Hand lowering of individual call sites (`lower_call` … `impl_method_owner`).

use super::*;

impl<'a> Lowering<'a> {
    pub(in crate::front::mir) fn lower_call(
        &mut self,
        mir_bb: usize,
        call: CallPayload,
        target: usize,
        on_unwind: usize,
    ) -> Result<(), LowerError> {
        if let Some(direct) = self.fn_item_call_once_direct_call(&call) {
            return self.lower_call(mir_bb, direct, target, on_unwind);
        }
        let bb_id = self.emit_bb(mir_bb);
        let callee_pin_gcref_arg = match &call.func {
            CallFunc::Regular(reg) => regular_call_name_path(reg, self.llbc)
                .as_deref()
                .is_some_and(gc_root_pin_path),
            _ => false,
        };
        let callee_identity_hash = match &call.func {
            CallFunc::Regular(reg) => regular_call_name_path(reg, self.llbc)
                .as_deref()
                .is_some_and(gc_identity_hash_path),
            _ => false,
        };

        // Destination must be a plain `Local(i)` — projection-typed
        // destinations are not produced for monomorphized calls in any
        // body we extract today; fail-loud if Charon surfaces one.
        let dest_local = match call.dest.kind {
            PlaceKind::Local(i) => i as usize,
            _ => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call with projection-destination not supported"
                )));
            }
        };
        // `core::panicking::*` / `std::panicking::*` (`assert_failed`,
        // `panic`, `panic_fmt`, …) is a Rust abort. Upstream that is
        // `ll_assert` / `fatalerror`: the same implicit AssertionError
        // raise `TermKind::Abort` already closes a block with. Do not
        // register a stub body for the panicking helper.
        if let CallFunc::Regular(reg) = &call.func
            && regular_call_is_panicking_abort(reg, self.llbc)
        {
            self.graph.set_raise_implicit(bb_id, "AssertionError");
            // `removeassert.py` `remove_asserts` / `kill_assertion_link`:
            // the abort is an implicit AssertionError (`ll_assert` with a
            // constant message). Melt the format operands rustc evaluated
            // before `panic_fmt` / `assert_failed`.
            melt_abort_message_chain(&mut self.graph, bb_id);
            return Ok(());
        }
        // `null_mut::<T>()` / `null::<T>()` of a Raw pointee is
        // `lltype.nullptr(T)` (`null_raw`). The generic call-result
        // `__cast_instance_intrinsic` tail below would recast a GC
        // `null_mut` onto that root (`rtype_cast_instance_intrinsic`).
        if self.try_lower_raw_adt_ptr_null(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        // A bracket [`RootBracketPlan`] erased: the opener, the pin, the
        // `base()` and the read-back all leave the jitcode here, before any
        // operand is resolved -- the guard and its borrow bind no Variable,
        // so resolving the receiver would fail.
        if self.lower_erased_root_bracket_call(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        if self.lower_erased_owner_root_call(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        if self.lower_owner_root_guard_call(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        self.refuse_residual_owner_root_guard(mir_bb, &call)?;
        if self.lower_global_initializer_call(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        if self.lower_item_addr_call(mir_bb, &call, dest_local, target)? {
            return Ok(());
        }
        // Whole-list `Vec<PyObjectRef>` views (`&vec`, `&mut vec`, `&vec[..]`,
        // `&mut vec[..]`) are one `SomeList` word. Take that arm before
        // `is_pair_call`: an object-pointer `&mut [T]` dest would otherwise
        // become a `{ptr, len}` pair, and a later one-word use
        // (`shadow_stack_copy_range`, a reborrow) declines with
        // `pair-slice local read as one word`. Slice methods on the mut
        // view (`rotate_right`) still take `is_pair_call`, which splits
        // the header.
        if let CallFunc::Regular(reg) = &call.func
            && self.try_lower_object_vec_whole_list_view(mir_bb, &call, dest_local, target, reg)?
        {
            return Ok(());
        }
        if self.is_pair_call(&call, dest_local) {
            return self.lower_pair_call(mir_bb, call, dest_local, target);
        }
        let slice_next = self.slice_iter_next_value(&call);
        // Consumed at the block close below, after the producer call op.
        self.owner_root_reroot = self.owner_root_reroot_call(mir_bb, &call, dest_local)?;
        let rerooted = self.owner_root_reroot.is_some();

        // The call result kind is the MIR-declared type of the
        // destination place. RPython `call.py` `getcalldescr` reads `FUNC.RESULT`
        // off the callee funcptr; the destination local's declared type
        // is that same value at the call site, so deriving it here keeps
        // `getcalldescr`'s `RESULT == FUNC.RESULT` check (`call.py`)
        // satisfied for non-`Int` returns such as
        // `new_for_call_with_closure_and_globals_obj` (Ref).
        //
        // A `-> ()` callee's graph reports a void result kind (its
        // `Return` lowers via `set_return(None)`, see [`is_unit_type`]),
        // and [`tyref_to_value_type`] paints unit `Void` so the call
        // site's `FUNC.RESULT` matches (`call.rs`,
        // `ExecutionContext.force_all_frames`).
        //
        // A raw pointer to a non-byte scalar is the address word.
        // `history.py` `getkind` banks every raw `Ptr` as `int`, and
        // [`pygraph_initial_block`] already records that parameter as
        // `Int`. The call that produced the pointer uses the same bank.
        // A byte pointer stays the `Ref` erasure of
        // [`tyref_to_value_type_with`]: pyre erases a GC reference to
        // `*mut u8`. A reference stays an address. Naming its pointee
        // here would retype a call-returned `&T` to the scalar, and
        // `*dest` is `rewrite_op_raw_load`.
        //
        // [`is_typed_array_base_adapter`] spells that same `*mut i64` /
        // `*mut f64`, then [`is_items_block_base_ptr_add`] collapses the
        // body onto the GC block header. `getkind` of a GC pointer is
        // `ref`, so the call keeps the `Ref` [`tyref_to_value_type_with`]
        // already recorded for that header. The items-block accessors
        // ([`items_block_accessor_returns_its_block`]) collapse onto their
        // block too (`rlist.py` `ll_items` is `l.items`), so their `*mut u8`
        // result keeps that `Ref` record, not `CCHARP`.
        let callee_returns_gc_block = match &call.func {
            CallFunc::Regular(reg) => regular_call_name_path(reg, self.llbc)
                .as_deref()
                .is_some_and(|p| {
                    is_typed_array_base_adapter(p) || items_block_accessor_returns_its_block(p)
                }),
            _ => false,
        };
        // The destination place sometimes carries a bare `Option` type_id
        // with no generics. The callee `FunDecl` output still names the
        // concrete instantiation (`Option<usize>`, `Option<*mut PyObject>`),
        // which is the ClassDef the producer and the match must share
        // (`bookkeeper.py` `getuniqueclassdef(cls)`).
        let fd_output = match &call.func {
            CallFunc::Regular(reg) => regular_call_fun_decl_id(&reg.kind)
                .and_then(|id| self.llbc.fn_by_id(id))
                .map(|fd| clone_tyref(&fd.signature.output)),
            _ => None,
        };
        let dest_class_ty: &TyRef = match &fd_output {
            Some(out)
                if tyref_enum_instantiation_suffix(&call.dest.ty, self.llbc).is_empty()
                    && !tyref_enum_instantiation_suffix(out, self.llbc).is_empty() =>
            {
                out
            }
            _ => &call.dest.ty,
        };
        let result_ty = if is_unit_type(&call.dest.ty, self.llbc) {
            ValueType::Void
        } else if !callee_returns_gc_block
            && !graph_is_items_block_base_accessor(&self.graph.name)
            && let Some(ccharp) = raw_u8_ccharp_bank(&call.dest.ty, self.llbc)
        {
            // `rffi.py` `CCHARP`. Int-banked so overlay Signed keeps the stamp.
            ccharp
        } else if !callee_returns_gc_block
            && let Some(raw) = raw_scalar_address_value_type(&call.dest.ty, self.llbc)
        {
            raw
        } else {
            tyref_to_value_type_with(
                dest_class_ty,
                self.llbc,
                self.tombstoned_leaves,
                self.gc_struct_ids,
                &self.raw_alloc_flavor_structs,
            )
        };
        // A typed `Ref(Some(root))` already carries the intern key
        // (`tyref_to_value_type` paints a payload-carrying enum this way).
        // `__cast_instance_intrinsic` then intern's that root through
        // `getuniqueclassdef_for_struct_root` so the call result is
        // `SomeInstance(classdef)` rather than the classdef-less shell
        // `valuetype_to_someshell(Ref)` would seed.
        //
        // A call returning `*mut <registered ADT>` now carries
        // `Ref(Some(pointee))` from `tyref_to_value_type_with`. Recast
        // only when the target root differs from that declared pointee
        // (a downcast). Identity at jitcode
        // (`__cast_instance_intrinsic` → `cast_pointer` → `same_as`)
        // when the roots already match.
        //
        // `tupleobject.py wraptuple` length 2 unpacks `list_w` through
        // `specialisedtupleobject.py makespecialisedtuple` (`w_arg1, w_arg2 =
        // list_w`). Those reads are `rlist.py ll_getitem_fast` —
        // `getarrayitem` of `GcArray(OBJECTPTR)`. An items-block base
        // accessor (`is_object_items_block_base_accessor`: `l.items`)
        // residualizes as a HostObject whose result is a classdef-less
        // instance, so `ArrayRead` → flowspace `getitem` rewrites to
        // `getattr("__getitem__")` (`getitem_SomeInstance`). Narrow through
        // the `[Class]` list spelling brick 3 already uses for
        // `*base.add(idx)` (`object_ref_items_list_root`).
        let items_block_items_list_root = match &call.func {
            CallFunc::Regular(reg)
                if regular_call_is_items_block_accessor(reg, self.llbc)
                    && !graph_is_string_items_owner(&self.graph.name) =>
            {
                object_ref_items_list_root(&call.dest.ty, self.llbc)
                    .or_else(|| object_ref_items_list_root(dest_class_ty, self.llbc))
            }
            _ => None,
        };
        let result_narrow_root: Option<String> = match &result_ty {
            // A producer's graph returns the rbigint its handle roots
            // ([`OwnerRootPlan`]), the instance `RBigInt::clone` narrows to.
            _ if self.owner_root.erases(dest_local) || rerooted => Some("RBigInt".to_string()),
            _ if items_block_items_list_root.is_some() => items_block_items_list_root,
            ValueType::Ref(Some(root)) => {
                // A pointer-niche Option is `SomeInstance(payload)`, not the
                // Option enum class, even when `tyref_to_value_type` names
                // a split-eligible instantiation. Prefer the payload class
                // the niche arm below already uses.
                //
                // A string/list niche has no instance ClassDef
                // (`option_niche_payload_class_root` is `None`); keep the
                // split `Option<String>` / `Option<Vec<_>>` spelling so
                // `project_struct_field_type` unions the payload with
                // `s_None` (`Bookkeeper.project_struct_field_type` Option
                // arm, `model.py` `noneify`). Dropping the cast left a
                // residual host call as a classdef-less instance, and
                // `if let Some(s)` then could not return a `SomeString`.
                if self.tyref_is_niche_option_ptr(dest_class_ty) {
                    self.option_niche_payload_class_root(dest_class_ty)
                        .or_else(|| {
                            self.option_niche_null_cast(dest_class_ty)
                                .is_some()
                                .then(|| root.clone())
                        })
                } else if crate::front::result_exc::tyref_is_result_of_carrier(
                    &call.dest.ty,
                    self.llbc,
                    self.static_addrs.error_carrier,
                ) || crate::front::result_exc::tyref_is_result_of_carrier(
                    dest_class_ty,
                    self.llbc,
                    self.static_addrs.error_carrier,
                ) {
                    // `result_exc` erases Result-of-carrier to `T` plus
                    // exception links. A recast here would be the last op,
                    // and checkgraph refuses that result on the exception
                    // link (`simplify.py`).
                    None
                } else {
                    Some(root.clone())
                }
            }
            ValueType::Ref(None) => tyref_node(dest_class_ty, self.llbc)
                .and_then(|n| strip_ty_wrappers(n, self.llbc))
                .and_then(|n| raw_ptr_pointee_class_root_with(n, self.llbc, self.tombstoned_leaves))
                // `Option<&mut RegisteredStruct>` is a nullable pointer
                // niche, not a boxed Option object. RPython represents it as
                // `SomeInstance(Struct, can_be_None=True)`: narrow to the
                // payload class directly so a generated gateway wrapper's
                // successful match arm can dispatch `self.method()`.
                .or_else(|| self.option_niche_payload_class_root(dest_class_ty))
                // A `dont_look_inside` residual returning `Option<*mut T>`
                // whose dest type is the `Option<…>` ADT, not a bare raw
                // pointer, misses `raw_ptr_pointee_class_root`. Narrow it
                // to the per-instantiation `Option` enum root a constructed
                // `Some(..)`/`None` of the same instantiation mints, so the
                // caller's `if let Some(x) = ..` (`Discriminant` →
                // `__discriminant` read, then the Some-arm `__pos_0`
                // payload) resolves against the Option classdef.
                .or_else(|| self.option_residual_narrow_root(dest_class_ty))
                // A tuple/array returned across a call boundary has the same
                // synthetic positional layout as one built in the caller,
                // but `tyref_to_value_type` still classifies those
                // non-scalars as `Ref(None)`.  Recover the exact shaped owner
                // here so the caller's `.N` reads see the TupleRepr fields
                // registered from the callee's aggregate construction.
                .or_else(|| tyref_positional_aggregate_root(&call.dest.ty, self.llbc))
                // A by-value `RBigInt` (and `&RBigInt`) return is the instance
                // `annotationoftype` builds via `getuniqueclassdef`. The
                // register kind is still `Ref(None)`, so
                // `valuetype_to_someshell` would mint
                // `SomeInstance(classdef=None)` and
                // `SomeInstance.getattr("_digits")` (`unaryop.py`) cannot
                // see the field. Narrow with the same cast a raw-pointer
                // pointee uses above. `RBigInt::clone` in `live_rbigint` is
                // the producer that feeds `RBigIntGcRoot::new`.
                .or_else(|| self.rbigint_result_class_root(&call.dest.ty))
                .or_else(|| {
                    let root =
                        tyref_class_root_with(&call.dest.ty, self.llbc, self.tombstoned_leaves)?;
                    root.rsplit("::")
                        .next()
                        .is_some_and(|leaf| {
                            matches!(leaf, "ClassDictStrategy" | "SysState" | "TypedItemsBlock")
                        })
                        .then_some(root)
                }),
            _ => None,
        };
        // The Call dest is already `SomeInstance(root)` when `result_ty`
        // names that root. Recast only a genuine downcast. Keep
        // `result_narrow_root` for the ARRAY layout stamp
        // (`do_fixed_newlist` `op.result.concretetype.TO`) and for an
        // array-item recast whose getarrayitem dest is still GCREF.
        let recast_result_root = match (&result_ty, result_narrow_root.as_ref()) {
            (ValueType::Ref(Some(declared)), Some(narrowed)) if declared == narrowed => None,
            (_, other) => other.cloned(),
        };

        // Resolve arguments before deciding the call shape so receiver
        // resolution and `dyn` operand handling share the same path.
        let mut args: Vec<Variable> = Vec::with_capacity(call.args.len());
        // STR/list views converted to `(data, len)` for this call
        // (`emit_str_to_window` / `emit_list_view_to_window`). Keepalive
        // after the last use of that raw `char *` (`keepalive_until_here`).
        let mut converted_window_owners: Vec<Variable> = Vec::new();
        // MIR local index behind each plain-local argument, kept
        // alongside the resolved Variables so call intercepts can
        // consult per-local lowering state
        // (`const_discriminant_locals`).
        let arg_locals: Vec<Option<usize>> = call
            .args
            .iter()
            .map(|op| match op {
                Operand::Copy(p) | Operand::Move(p) => match p.kind {
                    PlaceKind::Local(i) => Some(i as usize),
                    _ => None,
                },
                Operand::Const(_) => None,
            })
            .collect();
        // Kept beside `arg_locals`. A projection (`pair.0`) and a literal
        // have no bare local; `spill_address_source` reads the operand.
        let arg_operands = call.args.clone();
        // Every argument's place type, captured before the operands are
        // consumed. A residual `fn f<T>(x: &T)` keeps `T` as a `TypeVar`
        // on the declaration (`monomorphize:false`); the place type is the
        // type the call actually passes.
        let call_arg_tys: Vec<Option<TyRef>> = call
            .args
            .iter()
            .map(|op| operand_tyref(op).map(clone_tyref))
            .collect();
        // First argument's MIR-declared type, captured before the
        // operands are consumed — `reflexive_into_alias` compares it
        // against the destination type.
        let first_arg_ty: Option<TyRef> = call.args.first().and_then(|op| match op {
            Operand::Copy(p) | Operand::Move(p) => Some(clone_tyref(&p.ty)),
            Operand::Const(_) => None,
        });
        // Captured before `call.args` is consumed. A constant word
        // (`RBigInt::from(1)`) has no place type; its `ty` is what the
        // constructor residual below classifies.
        let const_arg_ty = call
            .args
            .first()
            .and_then(|op| const_operand_tyref(op, self.llbc));
        // Captured before `call.args` is consumed: `slice::len` on
        // `Copy(*byte_view)` must see the mark `Rvalue::Len` already follows.
        // Second argument's MIR-declared type. Captured before the operands
        // are consumed so later classifiers (slice index, `FnDef` identity,
        // combinator env) can read it.
        let second_arg_ty: Option<TyRef> = call.args.get(1).and_then(|op| match op {
            Operand::Copy(p) | Operand::Move(p) => Some(clone_tyref(&p.ty)),
            Operand::Const(_) => None,
        });
        let first_arg_is_string_byte_view = call
            .args
            .first()
            .is_some_and(|op| self.operand_is_string_byte_view(op));
        let first_arg_is_list_view = call
            .args
            .first()
            .is_some_and(|op| self.operand_is_list_view(op));
        let second_arg_is_string_byte_view = call
            .args
            .get(1)
            .is_some_and(|op| self.operand_is_string_byte_view(op));
        // A constant word (`vb.int_eq(1)`) has no place type. Its declared
        // `ty` is what the int-comparison residual classifies, the same way
        // `RBigInt::from(1)` uses `const_arg_ty`.
        let second_const_arg_ty = call
            .args
            .get(1)
            .and_then(|op| const_operand_tyref(op, self.llbc));
        // Function-item identity of arg #1, captured before `call.args` is
        // consumed.  `Option::map(opt, named_fn)` passes a `FnDef` constant
        // (or a Copy/Move of a `FnDef`-typed local); that shape has no
        // closure ADT, so `second_arg_ty` alone cannot name the callee.
        let second_arg_fn_item = operand_fn_item_segments(self.llbc, call.args.get(1));
        let second_arg_fn_decl = operand_fn_item_decl(self.llbc, call.args.get(1));
        // Third argument's MIR-declared type. Captured before the operands
        // are consumed so a 3-arg Regular call (`items_block_set_ref`,
        // `copy_nonoverlapping`) can classify that operand after `call.args`
        // is moved.
        let third_arg_ty: Option<TyRef> = call.args.get(2).and_then(|op| match op {
            Operand::Copy(p) | Operand::Move(p) => Some(clone_tyref(&p.ty)),
            Operand::Const(_) => None,
        });
        // A literal scalar index (`slice.get(1)`) has no Place type to retain
        // in `second_arg_ty`.  Charon still records the method's `I` type in
        // the regular-call generics, so classify the `get` before consuming
        // `call.func`.  This is the same two-source type test used for
        // `SliceIndex::index` below and keeps Range* implementations on the
        // separate getslice lowering.
        let (
            slice_get_index_is_scalar,
            slice_get_element,
            slice_first_element,
            slice_object_array_type_id,
        ) = match &call.func {
            CallFunc::Regular(reg) => {
                let scalar = self.is_slice_get_scalar_call(reg, second_arg_ty.as_ref());
                let is_get_mut = self.is_slice_get_mut_call(reg);
                let element = scalar
                    .then(|| self.slice_get_element(reg))
                    .flatten()
                    .filter(|(_, array_type_id)| {
                        crate::front::slice_get::slice_get_element_may_record(
                            is_get_mut,
                            array_type_id.as_ref(),
                        )
                    });
                // `first` / `last` share `get`'s `T` generic; the same
                // ARRAY identity / thin-pointer proof feeds their
                // `ArrayRead`. Skip the generic walk on every other
                // Regular call.
                let first_element = self
                    .is_slice_first_or_last_call(reg)
                    .then(|| self.slice_get_element(reg))
                    .flatten();
                // `arraylen_gc` only for an object-pointer receiver. The
                // element-read identity above is a different value: a
                // `[u8]` slice is length-prefixed and still not the GC
                // array object `bh_arraylen_gc` reads.
                let object_array_type_id = self.slice_object_array_type_id(reg);
                (scalar, element, first_element, object_array_type_id)
            }
            _ => (false, None, None, None),
        };
        // A void ZST stays in `Call.args`. The callee's `Input` still
        // lists that parameter, and the annotator counts it. The
        // codewriter drops it later, on both sides, where `FUNC.ARGS`
        // has already filtered `Void`. Dropping it here makes
        // `Arg<T>::get` lose `args[0]` and a `dont_look_inside` host
        // see one argument fewer than its signature.
        // A pair slice argument is its pointer here; its length joins the
        // argument list only in the call built for it below.
        // `call.args` is moved just below. An explicit `drop(move local)`
        // still needs that place so it can run the same drop as a
        // `TermKind::Drop` of it.
        let moved_local_arg = match call.args.first() {
            Some(Operand::Move(place)) if matches!(place.kind, PlaceKind::Local(_)) => {
                Some(place.clone())
            }
            _ => None,
        };
        // A monomorphic `&[T]` / `&[u8]` FunDecl takes the two-word ABI.
        // A generic translation subject is substituted with the call's
        // `generics.types` (`hash_of<Q>` with `Q = [u8]` → `&[u8]`).
        // Generic `core::slice::<Impl>::get` is opaque (no unstructured
        // body) and still spells `&[T]`; splicing the window length there
        // makes `pair_callee_decl` reject the call. Those sites keep the
        // data word as the receiver and leave the length in the frame
        // (`slice_get` reads it via FatLen / the recorded diamond).
        let (pair_input_is_pair_or_window, callee_output_is_window): (Vec<bool>, bool) =
            match &call.func {
                CallFunc::Regular(reg) => regular_call_fun_decl_id(&reg.kind)
                    .and_then(|id| self.llbc.fn_by_id(id))
                    .map(|fd| {
                        let (inputs, output) = self.pair_or_window_fun_inputs(fd, &reg.generics);
                        (
                            inputs,
                            output
                                && fd.is_global_initializer().is_none()
                                && !fundecl_is_promoted_const(fd),
                        )
                    })
                    .unwrap_or_default(),
                _ => (Vec::new(), false),
            };
        let mut pair_lens: Vec<(usize, Variable)> = Vec::new();
        let from_bytes_identity = match &call.func {
            CallFunc::Regular(reg) => self.is_wtf8_from_bytes_identity(reg, &call.dest.ty),
            _ => false,
        };
        // Len / index / to_vec / a list-view dest keep the list
        // (`rlist.py` `ll_length` / `ll_getitem_nonneg` / `ll_copy`).
        let keep_list_view_arg = match &call.func {
            CallFunc::Regular(reg) => {
                self.is_slice_len(reg)
                    || self.is_slice_is_empty(reg)
                    || self.is_slice_scalar_index_call(reg, second_arg_ty.as_ref())
                    || regular_call_is_slice_to_vec(reg, self.llbc)
                    || self.list_view_locals.contains(&dest_local)
            }
            _ => false,
        };
        let host_dict = match &call.func {
            CallFunc::Regular(reg) => is_rdict_host_rewritten_call(reg, self.llbc),
            _ => false,
        };
        let host_dict_name = match &call.func {
            CallFunc::Regular(reg) if host_dict => {
                regular_call_name_path(reg, self.llbc).unwrap_or_else(|| "RDict".to_string())
            }
            _ => String::new(),
        };
        // `<[u8]>::to_vec` of a STR view copies the STR itself
        // (`ll_str2uintlist`), so its receiver stays the one-word STR.
        let call_is_slice_to_vec = matches!(
            &call.func,
            CallFunc::Regular(reg) if regular_call_is_slice_to_vec(reg, self.llbc)
        );
        for op in call.args {
            let index = args.len();
            let decl_wants_pair = pair_input_is_pair_or_window
                .get(index)
                .copied()
                .unwrap_or(false);
            if let Some((ptr, len)) = self.pair_or_window_operand(mir_bb, &op)? {
                if host_dict && index != 0 {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: window argument of {host_dict_name}"
                    )));
                }
                let src_is_window =
                    operand_tyref(&op).is_some_and(|ty| tyref_is_u8_window(ty, self.llbc));
                if decl_wants_pair || !src_is_window {
                    pair_lens.push((index, len));
                }
                let ptr = if src_is_window {
                    self.ensure_window_ccharp(mir_bb, ptr)
                } else {
                    ptr
                };
                args.push(ptr);
                continue;
            }
            if !host_dict
                && decl_wants_pair
                && self.operand_is_string_byte_view(&op)
                && !call_is_slice_to_vec
                && !from_bytes_identity
                && !tyref_is_u8_window(&call.dest.ty, self.llbc)
                && !tyref_is_string_value(&call.dest.ty, self.llbc)
            {
                let var = self.resolve_operand(mir_bb, op.clone())?;
                converted_window_owners.push(var.clone());
                let (ptr, len) = self.owner_to_window(mir_bb, var, operand_tyref(&op))?;
                pair_lens.push((index, len));
                args.push(self.ensure_window_ccharp(mir_bb, ptr));
                continue;
            }
            if !host_dict
                && decl_wants_pair
                && self.operand_is_list_view(&op)
                && !keep_list_view_arg
            {
                let var = self.resolve_operand(mir_bb, op)?;
                converted_window_owners.push(var.clone());
                let (ptr, len) = self.emit_list_view_to_window(mir_bb, var)?;
                pair_lens.push((index, len));
                args.push(self.ensure_window_ccharp(mir_bb, ptr));
                continue;
            }
            args.push(self.resolve_operand(mir_bb, op)?);
        }
        let resolved_call_args = args.clone();
        let gc_mut_ref_sig = self.gc_mut_ref_signature(&call.func, &call_arg_tys);
        // Captured before `call.func` moves into the call-shape match.
        let raw_scalar_spills = self.raw_scalar_borrow_spills(&call.func, &call_arg_tys);
        let first_arg_is_string_array_view = args.first().is_some_and(|arg| {
            self.string_array_view_locals.iter().any(|(local, _)| {
                self.local_var
                    .get(*local)
                    .and_then(Option::as_ref)
                    .and_then(|current| current.one().ok())
                    .is_some_and(|current| current == *arg)
            })
        });

        let class = call.func.classify();
        // A STR view is one word, so `pair_lens` is empty. Rewrite to
        // `same_as` after Result-scope / combinator capture so identity
        // shares that tail; a true W3 window copies with `charpsize2str`.
        let from_bytes_one_word = from_bytes_identity && args.len() == 1 && pair_lens.is_empty();
        let from_bytes_window_len = if from_bytes_identity {
            pair_lens
                .iter()
                .find(|(index, _)| *index == 0)
                .map(|(_, len)| len.clone())
        } else {
            None
        };
        let window_recv_len = pair_lens
            .iter()
            .find(|(index, _)| *index == 0)
            .map(|(_, len)| len.clone());
        // Full `name_path()` of a `CallKind::Fun` callee, captured for the
        // `Result<T, PyError>` scope decision at the `?`-diamond capture
        // site below: the built `CallTarget::Method` keeps only the leaf,
        // losing the module path the scope predicate keys on.  It is the
        // same string the callee-side gate sees (`fd.item_meta.name_path()`,
        // `lower_fun_decl_with_static_addrs`), so caller and callee agree on
        // whether a given callee is scoped.
        let mut callee_name_path: Option<String> = None;
        // `v.extend_from_slice(s)` on a `Vec` of one-word items is
        // `ll_extend` from the pair.
        let vec_extend_kind = match &call.func {
            CallFunc::Regular(reg)
                if matches!(pair_lens.as_slice(), [(1, _)])
                    && regular_call_name_path(reg, self.llbc).as_deref()
                        == Some("alloc::vec::<Impl>::extend_from_slice") =>
            {
                first_arg_ty
                    .as_ref()
                    .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc))
            }
            _ => None,
        };
        // `v.extend_from_slice(s)` of a one-word object slice is
        // `rlist.ll_extend` with `l2` the GcArray word.
        let vec_extend_gcarray_kind = match &call.func {
            CallFunc::Regular(reg)
                if pair_lens.is_empty()
                    && args.len() == 2
                    && regular_call_name_path(reg, self.llbc).as_deref()
                        == Some("alloc::vec::<Impl>::extend_from_slice")
                    && second_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_object_pointer_slice(ty, self.llbc)) =>
            {
                first_arg_ty
                    .as_ref()
                    .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc))
                    .filter(|&kind| kind == majit_ir::rvec::VecItemKind::Ref)
            }
            _ => None,
        };
        // An indirect call through a fn pointer or a vtable slot passes each
        // pair slice as its two words, the way the callee's signature
        // spells a `&[T]` parameter.
        let passes_pairs_indirectly = matches!(
            (&class, &call.func),
            (CallClass::Dynamic, CallFunc::Dynamic(..))
        );
        // `<[u8]>::to_vec` by representation, before the pair-ABI match: a
        // W3 window is `rffi.charpsize2str`, not a pair call of `to_vec`.
        if let CallFunc::Regular(reg) = &call.func
            && args.len() == 1
            && regular_call_is_slice_to_vec(reg, self.llbc)
            && tyref_rust_vec_item_kind(&call.dest.ty, self.llbc).is_none()
        {
            let recv_is_w1 = first_arg_is_string_byte_view
                || arg_locals
                    .first()
                    .copied()
                    .flatten()
                    .is_some_and(|local| self.local_holds_str_view(local))
                || args.first().is_some_and(|v| self.var_is_str_repr(v));
            if recv_is_w1 {
                let dest = self.emit_ll_str_to_u8_list(bb_id, args[0].clone());
                self.local_var[dest_local] = Some(LocalValue::One(dest));
                let target_bb = self.block_id[target];
                let link_args = self.edge_args(mir_bb, target)?;
                self.graph.set_goto(bb_id, target_bb, link_args);
                return Ok(());
            }
            let recv_is_w2 = first_arg_is_list_view
                || first_arg_ty
                    .as_ref()
                    .is_some_and(|ty| tyref_is_u8_list(ty, self.llbc));
            if !recv_is_w2 {
                let mut window_parts = if let Some(op) = arg_operands.first() {
                    self.window_operand(mir_bb, op)?
                } else {
                    None
                };
                if window_parts.is_none()
                    && first_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_u8_window(ty, self.llbc))
                {
                    if let (Some(data), Some(len)) = (args.first(), window_recv_len.as_ref()) {
                        window_parts = Some((data.clone(), len.clone()));
                    }
                }
                if let Some((data, len)) = window_parts {
                    let dest = self.emit_charpsize2str_list(mir_bb, data, len);
                    self.local_var[dest_local] = Some(LocalValue::One(dest));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                let unclassified_u8 = first_arg_ty.as_ref().is_some_and(|ty| {
                    tyref_is_u8_window(ty, self.llbc)
                        || tyref_is_u8_list(ty, self.llbc)
                        || tyref_is_bare_u8_slice(ty, self.llbc)
                });
                if unclassified_u8 {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: to_vec: unclassified [u8] receiver"
                    )));
                }
            }
        }
        let op_kind = match (class, call.func) {
            (CallClass::Direct | CallClass::Trait, CallFunc::Regular(_))
                if let Some(kind) = vec_extend_kind =>
            {
                let args = splice_pair_lens(args, pair_lens);
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: majit_ir::rvec::vec_helper_path(
                            majit_ir::rvec::VecOp::ExtendFromSlice,
                            kind,
                        )
                        .split("::")
                        .map(str::to_string)
                        .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(args),
                    result_ty: ValueType::Void,
                }
            }
            (CallClass::Direct | CallClass::Trait, CallFunc::Regular(_))
                if let Some(kind) = vec_extend_gcarray_kind =>
            {
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: majit_ir::rvec::vec_helper_path(
                            majit_ir::rvec::VecOp::Extend,
                            kind,
                        )
                        .split("::")
                        .map(str::to_string)
                        .collect(),
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: crate::model::call_args(args),
                    result_ty: ValueType::Void,
                }
            }
            // A call passing pair slices is the plain call of a callee that
            // declares them, with each length after its pointer.
            // `<[u8]>::to_vec` of a window is `rffi.charpsize2str`, not the
            // pair-ABI residual of `to_vec` itself (`is_slice_to_vec`).
            (CallClass::Direct | CallClass::Trait, CallFunc::Regular(reg))
                if !pair_lens.is_empty()
                    && !from_bytes_identity
                    && !regular_call_is_slice_to_vec(&reg, self.llbc) =>
            {
                // Checked `Wtf8::from_bytes(&[u8])` is the window pair
                // (`rutf8.py check_utf8` / `unicodehelper.py str_decode_utf8`).
                if self.is_wtf8_from_bytes_checked(&reg, &call.dest.ty) {
                    let spliced = splice_pair_lens(args.clone(), pair_lens.clone());
                    if self.try_lower_wtf8_from_bytes_checked(
                        mir_bb,
                        &reg,
                        &spliced,
                        dest_local,
                        &call.dest.ty,
                        target,
                    )? {
                        return Ok(());
                    }
                }
                let name = regular_call_name_path(&reg, self.llbc).unwrap_or_default();
                self.pair_callee_decl(mir_bb, &reg, &name, &pair_lens, false)?;
                // The `?`-site capture below keys the callee's scope on it.
                callee_name_path = Some(name);
                let target = self.pair_call_target(mir_bb, &reg, &args, &pair_lens)?;
                // Stamp before splice so FunDecl input indices still match.
                let args = self.coerce_ccharp_call_args(mir_bb, &reg, args, &call_arg_tys);
                let args = splice_pair_lens(args, pair_lens);
                OpKind::Call {
                    target,
                    args: crate::model::call_args(args),
                    result_ty: result_ty.clone(),
                }
            }
            _ if !pair_lens.is_empty() && !passes_pairs_indirectly && !from_bytes_identity => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: pair slice through a dynamic call"
                )));
            }
            (CallClass::Direct, CallFunc::Regular(reg))
            | (CallClass::Trait, CallFunc::Regular(reg)) => {
                // One-word STR view `*as_ptr().add(i)` is `ll_stritem_nonneg`.
                // Coerce to CCHARP first would `rffi.cast` the header
                // (`coerce_ccharp_call_args`) and plant the marker on int-kind.
                if self.try_lower_byte_view_ptr_add(
                    mir_bb,
                    &reg,
                    &args,
                    &arg_locals,
                    arg_operands.get(1),
                    first_arg_ty.as_ref(),
                    dest_local,
                    target,
                )? {
                    return Ok(());
                }
                let mut args = self.coerce_ccharp_call_args(mir_bb, &reg, args, &call_arg_tys);
                // A boundary the consumer declared as one scalar field
                // store ([`crate::ScalarFieldStore`]).  Emit the
                // `setfield_gc_<bank>` its body performs instead of
                // descending into the `&mut i64/f64` borrow the body takes
                // into that field, which has no pointer-to-primitive repr
                // here, or rebuilding the container around a field that is
                // inline.  The owner/id are the same keys
                // `derive_program_metadata` installs from Charon's layouts,
                // so the write carries the descr an ordinary
                // `target.field = value` assignment would.
                if let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                    && let Some(store) = self
                        .static_addrs
                        .scalar_field_stores
                        .iter()
                        .find(|store| fd.item_meta.name_path() == store.function_path)
                {
                    if args.len() != 2 {
                        return Err(LowerError::Schema(format!(
                            "bb{mir_bb}: {} expects target and scalar",
                            store.function_path
                        )));
                    }
                    let field_ty = match store.bank {
                        crate::ScalarBank::Int => ValueType::Int,
                        crate::ScalarBank::Float => ValueType::Float,
                    };
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::FieldWrite {
                            base: args[0].clone(),
                            field: FieldDescriptor::new(
                                store.field,
                                Some(store.owner_root.to_string()),
                            )
                            .with_owner_id(Some(
                                majit_ir::descr::StructId::from_canonical(store.owner_root),
                            )),
                            value: args[1].clone().into(),
                            ty: field_ty,
                        },
                    });
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::mem::replace(&mut place, new)` is `old = place; place = new`
                // (`getarrayitem_gc`/`setarrayitem_gc` or `getfield_gc`/`setfield_gc`).
                // The borrowed place is on `atomic_ref_place` (or `index_elem_alias`
                // when the reference is an `index_mut`).  No place: leave the call.
                if self.try_lower_mem_exchange(
                    mir_bb,
                    &reg,
                    &args,
                    &arg_locals,
                    dest_local,
                    target,
                )? {
                    return Ok(());
                }
                // Fold an inline `core::mem::size_of::<T>()` /
                // `align_of::<T>()` call to its build-time byte size /
                // alignment — the same constant `fold_size_const_global`
                // produces for the NamedConst-initializer form, applied here
                // to the inline call shape (`gc_alloc_storage_box`'s
                // `try_gc_alloc_stable_raw(tid, size_of::<T>())`, and
                // `Layout::from_size_align(_, align_of::<usize>())`).
                // Removes the residual `<host std.mem.size_of>` /
                // `align_of` call the rtyper cannot register.  Declines
                // (falls through to the ordinary call path) for a pointer /
                // tuple / unresolved layout.
                if let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                {
                    let want_align = match fd.item_meta.name_path().as_str() {
                        "core::mem::size_of" => Some(false),
                        "core::mem::align_of" => Some(true),
                        _ => None,
                    };
                    if let Some(want_align) = want_align
                        && let Some(ty) = reg
                            .generics
                            .get("types")
                            .and_then(serde_json::Value::as_array)
                            .and_then(|a| a.first())
                        && let Some(size) = self.size_align_const_from_tyexpr(want_align, ty)
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind: OpKind::ConstInt(size),
                        });
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                }
                // `Layout::new::<T>()` is `size_of::<T>()` plus `align_of::<T>()`
                // (`Layout::from_size_align` with those constants, which a sized
                // `T` always satisfies).  The grow `unwrap_or_else` abort
                // closure (`object_array::grow_*_items_block` `call_once`)
                // residualised it as `["alloc", "layout", "Layout", "new"]`,
                // which `CallRegistry` cannot type.  Fold to the same
                // virtualized `{size, align}` aggregate
                // `from_size_align` already emits (`build_layout_aggregate`).
                if self.try_lower_layout_new(mir_bb, &reg, dest_local, &call.dest.ty, target)? {
                    return Ok(());
                }
                // Rust exposes RPython's generated shadow-stack publication as
                // a normal `pin_root(PyObjectRef)` call.  Its pointer argument
                // is actually `llmemory.GCREF`, irrespective of the source
                // pointer's external repr, so cast it the way
                // `gct_gc_identityhash` casts a GC helper's pointer argument
                // (`framework.py`) — before FunctionDesc propagation,
                // so strings, W_Root instances, and other GC pointers do not
                // union at the helper's single input cell.
                //
                // The pointer is the LAST argument: the `RootScope` method form
                // `gc_root_pin_path` also admits passes `&self` first, and
                // gating on `args.len() == 1` skipped it entirely, leaving the
                // definition-side stamp and the call site disagreeing about the
                // same callee.
                if callee_pin_gcref_arg && let Some(last) = args.len().checked_sub(1) {
                    args[last] = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            LinkArg::Value(args[last].clone()),
                            "GCREF",
                        )
                        .as_variable()
                        .expect("a materialized GC root stays a Variable")
                        .clone();
                }
                // `rarithmetic.r_uint`: the unique graph's `usize` input is
                // Unsigned. `cast_ptr_to_int` callers otherwise merge
                // `r_uint ∪ int` (`annrpython.py` `mergeinputargs`).
                if callee_identity_hash && let Some(arg) = args.first().cloned() {
                    args[0] = push_r_uint(&mut self.graph, bb_id, arg);
                }
                // `core::intrinsics::transmute::<A, B>(x)` is a bitwise
                // move.  `f64 ↔ i64` is `float2longlong` /
                // `longlong2float`.  Any other pair of equal-size
                // same-bank types is the bank's ordinary copy (`same_as`),
                // including `u8 -> #[repr(u8)]` fieldless enums: the enum
                // is already modelled as its integer tag by
                // `tyref_to_value_type`, and its `TypeDecl` layout supplies
                // the matching one-byte size.  A bank crossing or an unknown
                // / unequal size keeps the existing residual call.
                if args.len() == 1
                    && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                    && matches!(
                        fd.item_meta.name_path().as_str(),
                        "core::intrinsics::transmute" | "core::mem::transmute"
                    )
                {
                    if first_arg_ty.as_ref().is_some_and(|src| {
                        self.tyref_literal_float_atom(src) == Some("F64")
                            && self.tyref_literal_int_atom(&call.dest.ty) == Some("I64")
                    }) {
                        return self.emit_float_bytes_llop(
                            mir_bb,
                            dest_local,
                            target,
                            args[0].clone(),
                            false,
                        );
                    }
                    if first_arg_ty.as_ref().is_some_and(|src| {
                        self.tyref_literal_int_atom(src) == Some("I64")
                            && self.tyref_literal_float_atom(&call.dest.ty) == Some("F64")
                    }) {
                        return self.emit_float_bytes_llop(
                            mir_bb,
                            dest_local,
                            target,
                            args[0].clone(),
                            true,
                        );
                    }
                    if first_arg_ty.as_ref().is_some_and(|src_ty| {
                        transmute_is_same_layout_bank(src_ty, &call.dest.ty, self.llbc)
                    }) {
                        let src_vt = tyref_to_value_type(
                            first_arg_ty.as_ref().expect("same-bank transmute has src"),
                            self.llbc,
                        );
                        if let Some(raw) = self.emit_dest_overlay_cast(
                            bb_id,
                            Some(&src_vt),
                            args[0].clone(),
                            &call.dest.ty,
                        ) {
                            self.local_var[dest_local] = Some(LocalValue::One(raw));
                            let target_bb = self.block_id[target];
                            let link_args = self.edge_args(mir_bb, target)?;
                            self.graph.set_goto(bb_id, target_bb, link_args);
                            return Ok(());
                        }
                        // Same bank and same size: the destination local is
                        // the argument. `remove_same_as` deletes this copy.
                        self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                    // Equal-size ptr↔int: `history.getkind` banks
                    // `Ptr(FuncType)` as `int` and a data pointer as
                    // `ref`, so the pair is a bank crossing.  Residual
                    // `transmute` is not in CallRegistry; the same
                    // conversion is `lltype.cast_ptr_to_int` /
                    // `lltype.cast_int_to_ptr` (`rbuiltin.py`
                    // `rtype_cast_ptr_to_int` / `rtype_cast_int_to_ptr`).
                    if let Some(src_ty) = first_arg_ty.as_ref()
                        && tyref_exact_layout_size(src_ty, self.llbc)
                            == tyref_exact_layout_size(&call.dest.ty, self.llbc)
                        && tyref_exact_layout_size(src_ty, self.llbc).is_some()
                    {
                        let src_vt = tyref_to_value_type(src_ty, self.llbc);
                        let dst_vt = tyref_to_value_type(&call.dest.ty, self.llbc);
                        match (value_type_bank(&src_vt), value_type_bank(&dst_vt)) {
                            (1, 0) => {
                                return self.emit_host_cast_llop(
                                    mir_bb,
                                    dest_local,
                                    target,
                                    args[0].clone(),
                                    "cast_ptr_to_int",
                                    ValueType::Int,
                                );
                            }
                            (0, 1) => {
                                if let Some(raw) = self.emit_dest_overlay_cast(
                                    bb_id,
                                    Some(&src_vt),
                                    args[0].clone(),
                                    &call.dest.ty,
                                ) {
                                    self.local_var[dest_local] = Some(LocalValue::One(raw));
                                    let target_bb = self.block_id[target];
                                    let link_args = self.edge_args(mir_bb, target)?;
                                    self.graph.set_goto(bb_id, target_bb, link_args);
                                    return Ok(());
                                }
                                return self.emit_host_cast_llop(
                                    mir_bb,
                                    dest_local,
                                    target,
                                    args[0].clone(),
                                    "cast_int_to_ptr",
                                    ValueType::Ref(None),
                                );
                            }
                            _ => {}
                        }
                    }
                }
                // `core::intrinsics::saturating_add` / `saturating_sub`
                // (`rint.py` `rtype_add_ovf` / `rtype_sub_ovf` for signed).
                if args.len() == 2
                    && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                    && matches!(
                        fd.item_meta.name_path().as_str(),
                        "core::intrinsics::saturating_add" | "core::intrinsics::saturating_sub"
                    )
                {
                    return self.emit_saturating_intrinsic(
                        mir_bb,
                        dest_local,
                        target,
                        args[0].clone(),
                        args[1].clone(),
                        &call.dest.ty,
                        fd.item_meta
                            .name_path()
                            .as_str()
                            .ends_with("saturating_add"),
                    );
                }
                // `we_are_jitted()` is true during tracing and blackholing
                // (rlib/jit.py); the rtyper folds the surviving
                // `_we_are_jitted` symbolic to a constant True
                // (rlib/jit.py, jtransform.py:1636-1639).  Folding it  allow-line-citation
                // to `ConstBool(true)` here — at the front, before annotation —
                // lets `simplify_lowered_graph`'s `fold_constant_exitswitch`
                // drop each JIT-dead `if not we_are_jitted()` interpreter arm
                // before the graph is annotated, instead of at
                // `fold_we_are_jitted_calls` (jtransform post-annotation).  A
                // dead interpreter arm reading a residual-only static (e.g.
                // `baseobjspace::METHOD_CACHE` behind
                // `lookup_where_with_method_cache`) would otherwise fail the
                // phaseA lift and drop the whole graph to the legacy walker.
                if let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                    && path_ends_with_segments(&fd.item_meta.name_path(), "jit::we_are_jitted")
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ConstBool(true),
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `gc_alloc_storage_box(Wtf8Buf, tid) -> *mut Wtf8Buf` is
                // only the Rust ownership spelling of PyPy's direct `_utf8`
                // RPython-string field.  The input value has already become
                // the allocated `rpy_string`; a second nominal Wtf8Buf box
                // does not exist in the translated object model.  Alias this
                // one specialization to its value, exactly like
                // `malloc_raw_alloc` round-trips a `SomeString`.  Every other
                // `gc_alloc_storage_box<T>` keeps its translated GC-box body.
                if args.len() == 2
                    && self.is_string_storage_box_identity(
                        &reg,
                        first_arg_ty.as_ref(),
                        &call.dest.ty,
                    )
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Reflexive blanket `into` — the callsite selected
                // `impl<T> From<T> for T`, a pure `T -> T` identity
                // conversion.  Bind the destination local to the
                // argument directly (same shape as a transparent
                // ctor alias) instead of emitting a call to core's
                // identity body, which is not a registered callee.
                // `is_reflexive_from` takes the same path for the same
                // impl reached through `From::from` directly.
                //
                // The clause-bound variant — `msg.into()` inside a
                // generic body with `T: Into<String>` — has no
                // resolved impl to devirtualize through; for a
                // string-family target the lifted value model treats
                // the conversion as identity too (Rust `String` and
                // `&str` both lower to the immutable rpy_string), so
                // it takes the same alias path.
                // `core::mem::forget(x)` leaves nothing behind to record: it
                // deliberately suppresses the destructor.  `core::mem::drop(x)`
                // runs that destructor at the call.  A `Move` of a local is
                // the same drop [`TermKind::Drop`] lowers; a type with no
                // drop glue is a no-op.  Any other type keeps the residual
                // call: its `FOREIGN_STDLIB_EXTERNALS` entry types the
                // callsite but still names a `core` body the build has no
                // address for, and one residual target a portal can reach
                // whose address the build left unbound refuses every trace,
                // arm taken or not.  Binding one address instead is not open:
                // the path key collapses every `T`, so a single address would
                // serve monomorphisations that do not share a destructor.
                if args.len() == 1 && self.is_mem_forget(&reg) {
                    let void = self.emit_unit(bb_id);
                    self.local_var[dest_local] = Some(LocalValue::One(void));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `majit_ir::icf::icf_identity_token::<ID>()` /
                // `majit_ir::icf::icf_identity_token_of::<T>()` is a
                // machine-code distinguisher with no semantic effect. genc
                // `FunctionCodeGenerator` emits one C function per graph and
                // never merges two graphs, so the helper's flow graph has no
                // operation for it. Bind the unit destination to `emit_unit`
                // and continue to the call's target block.
                if args.is_empty() && self.is_icf_identity_token(&reg) {
                    let void = self.emit_unit(bb_id);
                    self.local_var[dest_local] = Some(LocalValue::One(void));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1
                    && self.is_mem_drop(&reg)
                    && let Some(place) = moved_local_arg.as_ref()
                    && self.emit_place_drop(mir_bb, place, None)?
                {
                    let void = self.emit_unit(bb_id);
                    self.local_var[dest_local] = Some(LocalValue::One(void));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1
                    && (matches!(self.blanket_into_devirt(&reg), Some(IntoDevirt::Identity))
                        || self.is_reflexive_from(&reg)
                        || self.is_widening_int_from(&reg)
                        || self.trait_clause_into_string_identity(&reg, &call.dest.ty)
                        || self.is_noop_ptr_cast(&reg)
                        || self.is_raw_ptr_as_ref(&reg)
                        || self.is_reflexive_into_iter(&reg)
                        || self.is_hint_must_use(&reg)
                        || self.is_arguments_from_str(&reg)
                        || self.is_rbigint_translated_alias(&reg, first_arg_ty.as_ref()))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String>::deref` / `<str>::deref` and the Wtf8 string
                // family: a `Deref` between two string-value types is a
                // pointer-follow with no transformation.  `String` / `&str`
                // / `Wtf8` / `Wtf8Buf` all project to the single immutable
                // `s_unicode0` value, so the deref is identity — alias the
                // destination to the argument directly (the same zero-op
                // shape as the reflexive `into` above) instead of falling
                // through to the residual `method("deref", …)` build, an
                // unregistered callee.  `deref_cast_root` returns `None` for
                // these (the `&str` dest resolves no struct root), so the
                // cast arm below declines; gate on the receiver AND the
                // destination both being string values so slice / `Vec`
                // derefs (`&[T]`, root `Builtin::Slice`) keep their ordinary
                // path.
                //
                // `String::from_utf8_lossy` is rewritten to a plain string
                // (`unicodehelper.py str_decode_utf8` replace arm).  The
                // consumer is `push_str(&*cow)`: dest is string-family and
                // the Cow local is already the produced string
                // (`string_byte_view_locals`).  A residual Cow ADT is not
                // in that set and keeps the ordinary path.
                //
                // `Deref` takes `&self`, so the receiver is often `&Cow<str>`
                // (a `Ref` of the dest local, or a later `&_cow` binding).
                // `arg_locals` is `None` for a projection; follow the mark
                // the same way `Rvalue::Len` does.
                if args.len() == 1
                    && is_deref_call(&reg, self.llbc)
                    && tyref_is_string_family(&call.dest.ty, self.llbc)
                {
                    let from_string = first_arg_ty
                        .as_ref()
                        .is_some_and(|t| tyref_is_string_family(t, self.llbc));
                    let from_lossy_cow = first_arg_ty
                        .as_ref()
                        .is_some_and(|t| self.tyref_is_cow_str(t))
                        && first_arg_is_string_byte_view;
                    if from_string || from_lossy_cow {
                        // A builder-mode accumulator's Shared borrow feeding this
                        // deref is `ll_build`
                        // (`AbstractStringBuilderRepr.rtype_method_build`): the
                        // builder is a STRINGBUILDERPTR, and identity assumes
                        // SomeString.
                        let arg = match self.builder_identity_string(mir_bb, &arg_locals)? {
                            Some(built) => built,
                            None => args[0].clone(),
                        };
                        if from_lossy_cow {
                            self.alias_dest_to_arg0(dest_local, arg, true);
                        } else {
                            self.alias_dest_to_arg0_inherit(dest_local, arg, &arg_locals);
                        }
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                }
                // A bodyless `Deref::deref` / `DerefMut::deref_mut` (`Box<T>`,
                // and a one-word transparent wrapper) whose `&T` is a
                // registered struct.  The handle is one pointer word and
                // this LLBC has no body for the impl, so `*p` is a typed
                // pointer reinterpret: emit the `cast_pointer(T, p)`
                // downcast marker (`cast_pointer_marker_op`) the flowspace
                // adapter rebuilds into
                // `simple_call(lltype.cast_pointer, T, p)`, yielding
                // `SomeInstance(T)` regardless of the receiver's
                // classdef-less annotation.  A `deref` / `deref_mut` whose
                // `FunDecl` has a body is an ordinary method:
                // [`Self::deref_cast_root`] declines and the call lowers
                // on the normal path, so that body is lowered and inlined
                // like any other callee.  `Rc` / `Arc` stay out of the
                // cast (their word addresses a refcount header).  Slice /
                // `str` derefs resolve no struct root and keep their
                // ordinary lowering.
                if args.len() == 1
                    && let Some(root) = self.deref_cast_root(&reg, &call.dest.ty)
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: cast_pointer_marker_op(root, args[0].clone()),
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String>::deref` / `<Vec<T>>::deref` (+ `deref_mut`) —
                // identity in the lifted value model (String/&str and
                // Vec/&[T] share one repr), so alias the destination to
                // the receiver instead of emitting a `deref` method call
                // the rtyper cannot route on the classdef-less receiver.
                //
                // A `Vec` of one-word items is the address of its raw
                // `{ptr, len, cap}` header, not a list, so its slice is not
                // the receiver.  Only `<[T]>::reverse` consumes the alias,
                // which the call arm retargets to `ll_vec_reverse`; any other
                // slice use would read the header words as items, so that
                // deref keeps the residual `deref` call, which the annotator
                // rejects on the `SomeRustVec` receiver.
                if args.len() == 1
                    && self.is_container_identity_deref(&reg)
                    && (first_arg_ty
                        .as_ref()
                        .is_none_or(|ty| tyref_rust_vec_item_kind(ty, self.llbc).is_none())
                        || rust_vec_deref_feeds_only_reverse(self.body, self.llbc, dest_local))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // A shared `&[u8]` dest produced by deref / as-slice of a W2
                // list owner aliases the list (`rlist.py` `ll_length` /
                // `ll_getitem_nonneg`). Convert to a window only at a
                // window boundary.
                if args.len() == 1
                    && self.list_view_locals.contains(&dest_local)
                    && (first_arg_is_list_view
                        || first_arg_ty
                            .as_ref()
                            .is_some_and(|ty| tyref_is_u8_list(ty, self.llbc)))
                {
                    self.bind_list_view_local(dest_local, args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String as AsRef<str>>::as_ref` / `String::as_str` /
                // `<String as Borrow<str>>::borrow` — a `&str` view of the
                // same string, identity in the lifted value model, so alias
                // the destination to the receiver instead of emitting an
                // `as_ref` method call the rtyper cannot route on the
                // classdef-less string receiver.
                if args.len() == 1 && self.is_std_borrow_identity(&reg) {
                    let arg = match self.builder_identity_string(mir_bb, &arg_locals)? {
                        Some(built) => built,
                        None => args[0].clone(),
                    };
                    self.alias_dest_to_arg0_inherit(dest_local, arg, &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1
                    && self.is_string_to_str_identity(&reg, first_arg_ty.as_ref(), &call.dest.ty)
                {
                    let arg = match self.builder_identity_string(mir_bb, &arg_locals)? {
                        Some(built) => built,
                        None => args[0].clone(),
                    };
                    self.alias_dest_to_arg0_inherit(dest_local, arg, &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `utf8_payload_wtf8` / `utf8_payload_bytes` are views of an
                // rstr `STR` header (`Function.name`, `W_UnicodeObject._utf8`).
                // Alias dest to that STR so `from_utf8_unchecked` /
                // `as_str_unchecked` keep `SomeString` instead of a byte list.
                if args.len() == 1 && self.is_utf8_payload_wtf8_identity(&reg) {
                    self.alias_dest_to_arg0(dest_local, args[0].clone(), false);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1 && self.is_utf8_payload_bytes_identity(&reg) {
                    self.bind_str_source_to_dest(mir_bb, dest_local, args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `prebuilt_str_storage(&str)` / `prebuilt_wtf8_storage(&Wtf8)` is
                // the name's prebuilt `STR` (`CONST_STR_CACHE`,
                // `BaseLLStringRepr.convert_const`), which the operand already is.
                if args.len() == 1 && self.is_prebuilt_str_storage_identity(&reg) {
                    self.alias_dest_to_arg0(dest_local, args[0].clone(), false);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `bytes_block_chars(block)` is `rstr.py` `ll_chars`: identity
                // on the `STR` header. Mark dest a byte view so `len` /
                // index plant `strlen` / `strgetitem`.
                if args.len() == 1 && self.is_bytes_block_chars(&reg) {
                    self.bind_str_source_to_dest(mir_bb, dest_local, args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `w_bytes_data` returns the `&[u8]` view of
                // `W_BytesObject.data`.  The `&[u8]` return type is painted
                // as a u8 window (Int) rather than `Ptr(STR)`, so the read
                // stays hand-lowered until that accessor returns the STR
                // itself.  `w_bytes_block` (returns `*const BytesBlock`),
                // `w_str_get_wtf8` and `w_str_storage` stay flowed.
                if args.len() == 1 && self.is_w_bytes_data(&reg) {
                    let dest = self.emit_w_bytes_data_str_read(bb_id, args[0].clone());
                    self.bind_str_source_to_dest(mir_bb, dest_local, dest);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `String|str|Wtf8|Wtf8Buf::as_bytes` — the UTF-8 / WTF-8
                // byte view of a string.  A string IS its byte sequence in
                // the lifted value model (the immutable `rpy_string`), so
                // the byte view is an identity on the receiver: a downstream
                // byte `==` lowers to `ll_streq`, a `len` to `ll_strlen`.
                // Alias the destination to the receiver instead of the
                // `as_bytes` getattr the rtyper cannot route on the string
                // receiver (the `Cannot find attribute "as_bytes" on String`
                // wall).
                //
                // Scalar consumers are recorded as byte views and lower to
                // RPython's literal `ord(s[i])` pair below. This is distinct
                // from unicode code-point indexing: the receiver here is the
                // UTF-8/WTF-8 storage exposed explicitly by `as_bytes`.
                if args.len() == 1
                    && self.is_string_as_bytes_identity(&reg, first_arg_ty.as_ref(), &call.dest.ty)
                {
                    // `buf.as_bytes()` on a builder-mode accumulator is
                    // `ll_build` then the existing byte-view identity
                    // (`AbstractStringBuilderRepr.rtype_method_build`).
                    let arg = match self.builder_identity_string(mir_bb, &arg_locals)? {
                        Some(built) => built,
                        None => args[0].clone(),
                    };
                    self.bind_str_source_to_dest(mir_bb, dest_local, arg);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `str::to_ascii_lowercase` / `str::replace` are the byte-string
                // methods `StringRepr` already rtypes (`ll_lower`,
                // `ll_replace_chr_chr`). Emit that method call instead of an
                // unregistered `alloc::str` FunctionPath.
                if let Some(method) =
                    self.string_repr_method_name(&reg, args.len(), first_arg_ty.as_ref())
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::Method {
                                name: method.to_string(),
                                receiver_root: Some("str".into()),
                                resolved_path: None,
                                fun_decl_id: None,
                                branch_payloads: None,
                            },
                            args: args.iter().cloned().map(LinkArg::Value).collect(),
                            result_ty: ValueType::Str,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Wtf8::from_bytes_unchecked` / `str::from_utf8_unchecked`
                // identity is rewritten to `same_as` after Result-scope /
                // combinator capture (`from_bytes_one_word`).
                // Checked `Wtf8::from_bytes(&[u8]) -> Option<&Wtf8>`:
                // `rutf8.py check_utf8` with `allow_surrogates=True`, then
                // Ok → `Option::Some` of the unchecked identity, Err → None.
                if self.try_lower_wtf8_from_bytes_checked(
                    mir_bb,
                    &reg,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                // `String::from_utf8_lossy(&[u8]) -> Cow<str>`:
                // `unicodehelper.py str_decode_utf8` replace / U+FFFD arm.
                if self.try_lower_from_utf8_lossy(
                    mir_bb,
                    &reg,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                // Blanket `ToString::to_string` (`alloc::string::<Impl>::to_string`)
                // for a non-string receiver is `str(value)` (`ll_str` /
                // `ll_chr2str` / `ll_int2dec`). A string-family receiver is
                // the identity arm below. A missing receiver type still
                // takes this arm: the dest is `String`, and identity would
                // alias a non-string header. A Float receiver is excluded:
                // `rfloat.py` `FloatRepr.ll_str` / `floatobject.py`
                // `float_repr` residualizes through
                // `rfloat_call::repr_residual_path`.
                if args.len() == 1
                    && self.is_to_string_path(&reg)
                    && !first_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_string_family(ty, self.llbc))
                    && !first_arg_ty.as_ref().is_some_and(|ty| {
                        to_string_receiver_is_float(
                            ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    })
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
                    stamp_as_rstr_str(&res);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "str".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Str,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<str|String as ToString>::to_string`, `Wtf8::to_wtf8_buf`,
                // `<str>::to_owned`, and a string-family `Clone::clone` on a
                // string-family receiver — each an owned copy of a value that
                // is already the immutable rpy_string in the lifted model, so
                // alias the destination to the receiver instead of emitting an
                // unregistered clone/to_owned/to_string.
                if args.len() == 1
                    && (self.is_to_string_identity(&reg, first_arg_ty.as_ref())
                        || self.is_string_clone_identity(&reg, first_arg_ty.as_ref())
                        || self.is_copy_scalar_or_thin_ptr_clone(&reg, first_arg_ty.as_ref()))
                    // A `Wtf8::to_wtf8_buf` that defines a proven mutable
                    // accumulator is not the ordinary immutable-string copy
                    // this identity arm models.  Its later `push*` calls and
                    // single materialisation spell RPython's
                    // `StringBuilder`; let the builder-ctor arm below emit
                    // `newstringbuilder` plus the seed append.  Otherwise the
                    // identity shortcut binds the accumulator to the source
                    // `SomeString`, and the terminal `build` asks that string
                    // for a builder-only method.
                    && !(self.builder_mode
                        && str_builder_ctor_leaf(self.llbc, &reg).is_some()
                        && self.accumulator_is_builder_mode(dest_local))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Trait `Clone::clone` of `Vec<T>` whose T-clone is
                // identity (`rlist.py` `ll_copy` / `ll_arraycopy`), or a
                // tuple of Copy and such `Vec` fields, specialized on the
                // receiver before the call becomes residual
                // `["Clone","clone"]`.
                if args.len() == 1
                    && self.try_lower_trait_clone(
                        mir_bb,
                        &reg,
                        first_arg_ty.as_ref(),
                        &call.dest.ty,
                        dest_local,
                        &args,
                        target,
                    )?
                {
                    return Ok(());
                }
                // `T::default()` for a Copy scalar or thin-pointer `T` is the
                // zero of that kind — an RPython constant, not a residual
                // `core` body.
                if args.is_empty()
                    && self.is_scalar_or_ptr_default(&reg, &call.dest.ty)
                    && let Some(zero) = self.emit_zero_constant_of_ty(bb_id, &call.dest.ty)
                {
                    self.local_var[dest_local] = Some(LocalValue::One(zero));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // A zero-sized std `default` is `lltype.Void`
                // (`ObjectDictStrategy.get_empty_storage`: the `r_dict`
                // hasher has no word). Layout size is the gate; a
                // `Default` with a payload stays a call.
                if args.is_empty() && self.is_void_zst_std_default(&reg, &call.dest.ty) {
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `(*p.add(i)).field` on a named struct pointee is
                // `getinteriorfield` (`rewrite_op_getinteriorfield`). The
                // add records the GcArray header and the index; the field
                // use emits `InteriorFieldRead` / `InteriorFieldWrite`.
                // `sub`, a constant index, and a pointer that escapes stay
                // on the `direct_ptradd` path below.
                if self.try_lower_struct_field_ptr_add(
                    mir_bb,
                    &reg,
                    &args,
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                    target,
                )? {
                    return Ok(());
                }
                // `<(A, B, …) as Default>::default` is `TupleRepr.newtuple`
                // of each element's default: malloc of the tuple GcStruct
                // plus one setfield per item. A residual
                // `core::tuple::<Impl>::default` is not in CallRegistry.
                if args.is_empty()
                    && self.is_tuple_default(&reg, &call.dest.ty)
                    && let Some(tuple) = self.emit_default_of_ty(bb_id, &call.dest.ty)
                {
                    self.local_var[dest_local] = Some(LocalValue::One(tuple));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `CallKind::Trait` `<T as Default>::default` in a copied
                // generic body (a `Clause` rewritten to `TraitImpl`). The
                // impl method is often Opaque on a Struct owner, so
                // `specialized_trait_target` cannot name it; emit the
                // destination's default instead of residual
                // `["Default","default"]`. A Fun Regular path
                // `core::default::Default::default` is the lookalike
                // that stays residual.
                if args.is_empty()
                    && self.is_trait_default_call(&reg)
                    && let Some(value) = self.emit_default_of_ty(bb_id, &call.dest.ty)
                {
                    self.local_var[dest_local] = Some(LocalValue::One(value));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1 && self.is_const_ptr_as_ref(&reg) {
                    self.alias_dest_to_arg0(dest_local, args[0].clone(), true);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<*mut T>::add` / `<*const T>::add` / `::sub` is
                // `lltype.direct_ptradd(ptr, unscaled_count)`.  `sub`
                // negates the count.  The pointer's concretetype carries
                // `TO.OF` so `rewrite_op_direct_ptradd` can scale; a char
                // pointer is `CCHARP` and stays unscaled.  The result is
                // a raw pointer (int kind).
                //
                // Brick-1 accessors and brick-3 getarrayitem `.add`s have
                // their own intercepts later in this match; do not steal
                // those (the items-base collapse aliases to the header
                // because the gcarray descr already folds `base_size`).
                if args.len() == 2
                    && let Some(pointee_size) =
                        self.ptr_add_pointee_size(&reg, first_arg_ty.as_ref())
                    && !self.ptr_add_has_later_intercept(
                        &reg,
                        args.len(),
                        &arg_locals,
                        first_arg_ty.as_ref(),
                        dest_local,
                    )
                {
                    // `(s as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET)` is
                    // `s.chars` (`rstr.py` `STR`) only when `s` is the rstr
                    // header and the add produces a `u8` pointer. Any other
                    // add/sub of that constant is `rtype_direct_ptradd` with
                    // its displacement.
                    if self.is_string_header_chars_byte_ptr_add(
                        &reg,
                        &arg_locals,
                        &arg_operands,
                        first_arg_ty.as_ref(),
                        &call.dest.ty,
                    ) {
                        self.alias_dest_to_arg0(dest_local, args[0].clone(), true);
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                    let offset = if pointee_size == 0 {
                        args[0].clone()
                    } else {
                        let mut count = args[1].clone();
                        if self.ptr_offset_is_sub(&reg) {
                            let neg = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(neg.clone()),
                                kind: OpKind::UnaryOp {
                                    op: "neg".to_string(),
                                    operand: count,
                                    result_ty: ValueType::Int,
                                },
                            });
                            count = neg;
                        }
                        let item = item_lltype_for_size(pointee_size);
                        push_direct_ptradd(&mut self.graph, bb_id, args[0].clone(), count, &item)
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(offset));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::ptr::write` of a Copy scalar or thin pointer is a
                // raw store into the already-allocated object.  An aggregate
                // write is left for `lower_struct_ptr_writes`.
                if args.len() == 2
                    && let Some((item_ty, itemsize, is_item_signed)) =
                        self.ptr_write_store_descr(&reg, second_arg_ty.as_ref())
                {
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
                            base: args[0].clone(),
                            offset,
                            value: args[1].clone(),
                            item_ty,
                            itemsize,
                            is_item_signed,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Option::as_ref` / `Option::<&T>::copied` /
                // `Option::<T>::cloned` on an `Option` receiver are identities
                // in the lifted value model: `T` and `&T` are the same value
                // word and the Option tag is unchanged.  Alias the destination
                // instead of emitting a method call the rtyper cannot route on
                // the foreign Option receiver.
                if args.len() == 1
                    && self.is_option_value_identity(&reg, first_arg_ty.as_ref(), &call.dest.ty)
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `slice::Iter::copied` on a `Copy` element. The bytearray
                // slice assign feeds `Vec<u8>::splice` with
                // `sequence2.iter().copied()`. The adapter's next value is
                // the iterator's element; alias it so the splice sees the
                // same iterator word.
                if args.len() == 1 && self.is_slice_iter_copied(mir_bb, &reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `s.as_bytes().iter().position(|&b| b == c)` is RPython
                // `str.find(c)` (`LLHelpers.ll_find_char` /
                // `rpython_str_find_char`). The residual FunDecl is an
                // empty GraphCache stub; mergeinputargs of its iterator
                // slot unions every callsite.
                if args.len() == 2
                    && self.try_lower_str_view_iter_position(
                        mir_bb,
                        &reg,
                        &args,
                        dest_local,
                        dest_class_ty,
                        target,
                        second_arg_ty.as_ref(),
                    )?
                {
                    return Ok(());
                }
                // Workspace `Index::index` / `IndexMut::index_mut`
                // impls (`FixedObjectArray` and friends) bottom out at
                // raw-slice construction (`as_mut_slice` →
                // `from_raw_parts`), which has no graph lowering.  The
                // callsite is RPython's getarrayitem: lower it as an
                // eager `ArrayRead` for value uses (`x = arr[i]`
                // desugars to `x = *index(&arr, i)` and the `Deref`
                // read collapses to the bound element), and record the
                // `(base, index)` pair so the paired `*p = v` write
                // (`arr[i] = v` desugar) emits `ArrayWrite` from the
                // `emit_projection_write` `Deref` arm.  `Vec<T>::index` /
                // `Vec<T>::index_mut` ([`is_vec_index_call`]) lower
                // identically — the resized-list indirection is supplied
                // downstream by the repr-dispatched `getitem` / `setitem`,
                // not here.
                //
                // `Vec<T>::index_mut` yields a `&mut T`; binding
                // `local_var[dest]` to the element value is sound only when
                // that borrow is consumed by a single in-place deref
                // (`*p` read or `*p = v` write) and never escapes as a
                // pointer.  Unlike the `pyre_`-fenced workspace types, std
                // `Vec::index_mut` matches any `Vec` in the crate, so the
                // mutable leaf is gated on the [`add_dest_used_only_as_single_deref`]
                // escape guard (a failing shape stays residual — safe).  The
                // read leaf (`index`, a value copy) needs no such guard.
                let workspace_index = self.is_workspace_index_call(&reg);
                // `&vec[..]` / `&mut vec[..]` is Rust's whole-list borrowed
                // view (`Index<RangeFull>`). RPython has no borrow wrapper:
                // both sides are the same `SomeList` value.  Preserve that
                // identity before the scalar element arm; Range/RangeFrom/
                // RangeTo remain real getslice operations.
                //
                // `Vec<PyObjectRef>[..]` is [`Self::try_lower_object_vec_whole_list_view`].
                if args.len() == 2 && is_vec_rangefull_index_regular(&reg, self.llbc) {
                    // A `Vec` of raw pointers is an address (`getkind` int).
                    // The slice view is `lltype.cast_int_to_ptr`.
                    // `Vec<PyObjectRef>` already took the whole-list arm above.
                    let object_vec = first_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_object_pointer_rust_vec(ty, self.llbc));
                    if !object_vec
                        && borrow_pointee_is_ptr_slice(&call.dest.ty, self.llbc, self.gc_struct_ids)
                    {
                        let casted = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(casted.clone()),
                            kind: cast_int_to_ptr_call(args[0].clone()),
                        });
                        self.local_var[dest_local] = Some(LocalValue::One(casted));
                    } else {
                        self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    }
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // RPython's UTF-8 carrier is a byte string. A Rust
                // `as_bytes()[i]` over the frontend's string identity is
                // therefore `ord(s[i])`, not a list/array element load and
                // not unicode code-point indexing. Keep it as a synthetic
                // marker until the flowspace adapter can create the Char
                // intermediate required by `ord`.
                if args.len() == 2
                    && (self.graph.name.ends_with("::copied_byte_vec")
                        || first_arg_is_string_byte_view)
                    && self.is_slice_scalar_index_call(&reg, second_arg_ty.as_ref())
                    && !self.is_slice_scalar_index_mut_call(&reg)
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let byte_copy = self.graph.name.ends_with("::copied_byte_vec");
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__string_byte_getitem".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone(), args[1].clone()]),
                            // `ord` returns RPython Signed. The byte-copy
                            // helper stores `u8`, so that graph's read is
                            // unsigned. Rust's `u8` spelling is otherwise
                            // representation detail here.
                            result_ty: if byte_copy {
                                ValueType::Unsigned
                            } else {
                                ValueType::Int
                            },
                        },
                    });
                    // No `index_elem_alias` entry.  That table exists so a
                    // paired `*p = v` becomes an `ArrayWrite` over the same
                    // base, and this base is a `StringRepr` value — upstream
                    // has no `setitem` on one (`rstr.py` offers only
                    // `ll_str2mutable` / `mutable_str` for writes), so the
                    // write half cannot exist.  The arm is already guarded by
                    // `!is_slice_scalar_index_mut_call`, but recording an
                    // alias whose pairing would mint an op RPython does not
                    // have is a latent illegal-op source, and it also feeds
                    // `compute_index_write_extra_live`'s liveness reasoning.
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `as_bytes()[a..b]` is rewritten by `front::slice_index` to
                // `__getslice_*`, a GC-array/list slice — not
                // `_ll_stringslice` (`rstr.py`), which would allocate a
                // new string of length `stop-start`.  Do not mark the dest
                // as a string byte view: `strlen` / string `eq` expect a
                // `W_UnicodeObject`, and this dest is the slice object.
                // `ArrayLen` on the getslice result is the length of
                // `[a..b]`.
                // `ArrayRead` addresses its element as `base + index *
                // itemsize`.  A scalar host element carries its spelling as
                // the array identity so the descr computes the exact width;
                // a thin pointer, or a source value translated to RPython's
                // one-word GC pointer repr, needs only that pointer identity.
                // Any other inline Rust ADT falls through to the residual
                // call: assigning it a target-word stride would be false.
                // The workspace arm asks nothing: its `pyre_` fence admits
                // exactly the three element banks named below, all one word.
                let element_node = tyref_index_element_node(&call.dest.ty, self.llbc);
                let element_scalar =
                    element_node.and_then(|elem| json_ty_scalar_element_spelling(elem, self.llbc));
                // A string-family or tuple element gets NO spelling here, even
                // though `StringRepr.lowleveltype = Ptr(STR)` (`rstr.py`,
                // `StringRepr`) and `TUPLE_TYPE` (`rtuple.py`) really are one
                // GC pointer inside an RPython list.  Those describe the repr
                // of a list the translation BUILDS; a host Rust container
                // that no translation replaces still strides over its real
                // memory.  `&str` is two words there, `String`/`Wtf8Buf`
                // three, and `(Wtf8Buf, PyObjectRef)` four, so naming any of
                // them a one-word item reads the wrong address from the
                // second element on.  A translated `Vec<String>` is that
                // built list (`translated_str_vec_index` below) and mints
                // `STRING_GCREF_GCARRAY_TYPE_ID` without asking here.  The
                // managed `GcArray(GCREF)` items view — the other container
                // that IS one word per item — carries its own proof
                // (`first_arg_is_string_array_view`).  Routing a host
                // container through the same `[str]` identity would also
                // alias it into that array's `_cache_array` slot.
                let element_spelling = element_scalar;
                // A `&[PyObjectRef]` is `Ptr(GcArray(Ptr(PyObject)))` in the
                // translated type model — the block a gateway wrapper's
                // argument array is built as — so it names that ARRAY like
                // every other access to it.  A `Vec` stays a host container.
                let slice_object_element = self
                    .is_slice_scalar_index_call(&reg, second_arg_ty.as_ref())
                    && element_node.is_some_and(|elem| json_ty_is_objectptr(elem, self.llbc));
                // `Option<Box<T>>` (and the other null niches) is one pointer
                // word: `None` is null, `Some` is the payload. Indexing it
                // as a residual `Vec::index_mut` hands a virtualizable array
                // to a call, which `jtransform.py` rejects.
                let element_is_niche_option =
                    element_node.is_some_and(|elem| json_ty_is_niche_option_word(elem, self.llbc));
                let element_is_addressable = element_spelling.is_some()
                    || element_node
                        .is_some_and(|elem| json_ty_is_thin_pointer_element(elem, self.llbc))
                    || element_is_niche_option;
                // A translated `Vec<String>` / `Vec<Wtf8Buf>` is `ListRepr` of
                // `SomeString` (`project_struct_field_type` maps the
                // string-family element to `s_str0`, then `Vec<` to
                // `SomeList`). Host `String` is three words, so it has no
                // `element_spelling` above; the list's items are one GCREF
                // each (`StringRepr`). Scalar `Vec::index` therefore emits
                // the same `ArrayRead` / `getitem` as any other `ListRepr`
                // (`rlist.py` `AbstractBaseListRepr.rtype_getitem` /
                // `ll_getitem_fast`) with `item_ty = Str`, on the list
                // itself — never the host buf pointer.
                let translated_str_vec_index = args.len() == 2
                    && self.is_vec_index_call(&reg, second_arg_ty.as_ref())
                    && tyref_is_string_family(&call.dest.ty, self.llbc);
                let vable_array_var = (args.len() == 2)
                    .then(|| declared_vable_array_var(&self.graph, &args[0]))
                    .flatten();
                // A declared virtualizable array is `setarrayitem_vable` /
                // `getarrayitem_vable`. Leaving `Vec::index_mut` as a call
                // makes the trace abort on a symbolic residual, and the
                // bounds check belongs on the array op as a guard.
                let index_leg = args.len() == 2
                    && (workspace_index
                        || vable_array_var.is_some()
                        || (self.is_vec_index_call(&reg, second_arg_ty.as_ref())
                            && (!is_vec_index_mut_call(
                                &reg,
                                second_arg_ty.as_ref(),
                                self.llbc,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            ) || index_mut_result_is_element(
                                self.llbc, self.body, dest_local,
                            )))
                        || (self.is_slice_scalar_index_call(&reg, second_arg_ty.as_ref())
                            && (!self.is_slice_scalar_index_mut_call(&reg)
                                || add_dest_used_only_as_single_deref(
                                    self.llbc, self.body, dest_local,
                                ))));
                // The element kind and the ARRAY identity are decided above the
                // gate, not at the emit below, because the gate turns on the
                // identity: minting one is one of the two ways the element's
                // width reaches the descr.  One expression has to answer both
                // questions or a later edit desynchronises them, admitting an
                // element whose width the descr never learns.
                //
                // `Index::index(_mut)` returns `&T`/`&mut T`, while RPython's
                // getarrayitem returns `T`.  Preserve that pointee kind:
                // treating the reference wrapper itself as the item makes an
                // integer Vec load flow into `int_mul/ri>i`.
                //
                // The `pyre_`-fenced workspace arm has exactly three element
                // banks — `FixedObjectArray`, `IntArray`, `FloatArray` — so a
                // `*mut PyObject` element there IS the length-prefixed object
                // block that `set_ref` and the `swap` decomposition name, and
                // this read has to share their descr or a cached element
                // survives their store.  An object-pointer slice is that
                // block too (`slice_object_element`).  The `Vec<T>` leg reaches
                // object pointers that are not that block, and it keeps the
                // identity-less descr, which `arraydescrof_concrete` mints
                // locally without a cache publish.
                //
                // Unlike the `swap` arm, the item kind is sound evidence here.
                // The `Ref(None)` fallback that forced `swap` onto a positive
                // `output_type_is_objectptr` proof is reached when
                // `monomorphize:false` leaves a `TypeVar` at the call site; the
                // impls behind this arm are non-generic inherent/trait impls on
                // three concrete types, so `call.dest.ty` is a resolved
                // associated `Output` at every one and never lands on that
                // fallback.
                //
                // Outside that arm the element is whatever the receiver spells,
                // and an int-banked one narrower than 8 bytes needs its
                // identity for `get_type_flag` to be reachable at all
                // ([`narrow_item_array_type_id`]); a `Vec<u8>` read strides by
                // 8 without it.
                let index_element = index_leg.then(|| {
                    if first_arg_is_string_array_view || translated_str_vec_index {
                        return (
                            ValueType::Str,
                            Some(STRING_GCREF_GCARRAY_TYPE_ID.to_string()),
                        );
                    }
                    let item_ty = tyref_deref_value_type_with(
                        &call.dest.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    // The `pyre_`-fenced workspace arm has exactly three
                    // element banks — `FixedObjectArray`, `IntArray`,
                    // `FloatArray` — so a `*mut PyObject` element there IS
                    // the length-prefixed object block that `set_ref` and the
                    // `swap` decomposition name, and this read has to share
                    // their descr or a cached element survives their store.
                    // An object-pointer slice is that block too
                    // (`slice_object_element`).  The `Vec<T>` leg reaches
                    // object pointers that are not that block, and it keeps
                    // the identity-less descr, which `arraydescrof_concrete`
                    // mints locally without a cache publish.
                    //
                    // Unlike the `swap` arm, the item kind is sound evidence
                    // here.  The `Ref(None)` fallback that forced `swap` onto
                    // a positive `output_type_is_objectptr` proof is reached
                    // when `monomorphize:false` leaves a `TypeVar` at the call
                    // site; the impls behind this arm are non-generic
                    // inherent/trait impls on three concrete types, so
                    // `call.dest.ty` is a resolved associated `Output` at
                    // every one and never lands on that fallback.
                    let array_type_id = if workspace_index {
                        matches!(item_ty, ValueType::Ref(_))
                            .then(|| OBJECT_REF_GCARRAY_TYPE_ID.to_string())
                    } else if self.is_vec_index_call(&reg, second_arg_ty.as_ref())
                        && !reaches_declared_vable_array(&self.graph, &args[0])
                    {
                        // The buffer is headerless. `[u8]` is the length-prefixed
                        // byte block, so a Vec of an addressable element keeps
                        // its own identity: element `u8`, `nolength`, cache key
                        // distinct from `[u8]`. A multi-word element has no
                        // width in that string, so the call stays residual.
                        // A declared virtualizable array keeps the length-prefixed
                        // descr the vable protocol already uses.
                        if element_is_addressable || element_spelling.is_some() {
                            first_arg_ty
                                .as_ref()
                                .map(|ty| tyref_to_ast_string(ty, self.llbc))
                                .filter(|identity| !identity.starts_with("??"))
                        } else {
                            None
                        }
                    } else {
                        // `get_array_descr` keys the descr cache on the ARRAY
                        // lltype, so two sites naming the same element name the
                        // same array — which is what lets the width travel.
                        // A thin-pointer element stays unnamed: it is the one
                        // shape the identity-less mint already sizes right.
                        //
                        // The element spelling is the authority wherever it
                        // resolves, so only one name reaches the cache key for
                        // a given element.  Where it does not resolve, an
                        // int-banked receiver can still say what it holds:
                        // `arraydescrof_concrete`'s identity-less arm has no
                        // element name to hand `get_type_flag` and banks every
                        // non-`Ref` element at 8 bytes, so a narrow int read
                        // through such a receiver would stride past its
                        // neighbours.
                        slice_object_element
                            .then(|| {
                                slice_array_type_id("*mut PyObject")
                                    .expect("object-pointer slice names the object gcarray")
                            })
                            .or_else(|| element_spelling.as_deref().and_then(slice_array_type_id))
                            .or_else(|| {
                                if !matches!(
                                    item_ty,
                                    ValueType::Int | ValueType::Unsigned | ValueType::Bool
                                ) {
                                    return None;
                                }
                                first_arg_ty.as_ref().and_then(|ty| {
                                    narrow_item_array_type_id(
                                        ty,
                                        self.llbc,
                                        &self.raw_alloc_flavor_structs,
                                    )
                                })
                            })
                    };
                    (item_ty, array_type_id)
                });
                // `ArrayRead` addresses its element as `base + index *
                // itemsize`, so the op is sound only where that itemsize is the
                // element's real width.  Two disjoint proofs supply it:
                //
                // * the ARRAY identity minted above, which gives
                //   `arraydescrof_concrete` an element name to hand
                //   `get_type_flag` and so stamps the true width — this is the
                //   narrow-int case, `Vec<u8>` striding 1 and `Vec<i16>` 2;
                // * `element_is_addressable`, for an element the identity-less
                //   descr already describes: a scalar names its own spelling,
                //   and a thin pointer IS the one target word that arm assumes.
                //
                // An element neither proof can name falls through to the
                // residual call, where Rust computes the address itself.
                // `ValueType` cannot stand in for either: `Unsigned` is one
                // width-less variant spanning `u8` through `usize`, and every
                // non-primitive shape flattens to `Ref(None)`, so a `Vec<u8>`
                // and a `Vec<String>` are indistinguishable there while
                // striding 1 and 24 bytes.  The workspace arm needs no proof:
                // its `pyre_` fence admits exactly the three element banks
                // named above, all of them one word.
                if let Some((item_ty, array_type_id)) = index_element
                    && (workspace_index || array_type_id.is_some() || element_is_addressable)
                {
                    let mut array_type_id = array_type_id;
                    // `Box<[u8]>[i]` addresses the data word. The `[u8]`
                    // identity is length-prefixed; the fat box is not.
                    // A list view of a W2 owner stays the list
                    // (`rlist.py` `ll_getitem_nonneg`).
                    let fat_data = if first_arg_is_list_view {
                        None
                    } else {
                        self.fat_component(bb_id, &args[0], crate::model::VecFieldPart::FatData)
                    };
                    if fat_data.is_some() {
                        let receiver = first_arg_ty
                            .as_ref()
                            .map(|ty| tyref_to_ast_string(ty, self.llbc));
                        let remapped = element_spelling
                            .as_deref()
                            .and_then(crate::front::typestr::fat_box_scalar_array_id)
                            .or_else(|| {
                                array_type_id
                                    .as_deref()
                                    .and_then(crate::front::typestr::fat_box_scalar_array_id)
                            })
                            .or_else(|| {
                                receiver
                                    .as_deref()
                                    .and_then(crate::front::typestr::fat_box_scalar_array_id)
                            });
                        if let Some(id) = remapped {
                            array_type_id = Some(id.to_string());
                        }
                    }
                    // `&frame.items[i]` is a bounds-checked index (`nolength:
                    // false` below). When `items` is a declared virtualizable
                    // array, the address mark would keep the field read out of
                    // `vable_array_vars` and the index would stay a plain
                    // `getarrayitem_gc`. A fat box is not that array.
                    let vable_array = if fat_data.is_some() {
                        false
                    } else if let Some(root) = &vable_array_var {
                        self.release_declared_vable_array_address(root)
                    } else {
                        self.release_declared_vable_array_address(&args[0])
                    };
                    // Name the field-read variable. A later deref is not the
                    // key `vable_array_vars` stores, so `setarrayitem` would
                    // miss the virtualizable rewrite and write the header.
                    // A `Vec` of one-word items has a raw item buffer.  An
                    // inline field keeps the `(base, field)` read of its
                    // buffer word, now an address (`Int`); any other receiver
                    // is the header address, read and written through
                    // `ll_vec_getitem_fast` / `ll_vec_setitem_fast`.
                    let plain_vec_index = vable_array_var.is_none()
                        && !vable_array
                        && self.is_vec_index_call(&reg, second_arg_ty.as_ref())
                        && !reaches_declared_vable_array(&self.graph, &args[0])
                        && !translated_str_vec_index;
                    let rust_vec_kind = plain_vec_index
                        .then(|| {
                            first_arg_ty
                                .as_ref()
                                .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc))
                        })
                        .flatten();
                    let rust_vec_field_buf = rust_vec_kind.and_then(|_| {
                        inline_vec_field_part(
                            &mut self.graph,
                            bb_id,
                            &args[0],
                            crate::model::VecFieldPart::Buf,
                            ValueType::Int,
                        )
                    });
                    let rust_vec_header = rust_vec_kind.filter(|_| rust_vec_field_buf.is_none());
                    let array_base = if let Some(data) = fat_data {
                        data
                    } else if let Some(root) = &vable_array_var {
                        root.clone()
                    } else if let Some(buf) = rust_vec_field_buf {
                        buf
                    } else if rust_vec_header.is_some() {
                        args[0].clone()
                    } else if plain_vec_index {
                        self.retarget_vec_part(bb_id, &args[0], crate::model::VecFieldPart::Buf)
                    } else {
                        args[0].clone()
                    };
                    // `BytesArray::index` / `UnicodeArray::index` are
                    // `&self.as_slice()[i]`. Reading the `{ block, len }`
                    // record is `getitem_SomeInstance`. `ll_items` reads
                    // `block`; the items are `GcArray(GCREF)`.
                    let string_index_owner = string_array_index_owner(&reg, self.llbc);
                    // A trait-associated `Index::Output` can stay a TypeVar
                    // in this call's destination even though the workspace
                    // gate has resolved the concrete receiver. In that case
                    // `result_narrow_root` cannot recover the pointee name,
                    // but the canonical object-array identity is already the
                    // positive proof that the element is `PyObjectRef` (the
                    // int/float workspaces have non-Ref banks). A string
                    // index's element is already `Str`; casting it to a
                    // class drops that.
                    let item_narrow_root = if string_index_owner.is_some() {
                        None
                    } else {
                        result_narrow_root.clone().or_else(|| {
                            (array_type_id.as_deref() == Some(OBJECT_REF_GCARRAY_TYPE_ID))
                                .then(|| "PyObject".to_string())
                        })
                    };
                    // `ll_setitem_fast` stores the item and does not read the
                    // old one. A write-only `index_mut` emits no `list.getitem`
                    // / `ArrayRead` and no element constant. The alias still
                    // names the base and index for the later
                    // `ll_vec_setitem_fast`. The fresh Variable keeps
                    // `local_var` bound so `mem_slot` answers `MemSlot::Index`;
                    // it is not the loaded item, and `edge_args` does not pass
                    // it.
                    let write_only = rust_vec_header.is_some()
                        && add_dest_used_only_as_single_write_deref(
                            self.llbc, self.body, dest_local,
                        );
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let element_is_pointer_word = string_index_owner.is_none()
                        && self.var_tracks_pointer_word_array(&args[0]);
                    if !write_only && (element_is_pointer_word || element_is_niche_option) {
                        self.pointer_word_vars.insert(res.clone());
                    }
                    let item_ty = if element_is_pointer_word || element_is_niche_option {
                        ValueType::Ref(None)
                    } else {
                        item_ty
                    };
                    let (array_base, item_ty, array_type_id) =
                        if let Some(owner) = string_index_owner {
                            let block =
                                self.emit_string_array_items_read(bb_id, args[0].clone(), owner);
                            let view = self
                                .narrow_value_to_instance_root(
                                    bb_id,
                                    LinkArg::Value(block),
                                    STRING_GCREF_GCARRAY_TYPE_ID,
                                )
                                .as_variable()
                                .expect("a string-items view stays a Variable")
                                .clone();
                            (
                                view,
                                ValueType::Str,
                                Some(STRING_GCREF_GCARRAY_TYPE_ID.to_string()),
                            )
                        } else {
                            (array_base, item_ty, array_type_id)
                        };
                    if !write_only {
                        let read = match rust_vec_header {
                            Some(kind) => OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: rust_vec_helper_segments(
                                        majit_ir::rvec::VecOp::GetItem,
                                        kind,
                                    ),
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![
                                    array_base.clone(),
                                    args[1].clone(),
                                ]),
                                result_ty: item_ty.clone(),
                            },
                            None => OpKind::ArrayRead {
                                base: array_base.clone(),
                                index: args[1].clone(),
                                item_ty: item_ty.clone(),
                                array_type_id: array_type_id.clone(),
                                nolength: crate::front::typestr::nolength_from_array_type_id(
                                    array_type_id.as_deref(),
                                ),
                                gckind: crate::front::typestr::gckind_from_array_type_id(
                                    array_type_id.as_deref(),
                                ),
                                pure: false,
                            },
                        };
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind: read,
                        });
                    }
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: array_base,
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                            item_ty,
                            array_type_id,
                            rust_vec: rust_vec_header,
                            list_root: None,
                            items_owner: string_index_owner,
                            no_element: write_only,
                        },
                    );
                    // This intercept returns before the generic call-result
                    // tail below.  Repeat its registered-pointee narrow so
                    // `rlist.py:recast` exposes the concrete W_Root-equivalent
                    // repr after the GCREF-backed getarrayitem.
                    if write_only {
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                        self.drop_unvalued_index_local(dest_local);
                    } else if let Some(root) = item_narrow_root {
                        let narrowed = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(narrowed.clone()),
                            kind: crate::model::cast_instance_call(root, res),
                        });
                        self.local_var[dest_local] = Some(LocalValue::One(narrowed));
                    } else {
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                    }
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `BytesArray::remove` / `UnicodeArray::remove` spell
                // `rlist.ll_delitem_nonneg`'s overlap-safe
                // `ll_arraymove(l, index + 1, index, newlength - index)` as
                // raw `ptr::copy(src, dst, count)`.  Recover the logical
                // GcArray plus those three indices here and route the call to
                // the exact RPython helper marker.  The match is both
                // graph-scoped and callee-exact; an arbitrary raw copy keeps
                // its Rust semantics.
                if args.len() == 3 && self.is_string_array_remove_ptr_copy(&reg) {
                    let owner = self
                        .string_array_remove_owner()
                        .expect("ptr-copy gate proved a string-array remove owner");
                    let list = self
                        .local_var
                        .get(1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: string-array remove lost self before ll_arraymove"
                            ))
                        })?
                        .one()?;
                    let index = self
                        .local_var
                        .get(2)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "bb{mir_bb}: string-array remove lost index before ll_arraymove"
                            ))
                        })?
                        .one()?;
                    let block = self.emit_string_array_items_read(bb_id, list, owner);
                    let array = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            LinkArg::Value(block),
                            STRING_GCREF_GCARRAY_TYPE_ID,
                        )
                        .as_variable()
                        .expect("a string-items view stays a Variable")
                        .clone();
                    let one = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(one.clone()),
                        kind: OpKind::ConstUInt(1),
                    });
                    let source_start = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(source_start.clone()),
                        kind: OpKind::BinOp {
                            op: "add".to_string(),
                            lhs: index.clone(),
                            rhs: one,
                            result_ty: ValueType::Unsigned,
                        },
                    });
                    let result = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(result.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    crate::runtime_names::shims::LL_ARRAYMOVE.to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![
                                array,
                                source_start,
                                index,
                                args[2].clone(),
                            ]),
                            result_ty: ValueType::Void,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(result));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Every raw pointer offset in those same adapters feeds only
                // the `ptr::copy` or the final null clear reconstructed above
                // and below.  Once the logical indices are recovered, the
                // interior pointer has no translated value of its own; alias
                // it to its base so no Rust pointer arithmetic leaks into the
                // RPython graph.
                if args.len() == 2
                    && self.string_array_remove_owner().is_some()
                    && regular_call_is_ptr_add(&reg, self.llbc)
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `*items_block_items_base(block).add(idx)` — a list /
                // tuple element load or store through an `ItemsBlock`
                // items pointer.  The same getarrayitem / setarrayitem
                // decomposition as the workspace `Index` arm above, but
                // reached through a raw `.add` whose base traces to the
                // header-returning accessor (brick 1) rather than
                // `Index::index`: emit an `ArrayRead` for `x = *p` (the
                // `Deref` read collapses to the bound element) and record
                // the `(base, index)` pair so a paired `*p = v` write
                // emits `ArrayWrite` from `emit_projection_write`'s
                // `Deref` arm.  The gate ([`is_list_items_elem_ptr_add`])
                // requires the `.add` result be dereferenced exactly once
                // and never escape as a raw pointer, so the residual
                // pointer-walking callers (`object_insert` / `_remove` /
                // `_splice`) keep their `add` and fall to the legacy
                // walker.
                // `BytesArray::base` / `UnicodeArray::base` address the same
                // physical `ItemsBlock` as the object arm below, but PyPy's
                // `BytesListStrategy` and `AsciiListStrategy` expose an
                // RPython `str` item.  `rmodel.externalvsinternal` stores that
                // GC pointer as GCREF; it does not erase the ListDef's
                // external SomeString annotation.  Recover `[str]` here so
                // getitem/setitem use StringRepr at the graph boundary and
                // GCRefRepr only inside the ARRAY.
                if let Some((item_ty, array_type_id)) = self.string_items_elem_ptr_add(
                    &reg,
                    args.len(),
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                ) {
                    let base = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            args[0].clone().into(),
                            &array_type_id,
                        )
                        .as_variable()
                        .expect("a materialized string-items base stays a Variable")
                        .clone();
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: args[1].clone(),
                            item_ty: item_ty.clone(),
                            array_type_id: Some(array_type_id.clone()),
                            nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            gckind: crate::front::typestr::gckind_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            pure: false,
                        },
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: base,
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                            item_ty,
                            array_type_id: Some(array_type_id),
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
                    return Ok(());
                }
                if self.is_list_items_elem_ptr_add(
                    &reg,
                    args.len(),
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                ) {
                    // The accessor returns a pointer to the element pointer,
                    // null when the block is null. That word is a classdef-less
                    // instance, and `getitem` on it rewrites to
                    // `getattr("__getitem__")` (`getitem_SomeInstance`). The
                    // descr key stays `OBJECT_REF_GCARRAY_TYPE_ID`; the list
                    // spelling is the element class, which
                    // `project_struct_field_type` models as `SomeList`
                    // (`ll_getitem_fast`, `rlist.py`).
                    let list_root = first_arg_ty
                        .as_ref()
                        .and_then(|ty| object_ref_items_list_root(ty, self.llbc));
                    let base = match list_root.as_deref() {
                        Some(root) => self
                            .narrow_value_to_instance_root(bb_id, args[0].clone().into(), root)
                            .as_variable()
                            .expect("a materialized object-items base stays a Variable")
                            .clone(),
                        None => args[0].clone(),
                    };
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: args[1].clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                            pure: false,
                        },
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: base,
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            rust_vec: None,
                            list_root,
                            items_owner: None,
                            no_element: false,
                        },
                    );
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Scalar twin of the object-items arm above:
                // `*typed_items_block_items_base(block).cast::<i64|f64>()
                // .add(idx)` is RPython's getarrayitem on
                // `GcArray(Signed|Float)`. The accessor has already collapsed
                // its interior pointer to the block header, so use the typed
                // array identity to restore both header offset and item bank.
                // The shared escape gate proves the `.add` result is consumed
                // by exactly one dereference and never used as a raw pointer.
                if let Some((item_ty, array_type_id)) = self.typed_items_elem_ptr_add(
                    &reg,
                    args.len(),
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                ) {
                    let base = self
                        .narrow_value_to_instance_root(
                            bb_id,
                            args[0].clone().into(),
                            &array_type_id,
                        )
                        .as_variable()
                        .expect("a materialized typed-items base stays a Variable")
                        .clone();
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: args[1].clone(),
                            item_ty: item_ty.clone(),
                            array_type_id: Some(array_type_id.clone()),
                            nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            gckind: crate::front::typestr::gckind_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            pure: false,
                        },
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: base,
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                            item_ty,
                            array_type_id: Some(array_type_id),
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
                    return Ok(());
                }
                // `*slice.as_ptr().add(i)` on a shared `&[u8]` is a
                // one-byte `raw_load` of the window (`CCHARP`); on
                // `&mut [u8]` it is `getarrayitem` on the list of
                // unsigned. A W1 `as_bytes()` view keeps
                // `__string_byte_getitem`.
                if let Some((item_ty, array_type_id)) = self.u8_slice_elem_ptr_add(
                    &reg,
                    args.len(),
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                ) {
                    let window_load = first_arg_ty
                        .as_ref()
                        .is_some_and(|ty| raw_ptr_u8_is_const(ty, self.llbc));
                    // A `u8` element is `CHAR` (`Signed`). `Unknown` banks
                    // as `'r'` and `rewrite_op_raw_load` wraps the byte.
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    let kind = if first_arg_is_string_byte_view {
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__string_byte_getitem".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone(), args[1].clone()]),
                            result_ty: ValueType::Int,
                        }
                    } else if window_load {
                        OpKind::RawLoad {
                            base: args[0].clone(),
                            offset: args[1].clone(),
                            item_ty: ValueType::Unsigned,
                            itemsize: 1,
                            is_item_signed: false,
                        }
                    } else {
                        OpKind::ArrayRead {
                            base: args[0].clone(),
                            index: args[1].clone(),
                            item_ty: item_ty.clone(),
                            array_type_id: Some(array_type_id.clone()),
                            nolength: crate::front::typestr::nolength_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            gckind: crate::front::typestr::gckind_from_array_type_id(Some(
                                array_type_id.as_str(),
                            )),
                            pure: false,
                        }
                    };
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind,
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: args[0].clone(),
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                            item_ty,
                            array_type_id: Some(array_type_id),
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
                    return Ok(());
                }
                // `FixedObjectArray::set_ref(self, index, value)` — the
                // length-prefixed array's GC-published element store.  The
                // method body hand-rolls the write-barrier dance
                // (`pin_root` / `try_gc_write_barrier` / reload-from-shadow-
                // stack) because Rust has no GC-transform pass; in the
                // lifted trace that is exactly RPython's single
                // `setarrayitem_gc` (`ll_setitem_fast`, `rlist.py`),
                // whose conditional write barrier is re-inserted by the
                // backend rewrite (`handle_write_barrier_setarrayitem`,
                // rewrite.py), not by a traced source call.  Collapse
                // the whole call to one `ArrayWrite` over the receiver
                // (a `FixedSizeListRepr`, `ll_fixed_items(l) = l`,
                // rlist.py) so the barrier-body accessors
                // (`items_mut_ptr`, `try_gc_owns_object`) never reach the
                // annotator.  The call returns `()`; its dead destination
                // binds to a defined unit constant.
                if args.len() == 3 && self.is_object_array_set_ref_call(&reg) {
                    // `FixedObjectArray._items` is `[PyObjectRef; 0]`, the
                    // Rust storage spelling of PyPy's fixed `[W_Root]` list.
                    // Preserve that external item annotation before the
                    // setitem generalizes its ListDef.  The internal list
                    // repr remains GCREF and rlist's recast lowers this marker
                    // to the same `cast_opaque_ptr` PyPy emits.
                    let value = self.narrow_value_to_instance_root(
                        bb_id,
                        args[2].clone().into(),
                        "PyObject",
                    );
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::ArrayWrite {
                            base: args[0].clone(),
                            index: args[1].clone(),
                            value,
                            item_ty: ValueType::Ref(None),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `items_block_set_ref(block, index, value)` is
                // `ll_setitem_fast` (`rlist.py`) on the object-strategy
                // items list. `W_ListObject.items` is the erased
                // `lstorage` (`rerased.py`); the object strategy
                // `unerase`s it at the point of use (`listobject.py`
                // `ListStrategy.unerase`) to `GcArray(OBJECTPTR)`. The
                // list root is `[T]` from the stored value's `*mut T`
                // (`object_stored_ptr_items_list_root`); the stored
                // value is narrowed to `T`
                // (`object_stored_ptr_class_root`). String-item
                // owners skip the base unerase
                // (`graph_is_string_items_owner`). The header-flag
                // test and the barrier call in its body are the GC
                // transform's work; the backend rewrite re-inserts
                // the conditional barrier in front of the store
                // (`handle_write_barrier_setarrayitem`). Collapse
                // the call to the one `ArrayWrite` over the unerased
                // block. A value operand whose type does not resolve
                // leaves the call in place.
                if args.len() == 3 && self.is_items_block_set_ref_call(&reg) {
                    let string_owner = graph_is_string_items_owner(&self.graph.name);
                    let const_value_ty = arg_operands
                        .get(2)
                        .and_then(|op| const_operand_tyref(op, self.llbc));
                    let value_ty = third_arg_ty.as_ref().or(const_value_ty.as_ref());
                    let value_root =
                        value_ty.and_then(|ty| object_stored_ptr_class_root(ty, self.llbc));
                    let list_root = if string_owner {
                        None
                    } else {
                        value_ty.and_then(|ty| object_stored_ptr_items_list_root(ty, self.llbc))
                    };
                    if let Some(value_root) = value_root.as_deref() {
                        let base = match list_root.as_deref() {
                            Some(root) => self.unerase_object_strategy_items_block(
                                bb_id,
                                args[0].clone(),
                                root,
                            ),
                            None => args[0].clone(),
                        };
                        let value = self.narrow_value_to_instance_root(
                            bb_id,
                            args[2].clone().into(),
                            value_root,
                        );
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: None,
                            kind: OpKind::ArrayWrite {
                                base,
                                index: args[1].clone(),
                                value,
                                item_ty: ValueType::Ref(None),
                                array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                                nolength: false,
                                gckind: crate::model::ArrayGcKind::Gc,
                            },
                        });
                        self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                }
                // Source-level write-barrier helpers guard the `setfield` the
                // caller performs next (`type_write_barrier`).
                // Jitcode is produced before the GC transform, so it carries
                // no barrier: the backend rewrite inserts `COND_CALL_GC_WB`
                // in front of that `setfield_gc` (`handle_write_barrier_setfield`).
                // The call returns `()`; its destination binds to a unit
                // constant.
                if args.len() == 1 && self.is_source_write_barrier_call(&reg) {
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::swap(s, a, b)` over a `&mut [T]` whose base is
                // the same `FixedObjectArray` shape the workspace index
                // path reads.  Lower to the getarrayitem/setarrayitem
                // decomposition: read both elements, then write each
                // back to the other's index.  The base operand feeds the
                // synthetic `ArrayWrite`s the MIR never spells, but every
                // arg here is live in the call block (no cross-block
                // rebind like the deferred `index_mut` write), so no
                // extra liveness threading is needed.  The call returns
                // `()`; its destination binds to a defined unit constant.
                if args.len() == 3 && self.is_slice_swap_call(&reg) {
                    let base = args[0].clone();
                    let idx_a = args[1].clone();
                    let idx_b = args[2].clone();
                    // The four synthetic ops below name the SLICE's own item
                    // kind and array identity.  `swap` is matched by name
                    // alone, and the name reaches three call sites: the
                    // generic `listsort::sort_with<T>` (through
                    // `reverse_slice`), the `array` module's `Vec<u8>` byte
                    // buffer, and `FixedObjectArray::swap`'s
                    // `&mut [PyObjectRef]`.  Only the last is the
                    // length-prefixed object block, whose identity every
                    // other devirtualized accessor arm spells explicitly;
                    // leaving it unnamed keeps the swap out of that descr's
                    // key, so a trace could cache an element through one arm
                    // and miss the invalidation this store owes it.
                    //
                    // The identity is gated on `output_type_is_objectptr`, a
                    // POSITIVE proof that the element is `*mut PyObject`, not
                    // on the item kind coming out `Ref` — see
                    // [`Self::slice_swap_elem_tyref`] for why the `Ref(None)`
                    // fallback cannot stand in for that proof.
                    let elem_tyref = self.slice_swap_elem_tyref(&reg);
                    let elem_ty = elem_tyref
                        .as_ref()
                        .map(|ty| {
                            tyref_to_value_type_with(
                                ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            )
                        })
                        .unwrap_or(ValueType::Ref(None));
                    let elem_array_type_id = elem_tyref
                        .as_ref()
                        .is_some_and(|ty| output_type_is_objectptr(ty, self.llbc))
                        .then(|| OBJECT_REF_GCARRAY_TYPE_ID.to_string());
                    let elem_a = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let elem_b = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(elem_a.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: idx_a.clone(),
                            item_ty: elem_ty.clone(),
                            array_type_id: elem_array_type_id.clone(),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                            pure: false,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(elem_b.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: idx_b.clone(),
                            item_ty: elem_ty.clone(),
                            array_type_id: elem_array_type_id.clone(),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                            pure: false,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::ArrayWrite {
                            base: base.clone(),
                            index: idx_a,
                            value: LinkArg::Value(elem_b),
                            item_ty: elem_ty.clone(),
                            array_type_id: elem_array_type_id.clone(),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::ArrayWrite {
                            base,
                            index: idx_b,
                            value: LinkArg::Value(elem_a),
                            item_ty: elem_ty.clone(),
                            array_type_id: elem_array_type_id.clone(),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `rstring.py` `StringBuilder(size)`, `.append(s)` and
                // `.build()` are `SomeStringBuilder` operations
                // (`rtyper/rbuilder.py` `AbstractStringBuilderRepr`), wherever
                // the builder lives: a local or an instance attribute such as
                // `rutf8.py` `Utf8StringBuilder._s`.
                if let Some(method) = string_builder_method_leaf(self.llbc, &reg) {
                    let (marker, result_ty, concrete) = match method {
                        "new" => (
                            crate::runtime_names::shims::STRINGBUILDER_NEW,
                            ValueType::Ref(None),
                            crate::model::ConcreteType::Unknown,
                        ),
                        "append" => (
                            crate::runtime_names::shims::STRINGBUILDER_APPEND,
                            ValueType::Void,
                            crate::model::ConcreteType::Void,
                        ),
                        _ => (
                            crate::runtime_names::shims::STRINGBUILDER_BUILD,
                            ValueType::Ref(None),
                            crate::model::ConcreteType::Unknown,
                        ),
                    };
                    let res = self.graph.alloc_value_var_with_type(concrete);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![marker.to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(args.clone()),
                            result_ty,
                        },
                    });
                    let stored = if method == "append" {
                        self.emit_unit(bb_id)
                    } else {
                        res
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(stored));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Builder-mode ctor. In the canonical builder-form lowering, the
                // single definition of a builder-mode accumulator is a
                // constructor call ([`single_def_is_str_builder_ctor`],
                // required on top of the owned single-def rule in
                // [`is_fresh_str_builder`]).  That call mints the
                // `StringBuilder` once via the `__majit_stringbuilder_new`
                // marker instead of the concat empty string; later appends
                // mutate it in place.  Placed before the method-specific arms
                // so it preempts any generic ctor lowering. Inert unless
                // [`enable_builder_mode`] selected this canonical form; the
                // accumulator test reads the per-body table.
                //
                // The ctor operands ride the marker verbatim: `new()` carries
                // none and `with_capacity(n)` carries the size `n`
                // ([`str_builder_ctor_leaf`]), so the
                // marker's arg count mirrors `len(hop.args_v)` in
                // `AbstractStringBuilderRepr.rtyper_new` (rtyper/rbuilder.py) —
                // no arg selects `ll_new(INIT_SIZE)`, the size arg threads
                // `ll_new(n)`.
                let builder_ctor_leaf = str_builder_ctor_leaf(self.llbc, &reg);
                if self.builder_mode && self.accumulator_is_builder_mode(dest_local) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    crate::runtime_names::shims::STRINGBUILDER_NEW.to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: if builder_ctor_leaf == Some("with_capacity") {
                                crate::model::call_args(args.clone())
                            } else {
                                Vec::new()
                            },
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    // `Wtf8Buf::from_string(initial)` and
                    // `Wtf8::to_wtf8_buf(&initial)` are the Rust ownership
                    // spellings of an RPython builder followed by its first
                    // append.  Keep the initial value as an append operand;
                    // passing it to `new` would misread it as the optional
                    // integer capacity argument.
                    //
                    // So the builder is born at `INIT_SIZE`, not at
                    // `len(initial)`.  That is the size RPython source with
                    // this shape gets: `b = StringBuilder(); b.append(s)`
                    // reaches `rtyper_new` with `len(hop.args_v) == 0`
                    // (`rtyper/rbuilder.py`).  `convert_const`'s
                    // `ll_new(len(s))` (`:57-63`) is the PREBUILT-constant
                    // path — a builder already built at translation time —
                    // and is not this callsite.
                    if matches!(builder_ctor_leaf, Some("from_string" | "to_wtf8_buf")) {
                        let void = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(void),
                            kind: OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec!["__majit_stringbuilder_append".to_string()],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![res.clone(), args[0].clone()]),
                                result_ty: ValueType::Void,
                            },
                        });
                    }
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `String::push(&mut buf, ch)`, `Wtf8Buf` /
                // `String`::push_str(&mut buf, s), or push_wtf8(&mut buf, w) —
                // and the argument-swapped `Wtf8Piece::push_onto(&s, &mut
                // buf)` — an accumulator append.  A `Wtf8Buf` / `String`
                // lifts to the single immutable `ValueType::Str`, so there is
                // no builder to mutate in place; model the append functionally
                // as `buf = ll_strconcat(buf, piece)` (the `add` BinOp the
                // rtyper routes to `ll_strconcat`) and rebind the accumulator's
                // MIR local so its later reads — and the loop-header phi that
                // threads it across the back-edge — observe the concatenated
                // string.  `str_builder_append_args` gives the accumulator and
                // piece arg positions (push / push_str / push_wtf8: acc 0, piece 1;
                // push_onto: acc 1, piece 0).  The `&mut buf` accumulator
                // arrives as a per-site ref-temp; the per-body table answers
                // [`append_accumulator_of_arg_temp`] for it, resolving
                // to `buf` only when `buf` is an owned single-def `Wtf8Buf` /
                // `String` (not local 0, not a parameter) whose every borrow
                // feeds such an append, so a `&mut`-parameter accumulator
                // (whose caller observes the real-buffer mutation) is excluded
                // and keeps its residual rather than miscompiling.  The call
                // returns `()`; its dead destination binds to a defined unit
                // constant.
                if args.len() == 2
                    && let Some((acc_i, piece_i)) = str_builder_append_args(self.llbc, &reg)
                    && let Some(buf_local) = arg_locals
                        .get(acc_i)
                        .copied()
                        .flatten()
                        .and_then(|rt| self.accumulator_of_append_arg(rt))
                {
                    // Concatenate onto the accumulator's current binding
                    // (`local_var[buf_local]`), not `args[acc_i]`: the latter is
                    // the value the `&mut buf` borrow resolved to, which an
                    // earlier append in the same block — or the loop-header phi —
                    // may already have superseded.  Fall back to the borrow value
                    // only if the slot is somehow unbound.
                    let acc_val = self.local_var[buf_local]
                        .as_ref()
                        .and_then(|value| value.one().ok())
                        .unwrap_or_else(|| args[acc_i].clone());
                    let piece_val = if self.builder_mode
                        && let Some(piece_temp) = arg_locals.get(piece_i).copied().flatten()
                        && let Some(piece_builder) =
                            self.piece_accumulator_of_append_arg(piece_temp)
                    {
                        self.resolve_builder_build(mir_bb, piece_builder)?
                    } else if str_builder_ctor_leaf(self.llbc, &reg) == Some("push") {
                        // `String::push(c)` / `Wtf8Buf::push(c)`: `c` is a
                        // code point (`rutf8.py unichr_as_utf8`), not a
                        // leftover SomeInteger append piece.
                        self.emit_codepoint_as_utf8(
                            mir_bb,
                            &args[piece_i],
                            arg_operands.get(piece_i),
                        )?
                    } else {
                        args[piece_i].clone()
                    };
                    if self.builder_mode && self.accumulator_is_builder_mode(buf_local) {
                        // Builder form: `b.append(piece)` mutates the buffer in
                        // place and returns `()`, so — unlike the concat rebind
                        // below — the accumulator binding is unchanged.  Emit the
                        // void `__majit_stringbuilder_append` marker
                        // (`flowspace_adapter` → getattr("append") + simple_call).
                        let void = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(void),
                            kind: OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec![
                                        crate::runtime_names::shims::STRINGBUILDER_APPEND
                                            .to_string(),
                                    ],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![acc_val, piece_val]),
                                result_ty: ValueType::Void,
                            },
                        });
                    } else {
                        let concat = emit_str_add(&mut self.graph, bb_id, &acc_val, &piece_val);
                        self.local_var[buf_local] = Some(LocalValue::One(concat));
                    }
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `*const T::cast_mut()` / `*mut T::cast_const()` /
                // `<ptr>::cast()` — address-preserving pointer
                // reinterprets the JIT models as identity (the
                // receiver-method twin of the `CastKind::RawPtr` Rvalue
                // → `same_as`).  Alias the destination to the receiver so
                // the method name never reaches the rtyper as a
                // `ptr.getattr`.
                if args.len() == 1 && self.is_ptr_identity_cast(&reg) {
                    // When the cast target names a registered struct root
                    // (`ptr.cast::<W_SRE_Pattern>()` then a field read),
                    // narrow the result to `SomeInstance(root)` exactly
                    // like the `Rvalue::Cast` arm: a bare alias leaves the
                    // pointer classdef-less and the downstream `getattr`
                    // blocks at the annotator.  The receiver is already a
                    // raw pointer (`is_ptr_identity_cast`), so this is a
                    // genuine `cast_pointer` (ptr→ptr).
                    //
                    // `addr_of_mut!((*header).items).cast::<Entry>()` is
                    // `lltype.py direct_arrayitems`: the address of a
                    // `[T; N]` field, `array_items_base` past the header
                    // (`rordereddict.py get_ll_dict`, `DICTENTRYARRAY`).
                    // The field read is the array value (`SomeList`), and
                    // casting that list to the element root is not a
                    // pointer cast. Recover the header and add the field
                    // offset, then `cast_int_to_ptr`. `direct_ptradd` is
                    // the wrong op here: the header is `SomeInstance`,
                    // `null_mut` joins that as an instance, and
                    // `ann_direct_arrayitems` is still deferred.
                    let dest_is_ref = matches!(
                        tyref_to_value_type_with(
                            &call.dest.ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                        ValueType::Ref(_)
                    );
                    let src_root = first_arg_ty.as_ref().and_then(|ty| {
                        tyref_class_root_with(ty, self.llbc, self.tombstoned_leaves)
                    });
                    if dest_is_ref
                        && let Some(kind) = self.instance_narrow_for_type(
                            &call.dest.ty,
                            &args[0],
                            src_root.as_deref(),
                        )
                    {
                        let array_tail =
                            self.taken_array_field_address(bb_id, &args[0], Some(&call.dest.ty))?;
                        let res = if let Some((base, offset)) = array_tail
                            && let Some(root) = tyref_class_root_with(
                                &call.dest.ty,
                                self.llbc,
                                self.tombstoned_leaves,
                            ) {
                            self.emit_array_tail_item_ptr(bb_id, base, offset, &root)
                        } else {
                            let res = self
                                .graph
                                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(res.clone()),
                                kind,
                            });
                            res
                        };
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                    } else if self.should_emit_rffi_cast_identity_to_raw_scalar(
                        &call.dest.ty,
                        first_arg_ty.as_ref(),
                    ) {
                        // `index_words_ptr` is `typed_items_block_items_base
                        // as *mut i64` and returns that address. The source
                        // is the `*mut u8` Ref erasure. Aliasing keeps
                        // `__cast_address_intrinsic`'s Instance, and the
                        // caller's `direct_ptradd` (`rffi.py` `ptradd`) then
                        // fails `ann_direct_ptradd`. `rffi.cast` onto the
                        // raw `CArrayPtr` is `cast_ptr_to_int` then
                        // `cast_int_to_ptr` (`lltype.py`). An already-int
                        // raw scalar (`HashStateStorage::as_mut_ptr`) stays
                        // in the int bank. `IntArray::base` and
                        // `set_sizehint_state_value` keep the alias
                        // (GcArray header).
                        let res = self.emit_rffi_cast_to_raw_scalar_ptr(
                            bb_id,
                            args[0].clone(),
                            &call.dest.ty,
                        );
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                    } else if self
                        .identity_cast_int_raw_ptr_to_ccharp(&call.dest.ty, first_arg_ty.as_ref())
                    {
                        // `libc::malloc` is `*mut c_void` (`lltype.py`
                        // `ann_malloc` flavor raw, kind Int/Unsigned).
                        // `ptr.cast::<u8>()` / `cast::<CHAR>()` is
                        // `lltype.py` `ann_cast_int_to_ptr` / `rffi.cast`
                        // onto `CCHARP`, the same wrap
                        // `ensure_window_ccharp` /
                        // `emit_window_from_raw_parts_ptr` emit. Other
                        // identity casts stay aliases.
                        let res = emit_cast_int_to_ccharp(&mut self.graph, bb_id, args[0].clone());
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                    } else {
                        self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    }
                    // Dest `*u8` is CCHARP (`rffi.py` `CCHARP`). An int-kind
                    // source is already that word. A Ref receiver
                    // (`p.cast::<u8>()` of `*mut i8`) `rffi.cast`s except
                    // inside an items-base accessor (`cast_ptr_to_adr`).
                    self.bind_raw_u8_dest(mir_bb, dest_local, &call.dest.ty);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `try_gc_current_object_address(p)` is a Rust-interpreter
                // adapter for the root reload RPython inserts after a
                // collecting call (`gc_pop_roots` -> `gc_restore_root`,
                // `memory/gctransform/shadowcolor.py`).  The translated JIT
                // graph already carries `p` as a GC reference whose backend
                // root map is rewritten when an object moves, so tracing the
                // forwarding-stub query would duplicate the GC transform and
                // expose a Rust-only `majit_gc` helper that has no PyPy graph
                // counterpart.  Erase only the adapter's own one-call body;
                // integer/raw-address uses elsewhere must keep their runtime
                // normalization.
                if args.len() == 1 && self.is_gc_current_object_address_adapter(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Atomic*::from_ptr(p)` creates a borrowed atomic view over
                // the same address; it neither allocates nor changes the
                // pointer representation.  The translated object model keeps
                // these slots as their upstream scalar fields (for example
                // unicode `hash`, `rstr.py` `UNICODE`), so alias the view to the
                // field pointer exactly like the raw-pointer casts above.
                if args.len() == 1 && self.is_atomic_from_ptr_identity(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `ONCE.get_or_init(f)` on a lazily-built prebuilt whose refs
                // row carries the built object is `Bookkeeper.immutablevalue`
                // of that object: the host initialized it before the row was
                // captured, so the call is the constant and `f` never runs.
                if args.len() == 2
                    && !self.multi_assigned_locals.contains(&dest_local)
                    && self.is_prebuilt_once_lock_get_or_init(&reg, &args[0])
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    self.prebuilt_once_value_locals.push(dest_local);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Atomic*::new(v)` builds the layout-transparent wrapper over
                // its inner scalar; the translated field is that scalar, so
                // the constructor is the value itself.
                if args.len() == 1 && self.is_atomic_new_identity(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // A Relaxed load is the scalar field value.
                //
                // An Acquire load of a field declared quasi-immutable is
                // that field's read: the codewriter emits
                // `record_quasiimmut_field` and the getfield, and the
                // descr's Acquire bit orders the executable read. SeqCst,
                // Release, an unknown ordering, and an Acquire load of a
                // field that is not quasi-immutable stay unsupported.
                // `&self` may already alias the loaded value, NOT its
                // address, so those cannot fall through to an ordinary
                // `Atomic::load` call.
                // FlowContext.record_block in flowcontext.py propagates an
                // unsupported operation rather than publishing a wrong graph.
                // This rejects the body, not the whole translation: the
                // whole-program builder's pre-existing residual fallback is
                // separate #346 work, not an implementation of atomic loads.
                if args.len() == 2 && self.is_atomic_load(&reg) {
                    let ordering = arg_locals
                        .get(1)
                        .copied()
                        .flatten()
                        .and_then(|local| self.atomic_ordering_locals.get(&local))
                        .map(String::as_str);
                    if ordering == Some("Relaxed") {
                        // A borrow recorded on `atomic_ref_place` already
                        // aliases the scalar field (`STR.hash` is Signed).
                        // `from_ptr` identity leaves the field *address*;
                        // loading that word is `raw_load`, not aliasing the
                        // pointer (Unsigned) onto the Signed memo.
                        if self.bind_dest_to_recorded_atomic_load(
                            mir_bb,
                            dest_local,
                            target,
                            &arg_locals,
                        )? {
                            return Ok(());
                        }
                        if let Some((item_ty, itemsize, is_item_signed)) =
                            self.raw_word_descr(&call.dest.ty)
                        {
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
                                    base: args[0].clone(),
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
                            return Ok(());
                        }
                        self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                    // Quasi-immutable field. The field is read at the load
                    // site through the recorded borrow (`atomic_ref_place`),
                    // so `record_quasiimmut_field` still guards it and the
                    // getfield executes as the Acquire load. A receiver with
                    // no recorded borrow (a copied local, or an
                    // `Atomic*::from_ptr` view) cannot be re-read there, so
                    // it stays unsupported rather than reusing an earlier
                    // read. A borrow whose place goes through a reassigned
                    // local or a pointer read from a field also stays
                    // unsupported.
                    let is_acquire = ordering == Some("Acquire");
                    let ordering_label = ordering.unwrap_or("unknown").to_string();
                    if is_acquire
                        && self.atomic_load_receiver_is_quasi_immutable_field(&args[0])
                        && self.atomic_load_referent_is_stable(&arg_locals)
                        && self.bind_dest_to_recorded_atomic_load(
                            mir_bb,
                            dest_local,
                            target,
                            &arg_locals,
                        )?
                    {
                        return Ok(());
                    }
                    return Err(LowerError::Unsupported(format!(
                        "atomic load ordering {ordering_label} requires address-preserving ordered lowering"
                    )));
                }
                // `<Atomic*>::store(&self, value, ordering)` — the write twin
                // of the load fold above.  A read needs no place, so the load
                // gets away with aliasing the receiver Variable; a write does,
                // so recover the borrowed slot from the `&self` operand
                // ([`Lowering::atomic_ref_place`]) and emit the same
                // `FieldWrite` a plain `slot = value` assignment would.
                //
                // PRECONDITION, and it is a real one: a `FieldWrite` lowers to
                // one plain machine store, which implements a RELAXED atomic
                // store only where that store is single-copy atomic — i.e. the
                // backend emits ONE store for the slot's width.  It does on
                // every target pyre emits for: `mov` / `str` for a naturally
                // aligned word or narrower on x86_64 and aarch64, and `iN.store`
                // for every atomic width on wasm32.  Relaxed buys freedom from
                // tearing and nothing else — no fence, no `lock` prefix — so
                // under that condition the emitted access IS the access the
                // residual call would have performed.  What would break it is a
                // slot the backend has to split: an `AtomicU64` field on a
                // 32-bit NATIVE target, which pyre has none of.  The threshold
                // is deliberately NOT checked here: it is a per-backend store
                // granularity, not a front-end fact, and spelling it as
                // `target_word_size()` would decline `AtomicI64` on wasm32,
                // where `i64.store` is a single instruction.  Adding such a
                // target means teaching the BACKEND, or declining there.
                // The load fold rests on the same precondition.
                //
                // ONLY a `Relaxed` store folds.  A stronger ordering is not
                // decoration: `w_type_set_version_tag` publishes `version_tag`
                // with `Release` against `w_type_get_version_tag`'s `Acquire`
                // reader, and today that store reaches the CALL, which performs
                // the release.  Replacing it with an unordered `FieldWrite`
                // would let a traced class mutation publish the new cache key
                // before the dictionary changes behind it are visible.  So the
                // ordering is read (`atomic_ordering_locals`) and anything but
                // `Relaxed` keeps the residual call it has today.
                if args.len() == 3
                    && self.is_atomic_store(&reg)
                    && arg_locals
                        .get(2)
                        .copied()
                        .flatten()
                        .and_then(|l| self.atomic_ordering_locals.get(&l))
                        .is_some_and(|ordering| ordering == "Relaxed")
                {
                    // A receiver this body did not bind as a borrow of a
                    // field — or one borrowing a whole local rather than a
                    // field of one, or one rebound elsewhere in the body —
                    // leaves no place to write to.  Fall through to the
                    // ordinary call path, which is what an unrecognised
                    // `store` already lowers to, rather than inventing a
                    // target or failing a graph that compiles today.  The
                    // place is copied, not taken: one borrow stored through
                    // twice folds both times.
                    let referent = arg_locals
                        .first()
                        .copied()
                        .flatten()
                        .and_then(|l| self.atomic_ref_place.get(&l))
                        .map(clone_place);
                    if let Some(referent) = referent {
                        let field_ty = clone_tyref(&referent.ty);
                        if let PlaceKind::Projection(inner, elem) = referent.kind {
                            let value = args[1].clone();
                            self.emit_projection_write(
                                mir_bb,
                                *inner,
                                elem,
                                value.into(),
                                &field_ty,
                            )?;
                            // The store's result is an actual unit constant.
                            self.local_var[dest_local] =
                                Some(LocalValue::One(self.emit_unit(bb_id)));
                            let target_bb = self.block_id[target];
                            let link_args = self.edge_args(mir_bb, target)?;
                            self.graph.set_goto(bb_id, target_bb, link_args);
                            return Ok(());
                        }
                    }
                }
                // `(block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET)` inside
                // an `ItemsBlock` items-base accessor — the interior items
                // pointer the runtime reads through a `base_size`-bearing
                // gcarray descr, so the JIT's items base is the header
                // pointer itself.  Alias the destination to the receiver,
                // dropping the unregistered raw-pointer `add`
                // ([`is_items_block_base_ptr_add`] keys on the enclosing
                // accessor so a dereferenced `.add` elsewhere is untouched).
                if args.len() == 2 && self.is_items_block_base_ptr_add(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `bytes_block_chars` is `rstr.py` `ll_chars`: recover the
                // owning `STR` from the address-taken `chars` FieldRead and
                // alias the returned slice to that header. Function-scoped;
                // arbitrary `from_raw_parts(p, n)` cannot discard `n` (a
                // list's logical length may be below capacity).
                if args.len() == 2
                    && let Some(block) =
                        self.bytes_block_header_from_raw_parts(bb_id, &reg, &args[0])
                {
                    self.alias_dest_to_arg0(dest_local, block, true);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `BytesArray::as_slice` / `UnicodeArray::as_slice` are the
                // Rust fat-slice spelling around RPython's `ll_items(l)`.
                // Their caller keeps the separate logical `l.length`; the
                // one-word translated value is therefore the backing
                // `GcArray(GCREF)`, with its external item kept as `str`.
                // Fold only these two adapter bodies: an arbitrary
                // `from_raw_parts(base, logical_len)` cannot discard its
                // second word.
                if args.len() == 2 && self.is_string_array_view_from_raw_parts(&reg) {
                    let view = self.narrow_value_to_instance_root(
                        bb_id,
                        args[0].clone().into(),
                        STRING_GCREF_GCARRAY_TYPE_ID,
                    );
                    self.local_var[dest_local] = Some(LocalValue::One(
                        view.as_variable()
                            .expect("a string-items view stays a Variable")
                            .clone(),
                    ));
                    // The dropped second word IS `l.length`; keep it so a
                    // `Rvalue::Len` over the view answers with it.
                    self.record_string_array_view_len(dest_local, args[1].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Body-side twin of [`Self::numeric_array_slice_view_owner`]:
                // `from_raw_parts(self.base(), self.len())` inside
                // `IntArray` / `FloatArray` `as_slice`. The second word is
                // `l.length` (`ll_length`).
                if args.len() == 2
                    && let Some(root) = self.numeric_array_view_from_raw_parts_root(&reg)
                {
                    let view =
                        self.narrow_value_to_instance_root(bb_id, args[0].clone().into(), root);
                    self.local_var[dest_local] = Some(LocalValue::One(
                        view.as_variable()
                            .expect("a numeric-items view stays a Variable")
                            .clone(),
                    ));
                    self.record_numeric_array_view_len(dest_local, args[1].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `RBigInt::digits{,_mut}` is a Rust view adapter around a
                // typed-items GcArray:
                // `slice::from_raw_parts(items_base(block), capacity)`.  The
                // items-base call above already aliases the header pointer so the
                // gcarray descr re-applies `base_size`; the resulting slice is
                // therefore the same array value in the translated model, and its
                // length is the block capacity the descr reports.  Fold only these
                // capacity-length enclosing accessors, not arbitrary raw slices
                // (an object list's logical length can differ from its items
                // block capacity — see [`is_container_items_view_from_raw_parts`]).
                if args.len() == 2 && self.is_container_items_view_from_raw_parts(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `from_raw_parts((base as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
                // bh_lowlevel_string_len(base))` is the rstr `STR` itself
                // (`rstr.py` `STR`): `chars` is the inline `Array(Char)`,
                // read by `ll_strlen` (`len(s.chars)`) and `strgetitem`
                // (`s.chars[i]`). There is no separate slice object. Bind
                // the destination to that same `base` and record it as a
                // string byte view so `view[i]` / `len(view)` lower to
                // `__string_byte_getitem` / `__strlen`. A different base, a
                // length that is not this string's own length, or a
                // different offset falls through: an arbitrary
                // `from_raw_parts(p, n)` cannot drop `n`. Not keyed on the
                // enclosing function name.
                if args.len() == 2
                    && let (Some(ptr_local), Some(len_local)) = (
                        arg_locals.first().copied().flatten(),
                        arg_locals.get(1).copied().flatten(),
                    )
                    && let Some(base) =
                        str_chars_view_base(self.body, self.llbc, &reg, ptr_local, len_local)
                    && let Some(base_var) = self
                        .local_var
                        .get(base)
                        .and_then(|slot| slot.as_ref())
                        .and_then(|value| value.one().ok())
                {
                    self.alias_dest_to_arg0(dest_local, base_var, true);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `f64::abs(x)` is `float_abs` (`lloperation.py` /
                // `rfloat.rtype_abs`).  Opaque in core, so the callsite
                // would skip as an unregistered FunctionPath.
                if args.len() == 1 && self.is_f64_abs(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "abs".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Float,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `i64::abs(x)` is `int_abs` (`rint.py rtype_abs`,
                // `hop.genop(self.opprefix + 'abs', ...)`).  jtransform
                // routes the int-bank `abs` to `_ll_1_int_abs`.
                if args.len() == 1 && self.is_word_int_abs(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "abs".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Opaque `f64::{hypot,atan2,copysign,floor,ceil,powf,ln,
                // exp,sin,cos,sqrt,log10,atan,tan,asin,acos,…}` are the C
                // llexternals in `ll_math.py` (`math_hypot`, …), listed by
                // `ll_math::f64_method_llexternal`.  The raising wrappers
                // (`ll_math_hypot`, …) stay around them.
                if let Some((arity, leaf)) = self.f64_ll_math_leaf(&reg)
                    && args.len() == arity
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["ll_math".to_string(), leaf.to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(args.iter().cloned()),
                            result_ty: ValueType::Float,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1 && self.is_f64_is_nan(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "ne".to_string(),
                            lhs: args[0].clone(),
                            rhs: args[0].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // ll_math.py ll_math_isfinite / ll_math_isinf (jitted arm):
                // `(x - x) == 0.0` / `(x + VERY_LARGE_FLOAT) == x`.
                // Like isnan above, these opaque Rust primitives become
                // ordinary graph operations, not unresolved host calls.
                let is_infinite = self.is_f64_is_infinite(&reg);
                if args.len() == 1 && (self.is_f64_is_finite(&reg) || is_infinite) {
                    let zero = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(zero.clone()),
                        // Upstream starts at 1.0 and multiplies by 64 until
                        // multiplying by 100 overflows: exactly 2**1020.
                        kind: OpKind::ConstFloat(if is_infinite {
                            (2.0_f64.powi(1020)).to_bits()
                        } else {
                            0
                        }),
                    });
                    let diff = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(diff.clone()),
                        kind: OpKind::BinOp {
                            op: if is_infinite { "add" } else { "sub" }.to_string(),
                            lhs: args[0].clone(),
                            rhs: if is_infinite {
                                zero.clone()
                            } else {
                                args[0].clone()
                            },
                            result_ty: ValueType::Float,
                        },
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: diff.clone(),
                            rhs: if is_infinite {
                                args[0].clone()
                            } else {
                                zero.clone()
                            },
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `f64::from_bits(bits)` is RPython's
                // `longlong2float.longlong2float(bits)`: reinterpret the integer
                // bit pattern as an f64 instead of following the opaque core body.
                if args.len() == 1 && self.is_f64_from_bits(&reg) {
                    return self.emit_longlong2float_call(
                        mir_bb,
                        dest_local,
                        target,
                        args[0].clone(),
                        true,
                    );
                }
                // `f64::to_bits(x)` is `longlong2float.float2longlong(x)`.
                if args.len() == 1 && self.is_f64_to_bits(&reg) {
                    return self.emit_longlong2float_call(
                        mir_bb,
                        dest_local,
                        target,
                        args[0].clone(),
                        false,
                    );
                }
                // Host llops spelled as calls: `pyre_*::longlong2float::*`
                // (`Float2LongLongEntry.specialize_call` /
                // `LongLong2FloatEntry.specialize_call`) and
                // `*::lltype::cast_{ptr_to_int,int_to_ptr}`
                // (`rewrite_op_cast_ptr_to_int`). Exact trailing
                // segments plus banks; a look-alike module or leaf
                // stays residual.
                if args.len() == 1
                    && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                {
                    let path = fd.item_meta.name_path();
                    let src = first_arg_ty.as_ref().map(|ty| {
                        tyref_to_value_type_with(
                            ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    });
                    let dst = tyref_to_value_type_with(
                        &call.dest.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    if let Some(to_float) = host_longlong2float_llop(&path, src.as_ref(), &dst) {
                        return self.emit_float_bytes_llop(
                            mir_bb,
                            dest_local,
                            target,
                            args[0].clone(),
                            to_float,
                        );
                    }
                    if let Some((op, result_ty)) = host_lltype_cast_llop(&path, src.as_ref(), &dst)
                    {
                        if op == "cast_int_to_ptr" {
                            if let Some(raw) = self.emit_dest_overlay_cast(
                                bb_id,
                                src.as_ref(),
                                args[0].clone(),
                                &call.dest.ty,
                            ) {
                                self.local_var[dest_local] = Some(LocalValue::One(raw));
                                let target_bb = self.block_id[target];
                                let link_args = self.edge_args(mir_bb, target)?;
                                self.graph.set_goto(bb_id, target_bb, link_args);
                                return Ok(());
                            }
                            if let Some(orig) = self.cast_ptr_to_int_src.get(&args[0]).cloned() {
                                // Same recast as the `as` identity fold:
                                // `lltype.py` `ann_cast_opaque_ptr` so a
                                // GCREF orig used as `PyObjectRef` becomes
                                // `SomeInstance(PyObject)`.
                                let recovered = match self.ptr_cast_marker(
                                    Some(&ValueType::Ref(None)),
                                    None,
                                    &call.dest.ty,
                                    &orig,
                                ) {
                                    Some((kind, res)) => {
                                        self.graph.block_mut(bb_id).operations.push(
                                            SpaceOperation {
                                                result: Some(res.clone()),
                                                kind,
                                            },
                                        );
                                        res
                                    }
                                    None => orig,
                                };
                                self.local_var[dest_local] = Some(LocalValue::One(recovered));
                                let target_bb = self.block_id[target];
                                let link_args = self.edge_args(mir_bb, target)?;
                                self.graph.set_goto(bb_id, target_bb, link_args);
                                return Ok(());
                            }
                            // Same recovery as the `as` cast above when
                            // `hint_promote` sits between the two halves
                            // (`Function.getcode`).
                            if let Some(orig) = self.address_cast_pointer(&args[0]) {
                                let recovered = match self.ptr_cast_marker(
                                    Some(&ValueType::Ref(None)),
                                    None,
                                    &call.dest.ty,
                                    &orig,
                                ) {
                                    Some((kind, res)) => {
                                        self.graph.block_mut(bb_id).operations.push(
                                            SpaceOperation {
                                                result: Some(res.clone()),
                                                kind,
                                            },
                                        );
                                        res
                                    }
                                    None => orig,
                                };
                                self.local_var[dest_local] = Some(LocalValue::One(recovered));
                                let target_bb = self.block_id[target];
                                let link_args = self.edge_args(mir_bb, target)?;
                                self.graph.set_goto(bb_id, target_bb, link_args);
                                return Ok(());
                            }
                        }
                        return self.emit_host_cast_llop(
                            mir_bb,
                            dest_local,
                            target,
                            args[0].clone(),
                            op,
                            result_ty,
                        );
                    }
                }
                // `llop.threadlocalref_get` / `threadlocalref_store`
                // (`rthread.py ThreadLocalField.getraw` / `setraw`).
                // The intrinsic body reads `THREADLOCALS`; project the
                // call so that body is never walked. Offset must be a
                // call-site constant (`CDefinedIntSymbolic`); otherwise
                // leave the call.
                if args.len() == 2
                    && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
                    && let Some(fd) = self.llbc.fn_by_id(*id)
                {
                    let path = fd.item_meta.name_path();
                    match host_rthread_threadlocalref_llop(&path) {
                        Some(ThreadLocalRefLlop::Get) if self.var_is_const_int(&args[0]) => {
                            let result_ty = tyref_to_value_type_with(
                                &call.dest.ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            );
                            return self.emit_threadlocalref_get(
                                mir_bb,
                                dest_local,
                                target,
                                args[0].clone(),
                                args[1].clone(),
                                result_ty,
                            );
                        }
                        Some(ThreadLocalRefLlop::Store) if self.var_is_const_int(&args[0]) => {
                            return self.emit_threadlocalref_store(
                                mir_bb,
                                dest_local,
                                target,
                                args[0].clone(),
                                args[1].clone(),
                            );
                        }
                        _ => {}
                    }
                }
                // `f64::to_int_unchecked::<i64>(x)` is `cast_float_to_int`.
                // Truncation toward zero, undefined outside the signed
                // range — the same contract as the llop.  A saturating
                // `as i64` helper is not this spelling and stays residual.
                if args.len() == 1
                    && self.is_f64_to_int_unchecked_i64(&reg, &call.dest.ty, first_arg_ty.as_ref())
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "cast_float_to_int".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `f64::is_sign_negative(x)` is `float2longlong(x) < 0` — reinterpret
                // the f64 bits as i64 (sign bit = MSB) instead of an unresolved call.
                if args.len() == 1 && self.is_f64_is_sign_negative(&reg) {
                    let bits = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(bits.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    "longlong2float".to_string(),
                                    "float2longlong".to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    });
                    let zero = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(zero.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "lt".to_string(),
                            lhs: bits.clone(),
                            rhs: zero.clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `f64::is_sign_positive(x)` is the complementary sign-bit
                // test, `float2longlong(x) >= 0`. PyPy spells the complex repr
                // decision as `math.copysign(1.0, x) == 1.0`; both must inspect
                // the sign bit so +0.0/-0.0 remain distinct.
                if args.len() == 1 && self.is_f64_is_sign_positive(&reg) {
                    let bits = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(bits.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    "longlong2float".to_string(),
                                    "float2longlong".to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    });
                    let zero = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(zero.clone()),
                        kind: OpKind::ConstInt(0),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "ge".to_string(),
                            lhs: bits,
                            rhs: zero,
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // The concrete container `IntoIterator::into_iter` impls
                // (`&[T]`/`Vec`/`[T;N]`) and `<[T]>::iter` / `iter_mut`
                // construct a container iterator. Emit the `iter` operation
                // on the container receiver — the `("slice","iter")` bridge
                // routes it to `Repr::rtype_iter` — instead of the FunDecl
                // `&[T]` signature, which would annotate a STR-view
                // receiver as List (`List ∪ String` at getslice). A window
                // `CCHARP` data word is not a container; it stays residual.
                // A STR-view receiver is copied through `ll_str2uintlist`
                // first so `iter` walks a list and the STR SSA stays
                // `String` for `__getslice` / `__strlen`.
                // `s.as_bytes().iter().position(|&b| b == c)` does not
                // walk the iterator: the successor is `str.find(c)`, so
                // the STR is forwarded as the haystack and the list copy
                // is not emitted (its inlined helper was a used-before-
                // definition of the find needle).
                if args.len() == 1
                    && self.is_concrete_iter_constructor(&reg)
                    && !args[0]
                        .concretetype()
                        .as_ref()
                        .is_some_and(lltype_is_ccharp)
                {
                    if (first_arg_is_string_byte_view || self.var_is_str_repr(&args[0]))
                        && self.mir_target_is_const_eq_iter_position(target)
                    {
                        self.bind_str_view_local(dest_local, args[0].clone());
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                    let recv = if first_arg_is_string_byte_view || self.var_is_str_repr(&args[0]) {
                        self.emit_ll_str_to_u8_list(bb_id, args[0].clone())
                    } else {
                        args[0].clone()
                    };
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    "core".to_string(),
                                    "slice".to_string(),
                                    "iter".to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![recv]),
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::str::<Impl>::chars` (and the autoderef `String` →
                // `str` receiver) constructs `rutf8.Utf8StringIterator`, not
                // a byte `StringIteratorRepr`.  Emit `__majit_utf8iter(s)`
                // with `fun_decl_id: None` so CallRegistry is not asked for
                // the Opaque `chars` body (`Utf8StringIteratorRepr`).
                if args.len() == 1 && self.is_str_chars_constructor(&reg, first_arg_ty.as_ref()) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![crate::runtime_names::shims::UTF8_ITER.to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::str::<Impl>::as_ptr` on a string-family receiver
                // is `rffi.get_raw_address_of_string` (`CCHARP`).  Emit
                // `__majit_get_raw_address_of_string(s)` with
                // `fun_decl_id: None` so CallRegistry is not asked for
                // the Opaque `as_ptr` body.  `AtomicIsize::as_ptr` and
                // slice/vec `as_ptr` have no `str` path segment.
                if args.len() == 1
                    && self.is_str_as_ptr(&reg, first_arg_ty.as_ref(), const_arg_ty.as_ref())
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    stamp_as_ccharp(&res);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    crate::runtime_names::shims::GET_RAW_ADDRESS_OF_STRING
                                        .to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::str::<Impl>::len` on a string-family receiver
                // is `len(s)` = `ll_strlen` (`rstr.py
                // AbstractStringRepr.rtype_len`), planted as the
                // `__strlen` marker the rtyper adapter and `jtransform`
                // both lower. Slice/Vec/Wtf8 `len` have their own arms.
                if args.len() == 1
                    && self.is_str_len(&reg, first_arg_ty.as_ref(), const_arg_ty.as_ref())
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__strlen".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::str::<Impl>::contains(haystack, needle)` is the
                // flowspace `contains` operation (`op.contains(container, item)`,
                // `rstr.py` `rtype_contains`).  A `char` Pattern — including
                // the utf8iter code-point Variable `decompose_valuefmt` feeds
                // `FMTS.contains(spec)` — becomes a one-code-point UTF-8
                // string (`rutf8.py unichr_as_utf8`) so the pairtype is
                // String+String.  `&str` / `&String` needles stay as the
                // string.  Any other Pattern (closure, `&[char]`, …) keeps
                // the residual FunctionPath.
                if args.len() == 2
                    && self.is_str_contains_call(&reg)
                    && let Some(kind) = self.str_contains_needle_kind(
                        &reg,
                        second_arg_ty.as_ref(),
                        second_const_arg_ty.as_ref(),
                    )
                {
                    let needle = match kind {
                        StrContainsNeedleKind::Char => {
                            self.emit_codepoint_as_utf8(mir_bb, &args[1], arg_operands.get(1))?
                        }
                        StrContainsNeedleKind::Str => args[1].clone(),
                    };
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "contains".to_string(),
                            lhs: args[0].clone(),
                            rhs: needle,
                            result_ty: ValueType::Bool,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::ptr::{const_ptr,mut_ptr}::<Impl>::is_null(p)` is
                // upstream's null test `p is None` (`flowspace` `is_`,
                // `binaryop.py` `is__default`, `rnone.py` `rtype_is_None` →
                // `ptr_iszero` on a pointer lowleveltype).  Every receiver
                // takes that operation: a GC string is a nullable
                // `SomeString`, and routing the call as
                // `getattr(p, "is_null")` dies on `String` (no such
                // attribute).  The `None` operand is `Constant(None)`.
                if args.len() == 1 && self.is_raw_ptr_is_null(&reg) {
                    // `Constant(None)` is Void (`rmodel.py`
                    // `pairtype(Repr, VoidRepr).convert_from_to`). jtransform
                    // erases the definition and folds `is_` to `ptr_iszero`.
                    let none = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Void);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(none.clone()),
                        kind: OpKind::ConstNone,
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "is_".to_string(),
                            lhs: args[0].clone(),
                            rhs: none,
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::is_empty` is `arraylen_gc(s) == 0`.  Keep both
                // operations in the graph instead of residualizing the
                // graph-less std helper.  A string-byte-view (`as_bytes()`)
                // is `ll_strlen == 0`, not a GcArray header read.
                if args.len() == 1 && self.is_slice_is_empty(&reg) {
                    let fat_len = if first_arg_is_string_byte_view {
                        None
                    } else {
                        self.fat_component(bb_id, &args[0], crate::model::VecFieldPart::FatLen)
                    };
                    let len = if let Some(len_var) = self.array_view_logical_len_for_var(&args[0]) {
                        self.bind_array_view_len_in_block(bb_id, len_var)
                    } else if let Some(len_var) = fat_len {
                        len_var
                    } else {
                        let len = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                        let len_kind = if first_arg_is_string_byte_view {
                            OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec!["__strlen".to_string()],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![args[0].clone()]),
                                result_ty: ValueType::Int,
                            }
                        } else if first_arg_is_list_view {
                            OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec!["__len".to_string()],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![args[0].clone()]),
                                result_ty: ValueType::Int,
                            }
                        } else {
                            OpKind::ArrayLen {
                                base: self.object_slice_arraylen_base(bb_id, &reg, args[0].clone()),
                                array_type_id: self.slice_object_array_type_id(&reg),
                                nolength: false,
                                gckind: crate::model::ArrayGcKind::Gc,
                            }
                        };
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(len.clone()),
                            kind: len_kind,
                        });
                        len
                    };
                    let zero = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(zero.clone()),
                        kind: OpKind::ConstInt(0),
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
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::len` is already a low-level GcArray operation in
                // the translated type model.  Emit ArrayLen directly, just
                // as `Rvalue::Len(place)` eventually does, so a gateway
                // wrapper's red `&[PyObjectRef]` argument never detours
                // through an unregistered host residual.
                //
                // A string-byte-view (`as_bytes()`) is the same place
                // `Rvalue::Len` rewrites to `__strlen`: the view is the
                // `W_UnicodeObject`, not a GcArray header.
                if args.len() == 1 && self.is_slice_len(&reg) {
                    // A `Box<[T]>` field read — list views included — reads the fat pointer's length word (`rlist.py` `ll_length` reads the list's own length);
                    // a list view with no fat word in the graph (a parameter, a `Vec`) keeps `__len`.
                    let fat_len = if first_arg_is_string_byte_view {
                        None
                    } else {
                        self.fat_component(bb_id, &args[0], crate::model::VecFieldPart::FatLen)
                    };
                    let res = if let Some(len_var) = self.array_view_logical_len_for_var(&args[0]) {
                        self.bind_array_view_len_in_block(bb_id, len_var)
                    } else if let Some(len_var) = fat_len {
                        len_var
                    } else {
                        // `.len()` of a declared virtualizable array is
                        // `arraylen_vable`, which needs the field read unmarked.
                        self.release_declared_vable_array_address(&args[0]);
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                        let kind = if first_arg_is_string_byte_view {
                            OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec!["__strlen".to_string()],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![args[0].clone()]),
                                result_ty: ValueType::Int,
                            }
                        } else if first_arg_is_list_view {
                            OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec!["__len".to_string()],
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                },
                                args: crate::model::call_args(vec![args[0].clone()]),
                                result_ty: ValueType::Int,
                            }
                        } else {
                            OpKind::ArrayLen {
                                base: self.object_slice_arraylen_base(bb_id, &reg, args[0].clone()),
                                array_type_id: self.slice_object_array_type_id(&reg),
                                nolength: false,
                                gckind: crate::model::ArrayGcKind::Gc,
                            }
                        };
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind,
                        });
                        res
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `items_block_items_base(block)` on a fixed-list GcArray
                // is `ll_fixed_items` (`rlist.py` `FixedSizeListRepr`):
                // the translated value is the array header itself. Alias
                // to the receiver so a subsequent `*base.add(idx)` is
                // `getarrayitem_gc` on that array, not a call whose
                // `*mut ItemsBlock` signature seeds `SomeInstance` and
                // unions with `SomeList` at `mergeinputargs`. The same
                // callee on a resizable list's `l.items` header stays a
                // call; the split is the operand's lowleveltype.
                if args.len() == 1
                    && regular_call_name_path(&reg, self.llbc)
                        .is_some_and(|path| items_block_accessor_returns_its_block(&path))
                    && arg_locals
                        .first()
                        .copied()
                        .flatten()
                        .is_some_and(|local| self.local_is_fixed_object_array(local))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `items_block_capacity` on a fixed-list GcArray is
                // `ll_fixed_length` (`rlist.py` `FixedSizeListRepr`): `len`
                // of the array itself. The same callee on a resizable
                // `ItemsBlock` is `len(l.items)` (`rlist.py`
                // `_ll_list_resize_hint`) and stays a call: pyre empty
                // lists keep a NULL `items` field and
                // `items_block_capacity` returns 0 for NULL, while
                // PyPy's `ll_newemptylist` allocates a zero-length array
                // so `len(l.items)` is always safe. pyre does not have
                // that non-null empty storage yet. The split is the
                // operand's lowleveltype (`[*mut PyObject]` /
                // `FixedObjectArray` vs the header).
                if args.len() == 1
                    && self.is_items_block_capacity(&reg)
                    && arg_locals
                        .first()
                        .copied()
                        .flatten()
                        .is_some_and(|local| self.local_is_fixed_object_array(local))
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayLen {
                            base: args[0].clone(),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `items_block_items_base` on a fixed-list GcArray is the
                // list itself (`rlist.py` `FixedSizeListRepr`,
                // `W_TupleObject.getitem` is `self.wrappeditems[index]`).
                // Alias the destination to the operand so `*base.add(idx)`
                // indexes that list. The same callee on an `ItemsBlock`
                // header stays a call. The split is the operand's
                // lowleveltype (`[*mut PyObject]` / `FixedObjectArray` vs
                // the header).
                if args.len() == 1
                    && self.is_items_block_items_base(&reg)
                    && arg_locals
                        .first()
                        .copied()
                        .flatten()
                        .is_some_and(|local| self.local_is_fixed_object_array(local))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `FixedObjectArray::len` reads the array's own length
                // header, which is `arraylen_gc` — so emit `ArrayLen`
                // rather than routing it through the `__len` call below.
                //
                // The receiver is the virtualizable array
                // (`PyFrame.locals_cells_stack_w`), and the codewriter's
                // virtualizable protocol requires it to be consumed by an
                // array operation in its defining block.  A `__len` call
                // passes it as a *call argument* instead, which reaches
                // `jtransform`'s `handle_residual_call` and trips
                // `_check_no_vable_array` ("a virtualizable array is passed
                // around") — witnessed on `pyframe::<Impl>::_check_stack_index`,
                // whose `locals_w!(self).len()` lowered to
                // `FieldRead(locals_cells_stack_w)` feeding
                // `Call(__len, [array])`.
                //
                // This completes the array-op family the front already
                // applies to the same receiver: `is_object_array_set_ref_call`
                // gives `set_ref` an `ArrayWrite`, and the workspace `Index`
                // interception gives `index` / `index_mut` an
                // `ArrayRead` / `ArrayWrite`.  The length-prefixed siblings
                // `IntArray` / `FloatArray` are the nested
                // `W_ListObject.int_items` / `float_items` structs
                // (`rlist.py` LIST `("length", Signed)`); `is_container_len`
                // reads that word as a `FieldRead`, matching
                // `jtransform.py` `_handle_list_call` `list.int_len` →
                // `getfield_gc_i(l, int_items.len)`.
                if args.len() == 1 && self.is_object_array_len(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayLen {
                            base: args[0].clone(),
                            array_type_id: Some(OBJECT_REF_GCARRAY_TYPE_ID.to_string()),
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Vec::len` returns the container element
                // count.  Emit the `__len` operation on the receiver — the
                // rtyper routes it through the `len` op
                // (`flowspace_adapter`), which on a `SomeList` receiver
                // lowers to `AbstractBaseListRepr.rtype_len` — instead of
                // the `getattr("len")` the generic method fallback emits,
                // which dead-ends at `Cannot find attribute "len"` on the
                // list annotation.  Same routing
                // [`is_concrete_iter_constructor`] gives the container
                // `iter`.
                // `IntArray::len` / `FloatArray::len` are the same call:
                // the receiver is the `{ block, len }` struct, so `__len`
                // becomes `getattr("__len__")`. [`Self::is_container_len`]
                // emits the `FieldRead` of `len` (`typed_array_len_owner`)
                // instead of entering that body.
                if args.len() == 1 && self.is_container_len(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
                    // `Vec::len` on a declared virtualizable array field is
                    // `arraylen_vable` (`rewrite_op_getarraysize`), not a
                    // residual `__len` that would carry the array out of the
                    // block. The index check uses the same length.
                    let vable_array = self.release_declared_vable_array_address(&args[0]);
                    // `Vec::len` is `rlist.ll_length`: `return l.length`,
                    // oopspec `list.len`. The rtyper's `len` op on a
                    // `SomeList` is that read. A `FieldRead` named `"len"`
                    // reaches the annotator as `getattr(list, "len")`, and
                    // the list struct has no such attribute
                    // (`rlist.py` `("length", Signed)`).
                    let rust_vec_kind = first_arg_ty
                        .as_ref()
                        .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc));
                    let kind = if vable_array {
                        OpKind::ArrayLen {
                            base: args[0].clone(),
                            array_type_id: None,
                            nolength: false,
                            gckind: crate::model::ArrayGcKind::Gc,
                        }
                    } else if let Some(kind) = rust_vec_kind {
                        // A `Vec` of one-word items: an inline field keeps the
                        // `(base, field)` length-word read; every other
                        // receiver is its header address and calls
                        // `ll_vec_length`.
                        if let Some(len) = inline_vec_field_part(
                            &mut self.graph,
                            bb_id,
                            &args[0],
                            crate::model::VecFieldPart::Len,
                            ValueType::Int,
                        ) {
                            // `rlist.ll_length` is `return l.length`. The
                            // component is a new op; the whole-field read
                            // stays the key (`retarget_fat_operand`: one
                            // field feeds `.len()` and an index). Dropping
                            // it here is unsound: `count_var_uses` only
                            // sees already-lowered uses, so a later
                            // `gcarray_from_pyobject_vec` / link of the
                            // same field would keep a dead operand.
                            self.local_var[dest_local] = Some(LocalValue::One(len));
                            let target_bb = self.block_id[target];
                            let link_args = self.edge_args(mir_bb, target)?;
                            self.graph.set_goto(bb_id, target_bb, link_args);
                            return Ok(());
                        }
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: rust_vec_helper_segments(
                                    majit_ir::rvec::VecOp::Length,
                                    kind,
                                ),
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        }
                    } else if let Some(owner) = self.typed_array_len_owner(&reg) {
                        // Nested `IntArray` / `FloatArray` is the LIST
                        // GcStruct's `length` word (`rlist.py`), not a
                        // `SomeList`.  `__len` on that struct is
                        // `getattr(__len__)` on a classdef with no such
                        // method (`ll_list_int_length`); the word is
                        // `getfield_gc_i` of `len`.  Signed, matching
                        // `("length", Signed)` and `getfield_gc_i`.
                        OpKind::FieldRead {
                            base: args[0].clone(),
                            field: FieldDescriptor::new("len", Some(owner.to_string())),
                            ty: ValueType::Int,
                            pure: true,
                        }
                    } else {
                        // `signature.py` `num_argnames` / `scope_length` are
                        // `len(self.argnames)`. A `Vec` whose item is wider
                        // than one word is still that header word
                        // (`rlist.ll_length`), not a value the codewriter can
                        // pass to `__len`. The address of the inline header
                        // on a GC struct is `rewrite_op_getsubstruct`'s
                        // refusal. The component read is the length; drop the
                        // address when nothing else consumes it.
                        let wide_vec = first_arg_ty
                            .as_ref()
                            .is_some_and(|ty| tyref_is_vec_value(ty, self.llbc));
                        if wide_vec
                            && let Some(len) = inline_vec_field_part(
                                &mut self.graph,
                                bb_id,
                                &args[0],
                                crate::model::VecFieldPart::Len,
                                ValueType::Int,
                            )
                        {
                            // Same as the one-word arm: keep the whole-field
                            // read; the length word is a second getfield.
                            self.local_var[dest_local] = Some(LocalValue::One(len));
                            let target_bb = self.block_id[target];
                            let link_args = self.edge_args(mir_bb, target)?;
                            self.graph.set_goto(bb_id, target_bb, link_args);
                            return Ok(());
                        }
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__len".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        }
                    };
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind,
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Vec::set_len(v, n)` on an inline field is `l.length = n`
                // (`rlist.py` `ll_set_length`). A standalone `Vec` is not
                // that LIST header; the generic call path keeps the std
                // method so a raw buffer's `set_len` is not recorded as a
                // GC setfield.
                if args.len() == 2
                    && self.is_vec_set_len(&reg)
                    && let Some(kind) = vec_set_len_write(&self.graph, &args[0], args[1].clone())
                {
                    self.graph
                        .block_mut(bb_id)
                        .operations
                        .push(SpaceOperation { result: None, kind });
                    // `ll_set_length` is void. Bind a unit dest only when
                    // the MIR dest is `()`; a dest typed as the header is
                    // the same pointer `l.length = n` leaves unchanged.
                    let dest_is_void = matches!(
                        tyref_to_value_type_with(
                            &call.dest.ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ),
                        ValueType::Void
                    );
                    self.local_var[dest_local] = Some(LocalValue::One(if dest_is_void {
                        self.emit_unit(bb_id)
                    } else {
                        args[0].clone()
                    }));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `BytesArray::as_slice` / `UnicodeArray::as_slice` expose
                // the backing items field, not the enclosing `{ block, len }`
                // storage record.  This is RPython `ll_items(l)`: read the
                // `block` field and project it to the logical `[str]` array
                // before the consumer's getitem.  The caller's explicit
                // `self.len` check remains the logical-length guard.
                if args.len() == 1
                    && let Some(owner) = self.string_array_slice_view_owner(&reg)
                {
                    let block = self.emit_string_array_items_read(bb_id, args[0].clone(), owner);
                    let logical_len =
                        self.emit_string_array_len_read(bb_id, args[0].clone(), owner);
                    let view = self.narrow_value_to_instance_root(
                        bb_id,
                        LinkArg::Value(block),
                        STRING_GCREF_GCARRAY_TYPE_ID,
                    );
                    self.local_var[dest_local] = Some(LocalValue::One(
                        view.as_variable()
                            .expect("a string-items view stays a Variable")
                            .clone(),
                    ));
                    // Same pairing as the `from_raw_parts` fold above: the
                    // view is the header, `l.length` is its own field.
                    self.record_string_array_view_len(dest_local, logical_len);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `IntArray::as_slice` / `FloatArray::as_slice` are
                // `from_raw_parts(self.base(), self.len())`. The receiver
                // is the `{ block, len }` struct, so aliasing it makes the
                // following index `getitem_SomeInstance`. Read `block` and
                // narrow it to `[i64]` / `[f64]`, the GcArray the
                // unwrapped push sites write. Record `l.length` so a
                // length read of the view is `ll_length`, not
                // `ll_fixed_length` of the items block.
                if args.len() == 1
                    && let Some((owner, root)) = self.numeric_array_slice_view_owner(&reg)
                {
                    let block = self.emit_numeric_array_items_read(bb_id, args[0].clone(), owner);
                    let logical_len =
                        self.emit_numeric_array_len_read(bb_id, args[0].clone(), owner);
                    let view =
                        self.narrow_value_to_instance_root(bb_id, LinkArg::Value(block), root);
                    self.local_var[dest_local] = Some(LocalValue::One(
                        view.as_variable()
                            .expect("a numeric-items view stays a Variable")
                            .clone(),
                    ));
                    self.record_numeric_array_view_len(dest_local, logical_len);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Vec::as_slice` / `<[T]>::as_slice` borrows the same
                // elements over shared storage — identity on the list
                // model.  Alias the result to the receiver so the consumer
                // reads the list directly, instead of the
                // `getattr("as_slice")` the generic method fallback emits,
                // which dead-ends at `Cannot find attribute …` on the list
                // annotation.  Same shape as the reflexive identity aliases
                // below.
                //
                // `Vec<PyObjectRef>::as_slice` is [`Self::try_lower_object_vec_whole_list_view`].
                if args.len() == 1 && self.is_container_slice_identity(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::as_ptr` — the slice's data pointer is the receiver
                // value in the erased array-pointer model.  Alias the result to
                // the receiver (twin of the `as_slice` identity above and
                // `NonNull::as_ptr`).  A `Vec::as_ptr` is deliberately excluded
                // (its items live behind a getfield, not the receiver) — see
                // `is_container_as_ptr_identity`.
                // A window's `as_ptr` is `FatData` only; it does not mark
                // `string_byte_view`. A W1 `as_bytes()` view keeps the mark
                // via `alias_dest_to_arg0_inherit`.
                if args.len() == 1 && self.is_container_as_ptr_identity(&reg) {
                    // A fat `Box<[T]>` receiver's data word is not the box
                    // variable. `arraylen_gc` on that word reads a constant.
                    let data = if let Some(data) =
                        self.fat_component(bb_id, &args[0], crate::model::VecFieldPart::FatData)
                    {
                        data
                    } else {
                        args[0].clone()
                    };
                    self.stamp_pair_window_as_ptr_ccharp(&call.dest.ty, &data);
                    self.alias_dest_to_arg0_inherit(dest_local, data, &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `UnsafeCell<T>` is `#[repr(transparent)]` over `T`. `new`,
                // `get`, and `get_mut` are the cell word itself — the same
                // identity as `<*mut T>::cast_mut`. `core` has no graph for
                // them, so alias the result to the single argument.
                if args.len() == 1 && self.is_unsafecell_word_identity(&reg) {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // The read is the field's own `getfield` (`jtransform.py rewrite_op_getfield`)
                // for a scalar `T`; upstream keeps such flags as plain attributes,
                // e.g. `nestedscope.py CellFamily.ever_mutated`.
                if args.len() == 1
                    && self.is_cell_get(&reg)
                    && tyref_is_primitive_scalar(&call.dest.ty, self.llbc)
                {
                    let dest_ty = tyref_to_value_type_with(
                        &call.dest.ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    );
                    let field_kind = {
                        let block = self.graph.block(bb_id);
                        block.operations.iter().rev().find_map(|op| {
                            let OpKind::FieldRead { field, .. } = &op.kind else {
                                return None;
                            };
                            if op.result.as_ref() == Some(&args[0])
                                && field.taken_by_address
                                && field.vec_part.is_none()
                                && !field.inline_vec
                            {
                                Some(op.kind.clone())
                            } else {
                                None
                            }
                        })
                    };
                    if let Some(OpKind::FieldRead {
                        base,
                        mut field,
                        pure,
                        ..
                    }) = field_kind
                    {
                        field.taken_by_address = false;
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind: OpKind::FieldRead {
                                base,
                                field,
                                ty: dest_ty,
                                pure,
                            },
                        });
                        self.local_var[dest_local] = Some(LocalValue::One(res));
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                }
                // `alloc::fmt::format` of a no-placeholder constant
                // message — `format!("literal")`, whose `format_args!`
                // lowered to `Arguments::from_str` (aliased to its
                // `__str_const` operand by the identity fold above).  The
                // render of a constant string is that string, so alias the
                // result to the constant instead of leaving an unregistered
                // `alloc::fmt::format` call.
                if args.len() == 1
                    && self.is_fmt_format_call(&reg)
                    && self.traces_to_str_const(&args[0])
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Resolve the target function's fully-qualified path
                // through the FunId → FunDecl table. `Trait` here is
                // Charon's "trait-bound generic resolved at extraction
                // time", which is itself a direct call once the impl
                // is selected — same OpKind shape as Direct.
                //
                // When the FunDecl's name path encodes an `Impl`
                // segment whose owner type is resolvable, emit
                // `CallTarget::Method` instead of `FunctionPath` so the
                // annotator's `MethodDesc.func_args`
                // (`annotator/description.rs`) prepends a
                // classdef-bound `SomeInstance` for `self`.  Without it,
                // the callee body's `self` lands with `classdef=None`
                // and any `.field` projection on it panics at
                // `unaryop.rs` (lib test
                // `generic_handler_graphs_keep_symbolic_fnaddr_surface`).
                let (segments, method_hint) =
                    self.call_target_segments(mir_bb, &reg, first_arg_ty.as_ref())?;
                let original_segments = segments.clone();
                let original_method_hint = method_hint.clone();
                // `extend_from_slice` of a `STR` onto `Vec<u8>` is the same
                // copy: convert the source first, then the existing
                // `list.extend` rewrite sees unsigned items.
                if args.len() == 2
                    && second_arg_is_string_byte_view
                    && segments.last().map(String::as_str) == Some("extend_from_slice")
                {
                    args[1] = self.emit_ll_str_to_u8_list(bb_id, args[1].clone());
                }
                // `<[T]>::to_vec(slice)` copies the slice into an owned Vec —
                // the RPython `list(slice)` builtin, whose `rtype_bltn_list`
                // (`rlist.py`) `gendirectcall`s `ll_copy`. Retarget the
                // call to the single-segment `list` builtin path so both the
                // annotator (SomeList result) and the rtyper (`rtype_n` ->
                // `rtype_bltn_list`) see a list construction. (The old
                // slice-identity alias was a miscompile: `to_vec` is a fresh
                // allocation, not an alias of the source.)
                // A `Vec` of one-word items is a raw header, not the list
                // `list(slice)` builds, so that copy stays a residual call
                // until the slice view is lowered.
                let (segments, method_hint) = if args.len() == 1
                    && is_slice_to_vec(&segments)
                    && tyref_rust_vec_item_kind(&call.dest.ty, self.llbc).is_none()
                {
                    (vec!["list".to_string()], None)
                } else if args.len() == 2
                    && second_arg_is_string_byte_view
                    && segments.last().map(String::as_str) == Some("extend_from_slice")
                {
                    (
                        vec![
                            "vec".to_string(),
                            "Vec".to_string(),
                            "extend_from_slice".to_string(),
                        ],
                        None,
                    )
                } else if args.is_empty() && is_alloc_vec_new_segments(&segments) {
                    // `Vec::new()` is the empty-list constructor.  Retarget
                    // onto the `vec::Vec::new` path `flowspace_adapter`
                    // already rewrites to `newlist()` / `ll_newemptylist`.
                    (
                        vec!["vec".to_string(), "Vec".to_string(), "new".to_string()],
                        None,
                    )
                } else if args.len() == 2
                    && is_vec_extend_segments(&segments)
                    && second_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_vec_value(ty, self.llbc))
                {
                    // PyPy expresses these source sites as list
                    // concatenation / `list.extend`, and
                    // `AbstractListRepr.rtype_method_extend` lowers them to
                    // `ll_extend`.  Rust's `Vec::extend(Vec<T>)` consumes its
                    // source, but the source is dead after the call and both
                    // values are translated lists, so route this exact
                    // by-value Vec input through the already-orthodox
                    // `extend_from_slice` → list.extend bridge.  An iterator
                    // value (`Vec::IntoIter`, Map, Range, or a custom iterator)
                    // is not a list and deliberately keeps the residual wall.
                    (
                        vec![
                            "vec".to_string(),
                            "Vec".to_string(),
                            "extend_from_slice".to_string(),
                        ],
                        None,
                    )
                } else if args.len() == 2
                    && crate::front::str_find::is_rpython_str_slice_prefix(&segments)
                {
                    // PyPy's `name[:dotindex]` reaches ordinary
                    // `rtype_getslice`.  Rust's core string index body is
                    // opaque to Charon, so route the exact interpreter
                    // compatibility helper through the existing marker that
                    // becomes `getslice(s, 0, stop)` after annotation.
                    (vec!["__getslice_rangeto".to_string()], None)
                } else {
                    (segments, method_hint)
                };
                // `Vec::set_len(n)` is `l.length = n` (`lltypesystem/rlist.py`
                // `ll_newlist` after `ll_newlist_hint`). There is no
                // `ll_vec_set_len` helper; write the header `len` word.
                // `VecFieldPart::Len` is the inline-field projection
                // (`inline_vec_field_part`); on this owner `fielddescrof`
                // would mint `len.len` with no parent SizeDescr
                // (`get_field_descr`). The codewriter stores the word
                // (`vec_set_header_word`).
                if args.len() == 2
                    && rust_vec_std_leaf(&original_segments) == Some("set_len")
                    && first_arg_ty
                        .as_ref()
                        .and_then(|ty| tyref_rust_vec_item_kind(ty, self.llbc))
                        .is_some()
                {
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::FieldWrite {
                            base: args[0].clone(),
                            field: FieldDescriptor::new("len", Some("alloc::vec::Vec".to_string())),
                            value: crate::model::LinkArg::Value(args[1].clone()),
                            ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `ptr::copy_nonoverlapping::<T>(src, dst, n)` is
                // `rgc.ll_arraycopy` / `llmemory.raw_memcopy` of `n`
                // items (`rlist.py` `ll_arraycopy`). A one-word `T` is
                // `ll_slice_arraycopy_*`; `typed_items_block_items_base`
                // dests are the header, so add `itemoffsetof` first.
                if args.len() == 3 && self.is_ptr_copy_nonoverlapping(&reg) {
                    let kind = first_arg_ty
                        .as_ref()
                        .and_then(|ty| raw_ptr_one_word_item_kind(ty, self.llbc))
                        .or_else(|| {
                            second_arg_ty
                                .as_ref()
                                .and_then(|ty| raw_ptr_one_word_item_kind(ty, self.llbc))
                        });
                    if let Some(kind) = kind
                        && let Some(src) = self.copy_nonoverlapping_item_addr(
                            mir_bb,
                            args[0].clone(),
                            arg_locals.first().copied().flatten(),
                            first_arg_ty.as_ref(),
                        )
                        && let Some(dst) = self.copy_nonoverlapping_item_addr(
                            mir_bb,
                            args[1].clone(),
                            arg_locals.get(1).copied().flatten(),
                            second_arg_ty.as_ref(),
                        )
                    {
                        let zero = self.emit_const_uint(mir_bb, 0);
                        let count_from = third_arg_ty
                            .as_ref()
                            .map(|ty| {
                                tyref_to_value_type_with(
                                    ty,
                                    self.llbc,
                                    self.tombstoned_leaves,
                                    self.gc_struct_ids,
                                    &self.raw_alloc_flavor_structs,
                                )
                            })
                            .unwrap_or(ValueType::Unsigned);
                        let count = self.retype_word(
                            mir_bb,
                            args[2].clone(),
                            &count_from,
                            &ValueType::Unsigned,
                        );
                        let _ = self.emit_path_call(
                            mir_bb,
                            majit_ir::rvec::slice_helper_path(
                                majit_ir::rvec::SliceOp::ArrayCopy,
                                kind,
                            ),
                            vec![src, dst, zero.clone(), zero, count],
                            ValueType::Void,
                        );
                        self.local_var[dest_local] = Some(LocalValue::One(self.emit_unit(bb_id)));
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                }
                // A `Vec` of one-word items is the raw header `RustVecRepr`
                // lowers; its constructors, `push`, and `as_ptr` /
                // `as_mut_ptr` call the `ll_vec_*` helper that repr names
                // for the item kind, as the rtyper writes
                // `gendirectcall(ll_newlist_hint | ll_append | ll_items, ...)`
                // into the graph it hands the codewriter. `as_ptr` of a W2
                // `Vec<u8>` list is `rgc.nonmoving_raw_ptr_for_resizable_list`
                // (CCHARP). Any other `Vec<T>` still reads the header pointer
                // word (`rlist.py` `ll_items` / `ll_vec_as_ptr`).
                if args.len() == 1
                    && matches!(
                        rust_vec_std_leaf(&original_segments),
                        Some("as_ptr" | "as_mut_ptr")
                    )
                    && (first_arg_is_list_view
                        || first_arg_ty
                            .as_ref()
                            .is_some_and(|ty| tyref_is_u8_list(ty, self.llbc)))
                {
                    let ptr = self.emit_list_items_ccharp(mir_bb, args[0].clone());
                    self.local_var[dest_local] = Some(LocalValue::One(ptr));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                let (segments, method_hint) = match rust_vec_std_call(
                    &original_segments,
                    args.len(),
                    &call.dest.ty,
                    first_arg_ty.as_ref(),
                    self.llbc,
                ) {
                    Some((op, kind)) => {
                        // `alloc::vec::from_elem(elem, n)` — the helper is
                        // `ll_alloc_and_set(count, item)`.
                        if op == majit_ir::rvec::VecOp::AllocAndSet && args.len() == 2 {
                            args.swap(0, 1);
                        }
                        (rust_vec_helper_segments(op, kind), None)
                    }
                    None if args.len() == 1
                        && matches!(
                            rust_vec_std_leaf(&original_segments),
                            Some("as_ptr" | "as_mut_ptr")
                        ) =>
                    {
                        // `ll_vec_as_ptr(header: usize)` reads `VEC_PTR_WORD`
                        // (`rvec.rs`). One-word `as_ptr` is `ll_vec_items_*`
                        // and takes the Vec; this arm is the kind-independent
                        // helper. Convert the Vec to that header the same way
                        // `pair_rustvec_integer_convert_from_to` /
                        // `gen_cast` do: `cast_ptr_to_int` then `r_uint`.
                        // Passing the Vec unique-graphs `vec_header_word`
                        // as `r_uint ∪ list` (`mergeinputargs`).
                        //
                        // Byte items (`Vec<u8>` / `CHAR`) are `rlist.py`
                        // `ll_items` / `rgc.py`
                        // `ll_nonmovable_raw_ptr_for_resizable_list`: the
                        // helper returns an integer word and the destination
                        // `*mut u8` / `*const u8` is `CCHARP`, so wrap with
                        // `cast_int_to_ptr(CCHARP, items)`. Other item kinds
                        // keep the integer result.
                        let signed = push_cast_ptr_to_int(&mut self.graph, bb_id, args[0].clone());
                        args[0] = push_r_uint(&mut self.graph, bb_id, signed);
                        if tyref_is_raw_byte_ptr(&call.dest.ty, self.llbc) {
                            let items = self.emit_path_call(
                                mir_bb,
                                &rust_vec_as_ptr_helper_segments().join("::"),
                                vec![args[0].clone()],
                                ValueType::Unsigned,
                            );
                            let ccharp = emit_cast_int_to_ccharp(&mut self.graph, bb_id, items);
                            self.local_var[dest_local] = Some(LocalValue::One(ccharp));
                            let target_bb = self.block_id[target];
                            let link_args = self.edge_args(mir_bb, target)?;
                            self.graph.set_goto(bb_id, target_bb, link_args);
                            return Ok(());
                        }
                        (rust_vec_as_ptr_helper_segments(), None)
                    }
                    None => (segments, method_hint),
                };
                // `v.reverse()` reaches `<[T]>::reverse` through
                // `Vec::deref_mut`, which aliases the slice to the `Vec`
                // header; that is `ll_reverse` on the list.
                let (segments, method_hint) = match arg_locals
                    .first()
                    .copied()
                    .flatten()
                    .filter(|_| args.len() == 1 && is_core_slice_reverse(&original_segments))
                    .and_then(|local| local_is_rust_vec_deref(self.body, self.llbc, local))
                {
                    Some(kind) => (
                        rust_vec_helper_segments(majit_ir::rvec::VecOp::Reverse, kind),
                        None,
                    ),
                    None => (segments, method_hint),
                };
                // `&mut vec[..]` of `Vec<PyObjectRef>` is the header address.
                // Write the shadow-stack range into that vec. The codewriter
                // sends the call back when that argument is not a signed word.
                let (segments, method_hint) = if args.len() == 2
                    && fmt_path_ends_with(
                        &original_segments,
                        &["gc_roots", "shadow_stack_copy_range"],
                    )
                    && self.var_is_object_vec_header(&args[1])
                {
                    (
                        vec![
                            "pyre_object".to_string(),
                            "gc_roots".to_string(),
                            "shadow_stack_copy_range_into_vec".to_string(),
                        ],
                        None,
                    )
                } else {
                    (segments, method_hint)
                };
                // PyPy's `__import__` fast path uses the signed
                // `str.find` operation directly.  Replace the
                // interpreter's native compatibility helper with that same
                // existing RPython method lowering.
                if crate::front::str_find::is_rpython_str_find_char(&segments) {
                    let index = crate::front::str_find::emit_rpython_str_find_char(
                        &mut self.graph,
                        bb_id,
                        &args,
                    )
                    .map_err(LowerError::Unsupported)?;
                    self.local_var[dest_local] = Some(LocalValue::One(index));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `vec![e0, …, eN]` lowers to
                // `box_assume_init_into_vec_unsafe(box [e0, …, eN])`, whose
                // `box_assume_init` primitive is unregistered, so the legacy
                // CodeWriter residualizes it as a plain call (the same
                // treatment `w_int_new` / `w_float_new` get).  A recognizer
                // that rewrites this shape to `OpKind::NewList` (feeding the
                // repr-generic `rtype_newlist`) is UNCONDITIONAL: it plants
                // `NewList` into a graph regardless of whether that graph
                // two-phase lifts or drops to the legacy walker, and the legacy
                // walker never runs `rtype_newlist`, so a dropped graph's raw
                // `NewList` reaches the assembler default arm and emits
                // `newlist/r>r` — an opname with no blackhole handler —
                // breaking `default_bh_builder_unwired_set_matches_task_85_snapshot`.
                //
                // Two graphs carry a vec! and still drop to the legacy walker,
                // each behind an independent non-vec! wall the recognizer does
                // not touch:
                //   • `_pypy_generic_alias::make_generic_alias`: `collect_parameters`
                //     → `push_unique` → `slice::iter::Iter::next` (unregistered),
                //     plus a `collect_parameters` `UnionError: r_uint ∪ int`.
                //   • `typedef::set_method_difference`: `&args[1..]` →
                //     `core::slice::index::<Impl>::index` (unregistered), and the
                //     `set_copy_real` → `w_set_copy_storage_from` storage-copy
                //     chain (its `__deref_write` wall is gone as of the off-GC
                //     storage box S1 — `w_set_copy_storage_from` reassigns the
                //     `items` GC field; the chain now walls only on the
                //     `w_set_new` `setattr("items")`/`d.len` `int ∪ r_uint`
                //     annotator UnionError).
                // Until BOTH graphs fully lift (slice-iterator + slice-index +
                // the collect_parameters UnionError — all annotator-completeness
                // axis walls, not vec! walls), the unconditional recognizer is
                // unsafe.  `rtype_newlist`'s Fixed arm stays implemented and
                // dormant; the residual call is the correct lowering meanwhile.
                // For a method/direct callee this equals the callee's
                // `name_path()`; the scope predicate keys on the module
                // path, which the built `CallTarget::Method` drops.
                callee_name_path = Some(segments.join("::"));
                if self.try_lower_checked_neg(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_wrapping_binop(
                    mir_bb, &reg.kind, &segments, &args, dest_local, target,
                )? {
                    return Ok(());
                }
                if self.try_lower_ptr_read(
                    mir_bb,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_wrapping_unop(
                    mir_bb, &reg.kind, &segments, &args, dest_local, target,
                )? {
                    return Ok(());
                }
                if self.try_lower_cmp_minmax(
                    mir_bb,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_cmp_binop(
                    mir_bb,
                    &segments,
                    &args,
                    dest_local,
                    first_arg_ty.as_ref(),
                    second_arg_ty.as_ref(),
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_word_rotate(
                    mir_bb, &reg.kind, &segments, &args, dest_local, target,
                )? {
                    return Ok(());
                }
                if self.try_lower_rbigint_divmod_pair(
                    mir_bb,
                    &segments,
                    &args,
                    first_arg_ty.as_ref(),
                    second_arg_ty.as_ref(),
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_usize_try_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_i64_to_i32_try_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    first_arg_ty.as_ref(),
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_slice_from_ref(
                    mir_bb,
                    &segments,
                    &args,
                    first_arg_ty.as_ref(),
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_vec_try_reserve_exact(
                    mir_bb,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_num_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_rbigint_toint(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_bigint_i64_try_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_wtf8_as_str(
                    mir_bb,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                // `RBigInt::{one,zero}()` — a 0-arg constructor returning the
                // translated one-GC-reference bigint. Emit the same constant
                // through the pointer-ABI `rbigint.fromint` residual instead
                // of exposing Rust's by-value struct return ABI.
                if args.is_empty()
                    && let [.., owner_a, owner_b, leaf] = segments.as_slice()
                    && owner_a == "rbigint"
                    && owner_b == "RBigInt"
                    && matches!(leaf.as_str(), "one" | "zero")
                {
                    let n = if leaf == "one" { 1 } else { 0 };
                    let cst = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(cst.clone()),
                        kind: OpKind::ConstInt(n),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    crate::runtime_names::crates::OBJECT.to_string(),
                                    "longobject".to_string(),
                                    "jit_bigint_from_i64".to_string(),
                                ],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![cst]),
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    // `one`/`zero` return before the shared call-result
                    // narrow. The destination is still the user class
                    // `annotationoftype` would build
                    // (`getuniqueclassdef`).
                    let stored = if let Some(root) = self.rbigint_result_class_root(&call.dest.ty) {
                        let narrowed = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(narrowed.clone()),
                            kind: crate::model::cast_instance_call(root, res),
                        });
                        narrowed
                    } else {
                        res
                    };
                    self.local_var[dest_local] = Some(LocalValue::One(stored));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<str as PartialEq>::eq` and the matching `ne` methods
                // are string equality (`rtype_eq` → `ll_streq`). A
                // rust-heap `String` is not `Ptr(STR)`; `<String as
                // PartialEq>::eq` stays residual so `ll_streq` does not
                // consume a `{cap, ptr, len}` header.
                if args.len() == 2
                    && let Some(leaf) = segments.last()
                    && matches!(leaf.as_str(), "eq" | "ne")
                    && (fmt_path_ends_with(&segments, &["str", "traits", "<Impl>", "eq"])
                        || fmt_path_ends_with(&segments, &["str", "traits", "<Impl>", "ne"]))
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: leaf.clone(),
                            lhs: args[0].clone(),
                            rhs: args[1].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `as_bytes()[a..b] == as_bytes()[c..d]` is `s[a:b] == t[c:d]`.
                // Rust types the byte view as `[u8]`, so `==` resolves to
                // `core::slice::cmp::<Impl>::eq` rather than `<str as
                // PartialEq>::eq`.  Both operands are the frontend's string
                // identity (`string_byte_view_locals`); emit the same
                // `BinOp("eq")` that becomes `ll_streq`.
                if args.len() == 2
                    && fmt_path_ends_with(&segments, &["slice", "cmp", "<Impl>", "eq"])
                    && first_arg_is_string_byte_view
                    && second_arg_is_string_byte_view
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: args[0].clone(),
                            rhs: args[1].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Wtf8::len` is `ll_strlen` only when the word is an rstr
                // (a marked byte view) or a `Wtf8Buf`, whose `Vec` length
                // sits at the same offset.  A `W_UnicodeObject` keeps its
                // class pointer there, so a host `&Wtf8` stays residual.
                if args.len() == 1
                    && (first_arg_is_string_byte_view
                        || first_arg_ty
                            .as_ref()
                            .is_some_and(|ty| tyref_is_wtf8buf(ty, self.llbc)))
                    && (fmt_path_ends_with(&segments, &["Wtf8", "len"])
                        || fmt_path_ends_with(&segments, &["rustpython_wtf8", "Wtf8", "len"]))
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__strlen".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Wtf8::code_points` is the string itself. `Iterator::count`
                // on that iterator is `Wtf8CodePoints::count`, one integer,
                // not an iterator object: `rutf8.codepoints_in_utf8` with
                // the same defaults as upstream (`start=0, end=sys.maxint`).
                if args.len() == 1
                    && (fmt_path_ends_with(&segments, &["Wtf8", "code_points"])
                        || fmt_path_ends_with(
                            &segments,
                            &["rustpython_wtf8", "Wtf8", "code_points"],
                        ))
                {
                    self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if args.len() == 1
                    && segments.last().is_some_and(|leaf| leaf == "count")
                    && segments.iter().any(|seg| seg.contains("Wtf8CodePoints"))
                {
                    let start = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(start.clone()),
                        kind: OpKind::ConstUInt(0),
                    });
                    let end = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(end.clone()),
                        kind: OpKind::ConstUInt(usize::MAX as u64),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::function_path([
                                "pyre_object",
                                "rutf8",
                                "codepoints_in_utf8",
                            ]),
                            args: crate::model::call_args(vec![args[0].clone(), start, end]),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `&a == &b` / `&a != &b` on two string-family references
                // (`&Wtf8`, `&str`, `&String`) resolves to the blanket
                // `impl PartialEq<&B> for &A` (`core::cmp::impls`), whose body
                // forwards to the pointee's own `eq`.  Both operands already
                // carry the single `SomeString` value the reference projects
                // to, so this is the same `BinOp("eq")` / `BinOp("ne")`
                // (`ll_streq`) the `<str as PartialEq>::eq` arm above emits;
                // `ne` is the same lowering: `AbstractStringRepr.rtype_ne`
                // is `ll_streq` followed by `bool_not`, and the flow graph's
                // string `BinOp("ne")` is that decomposition's canonical
                // spelling.  Left residual,
                // every `w_str_get_wtf8(a) == w_str_get_wtf8(b)`
                // and `name == "__doc__"` is an unlowered helper that blocks
                // the descent into its caller (`is_w`, `getattr_str_impl`).
                // The operand-type gate keeps the integer and float impls
                // that share this owner (`<i64 as PartialEq>::eq`) out.
                if args.len() == 2
                    && let [.., owner_a, owner_b, owner_c, leaf] = segments.as_slice()
                    && owner_a == "cmp"
                    && owner_b == "impls"
                    && owner_c == "<Impl>"
                    && matches!(leaf.as_str(), "eq" | "ne")
                    && ([first_arg_ty.as_ref(), second_arg_ty.as_ref()]
                        .iter()
                        .all(|t| t.is_some_and(|t| tyref_is_string_family(t, self.llbc)))
                        || arg_locals.iter().all(|local| {
                            local.is_some_and(|local| self.string_byte_view_locals.contains(&local))
                        }))
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: leaf.clone(),
                            lhs: args[0].clone(),
                            rhs: args[1].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<NonNull<T> as PartialEq>::eq` / `ne` compares the two
                // addresses (`as_ptr` with `as_ptr`). Each operand is
                // `&NonNull<T>`, and `Rvalue::Ref` of that local is the
                // pointer word, so the args are those two refs.
                // `rclass.py` `pairtype(InstanceRepr, InstanceRepr)`
                // `rtype_eq` / `rtype_ne` lower the comparison to `ptr_eq`
                // / `ptr_ne`.
                if args.len() == 2
                    && (fmt_path_ends_with_owner_method(
                        &segments,
                        &["non_null", "NonNull"],
                        &["core", "cmp", "PartialEq"],
                        "eq",
                    ) || fmt_path_ends_with_owner_method(
                        &segments,
                        &["non_null", "NonNull"],
                        &["core", "cmp", "PartialEq"],
                        "ne",
                    ))
                {
                    let leaf = if fmt_path_ends_with_owner_method(
                        &segments,
                        &["non_null", "NonNull"],
                        &["core", "cmp", "PartialEq"],
                        "ne",
                    ) {
                        "ne"
                    } else {
                        "eq"
                    };
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
                    return Ok(());
                }
                // FunctionPath `option::Option::eq` / `ne`. The Method
                // spelling is the same helper, called again before
                // `CallTarget::Method` is built.
                if self.try_lower_option_niche_eq_ne(
                    mir_bb,
                    &segments,
                    method_hint.as_ref(),
                    &args,
                    first_arg_ty.as_ref(),
                    second_arg_ty.as_ref(),
                    dest_local,
                    target,
                )? {
                    return Ok(());
                }
                // `<T as ToString>::to_string(x)` is `str(x)` (`ll_str`)
                // only when `x` already has an rstr `ll_str`: an integer
                // (`ll_int2dec`) or rstr `STR` (identity). A Rust-heap
                // `String` is Vec-shaped and a class instance has no
                // `ll_str`; lowering those to `UnaryOp("str")` assembles
                // as the unwired `int_str/r>r`. Leave them as the residual
                // `to_string` call.
                if args.len() == 1
                    && (fmt_path_ends_with(&segments, &["string", "<Impl>", "to_string"])
                        || self.callee_template_is(&reg, "alloc::string::<Impl>::to_string"))
                    && first_arg_ty.as_ref().is_some_and(|ty| {
                        matches!(
                            tyref_to_value_type_with(
                                ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            ),
                            ValueType::Str
                                | ValueType::Int
                                | ValueType::Unsigned
                                | ValueType::Bool
                                | ValueType::Int128
                                | ValueType::UInt128
                        )
                    })
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
                    stamp_as_rstr_str(&res);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "str".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Str,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<str>::is_empty(s)` is `len(s) == 0` (`ll_strlen` then a
                // `ConstInt(0)` compare) — emit the `__len` + `BinOp("eq")`
                // decomposition instead of leaving the graph-less
                // trait-method extern.  `__len` routes through the rtyper's
                // `len` op (`flowspace_adapter`), which on a `&str` operand
                // lowers to `StringRepr.rtype_len` → `ll_strlen` → the
                // `strlen` blackhole op; the `eq` of the two `Signed`
                // operands lowers to `int_eq`.
                if args.len() == 1 && fmt_path_ends_with(&segments, &["str", "<Impl>", "is_empty"])
                {
                    let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
                        let res =
                            graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind,
                        });
                        res
                    };
                    let len = push_op(
                        &mut self.graph,
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__len".to_string()],
                                fun_decl_id: None,
                                specialization: crate::parse::GraphSpecialization::Regular,
                            },
                            args: crate::model::call_args(vec![args[0].clone()]),
                            result_ty: ValueType::Int,
                        },
                    );
                    let zero = push_op(&mut self.graph, OpKind::ConstInt(0));
                    let res = push_op(
                        &mut self.graph,
                        OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: len,
                            rhs: zero,
                            result_ty: ValueType::Int,
                        },
                    );
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `core::intrinsics::discriminant_value(v)` reads the integer
                // tag of the enum `*v` — the intrinsic (call) form of
                // `Rvalue::Discriminant`.  Emit the same synthetic
                // `__discriminant` `FieldRead` instead of leaving the graph-less
                // intrinsic extern.  The owner is derived from the arg's enum
                // type (peeling the `&T` ref), keyed to the enum base identity
                // so it resolves at the tag's real byte position; an
                // unresolvable type falls back to the unowned read (offset 0),
                // matching `Rvalue::Discriminant`'s `None` path.
                if args.len() == 1
                    && fmt_path_ends_with(&segments, &["intrinsics", "discriminant_value"])
                {
                    let (owner_root, owner_id) = match first_arg_ty
                        .as_ref()
                        .and_then(|t| self.tyref_ref_adt_def_id(t))
                        .and_then(|id| self.llbc.type_by_id(id))
                        .map(|td| td.item_meta.name_path())
                    {
                        Some(name_path) => {
                            let canon = decl_path_for_tombstone(&name_path, self.tombstoned_leaves);
                            let sid = majit_ir::descr::StructId::from_canonical(&canon);
                            (Some(canon), Some(sid))
                        }
                        None => (None, None),
                    };
                    // A FIELDLESS enum is `Int`-valued (`tyref_to_value_type`),
                    // and `&enum` aliases the enum value (`Rvalue::Ref` is the
                    // identity), so `args[0]` already IS the discriminant int —
                    // alias it directly rather than reading a `__discriminant`
                    // field off an integer base (no rtype clsfield / blackhole
                    // handler covers a getfield off an int).  Mirrors the
                    // standalone `Rvalue::Discriminant` identity collapse.  A
                    // payload enum keeps the `__discriminant` `FieldRead`.
                    let pointee_fieldless = first_arg_ty
                        .as_ref()
                        .and_then(|t| self.tyref_ref_adt_def_id(t))
                        .and_then(|id| self.llbc.type_by_id(id))
                        .is_some_and(|td| type_decl_is_fieldless_enum(td, self.llbc));
                    if pointee_fieldless {
                        self.alias_dest_to_arg0_inherit(dest_local, args[0].clone(), &arg_locals);
                        let target_bb = self.block_id[target];
                        let link_args = self.edge_args(mir_bb, target)?;
                        self.graph.set_goto(bb_id, target_bb, link_args);
                        return Ok(());
                    }
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base: args[0].clone(),
                            field: crate::model::FieldDescriptor {
                                name: "__discriminant".to_string(),
                                owner_root,
                                owner_id,
                                base_is_deref: None,
                                taken_by_address: false,
                                inline_vec: false,
                                vec_part: None,
                                owner_declared_gc: None,
                                host_index: None,
                                scalar_word: None,
                            },
                            ty: ValueType::Int,
                            pure: true,
                        },
                    });
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                if regular_call_name_path(&reg, self.llbc).as_deref()
                    == Some("majit_ir::llop::gc_load_indexed")
                {
                    let res = self.lower_llop_gc_load_indexed(
                        mir_bb,
                        bb_id,
                        &args,
                        &arg_operands,
                        &call.dest.ty,
                    )?;
                    self.local_var[dest_local] = Some(LocalValue::One(res));
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                let alias = if let Some(payload) =
                    self.expect_on_const_ok(&segments, &args, &arg_locals)
                {
                    // Identity unwrap: the receiver variable was bound
                    // directly to the `Ok` payload, so the result is
                    // that variable — bind and close the block with no
                    // op emitted.
                    Some(payload)
                } else {
                    self.reflexive_into_alias(
                        &reg,
                        &segments,
                        &args,
                        first_arg_ty.as_ref(),
                        &call.dest.ty,
                    )
                    .or_else(|| {
                        self.reflexive_into_iter_alias(
                            &segments,
                            &args,
                            first_arg_ty.as_ref(),
                            &call.dest.ty,
                        )
                    })
                    .or_else(|| self.trait_into_string_alias(&reg, &segments, &args, &call.dest.ty))
                    .or_else(|| self.wtf8_string_identity_alias(&segments, &args))
                    .or_else(|| {
                        self.oparg_arg_get_alias(&reg.kind, &segments, &args, &call.dest.ty)
                    })
                    .or_else(|| self.oparg_value_alias(&segments, &args))
                    .or_else(|| self.identity_passthrough_alias(&segments, &args))
                    .or_else(|| self.nonnull_new_identity_alias(&segments, &args, &call.dest.ty))
                };
                if let Some(value) = alias {
                    self.local_var[dest_local] = Some(LocalValue::One(value));
                    let bb_id = self.emit_bb(mir_bb);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                {
                    self.refuse_borrowed_transparent_scalar_residual(mir_bb, &reg, &call_arg_tys)?;
                    // `jit::promote(x)` (and its `promote_string` /
                    // `promote_unicode` siblings) rewrites to the synthesised
                    // `hint_promote*` marker so the residual `OpKind::Call`
                    // reaches `jtransform::rewrite_op_hint`, which emits
                    // `[-live-, <kind>_guard_value(x)]`
                    // (`codewriter/jtransform.py`).  The rtyper
                    // lowers the marker to `same_as` for the dual-gate type
                    // projection (`flowspace_adapter`), and jtransform aliases
                    // the result back to `x`.  Same single-segment marker
                    // shape as the `elidable_promote` wrapper's
                    // `hint_promote_or_string`.
                    // Residual `w_bytes_getitem` yields a `u8` byte.
                    let byte_item = segments.last().is_some_and(|s| {
                        s == "w_bytes_getitem"
                            || s == "w_bytearray_getitem"
                            || s == "bytes_like_getitem"
                    });
                    let promote_marker = self.jit_promote_marker(&reg);
                    let fun_decl_id = regular_call_fun_decl_id(&reg.kind);
                    let target = if let Some((trait_root, method_name)) =
                        abstract_trait_call_target(&reg, self.llbc)
                    {
                        CallTarget::indirect(trait_root, method_name)
                    } else if let Some(path) = scalar_inherent_method_path(
                        &reg,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    ) {
                        CallTarget::FunctionPath {
                            segments: path,
                            fun_decl_id: None,
                            specialization: crate::parse::GraphSpecialization::Regular,
                        }
                    } else if args.len() == 1
                        && let Some(marker) = promote_marker
                    {
                        CallTarget::FunctionPath {
                            segments: vec![marker.to_string()],
                            fun_decl_id: None,
                            specialization: crate::parse::GraphSpecialization::Regular,
                        }
                    } else {
                        // `CallTarget::Method` requires a receiver in `args[0]`
                        // (the flowspace adapter lowers it to `getattr(recv,
                        // method_leaf) → simple_call(bound_method, …)`).
                        // Charon's `impl_method_owner` matches both inherent
                        // methods (which carry `&self`) *and* associated
                        // functions (e.g. `RootScope::new()` — no `self` arg).
                        // Only the former actually has a receiver in `args[0]`;
                        // routing a 0-arg associated function through `Method`
                        // panics at `flowspace_adapter.rs` ("Call::Method
                        // has empty args").  Fall back to the `FunctionPath`
                        // segments when there is no receiver to thread.
                        // Fold niche `Option` `eq`/`ne` before the Method
                        // target is chosen (`impl_method_owner` /
                        // `first_is_self`). Left as `CallTarget::Method
                        // { name: "eq" }`, the adapter getattr's `eq` off
                        // the niche payload (`SomeInstance(PyObject,
                        // can_be_None=True)`) and annotate blocks.
                        if self.try_lower_option_niche_eq_ne(
                            mir_bb,
                            &segments,
                            method_hint.as_ref(),
                            &args,
                            first_arg_ty.as_ref(),
                            second_arg_ty.as_ref(),
                            dest_local,
                            target,
                        )? {
                            return Ok(());
                        }
                        let same_callee =
                            segments == original_segments && method_hint == original_method_hint;
                        match method_hint {
                            Some((owner_root, leaf, _)) if !args.is_empty() => {
                                // Method is this FunDecl: stamp its registered
                                // path so `target_to_path` does not use the
                                // owner leaf as a suffix of the impl key.
                                let mut target = CallTarget::method(leaf, Some(owner_root));
                                if let Some(id) = fun_decl_id {
                                    target = target.with_fun_decl_id(id);
                                    if let Some(fd) = self.llbc.fn_by_id(id) {
                                        target = target.with_resolved_path(
                                            registered_path_for_fun_decl(self.llbc, fd),
                                        );
                                    }
                                }
                                target
                            }
                            _ => {
                                // Opaque cross-crate `self` (`RootScope` in the
                                // caller's LLBC). `impl_method_owner` declines
                                // `CallTarget::Method`, so `MethodDesc.func_args`
                                // never prepends `SomeInstance(selfclassdef)`
                                // and the traced body binds a classdef-less
                                // receiver. Paint the classdef onto this
                                // argument only.
                                if let Some(root) = self.opaque_traced_method_self_root(&reg)
                                    && let Some(recv) = args.first().cloned()
                                {
                                    let narrowed = self.graph.alloc_value_var_with_type(
                                        crate::model::ConcreteType::Unknown,
                                    );
                                    let bb_id = self.emit_bb(mir_bb);
                                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                        result: Some(narrowed.clone()),
                                        kind: crate::model::cast_instance_call(root, recv),
                                    });
                                    args[0] = narrowed;
                                }
                                // `ll_dict_setitem` / `ll_dict_lookup`
                                // (`rordereddict.py`) type the receiver as
                                // the ordered dict. Paint the instantiated
                                // `RDict<K,V,S>` so that cast is `SomeDict`;
                                // a bare `RDict` parameter stays an instance.
                                if let Some(root) = rdict_insert_get_cast_root(
                                    &segments,
                                    first_arg_ty.as_ref(),
                                    self.llbc,
                                ) && let Some(recv) = args.first().cloned()
                                {
                                    let narrowed = self.graph.alloc_value_var_with_type(
                                        crate::model::ConcreteType::Unknown,
                                    );
                                    let bb_id = self.block_id[mir_bb];
                                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                        result: Some(narrowed.clone()),
                                        kind: crate::model::cast_instance_call(root, recv),
                                    });
                                    args[0] = narrowed;
                                }
                                let mut target = CallTarget::FunctionPath {
                                    segments,
                                    fun_decl_id: None,
                                    specialization: crate::parse::GraphSpecialization::Regular,
                                };
                                if same_callee && let Some(id) = fun_decl_id {
                                    target = target.with_fun_decl_id(id);
                                }
                                target
                            }
                        }
                    };
                    if self.graph.name.ends_with("pyobject_from_constant_addr")
                        && args.len() == 1
                        && regular_call_name_path(&reg, self.llbc)
                            .is_some_and(|path| path.ends_with("pyobject_from_constant"))
                    {
                        // The address argument is an int word. The callee
                        // takes `&ConstantData`. `lltype.cast_int_to_ptr`
                        // is the host call `rtype_cast_int_to_ptr` accepts.
                        let cast = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
                        let bb_id = self.emit_bb(mir_bb);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(cast.clone()),
                            kind: cast_int_to_ptr_call(args[0].clone()),
                        });
                        args[0] = cast;
                    }
                    let result_ty = if byte_item {
                        ValueType::Unsigned
                    } else {
                        result_ty
                    };
                    if let Some(id) = fun_decl_id
                        && let Some(fd) = self.llbc.fn_by_id(id)
                        && fundecl_keeps_zst_self(
                            self.llbc,
                            fd,
                            None,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        )
                    {
                        self.keep_dyn_call_receiver(mir_bb, &mut args);
                    }
                    OpKind::Call {
                        target,
                        args: crate::model::call_args(args),
                        result_ty,
                    }
                }
            }
            (CallClass::Dynamic, CallFunc::Dynamic(dyn_operand, _)) => {
                // `dyn Trait` virtual call.  The inline-dispatch form reads
                // the method fn-ptr straight out of the receiver's vtable
                // (`(*recv).vtable.method_<name>`), so the operand's
                // terminal projection is `Field[Adt(vtable_id), slot]`.
                // Preserve that projection as the funcptr operand of an
                // `IndirectCall`; `rpbc::lower_indirect_calls` supplies the
                // full impl family, matching `ClassRepr.getclsfield` followed
                // by `FunctionReprBase.call`, instead of routing through the
                // synthetic `__dyn_call`.  The
                // call args already carry the receiver in `args[0]` (the
                // fat-pointer data word). RPython method `indirect_call`
                // always has self: keep that word as Ref (GC instance;
                // `history.getkind` of a Gc Ptr) even when the rustc
                // strategy type is ZST (`keep_dyn_call_receiver`). The
                // callee ICT jitcode keeps the matching unused `self`
                // inputarg (`fundecl_keeps_zst_self`).
                //
                // A stored fn-ptr — the one-hop `Option<fn-ptr>` deref, a
                // host-registered callback table, a settable hook — is not a
                // vtable slot, so `dyn_indirect_target` does not name it.  It
                // is still a PBC: RPython's rtyped call through a
                // `Ptr(FuncType)` value is `indirect_call(funcptr, *args,
                // c_graphs)`, not a synthetic opaque helper, and the family
                // it carries decides whether the codewriter looks inside the
                // callees or emits a residual call. Only a genuinely
                // non-pointer operand, such as a `dyn` fat pointer the vtable
                // arm could not resolve, keeps the synthetic `__dyn_call`
                // path with the receiver threaded into `args[0]`.
                let fn_ptr_family = operand_fn_ptr_family(&dyn_operand, self.llbc);
                let eval_hook_graphs =
                    eval_hook_graphs_for_call(self.llbc, self.body, &dyn_operand);
                // A monomorphized vtable stores every method slot as
                // `*const ()`: the call casts the slot to the method's fn-ptr
                // type into a temporary and calls the temporary.  The slot
                // read is that cast's operand.
                let vtable_slot = vtable_slot_under_fn_ptr_cast(self.llbc, self.body, &dyn_operand)
                    .unwrap_or_else(|| dyn_operand.clone());
                let indirect = self.dyn_indirect_target(&vtable_slot);
                let passes_pairs = !pair_lens.is_empty();
                let mut args = splice_pair_lens(args, pair_lens);
                if let Some((trait_root, method_name)) = indirect {
                    // `SetStrategy::{add,remove,has_key}` take
                    // `(&self, w_set, ObjectKey)`. The flow graph drops the
                    // ZST `&self` and leaves `ObjectKey` as one pointer, so
                    // the residual words are `(w_set, &ObjectKey)`. The vtable
                    // slot is the rustc method, which reads those words as
                    // `self` and `w_set`. Call the word trampoline that
                    // matches the residual instead. Do not recast the
                    // receiver before this trampoline.
                    if trait_root == "SetStrategy"
                        && matches!(method_name.as_str(), "add" | "remove" | "has_key")
                        && (args.len() == 2 || args.len() == 3)
                    {
                        let key_args = if args.len() == 3 {
                            vec![args[1].clone(), args[2].clone()]
                        } else {
                            vec![args[0].clone(), args[1].clone()]
                        };
                        let leaf = match method_name.as_str() {
                            "add" => "set_strategy_add_key_ptr",
                            "remove" => "set_strategy_remove_key_ptr",
                            _ => "set_strategy_has_key_ptr",
                        };
                        OpKind::Call {
                            target: CallTarget::function_path(["pyre_object", "setobject", leaf]),
                            args: crate::model::call_args(key_args),
                            result_ty,
                        }
                    } else {
                        // RPython `ClassRepr.getclsfield` emits the concrete
                        // vtable field read, then `FunctionReprBase.call` feeds
                        // that pointer to `indirect_call`.  `resolve_operand`
                        // already lowers this exact MIR Field projection, so keep
                        // it instead of replacing it with a blackhole-only
                        // name-based lookup that has no upstream counterpart.
                        // Keep the fat-pointer data word as the first Ref
                        // argument (the strategy instance) so the residual
                        // matches rustc `(&self, ...)` and the family's
                        // FUNC type.
                        self.keep_dyn_call_receiver(mir_bb, &mut args);
                        let funcptr = self.resolve_operand(mir_bb, vtable_slot)?;
                        OpKind::IndirectCall {
                            funcptr,
                            args,
                            graphs: None,
                            family_key: Some((trait_root, method_name)),
                            result_ty,
                        }
                    }
                } else if let Some(graphs) = eval_hook_graphs {
                    // `register_eval_override`'s function and the `FnDef`
                    // cast to that parameter's type are one prebuilt
                    // constant set. `FunctionsPBCRepr` /
                    // `SmallFunctionSetPBCRepr` (`rpbc.py`) lower it to
                    // `indirect_call(funcptr, *args, c_graphs)`.
                    // `guess_call_kind` answers `recursive` only when the
                    // funcptr is `portal_runner_ptr`; this pointer is not,
                    // so `rewrite_op_indirect_call` takes
                    // `handle_regular_indirect_call` or `handle_residual_call`.
                    let funcptr = self.resolve_operand(mir_bb, dyn_operand)?;
                    OpKind::IndirectCall {
                        funcptr,
                        args,
                        graphs: Some(graphs),
                        family_key: None,
                        result_ty,
                    }
                } else if let Some(family) = fn_ptr_family {
                    let funcptr = self.resolve_operand(mir_bb, dyn_operand)?;
                    let graphs = match family {
                        FnPtrFamily::BuiltinWrapper => Some(Vec::new()),
                        FnPtrFamily::Unknown => None,
                    };
                    OpKind::IndirectCall {
                        funcptr,
                        args,
                        graphs,
                        family_key: None,
                        result_ty,
                    }
                } else if passes_pairs {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: pair slice through a dynamic call"
                    )));
                } else {
                    let recv = self.resolve_operand(mir_bb, dyn_operand)?;
                    let mut full_args = Vec::with_capacity(args.len() + 1);
                    full_args.push(recv);
                    full_args.extend(args);
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec!["__dyn_call".to_string()],
                            fun_decl_id: None,
                            specialization: crate::parse::GraphSpecialization::Regular,
                        },
                        args: crate::model::call_args(full_args),
                        result_ty,
                    }
                }
            }
            (CallClass::Ptr, _) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call CallClass::Ptr (fn pointer) not yet supported"
                )));
            }
            (CallClass::Unknown, _) | (_, CallFunc::Unknown) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call with unknown CallFunc/CallClass"
                )));
            }
            // Class/payload mismatches shouldn't happen — `classify`
            // is total over the typed variants — but cover the arm so
            // the match is exhaustive without `_`.
            (CallClass::Dynamic, _) | (CallClass::Direct, _) | (CallClass::Trait, _) => {
                return Err(LowerError::Schema(format!(
                    "bb{mir_bb}: CallClass / CallFunc mismatch"
                )));
            }
        };

        // A hint-marker call (`jit::promote(x)` → `hint_promote`,
        // `#[elidable_promote]` → `hint_promote_or_string`) lowers to the
        // distinct `OpKind::Hint` op (RPython `flowspace/operation.py
        // add_operator('hint', None, dispatch=1)`) carrying the structured
        // hint `kind`, instead of a synthesised `Call` marker classified by
        // name downstream.  The flowspace oracle types it as `same_as(value)`
        // and `jtransform::rewrite_op_hint` rewrites it to the
        // `<kind>_guard_value` family.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && !args.is_empty()
            && let Some(kind) =
                crate::hints::classify_hint_segments(segments.iter().map(String::as_str))
        {
            OpKind::Hint {
                value: args[0].clone().into_variable(),
                kind,
            }
        } else {
            op_kind
        };

        // PyPy float repr and each complex repr component call
        // `rfloat.formatd`, an external returning one low-level `STR` GC
        // pointer. RustPython's equivalent host formatters return a three-word
        // Rust `String`, which the residual ABI cannot express. Retarget only
        // those exact formatting boundaries to one-word `Utf8Str*` (`rstr.py`
        // `rpy_string`) wrappers; the complex sign/parenthesis builder remains
        // in its graph.
        // The wrapper's own body is the one caller of the host formatter that
        // keeps it: retargeting there would make the wrapper call itself.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && let Some(residual) = crate::front::rfloat_call::repr_residual_path(segments)
            && residual.join("::") != self.graph.name
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Str,
            }
        } else {
            op_kind
        };

        // Rust's `RBigInt::clone` returns a new by-value handle sharing the
        // immutable digit array. In the translated one-GC-reference shape that
        // requires a fresh payload allocation: neither a reference-identity
        // assignment nor descending into the classdef-less `_digits` field is
        // correct. Retarget the exact receiver/result pair to the GC-aware
        // shallow-clone residual.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && tyref_is_rbigint(&call.dest.ty, self.llbc)
            && let Some(segments) = match target {
                CallTarget::FunctionPath { segments, .. } => segments
                    .last()
                    .and_then(|leaf| crate::front::rbigint_call::clone_residual_for_method(leaf)),
                CallTarget::Method { name, .. } => {
                    crate::front::rbigint_call::clone_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::function_path(segments),
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // A translated RBigInt value is already its one-GC-reference payload.
        // Rust's host-only `w_long_new(RBigInt)` / constant-boxing functions
        // accept the two-word by-value handle and allocate that payload first;
        // keeping that call in JIT IR would therefore use the wrong ABI and
        // duplicate the translated object. Wrap the existing payload pointer
        // directly, matching `W_LongObject(value)` in PyPy.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(residual) = crate::front::rbigint_call::long_box_residual_path(segments)
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Retarget an RBigInt Rust trait-operator call (`<RBigInt as
        // BitAnd>::bitand`, …) to its elidable `jit_bigint_*`
        // residual. The trait shell obscures the RPython method, and both operands
        // and the result are the classdef-less `*mut BigInt` GcRef the front
        // models a `BigInt` as, so the i64-ABI residual is a faithful pointer
        // pass.  Guarded on both operands resolving to the opaque `BigInt` ADT
        // so a same-named operator on another type is never mis-retargeted; a
        // pure target swap (args + result var unchanged), fail-safe by
        // construction (a non-BigInt / unlisted operator leaves the residual
        // `<Impl>` call the census Skips).
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|t| tyref_is_rbigint(t, self.llbc))
            && second_arg_ty
                .as_ref()
                .is_some_and(|t| tyref_is_rbigint(t, self.llbc))
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::front::bigint_binop::bigint_binop_residual_path(segments)
                }
                CallTarget::Method { name, .. } => {
                    crate::front::bigint_binop::bigint_binop_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // PyPy W_LongObject dispatches a mixed long/int add, subtract,
        // multiply, or bitwise operation directly to rbigint.int_* with the
        // W_IntObject's signed machine word. Preserve that one-GCREF +
        // one-Signed call shape: dependent-crate LLBC keeps RBigInt opaque,
        // so following the inherent Rust method would otherwise either Skip
        // or force a temporary RBigInt conversion. The exact I64 gate avoids
        // silently narrowing Rust's i128/u128 carriers, which RPython's JIT
        // cannot place in a register.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && second_arg_ty
                .as_ref()
                .is_some_and(|ty| self.tyref_literal_int_atom(ty) == Some("I64"))
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::front::rbigint_call::bigint_int_payload_residual_path(segments).or_else(
                        || {
                            segments.last().and_then(|leaf| {
                                crate::front::rbigint_call::int_binop_residual_for_method(leaf)
                            })
                        },
                    )
                }
                CallTarget::Method { name, .. } => {
                    crate::front::rbigint_call::int_binop_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // A derived `PartialEq` comparison between two FIELDLESS enums is an
        // integer comparison of their discriminants.  The enum is modelled
        // by-value as that discriminant (`tyref_to_value_type` colors it
        // `Int`), so the derived `eq` arrives as a `CallTarget::Method` whose
        // receiver already sits in the int bank — and `SomeInteger` has no
        // `eq` attribute, so leaving the call residual blocks the annotator
        // with `Cannot find attribute "eq" on Integer`.  Emit the `BinOp`
        // instead: RPython spells a C-like enum's dispatch as an `int_eq`
        // chain against named int constants (`pypy/interpreter/pyopcode.py`
        // `dispatch_bytecode`, folded by `merge_if_blocks`), never as a
        // method on the tag.  Twin of the `<str as PartialEq>::eq` fold.
        //
        // `!=` resolves to `PartialEq::ne`, which `derive` does not generate:
        // it is the trait's PROVIDED method, whose body lives in `core` and so
        // outside every extracted closure.  It therefore arrives fully
        // qualified as `<module>::<Enum>::ne` rather than as a method on the
        // receiver, and both spellings fold to the same discriminant `BinOp`.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && let Some(name) = match target {
                CallTarget::Method { name, .. } => Some(name.as_str()),
                CallTarget::FunctionPath { segments, .. } => {
                    segments.last().map(std::string::String::as_str)
                }
                _ => None,
            }
            && let Some(binop) = fieldless_enum_cmp_binop(name)
            && [first_arg_ty.as_ref(), second_arg_ty.as_ref()]
                .iter()
                .all(|t| t.is_some_and(|t| self.tyref_is_borrowed_fieldless_enum(t)))
        {
            OpKind::BinOp {
                op: binop.to_string(),
                lhs: args[0].clone().into_variable(),
                rhs: args[1].clone().into_variable(),
                result_ty: ValueType::Int,
            }
        } else {
            op_kind
        };

        // A `Method` `eq`/`ne` whose operands are already `SomeString`
        // (or a `[u8]` view of one) is the same `BinOp` as
        // `<str as PartialEq>::eq`.  Left as a method, the annotator
        // looks up `eq` on `String` and stops.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && let CallTarget::Method { name, .. } = target
            && matches!(name.as_str(), "eq" | "ne")
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| self.tyref_compares_as_string(ty))
            && second_arg_ty
                .as_ref()
                .is_none_or(|ty| self.tyref_compares_as_string(ty))
        {
            OpKind::BinOp {
                op: name.clone(),
                lhs: args[0].clone().into_variable(),
                rhs: args[1].clone().into_variable(),
                result_ty: ValueType::Int,
            }
        } else {
            op_kind
        };

        // Scalar `PartialEq` / `PartialOrd` on the rich spine: the same
        // `core::cmp::{eq..ge}` FunctionPath the adapter rewrites, or a
        // `Method` whose receiver is already in a scalar bank.  Twin of
        // the fieldless-enum fold above; strings stay on the dedicated
        // arm so they keep `ll_streq`.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && let Some(leaf) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::codewriter::minmax::cmp_binop_leaf(segments)
                }
                CallTarget::Method { name, .. } => match name.as_str() {
                    "eq" | "ne" | "lt" | "le" | "gt" | "ge" => Some(name.as_str()),
                    _ => None,
                },
                _ => None,
            }
            && {
                let lhs = first_arg_ty
                    .as_ref()
                    .and_then(|ty| self.scalar_cmp_bank(ty));
                let rhs = second_arg_ty
                    .as_ref()
                    .and_then(|ty| self.scalar_cmp_bank(ty));
                crate::codewriter::minmax::scalar_cmp_banks_compatible(
                    lhs.as_ref(),
                    rhs.as_ref(),
                    leaf,
                )
            } {
            let lhs = first_arg_ty
                .as_ref()
                .and_then(|ty| self.scalar_cmp_bank(ty));
            let rhs = second_arg_ty
                .as_ref()
                .and_then(|ty| self.scalar_cmp_bank(ty));
            OpKind::BinOp {
                op: crate::codewriter::minmax::scalar_cmp_opname(leaf, lhs.as_ref(), rhs.as_ref()),
                lhs: args[0].clone().into_variable(),
                rhs: args[1].clone().into_variable(),
                result_ty: ValueType::Int,
            }
        } else {
            op_kind
        };

        // Mixed W_LongObject/W_IntObject comparisons use rbigint.int_* in
        // both operand orders (the int-left descriptor reverses the relation).
        // The receiver remains one RBigInt GCREF, the other operand is exactly
        // Signed, and the result occupies the bool/int bank.
        let cmp_int_ty = second_arg_ty.as_ref().or(second_const_arg_ty.as_ref());
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && cmp_int_ty.is_some_and(|ty| {
                matches!(
                    self.tyref_literal_int_atom(ty),
                    Some("I8" | "I16" | "I32" | "I64" | "Isize")
                )
            })
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => segments.last().and_then(|leaf| {
                    crate::front::rbigint_call::int_comparison_residual_for_method(leaf)
                }),
                CallTarget::Method { name, .. } => {
                    crate::front::rbigint_call::int_comparison_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Bool,
            }
        } else {
            op_kind
        };

        // Zero-checked long/int leaves (`bigint_int_floordiv_nonzero`,
        // `bigint_int_modulo_int_result_nonzero`, `bigint_rshift`) are elidable
        // and either return `BigInt` by value or have no published address.
        // Retarget each to the `jit_bigint_*` wrapper that already exists.
        // The remainder wrapper returns the machine word; the other two return
        // the GC reference.
        let nonzero_int_ty = second_arg_ty.as_ref().or(second_const_arg_ty.as_ref());
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && nonzero_int_ty.is_some_and(|ty| {
                matches!(
                    self.tyref_literal_int_atom(ty),
                    Some("I8" | "I16" | "I32" | "I64" | "Isize")
                )
            })
            && let Some((residual, scalar_result)) =
                crate::front::rbigint_call::nonzero_leaf_residual_path(segments)
        {
            let result_ty = match scalar_result {
                crate::front::rbigint_call::ScalarResult::Int => ValueType::Int,
                crate::front::rbigint_call::ScalarResult::Bool => ValueType::Bool,
                crate::front::rbigint_call::ScalarResult::Ref => ValueType::Ref(None),
            };
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty,
            }
        } else {
            op_kind
        };

        // Retarget an RBigInt Rust trait shift call (`<RBigInt as Shl<usize>>::shl`,
        // …) to its `jit_bigint_{shl,shr}` residual.  Split from the binary
        // retarget above because the shift amount is a machine integer, not a
        // `BigInt`: the guard requires the first operand to be the opaque
        // `BigInt` ADT and the second to be an integer, so the residual reads
        // `b` as the shift count rather than as a `*mut BigInt` pointer.  Same
        // pure-target-swap, fail-safe contract.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|t| tyref_is_rbigint(t, self.llbc))
            && second_arg_ty.as_ref().is_some_and(|t| {
                matches!(
                    tyref_to_value_type_with(
                        t,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs
                    ),
                    ValueType::Int | ValueType::Unsigned
                )
            })
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::front::bigint_binop::bigint_shift_residual_path(segments)
                }
                CallTarget::Method { name, .. } => {
                    crate::front::bigint_binop::bigint_shift_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Retarget an RBigInt Rust trait unary call (`<RBigInt as Neg>::neg`)
        // to its `jit_bigint_neg` residual.  Split from the binary retarget
        // above because the operator takes a single `BigInt` operand: the guard
        // requires the sole operand to be the opaque `BigInt` ADT.  Same
        // pure-target-swap, fail-safe contract.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && first_arg_ty
                .as_ref()
                .is_some_and(|t| tyref_is_rbigint(t, self.llbc))
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::front::bigint_binop::bigint_unop_residual_path(segments)
                }
                CallTarget::Method { name, .. } => {
                    crate::front::bigint_binop::bigint_unop_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Retarget the interpreter's own unary RBigInt wrappers to the same
        // residuals.  A descent reaches `descroperation::bigint_neg` /
        // `bigint_invert` as an unlowered helper and never walks their
        // bodies, so the `<RBigInt as Neg>::neg` call inside is not a call the
        // retarget above ever sees.  Guarded like the divmod projections:
        // sole operand and destination both the opaque `BigInt` ADT.  Same
        // pure-target-swap, fail-safe contract.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && tyref_is_rbigint(&call.dest.ty, self.llbc)
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(residual) = crate::front::rbigint_call::unop_wrapper_residual_path(segments)
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // `RBigInt::digits{,_mut}` is only the Rust view adapter for
        // RPython's direct `self._digits` GcArray field.  Project the field
        // itself so the annotator receives the `[Signed]` SomeList registered
        // above instead of following `slice::from_raw_parts` and manufacturing
        // a classdef-less slice instance.  This is the literal rbigint.py
        // storage shape; `digits_mut` differs only in Rust borrow mutability,
        // which does not exist in the translated graph.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && match target {
                CallTarget::FunctionPath { segments, .. } => matches!(
                    segments.last().map(String::as_str),
                    Some("digits" | "digits_mut")
                ),
                CallTarget::Method { name, .. } => {
                    matches!(name.as_str(), "digits" | "digits_mut")
                }
                _ => false,
            } {
            OpKind::FieldRead {
                base: args[0].clone().into_variable(),
                field: FieldDescriptor::new("_digits", Some("RBigInt".to_string())).with_owner_id(
                    Some(majit_ir::descr::StructId::from_canonical(
                        "rbigint::RBigInt",
                    )),
                ),
                ty: ValueType::Ref(None),
                pure: true,
            }
        } else {
            op_kind
        };

        // rbigint.py marks its raw digit projections `_always_inline_ = True`.
        // A dependent-crate LLBC sees `RBigInt` as opaque and therefore drops
        // the ordinary impl-owner method hint, even though the combined
        // pyre-object LLBC owns the concrete classdef and method graph. Restore
        // that Method target after proving the exact nominal receiver so
        // MethodDesc seeds a classdef-bound `self`; otherwise `digit(0)` in
        // longobject.py's `-1 ** exponent` shortcut enters the concrete method
        // with `SomeInstance(classdef=None)` and fails on `_digits`.
        let op_kind = if let OpKind::Call {
            target,
            args,
            result_ty,
        } = &op_kind
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(leaf) = match target {
                CallTarget::FunctionPath { segments, .. } => segments.last(),
                _ => None,
            }
            && matches!(
                leaf.as_str(),
                "digit" | "widedigit" | "uwidedigit" | "udigit" | "numdigits"
            ) {
            let narrowed_receiver = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(narrowed_receiver.clone()),
                kind: crate::model::cast_instance_call("RBigInt", args[0].clone().into_variable()),
            });
            let mut narrowed_args = args.clone();
            narrowed_args[0] = narrowed_receiver.into();
            OpKind::Call {
                target: CallTarget::method(leaf.clone(), Some("RBigInt".to_string())),
                args: narrowed_args,
                result_ty: result_ty.clone(),
            }
        } else {
            op_kind
        };

        // RBigInt comparisons are `@jit.elidable` scalar residuals in
        // RPython. Keep the two GCREF operands and return an integer-bank
        // boolean instead of following Rust's trait shim into a classless
        // receiver method. `_divrem`'s lhs-remainder test takes the same
        // two-GCREF, one-boolean shape.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && second_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(residual) = match target {
                CallTarget::FunctionPath { segments, .. } => {
                    crate::front::bigint_binop::bigint_comparison_residual_path(segments).or_else(
                        || crate::front::rbigint_call::divrem_lhs_remainder_residual_path(segments),
                    )
                }
                CallTarget::Method { name, .. } => {
                    crate::front::bigint_binop::bigint_comparison_residual_for_method(name)
                }
                _ => None,
            } {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Bool,
            }
        } else {
            op_kind
        };

        // Scalar rbigint queries are pure residuals over one GCREF. This is
        // especially important for dependent-crate LLBCs, where Charon keeps
        // the nominal RBigInt type opaque and a Method getattr cannot inspect
        // its fields.
        let op_kind = if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some((segments, scalar_result)) = match target {
                CallTarget::FunctionPath { segments, .. } => segments
                    .last()
                    .and_then(|leaf| crate::front::rbigint_call::scalar_residual_for_method(leaf)),
                CallTarget::Method { name, .. } => {
                    crate::front::rbigint_call::scalar_residual_for_method(name)
                }
                _ => None,
            } {
            let result_ty = match scalar_result {
                crate::front::rbigint_call::ScalarResult::Int => ValueType::Int,
                crate::front::rbigint_call::ScalarResult::Bool => ValueType::Bool,
                crate::front::rbigint_call::ScalarResult::Ref => ValueType::Ref(None),
            };
            OpKind::Call {
                target: CallTarget::function_path(segments),
                args: args.clone(),
                result_ty,
            }
        } else {
            op_kind
        };

        // `rbigint.fromint` is `@jit.elidable` in RPython and returns one
        // GC reference. Retarget word-sized Rust constructors to wrappers
        // whose C ABI returns `*mut RBigInt`; never narrow i128/u128, which
        // RPython's JIT deliberately has no register kind for.
        //
        // A constant word argument (`RBigInt::from(1)` in `long_pow`) has no
        // place type; `const_arg_ty` is its declared type, so it retargets
        // the same way a place argument does.
        let ctor_arg_ty = first_arg_ty.as_ref().or(const_arg_ty.as_ref());
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && tyref_is_rbigint(&call.dest.ty, self.llbc)
            && let Some(arg_ty) = ctor_arg_ty
        {
            // `bool` is `frombool`: 0 or 1 in the signed word, same residual
            // as `I32`/`I64`. `U128`/`I128` stay out of this list.
            let signed = matches!(
                self.tyref_literal_int_atom(arg_ty),
                Some("I8" | "I16" | "I32" | "I64" | "Isize")
            ) || tyref_is_literal_bool(arg_ty, self.llbc);
            let unsigned = matches!(
                self.tyref_literal_uint_atom(arg_ty),
                Some("U8" | "U16" | "U32" | "U64" | "Usize")
            );
            if (signed || unsigned)
                && let Some(residual) =
                    crate::front::rbigint_call::constructor_residual_path(segments, unsigned)
            {
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: residual,
                        fun_decl_id: None,
                        specialization: crate::parse::GraphSpecialization::Regular,
                    },
                    args: args.clone(),
                    result_ty: ValueType::Ref(None),
                }
            } else {
                op_kind
            }
        } else {
            op_kind
        };

        // The interpreter has already rejected a zero divisor before these
        // quotient/remainder projections. Retarget their direct RBigInt
        // results to the two pointer-ABI halves of upstream `divmod`.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && tyref_is_rbigint(&call.dest.ty, self.llbc)
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && second_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(residual) =
                crate::front::rbigint_call::divmod_projection_residual_path(segments)
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Keep the RustPython compiler's foreign BigInt at the fixed
        // LOAD_CONST seam.  The conversion result is the same translated
        // one-GC-reference RBigInt shape as every local constructor.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && tyref_is_rbigint(&call.dest.ty, self.llbc)
            && let Some(residual) =
                crate::front::rbigint_call::compiler_bigint_residual_path(segments)
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Rust carries `rbigint.pow(..., None)` through
        // `Result<RBigInt, PyError>` in the interpreter source. RPython's
        // translated shape is instead one GCREF result plus an implicit
        // MemoryError edge. Retarget the exact host helper to that pointer ABI;
        // the Result-of-PyError capture immediately below rewires the compiler
        // generated `?` diamond into LastException exits. `bigint_add` returns
        // the payload pointer directly and takes the same GC-reference swap.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && first_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && second_arg_ty
                .as_ref()
                .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
            && let Some(residual) = crate::front::rbigint_call::pow_nomod_residual_path(segments)
                .or_else(|| crate::front::rbigint_call::bigint_add_residual_path(segments))
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Rust carries the fallible allocation of `rbigint.lshift` through a
        // `Result<RBigInt, PyError>`. RPython has a direct GCREF result plus
        // an implicit MemoryError edge, so retarget the validated
        // machine-count helper to that pointer ABI before `result_exc` erases
        // the Result diamond.  The `_make_ovf2long` seams take the same shape
        // with both operands already machine words.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && match segments.last().map(String::as_str) {
                Some("bigint_lshift_count" | "bigint_int_pow_nomod") => {
                    first_arg_ty
                        .as_ref()
                        .is_some_and(|ty| tyref_is_rbigint(ty, self.llbc))
                        && second_arg_ty.as_ref().is_some_and(|ty| {
                            tyref_to_value_type_with(
                                ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            ) == ValueType::Int
                        })
                }
                Some(
                    "bigint_lshift_int_int_result"
                    | "bigint_add_int_int"
                    | "bigint_sub_int_int"
                    | "bigint_mul_int_int",
                ) => {
                    first_arg_ty.as_ref().is_some_and(|ty| {
                        tyref_to_value_type_with(
                            ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ) == ValueType::Int
                    }) && second_arg_ty.as_ref().is_some_and(|ty| {
                        tyref_to_value_type_with(
                            ty,
                            self.llbc,
                            self.tombstoned_leaves,
                            self.gc_struct_ids,
                            &self.raw_alloc_flavor_structs,
                        ) == ValueType::Int
                    })
                }
                _ => false,
            }
            && let Some(residual) = crate::front::rbigint_call::lshift_count_residual_path(segments)
                .or_else(|| crate::front::rbigint_call::ovf2long_residual_path(segments))
                .or_else(|| crate::front::rbigint_call::pow_nomod_residual_path(segments))
        {
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: residual,
                    fun_decl_id: None,
                    specialization: crate::parse::GraphSpecialization::Regular,
                },
                args: args.clone(),
                result_ty: ValueType::Ref(None),
            }
        } else {
            op_kind
        };

        // Allocate the result Variable now. Bind it to the destination
        // local after copy-in so `p = f(&mut p)` still reads the previous
        // local, not this not-yet-defined result.
        let result_var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        // `*mut TypedItemsBlock` is `GcArray(Signed)` storage. Keep that
        // ARRAY layout on the call result so `do_fixed_newlist` reads
        // `op.result.concretetype.TO` instead of the callee's leaf name.
        if result_narrow_root
            .as_deref()
            .is_some_and(|root| root.rsplit("::").next() == Some("TypedItemsBlock"))
        {
            use crate::translator::rtyper::lltypesystem::lltype::{
                Array, LowLevelType, Ptr, PtrTarget, Struct,
            };
            let items = LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed)));
            let block = Struct::gc(
                "TypedItemsBlock",
                vec![
                    ("capacity".to_string(), LowLevelType::Unsigned),
                    ("items".to_string(), items),
                ],
            );
            result_var.set_concretetype(Some(LowLevelType::Ptr(Box::new(Ptr {
                TO: PtrTarget::Struct(block),
            }))));
        }
        // `newlist_clear` forks on `op.result.concretetype`
        // (`jtransform.py` `LIST = deref(op.result.concretetype)`).
        // A `Vec` result is `RustVecRepr.lowleveltype`
        // (`rrustvec.rs rust_vec_lltype`), kind `int`. Without that
        // pointer the arm emits GC `new_array_clear`, whose result
        // bank is `ref`.
        if result_var.concretetype().is_none() {
            stamp_rust_vec_header(&result_var, &call.dest.ty, self.llbc);
        }
        // A Ref-banked items-block result (`rlist.py` `ll_items`) keeps
        // the `tyref_to_value_type_with` record; CCHARP is Int-banked.
        if !callee_returns_gc_block && !graph_is_items_block_base_accessor(&self.graph.name) {
            stamp_raw_u8_ccharp(&result_var, &call.dest.ty, self.llbc);
        }
        // A call returning `&T` / `&mut T` for a primitive `T` yields the
        // address, not the word `Rvalue::Ref` would have aliased.
        // `*dest` is then `rewrite_op_raw_load` and `history.getkind` of
        // the loaded bool / int is `int`. A raw pointer stays on the
        // raw-pointer arm and is not recorded here.
        // The destination local stays bound to the previous value until
        // after copy-in, so `p = f(&mut p)` still reads that value.
        if matches!(op_kind, OpKind::Call { .. })
            && !self.multi_assigned_locals.contains(&dest_local)
            && !tyref_is_raw_pointer(&call.dest.ty, self.llbc)
            && tyref_scalar_pointee_value_type(&call.dest.ty, self.llbc).is_some()
            && !self.scalar_address_locals.contains(&dest_local)
        {
            self.scalar_address_locals.push(dest_local);
        }
        // Last-write-wins: a residual Call is not `as_bytes` /
        // `w_str_get_wtf8`.  Those arms mark dest and return above.
        // A STR-view receiver whose dest is still a string or a shared
        // `&[u8]` keeps the mark: `&view[a..b]` is `__getslice_*` of the
        // header (`rstr.py` `LLHelpers.ll_stringslice_startstop`).
        if first_arg_is_string_byte_view && tyref_is_one_word_str_view(&call.dest.ty, self.llbc) {
            if !self.string_byte_view_locals.contains(&dest_local) {
                self.string_byte_view_locals.push(dest_local);
            }
            stamp_as_rstr_str(&result_var);
        } else {
            self.string_byte_view_locals
                .retain(|&local| local != dest_local);
        }
        // `Result::expect`. The callee is that result-module item:
        // `core::result::<Impl>::expect` on the `Method` spelling, and
        // `result::Result::expect` when a raw `&Result` takes the static
        // `FunctionPath` (`impl_method_owner` declines the method hint).
        // `as_ref` is a regular call to the extracted core body
        // (`match *self { Ok(ref x) => Ok(x), Err(ref x) => Err(x) }`).
        let result_expect_callee = callee_name_path.as_deref().is_some_and(|path| {
            let segs: Vec<&str> = path.split("::").collect();
            *segs.last().unwrap_or(&"") == "expect"
                && matches!(
                    segs.as_slice(),
                    ["core" | "std", "result", "<Impl>" | "Result", _] | ["result", "Result", _]
                )
        });
        if result_expect_callee
            && let OpKind::Call { args, .. } = &op_kind
            && args.len() == 2
        {
            if let Some(site) = self.recognize_carrier_combinator_site(
                first_arg_ty.as_ref(),
                &result_var,
                crate::front::result_exc::CarrierCatchKind::RaiseImplicit,
            ) {
                self.carrier_combinator_sites.push(site);
            } else if first_arg_ty
                .as_ref()
                .is_some_and(|ty| crate::front::result_exc::tyref_is_result(ty, self.llbc))
                && let Some(site) = self.recognize_unwrap_site(first_arg_ty.as_ref(), &result_var)
            {
                self.result_expect_sites.push(site);
            }
        }
        // `Option::ok_or_else(opt, closure)?` is one combined value-or-raise
        // boundary in the translated graph.  Record it separately from an
        // ordinary scoped `Result` call: `ok_or_else` is a foreign core
        // combinator, and lowering only the downstream `?` would leave that
        // unregistered residual (while lowering only the Option select would
        // materialise the Result shell RPython never has).
        let captured_option_ok_or_else_try = if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && name == "ok_or_else"
            && args.len() == 2
            && let Some(site) = self.recognize_option_ok_or_else_try_site(
                first_arg_ty.as_ref(),
                second_arg_ty.as_ref(),
                &call.dest.ty,
                &result_var,
            ) {
            self.option_ok_or_else_try_sites.push(site);
            true
        } else {
            false
        };
        // `Option::ok_or_else` is a combinator (`DiscCombinator::OptionOkOrElse`),
        // not a transformed `Result<T, PyError>` callee.  Capturing it as a
        // scoped call remints the residual Method as a payload and leaves the
        // foreign `ok_or_else` in the graph.
        let option_ok_or_else_method = matches!(
            &op_kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                args,
                ..
            } if name == "ok_or_else" && args.len() == 2
        ) && first_arg_ty.as_ref().is_some_and(|ty| {
            crate::front::result_exc::tyref_is_option(ty, self.llbc)
                || self.peel_to_option_or_result(ty).is_some_and(|peeled| {
                    crate::front::result_exc::tyref_is_option(&peeled, self.llbc)
                })
        });
        // Capture scoped `Result<T, PyError>` call results for the
        // `?`-diamond rewiring pass (`front::result_exc`) that runs
        // after the body lowering completes.
        //
        // A dyn vtable slot is `OpKind::IndirectCall` with `family_key`
        // set. Trait methods stamp FUNC.RESULT as the Ok payload
        // (`dont_look_inside_return_token`). A scalar payload lives in
        // a different bank from the Result shell, so the same capture
        // unwraps that call. A reference payload already shares the
        // shell's Ref bank.
        let indirect_family_call = matches!(
            &op_kind,
            OpKind::IndirectCall {
                family_key: Some(_),
                ..
            }
        );
        if (matches!(&op_kind, OpKind::Call { .. }) && callee_name_path.is_some()
            || indirect_family_call)
            && !captured_option_ok_or_else_try
            && !option_ok_or_else_method
            && crate::front::result_exc::tyref_is_result_of_carrier(
                &call.dest.ty,
                self.llbc,
                self.static_addrs.error_carrier,
            )
        {
            // The per-instantiation suffix of the callee's `Result<T,
            // PyError>` keys the shells `result_exc` rebuilds at the
            // rewrap site, matching the front aggregate path's suffix so
            // both writers share one ClassDef per instantiation.
            let suffix = crate::front::result_exc::tyref_result_instantiation_suffix(
                &call.dest.ty,
                self.llbc,
            );
            // `Result<(), PyError>`: the transformed callee returns void
            // (`tyref_result_ok_is_unit`), so the payload is `Void`, not the
            // unit tuple's aggregate `Ref`.
            let payload_ty =
                if crate::front::result_exc::tyref_result_ok_is_unit(&call.dest.ty, self.llbc) {
                    ValueType::Void
                } else {
                    crate::front::result_exc::tyref_result_ok(&call.dest.ty, self.llbc)
                        .map(|ty| {
                            tyref_enum_payload_value_type(
                                &ty,
                                self.llbc,
                                self.tombstoned_leaves,
                                self.gc_struct_ids,
                                &self.raw_alloc_flavor_structs,
                            )
                        })
                        .unwrap_or(ValueType::Ref(None))
                };
            let payload_shares_shell_bank = matches!(
                payload_ty,
                ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder
            );
            if !indirect_family_call || !payload_shares_shell_bank {
                self.result_exc_call_results
                    .push((result_var.clone(), suffix, payload_ty));
            }
        }
        // Capture `Iterator::next()` results (`Option<T>`-typed) for the
        // `next`-diamond rewiring pass (`front::iter_next`).  Recognition
        // is liberal — any `next`-leaf call returning `Option` — because
        // the rewrite itself validates the surrounding for-loop match and
        // declines (leaving the residual call) on any other shape.
        if let OpKind::Call { target, .. } = &op_kind
            && crate::front::iter_next::is_iterator_next_target(target)
            && crate::front::result_exc::tyref_is_option(&call.dest.ty, self.llbc)
        {
            // A slice iterator yields `Option<&T>`; peel that one `&` so the
            // recorded kind is the item's own, the way `ll_listnext` hands
            // back the list's item repr rather than a pointer to it — and
            // only that one, so an element that is itself a reference stays
            // reference-typed.  Which iterator this is decides whether there
            // is a reference to peel at all: the receiver names the iterator
            // ADT, and a by-value one yields the element directly.  An
            // unreadable shape records `Ref(None)` — the answer the fold
            // assumed unconditionally before this was carried, so an
            // unreadable type keeps today's behaviour instead of inventing a
            // new one.
            let enumerate_next = crate::front::iter_adapter::is_enumerate_next_target(target)
                || first_arg_ty.as_ref().is_some_and(|receiver| {
                    self.tyref_ref_adt_path(receiver)
                        .or_else(|| {
                            self.tyref_peel_ref_to_pointee(receiver)
                                .and_then(|ty| adt_path_of_tyref(&ty, self.llbc))
                        })
                        .is_some_and(|path| {
                            crate::front::iter_adapter::is_enumerate_adapter_path(&path)
                        })
                });
            let iterator_added_a_reference = if enumerate_next {
                // `Enumerate<I>::next` yields `(usize, I::Item)`; the native
                // `next` runs on `I`, so the reference-peel follows I, not
                // the adapter.  `Enumerate`'s first type argument is I.
                first_arg_ty
                    .as_ref()
                    .and_then(|receiver| self.tyref_peel_ref_to_pointee(receiver))
                    .and_then(|enum_ty| {
                        let body = match &enum_ty {
                            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
                            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
                        };
                        let inner =
                            type_decl_ref_generics(body.get("Adt")?.as_object()?, self.llbc)?
                                .get("types")?
                                .get(0)?;
                        let inner_ty = serde_json::from_value::<TyRef>(inner.clone()).ok()?;
                        self.tyref_ref_adt_path(&inner_ty)
                    })
                    .is_some_and(|path| iterator_adds_a_reference(&path))
            } else {
                first_arg_ty
                    .as_ref()
                    .and_then(|receiver| self.tyref_ref_adt_path(receiver))
                    .is_some_and(|path| iterator_adds_a_reference(&path))
            };
            let item_ty = crate::front::result_exc::tyref_option_payload(&call.dest.ty, self.llbc)
                .and_then(|payload| {
                    let body = match &payload {
                        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
                        TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
                    };
                    // `Option<(usize, I::Item)>` — the recorded kind is the
                    // tuple's item, peeled like `I::next`'s own payload.
                    let body = if enumerate_next {
                        type_decl_ref_generics(body.get("Adt")?.as_object()?, self.llbc)?
                            .get("types")?
                            .get(1)?
                    } else {
                        body
                    };
                    let item =
                        iterator_payload_element(body, self.llbc, iterator_added_a_reference)?;
                    serde_json::from_value::<TyRef>(item.clone()).ok()
                })
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
            self.next_call_results.push((result_var.clone(), item_ty));
        }
        // Capture `RangeInclusive::new(lo, hi)` over an integer element so
        // `front::range_iter` can divert `for _ in lo..=hi` to `range(lo, hi + 1)`.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && crate::front::range_iter::is_range_inclusive_new_segments(segments)
            && args.len() == 2
            && let Some((narrow, unsigned)) = {
                let body = match &call.dest.ty {
                    TyRef::Inline { value: (_, v) } | TyRef::Other(v) => Some(v),
                    TyRef::Dedup { id } => self.llbc.dedup_body(*id),
                };
                body.and_then(|body| strip_ty_indirections(body, self.llbc))
                    .and_then(|body| {
                        let head = body.get("Adt")?.as_object()?;
                        self.int_range_head_narrow(head, "core::ops::range::RangeInclusive")
                    })
            }
        {
            self.range_iter_new_sites
                .push(crate::front::range_iter::RangeNewSite {
                    result_var: result_var.clone(),
                    kind: crate::front::range_iter::RangeSiteKind::Inclusive { narrow, unsigned },
                });
        }
        // Capture `Map::collect` at the construction site, where the
        // closure env is a concrete ADT.  The adapter body is
        // polymorphic (Charon does not monomorphise per closure); the
        // caller is not.  A miss — non-Map receiver, non-Vec dest, or
        // env that does not resolve to an ADT — leaves the residual.
        if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && crate::front::iter_adapter::is_map_collect_target(target)
            && tyref_is_vec_value(&call.dest.ty, self.llbc)
            && let Some(recv_ty) = first_arg_ty.as_ref()
            && adt_path_of_tyref(recv_ty, self.llbc)
                .as_deref()
                .is_some_and(crate::front::iter_adapter::is_map_adapter_path)
            && let Some(i_ty) = self.tyref_adt_type_arg(recv_ty, 0)
            && let Some(f_ty) = self.tyref_adt_type_arg(recv_ty, 1)
            && let Some(env_def_id) = self
                .tyref_adt_def_id(&f_ty)
                .or_else(|| self.tyref_ref_adt_def_id(&f_ty))
            && let Some(env_td) = self.llbc.type_by_id(env_def_id)
        {
            let call_once_owner = closure_env_call_once_owner(env_td);
            if !call_once_owner.is_empty() {
                let iter_ty = match self.tyref_peel_ref_to_pointee(&i_ty) {
                    Some(ty) => ty,
                    None => i_ty,
                };
                let item_tyref = self
                    .tyref_adt_type_arg(&iter_ty, 0)
                    .unwrap_or_else(|| clone_tyref(&iter_ty));
                let payload_ty = tyref_to_value_type_with(
                    &item_tyref,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                let payload_class_root = enum_payload_instance_class_root(
                    &item_tyref,
                    self.llbc,
                    self.tombstoned_leaves,
                );
                let args_tuple_suffix = payload_tuple_suffix(&item_tyref, self.llbc);
                let elem_tyref = self
                    .tyref_adt_type_arg(&call.dest.ty, 0)
                    .unwrap_or_else(|| clone_tyref(&call.dest.ty));
                let call_result_ty = tyref_to_value_type_with(
                    &elem_tyref,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                );
                self.map_collect_sites
                    .push(crate::front::iter_adapter::MapCollectSite {
                        result_var: result_var.clone(),
                        call_once_owner,
                        payload_ty: payload_ty.clone(),
                        payload_class_root,
                        args_tuple_suffix,
                        call_result_ty,
                        inner_item_ty: payload_ty,
                        rust_vec_kind: tyref_rust_vec_item_kind(&call.dest.ty, self.llbc),
                    });
            }
        }
        // Transparent std wrappers (`Cell::{new,get}`, `Atomic*::new`,
        // `Box::as_ref`, `Ref`/`MutexGuard` deref) are operations, not
        // residual calls.  Rewrite the Call before the capture gates
        // below see it.
        let identity_recv = first_arg_ty
            .as_ref()
            .and_then(|ty| adt_path_of_tyref(ty, self.llbc));
        let identity_dest = adt_path_of_tyref(&call.dest.ty, self.llbc);
        let identity_dest_ty = tyref_to_value_type_with(
            &call.dest.ty,
            self.llbc,
            self.tombstoned_leaves,
            self.gc_struct_ids,
            &self.raw_alloc_flavor_structs,
        );
        let dest_is_bool = matches!(identity_dest_ty, ValueType::Bool);
        // A borrow carries no representation of its own here -- `Rvalue::Ref`
        // aliases the place's Variable -- so the receiver's bank is the
        // pointee's.  `Box`/`Ref`/`MutexGuard` still read `Ref` through it
        // and keep their identity; a wrapper the LLBC leaves Opaque reads
        // `Ref` against a scalar destination and declines, because that
        // member is a load, not an alias.
        let identity_banks_agree = first_arg_ty.as_ref().is_some_and(|ty| {
            let recv_ty = match self.tyref_peel_ref_to_pointee(ty) {
                Some(pointee) => tyref_to_value_type_with(
                    &pointee,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
                None => tyref_to_value_type_with(
                    ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
            };
            value_type_bank(&recv_ty) == value_type_bank(&identity_dest_ty)
        });
        let identity_layout = if matches!(
            &op_kind,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } if segments.last().is_some_and(|leaf| leaf == "new")
        ) {
            self.adt_struct_fields(&call.dest.ty)
        } else {
            first_arg_ty
                .as_ref()
                .and_then(|ty| self.adt_struct_fields(ty))
        };
        // Only a thin `Box<T>` is one pointer word. `Box<[T]>`, `Box<str>` and
        // `Box<dyn Trait>` carry a metadata word that a `same_as` would drop.
        let receiver_is_thin_box = first_arg_ty.as_ref().is_some_and(|ty| {
            let peeled = self
                .tyref_peel_ref_to_pointee(ty)
                .unwrap_or_else(|| ty.clone());
            tyref_node(&peeled, self.llbc)
                .is_some_and(|node| type_node_is_thin_box(node, self.llbc))
        });
        let op_kind = crate::front::std_identity::lower_std_primitive_op(
            op_kind,
            identity_recv.as_deref(),
            identity_dest.as_deref(),
            self.tyref_literal_int_atom(&call.dest.ty),
            self.tyref_literal_uint_atom(&call.dest.ty),
            dest_is_bool,
            identity_banks_agree,
            identity_layout.as_deref(),
            receiver_is_thin_box,
        );
        // `Option::as_deref` / `as_deref_mut` on a null-niche
        // `Option<Box<T>>` is the pointer word (`Box::as_mut` is the
        // same identity on the box itself). `None` stays null.
        let niche_option_ptr = first_arg_ty.as_ref().is_some_and(|ty| {
            let peeled = self
                .tyref_peel_ref_to_pointee(ty)
                .unwrap_or_else(|| ty.clone());
            self.tyref_is_niche_option_ptr(&peeled)
        });
        // `CallTarget::Method` keeps the leaf. The inherent impl's owner
        // ADT is on the FunDecl (`impl_method_owner_for_fundecl`):
        // `option::Option::as_deref` / `as_deref_mut`.
        let decl_is_option_deref = match &op_kind {
            OpKind::Call { target, .. } => target
                .fun_decl_id()
                .and_then(|id| self.llbc.fn_by_id(id))
                .and_then(|fd| impl_method_owner_for_fundecl(self.llbc, fd))
                .is_some_and(|(owner, leaf)| {
                    (owner == "option::Option" || owner == "core::option::Option")
                        && matches!(leaf.as_str(), "as_deref" | "as_deref_mut")
                }),
            _ => false,
        };
        let op_kind = crate::front::std_identity::lower_niche_option_deref(
            op_kind,
            niche_option_ptr,
            decl_is_option_deref,
        );
        // Capture `i64::checked_{add,sub,mul}()` results (`Option<i64>`-
        // typed) for the checked-arith rewiring pass
        // (`front::checked_arith`), which rewrites each into the native
        // `*_ovf` op + OverflowError edge.  Recognition is liberal about the
        // SHAPE — any `core::num::<Impl>::checked_*` call returning `Option`
        // — because the rewrite itself validates the surrounding
        // overflow-fallback match and declines (leaving the residual call) on
        // any other shape.  It is not liberal about the WIDTH: the operand
        // atoms are only in hand here (Charon renders every inherent integer
        // impl as `<Impl>`, and `tyref_to_value_type` colors every signed
        // width `Int`), and the emitted `*_ovf` tests the machine-word bound,
        // so a narrower operand would answer the wrong question.  A literal
        // operand (`needed.checked_add(63)`) has no Place type, so the
        // destination `Option<Self>` payload is the width; either operand
        // is only a fallback.  Same placement rationale as the unsigned
        // pass's signedness gate below.
        if let OpKind::Call { target, .. } = &op_kind
            && crate::front::checked_arith::is_checked_arith_target(target)
            && crate::front::result_exc::tyref_is_option(&call.dest.ty, self.llbc)
            && {
                let dest_payload =
                    crate::front::result_exc::tyref_option_payload(&call.dest.ty, self.llbc);
                let dest_atom = dest_payload
                    .as_ref()
                    .and_then(|ty| self.tyref_literal_int_atom(ty));
                let peel0 = first_arg_ty
                    .as_ref()
                    .and_then(|ty| self.tyref_peel_ref_to_pointee(ty));
                let peel1 = second_arg_ty
                    .as_ref()
                    .and_then(|ty| self.tyref_peel_ref_to_pointee(ty));
                let op0 = first_arg_ty
                    .as_ref()
                    .and_then(|ty| self.tyref_literal_int_atom(ty))
                    .or_else(|| {
                        peel0
                            .as_ref()
                            .and_then(|ty| self.tyref_literal_int_atom(ty))
                    });
                let op1 = second_arg_ty
                    .as_ref()
                    .and_then(|ty| self.tyref_literal_int_atom(ty))
                    .or_else(|| {
                        peel1
                            .as_ref()
                            .and_then(|ty| self.tyref_literal_int_atom(ty))
                    });
                crate::front::checked_arith::signed_ovf_width_reaches(dest_atom, [op0, op1])
            }
        {
            self.checked_arith_call_results.push(result_var.clone());
        }
        // Capture `Option::ok_or_else(opt, closure)` as a possible immediate
        // continuation of signed checked arithmetic.  The post-pass still
        // validates the producer/consumer CFG adjacency; this site only keeps
        // the closure and Result type ownership that is unavailable after MIR
        // lowering.
        if let OpKind::Call {
            target:
                CallTarget::Method {
                    name,
                    receiver_root,
                    ..
                },
            args,
            ..
        } = &op_kind
            && name == "ok_or_else"
            && receiver_root.as_deref().is_some_and(|root| {
                majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(root))
                    == "Option"
            })
            && args.len() == 2
            && let Some(closure_arg) = args.get(1).and_then(|arg| arg.as_variable()).cloned()
            && let Some(site) = self.recognize_checked_arith_ok_or_else_site(
                first_arg_ty.as_ref(),
                second_arg_ty.as_ref(),
                &call.dest.ty,
                &result_var,
                closure_arg,
            )
        {
            self.checked_arith_ok_or_else_sites.push(site);
        }
        // Capture UNSIGNED `usize::checked_{add,sub,mul}()` results
        // (`Option<usize>`) for the native-overflow rewiring pass
        // (`front::checked_arith_uint`).  Signed arithmetic stays on
        // `front::checked_arith` (→ `*_ovf` + OverflowError); unsigned has no
        // signed-overflow op, so it lowers to `uint_mul_high` / `uint_lt`
        // overflow tests + a virtualized `Option`.  The operand-signedness
        // gate lives here because the operand types are only in hand at the
        // call site.
        //
        // Either operand answers for both: these are
        // `fn(self, rhs: Self) -> Option<Self>`, so the two are one type.  A
        // literal operand carries no `Place` to read a type from, which is
        // what `n.checked_sub(1)` spells, so demanding that both resolve would
        // leave the constant-operand form on the residual path for want of a
        // type the callee's signature already fixes.  Both-const
        // (`1usize.checked_add(2)`) has no operand Place at all; the
        // destination `Option<Self>` payload is then the width.
        if let OpKind::Call { target, .. } = &op_kind
            && let CallTarget::FunctionPath { segments, .. } = target
            && matches!(
                segments.last().map(String::as_str),
                Some("checked_add" | "checked_sub" | "checked_mul")
            )
            && crate::front::checked_arith::is_checked_arith_target(target)
            && crate::front::result_exc::tyref_is_option(dest_class_ty, self.llbc)
        {
            let dest_payload =
                crate::front::result_exc::tyref_option_payload(dest_class_ty, self.llbc).or_else(
                    || crate::front::result_exc::tyref_option_payload(&call.dest.ty, self.llbc),
                );
            let dest_atom = dest_payload
                .as_ref()
                .and_then(|ty| self.tyref_literal_uint_atom(ty));
            let peel0 = first_arg_ty
                .as_ref()
                .and_then(|ty| self.tyref_peel_ref_to_pointee(ty));
            let peel1 = second_arg_ty
                .as_ref()
                .and_then(|ty| self.tyref_peel_ref_to_pointee(ty));
            let op0 = first_arg_ty
                .as_ref()
                .and_then(|ty| self.tyref_literal_uint_atom(ty))
                .or_else(|| {
                    peel0
                        .as_ref()
                        .and_then(|ty| self.tyref_literal_uint_atom(ty))
                });
            let op1 = second_arg_ty
                .as_ref()
                .and_then(|ty| self.tyref_literal_uint_atom(ty))
                .or_else(|| {
                    peel1
                        .as_ref()
                        .and_then(|ty| self.tyref_literal_uint_atom(ty))
                });
            if crate::front::checked_arith_uint::unsigned_word_atom(dest_atom, [op0, op1]).is_some()
            {
                let payload_hint = dest_payload
                    .as_ref()
                    .filter(|ty| !tyref_is_type_parameter(ty, self.llbc))
                    .or(peel0.as_ref())
                    .or(peel1.as_ref())
                    .or(first_arg_ty.as_ref())
                    .or(second_arg_ty.as_ref());
                if let Some(site) =
                    self.recognize_checked_arith_uint_site(dest_class_ty, &result_var, payload_hint)
                {
                    self.checked_arith_uint_sites.push(site);
                }
            }
        }
        // Word-sized `{u64,usize}::saturating_mul`.  Narrow unsigned
        // saturating mul is not a word high-word test (`u32::MAX *
        // u32::MAX` fits in u64), so the dest atom is required here —
        // the same width gate as unsigned `checked_mul`.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && fmt_path_ends_with(segments, &["num", "<Impl>", "saturating_mul"])
            && self
                .tyref_literal_uint_atom(&call.dest.ty)
                .is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom)
        {
            self.saturating_mul_sites.push(result_var.clone());
        }
        // Capture `Layout::from_size_align(size, align)` itself
        // (`Result<Layout, LayoutError>`) for the producer-side rewrite
        // (`front::from_size_align`).  Opaque core, no extracted body; the
        // rewrite is int arithmetic plus a virtualized Result, the same
        // shape `checked_arith_uint` uses for `checked_mul`.  RPython has
        // no Layout object (`lltype.malloc(..., flavor='raw')` /
        // `llmemory.raw_malloc(size)`).  Downstream `Result::ok` / `expect`
        // / `unwrap` consume that Result.  A miss leaves the residual Call
        // for the existing Skip fallback.
        if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && crate::front::from_size_align::is_layout_from_size_align_target(target)
            && let Some(site) = self
                .recognize_from_size_align_call_site(dest_class_ty, &result_var)
                .or_else(|| self.recognize_from_size_align_call_site(&call.dest.ty, &result_var))
        {
            self.from_size_align_call_sites.push(site);
        }
        // Capture `Layout::size` / `Layout::align` on a virtualized
        // `{size, align}` aggregate.  Both are opaque core accessors with
        // no extracted body; the rewrite is a `__pos_0` / `__pos_1`
        // FieldRead.  The receiver type must peel to the Layout ADT.
        if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && let Some(field) = crate::front::from_size_align::layout_accessor_field(target)
            && let Some(site) =
                self.recognize_layout_accessor_site(field, first_arg_ty.as_ref(), &result_var)
        {
            self.layout_accessor_sites.push(site);
        }
        // Capture `Result::ok()` results whose payload is `Layout`
        // (`Option<Layout>`) for the `from_size_align` bound-check rewiring pass
        // (`front::from_size_align`).  The `.ok()` is the second residual of
        // `Layout::from_size_align(size, align).ok()`; the destination
        // `Option<Layout>` type resolves the `Option`/`Some`/`Layout` owners
        // here, and the post-pass validates the preceding `from_size_align`
        // residual + folded-const align before mutating.  A miss leaves both
        // residual calls for the existing Skip fallback.
        // Method-hint (`Result::ok`) and FunDecl (`result::<Impl>::ok`)
        // spellings both record: the rewriter already accepts both.
        if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 1
            && crate::front::slice_get::is_result_ok_call_target(target)
            && crate::front::result_exc::tyref_is_option(&call.dest.ty, self.llbc)
            && let Some(site) = self.recognize_from_size_align_ok_site(&call.dest.ty, &result_var)
        {
            self.from_size_align_sites.push(site);
        }
        // Capture `Result::expect(res, msg)` results whose payload is `Layout`
        // (the by-value `Layout`) for the `from_size_align` bound-check rewiring
        // pass (`front::from_size_align`).  The `.expect()` is the second
        // residual of `Layout::from_size_align(size, align).expect(msg)`; unlike
        // `.ok()`, the destination is `Layout` (not `Option<Layout>`), so the
        // receiver is `Result` (not `Option`) — distinguishing it from
        // `Option::expect`.  The post-pass validates the preceding
        // `from_size_align` residual before mutating; a miss leaves both
        // residual calls for the existing Skip fallback.
        if let OpKind::Call { target, args, .. } = &op_kind
            && args.len() == 2
            && crate::front::slice_get::is_result_expect_call_target(target)
            && let Some(site) =
                self.recognize_from_size_align_expect_site(&call.dest.ty, &result_var)
        {
            self.from_size_align_expect_sites.push(site);
        }
        // Capture `Try::branch(opt)` sites where the receiver is an
        // `Option<T>`, the compiler-generated core call behind `opt?`.  The
        // rewrite validates the surrounding `ControlFlow` diamond and the
        // enclosing function's Option return before mutating; a miss leaves the
        // residual `branch` call for the existing Skip fallback.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && name == "branch"
        {
            if let Some(site) = self.recognize_option_try_site(
                first_arg_ty.as_ref(),
                &result_var,
                args[0].as_variable(),
            ) {
                self.option_try_sites.push(site);
            } else if let Some(site) =
                self.recognize_result_try_site(first_arg_ty.as_ref(), &result_var)
            {
                self.result_try_sites.push(site);
            }
        }
        // Capture `bool::then(cond, closure_env)` sites for the
        // short-circuit `Option` diamond `front::bool_then` synthesizes.
        // `then_some` is a regular call to the extracted core body.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && fmt_path_ends_with(segments, &["bool", "<Impl>", "then"])
            && let Some(site) =
                self.recognize_bool_then_site(&call.dest.ty, second_arg_ty.as_ref(), &result_var)
        {
            self.bool_then_sites.push(site);
        }
        // Capture `<[T]>::first(slice)` sites for the length-checked
        // `Option<&T>` diamond `front::slice_first` synthesizes.  `first` is a
        // foreign leaf (its `Self` is the primitive slice `[T]`, so `lower_call`
        // keeps the raw `FunctionPath` segments), residualized as an
        // unregistered callee the rtyper census Skips; a resolution miss leaves
        // the residual call.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && fmt_path_ends_with(segments, &["slice", "<Impl>", "first"])
            && let Some(mut site) = self.recognize_slice_first_site(&call.dest.ty, &result_var)
        {
            self.annotate_slice_first_site(
                &mut site,
                &arg_locals,
                first_arg_ty.as_ref(),
                slice_first_element.clone(),
            );
            site.object_array_type_id = slice_object_array_type_id.clone();
            self.slice_first_sites.push(site);
        }
        // `<[T]>::last(slice)` is `first` with index `len-1` and the
        // same non-empty guard.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && fmt_path_ends_with(segments, &["slice", "<Impl>", "last"])
            && let Some(mut site) = self.recognize_slice_first_site(&call.dest.ty, &result_var)
        {
            site.access = crate::front::slice_first::SliceAccess::Last;
            self.annotate_slice_first_site(
                &mut site,
                &arg_locals,
                first_arg_ty.as_ref(),
                slice_first_element.clone(),
            );
            site.object_array_type_id = slice_object_array_type_id.clone();
            self.slice_first_sites.push(site);
        }
        // Capture `<[T]>::get(slice, i)` / `get_mut` sites for the
        // bounds-checked `Option<&T>` diamond `front::slice_get` synthesizes.
        // `get` is the general case of `first` — the same foreign leaf, the
        // same raw `FunctionPath` segments and the same unregistered-callee
        // census Skip — with the subscript in `args[1]`.  `get_mut` is the
        // same two-argument call; the rewriter already consumes that shape.
        // `recognize_slice_get_site` pins the scalar `SliceIndex` instantiation
        // off that operand's type, so a range instantiation (whose payload is
        // a sub-slice) and any other resolution miss both leave the residual
        // call.  `slice_get_element` is scalar / thin-pointer only: an
        // `AtomicU64` or `Option<Entry>` element never produces a proof, and
        // `get_mut` of a scalar is dropped because `ArrayRead` would copy it.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && crate::front::slice_get::is_slice_get_segments(segments)
            && slice_get_index_is_scalar
            // `first`/`last` carry the mark onto the site and answer a byte
            // view with `__strlen` plus `__string_byte_getitem`; the scalar
            // `get` rewriter has neither, so its diamond would read the string
            // object as a GC array.  Leave the residual call.
            && !self.slice_receiver_is_string_byte_view(&arg_locals, first_arg_ty.as_ref())
            && let Some((item_ty, array_type_id)) = slice_get_element.clone()
            && let Some(mut site) =
                self.recognize_slice_get_site(&call.dest.ty, &result_var, item_ty, array_type_id)
        {
            site.object_array_type_id = slice_object_array_type_id.clone();
            self.slice_get_sites.push(site);
        }
        // Capture `<[u8]>::is_ascii(slice)` sites for the
        // `rutf8.check_ascii` rewrite (`first_non_ascii_char(s) < 0`).
        // `is_ascii` is a foreign leaf on the primitive `[u8]` (FunDecl
        // `core::slice::ascii::<Impl>::is_ascii`), residualized as an
        // unregistered callee the rtyper census Skips; a resolution miss
        // leaves the residual call.  The receiver's `as_bytes()` mark
        // picks the StringRepr helper vs the `[u8]` GcArray helper.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && crate::front::slice_ascii::is_slice_is_ascii_segments(segments)
        {
            let string_byte_view =
                self.slice_receiver_is_string_byte_view(&arg_locals, first_arg_ty.as_ref());
            self.slice_is_ascii_sites
                .push(crate::front::slice_ascii::SliceIsAsciiSite {
                    result_var: result_var.clone(),
                    string_byte_view,
                });
        }
        // `<[T]>::get(slice, start..)` returns an Option subslice.  Preserve
        // that Option shape with the same guarded diamond as scalar `get`, but
        // make the successful payload RPython's `getslice(start, None)`.  The
        // RangeFrom aggregate was captured at construction time; exact value
        // identity keeps this arm separate from arbitrary SliceIndex impls.
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && fmt_path_ends_with(segments, &["slice", "<Impl>", "get"])
            && !slice_get_index_is_scalar
            && let Some(range_site) = self
                .slice_index_rangefrom_sites
                .iter()
                .find(|range_site| range_site.range_result == args[1])
                .cloned()
            && let Some(mut site) = self.recognize_slice_first_site(&call.dest.ty, &result_var)
        {
            site.access = crate::front::slice_first::SliceAccess::RangeFrom {
                range: range_site.range_result,
                start: range_site.start,
            };
            self.annotate_slice_first_site(&mut site, &arg_locals, first_arg_ty.as_ref(), None);
            site.object_array_type_id = slice_object_array_type_id.clone();
            self.slice_first_sites.push(site);
        }
        // Capture carrier `Result<T, PyError>::unwrap_or(res, default)` for
        // `front::result_exc` LastException fusion (`except OperationError`).
        // Plain Option / Result stay residual Method calls whose core bodies
        // CHARON_CORE_INCLUDE translates. A non-carrier miss leaves that call.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && name == "unwrap_or"
        {
            if let Some(site) = self.recognize_carrier_combinator_site(
                first_arg_ty.as_ref(),
                &result_var,
                crate::front::result_exc::CarrierCatchKind::JoinDefault,
            ) {
                self.carrier_combinator_sites.push(site);
            }
        }
        // Capture `Option::unwrap(opt)` / plain-error `Result::unwrap(res)`
        // sites for the discriminant guard `front::option_unwrap` synthesizes.
        // Like `unwrap_or`, `unwrap`'s body is Opaque (foreign `core`) and its
        // receiver is the enum ADT, so `first_is_self` routes it to a
        // `CallTarget::Method` (receiver in `args[0]`, the sole argument).
        // The exception-carrier Result is a `CarrierCombinatorSite` —
        // `front::result_exc` fuses `unwrap` onto `LastException` with an
        // implicit-`AssertionError` catch (`core` `Result::unwrap`;
        // `simplify.py` `remove_assertion_errors`). Plain-error Result and
        // Option keep the discriminant guard. A resolution miss leaves the
        // residual call — an unregistered callee the rtyper census Skips.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 1
            && name == "unwrap"
        {
            if let Some(site) = self.recognize_carrier_combinator_site(
                first_arg_ty.as_ref(),
                &result_var,
                crate::front::result_exc::CarrierCatchKind::RaiseImplicit,
            ) {
                self.carrier_combinator_sites.push(site);
            } else if let Some(site) =
                self.recognize_unwrap_site(first_arg_ty.as_ref(), &result_var)
            {
                self.unwrap_sites.push(site);
            }
        }
        // Capture `Option::expect(opt, msg)` and carrier
        // `Result::expect(res, msg)` sites. Option keeps the discriminant
        // guard `front::option_expect` synthesizes — the two-arg sibling of
        // `unwrap`. Its body is Opaque (foreign `core`) and its receiver is
        // the `Option` ADT, so `first_is_self` routes it to a
        // `CallTarget::Method` (receiver `args[0]`, panic message `args[1]`).
        // The exception-carrier Result is a `CarrierCombinatorSite`
        // (`interp_deque.py` `W_Deque.trimleft` / `popleft`; `core`
        // `Result::expect`; the `msg` operand is dropped).
        // `recognize_expect_site` confirms the receiver is an `Option`.
        // A resolution miss leaves the residual call — an unregistered
        // callee the rtyper census Skips.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && name == "expect"
        {
            if let Some(site) = self.recognize_carrier_combinator_site(
                first_arg_ty.as_ref(),
                &result_var,
                crate::front::result_exc::CarrierCatchKind::RaiseImplicit,
            ) {
                self.carrier_combinator_sites.push(site);
            } else if let Some(site) = self.recognize_expect_site(
                first_arg_ty.as_ref(),
                &result_var,
                args[0].as_variable(),
            ) {
                self.expect_sites.push(site);
            }
        }
        // `Result::map_err(result, closure)` is a foreign core combinator.
        // Record the concrete receiver/result instantiations while their MIR
        // types are available; the post-pass restores the ordinary Ok/Err
        // diamond before exceptiontransform-style Result lowering runs.
        let map_err_path = match &op_kind {
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                args,
                ..
            } if args.len() == 2 && name == "map_err" => callee_name_path.clone(),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if args.len() == 2 && segments.last().map(String::as_str) == Some("map_err") => {
                Some(segments.join("::"))
            }
            _ => None,
        };
        if map_err_path
            .as_deref()
            .is_some_and(is_core_result_map_err_path)
            && let Some(site) = self.recognize_result_map_err_site(
                first_arg_ty.as_ref(),
                second_arg_ty.as_ref(),
                second_arg_fn_item.as_deref(),
                &call.dest.ty,
                &result_var,
            )
        {
            self.result_map_err_sites.push(site);
        }
        // Capture `Option::map`/`and_then`/`unwrap_or_else(opt, callable)` sites
        // for the discriminant closure-select `front::option_closure_select`
        // synthesizes.  All five combinators are Opaque (foreign `core`) with
        // the `Option` ADT receiver, so `first_is_self` routes them to a
        // two-arg `CallTarget::Method` (receiver `args[0]`, callable
        // `args[1]`).  The callable is a closure ADT (`second_arg_ty` →
        // `call_once`) or a function item (`second_arg_fn_item` → a direct
        // `Call(FunctionPath)`).  Resolving the `Option` field owners plus
        // the callable needs the receiver type, the env type or `FnDef`
        // path, and the result type (`call.dest.ty`), all in hand here;
        // `recognize_closure_select_site` also confirms the receiver is an
        // `Option`.  A resolution miss leaves the residual call.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && args.len() == 2
            && let Some(kind) = match name.as_str() {
                "map" => Some(crate::front::option_closure_select::ClosureCombinator::Map),
                "and_then" => Some(crate::front::option_closure_select::ClosureCombinator::AndThen),
                "unwrap_or_else" => {
                    Some(crate::front::option_closure_select::ClosureCombinator::UnwrapOrElse)
                }
                "or_else" => Some(crate::front::option_closure_select::ClosureCombinator::OrElse),
                "is_some_and" => {
                    Some(crate::front::option_closure_select::ClosureCombinator::IsSomeAnd)
                }
                _ => None,
            }
            && let Some(site) = self.recognize_closure_select_site(
                kind,
                first_arg_ty.as_ref(),
                second_arg_ty.as_ref(),
                second_arg_fn_item.clone(),
                second_arg_fn_decl,
                &call.dest.ty,
                &result_var,
            )
        {
            self.closure_select_sites.push(site);
        }
        // Discriminant-switch combinators whose opaque core body is an
        // if/else on the receiver tag (plus `call_once` of a closure
        // argument when there is one).  Option `map`/`and_then`/
        // `unwrap_or_else`/`or_else`/`is_some_and` already have
        // their own capture above; this records the remaining Option/Result
        // methods whose semantics are the same shape.
        if let OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } = &op_kind
            && let Some(kind) = DiscCombinator::from_method(
                name,
                args.len(),
                callee_name_path.as_deref(),
                first_arg_ty.as_ref(),
                self.llbc,
            )
            && let Some(site) = self.recognize_disc_combinator_site(
                kind,
                args,
                first_arg_ty.as_ref(),
                second_arg_ty.as_ref(),
                second_arg_fn_decl,
                &call.dest.ty,
                &result_var,
            )
        {
            self.disc_combinator_sites.push(site);
        }
        let op_kind =
            self.rewrite_equal_layout_result_branch(op_kind, first_arg_ty.as_ref(), &call.dest.ty);
        let mut op_kind =
            self.stamp_result_branch_payloads(op_kind, first_arg_ty.as_ref(), &call.dest.ty);
        if from_bytes_identity && !resolved_call_args.is_empty() {
            // `rstr.STR` stores the UTF-8/WTF-8 bytes in the string itself
            // (`rutf8` reads that same buffer), so a Wtf8 view of those
            // bytes is the string. Bind dest to the argument with no op
            // after Result-scope / combinator capture so a following `?`
            // still sees an immediate `Result::branch` consumer.
            // A W3 window copies (`rffi.py` `charpsize2str`).
            let src = resolved_call_args[0].clone();
            if first_arg_is_string_byte_view || self.var_is_str_repr(&src) {
                self.alias_dest_to_arg0(dest_local, src, true);
                stamp_as_rstr_str(&resolved_call_args[0]);
                let target_bb = self.block_id[target];
                let link_args = self.edge_args(mir_bb, target)?;
                self.graph.set_goto(bb_id, target_bb, link_args);
                return Ok(());
            }
            let window_len = from_bytes_window_len
                .clone()
                .or_else(|| self.window_len_companion(&src));
            if from_bytes_one_word && window_len.is_none() {
                self.alias_dest_to_arg0(dest_local, src, true);
                stamp_as_rstr_str(&resolved_call_args[0]);
                let target_bb = self.block_id[target];
                let link_args = self.edge_args(mir_bb, target)?;
                self.graph.set_goto(bb_id, target_bb, link_args);
                return Ok(());
            }
            if let Some(len) = window_len {
                let header = self.emit_charpsize2str(mir_bb, src, len);
                self.bind_str_view_local(dest_local, header);
                let target_bb = self.block_id[target];
                let link_args = self.edge_args(mir_bb, target)?;
                self.graph.set_goto(bb_id, target_bb, link_args);
                return Ok(());
            }
        }
        // `lower_std_primitive_op` rewrites an identity wrapper
        // (`Box::as_ref` / `as_mut`, `Cell::get`, scalar `clone`) to
        // `same_as` only when the banks agree. Bind the destination
        // local to that operand. A separate `same_as` variable hides
        // the producer from `retarget_vec_operand`.
        if let OpKind::UnaryOp { op, operand, .. } = &op_kind
            && op == "same_as"
        {
            self.local_var[dest_local] = Some(LocalValue::One(operand.clone()));
            let target_bb = self.block_id[target];
            let link_args = self.edge_args(mir_bb, target)?;
            self.graph.set_goto(bb_id, target_bb, link_args);
            return Ok(());
        }
        // Copy the borrowed GC reference into the cell before the call.
        // The copy-out below rebinds the local before the successor edge.
        let gc_mut_ref_copies = self.install_gc_mut_ref_call(
            mir_bb,
            &mut op_kind,
            &gc_mut_ref_sig,
            &arg_locals,
            &arg_operands,
            &resolved_call_args,
        )?;
        let raw_scalar_reloads = self.install_raw_scalar_address_call(
            mir_bb,
            &mut op_kind,
            &raw_scalar_spills,
            &arg_locals,
            &arg_operands,
            &resolved_call_args,
            &call.dest.ty,
        )?;
        self.local_var[dest_local] = Some(match slice_next {
            Some(LocalValue::SliceIter { items, .. }) => LocalValue::SliceIter {
                items,
                index: result_var.clone(),
            },
            _ => LocalValue::One(result_var.clone()),
        });
        // A fat `&dyn` / `Box<dyn>` result is two words. Record it so
        // `front::dyn_fat` can splice the callee before the vtable word
        // is dropped on the one-word return.
        if matches!(&op_kind, OpKind::Call { .. }) && tyref_is_fat_dyn(&call.dest.ty, self.llbc) {
            self.dyn_return_calls.push(result_var.clone());
        }
        // A callee whose output is a window uses the two-word return ABI
        // wherever the call is lowered (`ret_len_slot_local` /
        // `SLICE_LEN_SLOT_NEW` / `TAKE`), including a dest that is a STR
        // view, a list, or unused. Identity wrappers returned above, so a
        // W1 source whose output type is also `&[u8]` does not leave a
        // dead slot. A STR-view range index stays two arguments so
        // `front::slice_index` can rewrite it to `__getslice_*`
        // (`rewire_one_slice_index_site` requires `args.len() == 2`).
        let range_index_consumer = matches!(
            &op_kind,
            OpKind::Call { args, .. }
                if args.len() == 2
                    && (self
                        .slice_index_rangefrom_sites
                        .iter()
                        .any(|site| site.range_result == args[1])
                        || self
                            .slice_index_rangeto_sites
                            .iter()
                            .any(|site| site.range_result == args[1])
                        || self
                            .slice_index_range_sites
                            .iter()
                            .any(|site| site.range_result == args[1]))
        );
        if range_index_consumer
            && !first_arg_is_string_byte_view
            && resolved_call_args.len() >= 2
            && !self.var_is_str_repr(&resolved_call_args[0])
        {
            let ptr = resolved_call_args[0].clone();
            if let Some(len) = window_recv_len
                .clone()
                .or_else(|| self.window_len_companion(&ptr))
            {
                let arg_is_range_full = second_arg_ty
                    .as_ref()
                    .is_some_and(|ty| tyref_to_ast_string(ty, self.llbc) == "RangeFull");
                let owner = self.owner_for_window_source(&ptr);
                let (ptr, len) = self.window_subslice(
                    mir_bb,
                    ptr,
                    len,
                    &resolved_call_args[1],
                    arg_is_range_full,
                    true,
                )?;
                let ptr = if tyref_is_u8_window(&call.dest.ty, self.llbc) {
                    self.ensure_window_ccharp(mir_bb, ptr)
                } else {
                    ptr
                };
                if self.len_shadow.get(dest_local).copied().flatten().is_some()
                    && tyref_is_u8_window(&call.dest.ty, self.llbc)
                {
                    self.bind_pair_local(mir_bb, dest_local, ptr, len, owner);
                } else {
                    self.local_var[dest_local] = Some(LocalValue::One(ptr));
                }
                let target_bb = self.block_id[target];
                let link_args = self.edge_args(mir_bb, target)?;
                self.graph.set_goto(bb_id, target_bb, link_args);
                return Ok(());
            }
        }
        let window_ret_slot =
            if callee_output_is_window && !first_arg_is_string_byte_view && !range_index_consumer {
                if let OpKind::Call { args, .. } = &mut op_kind {
                    let slot = self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::SLICE_LEN_SLOT_NEW,
                        vec![],
                        ValueType::Unsigned,
                    );
                    args.push(crate::model::LinkArg::from(slot.clone()));
                    Some(slot)
                } else {
                    None
                }
            } else {
                None
            };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result_var.clone()),
            kind: op_kind,
        });
        for owner in converted_window_owners {
            self.emit_keepalive(mir_bb, owner);
        }
        if let Some(slot) = window_ret_slot {
            let len = self.emit_path_call(
                mir_bb,
                majit_ir::rvec::SLICE_LEN_SLOT_TAKE,
                vec![slot],
                ValueType::Unsigned,
            );
            let ptr = self.ensure_window_ccharp(mir_bb, result_var.clone());
            self.string_byte_view_locals
                .retain(|&local| local != dest_local);
            if self.len_shadow.get(dest_local).copied().flatten().is_some() {
                self.bind_pair_local(mir_bb, dest_local, ptr, len, None);
            } else {
                self.local_var[dest_local] = Some(LocalValue::One(ptr));
            }
        }
        if self.len_shadow.get(dest_local).copied().flatten().is_some()
            && tyref_is_u8_window(&call.dest.ty, self.llbc)
            && let Some(var) = self
                .local_var
                .get(dest_local)
                .and_then(|slot| slot.as_ref())
                .and_then(|value| value.one().ok())
        {
            self.convert_one_word_owner_to_window_dest(
                mir_bb,
                dest_local,
                var,
                first_arg_ty.as_ref(),
            )?;
        }
        self.copy_out_gc_mut_ref(mir_bb, &gc_mut_ref_copies)?;
        // Reload before `edge_args` so the successor sees the word the
        // callee wrote through the raw out-parameter.
        self.reload_raw_scalar_addresses(mir_bb, &raw_scalar_reloads)?;
        // Narrow a call result whose declared pointee differs from the
        // target class (`recast_result_root`). Identity at jitcode
        // (`__cast_instance_intrinsic` → cast_pointer → `same_as`) when
        // the roots already match, so the Call dest is the value.
        if let Some(root) = recast_result_root {
            let narrowed = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(narrowed.clone()),
                kind: crate::model::cast_instance_call(root, result_var.clone()),
            });
            self.local_var[dest_local] = Some(LocalValue::One(narrowed));
        }
        // `from_raw_parts{,_mut}` of a pointer slice returns the data
        // address. `getkind` of that raw word is `int`, while every
        // `&[PyObjectRef]` / `&mut [PyObjectRef]` parameter stays `ref`.
        // Put the address back on the ref bank before the slice is passed.
        if callee_name_path.as_deref().is_some_and(|path| {
            path == "core::slice::raw::from_raw_parts"
                || path == "core::slice::raw::from_raw_parts_mut"
        }) && borrow_pointee_is_ptr_slice(&call.dest.ty, self.llbc, self.gc_struct_ids)
        {
            let source = match &self.local_var[dest_local] {
                Some(LocalValue::One(v)) => v.clone(),
                _ => result_var.clone(),
            };
            let casted = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(casted.clone()),
                kind: cast_int_to_ptr_call(source),
            });
            self.local_var[dest_local] = Some(LocalValue::One(casted));
        }

        // Close the block: forward to the success target. The call's
        // `on_unwind` successor is a Rust panic-cleanup path (destructor
        // drop-glue terminating in UnwindResume / Abort) with no Python
        // meaning — a Python exception raised by the callee rides the
        // SUCCESS edge as a `Result::Err` value, matched downstream as
        // ordinary control flow, not this unwind edge. The residual-call
        // `guard_no_exception` is re-derived op-locally from the callee
        // graph (`codewriter/call.rs` `_canraise`), so dropping the
        // front-graph unwind edge keeps the can-raise signal. A real
        // try/except handler would need a `LastException` edge here; the
        // interpreter expresses exceptions as `Result`, so none arises.
        if let Some(reroot) = self.owner_root_reroot.take() {
            // The producer value is bound to the destination; `new` reads it
            // there and rebinds the destination to the handle.
            return self.lower_call(mir_bb, reroot, target, on_unwind);
        }
        let _ = on_unwind;
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// Bind `dest_local` to a fresh read of the atomic slot recorded on
    /// the `&self` local, then close the block. The read happens at this
    /// call site (`resolve_place`), not at the borrow that built `&self`.
    fn bind_dest_to_recorded_atomic_load(
        &mut self,
        mir_bb: usize,
        dest_local: usize,
        target: usize,
        arg_locals: &[Option<usize>],
    ) -> Result<bool, LowerError> {
        let Some(referent) = arg_locals
            .first()
            .copied()
            .flatten()
            .and_then(|local| self.atomic_ref_place.get(&local))
            .map(clone_place)
        else {
            return Ok(false);
        };
        let value = self.resolve_place(mir_bb, referent)?;
        self.local_var[dest_local] = Some(LocalValue::One(value));
        let bb_id = self.emit_bb(mir_bb);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// `Cell<T>` is `#[repr(transparent)]` over `T`, so `get` of a field place
    /// is that field's value read. `core::cell::Cell` only. `RefCell` /
    /// `OnceCell` / `UnsafeCell` / `LazyCell` do not match: `OnceCell::get`
    /// returns `Option<&T>`.
    fn is_cell_get(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let leaf_is_get = |leaf: &str| leaf == "get";
        let owner_is_cell = |owner: &str| owner == "Cell" || owner.ends_with("::Cell");
        let path = fd.item_meta.name_path();
        if let Some((owner, leaf)) = path.rsplit_once("::")
            && leaf_is_get(leaf)
            && owner_is_cell(owner)
            && path.starts_with("core::cell::")
        {
            return true;
        }
        impl_method_owner_for_fundecl(self.llbc, fd).is_some_and(|(owner, leaf)| {
            leaf_is_get(&leaf) && owner_is_cell(&owner) && path.starts_with("core::cell::")
        })
    }

    pub(in crate::front::mir) fn set_spec(
        &mut self,
        spec: &'a std::cell::RefCell<crate::front::clause_spec::SpecQueue>,
    ) {
        self.spec = Some(spec);
    }

    pub(in crate::front::mir) fn set_spec_body(&mut self) {
        self.spec_body = true;
    }

    /// Specialized path for a direct call whose type args are concrete,
    /// or whose `generics.trait_refs` name impls. The bare path is
    /// unchanged when the callee is not generic or the call still
    /// carries a `Clause` / `TypeVar`.
    pub(in crate::front::mir) fn specialized_fun_segments(
        &self,
        fd: &FunDecl,
        reg: &RegularCall,
    ) -> Option<Vec<String>> {
        self.enqueue_spec(fd, &reg.generics)
    }

    /// Residual call path for a generic `dont_look_inside` helper.
    ///
    /// `enqueue_spec` refuses that body, so the unspecialized template
    /// path would bind every instantiation to one funcptr.
    /// `annlowlevel.specialize` / `getfunctionptr(graph)` give each
    /// specialized graph its own address; name the residual the same way
    /// from the call site's type arguments, without building a graph.
    pub(in crate::front::mir) fn residual_generic_fun_segments(
        &self,
        fd: &FunDecl,
        generics: &serde_json::Value,
    ) -> Option<Vec<String>> {
        if !crate::front::clause_spec::decl_is_generic(fd) {
            return None;
        }
        // A HOST_ENV ExtRegistry object is one builtin. Naming each
        // instantiation `leaf__inst_<types>` crate-strips to a prefix
        // `HOST_ENV.import_module` does not carry (`["jit", leaf]`), so
        // Layer-3b misses and Layer-1 / leaf-match walk the unique
        // template. Keep the unspecialized path (`enqueue_spec` already
        // refuses a host builtin).
        if self.callee_is_host_builtin(fd) {
            return None;
        }
        let path = fd.item_meta.name_path();
        if !self.dont_look_inside.contains(&strip_crate_prefix(&path)) {
            return None;
        }
        // Impl methods go through `enqueue_spec` (`cachedgraph`). A free
        // function is the residual whose address `jit_fnaddr` publishes
        // per instantiation.
        if impl_method_owner_for_fundecl(self.llbc, fd).is_some() {
            return None;
        }
        let names = crate::front::clause_spec::residual_inst_type_names(generics, self.llbc)?;
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let leaf = path.rsplit("::").next().unwrap_or("fn");
        let leaf = format!(
            "{leaf}{}",
            majit_ir::helper_fnaddr::residual_inst_suffix(&name_refs)
        );
        Some(spec_segments(self.llbc, fd, &leaf))
    }

    /// Impl method named by a `TraitImpl` trait ref, specialized when that
    /// impl's own clauses are concrete. A `Clause` ref returns `None` so
    /// the call keeps the trait-declaration path, unless the receiver
    /// type is a known ADT ([`trait_method_on_known_receiver`]).
    ///
    /// `description.py` `ClassDesc.lookup` / `MethodDesc`: rustc has
    /// already selected the impl, so the call is that impl's graph.
    /// Specialization (`FunctionDesc.cachedgraph`) is only queued from
    /// a spec body; a non-spec caller still names the impl method.
    pub(in crate::front::mir) fn specialized_trait_target(
        &self,
        payload: &serde_json::Value,
    ) -> Option<(Vec<String>, Option<(String, String, Option<String>)>)> {
        let (fn_id, generics) = crate::front::clause_spec::trait_impl_method(payload, self.llbc)?;
        let fd = self.llbc.fn_by_id(fn_id)?;
        // No unstructured body: keep the ordinary `CallKind::Trait` route.
        // A spec path for this method would name a graph nobody registers.
        // A zero-arg opaque impl method is the exception: it has no body
        // to copy, and `collect_foreign_opaque_method_externals` registers
        // the impl path when `foreign_opaque_method_result_valuetype`
        // accepts the result (`Default::default` on an opaque `Self`).
        if !crate::front::clause_spec::decl_has_unstructured_body(fd) {
            return self.opaque_nullary_impl_target(fd);
        }
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next().unwrap_or("fn");
        if self.spec_body {
            if let Some(segments) = self.enqueue_spec(fd, &generics) {
                return Some((segments, None));
            }
        }
        Some((spec_segments(self.llbc, fd, leaf), None))
    }

    /// Impl path of a zero-arg method whose declaration has no body.
    ///
    /// `enqueue_spec` refuses an opaque body. The call still has to name
    /// the path `collect_foreign_opaque_method_externals` registers, or a
    /// specialized caller keeps the unregistered trait path. A method
    /// with a receiver is left on that trait path: retargeting it would
    /// resolve graphs the prepass does not annotate.
    pub(in crate::front::mir) fn opaque_nullary_impl_target(
        &self,
        fd: &FunDecl,
    ) -> Option<(Vec<String>, Option<(String, String, Option<String>)>)> {
        if !fd.signature.inputs.is_empty() {
            return None;
        }
        let owner_def_id = impl_owner_adt_def_id_for_fundecl(self.llbc, fd)?;
        let owner_td = self.llbc.type_by_id(owner_def_id)?;
        if !matches!(owner_td.kind, TypeDeclKind::Opaque) && !owner_td.published_struct_copy {
            return None;
        }
        foreign_opaque_method_result_valuetype(&fd.signature.output, self.llbc)?;
        let path = fd.item_meta.name_path();
        let leaf = path.rsplit("::").next().unwrap_or("fn");
        Some((spec_segments(self.llbc, fd, leaf), None))
    }

    pub(in crate::front::mir) fn enqueue_spec(
        &self,
        fd: &FunDecl,
        generics: &serde_json::Value,
    ) -> Option<Vec<String>> {
        let spec = self.spec?;
        if !crate::front::clause_spec::decl_is_generic(fd) {
            return None;
        }
        // Only a body this LLBC extracted can be copied. An opaque
        // declaration (a foreign crate's or std's) keeps its bare path.
        if !crate::front::clause_spec::decl_has_unstructured_body(fd) {
            return None;
        }
        // A builtin is not a `FunctionDesc` (`BUILTIN_ANALYZERS`); it gets no specialized graph.
        if self.callee_is_host_builtin(fd) {
            return None;
        }
        // A free-function `dont_look_inside` helper is a residual call
        // (`residual_generic_fun_segments` / the monomorph's funcptr).
        // An impl method has no HELPER_FNADDRS row (`HelperFnAddrSkip::Generic`
        // and the impl-path split), so `FunctionDesc.cachedgraph` is its
        // specialized graph.
        let path = fd.item_meta.name_path();
        if self.dont_look_inside.contains(&strip_crate_prefix(&path))
            && impl_method_owner_for_fundecl(self.llbc, fd).is_none()
        {
            return None;
        }
        // `FunctionDesc.cachedgraph`: each concrete instantiation is
        // its own graph, including a look-inside generic whose body
        // names no `Clause`. The inner residual then sees the type
        // arguments. A still-open `Clause` ref refuses (`None`).
        let trait_refs =
            crate::front::clause_spec::concrete_trait_refs_or_empty(generics, self.llbc)?;
        let (types, const_generics) =
            crate::front::clause_spec::concrete_type_args(generics, self.llbc)?;
        let leaf = fd
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("fn")
            .to_string();
        let leaf = crate::front::clause_spec::spec_leaf(&leaf, fd.def_id, generics, self.llbc);
        spec.borrow_mut()
            .enqueue(crate::front::clause_spec::SpecRequest {
                fn_id: fd.def_id,
                leaf: leaf.clone(),
                trait_refs,
                types,
                const_generics,
            });
        Some(spec_segments(self.llbc, fd, &leaf))
    }

    /// `HOST_ENV` resolves the bare call segments, or the crate-qualified
    /// dotted path (`name_path` with `::` rewritten to `.`).
    pub(in crate::front::mir) fn callee_is_host_builtin(&self, fd: &FunDecl) -> bool {
        let (segments, _) = self.unspecialized_fun_segments(fd);
        crate::flowspace::model::host_env_callable(&segments).is_some()
            || fun_decl_name_path_is_host_env(fd)
    }

    /// Segments and method hint `call_target_segments` emits for a direct
    /// call once specialization and blanket `Into` devirtualization decline.
    pub(in crate::front::mir) fn unspecialized_fun_segments(
        &self,
        fd: &FunDecl,
    ) -> (Vec<String>, Option<(String, String, Option<String>)>) {
        let method_hint = self.impl_method_owner(fd);
        // A `#[dont_look_inside]` inherent method is a residual call, not a
        // trace target. Routing it as `CallTarget::Method` surfaces
        // `getattr(recv, method)` on a classed receiver whose classdict
        // carries no method source, which blocks at annotate
        // (`complete_pending_blocks failed: Blocked block`). Decline the
        // Method hint for such a callee so it routes as a `FunctionPath`.
        let method_hint = if method_hint.is_some()
            && self
                .dont_look_inside
                .contains(&strip_crate_prefix(&fd.item_meta.name_path()))
        {
            None
        } else {
            method_hint
        };
        // Declined-Method impl-owned callees still name the FunDecl
        // Charon recorded. `spec_segments` is that registry key
        // (inherent `for_impl_method`, trait-impl identity path) so std
        // matchers keep `[.., "<Impl>", leaf]` only on the `name_path`
        // branch below.
        let segments: Vec<String> = if method_hint.is_none()
            && let Some((_, leaf)) = impl_method_owner_for_fundecl(self.llbc, fd)
        {
            spec_segments(self.llbc, fd, &leaf)
        } else {
            fd.item_meta
                .name_path()
                .split("::")
                .map(|s| s.to_string())
                .collect()
        };
        (segments, method_hint)
    }

    /// Resolve a Charon `CallKind` to a flattened path segment list the
    /// codewriter consumes as `CallTarget::FunctionPath`, plus an
    /// optional `(owner_root_leaf, method_leaf, trait_impl_identity)`
    /// triple for impl methods, plus the `[Owner, leaf]` registration
    /// spelling when the callee is an inherent-impl *associated function*
    /// (no receiver). `trait_impl_identity` is `render_trait_impl` of the
    /// FunDecl's trait-impl id (`None` for an inherent method).
    ///
    /// The method hint is `Some` when the FunDecl's raw name segments
    /// encode an `Impl` block immediately before the leaf `Ident` —
    /// the standard Charon shape for inherent / trait-impl methods
    /// (e.g. `pyre_interpreter::pyframe::<Impl>::locals_w_mut`).  The
    /// caller uses the hint to pick `CallTarget::Method` over
    /// `CallTarget::FunctionPath` so the annotator can prepend a
    /// classdef-bound `SomeInstance` for `self`; see the comment at
    /// the use site in [`Self::lower_call`].
    ///
    /// The associated-function spelling is computed only when the
    /// method hint is `None` (a receiver-shaped callee never consumes
    /// it), and only the `CallTarget::FunctionPath` construction in
    /// `lower_call` applies it — the raw `name_path()` segments stay
    /// untouched for the std special-case matchers (`checked_neg` /
    /// `try_from` / `into` / `expect`) that key on the
    /// `[.., "<Impl>", leaf]` shape.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub(in crate::front::mir) fn call_target_segments(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
        first_arg_ty: Option<&TyRef>,
    ) -> Result<(Vec<String>, Option<(String, String, Option<String>)>), LowerError> {
        match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => self
                .llbc
                .fn_by_id(*id)
                .map(|fd| {
                    // A Charon-monomorphized callee is its own instance; its
                    // generics were substituted before the call was emitted.
                    if let Some(leaf) = instance_leaf(self.llbc, fd) {
                        return (spec_segments(self.llbc, fd, &leaf), None);
                    }
                    if let Some(segments) = self.specialized_fun_segments(fd, reg) {
                        return (segments, None);
                    }
                    if let Some(segments) = self.residual_generic_fun_segments(fd, &reg.generics) {
                        return (segments, None);
                    }
                    // Blanket `impl<T, U: From<T>> Into<U> for T`
                    // (core::convert) — `x.into()` is `U::from(x)`.
                    // The callsite's resolved `U: From<T>` obligation
                    // names the concrete From impl, so devirtualize to
                    // that impl's `from` the way rustc's
                    // monomorphization does.  The blanket body itself
                    // is generic-trait shaped and never lifts.  (The
                    // reflexive `Identity` outcome is intercepted at
                    // `lower_call` before reaching here.)
                    if let Some(IntoDevirt::Target(segments)) = self.blanket_into_devirt(reg) {
                        return (segments, None);
                    }
                    if let Some(segments) =
                        trait_default_call_segments(self.llbc, fd, &reg.generics)
                    {
                        return (segments, None);
                    }
                    self.unspecialized_fun_segments(fd)
                })
                .ok_or_else(|| {
                    LowerError::Schema(format!(
                        "bb{mir_bb}: Call references unknown FunDecl id {id}"
                    ))
                }),
            CallKind::Fun(FunId::Other(v)) => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Fun(Other) not yet supported: {v}"
            ))),
            // `CallKind::Trait([trait_ref, method_idx, fn_decl_id])` —
            // generic-trait method call.  Charon's `arr[2]` is the
            // `def_id` of the trait method declaration itself
            // (e.g. `pyre_interpreter::shared_opcode::SharedOpcodeHandler::
            // push_value`).
            //
            // `extract_trait_impls` parses the trait declaration's
            // default-body and registers it under BOTH
            // `["<default methods of <Trait>>", <method>]` (the
            // selfclassdef-bound `register_trait_method` path) and the
            // direct path `[<Trait>, <method>]` (lib.rs —
            // `register_function_graph(direct_path, …)`).  The direct
            // path is the call-site shape Rust code emits when calling
            // `<Trait>::<method>(receiver, …)` and the BFS-driven
            // `find_all_graphs` reaches it as a regular candidate.
            //
            // To stay PyPy-orthodox for generic-trait dispatch, route
            // the call through that same `[<Trait>, <method>]` path so:
            //   1. BFS discovers the trait default body as a
            //      candidate, which transitively pulls in the helpers
            //      it calls (e.g. `opcode_load_const`).
            //   2. `flowspace_adapter` emits a `simple_call(<
            //      callable>, args…)` shape (no `getattr` surface) so
            //      the classdef-less receiver does not surface as a
            //      panicking `SomeInstance.getattr`.
            //
            // Falls back to the `["__trait_method", <label>]` synthetic
            // path when the fn_decl cannot be resolved or does not have
            // the trait-method shape (e.g. when arr[2] is missing or
            // points at an `Impl` block).
            CallKind::Trait(v) => {
                if let Some(target) = self.specialized_trait_target(v) {
                    return Ok(target);
                }
                // `Vec<T>` as `Clone::clone` is `rlist.py` `ll_copy` when
                // T's clone is identity (`ll_arraycopy` never calls
                // `T::clone`). Bind it here so the call never becomes
                // `["Clone","clone"]`.
                if let Some(target) = specialized_trait_clone_list_target(v, self.llbc) {
                    return Ok(target);
                }
                // `description.py` `ClassDesc.lookup` / `MethodDesc`: a
                // `Clause` ref still names the impl once the receiver
                // ADT is known. rustc selected that impl; there is no
                // runtime dispatch.
                if let Some((fn_id, segments, impl_generics)) =
                    trait_method_on_known_receiver(first_arg_ty, v, self.llbc)
                {
                    if self.spec_body
                        && let Some(fd) = self.llbc.fn_by_id(fn_id)
                        && fd
                            .src
                            .as_ref()
                            .and_then(|s| s.get("TraitDefault"))
                            .is_none()
                    {
                        let combined =
                            merge_impl_and_method_generics(&impl_generics, &reg.generics);
                        if let Some(spec) = self.enqueue_spec(fd, &combined) {
                            return Ok((spec, None));
                        }
                    }
                    return Ok((segments, None));
                }
                if let Some((trait_path, method_leaf)) = trait_payload_path_and_method(v, self.llbc)
                {
                    // `extract_trait_impls` registers default bodies under
                    // `[<Trait leaf>, <method>]`. The residual path keeps
                    // that two-segment shape; recognizers compare the full
                    // Charon path separately.
                    let trait_leaf = trait_path
                        .rsplit("::")
                        .next()
                        .unwrap_or(trait_path.as_str())
                        .to_string();
                    Ok((vec![trait_leaf, method_leaf], None))
                } else {
                    let label = trait_call_label(v);
                    Ok((vec!["__trait_method".to_string(), label], None))
                }
            }
            CallKind::Ptr(v) => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Ptr not yet supported: {v}"
            ))),
            CallKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Unknown"
            ))),
        }
    }

    /// Class root of an inherent `&self` method whose owner ADT is opaque
    /// in this crate. `dont_look_inside` callees stay residual and are
    /// not painted. `core`/`std`/`alloc` owners are not struct classes.
    pub(in crate::front::mir) fn opaque_traced_method_self_root(
        &self,
        reg: &RegularCall,
    ) -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        if self
            .dont_look_inside
            .contains(&strip_crate_prefix(&fd.item_meta.name_path()))
        {
            return None;
        }
        if fd.first_arg_local_name().is_some_and(|n| n != "self") {
            return None;
        }
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
        let adt_def_id = self.resolve_impl_owner_adt_def_id(impl_payload)?;
        if !self.first_input_is_adt(fd, adt_def_id) {
            return None;
        }
        let td = self.llbc.type_by_id(adt_def_id)?;
        if !matches!(td.kind, TypeDeclKind::Opaque) && !td.published_struct_copy {
            return None;
        }
        // A fieldless enum is an int. Painting `self` as an instance
        // makes `eq` compare `InstanceRepr` with the `ConstInt` tag.
        if type_decl_is_fieldless_enum(td, self.llbc) {
            return None;
        }
        let name = td.item_meta.name_path();
        let crate_root = name.split("::").next().unwrap_or(&name);
        if matches!(crate_root, "core" | "std" | "alloc") {
            return None;
        }
        tyref_input_class_root(
            fd.signature.inputs.first()?,
            self.llbc,
            self.tombstoned_leaves,
        )
    }

    /// Return `(owner_root_leaf, method_leaf, trait_impl_identity)` when
    /// the FunDecl's name path encodes an impl block (inherent or
    /// trait-impl) whose owner type is resolvable through the LLBC
    /// tables. `trait_impl_identity` is `render_trait_impl` of the
    /// FunDecl's trait-impl id; an inherent method leaves it `None`.
    ///
    /// Charon serialises an impl method's name as:
    ///   `[Ident("crate"), Ident("mod"), Other({"Impl": ...}), Ident("method_name")]`
    /// where the `Impl` segment carries either
    ///   `{"Ty": {"skip_binder": {"Deduplicated": <type_id>}, "kind": "InherentImplBlock"}}`
    /// for inherent impls or `{"Trait": <trait_impl_id>}` for trait-impls.
    /// Trait-impl lookups indirect through the top-level `trait_impls`
    /// table, kept opaque (`schema::Translated.rest["trait_impls"]`)
    /// because no other consumer needs it typed.
    pub(in crate::front::mir) fn impl_method_owner(
        &self,
        fd: &FunDecl,
    ) -> Option<(String, String, Option<String>)> {
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
        let owner_leaf = match self.resolve_impl_owner_adt_def_id(impl_payload) {
            Some(adt_def_id) => {
                // The hint is only valid when the fn actually receives
                // the impl owner as `self` (`Owner` / `&Owner` /
                // `*mut Owner` first input).  Associated functions —
                // `PyError::type_error(msg)` — have no receiver;
                // routing them as `Method` makes the adapter getattr
                // the method name off `args[0]` (the message string).
                if !self.first_input_is_adt(fd, adt_def_id) {
                    return None;
                }
                let td = self.llbc.type_by_id(adt_def_id)?;
                // An opaque owner from a dependency artefact has no extracted
                // body in that LLBC, so the annotator never
                // mints a `ClassDef` for it — the receiver lands as a
                // classdef-less `SomeInstance` and a `CallTarget::Method`
                // getattr panics ("SomeInstance.getattr on classdef-less
                // instance").  The interpreter's overflow→long arms are the
                // cold `@jit.dont_look_inside` bailouts that operate on this
                // opaque value, so the faithful treatment is to residualize
                // the call, not trace into an unavailable body —
                // the `register_external` analog.  Declining the Method hint
                // routes the call through the `FunctionPath` form, which the
                // call registry resolves to an opaque external
                // (`register_foreign_opaque_method_externals`) and the
                // codewriter emits as a residual fnaddr call.
                if matches!(td.kind, TypeDeclKind::Opaque) || td.published_struct_copy {
                    return None;
                }
                // A FunDecl with no unstructured body cannot be getattr'd
                // off the receiver: Charon left the method Opaque, so
                // Method's `getattr(recv, leaf)` + simple_call finds no
                // classdict source. Keep Method for Enum combinators
                // (`Option::unwrap_or`) whose folds key on
                // `CallTarget::Method`. A Struct owner is minted as a
                // field-only class (`Range<i64>` has start/end); route as
                // FunctionPath so `Range::contains` reaches the graph of its
                // Charon-included body (`Bookkeeper.getdesc`).
                if !fd.has_unstructured_body() && matches!(td.kind, TypeDeclKind::Struct(_)) {
                    return None;
                }
                // A raw struct or enum is an address, not a GC instance.
                // `CallTarget::Method` would getattr the leaf off that
                // address. The direct path is the fun-decl itself.
                //
                // A configured JIT-driver receiver is the exception:
                // `jit_marker_key_from_target` keys on `CallTarget::Method`
                // plus `receiver_root`, and `ExtEnterLeaveMarker.specialize_call`
                // keeps the marker as a method on that receiver.
                let mut raw_stack = Vec::new();
                if adt_def_is_raw_storage(
                    adt_def_id,
                    self.llbc,
                    self.gc_struct_ids,
                    self.immutable_fields,
                    &mut raw_stack,
                ) {
                    let owner = td
                        .item_meta
                        .name_path()
                        .rsplit("::")
                        .next()
                        .unwrap_or("")
                        .to_string();
                    if !self
                        .jitdriver_receiver_roots
                        .iter()
                        .any(|root| root == &owner)
                    {
                        return None;
                    }
                }
                // A method whose name collides with a field of the owner ADT
                // cannot route as `CallTarget::Method`: the adapter spells
                // the call as `getattr(recv, leaf)` + `simple_call`, and
                // `SomeInstance.getattr` answers the projected field row — a
                // scalar — so the call walls at "Cannot prove that the
                // object is callable" on an `Integer`/`Bool`.  Rust
                // distinguishes `self.f` from `self.f()` syntactically; the
                // flat attr namespace does not.  Route through the
                // `FunctionPath` form, which resolves the callee by path.
                if adt_declares_field(td, &leaf) {
                    return None;
                }
                let owner = td
                    .item_meta
                    .name_path()
                    .rsplit("::")
                    .next()
                    .unwrap_or("")
                    .to_string();
                // Only an actual method (first signature input is the
                // owner ADT, possibly behind `&`/`&mut`) may route as
                // `CallTarget::Method`.  An associated function with
                // arguments (`PyError::type_error(msg)`) would
                // otherwise thread its first argument as the getattr
                // receiver and the annotator resolves the method name
                // against that argument's type.  Compared by nominal ADT,
                // not name leaf, so generic owners (`Result::branch` —
                // `?`'s Try::branch) still match.
                let first_is_self = fd
                    .signature
                    .inputs
                    .first()
                    .and_then(|t| tyref_node(t, self.llbc))
                    .and_then(|n| strip_ty_wrappers(n, self.llbc))
                    .and_then(adt_node_def_id)
                    .is_some_and(|id| same_nominal_adt(self.llbc, id, adt_def_id));
                if !first_is_self {
                    return None;
                }
                // The first input being the owner ADT is necessary but not
                // sufficient: an associated constructor
                // (`W_Range::allocate(payload: Self)` /
                // `allocate_stable(payload: Self)`) also presents the owner as
                // its first input, yet routing it as `Method` threads the
                // payload as the getattr receiver and blocks on
                // `getattr(payload, "allocate")`.  Only a genuine `self`
                // receiver may route as `Method`.  Rust forbids naming a
                // parameter `self`, so a receiver's local is the `self` keyword
                // (Charon names it `"self"`, or leaves it unnamed for a
                // monomorphised trait method), whereas a constructor's
                // `Self`-typed parameter keeps its source name (`payload`).
                // Decline the Method hint for an explicitly-named non-`self`
                // first input so the call routes as `FunctionPath` and the
                // call registry resolves the constructor (an associated
                // function, called through `FunctionPath`).
                if fd.first_arg_local_name().is_some_and(|n| n != "self") {
                    return None;
                }
                // `SomePtr.getattr` (`unaryop.py`) resolves struct fields and
                // `_adtmeths` only. An inherent method is neither, so
                // `CallTarget::Method`'s `getattr` + `simple_call` panics on a
                // raw-pointer receiver. A function that takes the pointer is a
                // `simple_call` of that constant (`rpbc.py` `FunctionsPBCRepr`
                // → `direct_call`), pointer first — the `FunctionPath` route.
                if fd.signature.inputs.first().is_some_and(|ty| {
                    ty_is_raw_address(
                        ty,
                        self.llbc,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    )
                }) {
                    return None;
                }
                // `&mut self` of a GC handle occupies a MutRef cell
                // (`mut_ref_cell_root_of`). MethodDesc.func_args prepends
                // SomeInstance(selfclassdef) of the owner, so the callee's
                // getattr(self, "value") blocks on that instance. Decline
                // the Method hint so the call stays FunctionPath and
                // FunctionDesc.parse_arguments sees the cell
                // `install_gc_mut_ref_call` already constructs.
                if fd.signature.inputs.first().is_some_and(|ty| {
                    tyref_gc_mut_ref_cell_root(
                        ty,
                        self.llbc,
                        self.tombstoned_leaves,
                        self.gc_struct_ids,
                        &self.raw_alloc_flavor_structs,
                    )
                    .is_some()
                }) {
                    return None;
                }
                // The ADT id above is still the wrapper. The name the
                // codewriter falls back to is the class the value carries.
                gc_handle_method_class_root(self.llbc, fd, adt_def_id, no_tombstoned_leaves())
                    .unwrap_or(owner)
            }
            // Non-ADT `Self` (primitive / raw pointer / slice): Charon leaves
            // the impl owner type unresolved, so the ADT table has no entry.
            // Do not invent a `CallTarget::Method` owner from the module
            // Ident — `core::ptr::{const_ptr,mut_ptr}::<Impl>::is_null` is
            // lowered to `is_(p, Constant(None))` before this hint is
            // consulted (`is_raw_ptr_is_null`), and every other primitive
            // method stays on the `FunctionPath` form.
            None => return None,
        };
        if owner_leaf.is_empty() {
            return None;
        }
        // A window or pair return uses the two-word ABI
        // (`pair_or_window_fun_inputs` / `SLICE_LEN_SLOT_NEW`). `getattr`
        // on the receiver would drop the length word
        // (`jtransform.py` `handle_regular_call` keeps the target graph).
        if self.pair_slice_kind(&fd.signature.output).is_some()
            || tyref_is_u8_window(&fd.signature.output, self.llbc)
        {
            return None;
        }
        Some((
            owner_leaf,
            leaf,
            trait_impl_id_for_fundecl(fd)
                .and_then(|id| crate::front::clause_spec::render_trait_impl(self.llbc, id)),
        ))
    }
}
