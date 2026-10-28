//! Constant folding of globals and const bodies.

use super::*;

impl<'a> Lowering<'a> {
    /// Resolve a global `def_id` to its fully-qualified path segments
    /// via the reader's `global_decls` table.
    ///
    /// Uses [`static_key_segments`] rather than `item_meta.name_path()`:
    /// `name_path` drops every Ident disambiguator except `closure#N`, so
    /// two function-local statics that share a leaf (`method_owner::OWNER`)
    /// collapse onto one key and neither `HostStaticAddrs` row can name
    /// just one of them. Charon stores the disambiguator as the `u64` of
    /// `NameSeg::Ident`; the key keeps it.
    pub(in crate::front::mir) fn global_segments(
        &self,
        mir_bb: usize,
        def_id: u64,
    ) -> Result<Vec<String>, LowerError> {
        self.llbc
            .global_by_id(def_id)
            .map(|g| static_key_segments(&g.item_meta.name))
            .ok_or_else(|| {
                LowerError::Schema(format!(
                    "bb{mir_bb}: Place::Global references unknown GlobalDecl id {def_id}"
                ))
            })
    }

    /// Emit the inline `RangeInclusive::new(lo, hi)` a promoted
    /// `(a..=b)` constant becomes once its AnonConst init is folded.
    pub(in crate::front::mir) fn emit_promoted_int_range_inclusive(
        &mut self,
        mir_bb: usize,
        lo: i64,
        hi: i64,
        place_ty: &TyRef,
    ) -> Variable {
        let bb_id = self.block_id[mir_bb];
        let lo_v = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(lo_v.clone()),
            kind: OpKind::ConstInt(lo),
        });
        let hi_v = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Signed);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(hi_v.clone()),
            kind: OpKind::ConstInt(hi),
        });
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        // Owner-qualified associated-function spelling
        // `unspecialized_fun_segments` emits for `RangeInclusive::new`
        // (`ops::range::RangeInclusive::new`).
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path(["ops", "range", "RangeInclusive", "new"]),
                args: crate::model::call_args(vec![lo_v.clone(), hi_v.clone()]),
                result_ty: tyref_to_value_type_with(
                    place_ty,
                    self.llbc,
                    self.tombstoned_leaves,
                    self.gc_struct_ids,
                    &self.raw_alloc_flavor_structs,
                ),
            },
        });
        res
    }

    /// Fold a `NamedConst` global (Rust `const`) whose initializer is a
    /// single literal assignment to its `ConstInt` / `ConstBool` /
    /// `ConstFloat` value.  `None` for statics (`global_kind` ≠
    /// `NamedConst`), absent initializers, or any non-trivial init body
    /// (a computed const keeps the accessor path so it is not
    /// mis-evaluated here).
    ///
    /// A foreign `const` is recorded `Opaque` in the caller's LLBC, so
    /// this also resolves the same path from the defining crate's harvest
    /// (seeded while that crate's artefact is loaded).  A `libc` errno
    /// constant is not harvested: its initializer is opaque and the host
    /// `libc` value is not the target's (`EAGAIN` is 35 on macOS and 11
    /// on Linux and wasm). Leave that read residual.
    pub(in crate::front::mir) fn fold_named_const_global(&self, def_id: u64) -> Option<OpKind> {
        fold_named_const_on_llbc(self.llbc, def_id).or_else(|| {
            let gd = self.llbc.global_by_id(def_id)?;
            if gd
                .rest
                .get("global_kind")
                .and_then(serde_json::Value::as_str)
                != Some("NamedConst")
            {
                return None;
            }
            named_const_fold_for_path(&gd.item_meta.name_path())
        })
    }

    /// Emit a folded named constant as a value. A `char` constant goes
    /// through [`Self::emit_decoded_const`] (`bookkeeper.py`
    /// `immutablevalue` `SomeUnicodeCodePoint`), the same two ops a
    /// char literal uses.
    pub(in crate::front::mir) fn emit_named_const_value(
        &mut self,
        mir_bb: usize,
        def_id: u64,
        place_ty: &TyRef,
    ) -> Option<Variable> {
        let decoded = decoded_named_const_on_llbc(self.llbc, def_id).or_else(|| {
            let inner = named_const_promoted_in_anon(self.llbc, def_id)?;
            decoded_named_const_on_llbc(self.llbc, inner)
        });
        let const_op = if decoded.is_none() {
            self.fold_named_const_global(def_id).or_else(|| {
                let inner = named_const_promoted_in_anon(self.llbc, def_id)?;
                self.fold_named_const_global(inner)
                    .or_else(|| self.const_eval_global(inner))
            })
        } else {
            None
        };
        if decoded.is_none() && const_op.is_none() {
            return None;
        }
        let bb_id = self.emit_bb(mir_bb);
        if let Some(decoded) = decoded {
            return Some(self.emit_decoded_const(bb_id, decoded));
        }
        let const_op = const_op?;
        if tyref_is_char(place_ty, self.llbc)
            && let OpKind::ConstInt(code) = const_op
            && let Ok(code) = u32::try_from(code)
        {
            return Some(self.emit_decoded_const(bb_id, DecodedConst::UniChar(code)));
        }
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: const_op,
        });
        Some(res)
    }

    /// Fold a `NamedConst` global whose initializer builds a fixed-size
    /// array of integer literals (`OP_STACKDEL: [i32; N] = [c0, c1, …]`),
    /// or borrows that literal (`HEX: &[u8; N] = b"..."`),
    /// into the synthetic prebuilt-constant-array define-op the adapter
    /// re-folds to `Constant(list)`
    /// (`flowspace_adapter.rs::is_const_int_array_define`).  A
    /// module-level constant array is a prebuilt `Constant(ll_array)`
    /// whose `arr[i]` runtime-index read is a foldable `getarrayitem`
    /// (the `ArrayRead` the `Index` projection arm emits); baking the
    /// literals here lets that read resolve against a constant base
    /// instead of a residual accessor `Call` no registry can bind.
    ///
    /// Accepts only the exact `_0 = [c0, c1, …]; return` shape, or the
    /// borrow spelling Charon emits for a byte string:
    ///
    /// ```text
    /// _array = [c0, c1, …]
    /// _borrow = &_array
    /// _0 = move _borrow
    /// return
    /// ```
    ///
    /// The array operands must all be integer literals and control flow may
    /// contain only linear `Goto`s to a `Return`.  Anything richer (computed
    /// elements, non-integer items, a `Repeat` fill, an extra assignment)
    /// returns `None` and keeps the residual accessor path; so does a foreign
    /// const whose init body Charon left opaque (no readable body to bake).
    pub(in crate::front::mir) fn fold_named_const_int_array_global(
        &self,
        def_id: u64,
    ) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
        let body = self.llbc.fn_by_id(init_id)?.unstructured()?;
        let mut array: Option<(u64, Vec<i64>)> = None;
        let mut borrow: Option<(u64, u64)> = None;
        let mut return_alias: Option<u64> = None;
        let forward_blocks = forward_reachable_mask(self.llbc, &body);
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                match stmt.stmt_kind_ref() {
                    Ok(StmtKind::StorageLive(_))
                    | Ok(StmtKind::StorageDead(_))
                    | Ok(StmtKind::PlaceMention(_))
                    | Ok(StmtKind::Borrowck(_)) => {}
                    Ok(StmtKind::Assign(place, Rvalue::Aggregate(kind, operands))) => {
                        // Only a fixed-size `Array` aggregate of integer
                        // literals; a `Repeat` fill or an Adt aggregate is
                        // out of scope, as is a second array definition.
                        let PlaceKind::Local(dst) = place.kind else {
                            return None;
                        };
                        if aggregate_ctor_name(kind) != "Array" || array.is_some() {
                            return None;
                        }
                        let mut vals = Vec::with_capacity(operands.len());
                        for op in operands {
                            let Operand::Const(value) = op else {
                                return None;
                            };
                            let n = match decode_constant(self.llbc, value).ok()? {
                                DecodedConst::Int(n) => n,
                                // Current Charon preserves a byte-string
                                // aggregate's `u8` bank. The RPython
                                // prebuilt ll_array constant uses the same
                                // signed integer literal carrier after the
                                // checked, value-preserving conversion.
                                DecodedConst::UInt(n) => i64::try_from(n).ok()?,
                                _ => return None,
                            };
                            vals.push(n);
                        }
                        array = Some((dst, vals));
                    }
                    Ok(StmtKind::Assign(place, Rvalue::Ref { place: source, .. })) => {
                        let (PlaceKind::Local(dst), PlaceKind::Local(source)) =
                            (&place.kind, &source.kind)
                        else {
                            return None;
                        };
                        if borrow.replace((*dst, *source)).is_some() {
                            return None;
                        }
                    }
                    Ok(StmtKind::Assign(place, Rvalue::Use(operand, _)))
                        if matches!(place.kind, PlaceKind::Local(0)) =>
                    {
                        let source = match operand {
                            Operand::Copy(place) | Operand::Move(place) => match place.kind {
                                PlaceKind::Local(source) => source,
                                _ => return None,
                            },
                            Operand::Const(_) => return None,
                        };
                        if return_alias.replace(source).is_some() {
                            return None;
                        }
                    }
                    // Any other statement means the array is computed, or
                    // the borrow escapes through a shape we cannot prove.
                    _ => return None,
                }
            }
            match block.term_ref(self.llbc) {
                Ok(TermKind::Return) | Ok(TermKind::Goto { .. }) => {}
                _ => return None,
            }
        }
        let (array_local, items) = array?;
        match (borrow, return_alias) {
            // Direct fixed-size array constant: `_0 = [..]`.
            (None, None) if array_local == 0 => {}
            // Borrowed byte-string constant: `_borrow = &_array; _0 = _borrow`.
            (Some((borrow_local, borrowed_local)), Some(returned_local))
                if borrowed_local == array_local && returned_local == borrow_local => {}
            _ => return None,
        }
        // The annotator infers the list's element type from the baked
        // literals, so an empty array carries no element type; leave it
        // to the residual path.
        if items.is_empty() {
            return None;
        }
        let mut segments = Vec::with_capacity(items.len() + 1);
        segments.push("__const_int_array".to_string());
        for n in items {
            segments.push(n.to_string());
        }
        Some(OpKind::Call {
            target: CallTarget::function_path(segments),
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        })
    }

    /// Fold a `NamedConst` global whose initializer is a fixed-size array
    /// of string literals (`SRE_FLAG_NAMES: [&str; 9] = ["re.TEMPLATE", …]`)
    /// into `__const_str_array`. `interp_sre.py FLAG_NAMES` is a prebuilt
    /// list; `repr_w` enumerates it. The residual accessor call has no
    /// registry entry (`SRE_FLAG_NAMES` is a const, not a function).
    ///
    /// Same `_0 = [c0, c1, …]; return` shape as
    /// [`Self::fold_named_const_int_array_global`]. Mixes of non-string
    /// elements, a `::` in a literal (FunctionPath cannot carry it), or a
    /// richer init body stay on the residual accessor.
    pub(in crate::front::mir) fn fold_named_const_str_array_global(
        &self,
        def_id: u64,
    ) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
        let body = self.llbc.fn_by_id(init_id)?.unstructured()?;
        let mut array: Option<(u64, Vec<String>)> = None;
        for block in &body.body {
            for stmt in &block.statements {
                match stmt.stmt_kind_ref() {
                    Ok(StmtKind::StorageLive(_))
                    | Ok(StmtKind::StorageDead(_))
                    | Ok(StmtKind::PlaceMention(_))
                    | Ok(StmtKind::Borrowck(_)) => {}
                    Ok(StmtKind::Assign(place, Rvalue::Aggregate(kind, operands))) => {
                        let PlaceKind::Local(dst) = place.kind else {
                            return None;
                        };
                        if aggregate_ctor_name(kind) != "Array" || array.is_some() {
                            return None;
                        }
                        let mut vals = Vec::with_capacity(operands.len());
                        for op in operands {
                            let Operand::Const(value) = op else {
                                return None;
                            };
                            let DecodedConst::Str(s) = decode_constant(self.llbc, value).ok()?
                            else {
                                return None;
                            };
                            if s.contains("::") {
                                return None;
                            }
                            vals.push(s);
                        }
                        array = Some((dst, vals));
                    }
                    _ => return None,
                }
            }
            match block.term_ref(self.llbc) {
                Ok(TermKind::Return) | Ok(TermKind::Goto { .. }) => {}
                _ => return None,
            }
        }
        let (array_local, items) = array?;
        if array_local != 0 || items.is_empty() {
            return None;
        }
        let mut segments = Vec::with_capacity(items.len() + 1);
        segments.push("__const_str_array".to_string());
        segments.extend(items);
        Some(OpKind::Call {
            target: CallTarget::function_path(segments),
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        })
    }

    /// A `NamedConst` whose initializer is one call to a const fn that
    /// builds a fixed integer array (`PTWOTABLE = make_ptwotable()`).
    /// `rbigint.py` `ptwotable` is a prebuilt list; the same
    /// `__const_int_array` define-op [`Self::fold_named_const_int_array_global`]
    /// emits for a literal aggregate is what an index read folds against.
    /// A const fn body Charon left as a loop is not that literal shape, so
    /// evaluate it. Anything the evaluator cannot prove stays a residual
    /// accessor.
    pub(in crate::front::mir) fn fold_const_fn_int_array_global(
        &self,
        def_id: u64,
    ) -> Option<OpKind> {
        const_fn_int_array_op(self.llbc, def_id)
    }

    pub(in crate::front::mir) fn static_addr_op(&self, segments: &[String]) -> Option<OpKind> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        for (key, addr) in self.static_addrs.refs {
            if static_key_matches(&full, &stripped, key) {
                return Some(OpKind::ConstRefAddr(*addr));
            }
        }
        None
    }

    /// Whether `def_id` is an immutable prebuilt instance of a configured
    /// JIT-driver receiver type.
    ///
    /// `GraphTransformConfig::jitdriver_receiver_roots` is the front-to-
    /// codewriter registration of the types whose marker methods are handled
    /// by `jit_marker_key_from_target`.  Match that set against the global's
    /// declared ADT path, not against any particular global binding, so every
    /// driver instance gets the same constant treatment.
    pub(in crate::front::mir) fn is_jitdriver_named_const(&self, def_id: u64, ty: &TyRef) -> bool {
        let Some(global) = self.llbc.global_by_id(def_id) else {
            return false;
        };
        if global
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return false;
        }
        let Some(type_path) = tyref_node(ty, self.llbc)
            .and_then(|node| strip_ty_wrappers(node, self.llbc))
            .and_then(adt_node_def_id)
            .and_then(|id| self.llbc.type_by_id(id))
            .map(|decl| decl.item_meta.name_path())
        else {
            return false;
        };
        let stripped = strip_crate_prefix(&type_path);
        self.jitdriver_receiver_roots
            .iter()
            .any(|root| static_key_matches(&type_path, &stripped, root))
    }

    /// Value of an immutable size `const` baked at build time
    /// (`HostStaticAddrs.int_values`).  The initializer is a
    /// `size_of::<T>()` the front-end cannot evaluate from the LLBC
    /// (Charon leaves the target-dependent layout symbolic), so the
    /// driver captures its compile-time value and the read folds to the
    /// same `ConstInt` an inline integer literal produces — rather than a
    /// 0-arg accessor call no registry resolves.  The value is identical
    /// at the call site (the JIT is native: host target == runtime
    /// target).
    pub(in crate::front::mir) fn static_int_value_op(&self, segments: &[String]) -> Option<OpKind> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        for (key, value) in self.static_addrs.int_values {
            if static_key_matches(&full, &stripped, key) {
                return Some(OpKind::ConstInt(*value));
            }
        }
        None
    }

    /// Address of a `pytypes`-bucket host static — a class singleton
    /// (`&SLICE_TYPE`, `&INT_TYPE`, …).  The `Global` reader lowers these
    /// to a `__cast_instance_intrinsic[<root>]` narrow of the raw address
    /// (a typed instance pointer) rather than the bare `ConstInt` the
    /// `refs` siblings avoid: a class static is the same kind of value
    /// as the `(*obj).ob_type` field-read it is compared against, so it
    /// must type `SomeInstance(<root>)` for `rtype_is_` (pointer
    /// identity) to lower the `ob_type == &TYPE` chain
    /// (`is_slice` / `is_cell` / `is_range`).
    ///
    /// The bucket carries only the address; the root comes from the
    /// static's own declared type at the read site ([`tyref_class_root`]),
    /// so a host crate whose class root is not pyre's `PyType` — the
    /// `charon-corpus` `CelClass` — narrows to its own root rather than
    /// being stamped with pyre's.
    pub(in crate::front::mir) fn pytype_static_addr(&self, segments: &[String]) -> Option<i64> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        self.static_addrs
            .pytypes
            .iter()
            .find(|(key, _)| static_key_matches(&full, &stripped, key))
            .map(|(_, addr)| *addr)
    }

    /// The class singleton a `PyreClassPyTypeOf::PYTYPE` read names,
    /// resolved through the impl's IDENTITY rather than through its
    /// rendered path.
    ///
    /// `#[pyre_class]` gives one `PyType` address two names — the static,
    /// and this associated const — and Charon renders the const's path
    /// `<module>::<Impl>::PYTYPE`, dropping the impl id
    /// ([`ItemMeta::trait_impl_id`]). That rendering is not injective, so
    /// [`Self::pytype_static_addr`]'s key matching cannot serve this
    /// family: a key that named one class here would name every class in
    /// its module.
    ///
    /// So this never consults a path. It reads the global's own impl id,
    /// takes that impl's `Self` type, and looks the owning ADT up in the
    /// struct-keyed bucket — the translation of
    /// `Bookkeeper.getdesc(Constant(pyobj))`, which resolves a prebuilt by
    /// the object rather than by any spelling of it, and of
    /// `rclass.InstanceRepr.convert_const_exact`, which caches one
    /// prebuilt structure per object in an `identity_dict`. It is the
    /// contract [`crate::translator::rtyper::cutover`] already imitates
    /// for functions, applied to prebuilt data.
    ///
    /// `None` for any global that is not a trait-impl associated const,
    /// whose impl has no resolvable `Self` ADT, or whose ADT owns no
    /// class singleton — each of which leaves the ordinary path-keyed
    /// handling below to answer.
    pub(in crate::front::mir) fn pytype_addr_by_impl_identity(&self, def_id: u64) -> Option<i64> {
        let global = self.llbc.global_by_id(def_id)?;
        let item_path = global.item_meta.name_path();
        let impl_id = global.item_meta.trait_impl_id()?;
        let row = self.llbc.trait_impl_by_id(impl_id)?;
        let trait_id = row.get("impl_trait")?.get("id")?.as_u64()?;
        let trait_path = self.llbc.trait_by_id(trait_id)?.item_meta.name_path();
        // The impl identity names the OWNER, not which associated item was
        // read.  `PyreClassPyTypeOf` also defines `DESCRIPTOR`, `PYNAME`, and
        // the CPython flag constants; mapping any of those to the owner's
        // PyType address would reinterpret a `PyreClassDescriptor`/scalar as
        // `PyType`.  Select the exact trait item first, then use identity only
        // for the part it actually proves: which class owns that `PYTYPE`.
        if !is_class_pytype_assoc_const(&item_path, &trait_path) {
            return None;
        }
        // `impl_trait.generics.types[0]` is the `Self` of `impl Trait for
        // Self`. Charon puts it first whatever else the trait is generic
        // over, so this is the implementing type, not a trait parameter.
        let self_ty: TyRef = serde_json::from_value(
            row.get("impl_trait")?
                .get("generics")?
                .get("types")?
                .as_array()?
                .first()?
                .clone(),
        )
        .ok()?;
        let node = strip_ty_wrappers(tyref_node(&self_ty, self.llbc)?, self.llbc)?;
        let struct_path = self
            .llbc
            .type_by_id(adt_node_def_id(node)?)?
            .item_meta
            .name_path();
        // Compared with [`path_eq_ignoring_raw`], not `==`: this key is
        // spelled by `module_path!()` on the host side and so keeps a
        // raw-identifier `r#`, while the `name_path()` just read is
        // Charon's canonical `DefPath` ident and has dropped it.  A class
        // under `mod r#struct` would otherwise never match.
        let stripped = strip_crate_prefix(&struct_path);
        self.static_addrs
            .pytypes_by_struct
            .iter()
            .find(|(key, _)| {
                path_eq_ignoring_raw(key, &struct_path) || path_eq_ignoring_raw(key, &stripped)
            })
            .map(|(_, addr)| *addr)
    }

    /// Evaluate a global's initializer to its literal when the body is
    /// the trivial `_0 = <literal>; return` shape.  The read then
    /// lowers to the same `Const*` op an inline literal produces —
    /// flow graphs carry module-level constants as `Constant(value)`,
    /// so config bools like `WITHPREBUILTINT` constant-fold their
    /// guarded branches instead of minting a synthetic 0-arg call no
    /// registry can resolve.  Aggregates and any `Call` other than
    /// `core::mem::size_of` / `align_of` return `None` and keep the
    /// Call fallback.
    ///
    /// A defining crate's initializer is absent from a dependent
    /// artefact (`opacity: Foreign`). After the local body and the
    /// `core::num` associated-const lanes fail, fold to the literal
    /// harvested from the crate that owns the const.
    pub(in crate::front::mir) fn const_eval_global(&self, def_id: u64) -> Option<OpKind> {
        let g = self.llbc.global_by_id(def_id)?;
        // Only an immutable, non-thread-local global folds to its init
        // literal.  A `static mut` is written at runtime and a
        // thread-local holds a per-thread value, so a post-init read of
        // either must reach the live accessor, not the initialiser.  (A
        // by-value read carries no address identity, so an immutable
        // `static`'s literal is safe to inline here even when its address
        // is taken elsewhere.)  `global_kind` does not distinguish
        // `static mut` from `static`, so the mutability comes from the
        // `static mut` keyword in Charon's recorded `source_text`.
        if global_is_thread_local(g) {
            return None;
        }
        if global_source_text_is_static_mut(g) {
            return None;
        }
        let local = crate::front::llbc_hints::marker_init_fun_id(g)
            .and_then(|init_id| self.llbc.fn_by_id(init_id))
            .and_then(|fd| {
                if let Some(u) = fd.unstructured() {
                    const_eval_init_body(self.llbc, &u)
                } else {
                    const_eval_core_num_associated_const(self.llbc, def_id)
                        .and_then(const_lit_to_op)
                }
            });
        local.or_else(|| attached_foldable_lit(self.llbc, def_id))
    }

    /// Fold a `NamedConst` global whose initializer is exactly
    /// `size_of::<T>()` / `align_of::<T>()` to the concrete byte size /
    /// alignment Charon resolved for `T`'s layout.  The const's value is
    /// a build-time constant of the extraction target, so folding it to
    /// a `ConstInt` removes the residual accessor call the layout-size
    /// consts (`FUNCTION_OBJECT_SIZE`, `W_DICT_OBJECT_SIZE`, …) otherwise
    /// lower to (a `FunctionPath` the rtyper cannot register).
    ///
    /// Accepts only the *exact* shape — `_0` (the const's value) is written
    /// once, by a single `size_of`/`align_of` call, unconditionally — and
    /// returns `None` for anything richer so it keeps the residual accessor
    /// path, mirroring the strict single-statement acceptance in
    /// [`Self::const_eval_global`].  Rejected:
    ///   * a `Switch` terminator (data-dependent control flow), so the call
    ///     can never be made conditional;
    ///   * any statement that assigns `_0` (a computed value such as
    ///     `size_of::<A>() + size_of::<B>()`, whose `+` writes `_0`);
    ///   * a second `_0`-defining call, or a `_0`-defining call to any
    ///     function other than `size_of`/`align_of`;
    ///   * a body with no `Return`.
    ///     Linear `Goto`s and panic-cleanup terminators (`Abort`,
    ///     `UnwindResume`, `Drop`, `Assert`), plus calls and assignments that
    ///     do *not* write `_0`, are permitted: none of them define the const
    ///     value, so with no `Switch` the single `size_of` call is its
    ///     unconditional sole definer.  Primitive widths (`usize`, `u64`, …)
    ///     fold through the same lane as an inline `size_of`/`align_of` call;
    ///     a pointer / tuple / unresolved layout stays residual.
    pub(in crate::front::mir) fn fold_size_const_global(&self, def_id: u64) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
        let body = self.llbc.fn_by_id(init_id)?.unstructured()?;

        // The single `size_of`/`align_of` call that defines `_0`, captured
        // as `(want_align, type_argument)`.  `term()`/`stmt_kind()` return
        // owned values, so the type expression is cloned out of the call.
        let mut found: Option<(bool, serde_json::Value)> = None;
        let mut saw_return = false;
        let forward_blocks = forward_reachable_mask(self.llbc, &body);
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &block.statements {
                match stmt.stmt_kind_ref() {
                    Ok(StmtKind::StorageLive(_))
                    | Ok(StmtKind::StorageDead(_))
                    | Ok(StmtKind::PlaceMention(_))
                    | Ok(StmtKind::Borrowck(_)) => {}
                    // A statement writing `_0` means the value is computed,
                    // not the bare `size_of`/`align_of` call result.
                    Ok(StmtKind::Assign(place, _)) if matches!(place.kind, PlaceKind::Local(0)) => {
                        return None;
                    }
                    // An assignment to a temporary cannot reach `_0` without
                    // an `_0` write we already reject, so it is inert here.
                    Ok(StmtKind::Assign(_, _)) => {}
                    // `Assert` statements and anything unparsed: bail.
                    _ => return None,
                }
            }
            match block.term_ref(self.llbc) {
                // Data-dependent control flow or an unreadable terminator:
                // the const value would not be the unconditional size_of.
                Ok(TermKind::Switch { .. }) | Ok(TermKind::Unknown) | Err(_) => {
                    return None;
                }
                Ok(TermKind::Return) => saw_return = true,
                Ok(TermKind::Call { call, .. })
                    if matches!(call.dest.kind, PlaceKind::Local(0)) =>
                {
                    let CallFunc::Regular(reg) = &call.func else {
                        return None;
                    };
                    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
                        return None;
                    };
                    let want_align = match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
                        "core::mem::size_of" => false,
                        "core::mem::align_of" => true,
                        // `_0` defined by some other function: not a bare
                        // size_of/align_of const.
                        _ => return None,
                    };
                    // A second `_0`-defining call makes the value
                    // order-dependent; only a single one is foldable.
                    if found.is_some() {
                        return None;
                    }
                    let ty = reg.generics.get("types")?.as_array()?.first()?.clone();
                    found = Some((want_align, ty));
                }
                // `Goto` / `Abort` / `UnwindResume` / `Drop` / `Assert`, and
                // any call that does not define `_0`: linear continuation or
                // panic cleanup, none of which write the const value.
                Ok(_) => {}
            }
        }
        let (want_align, ty) = found?;
        if !saw_return {
            return None;
        }
        Some(OpKind::ConstInt(
            self.size_align_const_from_tyexpr(want_align, &ty)?,
        ))
    }

    /// Fold a `NamedConst` whose initializer computes an integer and wraps
    /// it in a single-field ADT (`from_bits_retain` + `Aggregate`).
    ///
    /// `bitflags!` associated consts (`impl Flags { const NAME: Self =
    /// Self::from_bits_retain(1 << k) }`) compile to that shape. The init
    /// function is a Transparent Unstructured body, so the numeric value
    /// is in the LLBC; [`const_eval_init_body`] follows the wrapper Call
    /// and the Aggregate. Foreign-crate consts whose init is `Opaque`
    /// still miss this lane.
    pub(in crate::front::mir) fn fold_transparent_int_const_global(
        &self,
        def_id: u64,
    ) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
        let body = self.llbc.fn_by_id(init_id)?.unstructured()?;
        const_eval_init_body(self.llbc, &body)
    }

    /// The build-time byte size / alignment of a `size_of` / `align_of`
    /// type argument, shared by [`Self::fold_size_const_global`] (the
    /// NamedConst-initializer form) and the inline `size_of`/`align_of` call
    /// fold in [`Self::lower_call`].
    ///
    /// ADT arguments read Charon's `layout_for_target` (wasm32 folds the
    /// wasm32 width, not the host's). Primitive widths (`usize`, `u64`,
    /// …) reuse [`primitive_size_align`], the same lane a NamedConst
    /// initializer already uses, so an inline `align_of::<usize>()`
    /// folds too.  A pointer to a sized pointee is one address word.
    /// `None` for a fat pointer / tuple / unresolved layout.
    pub(in crate::front::mir) fn size_align_const_from_tyexpr(
        &self,
        want_align: bool,
        ty: &serde_json::Value,
    ) -> Option<i64> {
        if let Some(adt) = self.resolve_tyexpr_to_adt_def_id(ty) {
            // Read the layout for the build's `TARGET` (empty for the host
            // extraction target), matching the target-aware field-offset lookup
            // in `record_struct_id`. A wasm32 cross-build folds the wasm32 byte
            // size, not the host's.
            let target = std::env::var("TARGET").unwrap_or_default();
            let decl = self.llbc.type_by_id(adt)?;
            if let Some((size, align)) = decl.size_align_for_target(self.llbc, &target) {
                let value = if want_align { align } else { size };
                if let Some(value) = value {
                    return Some(value as i64);
                }
            }
            // `bytecode::CodeObject` is a foreign struct. Charon records
            // only an `AtLeast` guarantee, no chosen align. Its fields are
            // pointers, boxes and `u32`s, so the alignment is the target
            // word (`CARGO_CFG_TARGET_POINTER_WIDTH`).
            if want_align && decl.item_meta.name_path().ends_with("bytecode::CodeObject") {
                return Some(crate::layout::target_word_size() as i64);
            }
            // Charon leaves a generic struct's `chosen` layout null. The
            // declaration-order walk still yields a concrete size once the
            // type arguments are instantiated, when the recorded repr keeps
            // that order (`repr(C)` or `repr(transparent)`).
            if let Some((size, align)) = mem_size_align(self.llbc, ty, &[], 0) {
                return i64::try_from(if want_align { align } else { size }).ok();
            }
            return None;
        }
        let body = tyexpr_body(self.llbc, ty)?;
        // `{"RawPtr": [ty, kind]}` / `{"Ref": [region, ty, kind]}`.
        let pointee = match (body.get("RawPtr"), body.get("Ref")) {
            (Some(raw), _) => raw.get(0),
            (_, Some(reference)) => reference.get(1),
            _ => None,
        };
        if let Some(pointee) = pointee {
            // A pointer to a sized pointee is one address word.
            let pointee = tyexpr_body(self.llbc, pointee)?;
            let sized = pointee.get("Scalar").is_some()
                || pointee
                    .get("Adt")
                    .is_some_and(|adt| adt.get("builtin").is_none());
            return sized.then(|| crate::layout::target_word_size() as i64);
        }
        i64::try_from(primitive_size_align(want_align, body.get("Scalar")?)?).ok()
    }
}

pub(in crate::front::mir) fn encode_foldable_op(op: &OpKind) -> Option<String> {
    match op {
        OpKind::ConstInt(n) => Some(format!("i{n}")),
        OpKind::ConstUInt(n) => Some(format!("u{n}")),
        OpKind::ConstBool(b) => Some(format!("b{}", u8::from(*b))),
        OpKind::ConstFloat(bits) => Some(format!("f{bits}")),
        OpKind::ConstSingleFloat(bits) => Some(format!("s{bits}")),
        _ => None,
    }
}

pub(in crate::front::mir) fn decode_foldable_op(encoded: &str) -> Option<OpKind> {
    let (tag, rest) = encoded.split_at(encoded.chars().next()?.len_utf8());
    match tag {
        "i" => Some(OpKind::ConstInt(rest.parse().ok()?)),
        "u" => Some(OpKind::ConstUInt(rest.parse().ok()?)),
        "b" => Some(OpKind::ConstBool(rest == "1")),
        "f" => Some(OpKind::ConstFloat(rest.parse().ok()?)),
        "s" => Some(OpKind::ConstSingleFloat(rest.parse().ok()?)),
        _ => None,
    }
}

/// Literal previously written onto this artefact's global `def_id`.
pub(crate) fn attached_foldable_lit(llbc: &Llbc, def_id: u64) -> Option<OpKind> {
    decode_foldable_op(&llbc.foldable_const_lit(def_id)?)
}

pub(in crate::front::mir) fn global_is_thread_local(g: &GlobalDecl) -> bool {
    g.rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        == Some("ThreadLocal")
}

pub(in crate::front::mir) fn global_source_text_is_static_mut(g: &GlobalDecl) -> bool {
    g.item_meta
        .source_text
        .as_deref()
        .is_some_and(|s| s.contains("static mut"))
        || g.rest
            .get("item_meta")
            .and_then(|m| m.get("source_text"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| s.contains("static mut"))
}

/// Harvest foldable const literals from one artefact.
///
/// A global whose initializer Charon kept as a body folds under the
/// same gates as [`Lowering::const_eval_global`]: not a `ThreadLocal`,
/// not a `static mut`, body present, and [`const_eval_init_body_lit`]
/// succeeds — including a `size_of` / `align_of` Call, which is the
/// only Call shape that evaluator will fold. Dependency artefacts
/// record the same item as `opacity: Foreign` with no body; the
/// defining crate's harvest is what a later lowering consults for
/// those reads.
///
/// `item_meta.name_path()` renders every trait-impl segment as
/// `"<Impl>"`, so two associated consts in one module share a path
/// (`::<Impl>::CPYTHON_IMMUTABLETYPE`) and can fold to different
/// values. A path that names more than one global in this artefact is
/// not an identity, so it is left unharvested rather than folded or
/// asserted.
pub(crate) fn discover_foldable_const_lits(llbc: &Llbc) -> Vec<(String, OpKind)> {
    let mut paths: Vec<String> = llbc
        .iter_global_decls()
        .map(|g| g.item_meta.name_path())
        .collect();
    paths.sort();
    let mut ambiguous: Vec<String> = Vec::new();
    for pair in paths.windows(2) {
        if pair[0] == pair[1] && ambiguous.last() != Some(&pair[0]) {
            ambiguous.push(pair[0].clone());
        }
    }
    let mut discovered = Vec::new();
    for g in llbc.iter_global_decls() {
        if global_is_thread_local(g) || global_source_text_is_static_mut(g) {
            continue;
        }
        let path = g.item_meta.name_path();
        if ambiguous.binary_search(&path).is_ok() {
            continue;
        }
        let Some(init_id) = crate::front::llbc_hints::marker_init_fun_id(g) else {
            continue;
        };
        let Some(fd) = llbc.fn_by_id(init_id) else {
            continue;
        };
        let Some(u) = fd.unstructured() else {
            continue;
        };
        let Some(op) = const_eval_init_body(llbc, &u) else {
            continue;
        };
        discovered.push((path, op));
    }
    discovered
}

/// Impl associated consts share one `name_path` (`<Impl>`). Register each
/// by its defining global's `def_id` instead of that path.
#[cfg(any(test, feature = "mir-frontend"))]
pub(crate) fn register_ambiguous_impl_foldable_const_lits(llbc: &Llbc) {
    let mut paths: Vec<String> = llbc
        .iter_global_decls()
        .map(|g| g.item_meta.name_path())
        .collect();
    paths.sort();
    let mut ambiguous: Vec<String> = Vec::new();
    for pair in paths.windows(2) {
        if pair[0] == pair[1] && ambiguous.last() != Some(&pair[0]) {
            ambiguous.push(pair[0].clone());
        }
    }
    for g in llbc.iter_global_decls() {
        let path = g.item_meta.name_path();
        if !path.contains('<') || ambiguous.binary_search(&path).is_err() {
            continue;
        }
        if global_is_thread_local(g) || global_source_text_is_static_mut(g) {
            continue;
        }
        let Some(init_id) = crate::front::llbc_hints::marker_init_fun_id(g) else {
            continue;
        };
        let Some(fd) = llbc.fn_by_id(init_id) else {
            continue;
        };
        let Some(u) = fd.unstructured() else {
            continue;
        };
        let Some(op) = const_eval_init_body(llbc, &u) else {
            continue;
        };
        let Some(encoded) = encode_foldable_op(&op) else {
            continue;
        };
        llbc.register_foldable_const_lit(g.def_id, encoded);
    }
}

/// Merge foldable literals that share an injective path.
///
/// A path containing `<` is not an identity (`name_path` renders every
/// trait impl as `<Impl>`). Callers keep those on the defining global's
/// decl id and do not pass them here. Two artefacts that fold one
/// injective path to different values is a bug, not a silent winner.
pub(crate) fn register_foldable_const_lits(
    entries: impl IntoIterator<Item = (String, OpKind)>,
) -> Vec<(String, OpKind)> {
    let mut lits: Vec<(String, OpKind)> = Vec::new();
    for (path, lit) in entries {
        match lits.binary_search_by(|(known, _)| known.cmp(&path)) {
            Ok(index) => assert_eq!(
                lits[index].1, lit,
                "foldable const {path} has inconsistent linked definitions"
            ),
            Err(index) => lits.insert(index, (path, lit)),
        }
    }
    lits
}

/// Write each harvested literal onto the global in `llbc` whose
/// `name_path` matches, stored under that global's decl id.
pub(crate) fn attach_foldable_const_lits(llbc: &Llbc, entries: &[(String, OpKind)]) {
    if entries.is_empty() {
        return;
    }
    let mut entries = entries.to_vec();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    for g in llbc.iter_global_decls() {
        let path = g.item_meta.name_path();
        let Ok(index) = entries.binary_search_by(|(known, _)| known.as_str().cmp(path.as_str()))
        else {
            continue;
        };
        let Some(encoded) = encode_foldable_op(&entries[index].1) else {
            continue;
        };
        llbc.register_foldable_const_lit(g.def_id, encoded);
    }
}

/// Link complete transparent-scalar declarations to opaque dependency views.
///
/// Charon keeps `repr(transparent)` in an external declaration's layout but
/// omits its private field type. The defining crate's LLBC contains that field;
/// both declarations share the same qualified Rust path. RPython translates a
/// linked type universe, so collect that one register-bank fact before lowering
/// any body and publish it to every artefact in the same translation input.
pub(crate) fn discover_transparent_scalar_kinds(
    llbc: &Llbc,
) -> Vec<(String, majit_charon_reader::TransparentScalarKind)> {
    let mut discovered = Vec::new();
    for decl in llbc.iter_type_decls() {
        if !decl.is_repr_transparent() || decl.published_struct_copy {
            continue;
        }
        let TypeDeclKind::Struct(fields) = &decl.kind else {
            continue;
        };
        let Some((index, _)) = transparent_nonzst_field(decl, llbc) else {
            continue;
        };
        let Some(field) = fields.get(index) else {
            continue;
        };
        let kind = match tyref_to_value_type(&field.ty, llbc) {
            ValueType::Int => majit_charon_reader::TransparentScalarKind::Signed,
            ValueType::Unsigned => majit_charon_reader::TransparentScalarKind::Unsigned,
            ValueType::Bool => majit_charon_reader::TransparentScalarKind::Bool,
            ValueType::Float => majit_charon_reader::TransparentScalarKind::Float,
            _ => continue,
        };
        discovered.push((decl.item_meta.name_path(), kind));
    }
    // The defining crate is outside the extraction set, so the `u32` field
    // is known from `bitflags!` — the same knowledge [`code_flags_const`]'s
    // bit table already encodes. Keyed by the same crate, module, and type.
    discovered.extend(external_transparent_scalar_kinds());
    discovered
}

/// Transparent scalars whose defining crate is not in the extraction set.
///
/// `rustpython_compiler_core::bytecode::CodeFlags` is the `bitflags!`
/// `repr(transparent)` `u32` wrapper. No other external type is listed.
pub(in crate::front::mir) fn external_transparent_scalar_kinds()
-> Vec<(String, majit_charon_reader::TransparentScalarKind)> {
    vec![(
        "rustpython_compiler_core::bytecode::CodeFlags".to_string(),
        majit_charon_reader::TransparentScalarKind::Unsigned,
    )]
}

pub(in crate::front::mir) fn link_transparent_scalar_types<L: std::borrow::Borrow<Llbc>>(
    llbcs: &[L],
) {
    let llbc_refs: Vec<&Llbc> = llbcs.iter().map(std::borrow::Borrow::borrow).collect();
    let llbcs: &[&Llbc] = &llbc_refs;
    let mut discovered = Vec::new();
    let mut foldable_cross = Vec::new();
    let mut foldable_impl: Vec<Vec<(String, OpKind)>> = Vec::new();
    for llbc in llbcs {
        discovered.extend(discover_transparent_scalar_kinds(llbc));
        let mut impl_folds = Vec::new();
        for (path, op) in discover_foldable_const_lits(llbc) {
            if path.contains('<') {
                impl_folds.push((path, op));
            } else {
                foldable_cross.push((path, op));
            }
        }
        foldable_impl.push(impl_folds);
    }
    discovered.sort_by(|a, b| a.0.cmp(&b.0));
    discovered.dedup();
    let foldable_cross = register_foldable_const_lits(foldable_cross);
    for (llbc, impl_folds) in llbcs.iter().zip(foldable_impl) {
        llbc.register_transparent_scalar_kinds(discovered.iter().cloned());
        attach_foldable_const_lits(llbc, &foldable_cross);
        attach_foldable_const_lits(llbc, &impl_folds);
    }
}

/// Whether an impl-owned global is exactly
/// `PyreClassPyTypeOf::PYTYPE`.
///
/// The associated-item name selects the value; the trait path selects the
/// namespace that gives that name its meaning.  The impl id deliberately does
/// neither — it identifies only the implementing `Self` type.  Keeping these
/// three axes separate mirrors RPython's prebuilt lookup: object identity
/// chooses the owner while the attribute lookup still chooses one field of
/// that object.
pub(in crate::front::mir) fn is_class_pytype_assoc_const(
    item_path: &str,
    trait_path: &str,
) -> bool {
    item_path.rsplit("::").next() == Some("PYTYPE")
        && path_ends_with_segments(trait_path, "pyre_object::lltype::PyreClassPyTypeOf")
}

pub(in crate::front::mir) fn static_key_matches(full: &str, stripped: &str, key: &str) -> bool {
    path_eq_ignoring_raw(full, key)
        || path_eq_ignoring_raw(stripped, key)
        || path_has_suffix_ignoring_raw(full, key)
        || path_has_suffix_ignoring_raw(stripped, key)
}

/// Path segments of a Global place, including the Ident disambiguator
/// Charon records as `Ident [name, n]`.
///
/// `item_meta.name_path()` is the human-facing label and is not injective:
/// it keeps `closure#N` so co-located closure envs stay distinct ClassDefs,
/// but drops every other Ident's `n`. Function-local statics reuse a leaf
/// (`OWNER` inside `method_owner`, rustc `n` in `0..57`) and would all
/// look up as one key. `n == 0` stays the bare ident so existing
/// module-level table keys keep matching; `n > 0` is spelled `name#n`,
/// the same shape `name_path` already uses for closures. Other segments
/// still render as `<Variant>` — a label, recovered by
/// `item_meta.trait_impl_id()` when identity is required.
pub(in crate::front::mir) fn static_key_segments(name: &[NameSeg]) -> Vec<String> {
    name.iter().map(static_key_segment).collect()
}

pub(in crate::front::mir) fn static_key_segment(seg: &NameSeg) -> String {
    match seg {
        NameSeg::Ident {
            ident: (s, disambiguator),
        } => {
            if *disambiguator > 0 {
                format!("{s}#{disambiguator}")
            } else {
                s.clone()
            }
        }
        NameSeg::Other(v) => {
            if let Some(label) = majit_charon_reader::ullbc::builtin_path_label(v) {
                return label;
            }
            let label = v
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .unwrap_or_else(|| "?".into());
            format!("<{label}>")
        }
    }
}

/// Supply the value of a primitive `f64` associated constant whose
/// initializer Charon records as an `Opaque` body — `core` defines
/// `f64::INFINITY` as `1.0_f64 / 0.0_f64` and `f64::NAN` as
/// `0.0_f64 / 0.0_f64`, so no in-LLBC init survives for
/// [`Lowering::const_eval_global`] to evaluate.  The value is a
/// fixed IEEE-754 bit pattern the host (rustc) already computed, so
/// emit it as the same by-value `ConstFloat` an inline float literal
/// lowers to (mirroring `rfloat.INFINITY` reaching the flow graph as a
/// float `Constant`).  Matches on the `f64::<Impl>::<NAME>` tail so a
/// `core`- or `std`-rooted path resolves identically.
///
/// `f64::consts::<NAME>` reaches the same dead end for the same reason:
/// the `NamedConst` has no address and its initializer lives in `core`,
/// which is outside the extraction set, so `fold_named_const_global`
/// finds nothing to fold and the read would otherwise become a nullary
/// residual call on the constant's own path.  `math.pi` is a float
/// `Constant` in the flow graph upstream, and so is this.
pub(in crate::front::mir) fn primitive_float_const(segments: &[String]) -> Option<OpKind> {
    let tail: Vec<&str> = segments
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(String::as_str)
        .collect();
    let bits = match tail.as_slice() {
        ["f64", "<Impl>", name] => primitive_f64_assoc_bits(name)?,
        ["f64", "consts", name] => float_consts_value(name)?.to_bits(),
        _ => return None,
    };
    Some(OpKind::ConstFloat(bits))
}

/// IEEE-754 bits of `f64::<NAME>` associated constants.  Integer-valued
/// siblings (`RADIX`, `MANTISSA_DIGITS`, the `*_EXP` family) are not
/// floats and stay residual.
pub(in crate::front::mir) fn primitive_f64_assoc_bits(name: &str) -> Option<u64> {
    Some(match name {
        "INFINITY" => f64::INFINITY.to_bits(),
        "NEG_INFINITY" => f64::NEG_INFINITY.to_bits(),
        "NAN" => f64::NAN.to_bits(),
        "EPSILON" => f64::EPSILON.to_bits(),
        "MIN" => f64::MIN.to_bits(),
        "MAX" => f64::MAX.to_bits(),
        "MIN_POSITIVE" => f64::MIN_POSITIVE.to_bits(),
        _ => return None,
    })
}

/// The value `core::f64::consts::<NAME>` holds.  Spelled through the host's
/// own definitions so the bit pattern is the one rustc would have inlined at
/// the use site.
pub(in crate::front::mir) fn float_consts_value(name: &str) -> Option<f64> {
    use std::f64::consts;
    Some(match name {
        "PI" => consts::PI,
        "TAU" => consts::TAU,
        "E" => consts::E,
        "SQRT_2" => consts::SQRT_2,
        "LN_2" => consts::LN_2,
        "LN_10" => consts::LN_10,
        "LOG2_E" => consts::LOG2_E,
        "LOG10_E" => consts::LOG10_E,
        "LOG2_10" => consts::LOG2_10,
        "LOG10_2" => consts::LOG10_2,
        "FRAC_1_PI" => consts::FRAC_1_PI,
        "FRAC_2_PI" => consts::FRAC_2_PI,
        "FRAC_1_SQRT_2" => consts::FRAC_1_SQRT_2,
        "FRAC_2_SQRT_PI" => consts::FRAC_2_SQRT_PI,
        "FRAC_PI_2" => consts::FRAC_PI_2,
        "FRAC_PI_3" => consts::FRAC_PI_3,
        "FRAC_PI_4" => consts::FRAC_PI_4,
        "FRAC_PI_6" => consts::FRAC_PI_6,
        "FRAC_PI_8" => consts::FRAC_PI_8,
        _ => return None,
    })
}

/// Target-layout value of the scalar GcArray's items offset.  Charon leaves
/// Rust's `offset_of!(TypedItemsBlock, items)` initializer opaque, but the
/// translated lltype fixes the same layout structurally: a length header
/// followed inline by the `GcArray(Signed|Float)` items
/// (`get_itemarray_lowleveltype`, `rlist.py`).
///
/// The derivation is `align_of::<u64>()` padding of that header, not "one
/// target word": the Rust body is `#[repr(C)] { capacity: usize, items:
/// [u64; 0] }` (`majit/majit-rlib/src/lltypesystem/rlist.rs:165-171`), whose
/// 8-byte item alignment rounds a 4- or 8-byte header up to the same 8.  The
/// offset is therefore width-independent, which is why reading `word` from
/// [`crate::layout::target_word_size`] here rather than from Charon's
/// `target_pointer_size` cannot change the answer.
///
/// This is the front-end counterpart of `CallControl::array_items_base`; it
/// derives the target value instead of importing a host-runtime constant or
/// registering a fake residual getter.
pub(in crate::front::mir) fn known_array_layout_const(segments: &[String]) -> Option<OpKind> {
    let path = segments.join("::");
    // `{capacity: usize, items: [T; 0]}`: the items start after the one-word
    // length header, at `T`'s alignment.
    let header = crate::layout::target_word_size();
    let items_offset = if path_ends_with_segments(&path, "rlist::TYPED_ITEMS_BLOCK_ITEMS_OFFSET") {
        header.next_multiple_of(align_of::<u64>())
    } else if path_ends_with_segments(&path, "object_array::ITEMS_BLOCK_ITEMS_OFFSET") {
        header
    } else {
        return None;
    };
    Some(OpKind::ConstInt(items_offset as i64))
}

/// `GcRef::NULL` is the null GC reference (`GcRef(0)`).
///
/// The associated const's initializer is outside the extracted crates, so
/// the read would otherwise stay a nullary call to
/// `majit_ir::value::<Impl>::NULL`. Emit [`OpKind::ConstRefNull`], the
/// same sentinel `lltype.nullptr` lowers to.
pub(in crate::front::mir) fn gc_ref_null_const(segments: &[String]) -> Option<OpKind> {
    let [crate_name, module, impl_seg, leaf] = segments else {
        return None;
    };
    if crate_name.as_str() == "majit_ir"
        && module.as_str() == "value"
        && impl_seg.as_str() == "<Impl>"
        && leaf.as_str() == "NULL"
    {
        Some(OpKind::ConstRefNull)
    } else {
        None
    }
}

/// Supply the value of `bitflags::traits::Bits::EMPTY`.
///
/// Charon records the associated const's initializer as an `Opaque`
/// body: the impl lives in `bitflags`, which is outside the extraction
/// set, so [`Lowering::const_eval_global`] finds nothing to evaluate.
/// Every primitive `Bits` impl defines `EMPTY` as `0` (`impl_bits!`),
/// so emit that `0` as the same by-value `ConstInt` an inline integer
/// literal lowers to.  The crate + trait-module + associated-item leaf
/// select the value; the impl identity is unused, because every
/// primitive `Bits` impl holds the same `0` and Charon renders every
/// one as `<Impl>`.
///
/// `Bits::ALL` is type-width-specific (`u8::MAX` vs `u64::MAX`) under
/// the same non-injective path, so it stays residual.
pub(in crate::front::mir) fn bitflags_trait_empty_const(segments: &[String]) -> Option<OpKind> {
    // `Bits::EMPTY` is the impl's own initializer (`impl_bits!`), keyed by
    // the concrete Self type the way `fold_size_const_global` keys a layout.
    // Charon renders every impl as `<Impl>` and leaves the external impl
    // body opaque, so a leaf match cannot see that Self. Leave the read
    // residual; a defining initializer still folds through `const_eval_global`.
    let _ = segments;
    None
}

/// Supply the value of a `CodeFlags` associated constant. `bitflags!`
/// generates each flag as an `impl CodeFlags { const NAME: Self }` whose
/// initializer Charon records as an `Opaque` body (the `<Impl>` segment
/// anonymizes to `_`), so [`Lowering::const_eval_global`] finds no
/// in-LLBC init to evaluate and the read stays a `not registered` Call.
/// The value is a fixed compile-time `u32` bitmask, so emit it by-value
/// as the same `ConstUInt` an inline unsigned integer literal lowers to —
/// a `flags.contains(CodeFlags::VARARGS)` test then folds to the integer
/// bit-and the `bitflags` `.contains` inlines to. Mirrors
/// [`primitive_float_const`] for `f64::INFINITY`.
/// `rustpython_common::encodings::<codec>::ENCODING_NAME` is a `&str`
/// constant in another crate. Charon leaves the initializer opaque, so
/// the global read is the literal the codec module publishes.
pub(in crate::front::mir) fn encoding_name_const(segments: &[String]) -> Option<OpKind> {
    if !segments.iter().any(|segment| segment == "encodings") {
        return None;
    }
    let module = segments.get(segments.len().wrapping_sub(2))?.as_str();
    let leaf = segments.last()?.as_str();
    let text = match (module, leaf) {
        ("utf8", "ENCODING_NAME") => "utf-8",
        ("latin_1", "ENCODING_NAME") => "latin-1",
        ("ascii", "ENCODING_NAME") => "ascii",
        ("unicode_escape", "ENCODING_NAME") => "unicodeescape",
        ("raw_unicode_escape", "ENCODING_NAME") => "rawunicodeescape",
        ("utf7", "ENCODING_NAME") => "utf-7",
        ("utf16", "ENCODING_NAME") => "utf-16",
        ("utf16", "ENCODING_NAME_LE") => "utf-16-le",
        ("utf16", "ENCODING_NAME_BE") => "utf-16-be",
        ("utf32", "ENCODING_NAME") => "utf-32",
        ("utf32", "ENCODING_NAME_LE") => "utf-32-le",
        ("utf32", "ENCODING_NAME_BE") => "utf-32-be",
        _ => return None,
    };
    Some(OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: vec!["__str_const".to_string(), text.to_string()],
            fun_decl_id: None,
            specialization: crate::parse::GraphSpecialization::Regular,
        },
        args: crate::model::call_args(vec![]),
        result_ty: ValueType::Ref(None),
    })
}

pub(in crate::front::mir) fn code_flags_const(segments: &[String]) -> Option<OpKind> {
    let [first, second, .., impl_seg, leaf] = segments else {
        return None;
    };
    if first.as_str() != "rustpython_compiler_core"
        || second.as_str() != "bytecode"
        || impl_seg.as_str() != "<Impl>"
    {
        return None;
    }
    let bits: u32 = match leaf.as_str() {
        "OPTIMIZED" => 0x0001,
        "NEWLOCALS" => 0x0002,
        "VARARGS" => 0x0004,
        "VARKEYWORDS" => 0x0008,
        "NESTED" => 0x0010,
        "GENERATOR" => 0x0020,
        "COROUTINE" => 0x0080,
        "ITERABLE_COROUTINE" => 0x0100,
        "ASYNC_GENERATOR" => 0x0200,
        "FUTURE_DIVISION" => 0x20000,
        "FUTURE_ABSOLUTE_IMPORT" => 0x40000,
        "FUTURE_WITH_STATEMENT" => 0x80000,
        "FUTURE_PRINT_FUNCTION" => 0x100000,
        "FUTURE_UNICODE_LITERALS" => 0x200000,
        "FUTURE_BARRY_AS_BDFL" => 0x400000,
        "FUTURE_GENERATOR_STOP" => 0x800000,
        "FUTURE_ANNOTATIONS" => 0x1000000,
        "HAS_DOCSTRING" => 0x4000000,
        "METHOD" => 0x8000000,
        _ => return None,
    };
    Some(OpKind::ConstUInt(u64::from(bits)))
}

/// `bitflags` method on `rustpython_compiler_core::bytecode::CodeFlags`.
///
/// The impl body is outside the extracted crates, so the call would stay a
/// residual `FunctionPath`. The owner test is the same crate and module
/// [`code_flags_const`] uses, plus the `CodeFlags` type: a same-named method
/// on any other type stays a call. `contains` / `intersects` / `bitor` are
/// the integer operations the macro expands to; `bits` is the transparent
/// `u32` word those ops already read. Every other method stays residual.
pub(in crate::front::mir) fn code_flags_method(
    llbc: &Llbc,
    reg: &RegularCall,
) -> Option<(&'static str, TyRef)> {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let fd = llbc.fn_by_id(*id)?;
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    let leaf = match &segs[last_idx] {
        NameSeg::Ident { ident: (s, _) } => s.as_str(),
        _ => return None,
    };
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let owner_id = resolve_impl_owner_adt_def_id_free(llbc, impl_payload)?;
    let owner = llbc.type_by_id(owner_id)?;
    if owner.item_meta.name_path() != "rustpython_compiler_core::bytecode::CodeFlags" {
        return None;
    }
    let leaf = match leaf {
        "contains" => "contains",
        "intersects" => "intersects",
        "bitor" => "bitor",
        "bits" => "bits",
        _ => return None,
    };
    // `contains` / `intersects` / `bitor` take `other: Self`, the by-value
    // `CodeFlags` a `*self` read also has. `bits` is `&self` only, so the
    // same type is the pointee of the receiver.
    let flags_ty = if leaf == "bits" {
        let first = fd.signature.inputs.first()?;
        ref_pointee_ty(first, llbc).or_else(|| pointer_pointee_ty(first, llbc))?
    } else {
        clone_tyref(fd.signature.inputs.get(1)?)
    };
    Some((leaf, flags_ty))
}

pub(in crate::front::mir) fn place_kind_label(k: &PlaceKind) -> &'static str {
    match k {
        PlaceKind::Local(_) => "Local",
        PlaceKind::Projection(_, _) => "Projection",
        PlaceKind::Global { .. } => "Global",
        PlaceKind::Unknown => "Unknown",
    }
}

pub(in crate::front::mir) fn rvalue_variant_name(rv: &Rvalue) -> &'static str {
    match rv {
        Rvalue::Use(_, _) => "Use",
        Rvalue::BinaryOp(..) => "BinaryOp",
        Rvalue::UnaryOp(..) => "UnaryOp",
        Rvalue::Ref { .. } => "Ref",
        Rvalue::Aggregate(..) => "Aggregate",
        Rvalue::Discriminant(_) => "Discriminant",
        Rvalue::Cast(..) => "Cast",
        Rvalue::Len(_) => "Len",
        Rvalue::Repeat(..) => "Repeat",
        Rvalue::RawPtr { .. } => "RawPtr",
        Rvalue::NullaryOp(..) => "NullaryOp",
        Rvalue::ShallowInitBox(..) => "ShallowInitBox",
        Rvalue::Unknown => "Unknown",
    }
}

/// Subset of MIR constant kinds the driver currently knows how to
/// emit. Widen as the corpus grows past `straight_line_add`.
pub(in crate::front::mir) enum DecodedConst {
    Int(i64),
    UInt(u64),
    Int128(i128),
    UInt128(u128),
    Bool(bool),
    Float(u64),
    /// An `f32` literal, as its f32 bit pattern.  Charon records the
    /// width on the constant (`{"Float": {"value": "...", "ty": "F32"}}`),
    /// and parsing both widths into an f64 bit pattern is what erased it.
    SingleFloat(u32),
    /// String / byte-string literals. The IR has no dedicated string
    /// constant opkind; the codewriter treats these as opaque pointer-typed
    /// values. We carry the textual representation as a unique-string
    /// `ConstValue` so the generated IR is stable across runs.
    Str(String),
    /// A Rust `char` literal: one Unicode scalar
    /// (`bookkeeper.py` `immutablevalue` `SomeUnicodeCodePoint`).
    UniChar(u32),
    /// Constant function pointer (`FnDef`). Encoded as a synthetic
    /// `FunctionPath` so it shares the existing `Call` lowering path
    /// when threaded into an indirect call site.
    FnPath(Vec<String>),
}

/// Literal value tracked by [`const_eval_init_body`].
#[derive(Clone, Copy)]
pub(in crate::front::mir) enum ConstLit {
    Int(i64),
    /// See [`DecodedConst::SingleFloat`].
    SingleFloat(u32),
    /// Unsigned integer bits.  Keeping this distinct is required while
    /// evaluating const initializers: `(1_u64 << 63) - 1` cannot be
    /// evaluated correctly after prematurely reinterpreting the first
    /// shift result as a negative `i64`.
    UInt(u64),
    Bool(bool),
    Float(u64),
    /// `*Checked` binop result `(value, overflowed)`; read back via
    /// `Field[Tuple 2, 0|1]` projections.
    Checked(i64, bool),
    CheckedUInt(u64, bool),
    /// Fieldless enum discriminant. Not a folded constant by itself:
    /// `bitflagset`'s `from_element` shifts by this value (`element as u8`).
    EnumDisc(u64),
    /// `1 << disc` from an opaque `from_element`. `bits` reads the word back.
    BitflagBits(u64),
}

thread_local! {
    static NAMED_CONST_FOLDS: std::cell::RefCell<
        std::collections::HashMap<String, OpKind>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
    /// Crates whose folds the table already holds, so a repeat can be
    /// told from the next crate of the same invocation.
    static NAMED_CONST_CRATES: std::cell::RefCell<
        std::collections::HashSet<String>,
    > = std::cell::RefCell::new(std::collections::HashSet::new());
}

pub(in crate::front::mir) struct NamedConstFoldsGuard {
    pub(in crate::front::mir) previous: std::collections::HashMap<String, OpKind>,
    pub(in crate::front::mir) previous_crates: std::collections::HashSet<String>,
}

impl NamedConstFoldsGuard {
    pub(in crate::front::mir) fn push(folds: std::collections::HashMap<String, OpKind>) -> Self {
        let previous =
            NAMED_CONST_FOLDS.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), folds));
        let previous_crates =
            NAMED_CONST_CRATES.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
        Self {
            previous,
            previous_crates,
        }
    }
}

impl Drop for NamedConstFoldsGuard {
    fn drop(&mut self) {
        NAMED_CONST_FOLDS.with(|slot| {
            *slot.borrow_mut() = std::mem::take(&mut self.previous);
        });
        NAMED_CONST_CRATES.with(|slot| {
            *slot.borrow_mut() = std::mem::take(&mut self.previous_crates);
        });
    }
}

/// Install `folds` as this invocation's defining-crate NamedConst table
/// until the returned guard drops.
///
/// The production frontend parses one crate at a time (holding every
/// tree with the merged program blew a 12 GB container), so a
/// cross-crate read sees only the caller's `Opaque` initializer.  This
/// table is the defining crate's already-folded value, seeded per
/// pipeline invocation the same way [`crate::local_crates`] is: thread-
/// local because one translate run stays on one thread.
pub(crate) fn push_named_const_folds(
    folds: std::collections::HashMap<String, OpKind>,
) -> impl Drop {
    NamedConstFoldsGuard::push(folds)
}

pub(in crate::front::mir) fn named_const_fold_for_path(path: &str) -> Option<OpKind> {
    NAMED_CONST_FOLDS.with(|slot| slot.borrow().get(path).cloned())
}

/// Add one crate's folds to the table, restarting it when `crate_name`
/// repeats.
///
/// One frontend invocation loads each artefact once, so a crate that is
/// harvested a second time means a new invocation began.  Without the
/// restart the table would keep the previous invocation's entries, and a
/// later partial build could fold an Opaque foreign declaration with a
/// value harvested from unrelated input.
pub(in crate::front::mir) fn merge_named_const_folds(
    crate_name: &str,
    folds: std::collections::HashMap<String, OpKind>,
) {
    let restart = NAMED_CONST_CRATES.with(|slot| {
        let mut seen = slot.borrow_mut();
        if seen.insert(crate_name.to_string()) {
            return false;
        }
        seen.clear();
        seen.insert(crate_name.to_string());
        true
    });
    NAMED_CONST_FOLDS.with(|slot| {
        let mut table = slot.borrow_mut();
        if restart {
            table.clear();
        }
        table.extend(folds);
    });
}

/// Fold every local `NamedConst` whose initializer this LLBC actually
/// carries (a literal assign, or a computed body [`const_eval_init_body`]
/// can evaluate).  Foreign decls are skipped: Charon left their init
/// `Opaque`, and the defining crate's harvest is the one that binds
/// the path.
pub(crate) fn harvest_named_const_folds(llbc: &Llbc) -> std::collections::HashMap<String, OpKind> {
    let mut out = std::collections::HashMap::new();
    for gd in llbc.iter_global_decls() {
        if !gd.item_meta.is_local {
            continue;
        }
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            continue;
        }
        let Some(op) = fold_named_const_on_llbc(llbc, gd.def_id)
            .or_else(|| {
                let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
                const_eval_init_body(llbc, &llbc.fn_by_id(init_id)?.unstructured()?)
            })
            .or_else(|| const_fn_int_array_op(llbc, gd.def_id))
        else {
            continue;
        };
        out.insert(gd.item_meta.name_path(), op);
    }
    out
}

/// An `AnonConst` whose init copies one `NamedConst` (promoted `&FLAT`).
/// The named const still has no address.
pub(in crate::front::mir) fn named_const_promoted_in_anon(llbc: &Llbc, def_id: u64) -> Option<u64> {
    let g = llbc.global_by_id(def_id)?;
    if g.rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("AnonConst")
    {
        return None;
    }
    let init_id = crate::front::llbc_hints::marker_init_fun_id(g)?;
    let u = llbc.fn_by_id(init_id)?.unstructured()?;
    let mut named = None;
    for bb in &u.body {
        for st in &bb.statements {
            let Ok(StmtKind::Assign(_, rvalue)) = st.stmt_kind_ref() else {
                continue;
            };
            let Rvalue::Use(op, _) = rvalue else {
                continue;
            };
            let (Operand::Copy(place) | Operand::Move(place)) = op else {
                continue;
            };
            let PlaceKind::Global { id, .. } = place.kind else {
                continue;
            };
            let inner = llbc.global_by_id(id)?;
            if inner
                .rest
                .get("global_kind")
                .and_then(serde_json::Value::as_str)
                != Some("NamedConst")
            {
                continue;
            }
            if named.is_some() {
                return None;
            }
            named = Some(id);
        }
    }
    named
}

/// An `AnonConst` / promoted const whose init is `RangeInclusive::new`
/// of two integer constants.  rustc lifts `(a..=b)` out of the function;
/// Charon records the init as a 3-BB Call+Ref-return+Unwind, which
/// `promoted_init_from_body` will not splice.  The bounds are the live
/// host values RPython would already have at translation time.
pub(in crate::front::mir) fn promoted_int_range_inclusive_bounds(
    llbc: &Llbc,
    def_id: u64,
) -> Option<(i64, i64)> {
    let g = llbc.global_by_id(def_id)?;
    if g.rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("AnonConst")
    {
        return None;
    }
    let leaf = g.item_meta.name_path().rsplit("::").next()?.to_string();
    if !majit_charon_reader::ullbc::is_const_initializer_leaf(&leaf) {
        return None;
    }
    let init_id = crate::front::llbc_hints::marker_init_fun_id(g)?;
    let u = llbc.fn_by_id(init_id)?.unstructured()?;
    let mut bounds = None;
    for bb in &u.body {
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            continue;
        };
        let Some(fd) = llbc.fn_by_id(*id) else {
            continue;
        };
        let Some((owner, leaf)) = impl_method_owner_for_fundecl(llbc, fd) else {
            continue;
        };
        if leaf != "new"
            || !owner
                .rsplit("::")
                .next()
                .is_some_and(|s| s == "RangeInclusive")
        {
            continue;
        }
        if call.args.len() != 2 {
            return None;
        }
        let lo = operand_as_signed_int(llbc, &call.args[0])?;
        let hi = operand_as_signed_int(llbc, &call.args[1])?;
        if bounds.is_some() {
            return None;
        }
        bounds = Some((lo, hi));
    }
    bounds
}

/// Integer constant carried by a `RangeInclusive::new` operand: a
/// literal, or a `NamedConst` global (`KIND_PRIM_CHAR` / `KIND_PRIM_COMPLEX`).
pub(in crate::front::mir) fn operand_as_signed_int(llbc: &Llbc, op: &Operand) -> Option<i64> {
    match op {
        Operand::Const(value) => match decode_constant(llbc, value).ok()? {
            DecodedConst::Int(n) => Some(n),
            DecodedConst::UInt(n) => i64::try_from(n).ok(),
            DecodedConst::Int128(n) => i64::try_from(n).ok(),
            DecodedConst::UInt128(n) => i64::try_from(n).ok(),
            DecodedConst::UniChar(code) => Some(i64::from(code)),
            _ => None,
        },
        Operand::Copy(place) | Operand::Move(place) => {
            let PlaceKind::Global { id, .. } = &place.kind else {
                return None;
            };
            global_signed_int(llbc, *id)
        }
    }
}

pub(in crate::front::mir) fn global_signed_int(llbc: &Llbc, def_id: u64) -> Option<i64> {
    match fold_named_const_on_llbc(llbc, def_id).or_else(|| {
        let g = llbc.global_by_id(def_id)?;
        let init_id = crate::front::llbc_hints::marker_init_fun_id(g)?;
        const_eval_init_body(llbc, &llbc.fn_by_id(init_id)?.unstructured()?)
    })? {
        OpKind::ConstInt(n) => Some(n),
        OpKind::ConstUInt(n) => i64::try_from(n).ok(),
        _ => None,
    }
}

/// Fold a `NamedConst` whose initializer is the trivial
/// `_0 = const <lit>; return` shape in `llbc`.  `None` when the
/// initializer is missing, `Opaque`, or any non-literal assign.
/// A `char` stays the integer code point for const-eval
/// (`const_lit_from_decoded`); value emit goes through
/// [`Lowering::emit_decoded_const`].
pub(in crate::front::mir) fn decoded_named_const_on_llbc(
    llbc: &Llbc,
    def_id: u64,
) -> Option<DecodedConst> {
    let gd = llbc.global_by_id(def_id)?;
    if gd
        .rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("NamedConst")
    {
        return None;
    }
    let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
    let init = llbc.fn_by_id(init_id)?;
    let body = init.unstructured()?;
    // The initializer must be exactly one literal assignment to the
    // return local (`_0 = const <lit>`); anything else (arithmetic,
    // calls, multiple assigns) is a computed const left to the
    // accessor path.
    let mut found: Option<serde_json::Value> = None;
    let forward_blocks = forward_reachable_mask(llbc, &body);
    for (bb_idx, blk) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for st in &blk.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind_ref() else {
                continue;
            };
            let is_local0 = matches!(place.kind, PlaceKind::Local(0));
            let lit = match &rvalue {
                Rvalue::Use(Operand::Const(c), _) => llbc.const_expr_literal(c),
                _ => None,
            };
            match lit {
                Some(l) if is_local0 => {
                    if found.is_some() {
                        return None;
                    }
                    found = Some(l);
                }
                // A non-literal write to _0 (computed const) — bail.
                _ if is_local0 => return None,
                _ => {}
            }
        }
    }
    decode_literal(&found?).ok()
}

pub(in crate::front::mir) fn fold_named_const_on_llbc(llbc: &Llbc, def_id: u64) -> Option<OpKind> {
    match decoded_named_const_on_llbc(llbc, def_id)? {
        DecodedConst::Int(n) => Some(OpKind::ConstInt(n)),
        DecodedConst::UInt(n) => Some(OpKind::ConstUInt(n)),
        DecodedConst::Int128(n) => Some(OpKind::ConstInt128(n)),
        DecodedConst::UInt128(n) => Some(OpKind::ConstUInt128(n)),
        DecodedConst::Bool(b) => Some(OpKind::ConstBool(b)),
        DecodedConst::Float(bits) => Some(OpKind::ConstFloat(bits)),
        DecodedConst::UniChar(code) => Some(OpKind::ConstInt(i64::from(code))),
        // A `const NAME: &str = "..."` global reads as a named-const
        // fold, not a static address; without this arm the read falls
        // through to a residual `FunctionPath` Call on the const path
        // (`ATTR_W_OBJ_WEAK` etc.) the registry cannot bind.  Emit the
        // same synthetic `__str_const` the `build_rvalue` const path
        // uses (result kind `Ref` — a `&str` literal is `Ptr(STR)`).
        DecodedConst::Str(s) => Some(OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), s],
                fun_decl_id: None,
                specialization: crate::parse::GraphSpecialization::Regular,
            },
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(None),
        }),
        _ => None,
    }
}

/// Discriminant of a fieldless enum variant aggregate.
///
/// `{"Adt": [type, variant_index, null]}` with no operands. The index is
/// the variant's position, not the bit `from_element` shifts by.
pub(in crate::front::mir) fn unit_enum_discriminant(
    llbc: &Llbc,
    kind: &serde_json::Value,
) -> Option<u64> {
    let adt = kind.get("Adt")?.as_array()?;
    let head = adt.first()?;
    let id = head
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| head.as_u64())?;
    let variant = usize::try_from(adt.get(1)?.as_u64()?).ok()?;
    let TypeDeclKind::Enum(variants) = &llbc.type_by_id(id)?.kind else {
        return None;
    };
    let variant = variants.get(variant)?;
    if !variant.fields.is_empty() {
        return None;
    }
    u64::try_from(variant.discriminant_i64()?).ok()
}

/// A shared borrow of a flag carrier is the carrier. A promoted `&FLAG`
/// copies the NamedConst and borrows it; the transparent wrapper has
/// already peeled to the integer word, so that word is the same carrier.
/// Other borrows stay residual so a pointer-identity initializer is not
/// folded as an integer.
pub(in crate::front::mir) fn const_eval_borrowed_flag(
    llbc: &Llbc,
    locals: &std::collections::HashMap<u64, ConstLit>,
    place: &Place,
) -> Option<ConstLit> {
    let lit = match &place.kind {
        PlaceKind::Local(n) => locals.get(n).copied()?,
        PlaceKind::Projection(inner, elem) if elem.label() == "Deref" => {
            const_eval_borrowed_flag(llbc, locals, inner)?
        }
        _ => return None,
    };
    match lit {
        v @ (ConstLit::EnumDisc(_) | ConstLit::BitflagBits(_)) => Some(v),
        // The peeled flag word lives in an ADT local (`CodeFlags`). A
        // primitive integer (or `&i64`) is a pointer-identity borrow.
        v @ (ConstLit::Int(_) | ConstLit::UInt(_)) if const_ty_peels_to_adt(llbc, &place.ty) => {
            Some(v)
        }
        _ => None,
    }
}

/// Peel `Ref` / `RawPtr` / hash-cons wrappers and report whether the
/// pointee is an ADT. Used to tell a transparent flag wrapper from a
/// primitive integer the same borrow evaluator must not fold.
pub(in crate::front::mir) fn const_ty_peels_to_adt(llbc: &Llbc, ty: &TyRef) -> bool {
    let mut v: &serde_json::Value = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(body) => body,
            None => return false,
        },
    };
    loop {
        let Some(obj) = v.as_object() else {
            return false;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            v = match llbc.dedup_body(id) {
                Some(body) => body,
                None => return false,
            };
            continue;
        }
        if let Some(arr) = obj.get("Value").and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            v = &arr[1];
            continue;
        }
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
        return obj.contains_key("Adt");
    }
}

/// Opaque `from_element` / `bits` from `bitflagset`.
///
/// `from_element` is `(1 as repr) << (element as u8)`. `bits` returns that
/// word. Both callees are `Opaque` in the extraction (the crate is outside
/// the LLBC set), so the initializer would otherwise lower to a nullary
/// call on the const's own path.
pub(in crate::front::mir) fn const_eval_opaque_bitflag_call(
    llbc: &Llbc,
    fun_id: u64,
    args: &[ConstLit],
) -> Option<ConstLit> {
    let fd = llbc.fn_by_id(fun_id)?;
    if fd.unstructured().is_some() {
        return None;
    }
    let leaf = fd.item_meta.name_path();
    let leaf = leaf.rsplit("::").next()?;
    match (leaf, args) {
        ("from_element", [ConstLit::EnumDisc(disc)]) => {
            let shift = u8::try_from(*disc).ok()?;
            let mask = 1u64.checked_shl(u32::from(shift))?;
            Some(ConstLit::BitflagBits(mask))
        }
        ("bits", [ConstLit::BitflagBits(mask)]) => Some(ConstLit::UInt(*mask)),
        _ => None,
    }
}

/// Evaluate a global's single-value init body over literal locals.
///
/// Upstream needs no analog: RPython constants are live host values at
/// translation time, so a computed constant like `-(1i64 << 62)`
/// reaches the annotator already folded (flowspace constant folding).
/// ULLBC instead carries the unevaluated init MIR — in the dev profile
/// a shift or negation drags an overflow-assert diamond along — so fold
/// it here: execute literal assigns, follow each `Assert` whose
/// condition evaluates to its `expected` value (rustc already
/// const-checked the initializer), fold a `core::mem::size_of` /
/// `align_of` Call terminator to the target layout width, and bail to
/// the residual `Call` lowering on any other shape.
pub(in crate::front::mir) fn const_eval_init_body(llbc: &Llbc, u: &Unstructured) -> Option<OpKind> {
    const_lit_to_op(const_eval_init_body_lit(llbc, u, 0)?)
}

/// Fold a `NamedConst` whose init is a const fn returning `[i64; N]` /
/// `[u8; N]` (or that array directly) into `__const_int_array`.
///
/// `rbigint.py` `ptwotable` is a prebuilt list read by `_x_mul`. The Rust
/// const is `PTWOTABLE = make_ptwotable()`, a loop Charon keeps as MIR, so
/// the literal-aggregate lane never sees the elements. Evaluating that
/// const fn is the translation-time value RPython already has.
pub(in crate::front::mir) fn const_fn_int_array_op(llbc: &Llbc, def_id: u64) -> Option<OpKind> {
    let gd = llbc.global_by_id(def_id)?;
    if gd
        .rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("NamedConst")
    {
        return None;
    }
    let init_id = crate::front::llbc_hints::marker_init_fun_id(gd)?;
    let items = eval_const_int_array(llbc, &llbc.fn_by_id(init_id)?.unstructured()?, 0)?;
    if items.is_empty() {
        return None;
    }
    let mut segments = Vec::with_capacity(items.len() + 1);
    segments.push("__const_int_array".to_string());
    for n in items {
        segments.push(n.to_string());
    }
    Some(OpKind::Call {
        target: CallTarget::function_path(segments),
        args: Vec::new(),
        result_ty: ValueType::Ref(None),
    })
}

#[derive(Clone)]
pub(in crate::front::mir) enum ArrVal {
    Lit(ConstLit),
    Arr(Vec<i64>),
    /// Shared borrow of the local that holds the array.
    RefOf(u64),
    /// Unsized slice of that same array (`slice::len`'s receiver).
    SliceOf(u64),
}

pub(in crate::front::mir) fn eval_const_int_array(
    llbc: &Llbc,
    u: &Unstructured,
    depth: usize,
) -> Option<Vec<i64>> {
    if depth > 4 {
        return None;
    }
    let mut locals: std::collections::HashMap<u64, ArrVal> = std::collections::HashMap::new();
    let mut bb = 0usize;
    for _ in 0..8192 {
        let block = u.body.get(bb)?;
        for stmt in &block.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_))
                | Ok(StmtKind::StorageDead(_))
                | Ok(StmtKind::PlaceMention(_))
                | Ok(StmtKind::Borrowck(_)) => {}
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if let Some((arr_local, index_op)) = array_index_place(place) {
                        let index = arr_index(llbc, &locals, &index_op)?;
                        let value = arr_as_i64(&eval_arr_rvalue(llbc, &locals, rvalue, depth)?)?;
                        let ArrVal::Arr(items) = locals.get_mut(&arr_local)? else {
                            return None;
                        };
                        let slot = items.get_mut(index)?;
                        *slot = value;
                        continue;
                    }
                    let PlaceKind::Local(dst) = place.kind else {
                        return None;
                    };
                    let value = eval_arr_rvalue(llbc, &locals, rvalue, depth)?;
                    locals.insert(dst, value);
                }
                _ => return None,
            }
        }
        match block.term_ref(llbc).ok()? {
            TermKind::Return => {
                return match locals.get(&0)? {
                    ArrVal::Arr(items) => Some(items.clone()),
                    _ => None,
                };
            }
            TermKind::Goto { target } => bb = *target as usize,
            TermKind::Assert { assert, target, .. } => {
                let ArrVal::Lit(ConstLit::Bool(cond)) =
                    eval_arr_operand(llbc, &locals, &assert.cond)?
                else {
                    return None;
                };
                if cond != assert.expected {
                    return None;
                }
                bb = *target as usize;
            }
            TermKind::Switch { discr, targets } => {
                let ArrVal::Lit(ConstLit::Bool(cond)) = eval_arr_operand(llbc, &locals, discr)?
                else {
                    return None;
                };
                let SwitchTargets::If(then_bb, else_bb) = targets else {
                    return None;
                };
                bb = *(if cond { then_bb } else { else_bb }) as usize;
            }
            TermKind::Call { call, target, .. } => {
                let PlaceKind::Local(dst) = call.dest.kind else {
                    return None;
                };
                locals.insert(dst, eval_arr_call(llbc, &locals, call, depth)?);
                bb = *target as usize;
            }
            TermKind::UnwindResume
            | TermKind::UnwindTerminate
            | TermKind::Abort(_)
            | TermKind::Panic { .. }
            | TermKind::UndefinedBehavior => return None,
            _ => return None,
        }
    }
    None
}

pub(in crate::front::mir) fn eval_arr_call(
    llbc: &Llbc,
    locals: &std::collections::HashMap<u64, ArrVal>,
    call: &CallPayload,
    depth: usize,
) -> Option<ArrVal> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let path = llbc.fn_by_id(*id)?.item_meta.name_path();
    if path.ends_with("::len") && call.args.len() == 1 {
        let ArrVal::SliceOf(arr_local) = eval_arr_operand(llbc, locals, &call.args[0])? else {
            return None;
        };
        let ArrVal::Arr(items) = locals.get(&arr_local)? else {
            return None;
        };
        return Some(ArrVal::Lit(ConstLit::UInt(items.len() as u64)));
    }
    if !call.args.is_empty() {
        return None;
    }
    let body = llbc.fn_by_id(*id)?.unstructured()?;
    Some(ArrVal::Arr(eval_const_int_array(llbc, &body, depth + 1)?))
}

pub(in crate::front::mir) fn eval_arr_rvalue(
    llbc: &Llbc,
    locals: &std::collections::HashMap<u64, ArrVal>,
    rvalue: &Rvalue,
    _depth: usize,
) -> Option<ArrVal> {
    match rvalue {
        Rvalue::Use(op, _) => eval_arr_operand(llbc, locals, op),
        Rvalue::Repeat(op, _, count, _) => {
            let fill = arr_as_i64(&eval_arr_operand(llbc, locals, op)?)?;
            let n = match decode_constant(llbc, count)
                .ok()
                .and_then(const_lit_from_decoded)
            {
                Some(ConstLit::UInt(n)) => n,
                Some(ConstLit::Int(n)) if n >= 0 => n as u64,
                _ => return None,
            };
            let n = usize::try_from(n).ok()?;
            if n == 0 || n > 4096 {
                return None;
            }
            Some(ArrVal::Arr(vec![fill; n]))
        }
        Rvalue::Ref { place, .. } => {
            let PlaceKind::Local(n) = place.kind else {
                return None;
            };
            if matches!(locals.get(&n), Some(ArrVal::Arr(_))) {
                Some(ArrVal::RefOf(n))
            } else {
                None
            }
        }
        Rvalue::UnaryOp(kind, op) => {
            if operator_name(kind) == Some("Cast")
                && kind
                    .as_object()
                    .and_then(|m| m.get("Cast"))
                    .and_then(|c| c.get("Unsize"))
                    .is_some()
            {
                let ArrVal::RefOf(n) = eval_arr_operand(llbc, locals, op)? else {
                    return None;
                };
                return Some(ArrVal::SliceOf(n));
            }
            let ArrVal::Lit(lit) = eval_arr_operand(llbc, locals, op)? else {
                return None;
            };
            Some(ArrVal::Lit(const_eval_unop(kind, lit)?))
        }
        Rvalue::BinaryOp(kind, lhs, rhs) => {
            let ArrVal::Lit(l) = eval_arr_operand(llbc, locals, lhs)? else {
                return None;
            };
            let ArrVal::Lit(r) = eval_arr_operand(llbc, locals, rhs)? else {
                return None;
            };
            Some(ArrVal::Lit(const_eval_binop(kind, l, r)?))
        }
        _ => None,
    }
}

pub(in crate::front::mir) fn eval_arr_operand(
    llbc: &Llbc,
    locals: &std::collections::HashMap<u64, ArrVal>,
    op: &Operand,
) -> Option<ArrVal> {
    match op {
        Operand::Const(value) => Some(ArrVal::Lit(
            decode_constant(llbc, value)
                .ok()
                .and_then(const_lit_from_decoded)?,
        )),
        Operand::Copy(place) | Operand::Move(place) => match &place.kind {
            PlaceKind::Local(n) => locals.get(n).cloned(),
            PlaceKind::Projection(inner, elem) => {
                let PlaceKind::Local(n) = inner.kind else {
                    return None;
                };
                let idx = const_tuple_field_index(elem)?;
                match locals.get(&n)? {
                    ArrVal::Lit(ConstLit::Checked(v, _)) if idx == 0 => {
                        Some(ArrVal::Lit(ConstLit::Int(*v)))
                    }
                    ArrVal::Lit(ConstLit::Checked(_, o)) if idx == 1 => {
                        Some(ArrVal::Lit(ConstLit::Bool(*o)))
                    }
                    ArrVal::Lit(ConstLit::CheckedUInt(v, _)) if idx == 0 => {
                        Some(ArrVal::Lit(ConstLit::UInt(*v)))
                    }
                    ArrVal::Lit(ConstLit::CheckedUInt(_, o)) if idx == 1 => {
                        Some(ArrVal::Lit(ConstLit::Bool(*o)))
                    }
                    _ => None,
                }
            }
            _ => None,
        },
    }
}

pub(in crate::front::mir) fn array_index_place(place: &Place) -> Option<(u64, Operand)> {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return None;
    };
    let PlaceKind::Local(arr_local) = inner.kind else {
        return None;
    };
    let ProjectionElem::Tagged(v) = elem else {
        return None;
    };
    let offset = v.as_object()?.get("Index")?.get("offset")?.clone();
    let op = serde_json::from_value(offset).ok()?;
    Some((arr_local, op))
}

pub(in crate::front::mir) fn arr_index(
    llbc: &Llbc,
    locals: &std::collections::HashMap<u64, ArrVal>,
    op: &Operand,
) -> Option<usize> {
    let ArrVal::Lit(lit) = eval_arr_operand(llbc, locals, op)? else {
        return None;
    };
    let n = match lit {
        ConstLit::UInt(n) => n,
        ConstLit::Int(n) if n >= 0 => n as u64,
        _ => return None,
    };
    usize::try_from(n).ok()
}

pub(in crate::front::mir) fn arr_as_i64(val: &ArrVal) -> Option<i64> {
    match val {
        ArrVal::Lit(ConstLit::Int(n)) => Some(*n),
        ArrVal::Lit(ConstLit::UInt(n)) => i64::try_from(*n).ok(),
        _ => None,
    }
}

pub(in crate::front::mir) fn projection_is_named_field(elem: &ProjectionElem) -> bool {
    matches!(
        elem,
        ProjectionElem::Tagged(payload)
            if payload.as_object().is_some_and(|map| map.contains_key("Field"))
    )
}

/// `local.field` or a nested field of it, without a deref in the chain.
pub(in crate::front::mir) fn place_is_field_projection_of_local(
    place: &Place,
    dest: usize,
) -> bool {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return false;
    };
    if !projection_is_named_field(elem) {
        return false;
    }
    match &inner.kind {
        PlaceKind::Local(i) => *i as usize == dest,
        PlaceKind::Projection(_, _) => place_is_field_projection_of_local(inner, dest),
        _ => false,
    }
}

pub(in crate::front::mir) enum ConstStructUse {
    None,
    FieldRead,
    Other,
}

pub(in crate::front::mir) fn operand_const_struct_use(
    op: &Operand,
    aliases: &std::collections::HashSet<usize>,
) -> ConstStructUse {
    let (Operand::Copy(place) | Operand::Move(place)) = op else {
        return ConstStructUse::None;
    };
    if aliases
        .iter()
        .any(|&d| place_is_field_projection_of_local(place, d))
    {
        ConstStructUse::FieldRead
    } else if aliases.iter().any(|&d| place_references_local(place, d)) {
        ConstStructUse::Other
    } else {
        ConstStructUse::None
    }
}

pub(in crate::front::mir) fn bump_const_struct_use(
    use_kind: ConstStructUse,
    fields: &mut usize,
    other: &mut usize,
) {
    match use_kind {
        ConstStructUse::FieldRead => *fields += 1,
        ConstStructUse::Other => *other += 1,
        ConstStructUse::None => {}
    }
}

pub(in crate::front::mir) fn place_is_const_struct_alias(
    place: &Place,
    aliases: &std::collections::HashSet<usize>,
) -> bool {
    aliases
        .iter()
        .any(|&d| place_is_field_projection_of_local(place, d) || place_references_local(place, d))
}

pub(in crate::front::mir) fn rvalue_is_const_struct_alias_copy(
    rvalue: &Rvalue,
    place: &Place,
    aliases: &std::collections::HashSet<usize>,
) -> bool {
    matches!(place.kind, PlaceKind::Local(j) if j != 0 && aliases.contains(&(j as usize)))
        && matches!(
            rvalue,
            Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _)
                if matches!(src.kind, PlaceKind::Local(i) if aliases.contains(&(i as usize)))
        )
}

pub(in crate::front::mir) fn scan_rvalue_const_struct(
    rvalue: &Rvalue,
    aliases: &std::collections::HashSet<usize>,
    fields: &mut usize,
    other: &mut usize,
) {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => {
            bump_const_struct_use(operand_const_struct_use(op, aliases), fields, other);
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            bump_const_struct_use(operand_const_struct_use(lhs, aliases), fields, other);
            bump_const_struct_use(operand_const_struct_use(rhs, aliases), fields, other);
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            if aliases.iter().any(|&d| place_references_local(place, d)) {
                *other += 1;
            }
        }
        Rvalue::Len(place) | Rvalue::Discriminant(place) => {
            if aliases
                .iter()
                .any(|&d| place_is_field_projection_of_local(place, d))
            {
                *fields += 1;
            } else if aliases.iter().any(|&d| place_references_local(place, d)) {
                *other += 1;
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                bump_const_struct_use(operand_const_struct_use(op, aliases), fields, other);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

pub(in crate::front::mir) fn local_assignment_counts(
    llbc: &Llbc,
    body: &Unstructured,
) -> Vec<usize> {
    let mut counts = vec![0usize; body.locals.locals.len()];
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && let PlaceKind::Local(i) = place.kind
            {
                if let Some(count) = counts.get_mut(i as usize) {
                    *count += 1;
                }
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc)
            && let PlaceKind::Local(i) = call.dest.kind
        {
            if let Some(count) = counts.get_mut(i as usize) {
                *count += 1;
            }
        }
    }
    counts
}

/// Whole-local copies of `dest` that are themselves single-assignment.
/// The return place is a whole-value consumer and is never an alias.
pub(in crate::front::mir) fn const_struct_copy_aliases(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;
    let counts = local_assignment_counts(llbc, body);
    let forward_blocks = forward_reachable_mask(llbc, body);
    let mut aliases = HashSet::from([dest]);
    let mut grew = true;
    while grew {
        grew = false;
        for (bb_idx, bb) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            for stmt in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                    continue;
                };
                let PlaceKind::Local(j) = place.kind else {
                    continue;
                };
                let j = j as usize;
                if j == 0 || aliases.contains(&j) || counts.get(j).copied() != Some(1) {
                    continue;
                }
                let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = rvalue else {
                    continue;
                };
                let PlaceKind::Local(i) = src.kind else {
                    continue;
                };
                if aliases.contains(&(i as usize)) {
                    aliases.insert(j);
                    grew = true;
                }
            }
        }
    }
    aliases
}

/// `fold_op_list` keeps any operation it cannot prove safe. Fold a
/// const-struct local only when every use is a field read, including
/// through single-assignment whole-local copies: a projected store,
/// address-taking, or whole-value consumer leaves the aggregate.
pub(in crate::front::mir) fn const_struct_uses_are_field_reads(
    llbc: &Llbc,
    body: &Unstructured,
    dest: usize,
) -> bool {
    if dest == 0 {
        return false;
    }
    let aliases = const_struct_copy_aliases(llbc, body, dest);
    let mut fields = 0usize;
    let mut other = 0usize;
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    if rvalue_is_const_struct_alias_copy(&rvalue, &place, &aliases) {
                        continue;
                    }
                    scan_rvalue_const_struct(&rvalue, &aliases, &mut fields, &mut other);
                    if matches!(place.kind, PlaceKind::Local(i) if aliases.contains(&(i as usize)))
                    {
                    } else if place_is_const_struct_alias(&place, &aliases) {
                        other += 1;
                    }
                }
                Ok(StmtKind::Assert(assert)) => bump_const_struct_use(
                    operand_const_struct_use(&assert.cond, &aliases),
                    &mut fields,
                    &mut other,
                ),
                _ => {}
            }
        }
        match bb.term_ref(llbc) {
            Ok(TermKind::Switch { discr, .. }) => bump_const_struct_use(
                operand_const_struct_use(&discr, &aliases),
                &mut fields,
                &mut other,
            ),
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op, _) = &call.func {
                    bump_const_struct_use(
                        operand_const_struct_use(op, &aliases),
                        &mut fields,
                        &mut other,
                    );
                }
                for arg in &call.args {
                    bump_const_struct_use(
                        operand_const_struct_use(arg, &aliases),
                        &mut fields,
                        &mut other,
                    );
                }
                if matches!(call.dest.kind, PlaceKind::Local(i) if aliases.contains(&(i as usize)))
                {
                } else if place_is_const_struct_alias(&call.dest, &aliases) {
                    other += 1;
                }
            }
            Ok(TermKind::Assert { assert, .. }) => bump_const_struct_use(
                operand_const_struct_use(&assert.cond, &aliases),
                &mut fields,
                &mut other,
            ),
            _ => {}
        }
    }
    other == 0
}

/// Integer fields of a `NamedConst` whose initializer builds one
/// multi-field aggregate of integer literals (`ArrayToken { base_size,
/// item_size, len_offset }`). A whole-struct read of that const is an
/// unregistered nullary call; each field is the literal the aggregate stored.
///
/// `None` keeps today's call. A variant payload, a one-field aggregate, or
/// any operand [`const_eval_init_body_lit`] cannot reduce refuses the const.
pub(in crate::front::mir) fn const_struct_field_lits(
    llbc: &Llbc,
    global_id: u64,
) -> Option<Vec<u64>> {
    let global = llbc.global_by_id(global_id)?;
    if global
        .rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("NamedConst")
    {
        return None;
    }
    let init_id = crate::front::llbc_hints::marker_init_fun_id(global)?;
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    eval_const_struct_init(llbc, &body)
}

/// A struct aggregate, not an array or a tuple. An array const such as
/// `RBigInt::pow`'s `JMAPPING` is a value the edge has to carry; only a
/// nominal struct's fields are substituted one word at a time.
pub(in crate::front::mir) fn aggregate_is_nominal_adt(kind: &serde_json::Value) -> bool {
    let Some(adt) = kind.get("Adt").and_then(serde_json::Value::as_array) else {
        return false;
    };
    if adt.get(1).is_some_and(|variant| !variant.is_null()) {
        return false;
    }
    let Some(head) = adt.first() else {
        return false;
    };
    if head.as_u64().is_some() {
        return true;
    }
    match head.get("builtin") {
        None => head.get("id").is_some(),
        Some(builtin) => builtin.is_null(),
    }
}

pub(in crate::front::mir) fn eval_const_struct_init(
    llbc: &Llbc,
    u: &Unstructured,
) -> Option<Vec<u64>> {
    let n = u.locals.locals.len();
    let mut lits = vec![None; n];
    let mut structs: Vec<Option<Vec<u64>>> = vec![None; n];
    let mut bb = 0usize;
    for _ in 0..64 {
        let block = u.body.get(bb)?;
        for stmt in &block.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_))
                | Ok(StmtKind::StorageDead(_))
                | Ok(StmtKind::PlaceMention(_))
                | Ok(StmtKind::Borrowck(_)) => {}
                Ok(StmtKind::Assign(place, rvalue)) => {
                    let PlaceKind::Local(dst) = place.kind else {
                        return None;
                    };
                    let dst = usize::try_from(dst).ok()?;
                    if dst >= n {
                        return None;
                    }
                    match rvalue {
                        Rvalue::Use(op, _) => {
                            lits[dst] = Some(const_eval_struct_use_lit(llbc, &lits, op)?);
                            structs[dst] = None;
                        }
                        Rvalue::Aggregate(kind, operands)
                            if operands.len() >= 2 && aggregate_is_nominal_adt(kind) =>
                        {
                            let mut fields = Vec::with_capacity(operands.len());
                            for op in operands {
                                fields
                                    .push(lit_as_u64(const_eval_struct_use_lit(llbc, &lits, op)?)?);
                            }
                            structs[dst] = Some(fields);
                            lits[dst] = None;
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
        match block.term_ref(llbc).ok()? {
            TermKind::Return => {
                let fields = structs.first()?.clone()?;
                return (fields.len() >= 2).then_some(fields);
            }
            TermKind::Goto { target } => bb = *target as usize,
            TermKind::Call { call, target, .. } => {
                let PlaceKind::Local(dst) = call.dest.kind else {
                    return None;
                };
                let dst = usize::try_from(dst).ok()?;
                if dst >= n {
                    return None;
                }
                lits[dst] = Some(const_eval_size_align_call(llbc, call)?);
                structs[dst] = None;
                bb = *target as usize;
            }
            _ => return None,
        }
    }
    None
}

pub(in crate::front::mir) fn const_eval_struct_use_lit(
    llbc: &Llbc,
    lits: &[Option<ConstLit>],
    op: &Operand,
) -> Option<ConstLit> {
    match op {
        Operand::Const(value) => decode_const_lit(llbc, value),
        Operand::Copy(place) | Operand::Move(place) => match &place.kind {
            PlaceKind::Local(n) => lits.get(*n as usize).copied().flatten(),
            PlaceKind::Global { id, .. } => {
                let global = llbc.global_by_id(*id)?;
                let init_id = crate::front::llbc_hints::marker_init_fun_id(global)?;
                let body = llbc.fn_by_id(init_id)?.unstructured()?;
                const_eval_init_body_lit(llbc, &body, 0)
            }
            _ => None,
        },
    }
}

pub(in crate::front::mir) fn lit_as_u64(lit: ConstLit) -> Option<u64> {
    match lit {
        ConstLit::UInt(n) | ConstLit::EnumDisc(n) | ConstLit::BitflagBits(n) => Some(n),
        ConstLit::Int(n) if n >= 0 => u64::try_from(n).ok(),
        ConstLit::Bool(value) => Some(u64::from(value)),
        _ => None,
    }
}

pub(in crate::front::mir) fn const_eval_init_body_lit(
    llbc: &Llbc,
    u: &Unstructured,
    depth: usize,
) -> Option<ConstLit> {
    const_eval_init_body_with_locals(llbc, u, depth, std::collections::HashMap::new())
}

pub(in crate::front::mir) fn const_eval_init_body_with_locals(
    llbc: &Llbc,
    u: &Unstructured,
    depth: usize,
    mut locals: std::collections::HashMap<u64, ConstLit>,
) -> Option<ConstLit> {
    if depth > 32 {
        return None;
    }
    let eval_operand =
        |locals: &std::collections::HashMap<u64, ConstLit>, op: &Operand| -> Option<ConstLit> {
            match op {
                Operand::Copy(place) | Operand::Move(place) => match &place.kind {
                    PlaceKind::Local(n) => locals.get(n).copied(),
                    // RPython hands already-folded module constants to the
                    // flowspace.  Charon instead leaves a `Place::Global`
                    // inside computed const MIR (`MASK = (1 << SHIFT) - 1`).
                    // Evaluate an immutable NamedConst / AnonConst initializer
                    // recursively so the resulting flow graph still carries
                    // one literal Constant rather than a synthetic accessor
                    // call. AnonConst is rustc's promoted use of a const
                    // (`PathElem::Builtin(PromotedConst)`).
                    PlaceKind::Global { id, .. } => {
                        let global = llbc.global_by_id(*id)?;
                        // `AnonConst` is rustc's promoted `&FLAG` for a
                        // NamedConst used by reference (`FLAG.bits()`).
                        if !global.is_const_value() {
                            return None;
                        }
                        let init_id = crate::front::llbc_hints::marker_init_fun_id(global)?;
                        let init = llbc.fn_by_id(init_id)?;
                        if let Some(body) = init.unstructured() {
                            const_eval_init_body_lit(llbc, &body, depth + 1)
                        } else {
                            // Foreign primitive associated constants
                            // (`i64::MAX`, etc.) have an opaque stdlib
                            // initializer in Charon. RPython's flowspace sees
                            // their already-evaluated host value, so recover
                            // that literal from the primitive type and
                            // associated-const leaf instead of materialising
                            // a non-existent accessor call.
                            const_eval_core_num_associated_const(llbc, *id)
                        }
                    }
                    // `checked_pair.0` / `.1` — the only projection an
                    // init body needs (`(1i64 << 62) - 1` reads its
                    // `SubChecked` result back field-wise).
                    PlaceKind::Projection(inner, elem) => {
                        if elem.label() == "Deref" {
                            return const_eval_borrowed_flag(llbc, locals, inner);
                        }
                        let PlaceKind::Local(n) = inner.kind else {
                            return None;
                        };
                        match (locals.get(&n).copied()?, const_tuple_field_index(elem)?) {
                            (ConstLit::Checked(v, _), 0) => Some(ConstLit::Int(v)),
                            (ConstLit::Checked(_, o), 1) => Some(ConstLit::Bool(o)),
                            (ConstLit::CheckedUInt(v, _), 0) => Some(ConstLit::UInt(v)),
                            (ConstLit::CheckedUInt(_, o), 1) => Some(ConstLit::Bool(o)),
                            _ => None,
                        }
                    }
                    _ => None,
                },
                Operand::Const(value) => decode_const_lit(llbc, value),
            }
        };
    let mut bb = 0usize;
    // A genuine init body is a short assert-diamond chain; the step cap
    // only guards against cycles.
    for _ in 0..64 {
        let block = u.body.get(bb)?;
        for stmt in &block.statements {
            match stmt.stmt_kind_ref() {
                Ok(StmtKind::StorageLive(_))
                | Ok(StmtKind::StorageDead(_))
                | Ok(StmtKind::PlaceMention(_))
                | Ok(StmtKind::Borrowck(_)) => {}
                Ok(StmtKind::Assign(place, rvalue)) => {
                    let PlaceKind::Local(dst) = place.kind else {
                        return None;
                    };
                    let value = match &rvalue {
                        Rvalue::Use(op, _) => eval_operand(&locals, op)?,
                        Rvalue::UnaryOp(kind, op) => {
                            const_eval_unop(kind, eval_operand(&locals, op)?)?
                        }
                        Rvalue::BinaryOp(kind, lhs, rhs) => const_eval_binop(
                            kind,
                            eval_operand(&locals, lhs)?,
                            eval_operand(&locals, rhs)?,
                        )?,
                        // A `repr(transparent)` struct is its one sized
                        // field. A fieldless variant is its discriminant:
                        // `bitflagset`'s `from_element` shifts by that value
                        // (`CoFastFlag::Cell` is 6, so the mask is `1 << 6`).
                        Rvalue::Aggregate(kind, operands) => {
                            if operands.is_empty()
                                && let Some(disc) = unit_enum_discriminant(llbc, kind)
                            {
                                ConstLit::EnumDisc(disc)
                            } else {
                                let index = const_transparent_aggregate_field(llbc, kind)?;
                                eval_operand(&locals, operands.get(index)?)?
                            }
                        }
                        // Shared borrow of the flag word while `.bits()`
                        // reborrows it. Only the flag carriers propagate;
                        // any other borrow still refuses the initializer.
                        Rvalue::Ref { place, .. } => {
                            const_eval_borrowed_flag(llbc, &locals, place)?
                        }
                        _ => return None,
                    };
                    // rustc computes each assignment at the destination's
                    // width; re-narrow so a `Shl` that shifted bits out does
                    // not bake a too-wide constant.
                    locals.insert(
                        dst,
                        const_narrow_to_target(const_literal_ty(llbc, &place.ty), value),
                    );
                }
                _ => return None,
            }
        }
        match block.term_ref(llbc).ok()? {
            TermKind::Return => return locals.get(&0).copied(),
            TermKind::Goto { target } => bb = *target as usize,
            TermKind::Assert { assert, target, .. } => {
                let ConstLit::Bool(cond) = eval_operand(&locals, &assert.cond)? else {
                    return None;
                };
                if cond != assert.expected {
                    return None;
                }
                bb = *target as usize;
            }
            TermKind::Call { call, target, .. } => {
                let PlaceKind::Local(dst) = call.dest.kind else {
                    return None;
                };
                let value = match const_eval_size_align_call(llbc, call) {
                    Some(value) => value,
                    None => {
                        let args = call
                            .args
                            .iter()
                            .map(|arg| eval_operand(&locals, arg))
                            .collect::<Option<Vec<_>>>()?;
                        const_eval_const_fn_call(llbc, call, &args, depth)?
                    }
                };
                locals.insert(dst, value);
                bb = *target as usize;
            }
            _ => return None,
        }
    }
    None
}

/// Evaluate a call a const initializer makes to a `const fn` whose body
/// this LLBC carries, over literal arguments.
///
/// Only a `const fn` can be called from a const context, so the callee is
/// one rustc already evaluated to build the constant; running its body over
/// the same literals gives the host value RPython's flowspace would receive
/// as a prebuilt Constant.  A generic callee, a callee without a body, and
/// any body shape [`const_eval_init_body_with_locals`] cannot evaluate stay
/// unharvested.  Opaque `bitflagset` `from_element` / `bits` fold without a
/// body.
pub(in crate::front::mir) fn const_eval_const_fn_call(
    llbc: &Llbc,
    call: &CallPayload,
    args: &[ConstLit],
    depth: usize,
) -> Option<ConstLit> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    if let Some(lit) = const_eval_opaque_bitflag_call(llbc, *id, args) {
        return Some(const_narrow_to_target(
            const_literal_ty(llbc, &call.dest.ty),
            lit,
        ));
    }
    if reg
        .generics
        .get("types")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|types| !types.is_empty())
    {
        return None;
    }
    let body = llbc.fn_by_id(*id)?.unstructured()?;
    if body.locals.arg_count as usize != args.len() {
        return None;
    }
    let callee_locals = args
        .iter()
        .enumerate()
        .map(|(index, value)| (index as u64 + 1, *value))
        .collect();
    let value = const_eval_init_body_with_locals(llbc, &body, depth + 1, callee_locals)?;
    Some(const_narrow_to_target(
        const_literal_ty(llbc, &call.dest.ty),
        value,
    ))
}

/// The operand index that is the value of a `repr(transparent)` struct
/// aggregate: its one sized field.  `None` for any other aggregate.
pub(in crate::front::mir) fn const_transparent_aggregate_field(
    llbc: &Llbc,
    kind: &serde_json::Value,
) -> Option<usize> {
    let adt = kind.as_object()?.get("Adt")?.as_array()?;
    let head = adt.first()?;
    let type_id = match head.as_u64() {
        Some(id) => id,
        None => type_decl_ref_adt_id(head.as_object()?)?,
    };
    if adt.get(1).is_some_and(|variant| !variant.is_null()) {
        return None;
    }
    let decl = llbc.type_by_id(type_id)?;
    transparent_nonzst_field(decl, llbc).map(|(index, _)| index)
}

/// Fold a const-init `Call` terminator only when it is nullary
/// `core::mem::size_of` / `align_of`. Any other callee stays unharvested:
/// that refusal is the fail-closed default
/// [`const_eval_init_body_lit`] used before this exemption.
pub(in crate::front::mir) fn const_eval_size_align_call(
    llbc: &Llbc,
    call: &CallPayload,
) -> Option<ConstLit> {
    if !call.args.is_empty() {
        return None;
    }
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    let want_align = match llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
        "core::mem::size_of" => false,
        "core::mem::align_of" => true,
        _ => return None,
    };
    let ty = reg.generics.get("types")?.as_array()?.first()?;
    let bytes = size_align_of_tyexpr(llbc, want_align, ty)?;
    Some(const_narrow_to_target(
        const_literal_ty(llbc, &call.dest.ty),
        ConstLit::UInt(bytes),
    ))
}

/// Byte size / alignment of a `size_of` / `align_of` type argument.
///
/// Primitive widths (including `usize` / `isize`) come from the
/// translation target, not from the host `size_of::<usize>()` and not
/// from the artefact's extraction-host `target_pointer_size`. Harvest
/// runs inside the translation pipeline
/// (`discover_foldable_const_lits` via
/// `build_semantic_program_via_active_frontend` and the prepass), once
/// per TARGET. `layout::target_word_size` reads
/// `CARGO_CFG_TARGET_POINTER_WIDTH` — Cargo sets that to the crate
/// target even in a build.rs running on a wider host — and only falls
/// back to the host width when that cfg is unset (the host *is* the
/// target). A wasm32 prepass therefore harvests 4, not the host's 8.
///
/// ADT arguments reuse the same Charon `layout_for_target` lane
/// [`Lowering::fold_size_const_global`] uses. Anything else (tuple,
/// slice, an unresolved layout) stays unharvested.
pub(in crate::front::mir) fn size_align_of_tyexpr(
    llbc: &Llbc,
    want_align: bool,
    ty: &serde_json::Value,
) -> Option<u64> {
    let body = tyexpr_body(llbc, ty)?;
    if let Some(lit) = body.get("Scalar") {
        return primitive_size_align(want_align, lit);
    }
    // `size_align_const_from_tyexpr`: a thin `RawPtr` / `Ref` to a sized
    // pointee is one address word. A slice or trait object stays unharvested.
    let pointee = match (body.get("RawPtr"), body.get("Ref")) {
        (Some(raw), _) => raw.get(0),
        (_, Some(reference)) => reference.get(1),
        _ => None,
    };
    if let Some(pointee) = pointee {
        let pointee = tyexpr_body(llbc, pointee)?;
        // Charon spells a nominal ADT as `"builtin": null`. A missing key
        // and a null value are the same sized pointee; a present builtin
        // (`str`, a slice) is not one address word.
        let sized = pointee.get("Scalar").is_some() || nominal_adt_pointee(pointee);
        return sized.then(|| crate::layout::target_word_size() as u64);
    }
    let adt = inline_adt_def_id(body).or_else(|| resolve_tyexpr_to_adt_def_id_free(llbc, ty))?;
    let target = std::env::var("TARGET").unwrap_or_default();
    let layout = llbc.type_by_id(adt)?.layout_for_target(llbc, &target)?;
    if want_align {
        layout.align
    } else {
        layout.size
    }
}

/// `true` when `pointee` is a nominal ADT. Charon writes `"builtin": null`
/// for that shape; a present builtin is `str`, a slice, or `Box`.
pub(in crate::front::mir) fn nominal_adt_pointee(pointee: &serde_json::Value) -> bool {
    let Some(adt) = pointee.get("Adt") else {
        return false;
    };
    match adt.get("builtin") {
        None => true,
        Some(builtin) => builtin.is_null(),
    }
}

pub(in crate::front::mir) fn tyexpr_body<'a>(
    llbc: &'a Llbc,
    ty: &'a serde_json::Value,
) -> Option<&'a serde_json::Value> {
    if let Some(arr) = ty.get("Value").and_then(serde_json::Value::as_array) {
        return arr.get(1);
    }
    if let Some(id) = ty.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_body(id);
    }
    Some(ty)
}

pub(in crate::front::mir) fn primitive_size_align(
    want_align: bool,
    lit: &serde_json::Value,
) -> Option<u64> {
    let word = crate::layout::target_word_size() as u64;
    if matches!(lit.as_str(), Some("Bool") | Some("Char")) {
        return Some(if lit.as_str() == Some("Char") { 4 } else { 1 });
    }
    let int_width = lit.get("Integer").and_then(|int| {
        int.get("Unsigned")
            .or_else(|| int.get("Signed"))
            .and_then(serde_json::Value::as_str)
    });
    let (size, align) = if let Some(width) = int_width {
        match width {
            "U8" | "I8" => (1, 1),
            "U16" | "I16" => (2, 2),
            "U32" | "I32" => (4, 4),
            "U64" | "I64" => (8, 8),
            "U128" | "I128" => (16, 16),
            "Usize" | "Isize" => (word, word),
            _ => return None,
        }
    } else if let Some(width) = lit.get("Float").and_then(serde_json::Value::as_str) {
        match width {
            "F32" => (4, 4),
            "F64" => (8, 8),
            _ => return None,
        }
    } else if lit.get("Bool").is_some() {
        (1, 1)
    } else {
        return None;
    };
    Some(if want_align { align } else { size })
}

pub(in crate::front::mir) fn decode_const_lit(
    llbc: &Llbc,
    value: &serde_json::Value,
) -> Option<ConstLit> {
    let literal = llbc.const_expr_literal(value)?;
    if let Some(scalar) = literal.get("Scalar").and_then(serde_json::Value::as_object)
        && let Some((kind, payload)) = scalar.into_iter().next()
    {
        let pair = payload.as_array()?;
        let text = pair.get(1)?.as_str()?;
        return match kind.as_str() {
            "Signed" | "Isize" => text.parse::<i64>().ok().map(ConstLit::Int),
            "Unsigned" | "Usize" => text.parse::<u64>().ok().map(ConstLit::UInt),
            _ => None,
        };
    }
    if let Some(value) = literal.get("Bool").and_then(serde_json::Value::as_bool) {
        return Some(ConstLit::Bool(value));
    }
    if let Some(float) = literal.get("Float")
        && let Some(value) = float
            .get("value")
            .and_then(serde_json::Value::as_str)
            .and_then(|text| text.parse::<f64>().ok())
    {
        // See [`decode_literal_constant`]: the width rides beside the
        // value, and an `f32` keeps its own carrier.
        return Some(
            if float.get("ty").and_then(serde_json::Value::as_str) == Some("F32") {
                ConstLit::SingleFloat((value as f32).to_bits())
            } else {
                ConstLit::Float(value.to_bits())
            },
        );
    }
    None
}

/// Evaluate the integer `MIN` / `MAX` associated constants declared by
/// `core::num`.
///
/// Charon records foreign standard-library bodies as `Opaque`, including the
/// initializer behind `i64::MAX`. RPython's flowspace receives module
/// constants as live host values, so an expression such as pyre's
/// `MASK: Signed = i64::MAX` must likewise become one literal Constant in the
/// translated graph. Restrict this recovery to the exact `core::num::<Impl>`
/// owner, the `MIN` / `MAX` leaves, and a primitive fixed-width integer type;
/// user constants and target-width `isize` / `usize` remain residual.
pub(in crate::front::mir) fn const_eval_core_num_associated_const(
    llbc: &Llbc,
    def_id: u64,
) -> Option<ConstLit> {
    let global = llbc.global_by_id(def_id)?;
    if global
        .rest
        .get("global_kind")
        .and_then(serde_json::Value::as_str)
        != Some("NamedConst")
    {
        return None;
    }
    let path = global.item_meta.name_path();
    let leaf = path.rsplit("::").next()?;
    if !path.starts_with("core::num::<Impl>::") || !matches!(leaf, "MIN" | "MAX") {
        return None;
    }

    let tyref = global.rest.get("ty")?;
    let ty = if let Some(id) = tyref
        .get("Deduplicated")
        .and_then(serde_json::Value::as_u64)
    {
        llbc.dedup_body(id)?
    } else if let Some(pair) = tyref.get("Value").and_then(serde_json::Value::as_array) {
        pair.get(1)?
    } else {
        tyref
    };
    let literal = ty.get("Scalar")?.as_object()?;
    let literal = literal
        .get("Integer")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(literal);

    if let Some(width) = literal.get("Signed").and_then(serde_json::Value::as_str) {
        let (min, max) = match width {
            "I8" => (i8::MIN as i64, i8::MAX as i64),
            "I16" => (i16::MIN as i64, i16::MAX as i64),
            "I32" => (i32::MIN as i64, i32::MAX as i64),
            "I64" => (i64::MIN, i64::MAX),
            _ => return None,
        };
        return Some(ConstLit::Int(if leaf == "MIN" { min } else { max }));
    }
    if let Some(width) = literal.get("Unsigned").and_then(serde_json::Value::as_str) {
        let max = match width {
            "U8" => u8::MAX as u64,
            "U16" => u16::MAX as u64,
            "U32" => u32::MAX as u64,
            "U64" => u64::MAX,
            _ => return None,
        };
        return Some(ConstLit::UInt(if leaf == "MIN" { 0 } else { max }));
    }
    None
}

pub(in crate::front::mir) fn const_lit_to_op(value: ConstLit) -> Option<OpKind> {
    match value {
        ConstLit::Int(n) => Some(OpKind::ConstInt(n)),
        ConstLit::UInt(n) => Some(OpKind::ConstUInt(n)),
        ConstLit::Bool(b) => Some(OpKind::ConstBool(b)),
        ConstLit::Float(bits) => Some(OpKind::ConstFloat(bits)),
        ConstLit::SingleFloat(bits) => Some(OpKind::ConstSingleFloat(bits)),
        ConstLit::EnumDisc(n) => i64::try_from(n).ok().map(OpKind::ConstInt),
        ConstLit::Checked(..) | ConstLit::CheckedUInt(..) | ConstLit::BitflagBits(_) => None,
    }
}

/// Charon serializes an operator as an atom string (`"Add"`) or a
/// one-key object whose key is the operator and whose value is the
/// overflow mode (`{"Shl": "Wrap"}`) or cast payload.
pub(in crate::front::mir) fn operator_name(kind: &serde_json::Value) -> Option<&str> {
    if let Some(s) = kind.as_str() {
        return Some(s);
    }
    let obj = kind.as_object()?;
    if obj.len() == 1 {
        obj.keys().next().map(String::as_str)
    } else {
        None
    }
}

/// The overflow mode string of an operator object (`{"Add": "Wrap"}` →
/// `Some("Wrap")`). Atom operators (`"Add"`) and payloads that are not a
/// bare string (cast targets, multi-key objects) yield `None`.
pub(in crate::front::mir) fn operator_mode(kind: &serde_json::Value) -> Option<&str> {
    let obj = kind.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    obj.values().next()?.as_str()
}

/// Decode a variant-less `Field` projection — `{"Field": [null, idx]}` —
/// to its element index.  A downcast (`[variant, idx]`) is an enum field,
/// never a tuple element.
pub(in crate::front::mir) fn const_tuple_field_index(elem: &ProjectionElem) -> Option<u64> {
    let ProjectionElem::Tagged(v) = elem else {
        return None;
    };
    let payload = v.as_object()?.get("Field")?.as_array()?;
    if payload.len() != 2 || !payload[0].is_null() {
        return None;
    }
    payload[1].as_u64()
}

pub(in crate::front::mir) fn const_lit_from_decoded(value: DecodedConst) -> Option<ConstLit> {
    match value {
        DecodedConst::Int(n) => Some(ConstLit::Int(n)),
        DecodedConst::UInt(n) => Some(ConstLit::UInt(n)),
        DecodedConst::Bool(b) => Some(ConstLit::Bool(b)),
        DecodedConst::Float(bits) => Some(ConstLit::Float(bits)),
        DecodedConst::SingleFloat(bits) => Some(ConstLit::SingleFloat(bits)),
        DecodedConst::UniChar(code) => Some(ConstLit::Int(i64::from(code))),
        _ => None,
    }
}

pub(in crate::front::mir) fn const_eval_unop(
    kind: &serde_json::Value,
    v: ConstLit,
) -> Option<ConstLit> {
    let name = operator_name(kind)?;
    // A scalar int→int cast reads the operand's bits, and the two integer
    // carriers hold the same bits, so one evaluation serves both spellings
    // (`b'v' as i64` starts from an unsigned literal).  The destination's
    // own signedness is restored by `const_narrow_to_target`.
    let cast_bits = match v {
        ConstLit::Int(n) => Some(n),
        ConstLit::UInt(n) => Some(n as i64),
        _ => None,
    };
    if let ("Cast", Some(bits)) = (name, cast_bits) {
        // Truncate / re-extend to the dest width so `expr as uN` folds
        // exactly. `CastKind::Scalar` is dest-only (`ScalarTy`); the
        // source type is the operand's (`Remove the target types of casts`).
        let dest = kind.as_object()?.get("Cast")?.get("Scalar")?;
        return const_cast_int(bits, dest).map(ConstLit::Int);
    }
    match (name, v) {
        ("Neg", ConstLit::Int(n)) => n.checked_neg().map(ConstLit::Int),
        ("Not", ConstLit::Bool(b)) => Some(ConstLit::Bool(!b)),
        ("Not", ConstLit::Int(n)) => Some(ConstLit::Int(!n)),
        ("Not", ConstLit::UInt(n)) => Some(ConstLit::UInt(!n)),
        _ => None,
    }
}

pub(in crate::front::mir) fn const_cast_int(n: i64, target: &serde_json::Value) -> Option<i64> {
    const_cast_int_with_word(n, target, crate::layout::target_word_size())
}

/// Truncate / re-extend to `target` (`lltype.py` `cast_primitive`).
/// `Isize`/`Usize` width is the target word, not a hard-coded 64.
pub(in crate::front::mir) fn const_cast_int_with_word(
    n: i64,
    target: &serde_json::Value,
    word: usize,
) -> Option<i64> {
    let int = target
        .get("Integer")
        .and_then(serde_json::Value::as_object)?;
    let (signed, width) = if let Some(w) = int.get("Signed") {
        (true, w.as_str()?)
    } else {
        let w = int.get("Unsigned")?;
        (false, w.as_str()?)
    };
    let bits: u32 = match width {
        "I8" | "U8" => 8,
        "I16" | "U16" => 16,
        "I32" | "U32" => 32,
        "I64" | "U64" => 64,
        "Isize" | "Usize" => (word * 8) as u32,
        // 128-bit targets exceed the i64 carrier.
        _ => return None,
    };
    Some(if bits == 64 {
        n
    } else if signed {
        let shift = 64 - bits;
        (n << shift) >> shift
    } else {
        n & ((1i64 << bits) - 1)
    })
}

/// Resolve a Charon type reference to its `Scalar` payload, following
/// `Deduplicated` / `Value` the same way [`Lowering::tyref_literal_int_atom`]
/// does.
pub(in crate::front::mir) fn const_literal_ty<'t>(
    llbc: &'t Llbc,
    tyref: &'t TyRef,
) -> Option<&'t serde_json::Value> {
    let value = match tyref {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    value.as_object()?.get("Scalar")
}

/// Re-narrow a folded literal to the width and signedness of the local it
/// is assigned to.
///
/// The `ConstLit` carrier is 64-bit, so an operation whose true result is
/// computed at a narrower width can come out too wide — `Shl` is the live
/// case, because Rust does *not* count shifted-out bits as an overflow
/// (`200u8 << 1` is a legal const equal to 144, while the u64 carrier says
/// 400).  rustc truncates to the destination type, so doing the same here
/// keeps the fold bit-exact instead of baking a too-wide constant.  This is
/// the same width reasoning the `Add`/`Sub`/`Mul` `Wrap` arms use to decline
/// outright; with the destination type in hand the value can be corrected
/// rather than residualized.
///
/// Only integer destinations narrow: a `Bool`/`Float` destination, a
/// non-literal one (the tuple a `*Checked` pair is assigned to), or a
/// 128-bit width leaves the value untouched.
pub(in crate::front::mir) fn const_narrow_to_target(
    target: Option<&serde_json::Value>,
    v: ConstLit,
) -> ConstLit {
    let Some(target) = target else {
        return v;
    };
    let raw = match v {
        ConstLit::Int(n) => n,
        ConstLit::UInt(n) => n as i64,
        _ => return v,
    };
    let Some(narrowed) = const_cast_int(raw, target) else {
        return v;
    };
    if target
        .get("Integer")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|int| int.contains_key("Unsigned"))
    {
        ConstLit::UInt(narrowed as u64)
    } else {
        ConstLit::Int(narrowed)
    }
}

pub(in crate::front::mir) fn const_eval_binop(
    kind: &serde_json::Value,
    lhs: ConstLit,
    rhs: ConstLit,
) -> Option<ConstLit> {
    let name = operator_name(kind)?;
    // A mixed-signedness pair is not an ambiguous fold for every operator.
    // Charon emits one for two shapes that are exactly determined once the
    // signed side is non-negative, because then both carriers agree on the
    // mathematical value:
    //   * a shift, whose RHS is a bit *count* and not a value in the LHS's
    //     domain — `1u32 << 31` records `Unsigned(U32, 1)` against
    //     `Signed(I32, 31)` because an unsuffixed shift amount defaults to
    //     `i32`;
    //   * a comparison, where a `Cast` upstream dropped one side's
    //     signedness into the other carrier (`31i32 as u32 < 32u32`, the
    //     shift-overflow assert Charon plants ahead of the shift itself).
    // Arithmetic is deliberately excluded: its result *type* is what the
    // mixed pair leaves undetermined, and Rust needs an explicit cast to
    // write it, so it does not occur in a real initializer.
    let mixed = match (lhs, rhs) {
        (ConstLit::UInt(a), ConstLit::Int(b)) if b >= 0 => Some((a, b as u64)),
        (ConstLit::Int(a), ConstLit::UInt(b)) if a >= 0 => Some((a as u64, b)),
        _ => None,
    };
    if let Some((a, b)) = mixed {
        return match name {
            "Shl" => u32::try_from(b)
                .ok()
                .and_then(|shift| a.checked_shl(shift))
                .map(ConstLit::UInt),
            "Shr" => u32::try_from(b)
                .ok()
                .and_then(|shift| a.checked_shr(shift))
                .map(ConstLit::UInt),
            "Lt" => Some(ConstLit::Bool(a < b)),
            "Le" => Some(ConstLit::Bool(a <= b)),
            "Gt" => Some(ConstLit::Bool(a > b)),
            "Ge" => Some(ConstLit::Bool(a >= b)),
            "Eq" => Some(ConstLit::Bool(a == b)),
            "Ne" => Some(ConstLit::Bool(a != b)),
            _ => None,
        };
    }
    match (lhs, rhs) {
        (ConstLit::Int(a), ConstLit::Int(b)) => match name {
            // Wrapping Add/Sub/Mul: the correct result depends on the
            // operand width the i64 carrier has dropped (`255u8 + 1`
            // wraps to 0, not 256), so residualize instead of folding
            // at 64-bit width. Non-wrapping (panic-checked) forms reach
            // the arms below, where checked_* is exact because a valid
            // const initializer never overflows.
            "Add" | "Sub" | "Mul" if operator_mode(kind) == Some("Wrap") => None,
            "Add" => a.checked_add(b).map(ConstLit::Int),
            "Sub" => a.checked_sub(b).map(ConstLit::Int),
            "Mul" => a.checked_mul(b).map(ConstLit::Int),
            "Div" => a.checked_div(b).map(ConstLit::Int),
            "Rem" => a.checked_rem(b).map(ConstLit::Int),
            "Shl" => u32::try_from(b)
                .ok()
                .and_then(|s| a.checked_shl(s))
                .map(ConstLit::Int),
            // Right shift: the carrier lost the operand's signedness, so
            // a negative `a` is ambiguous between a signed value (which
            // wants an arithmetic shift) and an unsigned value past
            // i64::MAX (which wants a logical shift). Fold only for
            // `a >= 0`, where both shifts agree; residualize otherwise.
            "Shr" if a < 0 => None,
            "Shr" => u32::try_from(b)
                .ok()
                .and_then(|s| a.checked_shr(s))
                .map(ConstLit::Int),
            "BitAnd" => Some(ConstLit::Int(a & b)),
            "BitOr" => Some(ConstLit::Int(a | b)),
            "BitXor" => Some(ConstLit::Int(a ^ b)),
            // i64-width overflow flags: exact for I64 operands; a
            // narrower-width divergence would need an overflow the
            // initializer cannot contain (rustc const-checked it).
            "AddChecked" => {
                let (v, o) = a.overflowing_add(b);
                Some(ConstLit::Checked(v, o))
            }
            "SubChecked" => {
                let (v, o) = a.overflowing_sub(b);
                Some(ConstLit::Checked(v, o))
            }
            "MulChecked" => {
                let (v, o) = a.overflowing_mul(b);
                Some(ConstLit::Checked(v, o))
            }
            "Lt" => Some(ConstLit::Bool(a < b)),
            "Le" => Some(ConstLit::Bool(a <= b)),
            "Gt" => Some(ConstLit::Bool(a > b)),
            "Ge" => Some(ConstLit::Bool(a >= b)),
            "Eq" => Some(ConstLit::Bool(a == b)),
            "Ne" => Some(ConstLit::Bool(a != b)),
            _ => None,
        },
        (ConstLit::UInt(a), ConstLit::UInt(b)) => match name {
            // Width is intentionally not flattened to signedness, but the
            // literal carrier does not retain whether this was u8/u16/u32.
            // Rust's checked forms are only exact at u64 width, while plain
            // non-wrapping operations in a valid const initializer are exact
            // whenever they do not overflow u64.
            "Add" | "Sub" | "Mul" if operator_mode(kind) == Some("Wrap") => None,
            "Add" => a.checked_add(b).map(ConstLit::UInt),
            "Sub" => a.checked_sub(b).map(ConstLit::UInt),
            "Mul" => a.checked_mul(b).map(ConstLit::UInt),
            "Div" => a.checked_div(b).map(ConstLit::UInt),
            "Rem" => a.checked_rem(b).map(ConstLit::UInt),
            "Shl" => u32::try_from(b)
                .ok()
                .and_then(|shift| a.checked_shl(shift))
                .map(ConstLit::UInt),
            "Shr" => u32::try_from(b)
                .ok()
                .and_then(|shift| a.checked_shr(shift))
                .map(ConstLit::UInt),
            "BitAnd" => Some(ConstLit::UInt(a & b)),
            "BitOr" => Some(ConstLit::UInt(a | b)),
            "BitXor" => Some(ConstLit::UInt(a ^ b)),
            "AddChecked" => {
                let (value, overflow) = a.overflowing_add(b);
                Some(ConstLit::CheckedUInt(value, overflow))
            }
            "SubChecked" => {
                let (value, overflow) = a.overflowing_sub(b);
                Some(ConstLit::CheckedUInt(value, overflow))
            }
            "MulChecked" => {
                let (value, overflow) = a.overflowing_mul(b);
                Some(ConstLit::CheckedUInt(value, overflow))
            }
            "Lt" => Some(ConstLit::Bool(a < b)),
            "Le" => Some(ConstLit::Bool(a <= b)),
            "Gt" => Some(ConstLit::Bool(a > b)),
            "Ge" => Some(ConstLit::Bool(a >= b)),
            "Eq" => Some(ConstLit::Bool(a == b)),
            "Ne" => Some(ConstLit::Bool(a != b)),
            _ => None,
        },
        (ConstLit::Bool(a), ConstLit::Bool(b)) => match name {
            "Eq" => Some(ConstLit::Bool(a == b)),
            "Ne" => Some(ConstLit::Bool(a != b)),
            "BitAnd" => Some(ConstLit::Bool(a & b)),
            "BitOr" => Some(ConstLit::Bool(a | b)),
            _ => None,
        },
        _ => None,
    }
}

/// Decode `Operand::Const`'s value field, a `ConstantExpr` `[kind, ty]`
/// (inline, `{"Value": [id, [kind, ty]]}`, or `{"Deduplicated": id}`).
/// `kind` shapes:
///   - `{Integer: {Signed|Unsigned: [width, "v"]}}`
///   - `{Bool: bool}` / `{Char: "c"}` / `{Float: {value: "v", ty: "F32|F64"}}`
///   - `{Str: "..."}` / `{ByteStr: "..."}`
///   - `{FnDef: {kind: {Fun: id}, generics: ...}}`
///   - `{Opaque: "..."}` / `{VTableRef: ...}` / `{TraitConst: ...}`
pub(in crate::front::mir) fn decode_constant(
    llbc: &Llbc,
    value: &serde_json::Value,
) -> Result<DecodedConst, LowerError> {
    if let Some(literal) = llbc.const_expr_literal(value) {
        return decode_literal(&literal);
    }
    let kind = llbc
        .const_expr_kind(value)
        .and_then(|kind| kind.as_object().cloned())
        .ok_or_else(|| {
            LowerError::Unsupported(format!("Operand::Const value missing object kind: {value}"))
        })?;
    // `Opaque "<reason>"` — Charon itself bailed on the constant.
    // Forward the reason so it ends up in the synthetic path; the
    // codewriter sees a 0-arg Call it can ignore for analysis.
    if let Some(reason) = kind.get("Opaque").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(format!("opaque:{reason}")));
    }
    // `VTableRef { ... }` — vtable pointer for dynamic dispatch.
    // Treat as an opaque pointer-typed value; covering it faithfully
    // requires the trait dispatch widening.
    if kind.contains_key("VTableRef") {
        return Ok(DecodedConst::Str("__vtable_ref".to_string()));
    }
    // `TraitConst` — trait-associated const. Opaque for now; covering
    // it faithfully requires trait/impl resolution.
    if kind.contains_key("TraitConst") {
        return Ok(DecodedConst::Str("__trait_const".to_string()));
    }
    if let Some(fn_def) = kind.get("FnDef") {
        return decode_fn_def_const(llbc, fn_def);
    }
    // `offset_of!` / `size_of` / `align_of` survive in MIR as constant
    // operands (`OffsetOf` / `SizeOf` / `AlignOf`), not as calls, once
    // rustc has const-promoted them. A generic struct's Charon layout
    // stays symbolic; the instantiation's type arguments are on the
    // constant. The byte count is the declaration-order walk of those
    // fields when the recorded repr keeps that order, and a decline
    // otherwise (`mem_size_align` / `eval_offset_of` return `None`).
    if let Some(off) = kind.get("OffsetOf") {
        let n = eval_offset_of(llbc, off).ok_or_else(|| {
            LowerError::Unsupported(format!("OffsetOf not a concrete layout: {off}"))
        })?;
        return Ok(DecodedConst::UInt(n));
    }
    if let Some(ty) = kind.get("SizeOf") {
        let (n, _) = mem_size_align(llbc, ty, &[], 0).ok_or_else(|| {
            LowerError::Unsupported(format!("SizeOf not a concrete layout: {ty}"))
        })?;
        return Ok(DecodedConst::UInt(n));
    }
    if let Some(ty) = kind.get("AlignOf") {
        let (_, n) = mem_size_align(llbc, ty, &[], 0).ok_or_else(|| {
            LowerError::Unsupported(format!("AlignOf not a concrete layout: {ty}"))
        })?;
        return Ok(DecodedConst::UInt(n));
    }
    // `ConstantExprKind::Cast(expr, CastKind)`.  Upstream
    // `lltype.cast_primitive` / `rint.IntegerRepr.convert_const` convert
    // to the target type; identity-folding the inner drops that cast.
    if let Some(cast) = kind.get("Cast") {
        return fold_cast_constant(llbc, cast);
    }
    // `ConstantExprKind::Ptr` is a raw pointer.  10.04 `decode_constant`
    // (e6797c) had no Ptr arm; `Llbc::const_expr_literal` is Integer /
    // Bool / Char / Float / Str / ByteStr only.  Upstream
    // `rptr.PtrRepr.convert_const` keeps a pointer.  Peeling the
    // pointee would turn `*const u8` of 7 into the integer 7.  Reject.
    if kind.get("Ptr").is_some() {
        return Err(LowerError::Unsupported(format!(
            "raw pointer constant: {value}"
        )));
    }
    // `ConstantExprKind::Ref` is a shared reference.  10.04
    // `decode_constant` had no Ref arm either, so `&T` constants never
    // decoded to the pointee.  Keep that convention; do not peel.
    if kind.get("Ref").is_some() {
        return Err(LowerError::Unsupported(format!(
            "reference constant: {value}"
        )));
    }
    Err(LowerError::Unsupported(format!(
        "Operand::Const kind not yet handled: {value}"
    )))
}

/// Fold `ConstantExprKind::Cast(expr, CastKind)`.
///
/// Integer `Scalar` targets go through `const_cast_int` (same width /
/// signedness rule as `const_narrow_to_target`).  Same-width float
/// `Scalar` (`Float`↔F64, `SingleFloat`↔F32) is identity.  Everything
/// else — int↔float, float width change, 128-bit, Bool, RawPtr, FnPtr,
/// Unsize, Transmute, unknown — is `Unsupported`; no silent identity.
pub(in crate::front::mir) fn fold_cast_constant(
    llbc: &Llbc,
    cast: &Value,
) -> Result<DecodedConst, LowerError> {
    let arr = cast.as_array().ok_or_else(|| {
        LowerError::Unsupported(format!("Cast constant missing inner expr: {cast}"))
    })?;
    let inner = arr.first().ok_or_else(|| {
        LowerError::Unsupported(format!("Cast constant missing inner expr: {cast}"))
    })?;
    let cast_kind = arr
        .get(1)
        .ok_or_else(|| LowerError::Unsupported(format!("Cast constant not folded: {cast}")))?;
    let decoded = decode_constant(llbc, inner)?;
    if let Some(target) = cast_kind.get("Scalar") {
        if target.get("Integer").is_some() {
            let raw = match decoded {
                DecodedConst::Int(n) => n,
                DecodedConst::UInt(n) => n as i64,
                _ => {
                    return Err(LowerError::Unsupported(format!(
                        "Cast constant not folded: {cast}"
                    )));
                }
            };
            let narrowed = const_cast_int(raw, target).ok_or_else(|| {
                LowerError::Unsupported(format!("Cast constant not folded: {cast}"))
            })?;
            let unsigned = target
                .get("Integer")
                .and_then(Value::as_object)
                .is_some_and(|int| int.contains_key("Unsigned"));
            return Ok(if unsigned {
                DecodedConst::UInt(narrowed as u64)
            } else {
                DecodedConst::Int(narrowed)
            });
        }
        let float_width = target.get("Float").and_then(Value::as_str);
        match (&decoded, float_width) {
            (DecodedConst::Float(_), Some("F64")) => return Ok(decoded),
            (DecodedConst::SingleFloat(_), Some("F32")) => return Ok(decoded),
            _ => {}
        }
    }
    Err(LowerError::Unsupported(format!(
        "Cast constant not folded: {cast}"
    )))
}

pub(in crate::front::mir) fn tyref_json(ty: &TyRef) -> serde_json::Value {
    match ty {
        TyRef::Dedup { id } => serde_json::json!({"Deduplicated": id}),
        TyRef::Inline { value: (id, body) } => serde_json::json!({"Value": [id, body]}),
        TyRef::Other(v) => v.clone(),
    }
}

/// Index of a depth-0 type parameter. `None` when `body` is not a type
/// variable. A non-zero de Bruijn depth is `Some(usize::MAX)` so the
/// caller fails closed instead of substituting the wrong binder.
pub(in crate::front::mir) fn typevar_slot(body: &serde_json::Value) -> Option<usize> {
    let bound = body.get("TypeVar")?.get("Bound")?.as_array()?;
    let depth = bound.first()?.as_u64()?;
    let idx = bound.get(1)?.as_u64()? as usize;
    if depth != 0 {
        return Some(usize::MAX);
    }
    Some(idx)
}

pub(in crate::front::mir) fn align_up_bytes(offset: u64, align: u64) -> Option<u64> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    offset.checked_add(mask).map(|n| n & !mask)
}

/// Declaration order is the in-memory field order only when the recorded
/// repr fixes it. `ReprOptions::guarantees_fixed_field_order` is
/// `repr_algo == C`. `transparent` is a single-field wrapper whose layout
/// is that field. A missing repr, `repr(Rust)`, or a non-null
/// `align_modif` (pack or an explicit align) declines: the caller then
/// returns `None`.
pub(in crate::front::mir) fn decl_repr_keeps_declaration_order(
    decl: &TypeDecl,
    target: &str,
) -> bool {
    let Some(raw) = decl.layout.as_ref() else {
        return false;
    };
    let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(raw.get()) else {
        return false;
    };
    let entry = if let Some(found) = entries
        .iter()
        .find(|entry| entry.get("key").and_then(serde_json::Value::as_str) == Some(target))
    {
        found
    } else if entries.len() == 1 {
        &entries[0]
    } else {
        return false;
    };
    let Some(repr) = entry.get("value").and_then(|value| value.get("repr")) else {
        return false;
    };
    if !repr
        .get("align_modif")
        .is_none_or(serde_json::Value::is_null)
    {
        return false;
    }
    if repr
        .get("transparent")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return true;
    }
    repr.get("repr_algo").and_then(serde_json::Value::as_str) == Some("C")
}

/// Byte size and alignment of `ty`. `subst` replaces depth-0 type
/// parameters (an `OffsetOf` / `SizeOf` instantiation). A concrete
/// `layout_for_target` wins. A generic struct walks its fields in
/// declaration order only when [`decl_repr_keeps_declaration_order`]
/// accepts the recorded repr; otherwise the walk declines.
pub(in crate::front::mir) fn mem_size_align(
    llbc: &Llbc,
    ty: &serde_json::Value,
    subst: &[serde_json::Value],
    depth: u8,
) -> Option<(u64, u64)> {
    if depth > 32 {
        return None;
    }
    let body = tyexpr_body(llbc, ty)?;
    if let Some(slot) = typevar_slot(body) {
        let repl = subst.get(slot)?;
        return mem_size_align(llbc, repl, &[], depth + 1);
    }
    if let Some(lit) = body.get("Scalar") {
        let size = primitive_size_align(false, lit)?;
        let align = primitive_size_align(true, lit)?;
        return Some((size, align));
    }
    if body.get("Ref").is_some() || body.get("RawPtr").is_some() {
        let word = crate::layout::target_word_size() as u64;
        return Some((word, word));
    }
    if let Some(arr) = body.get("Array").and_then(serde_json::Value::as_array) {
        let elem = arr.first()?;
        let (elem_size, elem_align) = mem_size_align(llbc, elem, subst, depth + 1)?;
        let len: u64 = charon_array_len_to_string(arr.get(1)?, llbc).parse().ok()?;
        let size = elem_size.checked_mul(len)?;
        return Some((size, elem_align.max(1)));
    }
    let adt = body.get("Adt").unwrap_or(body);
    let id = adt.get("id")?.as_u64()?;
    let args = adt
        .get("generics")
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)
        .map(|a| {
            a.iter()
                .map(|arg| subst_type_args(llbc, arg, subst))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let decl = llbc.type_by_id(id)?;
    let target = std::env::var("TARGET").unwrap_or_default();
    if args.is_empty()
        && let Some(layout) = decl.layout_for_target(llbc, &target)
        && let (Some(size), Some(align)) = (layout.size, layout.align)
    {
        return Some((size, align));
    }
    if !decl_repr_keeps_declaration_order(decl, &target) {
        return None;
    }
    let fields: &[majit_charon_reader::ullbc::FieldDecl] = match &decl.kind {
        TypeDeclKind::Struct(fields) => fields,
        TypeDeclKind::Union(fields) => {
            let mut size = 0u64;
            let mut align = 1u64;
            for field in fields {
                let (fs, fa) = mem_size_align(llbc, &tyref_json(&field.ty), &args, depth + 1)?;
                size = size.max(fs);
                align = align.max(fa);
            }
            let size = align_up_bytes(size, align)?;
            return Some((size, align));
        }
        _ => return None,
    };
    let mut offset = 0u64;
    let mut max_align = 1u64;
    for field in fields {
        let (fs, fa) = mem_size_align(llbc, &tyref_json(&field.ty), &args, depth + 1)?;
        let fa = fa.max(1);
        offset = align_up_bytes(offset, fa)?;
        offset = offset.checked_add(fs)?;
        max_align = max_align.max(fa);
    }
    Some((align_up_bytes(offset, max_align)?, max_align))
}

pub(in crate::front::mir) fn subst_type_args(
    llbc: &Llbc,
    ty: &serde_json::Value,
    subst: &[serde_json::Value],
) -> serde_json::Value {
    let Some(body) = tyexpr_body(llbc, ty) else {
        return ty.clone();
    };
    if let Some(slot) = typevar_slot(body) {
        if slot == usize::MAX {
            return body.clone();
        }
        return subst.get(slot).cloned().unwrap_or_else(|| body.clone());
    }
    body.clone()
}

/// `OffsetOf([TypeDeclRef, variant, field])` → byte offset.
pub(in crate::front::mir) fn eval_offset_of(llbc: &Llbc, off: &serde_json::Value) -> Option<u64> {
    let arr = off.as_array()?;
    let declref = arr.first()?;
    let variant = arr.get(1).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
    let field_idx = arr.get(2)?.as_u64()? as usize;
    let id = declref.get("id")?.as_u64()?;
    let args = declref
        .get("generics")
        .and_then(|g| g.get("types"))
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let decl = llbc.type_by_id(id)?;
    let target = std::env::var("TARGET").unwrap_or_default();
    if args.is_empty()
        && let Some(layout) = decl.layout_for_target(llbc, &target)
        && let Some(n) = layout.field_offset(variant, field_idx)
        && n != u64::MAX
    {
        return Some(n);
    }
    if variant != 0 {
        return None;
    }
    if !decl_repr_keeps_declaration_order(decl, &target) {
        return None;
    }
    let fields = match &decl.kind {
        TypeDeclKind::Struct(fields) => fields,
        _ => return None,
    };
    let field = fields.get(field_idx)?;
    // Offset of this field is the declaration-order cursor before it.
    let mut offset = 0u64;
    for prior in fields.iter().take(field_idx) {
        let (fs, fa) = mem_size_align(llbc, &tyref_json(&prior.ty), &args, 0)?;
        offset = align_up_bytes(offset, fa.max(1))?;
        offset = offset.checked_add(fs)?;
    }
    let (_, fa) = mem_size_align(llbc, &tyref_json(&field.ty), &args, 0)?;
    align_up_bytes(offset, fa.max(1))
}

/// `FnDef { kind: Fun(id), generics }` — a constant function item.
pub(in crate::front::mir) fn decode_fn_def_const(
    llbc: &Llbc,
    fn_def: &serde_json::Value,
) -> Result<DecodedConst, LowerError> {
    let inner = fn_def
        .as_object()
        .and_then(|m| m.get("kind"))
        .and_then(|m| m.get("Fun"))
        .and_then(Value::as_u64)
        .ok_or_else(|| LowerError::Unsupported(format!("FnDef shape not yet handled: {fn_def}")))?;
    let fd = llbc.fn_by_id(inner).ok_or_else(|| {
        LowerError::Schema(format!(
            "FnDef constant references unknown FunDecl id {inner}"
        ))
    })?;
    // Spell an impl-owned callee the way the registration side keys it
    // (`CallPath::for_impl_method(self_ty_root, leaf)`), exactly as the
    // Call terminator does (`call_target_segments`). A free function keeps
    // `name_path()`: that is its registered spelling.
    Ok(DecodedConst::FnPath(fundecl_fn_item_segments(llbc, fd)))
}

pub(in crate::front::mir) fn decode_literal(
    lit: &serde_json::Value,
) -> Result<DecodedConst, LowerError> {
    let lit_obj = lit
        .as_object()
        .ok_or_else(|| LowerError::Schema(format!("Literal not object: {lit}")))?;
    if let Some(scalar_obj) = lit_obj.get("Scalar").and_then(Value::as_object) {
        for (k, payload) in scalar_obj {
            let arr = payload.as_array().ok_or_else(|| {
                LowerError::Schema(format!("Scalar {k}: payload not array: {payload}"))
            })?;
            if arr.len() != 2 {
                continue;
            }
            let v_str = arr[1].as_str().ok_or_else(|| {
                LowerError::Schema(format!("Scalar {k}: value not a string: {payload}"))
            })?;
            let width = arr[0].as_str().unwrap_or_default();
            return Ok(match k.as_str() {
                "Signed" if width == "I128" => DecodedConst::Int128(
                    v_str
                        .parse()
                        .map_err(|e| LowerError::Schema(format!("Scalar I128 parse: {e}")))?,
                ),
                "Unsigned" if width == "U128" => DecodedConst::UInt128(
                    v_str
                        .parse()
                        .map_err(|e| LowerError::Schema(format!("Scalar U128 parse: {e}")))?,
                ),
                "Signed" | "Isize" => DecodedConst::Int(
                    v_str
                        .parse()
                        .map_err(|e| LowerError::Schema(format!("Scalar Signed parse: {e}")))?,
                ),
                "Unsigned" | "Usize" => DecodedConst::UInt(
                    v_str
                        .parse::<u64>()
                        .map_err(|e| LowerError::Schema(format!("Scalar Unsigned parse: {e}")))?,
                ),
                _ => {
                    return Err(LowerError::Unsupported(format!(
                        "Scalar kind {k} not yet decoded"
                    )));
                }
            });
        }
    }
    if let Some(b) = lit_obj.get("Bool").and_then(Value::as_bool) {
        return Ok(DecodedConst::Bool(b));
    }
    if let Some(f) = lit_obj.get("Float") {
        if let Some(obj) = f.as_object()
            && let Some(s) = obj.get("value").and_then(Value::as_str)
            && let Ok(v) = s.parse::<f64>()
        {
            // The width is carried beside the value, and an `f32` keeps
            // its own carrier: narrowing to `f32` here is the conversion
            // rustc already applied to the literal, and the f64 bit
            // pattern of the same number is a different value.
            return Ok(if obj.get("ty").and_then(Value::as_str) == Some("F32") {
                DecodedConst::SingleFloat((v as f32).to_bits())
            } else {
                DecodedConst::Float(v.to_bits())
            });
        }
        return Err(LowerError::Schema(format!("Float shape: {f}")));
    }
    if let Some(s) = lit_obj.get("Str").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(s.to_string()));
    }
    // A `char` is one Unicode scalar (`bookkeeper.py` `immutablevalue`
    // `SomeUnicodeCodePoint`). SwitchInt arms are the same length-1
    // unicode (`AbstractUniCharRepr.convert_const`).
    if let Some(s) = lit_obj.get("Char").and_then(Value::as_str) {
        let mut chars = s.chars();
        return match (chars.next(), chars.next()) {
            (Some(c), None) => Ok(DecodedConst::UniChar(u32::from(c))),
            _ => Err(LowerError::Schema(format!(
                "Char literal is not one code point: {lit}"
            ))),
        };
    }
    if let Some(s) = lit_obj.get("ByteStr").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(s.to_string()));
    }
    Err(LowerError::Unsupported(format!(
        "Literal shape not yet decoded: {lit}"
    )))
}

pub(in crate::front::mir) fn scalar_to_const_value(
    v: &serde_json::Value,
    llbc: &Llbc,
) -> Option<ConstValue> {
    if let Some(id) = v.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        let body = llbc.dedup_const_body(id)?;
        return scalar_to_const_value(body.as_array()?.first()?, llbc);
    }
    let obj = v.as_object()?;
    if let Some(int) = obj.get("Integer") {
        return scalar_to_const_value(&serde_json::json!({"Scalar": int}), llbc);
    }
    // `{Scalar: {Signed|Unsigned|Isize|Usize: [ty, value]}}`
    // Charon also emits `{Scalar: {Char: ["Char", "97"]}}` for a `char`
    // SwitchInt arm; `AbstractUniCharRepr.convert_const` wants a length-1
    // unicode, not the code-point integer.
    if let Some(scalar) = obj.get("Scalar").and_then(Value::as_object) {
        for (k, payload) in scalar {
            let arr = payload.as_array()?;
            if arr.len() != 2 {
                continue;
            }
            return Some(match k.as_str() {
                "Char" => {
                    let n = u32::try_from(scalar_value_to_i64(&arr[1])?).ok()?;
                    ConstValue::UniStr(char::from_u32(n)?.to_string())
                }
                "Unsigned" | "Usize" => ConstValue::UInt(scalar_value_to_u64(&arr[1])?),
                _ => ConstValue::Int(scalar_value_to_i64(&arr[1])?),
            });
        }
    }
    // `{Char: "c"}` — character matched as a SwitchInt arm.
    // `AbstractUniCharRepr.convert_const` (`rstr.py`) accepts a length-1
    // unicode; the UniChar register bank is still `getkind == 'int'`.
    if let Some(c) = obj.get("Char").and_then(Value::as_str) {
        let mut chars = c.chars();
        let ch = chars.next()?;
        if chars.next().is_some() {
            return None;
        }
        return Some(ConstValue::UniStr(ch.to_string()));
    }
    // `{Bool: true}` — boolean matched as a SwitchInt arm.
    if let Some(b) = obj.get("Bool").and_then(Value::as_bool) {
        return Some(ConstValue::Int(b as i64));
    }
    // One-arm pointer/vtable `Ref` is the `Some` (1) case of a niche
    // `Option` switch; `lower_niche_option_switch` turns it into `If`.
    if obj.contains_key("Ref") {
        return Some(ConstValue::Int(1));
    }
    None
}

/// Extract a scalar value as `i64`. Accepts both string ("0") and
/// JSON-numeric (0) representations: Charon emits `["Char", "97"]`
/// for `'a'` but `["Bool", true]` for boolean discriminants.
pub(in crate::front::mir) fn scalar_value_to_i64(v: &serde_json::Value) -> Option<i64> {
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    if let Some(b) = v.as_bool() {
        return Some(b as i64);
    }
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(n) = v.as_u64() {
        return Some(n as i64);
    }
    None
}

pub(in crate::front::mir) fn scalar_value_to_u64(v: &serde_json::Value) -> Option<u64> {
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    if let Some(b) = v.as_bool() {
        return Some(u64::from(b));
    }
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(n) = v.as_i64() {
        return u64::try_from(n).ok();
    }
    None
}

#[cfg(test)]
mod tests;
