use super::*;

/// Legacy macro-time per-struct type id. Hashes the struct path tokens; retain
/// it for a caller which genuinely needs a compile-time value, but do not use
/// it for descriptor/layout identities: spelling one Rust type through two
/// paths gives two different token streams.
///
/// Descriptor/layout identities instead use [`struct_type_id_tokens`], which
/// emits `__majit_struct_type_id::<T>()` (resolved `type_name`, hashed like
/// `BhSizeSpec.type_id`). The same id keys the
/// builder's `struct_size_specs` cache so each
/// `setfield_gc_*` resolves its field's parent SizeDescr + `index_in_parent`
/// (`descr.py`).  Distinct struct paths collide only at `DefaultHasher`'s
/// 64-bit range, matching the runtime `LLType::Struct(type_id)` cache-key
/// identity.
///
/// `descr.py get_size_descr` keys the cache by the actual lltype STRUCT, so
/// a `GcStruct` and a raw `Struct` with identical fields are DISTINCT lltypes
/// with distinct descrs.  A GC `new_struct(T)` and a raw `ref(T)` state scalar
/// share a path and would otherwise share an id; the first-registered
/// `is_gc_managed` flag would then pin both, so a raw getfield could inherit a
/// GC descr's `GUARD_GC_TYPE` and read a non-existent type-id word at `ptr - 8`
/// (or a GC alloc could lose its type guard).  Fold `is_gc_managed` into the id
/// so the two layouts never alias: GC keeps the bare hash (existing
/// `new_struct` ids unchanged), raw appends a discriminator.
pub(super) fn struct_type_id(path: &syn::Path, is_gc_managed: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    quote!(#path).to_string().hash(&mut hasher);
    if !is_gc_managed {
        "raw".hash(&mut hasher);
    }
    hasher.finish()
}

/// Emit the runtime lltype-identity surrogate for `path`.
///
/// `descr.py` `get_size_descr` is keyed by the lltype `STRUCT` object, not
/// by a textual spelling in an individual graph. Every non-generic spelling
/// emits `__majit_struct_type_id::<#path>`, whose body hashes
/// `core::any::type_name::<T>()` — rustc resolves `crate::` / `self::` /
/// `super::` / `use` / re-exports to one definition — through the same
/// function the graph codewriter uses for `BhSizeSpec.type_id`.
///
/// A path with generic arguments can name a type parameter this proc macro
/// cannot prove `'static`, so `type_name::<#path>()` is not emitted there.
/// The legacy token hash stays. It does not alias a non-generic spelling:
/// `quote!(#path)` inserts spaces (`Foo < T >`) and `type_name` does not, so
/// the two hashes are of different strings. A 64-bit collision is the same
/// residual `path_hash` already accepts between distinct structs.
pub(super) fn struct_type_id_tokens(path: &syn::Path, is_gc_managed: bool) -> TokenStream {
    let has_generic_args = path
        .segments
        .iter()
        .any(|segment| !segment.arguments.is_none());
    if has_generic_args {
        let legacy = struct_type_id(path, is_gc_managed);
        quote! { #legacy }
    } else {
        quote! { majit_metainterp::__majit_struct_type_id::<#path>(#is_gc_managed) }
    }
}

/// Pointee of `*mut Struct` / `*const Struct` when `Struct` is a type name.
///
/// Primitive pointees (`*mut u8`) are not structs, so a cast to one does not
/// retarget a field read.
fn struct_pointee_of_pointer(ty: &syn::Type) -> Option<syn::Path> {
    let syn::Type::Ptr(ptr) = ty else {
        return None;
    };
    let syn::Type::Path(path) = ptr.elem.as_ref() else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let last = path.path.segments.last()?;
    if !last.arguments.is_none() {
        return None;
    }
    let name = last.ident.to_string();
    if !name.starts_with(|c: char| c.is_ascii_uppercase()) {
        return None;
    }
    Some(path.path.clone())
}

impl<'c> Lowerer<'c> {
    /// The value of an int operand that is a flow-graph `Constant`: an
    /// integer, byte or bool literal, or a SCREAMING_CASE constant path —
    /// the forms [`Self::lower_value_expr`] would otherwise load into a fresh
    /// register with `load_const_i_value`. `flatten.py` leaves a `Constant`
    /// in the operation and `assembler.py` `emit_const` encodes it there, so
    /// a consumer that takes this value emits no `int_copy` first.
    pub(super) fn int_constant_operand(&self, expr: &Expr) -> Option<TokenStream> {
        match expr {
            Expr::Paren(ExprParen { expr, .. }) => self.int_constant_operand(expr),
            Expr::Group(group) => self.int_constant_operand(&group.expr),
            Expr::Lit(ExprLit {
                lit: Lit::Byte(byte_lit),
                ..
            }) => {
                let value = i64::from(byte_lit.value());
                Some(quote! { #value })
            }
            Expr::Path(ExprPath {
                path, qself: None, ..
            }) => {
                if let Some(ident) = path.get_ident()
                    && self.bindings.contains_key(&ident.to_string())
                {
                    return None;
                }
                let last = path.segments.last()?;
                if !last.arguments.is_none()
                    || !last
                        .ident
                        .to_string()
                        .starts_with(|c: char| c.is_uppercase())
                {
                    return None;
                }
                Some(quote! { (#path as i64) })
            }
            _ => {
                let value = int_literal_value(expr)?;
                Some(quote! { #value })
            }
        }
    }

    pub(super) fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        // State field read (register/tape machines).
        if let Some(binding) = self.lower_state_field_read(expr) {
            return Some(binding);
        }
        // `state.<vable>.<field>` / `state.<vable>.<array>[i]` before the
        // heap getfield / getarrayitem rewrites, which match the same syntax.
        // `jtransform.py` `getfield_vable_*` / `getarrayitem_vable_*`.
        if let Some(binding) = self.lower_vable_field_read(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_vable_array_read(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_vable_array_len(expr) {
            return Some(binding);
        }
        // Field read through a `ref(T)` state scalar: `state.<ref>.<member>`
        // → getfield_gc_i. Placed after lower_state_field_read (which only
        // matches `state.<scalar>` whose base is `state`) so the plain scalar
        // read wins and this handles the `Field`-of-`Field` shape.
        if let Some(binding) = self.lower_state_ref_field_getfield(expr) {
            return Some(binding);
        }
        // Array element read on a local ref binding whose field is declared
        // in `array_fields`: headered → getfield_gc_r then getarrayitem_gc_*;
        // headerless → getfield_gc_i then getarrayitem_raw_*
        // (`rewrite_op_getarrayitem`). Before the plain field read, which
        // cannot match an `Index` shape anyway, and before
        // `lower_state_array_read`, which is the `state.<array>` form.
        if let Some(binding) = self.lower_ref_binding_array_read(expr) {
            return Some(binding);
        }
        // Field read on a local ref binding with known struct type:
        // `<binding>.<field>` → getfield_gc_i/getfield_gc_r.
        if let Some(binding) = self.lower_ref_binding_getfield(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_state_array_read(expr) {
            return Some(binding);
        }
        // RPython pyopcode.py:171 `ord(co_code[next_instr])` — an indexed
        // read of the green bytecode array fetches an immediate operand.
        // The dispatch-top opcode fetch is lowered by the stmt-level
        // `try_lower_opcode_fetch_stmt`, which arm-body lowering does not
        // reach; this handler lowers the identical `program[<idx>]` shape
        // in value position so operand reads inside opcode arm bodies emit
        // the same `getarrayitem_gc_i` on the green array (folded by the
        // optimizer since array + index are green).
        if let Some(binding) = self.lower_env_array_read(expr) {
            return Some(binding);
        }
        // RPython call.py passes the virtualizable object itself to residual
        // helpers. Rust spells that object argument `&mut state`; preserve the
        // existing ref register instead of trying to manufacture a raw array
        // pointer from a redirected virtualizable field.
        if let Some(binding) = self.lower_vable_reference(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:655 — suppress hint_access_directly(frame) /
        // hint_fresh_virtualizable(frame) function calls as identity.
        // These return the frame unchanged, so lower the argument instead.
        if let Some(binding) = self.lower_vable_hint_identity_call(expr) {
            return Some(binding);
        }
        // RPython jtransform.py rewrite_op_jit_conditional_call_value — conditional_call_elidable!(value, func, args...)
        if let Some(binding) = self.lower_conditional_call_elidable(expr) {
            return Some(binding);
        }
        // RPython jtransform.py handle_recursive_call — recursive_portal_call!(driver, greens...)
        if let Some(binding) = self.lower_recursive_portal_call(expr) {
            return Some(binding);
        }
        // `jtransform.py` `_rewrite_equality` recognises `Constant(nullptr)`.
        // The same constant is a value here, not only a compare operand.
        if expr_is_null_ptr(expr) {
            let reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstR, vec![], vec![Register::ref_(reg)]),
                quote! { __builder.load_const_r_value(#reg, 0i64); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: None,
            });
        }

        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) => {
                let value = int_lit.base10_parse::<i64>().ok()?;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            // RPython's bytecode loops compare `ord(code[p])` with integer
            // character constants. Rust spells the same one-byte constant as
            // `b'['`; it belongs on the int channel with its `u8` value.
            Expr::Lit(ExprLit {
                lit: Lit::Byte(byte_lit),
                ..
            }) => {
                let value = i64::from(byte_lit.value());
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            Expr::Lit(ExprLit {
                lit: Lit::Float(float_lit),
                ..
            }) => {
                let value = float_lit.base10_parse::<f64>().ok()?;
                let bits = value.to_bits() as i64;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstF, vec![], vec![Register::float(reg)]),
                    quote! { __builder.load_const_f_value(#reg, #bits); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Float,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            // Bool comparisons (`stackok == false`) ride the int channel:
            // bool bindings are IntLt/IntLe/… results carrying 0/1.
            Expr::Lit(ExprLit {
                lit: Lit::Bool(bool_lit),
                ..
            }) => {
                let value = i64::from(bool_lit.value);
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            Expr::Path(ExprPath { path, .. }) => {
                // A single-segment path naming a known local binding resolves
                // to that binding.
                if let Some(ident) = path.get_ident()
                    && let Some(binding) = self.bindings.get(&ident.to_string())
                {
                    return Some(binding.clone());
                }
                // Otherwise a path whose final segment is a SCREAMING_CASE
                // symbolic constant (`VAL_QUEUE`, `consts::VAL_PORT`) lowers to
                // a runtime int const — the same `#path as i64` form match-arm
                // patterns use (`extract_pat_value_tokens`); the Rust compiler
                // resolves the const value while building the JitCode, and a
                // green consumer folds it. A lowercase non-binding path stays
                // unlowerable (fail-closed).
                let last = path.segments.last()?;
                if !last.arguments.is_none()
                    || !last
                        .ident
                        .to_string()
                        .starts_with(|c: char| c.is_uppercase())
                {
                    return None;
                }
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #path as i64); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            // A pointer cast keeps the bits and the ref bank. Integer and
            // float targets are the arms below; a ref source cast to one of
            // those stays unlowered because the bytecode cast that moves
            // between the banks checks a tagged immediate.
            // `*mut Struct` / `*const Struct` names the pointee the next
            // `(*ident).field` resolves against. A cast to `*mut u8` and the
            // other non-struct pointees leaves the binding's struct as it was.
            Expr::Cast(ExprCast { expr, ty, .. })
                if !is_supported_int_cast(ty) && !is_supported_float_type(ty) =>
            {
                // `jtransform.py` `rewrite_op_cast_pointer`: a literal call
                // cast to a raw pointer is a prebuilt address, read when the
                // jitcode is built.
                if type_is_raw_pointer(ty) && expr_is_literal_call(expr) {
                    let reg = self.alloc_reg();
                    self.emit_op(
                        OpMeta::linear(OpKind::LoadConstR, vec![], vec![Register::ref_(reg)]),
                        quote! { __builder.load_const_r_value(#reg, (#expr as #ty) as i64); },
                    );
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type: struct_pointee_of_pointer(ty),
                    });
                }
                let binding = self.lower_value_expr(expr)?;
                match binding.kind {
                    BindingKind::Ref => {
                        let mut binding = binding;
                        if let Some(path) = struct_pointee_of_pointer(ty) {
                            binding.struct_type = Some(path);
                        }
                        Some(binding)
                    }
                    _ => None,
                }
            }
            Expr::Unsafe(inner) => self.lower_block_value(&inner.block),
            Expr::Cast(ExprCast { expr, ty, .. }) if is_supported_float_type(ty) => {
                let unsigned = expr_is_unsigned_int(expr);
                let binding = self.lower_value_expr(expr)?;
                match binding.kind {
                    // Same-type force_cast is a no-op (`rewrite_op_force_cast`
                    // when `v_arg.concretetype == v_result.concretetype`).
                    BindingKind::Float => Some(binding),
                    BindingKind::Int if unsigned => self.lower_uint_to_float_cast(binding),
                    BindingKind::Int => self.lower_int_to_float_cast(binding),
                    BindingKind::Ref => None,
                }
            }
            Expr::Cast(ExprCast { expr, ty, .. }) if is_supported_int_cast(ty) => {
                let binding = self.lower_value_expr(expr)?;
                match binding.kind {
                    BindingKind::Int => self.lower_int_cast(binding, ty),
                    // A FLOAT operand is the `cast_int_to_float` arm above run
                    // backwards. Only a word-width target lowers: Rust saturates
                    // an `as` cast at the TARGET type's bounds, so `300.0 as i8`
                    // is `127`, which a float->i64 followed by an int narrowing
                    // (`44`) would not reproduce.
                    BindingKind::Float if is_word_width_int(ty) => {
                        if type_is_unsigned_int(ty) {
                            self.lower_float_to_uint_cast(binding)
                        } else {
                            self.lower_float_to_int_cast(binding)
                        }
                    }
                    _ => None,
                }
            }
            Expr::Paren(ExprParen { expr, .. }) => self.lower_value_expr(expr),
            Expr::If(expr_if) => self.lower_if_value(expr_if),
            Expr::Match(expr_match) => match self.lower_checked_ovf_match(expr_match) {
                Some(result) => result,
                None => self.lower_match_value(expr_match),
            },
            Expr::Unary(ExprUnary { op, expr, .. }) => self.lower_unary(op, expr),
            Expr::Binary(binary) => self.lower_binary(binary),
            Expr::Call(call) => {
                // `jtransform.py` `rewrite_op_getinteriorfield`: `len(s)` /
                // `s[i]` on an RPython `str` become `strlen` / `strgetitem`.
                if let Some(binding) = self.lower_rstr_call(call) {
                    return Some(binding);
                }
                // jtransform.py rewrite_op_hint: promote → int_guard_value
                if let Some(binding) = self.lower_promote_call(call) {
                    return Some(binding);
                }
                // pyjitpl.py opimpl_assert_not_none — emit
                // BC_ASSERT_NOT_NONE and return the unwrapped binding.
                if let Some(binding) = self.lower_assert_not_none_call(call) {
                    return Some(binding);
                }
                // A `<fn>(state.<pool_base_ref>, <idx>)` marker call reading a
                // raw-pointer pool array lowers to getarrayitem_gc_r instead of
                // a residual CALL_R (re-producible heap read, no diverging red).
                if let Some(binding) = self.lower_pool_array_get_call(call) {
                    return Some(binding);
                }
                // jtransform.py _handle_int_special — pure int binop
                // aliases bypass the call-policy machinery and emit a native
                // IR op (IntAdd, IntSub, etc.) directly.
                if let Some(binding) = self.lower_native_int_binop_call(call) {
                    return Some(binding);
                }
                if let Some(binding) = self.lower_native_tag_small_call(call) {
                    return Some(binding);
                }
                // A helper that hands its argument straight back is opened
                // by `inline.auto_inline_graphs` upstream, before the
                // codewriter ever sees the graph. This front end has no such
                // stage, so the drop is declared and applied here.
                if let Some(binding) = self.lower_native_identity_call(call) {
                    return Some(binding);
                }
                // Raw native-memory load intrinsic `majit_raw_load_iXX(base, ea)`
                // → raw_load_i (jtransform.py rewrite_op_raw_load).
                if let Some(binding) = self.lower_raw_load_call(call) {
                    return Some(binding);
                }
                // Float<->int bitcast intrinsics `majit_f64_to_bits(f)` /
                // `majit_bits_to_f64(i)` → convert_{float_bytes_to_longlong,
                // longlong_bytes_to_float} (rpython convert ops).
                if let Some(binding) = self.lower_float_bitcast_call(call) {
                    return Some(binding);
                }
                // Unsigned compare intrinsics `majit_uint_lt(a, b)` /
                // `majit_uint_le(a, b)` → uint_lt / uint_le (the signed opcode
                // the generic binop path emits is wrong for uint).
                if let Some(binding) = self.lower_uint_intrinsic_call(call) {
                    return Some(binding);
                }
                // `<IntTy>::from_le_bytes([b0, b1, ...])` / `from_be_bytes`
                // is the array-literal spelling of the shift/or widening
                // already accepted as `(b0 as T) | ((b1 as T) << 8) | ...`.
                if let Some(binding) = self.lower_from_endian_bytes_call(call) {
                    return Some(binding);
                }
                self.lower_call_value(call)
            }
            Expr::MethodCall(call) => {
                if let Some(binding) = self.lower_wrapping_int_method_call(call) {
                    return Some(binding);
                }
                if let Some(binding) = self.lower_float_bitcast_method_call(call) {
                    return Some(binding);
                }
                if let Some(binding) = self.lower_ptr_is_null_method_call(call) {
                    return Some(binding);
                }
                self.lower_method_call_value(call)
            }
            Expr::Struct(s) => self.lower_struct_value(s),
            _ => None,
        }
    }

    /// `convert_float_bytes_to_longlong` for an already-lowered float source.
    fn emit_convert_float_bytes_to_longlong(&mut self, src: Binding) -> Option<Binding> {
        if !matches!(src.kind, BindingKind::Float) {
            return None;
        }
        let src_reg = src.reg;
        let dst = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                vec![Register::float(src_reg)],
                vec![Register::int(dst)],
            ),
            quote! { __builder.record_convert_float_bytes_to_longlong(#dst, #src_reg); },
        );
        Some(Binding {
            reg: dst,
            kind: BindingKind::Int,
            depends_on_stack: src.depends_on_stack,
            struct_type: None,
        })
    }

    /// `convert_longlong_bytes_to_float` for an already-lowered int source.
    fn emit_convert_longlong_bytes_to_float(&mut self, src: Binding) -> Option<Binding> {
        if !matches!(src.kind, BindingKind::Int) {
            return None;
        }
        let src_reg = src.reg;
        let dst = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                vec![Register::int(src_reg)],
                vec![Register::float(dst)],
            ),
            quote! { __builder.record_convert_longlong_bytes_to_float(#dst, #src_reg); },
        );
        Some(Binding {
            reg: dst,
            kind: BindingKind::Float,
            depends_on_stack: src.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower the float<->int bitcast intrinsics: `majit_f64_to_bits(f)` (float
    /// argument → its i64 bit pattern, `convert_float_bytes_to_longlong`) and
    /// `majit_bits_to_f64(i)` (int bits → float, `convert_longlong_bytes_to_float`).
    /// Both reinterpret the 64-bit pattern (no value change); a branchless float
    /// bit-select uses them to stay bit-exact where an arithmetic blend cannot.
    ///
    /// `f64::to_bits` / `f64::from_bits` are the same two conversions and are
    /// accepted here in their UFCS spelling (the method spelling `x.to_bits()`
    /// is `lower_float_bitcast_method_call`).  Recognizing only the free
    /// functions made an arm that reached for the inherent method degrade to an
    /// abort stub with nothing naming it — cel's `OP_RETURN_F` and its
    /// `OP_LOAD_CONST_F` mirror, the second of which the interpreter had already
    /// worked around by hoisting the op out of the traced loop body.
    fn lower_float_bitcast_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        if call.args.len() != 1 {
            return None;
        }
        let segments = canonical_expr_segments(&call.func)?;
        let owner = segments
            .len()
            .checked_sub(2)
            .map(|i| segments[i].as_str())
            .unwrap_or("");
        let to_bits = match (owner, segments.last()?.as_str()) {
            (_, "majit_f64_to_bits") | ("f64", "to_bits") => true,
            (_, "majit_bits_to_f64") | ("f64", "from_bits") => false,
            _ => return None,
        };
        let src = self.lower_value_expr(&call.args[0])?;
        if to_bits {
            self.emit_convert_float_bytes_to_longlong(src)
        } else {
            self.emit_convert_longlong_bytes_to_float(src)
        }
    }

    /// The inherent-method spelling of `majit_f64_to_bits`: `<f64>.to_bits()`.
    /// There is no method spelling of the reverse — `f64::from_bits` is an
    /// associated function and reaches `lower_float_bitcast_call`.
    fn lower_float_bitcast_method_call(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        if call.method != "to_bits" || !call.args.is_empty() {
            return None;
        }
        let src = self.lower_value_expr(&call.receiver)?;
        self.emit_convert_float_bytes_to_longlong(src)
    }

    /// Lower the unsigned integer intrinsics. Both operands are read from the
    /// int bank as unsigned 64-bit values. `opcode_for_binop` collapses every
    /// Rust binary operator to its SIGNED opcode (`<` to `IntLt`, `/` to
    /// `IntFloorDiv`) and `BindingKind` has no signedness at all, so an
    /// explicit intrinsic is the only way to select an unsigned operation from
    /// the tracing frontend.
    ///
    /// The two families lower differently, mirroring RPython:
    ///   * `majit_uint_lt`/`majit_uint_le` become the `uint_lt`/`uint_le`
    ///     primitives (`bhimpl_uint_lt`/`bhimpl_uint_le`), because
    ///     `resoperation.py:1017-1018` keeps `UINT_LT`/`UINT_LE` as real
    ///     resops.
    ///   * `majit_uint_div`/`majit_uint_mod` become the `int.udiv`/`int.umod`
    ///     oopspec residual calls, because `UINT_FLOORDIV` was REMOVED from the
    ///     resop set in 2016 and `rint.py:434,525` route unsigned `/` and `%`
    ///     through `ll_uint_py_div`/`ll_uint_py_mod` instead. This is the same
    ///     redirect `binop_i_emit_tokens` already performs for the signed
    ///     `IntFloorDiv`/`IntMod`.
    ///
    /// The division intrinsics carry `ll_uint_py_div`'s `rhs != 0`
    /// precondition: the caller must guard the zero divisor, exactly as
    /// RPython's inlined `ll_uint_py_div_zer` wrapper does.
    fn lower_uint_intrinsic_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        enum UintEmit {
            /// A real unsigned resop recorded through `record_binop_i`.
            Opcode(&'static str),
            /// An oopspec residual call recorded through its own `record_*`.
            OopspecCall(&'static str),
        }
        let emit = match canonical_expr_segments(&call.func)?.last()?.as_str() {
            "majit_uint_lt" => UintEmit::Opcode("UintLt"),
            "majit_uint_le" => UintEmit::Opcode("UintLe"),
            "majit_uint_div" => UintEmit::OopspecCall("record_uint_py_div"),
            "majit_uint_mod" => UintEmit::OopspecCall("record_uint_py_mod"),
            "majit_int_py_div" => UintEmit::OopspecCall("record_int_py_div"),
            "majit_int_py_mod" => UintEmit::OopspecCall("record_int_py_mod"),
            _ => return None,
        };
        if call.args.len() != 2 {
            return None;
        }
        let lhs = self.lower_value_expr(&call.args[0])?;
        let rhs = self.lower_value_expr(&call.args[1])?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let (lhs_reg, rhs_reg) = (lhs.reg, rhs.reg);
        let dst = self.alloc_reg();
        let tokens = match emit {
            UintEmit::Opcode(opcode_name) => {
                let opcode = format_ident!("{opcode_name}");
                binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs_reg)
            }
            UintEmit::OopspecCall(method_name) => {
                let method = format_ident!("{method_name}");
                quote! { __builder.#method(#dst, #lhs_reg, #rhs_reg); }
            }
        };
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(dst)],
            ),
            tokens,
        );
        Some(Binding {
            reg: dst,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower a raw native-memory load intrinsic
    /// `majit_raw_load_{i,u}{8,16,32,64}(base, ea)` to a `raw_load_i` op (the
    /// read-side analogue of `lower_raw_store_stmt`).  RPython parity:
    /// `jtransform.py rewrite_op_raw_load` lowers a
    /// `rffi.raw_storage_getitem` to `raw_load_i(base, offset,
    /// arraydescrof(CArray(T)))`.  Both operands are int-kind (raw address,
    /// byte offset); the op writes an int result.
    fn lower_raw_load_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        let segments = canonical_expr_segments(&call.func)?;
        if segments.last()?.as_str() == "majit_raw_load_f" {
            if call.args.len() != 2 {
                return None;
            }
            let base = self.lower_value_expr(&call.args[0])?;
            let ea = self.lower_value_expr(&call.args[1])?;
            if !matches!(base.kind, BindingKind::Int) || !matches!(ea.kind, BindingKind::Int) {
                return None;
            }
            let (base_reg, ea_reg) = (base.reg, ea.reg);
            let dst = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(
                    OpKind::RawLoad,
                    vec![Register::int(base_reg), Register::int(ea_reg)],
                    vec![Register::float(dst)],
                ),
                quote! {
                    let __raw_descr = __builder.add_raw_float_array_descr();
                    __builder.raw_load_f(
                        #dst as u16,
                        #base_reg as u16,
                        #ea_reg as u16,
                        __raw_descr,
                    );
                },
            );
            return Some(Binding {
                reg: dst,
                kind: BindingKind::Float,
                depends_on_stack: base.depends_on_stack || ea.depends_on_stack,
                struct_type: None,
            });
        }
        let (item_size, is_signed) = match segments.last()?.as_str() {
            "majit_raw_load_i8" => (1usize, true),
            "majit_raw_load_u8" => (1usize, false),
            "majit_raw_load_i16" => (2usize, true),
            "majit_raw_load_u16" => (2usize, false),
            "majit_raw_load_i32" => (4usize, true),
            "majit_raw_load_u32" => (4usize, false),
            "majit_raw_load_i64" => (8usize, true),
            // At 8 bytes signed/unsigned load into an i64 register is
            // identical (no sign-extension gap), but the intrinsic is
            // documented as supported, so accept it for parity.
            "majit_raw_load_u64" => (8usize, false),
            _ => return None,
        };
        if call.args.len() != 2 {
            return None;
        }
        let base = self.lower_value_expr(&call.args[0])?;
        let ea = self.lower_value_expr(&call.args[1])?;
        if !matches!(base.kind, BindingKind::Int) || !matches!(ea.kind, BindingKind::Int) {
            return None;
        }
        let (base_reg, ea_reg) = (base.reg, ea.reg);
        let dst = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::RawLoad,
                vec![Register::int(base_reg), Register::int(ea_reg)],
                vec![Register::int(dst)],
            ),
            quote! {
                let __raw_descr = __builder.add_raw_int_array_descr_signed(#item_size, #is_signed);
                __builder.raw_load_i(
                    #dst as u16,
                    #base_reg as u16,
                    #ea_reg as u16,
                    __raw_descr,
                );
            },
        );
        Some(Binding {
            reg: dst,
            kind: BindingKind::Int,
            depends_on_stack: base.depends_on_stack || ea.depends_on_stack,
            struct_type: None,
        })
    }

    /// `inlined_prefix` names this field as the leading substructure, and the
    /// literal's type is that base (leaf names match; the body may spell an
    /// imported path while the attribute spells a crate path).
    fn field_is_inlined_prefix(
        &self,
        outer: &syn::Path,
        member: &syn::Member,
        inner: &syn::Path,
    ) -> bool {
        let Some(config) = self.config else {
            return false;
        };
        let Some(name) = named_member(member) else {
            return false;
        };
        let Some((field, base)) = crate::jit_interp::inlined_prefix_entry(
            &config.inlined_prefix,
            &super::canonical_path_segments(outer),
        ) else {
            return false;
        };
        if field != &name {
            return false;
        }
        let base_segs = super::canonical_path_segments(base);
        let inner_segs = super::canonical_path_segments(inner);
        inner_segs == base_segs || inner_segs.last() == base_segs.last()
    }

    /// This struct is the header of an `Elem in Header` array
    /// (`array_fields`), so a literal of it is `malloc_varsize`.
    ///
    /// Returns the header path the descr is built from (basesize =
    /// `offset_of!(Header, items)`, lendescr = the header's length field).
    fn varsize_header(&self, struct_path: &syn::Path) -> Option<syn::Path> {
        let config = self.config?;
        let segs = super::canonical_path_segments(struct_path);
        let last = segs.last()?;
        config
            .array_headers
            .values()
            .find(|header| {
                let header_segs = super::canonical_path_segments(header);
                header_segs == segs || header_segs.last() == Some(last)
            })
            .cloned()
    }

    /// `jtransform.py` `rewrite_op_malloc_varsize`: pointer elements and a
    /// zeroed block are `new_array_clear`; a non-pointer block that is not
    /// zeroed is `new_array`. The length argument is the header's length
    /// field (`descr.py` `get_field_arraylen_descr`).
    fn emit_varsize_array(
        &mut self,
        struct_path: &syn::Path,
        header: &syn::Path,
        fields: &[(syn::Path, syn::Member, Binding)],
        depends_on_stack: bool,
    ) -> Option<Binding> {
        let config = self.config?;
        let len_name = super::lower_vable::varsize_length_field_name(config, header)?;
        // `rewrite_op_malloc_varsize` allocates and stamps the length only.
        // Another initialised header field would need its own setfield;
        // decline the literal so the helper stays a residual call.
        if fields
            .iter()
            .any(|(_, member, _)| named_member(member).as_deref() != Some(len_name.as_str()))
        {
            return None;
        }
        let length = fields.iter().find(|(_, member, value)| {
            named_member(member).as_deref() == Some(len_name.as_str())
                && matches!(value.kind, BindingKind::Int)
        })?;
        let length_reg = length.2.reg;
        let pointer_items = super::lower_vable::header_items_are_pointers(config, header);
        // `rewrite_op_malloc_varsize`: a pointer or struct element is
        // `new_array_clear`. A primitive element is `new_array` unless the
        // malloc asked for `zero` (the empty `items: []` of a pointer
        // block is the clear arm above).
        let result_reg = self.alloc_reg();
        let descr = super::lower_vable::gc_varsize_descr_tokens(config, header);
        let op_tokens = if pointer_items {
            quote! {
                let __descr_idx = #descr;
                __builder.new_array_clear(#result_reg, #length_reg, __descr_idx);
            }
        } else {
            quote! {
                let __descr_idx = #descr;
                __builder.new_array(#result_reg, #length_reg, __descr_idx);
            }
        };
        self.emit_op(
            OpMeta::linear(
                OpKind::New,
                vec![Register::int(length_reg)],
                vec![Register::ref_(result_reg)],
            ),
            op_tokens,
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Ref,
            depends_on_stack,
            struct_type: Some(struct_path.clone()),
        })
    }

    /// Lower a struct literal `Path { f0: v0, f1: v1, .. }` to a JIT
    /// allocation plus per-field stores: `new` (size from `size_of`) then
    /// `setfield_gc_<kind>` at each field's `offset_of`.  Mirrors
    /// `jtransform.py` `rewrite_op_malloc` + `rewrite_op_setfield`; the
    /// optimizer's virtualize pass folds the New away when the struct does
    /// not escape. A float field is `setfield_gc_f` (`getkind` `f`).
    /// Returns the result ref binding, or `None` (helper falls back to a
    /// residual call) for struct-update base syntax.
    fn lower_struct_value(&mut self, s: &syn::ExprStruct) -> Option<Binding> {
        if s.rest.is_some() {
            return None;
        }
        let struct_path = &s.path;
        // An inlined leading substructure is not its own allocation.
        // `lltype.py` admits it only at `_names[0]` (offset 0); its fields
        // are stores into the outer object. A constant `ob_type` / `typeptr`
        // there is the vtable `rewrite_op_malloc` reads, not a setfield
        // (`heaptracker.all_fielddescrs` drops `typeptr`).
        let mut flat: Vec<(syn::Path, syn::Member, syn::Expr)> = Vec::new();
        for field in &s.fields {
            if let syn::Expr::Struct(inner) = &field.expr
                && self.field_is_inlined_prefix(struct_path, &field.member, &inner.path)
            {
                for inner_field in &inner.fields {
                    flat.push((
                        inner.path.clone(),
                        inner_field.member.clone(),
                        inner_field.expr.clone(),
                    ));
                }
            } else {
                flat.push((
                    struct_path.clone(),
                    field.member.clone(),
                    field.expr.clone(),
                ));
            }
        }
        let mut vtable: Option<TokenStream> = None;
        let mut value_fields: Vec<(syn::Path, syn::Member, Binding)> = Vec::new();
        let mut depends_on_stack = false;
        let varsize = self.varsize_header(struct_path);
        for (owner, member, expr) in &flat {
            if Self::is_typeptr_member(member)
                && let Some(tokens) = Self::const_vtable_tokens(expr)
            {
                if vtable.is_some() {
                    return None;
                }
                vtable = Some(tokens);
                continue;
            }
            // The inlined `items` array is the varsize payload, not a
            // field store. `rewrite_op_malloc_varsize` consumes the
            // length and emits `new_array` / `new_array_clear`.
            if varsize.is_some() && named_member(member).as_deref() == Some("items") {
                continue;
            }
            let value = self.lower_value_expr(expr)?;
            depends_on_stack |= value.depends_on_stack;
            value_fields.push((owner.clone(), member.clone(), value));
        }
        let fields = value_fields;
        if let Some(header) = varsize {
            // A varsize allocation has no typeptr store to carry a vtable.
            if vtable.is_some() {
                return None;
            }
            return self.emit_varsize_array(struct_path, &header, &fields, depends_on_stack);
        }
        // Same id a later `getfield` mints (`struct_gc_kind_is_managed`), so
        // the virtual's slot and the read name one descriptor.
        let gc_managed = self
            .config
            .is_some_and(|cfg| cfg.struct_gc_kind_is_managed(struct_path));
        let type_id = struct_type_id_tokens(struct_path, gc_managed);
        let result_reg = self.alloc_reg();
        // descr.py init_size_descr: the SizeDescr carries the
        // struct's full `(offset, is_ref, name)` layout so the optimizer can
        // virtualize the New and resolve each field's parent SizeDescr +
        // index. A Rust struct literal may list fields in any order, so the
        // builder canonicalizes `index_in_parent` by byte offset; we only
        // need to hand it each field's `(offset, is_ref, name)`.
        let field_layout: Vec<TokenStream> = fields
            .iter()
            .map(|(owner, member, value)| {
                let is_ref = matches!(value.kind, BindingKind::Ref);
                // `descr.py` `get_type_flag`: the value's kind is the field's
                // lltype. A float binding is `FLOAT` at registration.
                let is_float = matches!(value.kind, BindingKind::Float);
                // `rewrite_op_malloc` + `rewrite_op_setfield` register
                // each field through `fielddescrof`, which reads width
                // and signedness from FIELDTYPE. Use the same
                // `int_fields` / `ref_fields` witness the getfield path
                // already consults.
                let struct_name = owner
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let member_name = named_member(member).unwrap_or_default();
                let key = format!("{struct_name}::{member_name}");
                let (size, signed, witness) = match self.config {
                    Some(config) => {
                        super::lower_vable::field_scalar_tokens(config, &key, owner, member)
                    }
                    None => (
                        quote! { ::core::mem::size_of::<i64>() },
                        quote! { true },
                        quote! {},
                    ),
                };
                // A prefix field sits at offset 0 of the outer struct, so
                // its own `offset_of` is the outer offset.
                quote! {{
                    #witness
                    (
                        ::core::mem::offset_of!(#owner, #member),
                        #is_ref,
                        stringify!(#member),
                        #size,
                        #signed,
                        #is_float,
                    )
                }}
            })
            .collect();
        let headerless = self
            .config
            .is_some_and(|cfg| cfg.is_headerless_struct(struct_path));
        let alloc = if let Some(vtable) = &vtable {
            quote! {
                __builder.new_with_vtable_struct(
                    #result_reg,
                    ::core::mem::size_of::<#struct_path>(),
                    #type_id,
                    #vtable,
                    #headerless,
                    #gc_managed,
                    &[ #(#field_layout),* ],
                    {
                        #[allow(unused_imports)]
                        use majit_metainterp::MajitImmutableFields as _;
                        <#struct_path>::__MAJIT_IMMUTABLE_FIELDS
                    },
                );
            }
        } else {
            quote! {
                __builder.new_struct(
                    #result_reg,
                    ::core::mem::size_of::<#struct_path>(),
                    #type_id,
                    #headerless,
                    &[ #(#field_layout),* ],
                    {
                        // The struct's own `_immutable_fields_` declaration.  Read through
                        // an in-scope trait rather than a qualified
                        // `<T as MajitImmutableFields>::...`, which would select the blanket
                        // empty default and never see a struct's own declaration.  The import
                        // is unused exactly when the inherent const wins, which is the common
                        // case.
                        #[allow(unused_imports)]
                        use majit_metainterp::MajitImmutableFields as _;
                        <#struct_path>::__MAJIT_IMMUTABLE_FIELDS
                    },
                );
            }
        };
        self.emit_op(
            OpMeta::linear(OpKind::New, vec![], vec![Register::ref_(result_reg)]),
            alloc,
        );
        for (owner, member, value) in fields.iter() {
            let value_reg = value.reg;
            let (reads, tokens) = match value.kind {
                BindingKind::Int => (
                    vec![Register::ref_(result_reg), Register::int(value_reg)],
                    quote! {
                        __builder.setfield_gc_i(
                            #result_reg,
                            #value_reg,
                            ::core::mem::offset_of!(#owner, #member),
                            #type_id,
                            stringify!(#member),
                        );
                    },
                ),
                BindingKind::Ref => (
                    vec![Register::ref_(result_reg), Register::ref_(value_reg)],
                    quote! {
                        __builder.setfield_gc_r(
                            #result_reg,
                            #value_reg,
                            ::core::mem::offset_of!(#owner, #member),
                            #type_id,
                            stringify!(#member),
                        );
                    },
                ),
                // `jtransform.py` `rewrite_op_setfield`: `kind = getkind(RESULT)[0]`
                // is `f`, so the store is `setfield_gc_f`. The layout entry
                // already named this field `FLOAT` (`descr.py` `get_type_flag`).
                BindingKind::Float => (
                    vec![Register::ref_(result_reg), Register::float(value_reg)],
                    quote! {
                        __builder.setfield_gc_f(
                            #result_reg,
                            #value_reg,
                            ::core::mem::offset_of!(#owner, #member),
                            #type_id,
                            stringify!(#member),
                        );
                    },
                ),
            };
            self.emit_op(OpMeta::linear(OpKind::SetfieldGc, reads, vec![]), tokens);
        }
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Ref,
            depends_on_stack,
            struct_type: Some(struct_path.clone()),
        })
    }

    fn is_typeptr_member(member: &syn::Member) -> bool {
        matches!(
            named_member(member).as_deref(),
            Some("ob_type") | Some("typeptr")
        )
    }

    /// `&PATH` — a compile-time class address. `rewrite_op_malloc` only emits
    /// `new_with_vtable` when the vtable is static.
    fn const_vtable_tokens(expr: &syn::Expr) -> Option<TokenStream> {
        let syn::Expr::Reference(reference) = expr else {
            return None;
        };
        if reference.mutability.is_some() {
            return None;
        }
        if !matches!(reference.expr.as_ref(), syn::Expr::Path(_)) {
            return None;
        }
        Some(quote! { (#expr as *const _ as usize) })
    }

    /// Statement-context lowering for `hint(x, promote=True)`:
    ///
    /// - `x = promote(arg);` (plain local re-assignment — `lower_state_
    ///   field_write` falls through because LHS isn't a `state.foo` field;
    ///   without this site `stmt_modifies_jit_state` returns false and
    ///   `lower_stmt`'s config-aware fallback silently consumes the stmt
    ///   without emitting the guard).
    /// - `promote(x);` (bare statement-form — same fall-through path as
    ///   plain assign).
    ///
    /// In both forms the actual guard emission is delegated to
    /// `lower_promote_call` via `lower_value_expr` so the resulting op
    /// shape (`-live-` + `<kind>_guard_value`) is identical to the
    /// value-context lowering.  For assignment form, mirror
    /// jtransform.py:613-615: the `None` sentinel means the result is
    /// considered equal to arg0, so the LHS aliases the RHS binding
    /// (`x = promote(y)` makes `x` read from y's register).
    /// Reassign a local variable that already has a binding: `pc = expr`.
    /// Lowers the RHS via `lower_value_expr`. On a straight-line lowerer the
    /// name is rebound to the RHS register. Across a join (`join_merge`,
    /// from `FlowContext.mergeblock`) the enclosing register is the one
    /// variable, so the RHS is moved into it (`GraphFlattener.insert_renamings`)
    /// and the name keeps that register.
    /// RPython parity: the flat dispatch's `pc = target; continue` updates
    /// the JitCode's pc register in-place (`flatten.py` emits a goto to the
    /// bytecode label, not an SSA rename, but majit's binding model is
    /// SSA-like — rebinding the name achieves the same effect because
    /// subsequent reads of `pc` pick up the new register).
    pub(super) fn lower_local_reassign(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Path(lhs_path) = &*assign.left else {
            return None;
        };
        let lhs_ident = lhs_path.path.get_ident()?.to_string();
        if !self.bindings.contains_key(&lhs_ident) {
            return None;
        }
        if lhs_ident == "pc" && !self.pc_pinned {
            return None;
        }
        let lhs = self.bindings.get(&lhs_ident)?.clone();
        // Still the enclosing binding, not a `let` that shadowed it. The
        // join reads this register; a kind change has no single link
        // argument (`FrameState.union`).
        let merges = self.join_merge.get(&lhs_ident).copied() == Some(lhs.reg);
        // Same refusal as `lower_local_update`: in a sub-JitCode a green is a
        // caller local threaded through the merge point, and writing this
        // body's register would not carry the value back. Refuse before the
        // RHS emits. An inline dispatch arm (`pc_pinned`) writes the
        // merge-point register itself, which `can_enter_jit!` then reads, so
        // a green there merges like `pc`.
        if merges
            && !self.pc_pinned
            && let Some(config) = self.config
            && super::lower_stmt::green_idents(config).contains(&lhs_ident)
        {
            self.body_failure_reason = Some(format!(
                "arm body loop-carried green `{lhs_ident}` reassignment in a \
                 non-inlined arm (lower_local_reassign): the header register \
                 would stay stale"
            ));
            return None;
        }
        let binding = self.lower_value_expr(&assign.right)?;
        if merges {
            if lhs.kind != binding.kind {
                return None;
            }
            if lhs.reg != binding.reg {
                let dst = lhs.reg;
                let src = binding.reg;
                let (op_kind, tokens) = match binding.kind {
                    BindingKind::Int => (OpKind::MoveI, quote! { __builder.move_i(#dst, #src); }),
                    BindingKind::Ref => (OpKind::MoveR, quote! { __builder.move_r(#dst, #src); }),
                    BindingKind::Float => (OpKind::MoveF, quote! { __builder.move_f(#dst, #src); }),
                };
                let register = Register::new(binding.kind, dst);
                self.emit_op(
                    OpMeta::linear(
                        op_kind,
                        vec![Register::from_binding(&binding)],
                        vec![register],
                    ),
                    tokens,
                );
            }
            let mut kept = lhs;
            kept.struct_type = binding.struct_type;
            self.bindings.insert(lhs_ident, kept);
            return Some(());
        }
        self.bindings.insert(lhs_ident, binding);
        Some(())
    }

    /// Lower a compound update of an existing local into that local's
    /// physical register: `need -= 1`, `target += offset`, and so on.
    ///
    /// This must not allocate and rebind an SSA-like result. A local carried
    /// around a translated `while` back-edge is read from the register that
    /// existed when the loop header was emitted; changing the binding after
    /// the body has been lowered would leave the header reading the stale
    /// value. RPython's `BrainInterpreter.interp_char` bracket scans depend on
    /// these loop-carried updates.
    pub(super) fn lower_local_update(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Binary(binary) = expr else {
            return None;
        };
        let Expr::Path(lhs_path) = &*binary.left else {
            return None;
        };
        let lhs_ident = lhs_path.path.get_ident()?.to_string();
        // An unpinned `pc` write in a sub-JitCode updates the callee copy
        // and never reaches the merge-point register (`BC_INLINE_CALL`
        // copies caller→callee only). Refuse so the arm degrades instead
        // of compiling a stale-pc loop. `pc_pinned` (inline dispatch)
        // intercepts these writes first.
        if lhs_ident == "pc" && !self.pc_pinned {
            return None;
        }
        // `lower_expr_stmt` reaches this before `lower_stmt_fallback`, so the
        // green-write refusal there never sees the statement.  A green is a
        // caller local threaded through the merge point: writing it into this
        // body's own register drops the advance instead of carrying it back,
        // and the arm then records, compiles and answers with a stale green.
        // The sub-JitCode paths that legitimately advance a green take it
        // through `pc_return_increment`, which removes the statement before
        // lowering, so refusing here cannot reach them.
        if let Some(config) = self.config
            && super::lower_stmt::green_idents(config).contains(&lhs_ident)
        {
            return None;
        }
        let lhs = self.bindings.get(&lhs_ident)?.clone();
        let (op_kind, opcode) = match lhs.kind {
            BindingKind::Int => (OpKind::BinopI, opcode_for_assign_binop(&binary.op)?),
            BindingKind::Float => (OpKind::BinopF, opcode_for_assign_binop_f(&binary.op)?),
            BindingKind::Ref => return None,
        };
        if matches!(lhs.kind, BindingKind::Int)
            && let Some(value) = self.int_constant_operand(&binary.right)
            && let Some(tokens) = binop_i_const_emit_tokens(lhs.reg, &opcode, lhs.reg, &value)
        {
            let register = Register::int(lhs.reg);
            self.emit_op(
                OpMeta::linear(op_kind, vec![register], vec![register]),
                tokens,
            );
            return Some(());
        }
        let rhs = self.lower_value_expr(&binary.right)?;
        if lhs.kind != rhs.kind {
            return None;
        }

        let tokens = match lhs.kind {
            BindingKind::Int => binop_i_emit_tokens(lhs.reg, &opcode, lhs.reg, rhs.reg),
            BindingKind::Float => binop_f_emit_tokens(lhs.reg, &opcode, lhs.reg, rhs.reg),
            BindingKind::Ref => return None,
        };
        let register = Register::new(lhs.kind, lhs.reg);
        self.emit_op(
            OpMeta::linear(
                op_kind,
                vec![register, Register::from_binding(&rhs)],
                vec![register],
            ),
            tokens,
        );
        Some(())
    }

    pub(super) fn lower_promote_stmt(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Assign(assign) => {
                let Expr::Path(lhs_path) = &*assign.left else {
                    return None;
                };
                let lhs_ident = lhs_path.path.get_ident()?.to_string();
                let Expr::Call(call) = &*assign.right else {
                    return None;
                };
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                let binding = self.lower_value_expr(&assign.right)?;
                self.bindings.insert(lhs_ident, binding);
                Some(())
            }
            Expr::Call(call) => {
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                self.lower_value_expr(expr)?;
                Some(())
            }
            _ => None,
        }
    }

    /// Statement-context lowering for `jit::assert_not_none(x);`.
    ///
    /// pyjitpl.py `opimpl_assert_not_none`. Bare statement form
    /// discards the unwrapped value but still emits the
    /// `BC_ASSERT_NOT_NONE` op so the trace records the nullity hint.
    /// The `let x = jit::assert_not_none(y);` form goes through the
    /// value-context lowerer instead (lower_value_expr →
    /// lower_assert_not_none_call); this stmt site catches only the
    /// bare-call shape.
    pub(super) fn lower_assert_not_none_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        if !is_assert_not_none_call_path(&call.func) {
            return None;
        }
        self.lower_value_expr(expr)?;
        Some(())
    }

    /// Statement-context lowering for `jit::record_exact_class(value, cls);`.
    ///
    /// pyjitpl.py `opimpl_record_exact_class`. RPython
    /// `rlib/jit.py record_exact_class` is a void hint, so this is
    /// the only emission shape — there is no value-form (the runtime
    /// stub in `jit.rs` returns `()`).
    ///
    /// `value` must be Ref-kind and `cls` must be Int-kind, matching
    /// `blackhole.py @arguments("r", "i")`.  Non-matching kinds
    /// silently skip per `pyjitpl.py MIFrame.opimpl_record_exact_class if isinstance(clsbox, Const):` —
    /// dispatch-time `trace_record_exact_class` also gates on
    /// `cls_const.is_constant()`.
    pub(super) fn lower_record_exact_class_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        if !is_record_exact_class_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 2 {
            return None;
        }
        let value_binding = self.lower_value_expr(&call.args[0])?;
        let cls_binding = self.lower_value_expr(&call.args[1])?;
        if !matches!(value_binding.kind, BindingKind::Ref) {
            return None;
        }
        if !matches!(cls_binding.kind, BindingKind::Int) {
            return None;
        }
        let value_reg = value_binding.reg;
        let cls_reg = cls_binding.reg;
        // jtransform.py rewrite_op_jit_record_exact_class emits a single SpaceOperation with no
        // preceding `-live-` annotation (-live- is only attached to
        // promote, jtransform.py:611).
        self.emit_op(
            OpMeta::linear(
                OpKind::RecordExactClass,
                vec![Register::ref_(value_reg), Register::int(cls_reg)],
                vec![],
            ),
            quote! { __builder.record_exact_class(#value_reg, #cls_reg); },
        );
        Some(())
    }

    /// Lower `promote(x)` → emit `-live-` + `<kind>_guard_value(x_reg)`,
    /// return x binding.
    ///
    /// RPython: `hint(x, promote=True)` rewrites to a `-live-` marker
    /// (jtransform.py:611) immediately followed by `int_guard_value(x)`
    /// (jtransform.py:612).  The leading `-live-` pins the per-marker
    /// liveness triple at the source position so the resume protocol can
    /// rebuild the live frame state if the guard fails.  Without this
    /// pair, the snapshot path falls back to the canonical "everything-
    /// alive" entry — correct for blackhole resume but a per-marker
    /// liveness parity loss vs RPython.
    ///
    /// Blackhole: no-op (the live marker is metadata, the guard a no-op
    /// at non-trace time).  Tracing: emits GUARD_VALUE to specialize on
    /// current value with the per-pc live set saved into all_liveness.
    ///
    /// Recognizes: `assert_not_none(x)`, `jit::assert_not_none(x)`.
    ///
    /// pyjitpl.py `opimpl_assert_not_none(box)`. Emits
    /// `BC_ASSERT_NOT_NONE`; trace-time dispatcher routes through
    /// `TraceCtx::trace_assert_not_none` which gates on
    /// `heap_cache.is_nullity_known` + bumps `HEAPCACHED_OPS` on cache
    /// hit. Only fires for ref-typed bindings — `jit::assert_not_none<T>`
    /// is documented as the `Option<T>::expect` analog at the runtime
    /// stub (`jit.rs`), so non-ref bindings (int/float) cannot
    /// reach this site through `Option<T>` unwrap.
    fn lower_assert_not_none_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if !is_assert_not_none_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        let binding = self.lower_value_expr(&call.args[0])?;
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let reg = binding.reg;
        // jtransform.py rewrite_op_debug_assert_not_none emits a single SpaceOperation with no
        // preceding `-live-` annotation (-live- is only attached to
        // promote, jtransform.py:611).
        self.emit_op(
            OpMeta::linear(OpKind::AssertNotNone, vec![Register::ref_(reg)], vec![]),
            quote! { __builder.assert_not_none(#reg); },
        );
        Some(binding)
    }

    /// `<IntTy>::from_le_bytes([e0, e1, ...])` / `from_be_bytes`.
    ///
    /// The byte-load, the `as T` cast, the shift and the or already lower;
    /// only this array-literal associated-function spelling was unrecognised.
    /// Desugar to the equivalent shift/or chain and hand it back to
    /// `lower_value_expr`. A mismatched element count is not a guess, so
    /// decline rather than invent a reconstruction.
    ///
    /// The chain is assembled in the UNSIGNED sibling of the target and cast
    /// once at the end. Only the casts narrow: `lower_int_cast` sign-extends
    /// or masks, while the shift and the or are lowered in the machine-word
    /// bank and keep every bit. Assembling `i16` directly would therefore
    /// diverge from Rust, which evaluates `(e1 as i16) << 8` in `i16` and
    /// wraps whatever reaches the sign bit -- `i16::from_le_bytes([0xff,
    /// 0xff])` is `-1` there and `65535` in a chain that never narrows. No
    /// intermediate overflows in the unsigned sibling, because a byte shifted
    /// by at most `width - 8` still fits it, so the one trailing cast is the
    /// only narrowing the value needs.
    fn lower_from_endian_bytes_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if call.args.len() != 1 {
            return None;
        }
        let info = super::helpers::primitive_from_endian_bytes_path(&call.func)?;
        let little_endian = info.little_endian;
        let ty_name = info.ty_name;
        let byte_width = info.byte_width;
        let acc_ty_name = info.acc_ty_name;
        let Expr::Array(array) = &call.args[0] else {
            return None;
        };
        if array.elems.len() != byte_width {
            return None;
        }
        let ty = syn::Ident::new(ty_name, proc_macro2::Span::call_site());
        let acc_ty = syn::Ident::new(acc_ty_name, proc_macro2::Span::call_site());
        let n = array.elems.len();
        let mut acc: Option<Expr> = None;
        for (i, elem) in array.elems.iter().enumerate() {
            let amount = if little_endian {
                8 * i
            } else {
                8 * (n - 1 - i)
            };
            let term: Expr = if amount == 0 {
                syn::parse_quote! { (#elem as #acc_ty) }
            } else {
                let shift = syn::LitInt::new(&amount.to_string(), proc_macro2::Span::call_site());
                syn::parse_quote! { ((#elem as #acc_ty) << #shift) }
            };
            acc = Some(match acc {
                None => term,
                Some(prev) => syn::parse_quote! { #prev | #term },
            });
        }
        let combined = acc?;
        let narrowed: Expr = syn::parse_quote! { ((#combined) as #ty) };
        self.lower_value_expr(&narrowed)
    }

    /// Recognizes: `promote(x)`, `hint_promote(x)`, `jit::promote(x)`.
    fn lower_promote_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if !is_promote_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        let binding = self.lower_value_expr(&call.args[0])?;
        let reg = binding.reg;
        // jtransform.py:611 — emit `-live-` before the guard op so the
        // codewriter's per-marker liveness analysis records the alive
        // set at this CFG position.  `live_placeholder_with_triple` will
        // patch the BC_LIVE 2-byte slot to the dedup'd offset at
        // `finalize_liveness` time.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { __builder.live_placeholder(); },
        );
        match binding.kind {
            BindingKind::Int => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::int(reg)], vec![]),
                    quote! { __builder.int_guard_value(#reg); },
                );
            }
            BindingKind::Ref => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::ref_(reg)], vec![]),
                    quote! { __builder.ref_guard_value(#reg); },
                );
            }
            BindingKind::Float => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::float(reg)], vec![]),
                    quote! { __builder.float_guard_value(#reg); },
                );
            }
        }
        Some(binding)
    }

    /// The struct `call_returns` declares for `func`'s result.
    ///
    /// A ref-returning call produces a ref binding, and the `result.field`
    /// after it needs a struct to resolve the field against —
    /// `lower_ref_binding_getfield` reads exactly this and bails without it,
    /// which takes the whole dispatch arm down. The concrete path reads the
    /// same declaration through `RefFieldRewriter`, so a binding that misses
    /// it still computes the right answer and the loss shows up only as a
    /// degraded arm.
    fn declared_return_struct(&self, func: &Expr) -> Option<syn::Path> {
        let segments = canonical_expr_segments(func)?;
        self.config?.call_returns.get(&segments).cloned()
    }

    /// `ll_strlen(s)` / `ll_strgetitem(s, i)` on a `str` red.
    ///
    /// Real functions so the interpreter fallback compiles; the trace records
    /// `OpCode::Strlen` / `OpCode::Strgetitem` instead of a residual call.
    /// `jtransform.py` `rewrite_op_getinteriorfield` emits the same ops for
    /// `rstr.STR`.
    fn lower_rstr_call(&mut self, call: &ExprCall) -> Option<Binding> {
        let segs = canonical_expr_segments(&call.func)?;
        let name = segs.last()?.as_str();
        match (name, call.args.len()) {
            ("ll_strlen", 1) => {
                let arg = self.lower_value_expr(&call.args[0])?;
                if arg.kind != BindingKind::Ref {
                    return None;
                }
                let reg = self.alloc_reg();
                let src = arg.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        vec![Register::ref_(src)],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.strlen(#reg, #src); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: arg.depends_on_stack,
                    struct_type: None,
                })
            }
            ("ll_strgetitem", 2) => {
                let string = self.lower_value_expr(&call.args[0])?;
                let index = self.lower_value_expr(&call.args[1])?;
                if string.kind != BindingKind::Ref || index.kind != BindingKind::Int {
                    return None;
                }
                let reg = self.alloc_reg();
                let src = string.reg;
                let idx = index.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        vec![Register::ref_(src), Register::int(idx)],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.strgetitem(#reg, #src, #idx); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: string.depends_on_stack || index.depends_on_stack,
                    struct_type: None,
                })
            }
            _ => None,
        }
    }

    pub(super) fn lower_call_value(&mut self, call: &ExprCall) -> Option<Binding> {
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        let mut depends_on_stack = false;
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding.clone());
            depends_on_stack |= binding.depends_on_stack;
        }

        let reg = self.alloc_reg();
        let func = &call.func;
        let arg_kinds: Vec<BindingKind> = arg_bindings.iter().map(|b| b.kind).collect();
        let word_result_addr = word_result_addr_tokens(func, &arg_kinds);
        let mut result_kind = BindingKind::Int;
        // jtransform.py:467-471 / 480-482: `-live-` follows the call, it does
        // not precede it.  Decide here whether the explicit arm below needs a
        // trailing marker; the inline arms emit their own (`post_live`) and
        // the inferred arm emits a runtime-conditional one, so both report
        // `false` and are excluded.
        let post_live_after_call = match &policy {
            CallPolicySpec::Explicit(kind) => explicit_call_emits_post_live(*kind),
            CallPolicySpec::Infer => false,
        };
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let canonical_call = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            quote! { residual_call_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceInt => {
                            quote! { call_may_force_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                            quote! { call_loopinvariant_int_canonical_via_target }
                        }
                        _ => unreachable!(),
                    };
                    // A declared `residual_writes` mutator that RETURNS a
                    // used value routes through this value-call path; it must
                    // still carry the field write-set `EffectInfo` so the
                    // optimizer invalidates the cached `getfield_gc_i` after
                    // the call — exactly as the statement form does.  Without
                    // it a stack-mutating residual (e.g. `jit_pop_is_zero`
                    // feeding a branch) leaves the cached `selected.size`
                    // stale and loop-peel const-folds it.  Only the residual
                    // policy qualifies (may-force / loop-invariant carry their
                    // own effects).
                    let write_ei = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            self.residual_write_effect_info_tokens(func, true)
                        }
                        _ => None,
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #reg,
                                );
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                Register::ints(&arg_regs),
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                                #call_invocation
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(__fn_idx, #typed_args, #reg);
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                __arg_regs,
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                                #call_invocation
                            },
                        );
                    }
                }
                // `call.py:303` non-elidable EF_CANNOT_RAISE int — explicit policy.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // See the stmt-form
                    // ElidableInt arm earlier in this file for the
                    // canonical migration rationale and the 3-way
                    // `_canraise(op)` pick from `call.py`.
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            #call_stmt
                        },
                    );
                }
                // Non-wrapped ref-returning call (value position).
                crate::jit_interp::CallPolicyKind::ResidualRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // The register literals must be spelled inside the
                    // `__builder` call itself: `regalloc`'s rewriter recolors
                    // only the arguments of a non-aux builder method call, so
                    // arguments bound to a local first would keep the register
                    // numbers the lowerer handed out before coloring.
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #reg);
                        },
                    );
                    // jtransform.py:467-470 — a residual ref call can raise, so
                    // the trailing `-live-` follows it exactly as the shared
                    // path below emits for the other explicit residual arms.
                    // This arm returns early (to attach `struct_type`), so it
                    // cannot fall through to that shared emission and must emit
                    // the marker itself.
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type: self.declared_return_struct(func),
                    });
                }
                crate::jit_interp::CallPolicyKind::AllocRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #reg, majit_metainterp::can_raise_effect_info());
                        },
                    );
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type: self.declared_return_struct(func),
                    });
                }
                crate::jit_interp::CallPolicyKind::StrConcat => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let ref_word = assert_ref_result_one_word(func, arg_bindings.len());
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            #ref_word
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #reg, majit_metainterp::str_concat_effect_info());
                        },
                    );
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type: self.declared_return_struct(func),
                    });
                }
                crate::jit_interp::CallPolicyKind::NurseryAllocRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // The register literals must be spelled inside the
                    // `__builder` call itself: `regalloc`'s rewriter recolors
                    // only the arguments of a non-aux builder method call, so
                    // arguments bound to a local first would keep the register
                    // numbers the lowerer handed out before coloring.
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #reg, majit_metainterp::nursery_alloc_effect_info());
                        },
                    );
                    // jtransform.py:467-470 — a residual ref call can raise, so
                    // the trailing `-live-` follows it exactly as the shared
                    // path below emits for the other explicit residual arms.
                    // This arm returns early (to attach `struct_type`), so it
                    // cannot fall through to that shared emission and must emit
                    // the marker itself.
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type: self.declared_return_struct(func),
                    });
                }
                crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceFloatWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        _ => unreachable!(),
                    };
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::InlineInt
                | crate::jit_interp::CallPolicyKind::InlineRef
                | crate::jit_interp::CallPolicyKind::InlineFloat => {
                    result_kind = binding_kind_for_inline_policy(kind).unwrap();
                    let shared_path = inline_shared_path(&call.func)?;
                    let sub_return_kind = jit_arg_kind_tokens(result_kind);
                    let prebuild_path = inline_prebuild_path(&call.func)?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                    let __arg_regs = inline_call_read_regs(&arg_bindings);
                    // RPython `pyjitpl.py finish_setup` order: the
                    // helper's per-marker `-live-` triples must land in
                    // `asm.all_liveness` before the parent's
                    // `JitDriver::install_canonical_liveness` snapshot.
                    // Queue the helper's `__majit_inline_jitcode_<name>_
                    // prebuild(__asm)` call into the parent's prebuild
                    // accumulator; `liveness_prebuild_tokens` will splice
                    // it ahead of the parent arm's own
                    // `__asm._register_liveness_offset` calls.
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __sub_edge = #shared_path(__asm);
                            let __sub_return_kind = #sub_return_kind;
                            let __sub_idx = __builder.add_sub_jitcode_ref(__sub_edge);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                crate::jit_interp::CallPolicyKind::InlinePipelineInt
                | crate::jit_interp::CallPolicyKind::InlinePipelineRef
                | crate::jit_interp::CallPolicyKind::InlinePipelineFloat => {
                    result_kind = binding_kind_for_inline_policy(kind).unwrap();
                    // Resolve the callee by name through the host's
                    // `__majit_pipeline_jitcode`, which returns the
                    // pipeline-built (`make_jitcodes()`, codewriter.py)
                    // sub-jitcode shell. Its build-time liveness table is
                    // installed through the host prebuild hook before the
                    // driver snapshots the shared assembler table.
                    let pipeline_name = match &*call.func {
                        syn::Expr::Path(ep) => ep.path.segments.last().map(|s| s.ident.to_string()),
                        _ => None,
                    }?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                    let __arg_regs = inline_call_read_regs(&arg_bindings);
                    self.inline_liveness_prebuild.push(quote! {
                        __majit_pipeline_liveness_prebuild(__asm);
                    });
                    let untyped_return = self.inference_failure_tokens(
                        "pipeline jitcode must end in a typed return opcode",
                    );
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            use majit_metainterp::jitcode::JitCodeRuntimeExt as _;
                            let __sub_jitcode = __majit_pipeline_jitcode(#pipeline_name);
                            // The callee is resolved by name at install time,
                            // so "does it end in a typed return" is not known
                            // until then. A body that does not is a body the
                            // parent's INLINE_CALL has no return kind for, and
                            // `make_jitcodes()` installs only completed
                            // jitcodes -- so the parent declines rather than
                            // registering a half-typed body. Under
                            // `InferenceFailureMode::Panic` the enclosing
                            // builder has no `None` to return and this stays a
                            // panic, as every other rejection there does.
                            let (__sub_return_kind, _) =
                                match __sub_jitcode.trailing_return_info() {
                                    Some(__info) => __info,
                                    None => {
                                        #untyped_return
                                    }
                                };
                            let __sub_idx = __builder.add_sub_jitcode_arc(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                let int_arg_regs = int_arg_regs(&arg_bindings);
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy only supports int-return value calls here; use an explicit inline_ref/inline_float or *_ref_wrapped/*_float_wrapped policy",
                );
                // RPython `codewriter.py:55` precomputes per-helper
                // `-live-` triples and `pyjitpl.py finish_setup`
                // snapshots `metainterp_sd.liveness_info` after every
                // helper has had its triples registered.  When the
                // inferred path's runtime policy resolves to `4u8`
                // (`call.py` `EF_INLINE_HELPER`), the inline-helper
                // builder at line 4u8's runtime arm below executes
                // `__inline_builder(__asm)` which materialises a sub-
                // JitCode containing the helper's `-live-` markers.
                // Without an install-time prebuild call queuing into
                // `inline_liveness_prebuild`, the helper's per-marker
                // triples never enter `asm.all_liveness` ahead of the
                // parent driver's `install_canonical_liveness`
                // snapshot — diverging from the
                // `codewriter.py:55 → finish_setup` order.
                //
                // Queue the prebuild path when the helper exposes one
                // (only inline-able helpers do).  At runtime, the
                // prebuild call is idempotent — it registers triples
                // even when the runtime-resolved policy is non-inline,
                // matching the explicit `InlineInt/InlineRef/InlineFloat`
                // arms above which queue unconditionally.  Non-inline
                // helpers don't expose a `inline_prebuild_path` symbol,
                // so the `.ok()` swallow here is the parity-correct
                // "no triples to register" signal.
                if let Some(prebuild_path) = inline_prebuild_path(&call.func) {
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                }
                let __arg_regs = inline_call_read_regs(&arg_bindings);
                let __slot_tokens = CondCallEffectSlot::slot_from_policy_tokens();
                if let Some(_arg_regs) = int_arg_regs {
                    // Inferred path: result is Int (the explicit-Inline case is
                    // handled at line 4u8 of the runtime match below, but the
                    // emit-time OpMeta destination still tracks the call's
                    // post-condition register slot — Int matches the typed-call
                    // helper that produces it).
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs.clone(),
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #word_result_addr
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                    // The policy byte hands over `_shared`, not `_with_asm`:
                                    // the cache probe is what a recursive helper needs.
                                    let __builder_fn: fn(&mut majit_metainterp::Assembler)
                                        -> majit_metainterp::jitcode::InlineJitCodeRef =
                                        unsafe { std::mem::transmute(__inline_builder) };
                                    let __sub_edge = __builder_fn(__asm);
                                    let __sub_return_kind = majit_metainterp::JitArgKind::Int;
                                    let __sub_idx = __builder.add_sub_jitcode_ref(__sub_edge);
                                    #inline_call
                                }
                                #INT_MAY_FORCE => {
                                    __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                #INT_LOOP_INVARIANT => {
                                    __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                _ => {
                                    #unsupported
                                }
                            }
                        },
                    );
                    // jtransform.py:467/480-482 — `inline_call_*` is always
                    // followed by `-live-`; a residual call (`call_*`)
                    // appends `-live-` only when `calldescr_canraise(calldescr)`
                    // (`call.py`).  In inferred mode the policy
                    // byte selects the calldescr at runtime, so emit the
                    // marker conditional on the can-raise codes plus the
                    // inline byte (4u8) which forces emit per
                    // jtransform.py:480-482.  LoopInvariantInt (18u8) and
                    // ElidableCannotRaiseInt (19u8) skip the marker
                    // because their calldescrs are statically cannot-raise.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live.clone(),
                    );
                } else {
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #word_result_addr
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                // The policy byte hands over `_shared`, not `_with_asm`:
                                // the cache probe is what a recursive helper needs.
                                let __builder_fn: fn(&mut majit_metainterp::Assembler)
                                    -> majit_metainterp::jitcode::InlineJitCodeRef =
                                    unsafe { std::mem::transmute(__inline_builder) };
                                let __sub_edge = __builder_fn(__asm);
                                let __sub_return_kind = majit_metainterp::JitArgKind::Int;
                                let __sub_idx = __builder.add_sub_jitcode_ref(__sub_edge);
                                #inline_call
                            }
                            #INT_MAY_FORCE => {
                                __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            #INT_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            _ => {
                                #unsupported
                            }
                        }
                    });
                    // jtransform.py:467/480-482 — see int_arg_regs branch above.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live,
                    );
                }
            }
        }

        // jtransform.py:467-471 / 480-482 — trailing `-live-` for the explicit
        // residual / elidable / may-force arms (computed above).
        if post_live_after_call {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }

        // Every `*_ref_wrapped` policy lands here rather than returning
        // early above, and its result is a ref like theirs.
        let struct_type = match result_kind {
            BindingKind::Ref => self.declared_return_struct(func),
            BindingKind::Int | BindingKind::Float => None,
        };

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
            struct_type,
        })
    }

    /// Lower Rust's explicit wrapping integer methods to the same native int
    /// binops as RPython's `int_add` / `int_sub` / `int_mul` special cases.
    ///
    /// RPython parity: `jtransform.py _handle_int_special` emits the
    /// arithmetic op directly instead of residualizing a helper call. Rust
    /// kernels use method syntax (`x.wrapping_add(y)`) to spell the same
    /// wraparound semantics, so lower that syntax to `IntAdd` / `IntSub` /
    /// `IntMul` rather than aborting the trace at the method call.
    fn lower_wrapping_int_method_call(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        if call.args.len() != 1 {
            return None;
        }
        let opcode_name = match call.method.to_string().as_str() {
            "wrapping_add" => "IntAdd",
            "wrapping_sub" => "IntSub",
            "wrapping_mul" => "IntMul",
            _ => return None,
        };

        let lhs = self.lower_value_expr(&call.receiver)?;
        let rhs = self.lower_value_expr(&call.args[0])?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }

        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        let opcode = Ident::new(opcode_name, proc_macro2::Span::call_site());
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            binop_i_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
            struct_type: None,
        })
    }

    /// Normalize an `Expr::MethodCall` to the associated free-call shape used
    /// by the codewriter call-policy machinery.  Both value- and
    /// statement-position lowering call this helper: RPython
    /// `Transformer.rewrite_call` receives the receiver as the first ordinary
    /// argument and does not distinguish the two Rust surface syntaxes.
    ///
    /// Receiver-resolution policy: only the env parameter (`program`) and
    /// the state parameter (`state`) are accepted; any other receiver
    /// returns `None`. RPython `call.py getcalldescr` keys on
    /// graph identity (no naming collision possible); pyre keys on
    /// canonical path so the `<state_type|env_type>::<method>` lookup
    /// preserves that fidelity. Arbitrary receivers cannot be resolved
    /// without the owning type identity, so they fall through.
    ///
    pub(super) fn normalize_method_call(&self, call: &ExprMethodCall) -> Option<ExprCall> {
        let receiver_ident = match &*call.receiver {
            Expr::Path(ExprPath { path, .. }) => path.get_ident()?,
            _ => return None,
        };
        let receiver_name = receiver_ident.to_string();
        let config = self.config?;
        let type_name = match receiver_name.as_str() {
            "program" => config.env_type_name.clone(),
            "state" => config.state_type_name.clone(),
            _ => return None,
        };

        // Construct the `Type::method` function item.  Preserve an explicit
        // method turbofish: `recv.method::<T>(x)` becomes
        // `Type::method::<T>(recv, x)`.
        let type_ident = format_ident!("{}", type_name);
        let method_ident = &call.method;
        let func: Expr = if let Some(turbofish) = &call.turbofish {
            syn::parse_quote! { #type_ident::#method_ident #turbofish }
        } else {
            syn::parse_quote! { #type_ident::#method_ident }
        };
        let mut args = syn::punctuated::Punctuated::new();
        args.push((*call.receiver).clone());
        args.extend(call.args.iter().cloned());
        Some(ExprCall {
            attrs: call.attrs.clone(),
            func: Box::new(func),
            paren_token: call.paren_token,
            args,
        })
    }

    /// Method syntax does not select a smaller policy/result-kind universe.
    /// Reuse `lower_call_value` so residual, elidable, may-force, release-gil,
    /// loop-invariant, wrapped, inline and inferred policies all keep the same
    /// `Transformer.rewrite_call` contract.
    fn lower_method_call_value(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        let normalized = self.normalize_method_call(call)?;
        self.lower_call_value(&normalized)
    }

    /// `jtransform.py` `_rewrite_equality` via `rewrite_op_ptr_eq` /
    /// `rewrite_op_ptr_ne`: a comparison against the null pointer is the
    /// unary `ptr_iszero` / `ptr_nonzero`. The null operand is recognised
    /// on the source tree; the value lowerer also emits `Constant(nullptr)`.
    fn lower_ptr_equality_against_null(&mut self, expr: &ExprBinary) -> Option<Binding> {
        let left_null = expr_is_null_ptr(&expr.left);
        let right_null = expr_is_null_ptr(&expr.right);
        if !left_null && !right_null {
            return None;
        }
        if left_null && right_null {
            let value = i64::from(matches!(expr.op, BinOp::Eq(_)));
            let reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                quote! { __builder.load_const_i_value(#reg, #value); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        let other = if right_null { &expr.left } else { &expr.right };
        let binding = self.lower_value_expr(other)?;
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        self.emit_ptr_nullity_test(binding, matches!(expr.op, BinOp::Eq(_)))
    }

    /// `x.is_null()` is RPython `ptr_iszero`.
    fn lower_ptr_is_null_method_call(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        if call.method != "is_null" || !call.args.is_empty() {
            return None;
        }
        let recv = self.lower_value_expr(&call.receiver)?;
        if !matches!(recv.kind, BindingKind::Ref) {
            return None;
        }
        self.emit_ptr_nullity_test(recv, true)
    }

    /// `ptr_iszero` when `is_zero`, otherwise `ptr_nonzero`.
    fn emit_ptr_nullity_test(&mut self, binding: Binding, is_zero: bool) -> Option<Binding> {
        let src_reg = binding.reg;
        let reg = self.alloc_reg();
        let emit = if is_zero {
            quote! { __builder.ptr_iszero(#reg, #src_reg); }
        } else {
            quote! { __builder.ptr_nonzero(#reg, #src_reg); }
        };
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                vec![Register::ref_(src_reg)],
                vec![Register::int(reg)],
            ),
            emit,
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        })
    }

    fn lower_if_value(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        if let Some(binding) = self.lower_bool_if(expr_if) {
            return Some(binding);
        }

        let cond = self.lower_condition(&expr_if.cond)?;
        let cond_depends_on_stack = cond.depends_on_stack();
        let (_, else_expr) = expr_if.else_branch.as_ref()?;
        // A branch that fails to lower fails the whole if-expression as
        // unsupported (`?`): the codewriter lowers expressible ops exactly
        // or rejects the construct (`jtransform.py`), never replacing one
        // branch with a runtime abort stub. The caller degrades cleanly
        // (sub-JitCode entry returns `None`, arm runs in the interpreter).
        let (then_seq, then_binding) =
            self.lower_branch_value_expr(&Expr::Block(syn::ExprBlock {
                attrs: Vec::new(),
                label: None,
                block: expr_if.then_branch.clone(),
            }))?;
        let (else_seq, else_binding) = self.lower_branch_value_expr(else_expr)?;
        let int_branch = matches!(then_binding.kind, BindingKind::Int)
            && matches!(else_binding.kind, BindingKind::Int);
        // `jtransform.py` `rewrite_op_same_as`: both arms are refs, joined by
        // a ref copy. A static ref on the taken arm stays a constant.
        let ref_branch = matches!(then_binding.kind, BindingKind::Ref)
            && matches!(else_binding.kind, BindingKind::Ref);
        if !int_branch && !ref_branch {
            return None;
        }
        let then_reg = then_binding.reg;
        let else_reg = else_binding.reg;

        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_lowered_condition_guard(&cond, &else_label);
        self.append_lowered_sequence(then_seq);
        if int_branch {
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(then_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #then_reg); },
            );
        } else {
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveR,
                    vec![Register::ref_(then_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! { __builder.move_r(#result_reg, #then_reg); },
            );
        }
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);
        self.append_lowered_sequence(else_seq);
        if int_branch {
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(else_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #else_reg); },
            );
        } else {
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveR,
                    vec![Register::ref_(else_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! { __builder.move_r(#result_reg, #else_reg); },
            );
        }
        self.emit_label_def(&end_label);

        let struct_type = if ref_branch {
            match (&then_binding.struct_type, &else_binding.struct_type) {
                (Some(a), Some(b)) if a == b => Some(a.clone()),
                _ => None,
            }
        } else {
            None
        };
        Some(Binding {
            reg: result_reg,
            kind: if int_branch {
                BindingKind::Int
            } else {
                BindingKind::Ref
            },
            depends_on_stack: cond_depends_on_stack
                || then_binding.depends_on_stack
                || else_binding.depends_on_stack,
            struct_type,
        })
    }

    fn lower_bool_if(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        let (then_value, else_value) = extract_bool_branch_values(expr_if)?;
        let cond = self.lower_value_expr(&expr_if.cond)?;
        if !matches!(cond.kind, BindingKind::Int) {
            return None;
        }
        match (then_value, else_value) {
            (1, 0) => Some(cond),
            (0, 1) => {
                // `_rewrite_equality` / `bool_not` → `int_is_zero`.
                let cond_reg = cond.reg;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        Register::ints(&[cond_reg]),
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::IntIsZero, #cond_reg); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: cond.depends_on_stack,
                    struct_type: None,
                })
            }
            _ => None,
        }
    }

    fn lower_unary(&mut self, op: &UnOp, expr: &Expr) -> Option<Binding> {
        match op {
            UnOp::Neg(_) => {
                let inner = self.lower_value_expr(expr)?;
                let reg = self.alloc_reg();
                let src_reg = inner.reg;
                // Negate on the int bank (`IntNeg`) and the float bank
                // (`FloatNeg`) share the one-src/one-dst unary shape; the bank
                // is carried by the `Register` tag, not the `OpKind`. A `Ref`
                // operand aborts to the interpreter.
                let (reads, writes, emit, kind) = match inner.kind {
                    BindingKind::Int => (
                        vec![Register::int(src_reg)],
                        vec![Register::int(reg)],
                        quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::IntNeg, #src_reg); },
                        BindingKind::Int,
                    ),
                    BindingKind::Float => (
                        vec![Register::float(src_reg)],
                        vec![Register::float(reg)],
                        quote! { __builder.record_unary_f(#reg, majit_ir::OpCode::FloatNeg, #src_reg); },
                        BindingKind::Float,
                    ),
                    BindingKind::Ref => return None,
                };
                self.emit_op(OpMeta::linear(OpKind::UnaryI, reads, writes), emit);
                Some(Binding {
                    reg,
                    kind,
                    depends_on_stack: inner.depends_on_stack,
                    struct_type: None,
                })
            }
            _ => None,
        }
    }

    /// Recognize a function call registered as a native int binop alias.
    /// Emits the corresponding IR op (IntAdd, IntSub, etc.) directly,
    /// bypassing the call-policy machinery.  The concrete path calls the
    /// original function normally — no observer replay needed (pure op).
    ///
    /// jtransform.py `_handle_int_special()` parity.
    fn lower_native_int_binop_call(&mut self, call: &ExprCall) -> Option<Binding> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;
        let opcode_name = config
            .native_int_binops
            .iter()
            .find(|(path, _)| *path == func_segments)
            .map(|(_, op)| op.clone())?;

        if call.args.len() != 2 {
            return None;
        }

        let lhs = self.lower_value_expr(&call.args[0])?;
        let rhs = self.lower_value_expr(&call.args[1])?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }

        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        let opcode = Ident::new(&opcode_name, proc_macro2::Span::call_site());
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            binop_i_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
            struct_type: None,
        })
    }

    /// Recognize a unary function call registered in `native_identity` and
    /// bind its argument in place of the call, emitting nothing.
    ///
    /// The argument is still lowered, so a receiver that mutates state keeps
    /// running; only the wrapper around its result goes away.
    ///
    /// This runs ahead of the `calls` policy machinery, so a path in both
    /// would take this route and leave its policy dead; the config parse
    /// rejects that pair rather than let it lower
    /// (`reject_overlapping_call_vocabularies`).
    fn lower_native_identity_call(&mut self, call: &ExprCall) -> Option<Binding> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;
        if !config.native_identity.contains(&func_segments) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        self.lower_value_expr(&call.args[0])
    }

    /// Recognize a unary function call registered in `native_tag_small` and
    /// emit native `(x << 1) | 1` IR (IntLshift then IntOr) instead of a CallI.
    /// The concrete/naive path calls the original fn normally.
    fn lower_native_tag_small_call(&mut self, call: &ExprCall) -> Option<Binding> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;
        if !config.native_tag_small.contains(&func_segments) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }

        let inner = self.lower_value_expr(&call.args[0])?;
        if !matches!(inner.kind, BindingKind::Int) {
            return None;
        }

        let x_reg = inner.reg;
        let one = quote! { 1i64 };
        let shl_reg = self.alloc_reg();
        let lshift = syn::Ident::new("IntLshift", proc_macro2::Span::call_site());
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[x_reg]),
                vec![Register::int(shl_reg)],
            ),
            binop_i_const_emit_tokens(shl_reg, &lshift, x_reg, &one)
                .expect("int_lshift accepts a constant operand"),
        );

        let res_reg = self.alloc_reg();
        let or = syn::Ident::new("IntOr", proc_macro2::Span::call_site());
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[shl_reg]),
                vec![Register::int(res_reg)],
            ),
            binop_i_const_emit_tokens(res_reg, &or, shl_reg, &one)
                .expect("int_or accepts a constant operand"),
        );
        Some(Binding {
            reg: res_reg,
            kind: BindingKind::Int,
            depends_on_stack: inner.depends_on_stack,
            struct_type: None,
        })
    }

    fn lower_binary(&mut self, expr: &ExprBinary) -> Option<Binding> {
        if matches!(expr.op, BinOp::Eq(_) | BinOp::Ne(_))
            && let Some(binding) = self.lower_ptr_equality_against_null(expr)
        {
            return Some(binding);
        }
        if let Some(binding) = self.transactional(|s| s.lower_binary_int_constant(expr)) {
            return Some(binding);
        }
        let lhs = self.lower_value_expr(&expr.left)?;
        let rhs = self.lower_value_expr(&expr.right)?;
        if matches!(lhs.kind, BindingKind::Float) && matches!(rhs.kind, BindingKind::Float) {
            let lhs_reg = lhs.reg;
            let rhs_reg = rhs.reg;
            // A float comparison materializes a `0`/`1` int result (like the
            // int compare path), so its operands stay float but the write is
            // int-banked. `record_compare_f` emits `float_lt/ff>i` etc.
            if let Some(opcode) = opcode_for_compare_f(&expr.op) {
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopF,
                        Register::floats(&[lhs_reg, rhs_reg]),
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_compare_f(#reg, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg); },
                );
                return Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
                    struct_type: None,
                });
            }
            let opcode = opcode_for_binop_f(&expr.op)?;
            let reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopF,
                    Register::floats(&[lhs_reg, rhs_reg]),
                    vec![Register::float(reg)],
                ),
                binop_f_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Float,
                depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
                struct_type: None,
            });
        }
        // jtransform.py `rewrite_op_ptr_eq` / `rewrite_op_ptr_ne`: comparing
        // two pointers is `ptr_eq` / `ptr_ne`, not the integer compare.  Both
        // operands stay ref-banked and the result is int (`ptr_eq/rr>i`) --
        // the same split the float arm above makes, with the banks carried on
        // the `Register`s so `BinopI` is only the op-kind label.  Before this
        // arm a Ref pair fell through to the Int check below and declined,
        // which degraded the whole dispatch arm to the interpreter.
        //
        // RPython promotes the pair to `instance_ptr_eq` / `instance_ptr_ne`
        // when `_is_rclass_instance(args[0])`, which is
        // `lltype._castdepth(v.concretetype.TO, rclass.OBJECT) >= 0` -- the
        // pointee descends from the instance base.  `struct_type` is a
        // different fact: "the lowering tracked a pointee layout", which a raw
        // struct, a `ref_params` entry, a field pointee and an array element
        // all satisfy.  Reading it as the instance predicate hands non-instance
        // pointers to instance-only reasoning: `optimize_oois_ooisnot` folds a
        // pair with two known-but-different classes to a constant, and two
        // structural views of one raw address are known-different classes that
        // do alias.
        //
        // Nothing in this tree carries the real predicate, so the pair stays on
        // the plain pointer compare until something does.  That is upstream's
        // own fallthrough -- `rewrite_op_ptr_eq` promotes only under the test
        // and otherwise leaves `ptr_eq` alone -- and it costs nothing the
        // branch had before, which declined the whole dispatch arm here.
        //
        // RPython has no ordered pointer comparison, so any other operator
        // declines and the arm keeps running in the interpreter.
        if matches!(lhs.kind, BindingKind::Ref) && matches!(rhs.kind, BindingKind::Ref) {
            let is_eq = match expr.op {
                BinOp::Eq(_) => true,
                BinOp::Ne(_) => false,
                _ => return None,
            };
            let name = if is_eq { "PtrEq" } else { "PtrNe" };
            let opcode = Ident::new(name, proc_macro2::Span::call_site());
            let reg = self.alloc_reg();
            let lhs_reg = lhs.reg;
            let rhs_reg = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::refs(&[lhs_reg, rhs_reg]),
                    vec![Register::int(reg)],
                ),
                quote! { __builder.record_binop_r(#reg, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
                struct_type: None,
            });
        }
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let depends_on_stack = lhs.depends_on_stack || rhs.depends_on_stack;
        // jtransform.py `_rewrite_symmetric` -- "rewrite 'c1+v2' into 'v2+c1'
        // in an attempt to avoid generating too many variants of the
        // bytecode".  Upstream reads `isinstance(op.args[0], Constant)` off
        // the flow graph; here the constant is still a source literal, so the
        // syn expression answers the same question.  Swapping the operands of
        // an ordered comparison mirrors it; every other operator keeps its
        // spelling, which is upstream's `.get(op.opname, op.opname)` default.
        let swap = binop_is_symmetric(&expr.op)
            && int_literal_value(&expr.left).is_some()
            && int_literal_value(&expr.right).is_none();
        let (op, rhs_expr, lhs_reg, rhs_reg) = if swap {
            (
                mirrored_compare_binop(&expr.op),
                &*expr.left,
                rhs.reg,
                lhs.reg,
            )
        } else {
            (expr.op, &*expr.right, lhs.reg, rhs.reg)
        };
        // jtransform.py `_rewrite_equality` via `rewrite_op_int_eq` /
        // `rewrite_op_int_ne`: a comparison against zero is the unary
        // `int_is_zero` / `int_is_true`, which is also the form
        // `goto_if_not_int_is_zero` can fuse.  The swap above already moved a
        // literal left operand to the right, so both spellings arrive here as
        // `<value> <op> 0`.
        if matches!(op, BinOp::Eq(_) | BinOp::Ne(_)) && int_literal_value(rhs_expr) == Some(0) {
            let name = if matches!(op, BinOp::Eq(_)) {
                "IntIsZero"
            } else {
                "IntIsTrue"
            };
            let opcode = Ident::new(name, proc_macro2::Span::call_site());
            let reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(
                    OpKind::UnaryI,
                    Register::ints(&[lhs_reg]),
                    vec![Register::int(reg)],
                ),
                quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::#opcode, #lhs_reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack,
                struct_type: None,
            });
        }
        let opcode = opcode_for_binop(&op)?;
        let reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            binop_i_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack,
            struct_type: None,
        })
    }

    /// An int operation with one `Constant` operand keeps the constant in the
    /// operation (`assembler.py` `emit_const`) instead of loading it into a
    /// register. `jtransform.py` `_rewrite_symmetric` first moves a constant
    /// left operand to the right, mirroring an ordered comparison, and
    /// `_rewrite_equality` turns a compare against zero into the unary
    /// `int_is_zero` / `int_is_true`. `None` leaves the operation to the
    /// register-operand lowering.
    fn lower_binary_int_constant(&mut self, expr: &ExprBinary) -> Option<Binding> {
        let (op, var_expr, const_expr, value) = match (
            self.int_constant_operand(&expr.left),
            self.int_constant_operand(&expr.right),
        ) {
            (None, Some(value)) => (expr.op, &*expr.left, &*expr.right, value),
            (Some(value), None) if binop_is_symmetric(&expr.op) => (
                mirrored_compare_binop(&expr.op),
                &*expr.right,
                &*expr.left,
                value,
            ),
            _ => return None,
        };
        let var = self.lower_value_expr(var_expr)?;
        if !matches!(var.kind, BindingKind::Int) {
            return None;
        }
        let reg = self.alloc_reg();
        if matches!(op, BinOp::Eq(_) | BinOp::Ne(_)) && int_literal_value(const_expr) == Some(0) {
            let name = if matches!(op, BinOp::Eq(_)) {
                "IntIsZero"
            } else {
                "IntIsTrue"
            };
            let opcode = Ident::new(name, proc_macro2::Span::call_site());
            let var_reg = var.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::UnaryI,
                    Register::ints(&[var_reg]),
                    vec![Register::int(reg)],
                ),
                quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::#opcode, #var_reg); },
            );
        } else {
            let opcode = opcode_for_binop(&op)?;
            let tokens = binop_i_const_emit_tokens(reg, &opcode, var.reg, &value)?;
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[var.reg]),
                    vec![Register::int(reg)],
                ),
                tokens,
            );
        }
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: var.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower an `<int> as f64` cast to RPython's `cast_int_to_float`
    /// (`bhimpl_cast_int_to_float`). The operand is read from the int bank and
    /// widened to the float bank; the result is a float binding. `UnaryI` is the
    /// op-kind label (a single int read) — the float bank of the result is
    /// carried by the write `Register`, which is what liveness/regalloc read.
    fn lower_int_to_float_cast(&mut self, binding: Binding) -> Option<Binding> {
        let src_reg = binding.reg;
        let reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                Register::ints(&[src_reg]),
                vec![Register::float(reg)],
            ),
            quote! { __builder.record_cast_int_to_float(#reg, #src_reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Float,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        })
    }

    /// `rewrite_op_force_cast` when the source is unsigned and the
    /// destination is float: residual `cast_uint_to_float`.
    fn lower_uint_to_float_cast(&mut self, binding: Binding) -> Option<Binding> {
        let src_reg = binding.reg;
        let reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                Register::ints(&[src_reg]),
                vec![Register::float(reg)],
            ),
            quote! { __builder.record_cast_uint_to_float(#reg, #src_reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Float,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower a `<float> as i64` cast to RPython's `cast_float_to_int`
    /// (`bhimpl_cast_float_to_int`), the inverse of
    /// [`Self::lower_int_to_float_cast`]. The operand is read from the float
    /// bank and truncated toward zero into the int bank; the result is an int
    /// binding. `UnaryI` is the op-kind label, as for the neighbouring
    /// `convert_float_bytes_to_longlong`; the float bank of the SOURCE and the
    /// int bank of the result are carried by the `Register`s, which is what
    /// liveness/regalloc read.
    fn lower_float_to_int_cast(&mut self, binding: Binding) -> Option<Binding> {
        let src_reg = binding.reg;
        let reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                vec![Register::float(src_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.record_cast_float_to_int(#reg, #src_reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        })
    }

    /// `rewrite_op_force_cast` float → unsigned word: residual
    /// `cast_float_to_uint`.
    fn lower_float_to_uint_cast(&mut self, binding: Binding) -> Option<Binding> {
        let src_reg = binding.reg;
        let reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::UnaryI,
                vec![Register::float(src_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.record_cast_float_to_uint(#reg, #src_reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower an integer `as <ty>` cast to the truncate + (sign|zero)-extend
    /// the Rust `as` operator performs (RPython's rtyper inserts the same
    /// `int_signext` / mask when an interpreter narrows a wider value).
    /// A cast to a 64-bit or pointer-width target is a no-op on the
    /// already-machine-word binding.  A narrowing cast to a signed type
    /// sign-extends from the type's high bit (`(x << s) >> s`, arithmetic
    /// shift); to an unsigned type it masks the low bits (`x & ((1<<n)-1)`).
    /// Previously every int cast was a no-op, dropping the sign extension a
    /// `program[pc] as i8 as i64` operand fetch relies on (an 8-bit `0xEE`
    /// stayed `238` instead of `-18`, so backward branch targets computed
    /// out of range).
    fn lower_int_cast(&mut self, binding: Binding, ty: &Type) -> Option<Binding> {
        let ident = match ty {
            Type::Path(tp) => tp.path.get_ident()?,
            _ => return None,
        };
        let (signed, bits): (bool, u32) = match ident.to_string().as_str() {
            "i8" => (true, 8),
            "i16" => (true, 16),
            "i32" => (true, 32),
            "u8" => (false, 8),
            "u16" => (false, 16),
            "u32" => (false, 32),
            // `_int_to_int_cast` to Bool is `int_is_true`.
            "bool" => {
                let src_reg = binding.reg;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        Register::ints(&[src_reg]),
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::IntIsTrue, #src_reg); },
                );
                return Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: binding.depends_on_stack,
                    struct_type: None,
                });
            }
            // 64-bit targets keep the full machine word. Pointer-width
            // targets are identity on a 64-bit host (`jtransform.py`
            // `rewrite_op_cast_int_to_uint` / `rewrite_op_cast_uint_to_int`
            // are `pass` — same representation, drop-and-alias) and a
            // 32-bit mask / sign-extend on wasm32.
            "i64" | "u64" => return Some(binding),
            "isize" => return Some(self.emit_pointer_width_int_cast(binding, true)),
            "usize" => return Some(self.emit_pointer_width_int_cast(binding, false)),
            _ => return None,
        };
        // `jtransform.py` `_int_to_int_cast`: a narrowing to a signed type is
        // `int_signext(v, size2)`, to an unsigned one `int_and(v, mask)`,
        // both with a Constant second operand.
        let depends_on_stack = binding.depends_on_stack;
        let x_reg = binding.reg;
        let result_reg = self.alloc_reg();
        let (opcode, operand) = if signed {
            ("IntSignext", i64::from(bits / 8))
        } else {
            ("IntAnd", ((1u64 << bits) - 1) as i64)
        };
        let opcode = syn::Ident::new(opcode, proc_macro2::Span::call_site());
        let operand = quote! { #operand };
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[x_reg]),
                vec![Register::int(result_reg)],
            ),
            binop_i_const_emit_tokens(result_reg, &opcode, x_reg, &operand)
                .expect("int_signext and int_and take a Constant operand"),
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack,
            struct_type: None,
        })
    }

    /// `as usize` / `as isize` on the already-machine-word int bank.
    ///
    /// RPython `jtransform.py rewrite_op_cast_int_to_uint` /
    /// `rewrite_op_cast_uint_to_int` are explicit no-ops: both kinds share
    /// the `'int'` box, so the cast is a rename. The target width is known
    /// only when the jitcode is built, so the op is one
    /// [`OpKind::CastIntToWord`] that register allocation coalesces like a
    /// renaming; `JitCodeBuilder::cast_int_to_word` then emits nothing on a
    /// 64-bit word and the 32-bit narrowing on wasm32.
    fn emit_pointer_width_int_cast(&mut self, binding: Binding, signed: bool) -> Binding {
        let x_reg = binding.reg;
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::CastIntToWord,
                Register::ints(&[x_reg]),
                Register::ints(&[result_reg]),
            ),
            quote! { __builder.cast_int_to_word(#result_reg as u16, #x_reg as u16, #signed); },
        );
        Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: binding.depends_on_stack,
            struct_type: None,
        }
    }

    pub(super) fn lower_branch_expr(&mut self, expr: &Expr) -> Option<LoweredSequence> {
        let stmts = extract_stmts(expr);
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            body_failure_reason: None,
            nested_failure_reasons: Vec::new(),
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
            sub_jitcode_identity_slot: None,
            dispatch_loop_label: self.dispatch_loop_label.clone(),
            pc_pinned: self.pc_pinned,
            join_merge: self.enclosing_join_merge(),
            // Never inherited: a nested block statement is not the arm body's
            // tail, so a `return` inside it must be rejected, not lowered.
            inline_arm_tail_stmt: false,
        };

        for stmt in &stmts {
            if nested.lower_stmt(stmt).is_none() {
                // Carry the diagnosis out before the child is dropped; a bare
                // `?` here propagates the failure and loses the reason.
                self.absorb_nested_failure(&mut nested);
                return None;
            }
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        self.absorb_nested_identity_slot(&mut nested);
        self.inline_liveness_prebuild
            .extend(nested.inline_liveness_prebuild);
        Some(LoweredSequence::new(nested.statements, nested.op_metadata))
    }

    pub(super) fn lower_branch_value_expr(
        &mut self,
        expr: &Expr,
    ) -> Option<(LoweredSequence, Binding)> {
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            body_failure_reason: None,
            nested_failure_reasons: Vec::new(),
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
            sub_jitcode_identity_slot: None,
            dispatch_loop_label: self.dispatch_loop_label.clone(),
            pc_pinned: self.pc_pinned,
            join_merge: self.enclosing_join_merge(),
            // Never inherited: a branch arm's value expression is not the arm
            // body's tail, so a `return` inside it must be rejected.
            inline_arm_tail_stmt: false,
        };

        let Some(binding) = nested.lower_scoped_value_expr(expr) else {
            // Carry the diagnosis out before the child is dropped; a bare `?`
            // here propagates the failure and loses the reason.
            self.absorb_nested_failure(&mut nested);
            return None;
        };
        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        self.absorb_nested_identity_slot(&mut nested);
        self.inline_liveness_prebuild
            .extend(nested.inline_liveness_prebuild);
        Some((
            LoweredSequence::new(nested.statements, nested.op_metadata),
            binding,
        ))
    }

    fn lower_scoped_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        match expr {
            Expr::Block(block) => self.lower_block_value(&block.block),
            _ => self.lower_value_expr(expr),
        }
    }

    pub(super) fn lower_block_value(&mut self, block: &Block) -> Option<Binding> {
        let (tail, prefix) = block.stmts.split_last()?;

        for stmt in prefix {
            self.lower_stmt(stmt)?;
        }

        match tail {
            Stmt::Expr(expr, None) => self.lower_value_expr(expr),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(reg: u16, kind: BindingKind) -> Binding {
        Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    #[test]
    fn non_generic_spellings_emit_the_resolved_type_id_helper() {
        for path in ["crate::a::X", "self::X", "super::a::X", "X", "reexport::X"] {
            let parsed: syn::Path = syn::parse_str(path).unwrap();
            let emitted = struct_type_id_tokens(&parsed, true).to_string();
            assert!(
                emitted.contains("__majit_struct_type_id"),
                "{path} -> {emitted}"
            );
            assert!(
                !emitted.contains("struct_type_id_path"),
                "{path} -> {emitted}"
            );
            assert!(
                emitted.contains(&path.replace("::", " :: ")),
                "{path} -> {emitted}"
            );
        }
        let generic: syn::Path = syn::parse_str("Foo<Bar>").unwrap();
        let emitted = struct_type_id_tokens(&generic, true).to_string();
        assert!(
            !emitted.contains("__majit_struct_type_id"),
            "generic fallback must stay a token hash, got {emitted}"
        );
    }

    #[test]
    fn unsafe_block_lowers_its_body() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("x".to_string(), binding(4, BindingKind::Int));
        let expr: Expr = syn::parse_str("unsafe { x + 1 }").expect("parse unsafe");
        let result = lowerer
            .lower_value_expr(&expr)
            .expect("unsafe block should lower");
        assert_eq!(result.kind, BindingKind::Int);
    }

    #[test]
    fn ref_pointer_cast_keeps_the_ref_register() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("p".to_string(), binding(3, BindingKind::Ref));
        let expr: Expr = syn::parse_str("p as *mut u8").expect("parse cast");
        let result = lowerer
            .lower_value_expr(&expr)
            .expect("ref to pointer cast should lower");
        assert_eq!(result.kind, BindingKind::Ref);
        assert_eq!(result.reg, 3);
        assert!(result.struct_type.is_none());
    }

    #[test]
    fn pointer_cast_to_a_struct_names_the_pointee() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("p".to_string(), binding(3, BindingKind::Ref));
        let expr: Expr = syn::parse_str("p as *mut Node").expect("parse cast");
        let result = lowerer
            .lower_value_expr(&expr)
            .expect("struct pointer cast should lower");
        assert_eq!(result.reg, 3);
        assert_eq!(result.struct_type.unwrap(), syn::parse_quote!(Node));
    }

    #[test]
    fn deref_field_read_lowers_to_getfield() {
        let config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "p".to_string(),
            Binding {
                reg: 1,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: Some(syn::parse_quote!(W_IntObject)),
            },
        );
        let expr: Expr = syn::parse_quote!(unsafe { (*p).intval });
        let result = lowerer
            .lower_value_expr(&expr)
            .expect("(*p).field should lower");
        assert_eq!(result.kind, BindingKind::Int);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            emitted.contains("getfield_gc_i"),
            "deref field read must be a getfield, got {emitted}"
        );
    }

    #[test]
    fn pointer_width_usize_cast_is_one_coalescable_cast() {
        // jtransform.py rewrite_op_cast_int_to_uint / rewrite_op_cast_uint_to_int
        // are identity. The builder decides the width, so the lowering emits
        // one op register allocation can coalesce.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("x".to_string(), binding(4, BindingKind::Int));
        let expr: Expr = syn::parse_str("x as usize").expect("parse usize cast");

        let result = lowerer
            .lower_value_expr(&expr)
            .expect("usize cast should lower");
        assert_eq!(result.kind, BindingKind::Int);

        assert_eq!(lowerer.op_metadata.len(), 1);
        let meta = &lowerer.op_metadata[0];
        assert_eq!(meta.kind, OpKind::CastIntToWord);
        assert_eq!(meta.reads, vec![Register::int(4)]);
        assert_eq!(meta.writes, vec![Register::int(result.reg)]);
        let emitted = lowerer.statements[0].to_string();
        assert!(
            emitted.contains("cast_int_to_word") && emitted.contains("false"),
            "got {emitted}"
        );
    }

    #[test]
    fn narrowing_int_casts_are_signext_and_mask_with_a_constant() {
        // jtransform.py `_int_to_int_cast`: int_signext(v, size2) for a
        // signed target, int_and(v, mask) for an unsigned one.
        for (ty, opcode, operand) in [("i8", "IntSignext", "1i64"), ("u16", "IntAnd", "65535i64")] {
            let mut lowerer = Lowerer::new(None);
            lowerer
                .bindings
                .insert("x".to_string(), binding(4, BindingKind::Int));
            let expr: Expr = syn::parse_str(&format!("x as {ty}")).expect("parse cast");
            lowerer.lower_value_expr(&expr).expect("cast should lower");

            assert_eq!(lowerer.op_metadata.len(), 1, "{ty}");
            assert_eq!(lowerer.op_metadata[0].reads, vec![Register::int(4)]);
            let emitted = lowerer.statements[0].to_string();
            assert!(
                emitted.contains("record_binop_i_const")
                    && emitted.contains(opcode)
                    && emitted.contains(operand),
                "{ty}: got {emitted}"
            );
        }
    }

    #[test]
    fn loop_carried_local_update_writes_the_existing_register() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("need".to_string(), binding(3, BindingKind::Int));
        let expr: Expr = syn::parse_str("need -= 1").expect("parse local update");

        assert_eq!(lowerer.lower_local_update(&expr), Some(()));

        assert_eq!(lowerer.op_metadata.len(), 1);
        let op = &lowerer.op_metadata[0];
        assert_eq!(op.kind, OpKind::BinopI);
        assert_eq!(op.reads, vec![Register::int(3)]);
        assert_eq!(op.writes, vec![Register::int(3)]);
        assert_eq!(lowerer.bindings["need"].reg, 3);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("record_binop_i_const"));
        assert!(emitted.contains("IntSub"));
    }

    /// A literal operand stays in the operation: no `load_const_i_value`, and
    /// the op reads only the variable's register.
    #[test]
    fn a_literal_operand_is_not_loaded_into_a_register() {
        for (src, opcode) in [
            ("n - 1", "IntSub"),
            ("1 + n", "IntAdd"),
            ("3 < n", "IntGt"),
            ("n == KIND", "IntEq"),
        ] {
            let mut lowerer = Lowerer::new(None);
            lowerer
                .bindings
                .insert("n".to_string(), binding(3, BindingKind::Int));
            let expr: Expr = syn::parse_str(src).expect("parse binary");
            let result = lowerer.lower_value_expr(&expr).expect("binary lowers");
            assert_eq!(result.kind, BindingKind::Int);
            assert_eq!(lowerer.op_metadata.len(), 1, "{src}");
            let op = &lowerer.op_metadata[0];
            assert_eq!(op.kind, OpKind::BinopI, "{src}");
            assert_eq!(op.reads, vec![Register::int(3)], "{src}");
            let emitted = lowerer
                .statements
                .iter()
                .map(ToString::to_string)
                .collect::<String>();
            assert!(emitted.contains("record_binop_i_const"), "{src}: {emitted}");
            assert!(emitted.contains(opcode), "{src}: {emitted}");
        }
    }

    /// `_rewrite_equality`: `n == 0` is the unary `int_is_zero`, and the zero
    /// is never loaded.
    #[test]
    fn an_int_compare_against_zero_loads_no_constant() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("n".to_string(), binding(3, BindingKind::Int));
        let expr: Expr = syn::parse_str("n == 0").expect("parse compare");
        lowerer.lower_value_expr(&expr).expect("compare lowers");
        assert_eq!(lowerer.op_metadata.len(), 1);
        assert_eq!(lowerer.op_metadata[0].kind, OpKind::UnaryI);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("IntIsZero"), "{emitted}");
    }

    #[test]
    fn byte_literal_uses_the_integer_constant_channel() {
        let mut lowerer = Lowerer::new(None);
        let expr: Expr = syn::parse_str("b']'").expect("parse byte literal");

        let binding = lowerer
            .lower_value_expr(&expr)
            .expect("byte literal lowers");

        assert_eq!(binding.kind, BindingKind::Int);
        assert_eq!(lowerer.op_metadata[0].kind, OpKind::LoadConstI);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("93i64"));
    }

    #[test]
    fn struct_layout_witnesses_are_array_element_blocks() {
        let config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("v".into(), binding(1, BindingKind::Int));
        let expr = syn::parse_str("Node { value: v }").unwrap();
        lowerer.lower_value_expr(&expr).expect("struct lowers");
        let statement = &lowerer.statements[0];
        // Layout witnesses contain const items; each layout must remain one
        // array expression when register allocation reparses builder calls.
        syn::parse2::<syn::Block>(quote!({ #statement }))
            .expect("new_struct with a field witness is valid Rust");
        assert!(statement.to_string().contains("const _"));
    }

    #[test]
    fn struct_literal_lowers_to_new_plus_setfield_gc() {
        // `Node { value: v, next: n }` with v:Int, n:Ref lowers to a New
        // allocation followed by a per-field setfield_gc_{i,r}.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("v".to_string(), binding(1, BindingKind::Int));
        lowerer
            .bindings
            .insert("n".to_string(), binding(2, BindingKind::Ref));
        let expr: Expr =
            syn::parse_str("Node { value: v, next: n }").expect("parse struct literal");

        let result = lowerer.lower_value_expr(&expr).expect("struct lowers");
        assert!(matches!(result.kind, BindingKind::Ref));

        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(
            kinds,
            vec![OpKind::New, OpKind::SetfieldGc, OpKind::SetfieldGc]
        );
        // The int field reads the int bank, the ref field the ref bank.
        assert_eq!(
            lowerer.op_metadata[1].reads,
            vec![Register::ref_(result.reg), Register::int(1)]
        );
        assert_eq!(
            lowerer.op_metadata[2].reads,
            vec![Register::ref_(result.reg), Register::ref_(2)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("new_struct"));
        assert!(emitted.contains("setfield_gc_i"));
        assert!(emitted.contains("setfield_gc_r"));
        assert!(emitted.contains("offset_of"));
        assert!(emitted.contains("size_of"));
    }

    fn varsize_config(element: &str) -> LowererConfig {
        let element_type: syn::Path = syn::parse_str(element).unwrap();
        let items = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Items),
            field: syn::parse_quote!(items),
            element_type,
            header: Some(syn::parse_quote!(Items)),
        };
        let capacity = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Items),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(i64),
        };
        LowererConfig::inline_helper(&[], &[items], &[capacity], &[], &[], &[], &[], &[])
    }

    #[test]
    fn pointer_varsize_literal_lowers_to_new_array_clear() {
        // `rewrite_op_malloc_varsize`: pointer elements are `new_array_clear`.
        // The descr carries the header's item offset and length offset
        // (`get_array_descr`).
        let config = varsize_config("Cell");
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("n".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("Items { capacity: n, items: [] }").unwrap();
        let result = lowerer.lower_value_expr(&expr).expect("varsize lowers");
        assert!(matches!(result.kind, BindingKind::Ref));
        assert!(result.struct_type.is_some());
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            emitted.contains("new_array_clear"),
            "pointer varsize must be new_array_clear, got {emitted}"
        );
        assert!(emitted.contains("add_gc_varsize_array_descr"));
        assert!(
            emitted.contains("__majit_struct_gc_type_id"),
            "varsize descr must stamp the registered GC type id, got {emitted}"
        );
        assert!(!emitted.contains("new_struct"));
    }

    #[test]
    fn int_varsize_literal_lowers_to_new_array() {
        // A primitive element is `new_array`, not `new_array_clear`.
        let config = varsize_config("i64");
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("n".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("Items { capacity: n, items: [] }").unwrap();
        lowerer.lower_value_expr(&expr).expect("varsize lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            emitted.contains("new_array"),
            "int varsize must be new_array, got {emitted}"
        );
        assert!(
            !emitted.contains("new_array_clear"),
            "int varsize must not clear, got {emitted}"
        );
    }

    #[test]
    fn float_varsize_literal_lowers_to_new_array_with_the_float_descr() {
        // An `f64` element is not a GC pointer: `new_array`, and the
        // descr is the one `getarrayitem_gc_f` reads.
        let config = varsize_config("f64");
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("n".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("Items { capacity: n, items: [] }").unwrap();
        lowerer.lower_value_expr(&expr).expect("varsize lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            !emitted.contains("new_array_clear"),
            "float varsize must not clear, got {emitted}"
        );
        assert!(
            emitted.contains("add_gc_varsize_float_array_descr"),
            "float varsize must use the float descr, got {emitted}"
        );
        assert!(
            emitted.contains("__majit_struct_gc_type_id"),
            "float varsize descr must stamp the registered GC type id, got {emitted}"
        );
    }

    #[test]
    fn varsize_literal_with_another_header_field_is_declined() {
        // Only the length is stamped by the allocation, so a literal that
        // also initialises another header field is not lowered.
        let config = varsize_config("i64");
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer
            .bindings
            .insert("n".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("Items { capacity: n, flag: 1i64, items: [] }").unwrap();
        assert!(lowerer.lower_value_expr(&expr).is_none());
    }

    #[test]
    fn struct_literal_float_field_lowers_to_setfield_gc_f() {
        // `jtransform.py` `rewrite_op_setfield` with `getkind` `f`.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("f".to_string(), binding(1, BindingKind::Float));
        let expr: Expr = syn::parse_str("Num { floatval: f }").expect("parse struct literal");
        let result = lowerer.lower_value_expr(&expr).expect("struct lowers");
        assert!(matches!(result.kind, BindingKind::Ref));
        assert_eq!(
            lowerer.op_metadata[1].reads,
            vec![Register::ref_(result.reg), Register::float(1)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(!emitted.contains("retag_struct_field_float"));
        assert!(emitted.contains("setfield_gc_f"));
        assert!(emitted.contains("true"));
    }

    #[test]
    fn raw_store_intrinsic_lowers_to_raw_store_i() {
        // `majit_raw_store_i64(base, ea, val)` lowers to a single RawStore
        // op reading the three int operands (base, ea, val) and emitting a
        // `raw_store_i` with an 8-byte signed raw-int array descr.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("base".to_string(), binding(3, BindingKind::Int));
        lowerer
            .bindings
            .insert("ea".to_string(), binding(4, BindingKind::Int));
        lowerer
            .bindings
            .insert("val".to_string(), binding(5, BindingKind::Int));
        let stmt: syn::Stmt =
            syn::parse_str("majit_raw_store_i64(base, ea, val);").expect("parse raw store");

        lowerer.lower_stmt(&stmt).expect("raw store lowers");

        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(kinds, vec![OpKind::RawStore]);
        assert_eq!(
            lowerer.op_metadata[0].reads,
            vec![Register::int(3), Register::int(4), Register::int(5)]
        );
        assert!(lowerer.op_metadata[0].writes.is_empty());
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("raw_store_i"));
        assert!(emitted.contains("add_raw_int_array_descr_signed"));
        assert!(emitted.contains("8usize"));
        assert!(emitted.contains("true"));
    }

    #[test]
    fn raw_store_intrinsics_lower_with_width_and_sign() {
        // `jtransform.py rewrite_op_raw_store` takes the descr from
        // the STORED value's own type, so a sub-word store carries its own
        // width and signedness exactly as the load side does.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("base".to_string(), binding(3, BindingKind::Int));
        lowerer
            .bindings
            .insert("ea".to_string(), binding(4, BindingKind::Int));
        lowerer
            .bindings
            .insert("val".to_string(), binding(5, BindingKind::Int));
        let stmt: syn::Stmt =
            syn::parse_str("majit_raw_store_u16(base, ea, val);").expect("parse raw store");

        lowerer.lower_stmt(&stmt).expect("raw store lowers");

        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(kinds, vec![OpKind::RawStore]);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("add_raw_int_array_descr_signed"));
        assert!(emitted.contains("2usize"));
        assert!(emitted.contains("false"));
    }

    #[test]
    fn raw_load_intrinsics_lower_with_width_and_sign() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("base".to_string(), binding(3, BindingKind::Int));
        lowerer
            .bindings
            .insert("ea".to_string(), binding(4, BindingKind::Int));
        let expr: Expr = syn::parse_str("majit_raw_load_i32(base, ea)").expect("parse raw load");

        let result = lowerer
            .lower_value_expr(&expr)
            .expect("raw load should lower");

        assert_eq!(result.kind, BindingKind::Int);
        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(kinds, vec![OpKind::RawLoad]);
        assert_eq!(
            lowerer.op_metadata[0].reads,
            vec![Register::int(3), Register::int(4)]
        );
        assert_eq!(
            lowerer.op_metadata[0].writes,
            vec![Register::int(result.reg)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("raw_load_i"));
        assert!(emitted.contains("add_raw_int_array_descr_signed"));
        assert!(emitted.contains("4usize"));
        assert!(emitted.contains("true"));
    }

    /// One lowering of `source`, seeded with the same three int bindings every
    /// spelling gets, reduced to the facts a caller can compare.
    fn lower_source(source: &str) -> (Vec<OpKind>, Vec<Vec<Register>>, Vec<Vec<Register>>, String) {
        let mut lowerer = Lowerer::new(None);
        for (name, reg) in [("base", 3u16), ("ea", 4), ("val", 5)] {
            lowerer
                .bindings
                .insert(name.to_string(), binding(reg, BindingKind::Int));
        }
        if source.ends_with(';') {
            let stmt: syn::Stmt = syn::parse_str(source).expect("parse statement");
            lowerer.lower_stmt(&stmt).expect("statement lowers");
        } else {
            let expr: Expr = syn::parse_str(source).expect("parse expression");
            lowerer.lower_value_expr(&expr).expect("expression lowers");
        }
        (
            lowerer.op_metadata.iter().map(|m| m.kind).collect(),
            lowerer
                .op_metadata
                .iter()
                .map(|m| m.reads.clone())
                .collect(),
            lowerer
                .op_metadata
                .iter()
                .map(|m| m.writes.clone())
                .collect(),
            lowerer
                .statements
                .iter()
                .map(ToString::to_string)
                .collect::<String>(),
        )
    }

    /// An IMPORTED intrinsic lowers exactly as a locally-defined one does.
    ///
    /// The lowerer matches the last segment of the call expression's path and
    /// never looks at where the function is defined, so a `use` of an intrinsic
    /// from another crate is recognized as readily as a `fn` in this module.
    /// The two are indistinguishable here on purpose: an imported name is
    /// spelled as a bare call, which is the same token stream a local
    /// definition produces, so the bare arm below IS the imported spelling and
    /// the qualified arm covers the other way to reach the same import.
    ///
    /// This matters because the untraced bodies these names need can then live
    /// in one library instead of being rewritten in every module that traces
    /// them — a duplication that silently diverges the two tiers the day one
    /// copy is fixed and the others are not.
    ///
    /// The op-kind assertion is load-bearing beyond naming the right op: the
    /// intrinsic arms run ahead of the generic call path, so a spelling that
    /// failed to be recognized would lower as a residual `Call` — and a
    /// residual needs a call-policy the imported spelling has no local
    /// definition to derive one from.
    #[test]
    fn an_imported_intrinsic_lowers_like_a_local_one() {
        for (bare, qualified, expected_kind, expected_emit) in [
            (
                "majit_raw_load_i64(base, ea)",
                "majit_metainterp::intrinsics::majit_raw_load_i64(base, ea)",
                OpKind::RawLoad,
                "raw_load_i",
            ),
            (
                "majit_raw_store_i64(base, ea, val);",
                "majit_metainterp::intrinsics::majit_raw_store_i64(base, ea, val);",
                OpKind::RawStore,
                "raw_store_i",
            ),
            (
                "majit_uint_lt(base, ea)",
                "majit_metainterp::intrinsics::majit_uint_lt(base, ea)",
                OpKind::BinopI,
                "UintLt",
            ),
        ] {
            let local = lower_source(bare);
            let imported = lower_source(qualified);
            assert_eq!(
                local.0,
                vec![expected_kind],
                "{bare} lowered to {:?}",
                local.0
            );
            assert_eq!(local, imported, "`{qualified}` lowered unlike `{bare}`");
            assert!(
                local.3.contains(expected_emit),
                "`{bare}` emitted no `{expected_emit}`: {}",
                local.3
            );
        }
    }

    /// The control for the test above: the match really is on the LAST segment.
    ///
    /// Without this, "the qualified spelling lowered the same" would also hold
    /// of a lowerer that ignored the path and matched on argument shape alone.
    #[test]
    fn only_the_last_path_segment_selects_an_intrinsic() {
        fn raw_load_lowers(source: &str) -> bool {
            let mut lowerer = Lowerer::new(None);
            for (name, reg) in [("base", 3u16), ("ea", 4)] {
                lowerer
                    .bindings
                    .insert(name.to_string(), binding(reg, BindingKind::Int));
            }
            let Expr::Call(call) = syn::parse_str::<Expr>(source).expect("parse call") else {
                unreachable!("the sources below are all call expressions");
            };
            lowerer.lower_raw_load_call(&call).is_some()
        }

        // An owner segment nobody declared still lowers: only the tail is read.
        assert!(raw_load_lowers("whatever::majit_raw_load_i64(base, ea)"));
        // A tail that is not an intrinsic name declines, however it is spelled.
        assert!(!raw_load_lowers(
            "majit_metainterp::intrinsics::majit_raw_load_i128(base, ea)"
        ));
        assert!(!raw_load_lowers("majit_raw_load_i128(base, ea)"));
    }

    fn emit_int_bytes_expr(src: &str, names: &[&str]) -> Option<String> {
        let mut lowerer = Lowerer::new(None);
        for (i, name) in names.iter().enumerate() {
            lowerer
                .bindings
                .insert((*name).to_string(), binding(i as u16, BindingKind::Int));
        }
        let expr: Expr = syn::parse_str(src).expect("parse bytes expr");
        lowerer.lower_value_expr(&expr)?;
        Some(lowerer.statements.iter().map(ToString::to_string).collect())
    }

    #[test]
    fn from_le_bytes_desugars_to_the_shift_or_chain() {
        let names = ["b0", "b1"];
        let via_from = emit_int_bytes_expr("u16::from_le_bytes([b0, b1])", &names)
            .expect("from_le_bytes must lower");
        let via_chain = emit_int_bytes_expr("((b0 as u16) | ((b1 as u16) << 8)) as u16", &names)
            .expect("shift/or chain must lower");
        assert_eq!(via_from, via_chain);
    }

    #[test]
    fn from_be_bytes_reverses_the_shift_amounts() {
        let names = ["b0", "b1"];
        let via_from = emit_int_bytes_expr("i16::from_be_bytes([b0, b1])", &names)
            .expect("from_be_bytes must lower");
        let via_chain = emit_int_bytes_expr("(((b0 as u16) << 8) | (b1 as u16)) as i16", &names)
            .expect("reversed shift/or chain must lower");
        assert_eq!(via_from, via_chain);
    }

    /// The chain a signed target desugars to is assembled unsigned and narrowed
    /// once. Assembling it in the signed type instead drops the narrowing,
    /// because only the casts narrow and the shift and the or run in the
    /// machine-word bank -- so those two streams must NOT be equal. Without
    /// this, `from_be_bytes_reverses_the_shift_amounts` would still pass with
    /// the sign extension removed.
    #[test]
    fn a_signed_target_does_not_desugar_into_its_own_type() {
        let names = ["b0", "b1"];
        let via_from = emit_int_bytes_expr("i16::from_le_bytes([b0, b1])", &names)
            .expect("from_le_bytes must lower");
        let signed_chain = emit_int_bytes_expr("(b0 as i16) | ((b1 as i16) << 8)", &names)
            .expect("signed shift/or chain must lower");
        assert_ne!(
            via_from, signed_chain,
            "`i16::from_le_bytes([0xff, 0xff])` is -1, and a chain assembled \
             in `i16` that never narrows the combined value answers 65535"
        );
    }

    #[test]
    fn from_endian_bytes_refuses_a_mismatched_count_or_non_array() {
        let names = ["b0", "b1", "b2", "b3"];
        assert!(
            emit_int_bytes_expr("u32::from_le_bytes([b0, b1])", &names).is_none(),
            "a 2-byte array is not a u32"
        );
        assert!(
            emit_int_bytes_expr("i8::from_le_bytes([b0])", &names).is_none(),
            "i8 is not in the recognised integer set"
        );
        assert!(
            emit_int_bytes_expr("i64::from_le_bytes(b0)", &names).is_none(),
            "a non-array argument is not a reconstruction"
        );
    }

    #[test]
    fn record_exact_class_statement_uses_ref_int_argcodes() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("value".to_string(), binding(1, BindingKind::Ref));
        lowerer
            .bindings
            .insert("cls".to_string(), binding(2, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("jit::record_exact_class(value, cls)").expect("parse hint call");

        assert_eq!(lowerer.lower_record_exact_class_stmt(&expr), Some(()));
        assert_eq!(lowerer.op_metadata.len(), 1);
        assert_eq!(lowerer.op_metadata[0].kind, OpKind::RecordExactClass);
        assert_eq!(
            lowerer.op_metadata[0].reads,
            vec![Register::ref_(1), Register::int(2)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("record_exact_class"));
    }

    #[test]
    fn record_exact_class_statement_rejects_ref_class_operand() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("value".to_string(), binding(1, BindingKind::Ref));
        lowerer
            .bindings
            .insert("cls".to_string(), binding(2, BindingKind::Ref));
        let expr: Expr =
            syn::parse_str("jit::record_exact_class(value, cls)").expect("parse hint call");

        assert_eq!(lowerer.lower_record_exact_class_stmt(&expr), None);
        assert!(lowerer.op_metadata.is_empty());
        assert!(lowerer.statements.is_empty());
    }

    /// `rewrite_op_ptr_eq` / `_rewrite_equality` / `_rewrite_symmetric` at the
    /// jitcode-lowering layer.  Each test drives `lower_binary` through
    /// `lower_value_expr` so the operand bindings are the ones production
    /// builds.
    mod ptr_and_equality_tests {
        use super::*;

        fn emitted(lowerer: &Lowerer<'_>) -> String {
            lowerer
                .statements
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        }

        fn lower(
            src: &str,
            setup: impl FnOnce(&mut Lowerer<'_>),
        ) -> (Lowerer<'static>, Option<Binding>) {
            let mut lowerer = Lowerer::new(None);
            setup(&mut lowerer);
            let expr: Expr = syn::parse_str(src).expect("parse binary expr");
            let Expr::Binary(binary) = expr else {
                panic!("expected a binary expression");
            };
            let out = lowerer.lower_binary(&binary);
            (lowerer, out)
        }

        fn ref_binding(reg: u16, struct_type: Option<syn::Path>) -> Binding {
            Binding {
                reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type,
            }
        }

        #[test]
        fn two_ref_operands_compare_with_ptr_eq() {
            let (lowerer, out) = lower("a == b", |l| {
                l.bindings.insert("a".into(), ref_binding(0, None));
                l.bindings.insert("b".into(), ref_binding(1, None));
            });
            assert!(out.is_some(), "a Ref pair must not decline the arm");
            let text = emitted(&lowerer);
            assert!(
                text.contains("record_binop_r") && text.contains("PtrEq"),
                "expected ptr_eq, got:\n{text}"
            );
        }

        #[test]
        fn two_ref_operands_compare_with_ptr_ne() {
            let (lowerer, out) = lower("a != b", |l| {
                l.bindings.insert("a".into(), ref_binding(0, None));
                l.bindings.insert("b".into(), ref_binding(1, None));
            });
            assert!(out.is_some());
            assert!(emitted(&lowerer).contains("PtrNe"));
        }

        #[test]
        fn null_mut_as_a_value_is_a_null_ref_constant() {
            let mut lowerer = Lowerer::new(None);
            let expr: Expr = syn::parse_str("core::ptr::null_mut()").expect("parse");
            let out = lowerer.lower_value_expr(&expr);
            assert!(out.is_some(), "null_mut() is Constant(nullptr)");
            let text = emitted(&lowerer);
            assert!(
                text.contains("load_const_r_value") && text.contains("0i64"),
                "assignment/return RHS must be a null ref, got:\n{text}"
            );
        }

        #[test]
        fn std_ptr_null_as_a_value_is_a_null_ref_constant() {
            let mut lowerer = Lowerer::new(None);
            let expr: Expr = syn::parse_str("std::ptr::null()").expect("parse");
            let out = lowerer.lower_value_expr(&expr);
            assert!(out.is_some());
            assert!(emitted(&lowerer).contains("load_const_r_value"));
        }

        #[test]
        fn foo_null_is_an_ordinary_call() {
            let mut lowerer = Lowerer::new(None);
            let expr: Expr = syn::parse_str("foo::null()").expect("parse");
            let _ = lowerer.lower_value_expr(&expr);
            let text = emitted(&lowerer);
            assert!(
                !text.contains("load_const_r_value"),
                "foo::null() must stay a call, got:\n{text}"
            );
        }

        #[test]
        fn assigning_null_mut_stores_the_null_ref() {
            let mut lowerer = Lowerer::new(None);
            lowerer.next_reg = 1;
            lowerer.bindings.insert("p".into(), ref_binding(0, None));
            let stmt: syn::Stmt = syn::parse_str("p = core::ptr::null_mut();").expect("parse");
            assert!(lowerer.lower_stmt(&stmt).is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("load_const_r_value") && text.contains("0i64"),
                "assignment RHS must lower, got:\n{text}"
            );
            assert_eq!(
                lowerer.bindings.get("p").map(|b| b.reg),
                Some(1),
                "the local must take the null ref register"
            );
        }

        #[test]
        fn returning_null_mut_is_a_null_ref_return() {
            let mut lowerer = Lowerer::new(None);
            lowerer.inline_arm_tail_stmt = true;
            let stmt: syn::Stmt = syn::parse_str("return core::ptr::null_mut();").expect("parse");
            assert!(lowerer.lower_stmt(&stmt).is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("load_const_r_value") && text.contains("ref_return"),
                "return value must lower, got:\n{text}"
            );
        }

        #[test]
        fn a_ref_equality_against_null_lowers_to_ptr_iszero() {
            let (lowerer, out) = lower("p == std::ptr::null_mut()", |l| {
                l.bindings.insert("p".into(), ref_binding(0, None));
            });
            assert!(out.is_some(), "null_mut() must not refuse the compare");
            let text = emitted(&lowerer);
            assert!(
                text.contains("ptr_iszero"),
                "`p == null_mut()` is `_rewrite_equality` to ptr_iszero, got:\n{text}"
            );
            assert!(
                !text.contains("PtrEq"),
                "must not keep the binary pointer compare:\n{text}"
            );
        }

        #[test]
        fn a_ref_inequality_against_null_lowers_to_ptr_nonzero() {
            let (lowerer, out) = lower("p != core::ptr::null()", |l| {
                l.bindings.insert("p".into(), ref_binding(0, None));
            });
            assert!(out.is_some());
            assert!(
                emitted(&lowerer).contains("ptr_nonzero"),
                "`p != null()` is ptr_nonzero"
            );
        }

        #[test]
        fn a_zero_cast_to_raw_pointer_is_the_null_constant() {
            let (lowerer, out) = lower("p == (0 as *mut u8)", |l| {
                l.bindings.insert("p".into(), ref_binding(0, None));
            });
            assert!(out.is_some());
            assert!(
                emitted(&lowerer).contains("ptr_iszero"),
                "`0 as *mut T` is the same null constant"
            );
        }

        #[test]
        fn is_null_lowers_to_ptr_iszero() {
            let mut lowerer = Lowerer::new(None);
            lowerer.bindings.insert("p".into(), ref_binding(0, None));
            let expr: Expr = syn::parse_str("p.is_null()").expect("parse method call");
            let out = lowerer.lower_value_expr(&expr);
            assert!(out.is_some(), "is_null() must not refuse");
            assert!(
                emitted(&lowerer).contains("ptr_iszero"),
                "`p.is_null()` is ptr_iszero"
            );
        }

        #[test]
        fn an_ordered_compare_over_refs_still_declines() {
            let (_lowerer, out) = lower("a < b", |l| {
                l.bindings.insert("a".into(), ref_binding(0, None));
                l.bindings.insert("b".into(), ref_binding(1, None));
            });
            assert!(out.is_none(), "RPython has no ordered pointer comparison");
        }

        #[test]
        fn two_tracked_struct_refs_still_use_the_plain_ptr_eq() {
            let path: syn::Path = syn::parse_quote!(Frame);
            let (lowerer, out) = lower("a == b", |l| {
                l.bindings
                    .insert("a".into(), ref_binding(0, Some(path.clone())));
                l.bindings.insert("b".into(), ref_binding(1, Some(path)));
            });
            assert!(out.is_some());
            let emitted = emitted(&lowerer);
            assert!(
                emitted.contains("PtrEq"),
                "a tracked pointee layout is not `_is_rclass_instance`"
            );
            assert!(
                !emitted.contains("InstancePtrEq"),
                "`instance_ptr_eq` licenses `optimize_oois_ooisnot` to fold a \
                 known-different-class pair, which raw pointers may not honour"
            );
        }

        #[test]
        fn an_int_equality_against_zero_lowers_to_int_is_zero() {
            let (lowerer, out) = lower("n == 0", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("record_unary_i") && text.contains("IntIsZero"),
                "expected the unary zero test, got:\n{text}"
            );
        }

        #[test]
        fn an_int_inequality_against_zero_lowers_to_int_is_true() {
            let (lowerer, out) = lower("n != 0", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            assert!(emitted(&lowerer).contains("IntIsTrue"));
        }

        #[test]
        fn a_zero_on_the_left_folds_the_same_way() {
            let (lowerer, out) = lower("0 == n", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            assert!(
                emitted(&lowerer).contains("IntIsZero"),
                "`0 == n` is `_rewrite_symmetric` then the same fold"
            );
        }

        #[test]
        fn rust_div_records_c_truncating_floordiv() {
            let (lowerer, out) = lower("a / b", |l| {
                l.bindings.insert("a".into(), binding(1, BindingKind::Int));
                l.bindings.insert("b".into(), binding(2, BindingKind::Int));
            });
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("record_int_floordiv"),
                "Rust `/` is `_ll_2_int_floordiv`, got:\n{text}"
            );
            assert!(
                !text.contains("record_int_py_div"),
                "must not emit Python-floor `ll_int_py_div`:\n{text}"
            );
        }

        #[test]
        fn majit_int_py_div_records_the_expandable_oopspec() {
            let mut lowerer = Lowerer::new(None);
            lowerer
                .bindings
                .insert("a".into(), binding(1, BindingKind::Int));
            lowerer
                .bindings
                .insert("b".into(), binding(2, BindingKind::Int));
            let expr: Expr = syn::parse_str("majit_int_py_div(a, b)").expect("parse call");
            let out = lowerer.lower_value_expr(&expr);
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("record_int_py_div"),
                "the constant-divisor expander matches this oopspec, got:\n{text}"
            );
            assert!(
                !text.contains("record_int_floordiv"),
                "must not take the C-truncating residual:\n{text}"
            );
        }

        #[test]
        fn a_literal_left_operand_of_an_ordered_compare_mirrors_it() {
            let (lowerer, out) = lower("5 < n", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("IntGt"),
                "`5 < n` must become `n > 5`, got:\n{text}"
            );
        }

        #[test]
        fn a_subtraction_keeps_its_literal_left_operand() {
            let (lowerer, out) = lower("5 - n", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("IntSub"),
                "`-` is not symmetric, got:\n{text}"
            );
        }

        #[test]
        fn a_nonzero_literal_keeps_the_binary_compare() {
            let (lowerer, out) = lower("n == 5", |l| {
                l.bindings.insert("n".into(), binding(4, BindingKind::Int));
            });
            assert!(out.is_some());
            let text = emitted(&lowerer);
            assert!(
                text.contains("IntEq") && !text.contains("IntIsZero"),
                "only the zero literal folds, got:\n{text}"
            );
        }
    }
}
