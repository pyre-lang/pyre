use super::lower_value::struct_type_id_tokens;
use super::*;
use syn::spanned::Spanned;

/// Length field of a varsize header (`descr.py` `get_field_arraylen_descr`).
///
/// Prefer the names a `GcArray` length word actually uses. A header with
/// a single `int_fields` entry uses that entry.
pub(super) fn varsize_length_field_name(
    config: &LowererConfig,
    header: &syn::Path,
) -> Option<String> {
    let last = header.segments.last()?.ident.to_string();
    let prefix = format!("{last}::");
    let names: Vec<String> = config
        .int_fields
        .keys()
        .filter_map(|key| key.strip_prefix(&prefix).map(|rest| rest.to_string()))
        .collect();
    for prefer in ["capacity", "length", "len"] {
        if names.iter().any(|name| name == prefer) {
            return Some(prefer.to_string());
        }
    }
    if names.len() == 1 {
        return Some(names[0].clone());
    }
    // A header declared only through `array_fields` still has a length
    // word. `capacity` is the name `GcArray` headers use when `int_fields`
    // does not name it (`rewrite_op_getarraysize`).
    if struct_is_array_header(config, header) {
        return Some("capacity".to_string());
    }
    None
}

fn struct_is_array_header(config: &LowererConfig, header: &syn::Path) -> bool {
    let last = header.segments.last().map(|seg| &seg.ident);
    config
        .array_headers
        .values()
        .any(|path| path.segments.last().map(|seg| &seg.ident) == last)
}

/// `rewrite_op_getarraysize`: a read of the header's length word is
/// `arraylen_gc`, including when that word is not listed in `int_fields`.
pub(super) fn is_varsize_length_member(
    config: &LowererConfig,
    header: &syn::Path,
    member: &str,
) -> bool {
    if !struct_is_array_header(config, header) {
        return false;
    }
    varsize_length_field_name(config, header).as_deref() == Some(member)
}

/// `rewrite_op_malloc_varsize`: pointer and struct elements are cleared.
/// An integer or `f64` element type (`i64`, `usize`, …) is `new_array`.
pub(super) fn header_items_are_pointers(config: &LowererConfig, header: &syn::Path) -> bool {
    let last = header
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_default();
    let mut saw = false;
    let mut primitive = false;
    for (key, header_path) in &config.array_headers {
        let header_last = header_path
            .segments
            .last()
            .map(|seg| seg.ident.to_string())
            .unwrap_or_default();
        if header_last != last {
            continue;
        }
        saw = true;
        if let Some((_, _, elem)) = config.array_fields.get(key)
            && (path_is_int_elem(elem) || path_is_f64(elem))
        {
            primitive = true;
        }
    }
    saw && !primitive
}

fn header_element_path<'a>(config: &'a LowererConfig, header: &syn::Path) -> Option<&'a syn::Path> {
    let last = header.segments.last().map(|seg| seg.ident.to_string())?;
    for (key, header_path) in &config.array_headers {
        let header_last = header_path
            .segments
            .last()
            .map(|seg| seg.ident.to_string())
            .unwrap_or_default();
        if header_last == last
            && let Some((_, _, elem)) = config.array_fields.get(key)
        {
            return Some(elem);
        }
    }
    None
}

/// `descr.py` `get_array_descr` for a `GcArray` or an array inlined after
/// `header`: basesize is `offset_of!(Header, items)`, lendescr offset is
/// the length field.
pub(super) fn gc_varsize_descr_tokens(config: &LowererConfig, header: &syn::Path) -> TokenStream {
    let len_name =
        varsize_length_field_name(config, header).unwrap_or_else(|| "capacity".to_string());
    let len_ident = syn::Ident::new(&len_name, proc_macro2::Span::call_site());
    let pointers = header_items_are_pointers(config, header);
    // A float item is the descr `getarrayitem_gc_f` reads, so the
    // allocation and the reads name one descriptor.
    if !pointers && header_element_path(config, header).is_some_and(path_is_f64) {
        return float_array_descr_tokens(Some(config), &Some(header.clone()));
    }
    // `symbolic.py` `get_array_token`: `itemsize = sizeof(SUBARRAY.OF)`.
    let (itemsize, is_signed) = if pointers {
        (quote! { ::core::mem::size_of::<usize>() }, quote! { false })
    } else if let Some(elem) = header_element_path(config, header) {
        (
            quote! { ::core::mem::size_of::<#elem>() },
            quote! { (<#elem>::MIN as i128) < 0 },
        )
    } else {
        (quote! { ::core::mem::size_of::<i64>() }, quote! { true })
    };
    let type_id = struct_type_id_tokens(header, true);
    let gc_type_id = quote! { majit_metainterp::__majit_struct_gc_type_id::<#header>() };
    quote! {
        __builder.add_gc_varsize_array_descr(
            ::core::mem::offset_of!(#header, items),
            ::core::mem::offset_of!(#header, #len_ident),
            #itemsize,
            #pointers,
            #is_signed,
            #type_id,
            #gc_type_id,
        )
    }
}

/// Descr for an `f64` array item. A header uses `offset_of!(Header, items)`
/// as `base_size` (`symbolic.py` `get_array_token`).
fn float_array_descr_tokens(
    config: Option<&LowererConfig>,
    header: &Option<syn::Path>,
) -> TokenStream {
    let Some(header) = header else {
        return quote! { __builder.add_raw_float_array_descr() };
    };
    let Some(config) = config else {
        return quote! { __builder.add_raw_float_array_descr() };
    };
    let len_name =
        varsize_length_field_name(config, header).unwrap_or_else(|| "capacity".to_string());
    let len_ident = syn::Ident::new(&len_name, proc_macro2::Span::call_site());
    let type_id = struct_type_id_tokens(header, true);
    let gc_type_id = quote! { majit_metainterp::__majit_struct_gc_type_id::<#header>() };
    quote! {
        __builder.add_gc_varsize_float_array_descr(
            ::core::mem::offset_of!(#header, items),
            ::core::mem::offset_of!(#header, #len_ident),
            #type_id,
            #gc_type_id,
        )
    }
}

impl<'c> Lowerer<'c> {
    /// The virtualizable object in an access.
    ///
    /// `jtransform.py` `rewrite_op_getfield` matches the virtualizable
    /// variable itself (`frame.field`). A portal stores that object in a
    /// `ref` state field, spelled `state.frame.field`; both are the same
    /// object, so both take `getfield_vable_*` / `getarrayitem_vable_*`.
    fn expr_is_vable_object(&self, expr: &Expr) -> bool {
        let Some(config) = self.config else {
            return false;
        };
        let Some(vable_var) = config.vable_var.as_deref() else {
            return false;
        };
        if expr_matches_local_name(expr, vable_var) {
            return true;
        }
        let Expr::Field(field) = expr else {
            return false;
        };
        if !expr_matches_local_name(&field.base, "state") {
            return false;
        }
        let Some(member) = named_member(&field.member) else {
            return false;
        };
        member == vable_var && config.state_ref_scalars.contains_key(vable_var)
    }
}

fn path_is_f64(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|segment| segment.ident == "f64")
}

/// Primitive integer element of a headered array (`CelIntWords::items`).
/// A header does not make the element a GC pointer. `bool` is an int item
/// of size 1, unsigned.
fn path_is_int_elem(path: &syn::Path) -> bool {
    const INTS: &[&str] = &[
        "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize", "bool",
    ];
    path.segments
        .last()
        .is_some_and(|segment| INTS.contains(&segment.ident.to_string().as_str()))
}

/// The `(field_size, is_signed)` a struct-layout registration reports for one
/// field, plus a compile-time check that the declaration matches the Rust
/// struct.  `descr.py get_field_descr` derives both from `FIELDTYPE`,
/// but the macro sees only the field's name at the access site, so a sub-word
/// integer field has to be named in `int_fields` to be registered as one.
/// Anything undeclared is registered as an `i64` — and has to be that wide to
/// stay so, which the returned witness enforces. `i64`, not `usize`: the two
/// part company on a 32-bit target such as wasm32, and it is the eight-byte
/// word that is claimed here.
pub(super) fn field_scalar_tokens(
    config: &LowererConfig,
    key: &str,
    struct_path: &syn::Path,
    member: &syn::Member,
) -> (TokenStream, TokenStream, TokenStream) {
    // Every consultation THIS lowering makes reaches the maps through here, so
    // one line records which declared keys an access site asked about.
    // Recorded whether or not the key is declared: the report is over the
    // DECLARED keys, and an undeclared one describes nothing to begin with.
    //
    // Not the only reader of `ref_fields`, and the difference bounds what the
    // report can claim.  `RefFieldRewriter` consults it too, to rewrite the
    // CONCRETE body's `x.field` into a deref of the raw-pointer carrier.  Both
    // walk the same source, so they normally agree — they diverge exactly where
    // this lowering stopped early and the rewriter did not, which is a degraded
    // arm.  That is why an unconsulted key is reported rather than rejected,
    // and why the gate prints the degraded arms beside it.
    config
        .consulted_field_keys
        .borrow_mut()
        .insert(key.to_string());
    if config.float_fields.contains(key) {
        // `descr.py` `get_type_flag` for `lltype.Float`: eight bytes, not an
        // integer. The access site still passes `is_float` separately.
        return (
            quote! { ::core::mem::size_of::<f64>() },
            quote! { false },
            quote! {
                const _: () = {
                    const fn __majit_float_field_width<T>(_: fn(&#struct_path) -> &T) {
                        assert!(
                            ::core::mem::size_of::<T>() == ::core::mem::size_of::<f64>(),
                            "float field width does not match f64",
                        );
                    }
                    __majit_float_field_width(|__s: &#struct_path| &__s.#member);
                };
            },
        );
    }
    match config.int_fields.get(key) {
        Some((ty, signed)) => (
            quote! { ::core::mem::size_of::<#ty>() },
            quote! { #signed },
            // `jtransform.py` `rewrite_op_getfield` reads the field's integer
            // storage. A `#[repr(u8)]` tag is that integer, not the enum's
            // name, so the witness is the width: the declared type and the
            // field must be the same size. A wider or narrower declaration
            // still fails here.
            quote! {
                const _: () = {
                    const fn __majit_int_field_width<T, U>(_: fn(&#struct_path) -> &T) {
                        assert!(
                            ::core::mem::size_of::<T>() == ::core::mem::size_of::<U>(),
                            "int_fields width does not match the field",
                        );
                    }
                    __majit_int_field_width::<_, #ty>(|__s: &#struct_path| &__s.#member);
                };
            },
        ),
        // `scalar_size`'s own default for a non-`Ref` field: the `i64` storage,
        // which is 8 on every target and NOT the machine word (a `usize` here
        // would report 4 on wasm32).
        //
        // That default is a CLAIM about the field — eight bytes — and it used to
        // be the one claim here that carried no witness.  The declared arm above
        // cannot drift from the struct; the undeclared arm could say eight bytes
        // about a `u32` and register a descr four bytes too wide, with the
        // sub-word range a declaration exists to buy silently gone.  Omission is
        // the dangerous direction precisely because it is spelled as nothing at
        // all, so the default witnesses itself too.
        //
        // WIDTH, not type identity, because width is what the claim is about.
        // The storage this registers is reached as a raw eight-byte word, and a
        // `#[repr(transparent)]` newtype over `i64` — a tagged value word is the
        // usual one — is exactly that word.  Demanding the field spell `i64`
        // would reject those for no defect.  What stays uncheckable either way
        // is the sign, which no Rust type-level test can recover from a field
        // the macro only knows by name.
        None => (
            quote! { ::core::mem::size_of::<i64>() },
            quote! { true },
            match ref_field_witness_tokens(&config.ref_fields, key, struct_path, member) {
                // A ref field is a pointer word and carries its own witness.
                witness if !witness.is_empty() => witness,
                _ => {
                    let message = format!(
                        "`{key}` is not the eight-byte scalar an undeclared field is \
                         registered as. Name its Rust integer type in `int_fields`, or its \
                         pointee in `ref_fields` / its element type in `array_fields`, so \
                         the emitted descr reports the field's own width instead of this \
                         default.",
                    );
                    quote! {
                        const _: () = {
                            // Names the field's type without spelling it, which
                            // is the only handle available here: the macro sees
                            // the member, not its declaration.
                            //
                            // Through a reference, not by value. `|s| s.field`
                            // returns `T` and so moves out of a shared borrow,
                            // which only compiles for a `Copy` field — an
                            // eight-byte field that is not `Copy` would fail
                            // expansion with a borrow error naming generated
                            // code, for a struct with nothing wrong with it.
                            const fn __field_width<T>(_: fn(&#struct_path) -> &T) -> usize {
                                ::core::mem::size_of::<T>()
                            }
                            assert!(
                                __field_width(|__s: &#struct_path| &__s.#member)
                                    == ::core::mem::size_of::<i64>(),
                                #message,
                            );
                        };
                    }
                }
            },
        ),
    }
}

/// A compile-time check that a `ref_fields` entry describes the field it names.
///
/// The lowering trusts the declaration twice and verifies it nowhere: the read
/// becomes `getfield_gc_r` into the ref bank, and the binding's `struct_type`
/// is what the NEXT hop resolves `offset_of!` against.  So a field that is not
/// a pointer to the declared pointee produces either a ref-bank read of
/// something that is not a reference, or an offset computed in the wrong
/// struct, and nothing between the declaration and the emitted descr says so.
///
/// `usize` is admitted alongside the two raw-pointer spellings because it is
/// the sanctioned carrier for a pointer whose declaring crate would rather not
/// name the pointee's type.  A carrier's pointee is not checkable, which is the
/// price of the carrier — what survives for it is that the field is
/// pointer-width and pointer-kind, not an `i64` or a `u32` that drifted into
/// the ref map.
///
/// The witness names the pointee rather than a whole pointer type so that
/// `*mut T` and `*const T` both satisfy it, mirroring `emit_array_field_base`.
///
/// Where this adds coverage, measured rather than assumed.  On the
/// `#[jit_inline]` path a drifted pointee on a raw-pointer field already fails
/// the build: the concrete rewriter types the loaded value against the
/// declaration and reports E0308.  On the `#[jit_interp]` state-field path it
/// does not — a machine declaring `Holder::link => Wrong` over a
/// `link: *mut Holder` compiles clean and faults at run time.  That path is
/// what this witness closes; on the inline path it is a second, earlier line.
///
/// Empty for a key `ref_fields` does not declare, and that is a decision, not
/// an omission: an undeclared field reads into the Int bank, and stable Rust
/// has no way to assert that a type is *not* a pointer.  Catching that half
/// takes a mechanism that works by disagreement instead of by declaration —
/// `Assembler::register_struct_layout`'s conflict check, which reports one word
/// registered as a pointer at one emit site and as a scalar at another.
///
/// It is not a general second line for this one, and the reason is worth
/// stating: each jitcode gets its OWN layout map, so two *declarations* never
/// meet there.  What meets is two emit sites within one jitcode — the same
/// member reached by two lowering paths.  A machine whose sole access to an
/// undeclared pointer field goes through one path stays undetected by both
/// mechanisms.
fn ref_field_witness_tokens(
    ref_fields: &HashMap<String, (syn::Path, Ident, syn::Path)>,
    key: &str,
    struct_path: &syn::Path,
    member: &syn::Member,
) -> TokenStream {
    let Some((_, _, pointee)) = ref_fields.get(key) else {
        return quote! {};
    };
    quote! {
        const _: () = {
            trait __MajitRefField {}
            impl __MajitRefField for *mut #pointee {}
            impl __MajitRefField for *const #pointee {}
            impl __MajitRefField for usize {}
            #[allow(dead_code)]
            fn __majit_ref_field_witness(__s: &#struct_path) {
                fn __accept<T: __MajitRefField>(_: T) {}
                __accept(__s.#member);
            }
        };
    }
}

/// The `(base_size, len_offset, witness)` a `pool_arrays` declaration reports
/// for its array, and the compile-time checks that the declaration describes
/// the struct it names.
///
/// Both numbers are `offset_of!` on the declaration's own field names, so a
/// field reordered ahead of the items moves the read with it instead of leaving
/// it behind.  The witnesses cover what an offset cannot: that `items` really
/// is an array of pointer-width elements, and that `len` really is a `usize` —
/// the width the descr's lendescr reads it at.  Without the second one, a `u32`
/// length word would be read as a machine word with its top half taken from
/// whatever follows.
///
/// The element witness needs the declared pointee, so it is emitted only for a
/// declaration that names one.  `offset_of!` is unconditional.
fn pool_array_layout_tokens(
    entry: &super::PoolArrayLowering,
) -> (TokenStream, TokenStream, TokenStream) {
    let struct_path = &entry.struct_path;
    let items = &entry.items_field;
    let base_size = quote! { ::core::mem::offset_of!(#struct_path, #items) };
    let (len_offset, len_witness) = match &entry.len_field {
        Some(len) => (
            quote! { ::core::option::Option::Some(::core::mem::offset_of!(#struct_path, #len)) },
            quote! {
                const _: fn(&#struct_path) -> usize = |__s| __s.#len;
            },
        ),
        None => (quote! { ::core::option::Option::None }, quote! {}),
    };
    let element_witness = match &entry.element_type {
        Some(element) => quote! {
            const _: fn(&#struct_path) -> *mut #element = |__s| __s.#items[0];
        },
        None => quote! {},
    };
    (
        base_size,
        len_offset,
        quote! { #len_witness #element_witness },
    )
}

/// Where the object holding the array base pointer comes from.
///
/// A local `ref_params` binding is already in a register, but a `ref(T)` state
/// scalar still has to be read out of the state frame, and that read emits.
/// Keeping the two apart is what lets [`JitCodeLowerer::match_array_field_base`]
/// stay non-emitting for both.
enum ArrayFieldHolder {
    /// An already-materialized ref binding.
    Reg(u16),
    /// `state.<ref_scalar>`, to be read at emit time.
    StateRef(Expr),
}

/// `jtransform.py` `rewrite_op_getarrayitem`: `_immutable_fields_ = ['x[*]']`
/// (`ImmutableRank::ImmutableArray`) makes the load `getarrayitem_gc_*_pure`.
/// The check runs while the jitcode is built, against the struct's own
/// `__MAJIT_IMMUTABLE_FIELDS` (`rclass.py` `_parse_field_list`).
fn immutable_array_item_pure(struct_path: &syn::Path, member: &syn::Member) -> TokenStream {
    let name = named_member(member).unwrap_or_default();
    let starred = format!("{name}[*]");
    let quasi = format!("{name}?[*]");
    quote! {
        {
            #[allow(unused_imports)]
            use majit_metainterp::MajitImmutableFields as _;
            let __fields = <#struct_path>::__MAJIT_IMMUTABLE_FIELDS;
            __fields.split(',').any(|__entry| {
                let __entry = __entry.trim();
                __entry == #starred || __entry == #quasi
            })
        }
    }
}

/// A `<ref binding or state ref scalar>.<field>` access that `array_fields`
/// declares, resolved but not yet emitted.  See
/// [`JitCodeLowerer::match_array_field_base`].
struct ArrayFieldBase {
    holder: ArrayFieldHolder,
    struct_path: syn::Path,
    member: syn::Member,
    element_type: syn::Path,
    /// Header whose `items` field is element 0. Pointer elements; the
    /// integer path leaves this empty.
    header: Option<syn::Path>,
}

impl<'c> Lowerer<'c> {
    /// Read an immediate operand byte from the green bytecode array:
    /// `program[<index>]`.  Mirrors the dispatch-top opcode fetch
    /// (`try_lower_opcode_fetch_stmt`) so the same `getarrayitem_gc_i_pure`
    /// is emitted for operand reads inside opcode arm bodies.  RPython
    /// `pyopcode.py ord(co_code[next_instr])`: the bytecode array and
    /// `pc` are green, so the optimizer constant-folds the load.
    ///
    /// The env parameter name is the macro convention `program`, matching
    /// `try_lower_opcode_fetch_stmt`'s hard-coded recognition.
    pub(super) fn lower_env_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        // The constant-fold this read relies on ("array + index are green",
        // `pyopcode.py:171 ord(co_code[next_instr])`) only holds when `pc` is
        // a declared green — PyPy `tl.py greens=['pc','code']` makes both the
        // bytecode array and the instruction pointer green, so the load folds
        // away.  With a red `pc` the read does not fold and the
        // surrounding state-field dispatch loop cannot close on it; leave the
        // operand read unlowered so the arm aborts and the interpreter handles
        // that opcode (the pre-green-pc behaviour).
        if !self.config.map(pc_is_green).unwrap_or(false) {
            return None;
        }
        let Expr::Index(index_expr) = expr else {
            return None;
        };
        if !expr_matches_local_name(&index_expr.expr, "program") {
            return None;
        }
        let program_binding = self.bindings.get("program").cloned()?;
        if !matches!(program_binding.kind, BindingKind::Ref) {
            return None;
        }
        let program_reg = program_binding.reg;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = idx_binding.reg;
        let result_reg = self.alloc_reg();
        // The descr scales the index by the env element size (see
        // `env_array_descr_expr`): `&[u8]` reads a raw byte, a wider env
        // (`&[i64]`) reads the element at byte offset `size_of::<elem>() *
        // index`. Shared with the opcode-fetch sites so operands and opcodes
        // read at the same stride.
        let descr_tok = env_array_descr_expr(self.config);
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(program_reg), Register::int(index_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! {
                let __descr_idx = #descr_tok;
                // Same array and same immutability as the dispatch-top fetch,
                // so the same always-pure spelling.
                __builder.getarrayitem_gc_i_pure(
                    #result_reg as u16,
                    #program_reg as u16,
                    #index_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    pub(super) fn lower_vable_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !self.expr_is_vable_object(&field.base) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(field_index, field_type) = config.vable_fields.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let fi = field_index as u16;
        let binding = self.lower_value_expr(&assign.right)?;
        let src = binding.reg;
        // vable_reg is always Ref (the virtualizable input register); src bank
        // follows `field_type` per `assembler.py:217` argcode mapping.
        // jtransform.py:926 — `-live-` precedes `setfield_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        match field_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::ref_(src)], vec![]),
                quote! { __builder.vable_setfield_ref_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::float(src)], vec![]),
                quote! { __builder.vable_setfield_float_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::int(src)], vec![]),
                quote! { __builder.vable_setfield_int_with_base(#vable_reg, #fi, #src); },
            ),
        }
        Some(())
    }

    /// RPython jtransform.py:794 `setarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i] = val` and emits vable_setarrayitem.
    pub(super) fn lower_vable_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        // LHS: frame.array_field[index]
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !self.expr_is_vable_object(&field.base) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let ai = array_index as u16;

        // Lower index and value
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        // vable_reg: Ref. idx_reg: Int (array index). val_reg: bank by item_type.
        // jtransform.py:798 — `-live-` precedes `setarrayitem_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        match item_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::ref_(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_ref_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::float(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_float_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::int(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_int_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
        }
        Some(())
    }

    /// Recognizes `frame.array_field[index] += value` and the other numeric
    /// compound assignments. Rust evaluates `index` once, so the lowering
    /// shares one index register between the vable read and write.
    pub(super) fn lower_vable_array_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let index_expr = match &*binary.left {
            Expr::Index(index) => index,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !self.expr_is_vable_object(&field.base) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let opcode = match item_type {
            ValueKind::Float => opcode_for_assign_binop_f(&binary.op)?,
            ValueKind::Int => opcode_for_assign_binop(&binary.op)?,
            ValueKind::Ref => return None,
        };
        let vable_reg = self.vable_base_reg()?;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let idx_reg = idx_binding.reg;
        let rhs = self.lower_value_expr(&binary.right)?;
        match item_type {
            ValueKind::Float if !matches!(rhs.kind, BindingKind::Float) => return None,
            ValueKind::Int if !matches!(rhs.kind, BindingKind::Int) => return None,
            ValueKind::Ref => unreachable!(),
            _ => {}
        }
        let ai = array_index as u16;
        let lhs_reg = self.alloc_reg();

        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(vable_reg), Register::int(idx_reg)],
                    vec![Register::float(lhs_reg)],
                ),
                quote! { __builder.vable_getarrayitem_float_with_base(#lhs_reg, #vable_reg, #ai, #idx_reg); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(vable_reg), Register::int(idx_reg)],
                    vec![Register::int(lhs_reg)],
                ),
                quote! { __builder.vable_getarrayitem_int_with_base(#lhs_reg, #vable_reg, #ai, #idx_reg); },
            ),
            ValueKind::Ref => unreachable!(),
        }

        let dst = self.alloc_reg();
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopF,
                    vec![Register::float(lhs_reg), Register::float(rhs.reg)],
                    vec![Register::float(dst)],
                ),
                binop_f_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[lhs_reg, rhs.reg]),
                    vec![Register::int(dst)],
                ),
                binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Ref => unreachable!(),
        }

        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(vable_reg),
                        Register::int(idx_reg),
                        Register::float(dst),
                    ],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_float_with_base(#vable_reg, #ai, #idx_reg, #dst); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(vable_reg),
                        Register::int(idx_reg),
                        Register::int(dst),
                    ],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_int_with_base(#vable_reg, #ai, #idx_reg, #dst); },
            ),
            ValueKind::Ref => unreachable!(),
        }
        Some(())
    }

    /// Recognizes `state.field = expr` for scalar state fields.
    pub(super) fn lower_state_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if config.vable_var.as_deref() == Some("state")
            && config.vable_fields.contains_key(&member_name)
        {
            return self.lower_vable_field_write(expr);
        }
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            let src = binding.reg;
            // store_state_field/di — `src` is Int per assembler.py:217 'i' argcode.
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::int(src)], vec![]),
                quote! { __builder.store_state_field(#fi, #src); },
            );
            return Some(());
        }
        // ref(T) scalar: the RHS must lower to a ref binding (another ref
        // state read or a residual ref-returning call).
        if let Some((field_index, _)) = config.state_ref_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = *field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            if !matches!(binding.kind, BindingKind::Ref) {
                return None;
            }
            let src = binding.reg;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::ref_(src)], vec![]),
                quote! { __builder.store_state_field_ref(#fi, #src); },
            );
            return Some(());
        }
        if let Some(&field_index) = config.state_float_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            if !matches!(binding.kind, BindingKind::Float) {
                return None;
            }
            let src = binding.reg;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::float(src)], vec![]),
                quote! { __builder.store_state_field_float(#fi, #src); },
            );
            return Some(());
        }
        None
    }

    /// Recognizes `state.field += expr` for scalar state fields.
    pub(super) fn lower_state_field_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let field = match &*binary.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            let opcode = opcode_for_assign_binop(&binary.op)?;
            let lhs = self.lower_state_field_read(&binary.left)?;
            if !matches!(lhs.kind, BindingKind::Int) {
                return None;
            }
            let lhs_reg = lhs.reg;
            // A `Constant` right operand stays in the operation
            // (`assembler.py` `emit_const`).
            let rhs = match self.int_constant_operand(&binary.right) {
                Some(value) if binop_i_accepts_constant(&opcode) => Ok(value),
                _ => {
                    let rhs = self.lower_value_expr(&binary.right)?;
                    if !matches!(rhs.kind, BindingKind::Int) {
                        return None;
                    }
                    Err(rhs.reg)
                }
            };
            let dst = self.alloc_reg();
            let (reads, tokens) = match rhs {
                Ok(value) => (
                    Register::ints(&[lhs_reg]),
                    binop_i_const_emit_tokens(dst, &opcode, lhs_reg, &value)?,
                ),
                Err(rhs_reg) => (
                    Register::ints(&[lhs_reg, rhs_reg]),
                    binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs_reg),
                ),
            };
            self.emit_op(
                OpMeta::linear(OpKind::BinopI, reads, vec![Register::int(dst)]),
                tokens,
            );
            let fi = field_index as u16;
            if config.vable_var.as_deref() == Some("state")
                && let Some(&(index, _)) = config.vable_fields.get(&member_name)
            {
                let base = self.vable_base_reg()?;
                let index = index as u16;
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![Register::ref_(base), Register::int(dst)],
                        vec![],
                    ),
                    quote! { __builder.vable_setfield_int_with_base(#base, #index, #dst); },
                );
            } else {
                self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
                self.emit_op(
                    OpMeta::linear(OpKind::StateField, vec![Register::int(dst)], vec![]),
                    quote! { __builder.store_state_field(#fi, #dst); },
                );
            }
            return Some(());
        }

        let &field_index = config.state_float_scalars.get(&member_name)?;
        let opcode = opcode_for_assign_binop_f(&binary.op)?;
        let lhs = self.lower_state_field_read(&binary.left)?;
        let rhs = self.lower_value_expr(&binary.right)?;
        if !matches!(lhs.kind, BindingKind::Float) || !matches!(rhs.kind, BindingKind::Float) {
            return None;
        }
        let dst = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopF,
                vec![Register::float(lhs_reg), Register::float(rhs_reg)],
                vec![Register::float(dst)],
            ),
            binop_f_emit_tokens(dst, &opcode, lhs_reg, rhs_reg),
        );
        let fi = field_index as u16;
        if config.vable_var.as_deref() == Some("state")
            && let Some(&(index, _)) = config.vable_fields.get(&member_name)
        {
            let base = self.vable_base_reg()?;
            let index = index as u16;
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base), Register::float(dst)],
                    vec![],
                ),
                quote! { __builder.vable_setfield_float_with_base(#base, #index, #dst); },
            );
        } else {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::float(dst)], vec![]),
                quote! { __builder.store_state_field_float(#fi, #dst); },
            );
        }
        Some(())
    }

    /// Recognizes `state.array_field[index] += value` and the other numeric
    /// compound assignments on a plain `[int]`/`[float]` state array. Rust
    /// evaluates `index` once, so the lowering shares one index register
    /// between the raw get and set. Python's `a[i] += x` is a load, a binop,
    /// and a store; `jtransform.py` `rewrite_op_getarrayitem` /
    /// `rewrite_op_setarrayitem` raw arms emit `getarrayitem_raw_{i,f}` /
    /// `setarrayitem_raw_{i,f}`. Shape matches
    /// [`Self::lower_vable_array_update`]. An unsupported op or a kind
    /// mismatch returns `None` the same way that sibling does.
    pub(super) fn lower_state_array_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let index_expr = match &*binary.left {
            Expr::Index(index) => index,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        // `[int; virt]` arrays are NOT handled here — they lower through
        // `lower_vable_array_update`. Bail before lowering the index/rhs so
        // they are not emitted twice when the dispatch falls through.
        if config.state_virt_arrays.contains_key(&member_name) {
            return None;
        }
        let &(_, item_kind) = config.state_arrays.get(&member_name)?;
        let opcode = match item_kind {
            ValueKind::Float => opcode_for_assign_binop_f(&binary.op)?,
            ValueKind::Int => opcode_for_assign_binop(&binary.op)?,
            ValueKind::Ref => return None,
        };
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let idx_reg = idx_binding.reg;
        let rhs = self.lower_value_expr(&binary.right)?;
        match item_kind {
            ValueKind::Float if !matches!(rhs.kind, BindingKind::Float) => return None,
            ValueKind::Int if !matches!(rhs.kind, BindingKind::Int) => return None,
            ValueKind::Ref => unreachable!(),
            _ => {}
        }
        // The `{ptr, len}` words are declared int arguments of this
        // JitCode (`MIFrame.setup_call` / `bhimpl_inline_call_ir_v`). The
        // get/set are the `dum_nocheck` item ops off the pointer
        // (`rlist.py` `rtype_getitem` / `rtype_setitem`).
        let base_reg = self.plain_array_ptr_reg(&member_name)?;
        let lhs_reg = self.alloc_reg();

        match item_kind {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::int(base_reg), Register::int(idx_reg)],
                    vec![Register::float(lhs_reg)],
                ),
                quote! {
                    let __descr_idx = __builder.add_raw_float_array_descr();
                    __builder.getarrayitem_raw_f(#lhs_reg, #base_reg, #idx_reg, __descr_idx);
                },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::int(base_reg), Register::int(idx_reg)],
                    vec![Register::int(lhs_reg)],
                ),
                quote! {
                    let __descr_idx = __builder.add_raw_int_array_descr_signed(
                        ::core::mem::size_of::<i64>(),
                        true,
                    );
                    __builder.getarrayitem_raw_i(#lhs_reg, #base_reg, #idx_reg, __descr_idx);
                },
            ),
            ValueKind::Ref => unreachable!(),
        }

        let dst = self.alloc_reg();
        match item_kind {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopF,
                    vec![Register::float(lhs_reg), Register::float(rhs.reg)],
                    vec![Register::float(dst)],
                ),
                binop_f_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[lhs_reg, rhs.reg]),
                    vec![Register::int(dst)],
                ),
                binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Ref => unreachable!(),
        }

        match item_kind {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![
                        Register::int(base_reg),
                        Register::int(idx_reg),
                        Register::float(dst),
                    ],
                    vec![],
                ),
                quote! {
                    let __descr_idx = __builder.add_raw_float_array_descr();
                    __builder.setarrayitem_raw_f(#base_reg, #idx_reg, #dst, __descr_idx);
                },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![
                        Register::int(base_reg),
                        Register::int(idx_reg),
                        Register::int(dst),
                    ],
                    vec![],
                ),
                quote! {
                    let __descr_idx = __builder.add_raw_int_array_descr_signed(
                        ::core::mem::size_of::<i64>(),
                        true,
                    );
                    __builder.setarrayitem_raw_i(#base_reg, #idx_reg, #dst, __descr_idx);
                },
            ),
            ValueKind::Ref => unreachable!(),
        }
        Some(())
    }

    /// Recognizes `state.array[index] = expr` for array state fields.
    /// Virtualizable arrays bail to the standard vable path. A plain
    /// `[int]`/`[float]` store is `setarrayitem_raw_i` / `setarrayitem_raw_f`
    /// off the ptr identity, the store counterpart of
    /// [`Self::lower_state_array_read`]. `rlist.py` `rtype_setitem` takes
    /// `dum_nocheck` when there is no IndexError handler; majit-translate
    /// strips the bounds `Assert` (`removeassert.py`). The store writes the
    /// native buffer in place (`jtransform.py` `rewrite_op_setarrayitem`
    /// raw arm), so no writeback is needed on exit or abort.
    pub(super) fn lower_state_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        // `[int; virt]` arrays are NOT handled here — they lower through the
        // standard virtualizable path (`lower_vable_array_write`), reached
        // because `LowererConfig::new` registers the state binding as the
        // vable identity var. Bail before lowering the index/value so they
        // are not emitted twice when the dispatch falls through.
        if config.state_virt_arrays.contains_key(&member_name) {
            return None;
        }
        let &(_, item_kind) = config.state_arrays.get(&member_name)?;
        // Assignment evaluates the RHS first, then the place (index).
        let value = self.lower_value_expr(&assign.right)?;
        let value_kind = match item_kind {
            ValueKind::Float => BindingKind::Float,
            ValueKind::Int => BindingKind::Int,
            ValueKind::Ref => return None,
        };
        if value.kind != value_kind {
            return None;
        }
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let idx_reg = idx_binding.reg;
        let value_reg = value.reg;
        // The `{ptr, len}` words are declared int arguments of this
        // JitCode (`MIFrame.setup_call` / `bhimpl_inline_call_ir_v`); this
        // op is the `dum_nocheck` item store off the pointer.
        let base_reg = self.plain_array_ptr_reg(&member_name)?;

        let (value_register, descr, set_raw) = match item_kind {
            ValueKind::Float => (
                Register::float(value_reg),
                quote! { __builder.add_raw_float_array_descr() },
                format_ident!("setarrayitem_raw_f"),
            ),
            ValueKind::Int => (
                Register::int(value_reg),
                quote! {
                    __builder.add_raw_int_array_descr_signed(
                        ::core::mem::size_of::<i64>(),
                        true,
                    )
                },
                format_ident!("setarrayitem_raw_i"),
            ),
            ValueKind::Ref => return None,
        };
        self.emit_op(
            OpMeta::linear(
                OpKind::StateField,
                vec![
                    Register::int(base_reg),
                    Register::int(idx_reg),
                    value_register,
                ],
                vec![],
            ),
            quote! {
                let __descr_idx = #descr;
                __builder.#set_raw(#base_reg, #idx_reg, #value_reg, __descr_idx);
            },
        );
        Some(())
    }

    /// RPython jtransform.py `hint_force_virtualizable`.
    ///
    /// Recognizes `hint_force_virtualizable!(frame)` macro invocation.
    pub(super) fn lower_vable_force(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let _vable_var = config.vable_var.as_ref()?;

        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let hint = classify_virtualizable_hint_syn_path(&mac.mac.path)?;
        if hint != VirtualizableHintKind::ForceVirtualizable {
            return None;
        }
        let arg: Expr = syn::parse2(mac.mac.tokens.clone()).ok()?;
        let binding = self.lower_value_expr(&arg)?;
        let vable_reg = binding.reg;
        // vable_force/r — vable_reg is Ref per assembler.py:217 'r' argcode.
        self.emit_op(
            OpMeta::linear(OpKind::Vable, vec![Register::ref_(vable_reg)], vec![]),
            quote! { __builder.vable_force_with_base(#vable_reg); },
        );
        Some(())
    }

    /// RPython jtransform.py `rewrite_op_hint` — suppress the identity hint
    /// function call.
    ///
    /// `hint_access_directly(frame)` is an identity function whose whole
    /// effect at this layer is to be dropped: upstream's hint arm answers
    /// `access_directly` with a bare `return`, and the flag it records is
    /// consulted only for `fresh_virtualizable` (`is_vable_getfield` reads
    /// `'fresh_virtualizable' in flags`).  So lowering the argument directly
    /// is the whole port.
    ///
    /// `hint_fresh_virtualizable` is NOT dropped here.  It asks the following
    /// field accesses to stay OFF the `getfield_vable_*` / `setfield_vable_*`
    /// path, which upstream arranges by recording the flag and reading it back
    /// in `is_vable_getfield` / `is_vable_setfield`.  This lowering has no
    /// carrier for the flag, so the identity answer keeps every following
    /// access on the vable path -- the opposite of what the hint asks, with no
    /// diagnostic.  Refuse instead: `stmt_contains_fresh_virtualizable_hint`
    /// turns that refusal into a declined body, and the arm runs interpreted.
    pub(super) fn lower_vable_hint_identity_call(&mut self, expr: &Expr) -> Option<Binding> {
        let call = match expr {
            Expr::Call(c) => c,
            _ => return None,
        };
        let func_name = match &*call.func {
            Expr::Path(p) => classify_virtualizable_hint_syn_path(&p.path),
            _ => return None,
        };
        match func_name {
            Some(VirtualizableHintKind::AccessDirectly) => {
                let arg = call.args.first()?;
                self.lower_value_expr(arg)
            }
            _ => None,
        }
    }

    /// Preserve a borrow of the live virtualizable as its existing ref input.
    ///
    /// RPython residual helpers receive the virtualizable object directly;
    /// `call.py::getcalldescr` can then classify writes to redirected fields
    /// as forcing the virtualizable. Rust needs an explicit borrow at the
    /// concrete call site, but that borrow is representation-only: in JitCode
    /// it is the same object reference, not a freshly computed integer
    /// address. Restrict this identity rule to the configured virtualizable so
    /// ordinary Rust references do not acquire ref semantics accidentally.
    pub(super) fn lower_vable_reference(&self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let reference = match expr {
            Expr::Reference(reference) => reference,
            _ => return None,
        };
        if !expr_matches_local_name(&reference.expr, vable_var) {
            return None;
        }
        let binding = self.bindings.get(vable_var)?;
        matches!(binding.kind, BindingKind::Ref).then(|| binding.clone())
    }

    /// RPython jtransform.py:655 `hint(access_directly=True)` /
    /// `hint(fresh_virtualizable=True)`.
    ///
    /// These hints are consumed by the translator — jtransform suppresses
    /// them (returns None = no opcode generated). The codewriter has already
    /// rewritten field accesses to use vable_getfield/setfield, so the
    /// access_directly hint is redundant at this point.
    ///
    /// In majit, the Lowerer recognizes these macro calls and emits nothing,
    /// which matches RPython's behavior exactly.
    pub(super) fn lower_vable_hint_suppress(&self, expr: &Expr) -> Option<()> {
        let _config = self.config?;
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        // Only `access_directly`.  `fresh_virtualizable` is refused by
        // `stmt_contains_fresh_virtualizable_hint` rather than suppressed --
        // see `lower_vable_hint_identity_call` for why dropping it is wrong.
        match classify_virtualizable_hint_syn_path(&mac.mac.path) {
            Some(VirtualizableHintKind::AccessDirectly) => Some(()),
            _ => None,
        }
    }

    // ── conditional_call / record_known_result JIT op emission ──────

    /// RPython jtransform.py rewrite_op_getfield `getfield_vable_*`.
    ///
    /// Recognizes `frame.field` where `frame` is the virtualizable variable
    /// and `field` is a declared virtualizable scalar field.
    pub(super) fn lower_vable_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;

        if let Expr::Field(field) = expr {
            if !self.expr_is_vable_object(&field.base) {
                return None;
            }
            let member_name = named_member(&field.member)?;

            if let Some(&(field_index, field_type)) = config.vable_fields.get(&member_name) {
                let vable_reg = self.vable_base_reg()?;
                let reg = self.alloc_reg();
                let fi = field_index as u16;
                // vable_reg is Ref; result `reg` bank follows field_type.
                // jtransform.py:845 — `-live-` precedes `getfield_vable_*`.
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                let vable_r = Register::ref_(vable_reg);
                let kind = match field_type {
                    ValueKind::Ref => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::ref_(reg)],
                            ),
                            quote! { __builder.vable_getfield_ref_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Ref
                    }
                    ValueKind::Float => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::float(reg)],
                            ),
                            quote! { __builder.vable_getfield_float_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Float
                    }
                    ValueKind::Int => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::int(reg)],
                            ),
                            quote! { __builder.vable_getfield_int_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Int
                    }
                };
                return Some(Binding {
                    reg,
                    kind,
                    depends_on_stack: false,
                    struct_type: config
                        .state_ref_scalars
                        .get(&member_name)
                        .map(|(_, path)| path.clone()),
                });
            }
        }
        None
    }

    /// RPython jtransform.py:760 `getarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i]` where `frame` is the virtualizable
    /// variable and `locals_w` is a declared virtualizable array field.
    pub(super) fn lower_vable_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;

        // Pattern: Expr::Index where base is Expr::Field on the virtualizable
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !self.expr_is_vable_object(&field.base) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;

        // Lower the index expression to a register
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;

        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // vable_reg: Ref. idx_reg: Int. result `reg` bank by item_type.
        // jtransform.py:764 — `-live-` precedes `getarrayitem_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        let kind = match item_type {
            ValueKind::Ref => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::ref_(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_ref_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Ref
            }
            ValueKind::Float => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::float(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_float_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Float
            }
            ValueKind::Int => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_int_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Int
            }
        };
        Some(Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py:815 `arraylen_vable`.
    ///
    /// Recognizes `frame.locals_w.len()` for declared virtualizable arrays.
    pub(super) fn lower_vable_array_len(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let call = match expr {
            Expr::MethodCall(call) => call,
            _ => return None,
        };
        if call.method != "len" || !call.args.is_empty() {
            return None;
        }
        let field = match &*call.receiver {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !self.expr_is_vable_object(&field.base) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &array_index = config.vable_arrays.get(&member_name).map(|(idx, _)| idx)?;
        let vable_reg = self.vable_base_reg()?;
        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // jtransform.py:814 — `-live-` precedes `arraylen_vable`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        // vable_arraylen reads vable_reg (Ref) and writes the length to an int reg.
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(vable_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.vable_arraylen_with_base(#reg, #vable_reg, #ai); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// Recognizes `state.field` for scalar state fields.
    pub(super) fn lower_state_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let field = match expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if config.vable_var.as_deref() == Some("state")
            && config.vable_fields.contains_key(&member_name)
        {
            return self.lower_vable_field_read(expr);
        }
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = field_index as u16;
            let reg = self.alloc_reg();
            // load_state_field reads the field at its int identity slot into
            // int `reg`.  Declare the identity slot as a read (liveness.py:67
            // adds every Register arg to the alive set) so the backward
            // liveness walk keeps the slot live across every downstream
            // `-live-` marker — without it the marker omits the slot, the
            // blackhole resume seeder never restores it, and forward
            // re-execution reads garbage.  Mirrors `store_state_field`
            // declaring its source and `vable_field_read` declaring `vable_r`.
            // The slot sits at `int_identity_base() + fi`, past the dispatch
            // JitCode's int argument (`pc` at i0).
            let slot = config.int_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    self.identity_use(Register::int(slot)),
                    vec![Register::int(reg)],
                ),
                quote! { __builder.load_state_field(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        // ref(T) scalar: read into the ref register bank so a subsequent
        // getfield_gc reads its struct base from a real ref value.
        if let Some((field_index, struct_path)) = config.state_ref_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = *field_index as u16;
            let reg = self.alloc_reg();
            // Declare the ref identity slot as a read so the backward
            // liveness walk keeps it live across guards (mirrors the int
            // load_state_field). Without it the slot drops from interior
            // -live- markers and the blackhole resume seeder leaves
            // registers_r[ref_scalar_slot(fi)] uninitialized. The slot
            // sits at `ref_identity_base() + fi`, past the dispatch
            // JitCode's ref-bank arguments.
            let slot = config.ref_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    self.identity_use(Register::ref_(slot)),
                    vec![Register::ref_(reg)],
                ),
                quote! { __builder.load_state_field_ref(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                // The `ref(T)` declaration's own `T`. `state.<ref>.<member>`
                // resolves it from this same map one level up, so leaving it
                // off here is what made the read stop lowering as soon as the
                // ref was bound to a local first.
                struct_type: Some(struct_path.clone()),
            });
        }
        if let Some(&field_index) = config.state_float_scalars.get(&member_name) {
            self.note_sub_jitcode_identity_slot(&member_name, field.member.span());
            let fi = field_index as u16;
            let reg = self.alloc_reg();
            let slot = config.float_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    self.identity_use(Register::float(slot)),
                    vec![Register::float(reg)],
                ),
                quote! { __builder.load_state_field_float(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        None
    }

    /// Recognizes a field READ through a `ref(T)` state scalar:
    /// `state.<ref_scalar>.<member>` → `getfield_gc_i` on the heap object the
    /// ref points at.  Unlike the residual-call form (an opaque CALL_I whose
    /// result the optimizer can neither re-produce in the short preamble nor
    /// invalidate on a write), a `getfield_gc` on a non-immutable field is
    /// re-readable each loop entry and is invalidated by a matching
    /// `setfield_gc` on the same `(struct_type_id, field)` — mirroring an
    /// RPython `len(obj)`/`obj.field` getfield_gc_i on a mutable field.
    ///
    /// Only int fields are lowered here (the length read); a ref field
    /// read would route through `getfield_gc_r` and is left unimplemented
    /// until a caller needs it.
    pub(super) fn lower_state_ref_field_getfield(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let Expr::Field(field) = expr else {
            return None;
        };
        // The base must be `state.<ref_scalar>` (a ref(T) state field).
        let Expr::Field(base_field) = &*field.base else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        let (_, struct_path) = config.state_ref_scalars.get(&base_name).cloned()?;
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Check if this field is declared ref-kind in `ref_fields`.
        // Resolve the access against the struct that DECLARES this field.
        // Everything below reads the one `struct_path` local -- the field key,
        // the ref/int/array lookup, the gc-kind, the type id, `size_of` and
        // `offset_of!` -- so rebinding it here moves all of them onto the
        // declaring struct together, which is what gives one physical field
        // one descriptor across every struct that embeds it.
        // `rclass.py getfield` recurses to `self.rbase` for exactly this, and
        // its `cast_pointer` costs nothing at runtime (`jtransform.py`
        // lowers it to `same_as`): the base register is unchanged, because an
        // inlined base is required to start at offset 0.
        let struct_path = config.declaring_struct(&struct_path, &member_name);
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        // The pointee's own gc-kind, so a `new_struct` of this type and an
        // access to it name one descriptor id rather than two.
        let gc_managed = config.struct_gc_kind_is_managed(&struct_path);
        let headerless = config.is_headerless_struct(&struct_path);
        let tid = struct_type_id_tokens(&struct_path, gc_managed);
        // The inlined base's own fields, so this struct's fields are numbered
        // after them rather than starting again at slot 0.
        let (prefix_fields, prefix_witness) = config.prefix_field_entries_tokens(&struct_path);
        // Lower the `state.<ref_scalar>` base to a ref binding (its
        // load_state_field_ref already declares the ref identity slot live for
        // resume), then read the field off that concrete ref.
        let base = self.lower_state_field_read(&field.base)?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let result_reg = self.alloc_reg();
        if is_varsize_length_member(config, &struct_path, &member_name) {
            let descr_tokens = gc_varsize_descr_tokens(config, &struct_path);
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    __builder.arraylen_gc(#result_reg, #base_reg, __descr_idx);
                },
            );
            return Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        if is_ref_field {
            // Ref-kind field → getfield_gc_r, result is a ref binding.
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: ref_field_entry.map(|(_, _, pointee_path)| pointee_path.clone()),
            })
        } else {
            // Int-kind field (default) → getfield_gc_i.
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            })
        }
    }

    /// The local a field read is on: `ident` or `(*ident)`, parentheses aside.
    ///
    /// One deref. `(*(*ident)).field` is a different load and stays unlowered.
    fn field_base_ident(base: &Expr) -> Option<&syn::Ident> {
        let mut expr = base;
        loop {
            expr = match expr {
                Expr::Paren(paren) => &paren.expr,
                Expr::Group(group) => &group.expr,
                Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                    let mut inner = unary.expr.as_ref();
                    loop {
                        inner = match inner {
                            Expr::Paren(paren) => &paren.expr,
                            Expr::Group(group) => &group.expr,
                            Expr::Path(path) => return path.path.get_ident(),
                            _ => return None,
                        };
                    }
                }
                Expr::Path(path) => return path.path.get_ident(),
                _ => return None,
            };
        }
    }

    /// Field read on an arbitrary local ref binding with known struct type:
    /// `<ident>.<field>` where `<ident>` was bound as `BindingKind::Ref` with
    /// a non-None `struct_type`.  Uses the same `ref_fields` config as
    /// `lower_state_ref_field_getfield` to determine whether the field is
    /// ref-kind or int-kind.
    pub(super) fn lower_ref_binding_getfield(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let Expr::Field(field) = expr else {
            return None;
        };
        // Base must be a simple ident (a local variable), not `state.x.y`.
        // `(*ident).field` is that same ident: the parentheses are part of
        // the Rust spelling, because `*ident.field` binds the other way.
        let Some(ident) = Self::field_base_ident(&field.base) else {
            return None;
        };
        let binding = self.bindings.get(&ident.to_string())?.clone();
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let struct_path = binding.struct_type.as_ref()?.clone();
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Resolve the access against the struct that DECLARES this field.
        // Everything below reads the one `struct_path` local -- the field key,
        // the ref/int/array lookup, the gc-kind, the type id, `size_of` and
        // `offset_of!` -- so rebinding it here moves all of them onto the
        // declaring struct together, which is what gives one physical field
        // one descriptor across every struct that embeds it.
        // `rclass.py getfield` recurses to `self.rbase` for exactly this, and
        // its `cast_pointer` costs nothing at runtime (`jtransform.py`
        // lowers it to `same_as`): the base register is unchanged, because an
        // inlined base is required to start at offset 0.
        let struct_path = config.declaring_struct(&struct_path, &member_name);
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        let gc_managed = config.struct_gc_kind_is_managed(&struct_path);
        let headerless = config.is_headerless_struct(&struct_path);
        let tid = struct_type_id_tokens(&struct_path, gc_managed);
        // The inlined base's own fields, so this struct's fields are numbered
        // after them rather than starting again at slot 0.
        let (prefix_fields, prefix_witness) = config.prefix_field_entries_tokens(&struct_path);
        let base_reg = binding.reg;
        let result_reg = self.alloc_reg();
        if is_ref_field {
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: ref_field_entry.map(|(_, _, pointee_path)| pointee_path.clone()),
            })
        } else if config.float_fields.contains(&ref_field_key) {
            // `jtransform.py` `rewrite_op_getfield`: `getkind` `f` →
            // `getfield_gc_f`.
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::float(result_reg)],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            true,
                        )],
                        {
                            #[allow(unused_imports)]
                            use majit_metainterp::MajitImmutableFields as _;
                            <#struct_path>::__MAJIT_IMMUTABLE_FIELDS
                        },
                    );
                    __builder.getfield_gc_f(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            })
        } else if is_varsize_length_member(config, &struct_path, &member_name) {
            // The length word of a varsize block is `arraylen_gc`
            // (`opimpl_arraylen_gc`), so a virtual `new_array` folds it.
            // A `getfield` of that word would not see `VArrayValue`.
            let descr_tokens = gc_varsize_descr_tokens(config, &struct_path);
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    __builder.arraylen_gc(#result_reg, #base_reg, __descr_idx);
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            })
        } else {
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            })
        }
    }

    /// Match `<local ref binding>.<field>` against `array_fields` WITHOUT
    /// emitting anything, so a caller can finish rejecting before it commits
    /// any op to the trace.
    ///
    /// Split from [`Self::emit_array_field_base`] because the element write
    /// has to emit its right-hand side first: Rust evaluates the assigned
    /// value before the assignee place, and a lowering that emitted the base
    /// load first would run the two in the opposite order from the
    /// interpreter.
    fn match_array_field_base(&self, field: &syn::ExprField) -> Option<ArrayFieldBase> {
        let config = self.config?;
        // Either a local ref binding (`jit_inline`'s `ref_params`) or a
        // `ref(T)` state scalar (`jit_interp`'s `state_fields`). Both name an
        // object whose field holds the buffer base; only where the object
        // pointer comes from differs.
        let (holder, struct_path) = match &*field.base {
            Expr::Path(path) => {
                let ident = path.path.get_ident()?;
                let binding = self.bindings.get(&ident.to_string())?.clone();
                if !matches!(binding.kind, BindingKind::Ref) {
                    return None;
                }
                (
                    ArrayFieldHolder::Reg(binding.reg),
                    binding.struct_type.as_ref()?.clone(),
                )
            }
            Expr::Field(base_field) if expr_matches_local_name(&base_field.base, "state") => {
                let base_name = named_member(&base_field.member)?;
                let (_, struct_path) = config.state_ref_scalars.get(&base_name).cloned()?;
                (
                    ArrayFieldHolder::StateRef((*field.base).clone()),
                    struct_path,
                )
            }
            _ => return None,
        };
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Resolve the access against the struct that DECLARES this field.
        // Everything below reads the one `struct_path` local -- the field key,
        // the ref/int/array lookup, the gc-kind, the type id, `size_of` and
        // `offset_of!` -- so rebinding it here moves all of them onto the
        // declaring struct together, which is what gives one physical field
        // one descriptor across every struct that embeds it.
        // `rclass.py getfield` recurses to `self.rbase` for exactly this, and
        // its `cast_pointer` costs nothing at runtime (`jtransform.py`
        // lowers it to `same_as`): the base register is unchanged, because an
        // inlined base is required to start at offset 0.
        let struct_path = config.declaring_struct(&struct_path, &member_name);
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let key = format!("{}::{}", struct_last, member_name);
        let element_type = config.array_fields.get(&key)?.2.clone();
        let header = config.array_headers.get(&key).cloned();
        Some(ArrayFieldBase {
            holder,
            struct_path,
            member,
            element_type,
            header,
        })
    }

    /// Emit the `getfield_gc_r` that loads the array's base pointer for a
    /// shape [`Self::match_array_field_base`] already accepted.
    ///
    /// Shared by the element read and the element write so both take the
    /// buffer pointer through the SAME interned field descr — a store to the
    /// base (a realloc on growth) then invalidates the load the heapcache
    /// served, instead of leaving a stale buffer pointer live in the trace.
    fn emit_array_field_base(&mut self, shape: &ArrayFieldBase) -> Option<u16> {
        let ArrayFieldBase {
            holder,
            struct_path,
            member,
            element_type,
            header,
        } = shape;
        // A header array's field is the block pointer, not `*mut Element`.
        // Dereferencing it is not an element. Check the pointer width and
        // that `header.items` exists; the element stride comes from the descr.
        let element_witness = if let Some(header) = header {
            quote! {
                const _: () = {
                    const fn __majit_field_width<T>(_: fn(&#struct_path) -> &T) -> usize {
                        ::core::mem::size_of::<T>()
                    }
                    assert!(
                        __majit_field_width(|__s: &#struct_path| &__s.#member)
                            == ::core::mem::size_of::<usize>()
                    );
                };
                const _: usize = ::core::mem::offset_of!(#header, items);
            }
        } else {
            quote! {
                const _: fn(&#struct_path) = |__s| {
                    let _: #element_type =
                        unsafe { ::core::mem::transmute(*__s.#member) };
                };
            }
        };
        let (struct_path, member, _element_type) =
            (struct_path.clone(), member.clone(), element_type.clone());
        // A state scalar is read here rather than in the matcher, so nothing is
        // emitted until the caller has committed to this shape.
        let base_reg = match holder {
            ArrayFieldHolder::Reg(reg) => *reg,
            ArrayFieldHolder::StateRef(base_expr) => {
                let base = self.lower_state_field_read(&base_expr.clone())?;
                if !matches!(base.kind, BindingKind::Ref) {
                    return None;
                }
                base.reg
            }
        };
        // No config means nothing was declared gc-managed or headerless,
        // which is the same answer the raw default gave before.
        let gc_managed = self
            .config
            .is_some_and(|config| config.struct_gc_kind_is_managed(&struct_path));
        let headerless = self
            .config
            .is_some_and(|config| config.is_headerless_struct(&struct_path));
        let tid = struct_type_id_tokens(&struct_path, gc_managed);
        // The inlined base's own fields, so this struct's fields are numbered
        // after them rather than starting again at slot 0.
        let (prefix_fields, prefix_witness) = self
            .config
            .map(|config| config.prefix_field_entries_tokens(&struct_path))
            .unwrap_or_default();
        let buffer_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(base_reg)],
                vec![Register::ref_(buffer_reg)],
            ),
            quote! {
                // The `int_fields` twin of this check lives in
                // `field_scalar_tokens`.  Nothing else ties the declared
                // element type to the field: the concrete rewrite indexes the
                // real pointer while the emitted descr takes its stride and
                // signedness from the declaration, so a declaration that
                // drifted from the struct (`Stack::data => u16` over a
                // `*mut u8`) would compile and read past the buffer.
                //
                // The claim is the element's WIDTH, not its name.  A
                // `#[repr(transparent)]` wrapper over the declared type has
                // the declared type's size and, by the definition of
                // `repr(transparent)`, its representation — a legitimate
                // element that nominal equality rejects.  `transmute` states
                // the width, so the drifted `u16`-over-`*mut u8` declaration
                // above still fails to compile.
                //
                // What this no longer catches: a declaration that keeps the
                // width and flips the sign (`=> i8` over a `*mut u8`).  The
                // stride is what bounds the access, so that mistake stays
                // in-buffer and misreads a value rather than the heap, but it
                // IS unchecked here — the sign now comes from the declaration
                // alone.
                //
                // Reading through the field rather than naming a pointer type
                // accepts `*const T` as well as `*mut T`; the closure body is
                // type-checked but never evaluated. A header array checks the
                // block pointer's width instead, because the field is not
                // itself `*mut Element`.
                #element_witness
                #prefix_witness
                __builder.register_struct_layout(
                    ::core::mem::size_of::<#struct_path>(),
                    #tid,
                    #gc_managed,
                    #headerless,
                    &[#(#prefix_fields,)* (
                        ::core::mem::offset_of!(#struct_path, #member),
                        true,
                        stringify!(#member),
                        ::core::mem::size_of::<usize>(),
                        false,
                        false,
                    )],
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
                __builder.getfield_gc_r(
                    #buffer_reg,
                    #base_reg,
                    ::core::mem::offset_of!(#struct_path, #member),
                    #tid,
                    stringify!(#member),
                );
            },
        );
        Some(buffer_reg)
    }

    /// Recognizes an array element READ on a local ref binding:
    /// `<binding>.<array_field>[<int expr>]` → `getfield_gc_r` for the buffer
    /// base followed by `getarrayitem_gc_i`.
    ///
    /// This is the non-virtualizable counterpart of the `[..; virt]` element
    /// read: the loaded value is an ordinary trace value, so it does NOT ride
    /// the guard's `vable_array` resume section and the snapshot stays O(1) in
    /// the array's length.
    ///
    /// No `arraylen_gc` is emitted anywhere on this path — the buffer has no
    /// object header to read a length from. A bound belongs on a sibling
    /// `int_fields` length field.
    pub(super) fn lower_ref_binding_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let Expr::Index(index_expr) = expr else {
            return None;
        };
        let Expr::Field(field) = &*index_expr.expr else {
            return None;
        };
        // Match before emitting anything, so a shape this does not handle
        // leaves the trace untouched.  A place expression evaluates its base
        // before its index, which is the order emitted here.
        let shape = self.match_array_field_base(field)?;
        let element_type = shape.element_type.clone();
        let header = shape.header.clone();
        let element_is_float = path_is_f64(&element_type);
        let buffer_reg = self.emit_array_field_base(&shape)?;
        let index = self.lower_value_expr(&index_expr.index)?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let result_reg = self.alloc_reg();
        if element_is_float {
            // `jtransform.py` `rewrite_op_getarrayitem`: `getkind(ARRAY.OF)`
            // selects `getarrayitem_gc_f` for `lltype.Float`, including an
            // element declared `Elem in Header`. `pure = '_pure'` when
            // `ARRAY._immutable_field(None)`.
            let item_pure = immutable_array_item_pure(&shape.struct_path, &shape.member);
            let descr_tokens = float_array_descr_tokens(self.config, &header);
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(buffer_reg), Register::int(index_reg)],
                    vec![Register::float(result_reg)],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    if #item_pure {
                        __builder.getarrayitem_gc_f_pure(
                            #result_reg as u16,
                            #buffer_reg as u16,
                            #index_reg as u16,
                            __descr_idx,
                        );
                    } else {
                        __builder.getarrayitem_gc_f(
                            #result_reg as u16,
                            #buffer_reg as u16,
                            #index_reg as u16,
                            __descr_idx,
                        );
                    }
                },
            );
            return Some(Binding {
                reg: result_reg,
                kind: BindingKind::Float,
                depends_on_stack: index.depends_on_stack,
                struct_type: None,
            });
        }
        if let Some(header) = header.clone().filter(|_| path_is_int_elem(&element_type)) {
            let descr_tokens = self
                .config
                .map(|config| gc_varsize_descr_tokens(config, &header))
                .unwrap_or_else(|| {
                    quote! {
                        __builder.add_raw_int_array_descr_signed(
                            ::core::mem::size_of::<#element_type>(),
                            (<#element_type>::MIN as i128) < 0,
                        )
                    }
                });
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(buffer_reg), Register::int(index_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    __builder.getarrayitem_gc_i(
                        #result_reg as u16,
                        #buffer_reg as u16,
                        #index_reg as u16,
                        __descr_idx,
                    );
                },
            );
            return Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: index.depends_on_stack,
                struct_type: None,
            });
        }
        if let Some(header) = header {
            // Pointer elements living after `header.items`. The base
            // register is the header, so the descr's base_size is that
            // field's offset rather than zero. The length offset is the
            // header's lendescr (`get_array_descr`).
            let descr_tokens = self
                .config
                .map(|config| gc_varsize_descr_tokens(config, &header))
                .unwrap_or_else(|| {
                    quote! {
                        __builder.add_ptr_array_descr(
                            ::core::mem::offset_of!(#header, items),
                            ::core::option::Option::None,
                        )
                    }
                });
            let item_pure = immutable_array_item_pure(&shape.struct_path, &shape.member);
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(buffer_reg), Register::int(index_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    // `rewrite_op_getarrayitem`: `pure = '_pure'` when
                    // `ARRAY._immutable_field(None)` (`_immutable_fields_ = ['x[*]']`).
                    if #item_pure {
                        __builder.getarrayitem_gc_r_pure(
                            #result_reg as u16,
                            #buffer_reg as u16,
                            #index_reg as u16,
                            __descr_idx,
                        );
                    } else {
                        __builder.getarrayitem_gc_r(
                            #result_reg as u16,
                            #buffer_reg as u16,
                            #index_reg as u16,
                            __descr_idx,
                        );
                    }
                },
            );
            return Some(Binding {
                reg: result_reg,
                kind: BindingKind::Ref,
                depends_on_stack: index.depends_on_stack,
                struct_type: None,
            });
        }
        let item_pure = immutable_array_item_pure(&shape.struct_path, &shape.member);
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(buffer_reg), Register::int(index_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! {
                let __descr_idx = __builder.add_raw_int_array_descr_signed(
                    ::core::mem::size_of::<#element_type>(),
                    // descr.py get_type_flag reads signedness off the
                    // declared element type.  Hard-coding signed would
                    // sign-extend a `u8` element of 0x80 to -128 where the
                    // concrete Rust read yields 128.  `MIN` resolves through a
                    // type alias, and a non-integer element type fails to
                    // compile here rather than loading as a signed word.
                    (<#element_type>::MIN as i128) < 0,
                );
                // `rewrite_op_getarrayitem`: `pure = '_pure'` when
                // `ARRAY._immutable_field(None)` (`_immutable_fields_ = ['x[*]']`).
                if #item_pure {
                    __builder.getarrayitem_gc_i_pure(
                        #result_reg as u16,
                        #buffer_reg as u16,
                        #index_reg as u16,
                        __descr_idx,
                    );
                } else {
                    __builder.getarrayitem_gc_i(
                        #result_reg as u16,
                        #buffer_reg as u16,
                        #index_reg as u16,
                        __descr_idx,
                    );
                }
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: index.depends_on_stack,
            struct_type: None,
        })
    }

    /// Recognizes an array element WRITE on a local ref binding:
    /// `<binding>.<array_field>[<int expr>] = <int expr>` → `getfield_gc_r`
    /// for the buffer base followed by `setarrayitem_gc_i`.
    ///
    /// Store counterpart of [`Self::lower_ref_binding_array_read`]; both take
    /// the element descr from `add_raw_int_array_descr_signed`, so the store
    /// invalidates the heapcache entry the matching load populated.
    pub(super) fn lower_ref_binding_array_write(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Index(index_expr) = &*assign.left else {
            return None;
        };
        let Expr::Field(field) = &*index_expr.expr else {
            return None;
        };
        // Match before emitting, then emit in Rust's evaluation order for an
        // assignment: the assigned value first, then the assignee place (base,
        // then index).  Emitting the base load first would run the array's
        // side effects ahead of the right-hand side's.
        let shape = self.match_array_field_base(field)?;
        let element_type = shape.element_type.clone();
        let header = shape.header.clone();
        let element_is_float = path_is_f64(&element_type);
        let element_is_int = path_is_int_elem(&element_type);
        let value = self.lower_value_expr(&assign.right)?;
        let value_is_ref = header.is_some() && !element_is_int;
        if element_is_float {
            if !matches!(value.kind, BindingKind::Float) {
                return None;
            }
        } else if value_is_ref {
            if !matches!(value.kind, BindingKind::Ref) {
                return None;
            }
        } else if !matches!(value.kind, BindingKind::Int) {
            return None;
        }
        let buffer_reg = self.emit_array_field_base(&shape)?;
        let index = self.lower_value_expr(&index_expr.index)?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let value_reg = value.reg;
        if element_is_float {
            let descr_tokens = float_array_descr_tokens(self.config, &header);
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(buffer_reg),
                        Register::int(index_reg),
                        Register::float(value_reg),
                    ],
                    vec![],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    __builder.setarrayitem_gc_f(
                        #buffer_reg as u16,
                        #index_reg as u16,
                        #value_reg as u16,
                        __descr_idx,
                    );
                },
            );
            return Some(());
        }
        if let Some(header) = header {
            let descr_tokens = self
                .config
                .map(|config| gc_varsize_descr_tokens(config, &header))
                .unwrap_or_else(|| {
                    quote! {
                        __builder.add_ptr_array_descr(
                            ::core::mem::offset_of!(#header, items),
                            ::core::option::Option::None,
                        )
                    }
                });
            if element_is_int {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![
                            Register::ref_(buffer_reg),
                            Register::int(index_reg),
                            Register::int(value_reg),
                        ],
                        vec![],
                    ),
                    quote! {
                        let __descr_idx = #descr_tokens;
                        __builder.setarrayitem_gc_i(
                            #buffer_reg as u16,
                            #index_reg as u16,
                            #value_reg as u16,
                            __descr_idx,
                        );
                    },
                );
                return Some(());
            }
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(buffer_reg),
                        Register::int(index_reg),
                        Register::ref_(value_reg),
                    ],
                    vec![],
                ),
                quote! {
                    let __descr_idx = #descr_tokens;
                    __builder.setarrayitem_gc_r(
                        #buffer_reg as u16,
                        #index_reg as u16,
                        #value_reg as u16,
                        __descr_idx,
                    );
                },
            );
            return Some(());
        }
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![
                    Register::ref_(buffer_reg),
                    Register::int(index_reg),
                    Register::int(value_reg),
                ],
                vec![],
            ),
            quote! {
                let __descr_idx = __builder.add_raw_int_array_descr_signed(
                    ::core::mem::size_of::<#element_type>(),
                    // descr.py get_type_flag reads signedness off the
                    // declared element type.  Hard-coding signed would
                    // sign-extend a `u8` element of 0x80 to -128 where the
                    // concrete Rust read yields 128.  `MIN` resolves through a
                    // type alias, and a non-integer element type fails to
                    // compile here rather than loading as a signed word.
                    (<#element_type>::MIN as i128) < 0,
                );
                __builder.setarrayitem_gc_i(
                    #buffer_reg as u16,
                    #index_reg as u16,
                    #value_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(())
    }

    /// Recognizes a pool-array element read through the registered getter call
    /// `<getter>(state.<pool_base_ref>, <int index>)` → `getarrayitem_gc_r` on
    /// the raw-pointer array (`[*mut U; N]` at the declared `items` offset) the
    /// ref-scalar points at — the `pools[selected]` read.  Unlike the
    /// residual-call form (an
    /// opaque CALL_R the optimizer can neither re-produce in the short preamble
    /// nor invalidate), the getarrayitem on the immutable `pools` array
    /// re-derives the element each loop entry from the consistent `selected`
    /// index, so the loaded ref can no longer be carried as an independent
    /// loop-red that diverges from the promoted index.
    ///
    /// Selection is keyed on OPERATION IDENTITY: the call's function path must
    /// match the `getter` registered for this `base` in `pool_arrays`, and arg0
    /// must be `state.<base>`.  An unrelated helper that happens to share the
    /// `(state.<base>, int)` arg shape does NOT match, so it is not miscompiled
    /// into a pool read — it falls through to its own residual body (which is
    /// also the getter's concrete fallback when no `pool_arrays` is configured).
    /// Elements are pointer-width; where they start, and whether a length word
    /// precedes them, come from the declaration (`pool_array_layout_tokens`).
    pub(super) fn lower_pool_array_get_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        let config = self.config?;
        if call.args.len() != 2 {
            return None;
        }
        // arg0 must be `state.<base>` where <base> is a declared pool-array base.
        let Expr::Field(base_field) = &call.args[0] else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        // Operation identity: the call's function must be the registered getter
        // for this base, not merely any call sharing the `(state.<base>, int)`
        // arg shape.  A function mismatch falls through to the residual CALL_R
        // fallback (the marker function's own body) rather than miscompiling an
        // unrelated helper into a `getarrayitem_gc_r`.
        let func_segments = canonical_expr_segments(&call.func)?;
        let entry = config
            .pool_arrays
            .iter()
            .find(|entry| entry.base == base_name && entry.getter == func_segments)?;
        let element_type = entry.element_type.clone();
        let (base_size, len_offset, layout_witness) = pool_array_layout_tokens(entry);
        // Lower the `state.<base>` ref-scalar (declares its ref identity slot
        // live for resume) and the index, then read the pointer element.
        let base = self.lower_state_field_read(&call.args[0])?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let index = self.lower_value_expr(&call.args[1])?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(base_reg), Register::int(index_reg)],
                vec![Register::ref_(result_reg)],
            ),
            quote! {
                #layout_witness
                let __descr_idx = __builder.add_ptr_array_descr(#base_size, #len_offset);
                __builder.getarrayitem_gc_r(
                    #result_reg as u16,
                    #base_reg as u16,
                    #index_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Ref,
            depends_on_stack: base.depends_on_stack || index.depends_on_stack,
            struct_type: element_type,
        })
    }

    /// Recognizes a field WRITE through a `ref(T)` state scalar:
    /// `state.<ref_scalar>.<member> = <int expr>` → `setfield_gc_i`.  The
    /// store shares the same `(struct_type_id, field)` interned `Field` descr
    /// as [`Self::lower_state_ref_field_getfield`], so the heapcache
    /// invalidates the cached getfield on every write — the in-trace
    /// counterpart of an RPython inlined `self.field = ...` store that keeps a
    /// length getfield from freezing to a loop-invariant constant.
    pub(super) fn lower_state_ref_field_setfield(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Field(field) = &*assign.left else {
            return None;
        };
        let Expr::Field(base_field) = &*field.base else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        let (_, struct_path) = config.state_ref_scalars.get(&base_name).cloned()?;
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Check if this field is declared ref-kind in `ref_fields`.
        // Resolve the access against the struct that DECLARES this field.
        // Everything below reads the one `struct_path` local -- the field key,
        // the ref/int/array lookup, the gc-kind, the type id, `size_of` and
        // `offset_of!` -- so rebinding it here moves all of them onto the
        // declaring struct together, which is what gives one physical field
        // one descriptor across every struct that embeds it.
        // `rclass.py getfield` recurses to `self.rbase` for exactly this, and
        // its `cast_pointer` costs nothing at runtime (`jtransform.py`
        // lowers it to `same_as`): the base register is unchanged, because an
        // inlined base is required to start at offset 0.
        let struct_path = config.declaring_struct(&struct_path, &member_name);
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        // The same id the matching getfield uses, so this setfield
        // invalidates it — see `struct_gc_kind_is_managed`.
        let gc_managed = config.struct_gc_kind_is_managed(&struct_path);
        let headerless = config.is_headerless_struct(&struct_path);
        let tid = struct_type_id_tokens(&struct_path, gc_managed);
        // The inlined base's own fields, so this struct's fields are numbered
        // after them rather than starting again at slot 0.
        let (prefix_fields, prefix_witness) = config.prefix_field_entries_tokens(&struct_path);
        let base = self.lower_state_field_read(&field.base)?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let rhs = self.lower_value_expr(&assign.right)?;
        if is_ref_field {
            // Ref-kind field → setfield_gc_r.
            if !matches!(rhs.kind, BindingKind::Ref) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::ref_(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        } else {
            // Int-kind field (default) → setfield_gc_i.
            if !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::int(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        }
        Some(())
    }

    /// Field WRITE on an arbitrary local ref binding with known struct type:
    /// `<ident>.<field> = <expr>` where `<ident>` was bound as
    /// `BindingKind::Ref` with a non-None `struct_type`.
    pub(super) fn lower_ref_binding_setfield(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Field(field) = &*assign.left else {
            return None;
        };
        // Base must be a simple ident (a local variable), not `state.x.y`.
        let Expr::Path(path) = &*field.base else {
            return None;
        };
        let ident = path.path.get_ident()?;
        let binding = self.bindings.get(&ident.to_string())?.clone();
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let struct_path = binding.struct_type.as_ref()?.clone();
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Resolve the access against the struct that DECLARES this field.
        // Everything below reads the one `struct_path` local -- the field key,
        // the ref/int/array lookup, the gc-kind, the type id, `size_of` and
        // `offset_of!` -- so rebinding it here moves all of them onto the
        // declaring struct together, which is what gives one physical field
        // one descriptor across every struct that embeds it.
        // `rclass.py getfield` recurses to `self.rbase` for exactly this, and
        // its `cast_pointer` costs nothing at runtime (`jtransform.py`
        // lowers it to `same_as`): the base register is unchanged, because an
        // inlined base is required to start at offset 0.
        let struct_path = config.declaring_struct(&struct_path, &member_name);
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        let is_float_field = config.float_fields.contains(&ref_field_key);
        let gc_managed = config.struct_gc_kind_is_managed(&struct_path);
        let headerless = config.is_headerless_struct(&struct_path);
        let tid = struct_type_id_tokens(&struct_path, gc_managed);
        // The inlined base's own fields, so this struct's fields are numbered
        // after them rather than starting again at slot 0.
        let (prefix_fields, prefix_witness) = config.prefix_field_entries_tokens(&struct_path);
        let base_reg = binding.reg;
        let rhs = self.lower_value_expr(&assign.right)?;
        if is_float_field {
            if !matches!(rhs.kind, BindingKind::Float) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::float(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            true,
                        )],
                        {
                            #[allow(unused_imports)]
                            use majit_metainterp::MajitImmutableFields as _;
                            <#struct_path>::__MAJIT_IMMUTABLE_FIELDS
                        },
                    );
                    __builder.setfield_gc_f(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        } else if is_ref_field {
            if !matches!(rhs.kind, BindingKind::Ref) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::ref_(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        } else {
            if !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::int(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    #prefix_witness
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        #gc_managed,
                        #headerless,
                        &[#(#prefix_fields,)* (
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                            false,
                        )],
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
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        }
        Some(())
    }

    /// Recognizes `state.array[index]` for array state fields.
    /// Virtualizable arrays bail to the standard vable path. A plain
    /// `[int]`/`[float]` is ptr then len. `rlist.py` `rtype_getitem` uses
    /// `dum_checkidx` (`raise IndexError`) only for `getitem_idx`, i.e. when
    /// the source has an `except IndexError` handler around the read
    /// (`rtype_getitem_idx`). A plain `l[i]` uses `dum_nocheck`. In that
    /// case `ll_getitem_nonneg`'s only check is `ll_assert`, which
    /// translation removes, and the JIT sees the bare `basegetitem` item
    /// read. A Rust slice index has no handler: its panic is a debug
    /// assertion in that sense. majit-translate already strips it the same
    /// way (`TermKind::Assert`, citing `backendopt/removeassert.py`).
    /// `jtransform.py` `rewrite_op_getarrayitem` emits
    /// `getarrayitem_raw_<kind>` from `getkind(ARRAY.OF)`.
    pub(super) fn lower_state_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        // `[int; virt]` arrays are NOT handled here — they lower through the
        // standard virtualizable path (`lower_vable_array_read`). Bail before
        // lowering the index expression so it is not emitted twice when the
        // dispatch falls through to the vable read.
        if config.state_virt_arrays.contains_key(&member_name) {
            return None;
        }
        let &(_, item_kind) = config.state_arrays.get(&member_name)?;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let reg = self.alloc_reg();
        // The `{ptr, len}` words are declared int arguments of this
        // JitCode (`MIFrame.setup_call` / `bhimpl_inline_call_ir_v`); this
        // op is the `dum_nocheck` item read off the pointer.
        let base_reg = self.plain_array_ptr_reg(&member_name)?;

        let (dst, descr, get_raw, binding_kind) = match item_kind {
            ValueKind::Float => (
                Register::float(reg),
                quote! { __builder.add_raw_float_array_descr() },
                format_ident!("getarrayitem_raw_f"),
                BindingKind::Float,
            ),
            ValueKind::Int => (
                Register::int(reg),
                quote! {
                    __builder.add_raw_int_array_descr_signed(
                        ::core::mem::size_of::<i64>(),
                        true,
                    )
                },
                format_ident!("getarrayitem_raw_i"),
                BindingKind::Int,
            ),
            ValueKind::Ref => return None,
        };
        self.emit_op(
            OpMeta::linear(
                OpKind::StateField,
                vec![Register::int(base_reg), Register::int(idx_reg)],
                vec![dst],
            ),
            quote! {
                let __descr_idx = #descr;
                __builder.#get_raw(#reg, #base_reg, #idx_reg, __descr_idx);
            },
        );
        Some(Binding {
            reg,
            kind: binding_kind,
            depends_on_stack: false,
            struct_type: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_fields_map(
        entries: &[(&str, &str, &str, &str)],
    ) -> HashMap<String, (syn::Path, Ident, syn::Path)> {
        entries
            .iter()
            .map(|(key, struct_path, field, pointee)| {
                (
                    (*key).to_string(),
                    (
                        syn::parse_str::<syn::Path>(struct_path).expect("struct path"),
                        syn::parse_str::<Ident>(field).expect("field ident"),
                        syn::parse_str::<syn::Path>(pointee).expect("pointee path"),
                    ),
                )
            })
            .collect()
    }

    fn witness_for(key: &str) -> String {
        let map = ref_fields_map(&[("Stack::head", "Stack", "head", "Node")]);
        let struct_path: syn::Path = syn::parse_str("Stack").expect("struct path");
        let member: syn::Member = syn::parse_str("head").expect("member");
        ref_field_witness_tokens(&map, key, &struct_path, &member).to_string()
    }

    #[test]
    fn a_void_dispatch_callee_has_an_explicit_return() {
        let body: Expr = syn::parse_quote!({
            let value = 1;
        });
        let (generated, _) =
            try_generate_jitcode_body_parts_with_caller_bindings(&body, None, &[]).unwrap();
        assert!(
            generated
                .body
                .to_string()
                .ends_with("__builder . void_return () ;")
        );
    }

    #[test]
    fn state_scalar_liveness_uses_the_callee_frame() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(0);
        config
            .state_ref_scalars
            .insert("selected_ref".into(), (0, syn::parse_quote!(ListBase)));
        config
            .vable_fields
            .insert("selected_ref".into(), (0, ValueKind::Ref));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let expr: Expr = syn::parse_quote!(state.selected_ref);
        let binding = lowerer.lower_state_field_read(&expr).unwrap();
        assert_eq!(binding.kind, BindingKind::Ref);
        assert_eq!(binding.struct_type.unwrap(), syn::parse_quote!(ListBase));
        let read = lowerer.op_metadata.last().unwrap();
        assert_eq!(read.reads, vec![Register::ref_(0)]);
        let live = compute_per_marker_liveness(&lowerer.op_metadata);
        assert_eq!(live[0], BTreeSet::from([Register::ref_(0)]));
        assert!(
            !lowerer
                .statements
                .iter()
                .any(|op| op.to_string().contains("load_state_field_ref"))
        );
    }

    #[test]
    fn a_borrow_of_the_virtualizable_preserves_its_ref_binding() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(4);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let expr: Expr = syn::parse_str("&mut state").expect("virtualizable borrow");

        let binding = lowerer
            .lower_vable_reference(&expr)
            .expect("virtualizable borrow lowers");

        assert_eq!(binding.reg, 4);
        assert_eq!(binding.kind, BindingKind::Ref);
        assert!(lowerer.op_metadata.is_empty());
    }

    /// A declared ref field admits the three spellings the lowering can
    /// actually receive, and no others. The pointee is named once per pointer
    /// spelling, which is what turns a drifted declaration into a type error.
    /// The control: `hint_access_directly` is still answered as the identity
    /// it is.  Upstream drops it at the same site, and the flag it records is
    /// read back for `fresh_virtualizable` alone, so nothing is lost.
    #[test]
    fn an_access_directly_hint_still_lowers_to_its_argument() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(4);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let expr: Expr = syn::parse_quote! { hint_access_directly(state) };

        let binding = lowerer
            .lower_value_expr(&expr)
            .expect("the access_directly hint lowers to its argument");

        assert_eq!(binding.reg, 4);
        assert_eq!(binding.kind, BindingKind::Ref);
    }

    /// `hint_fresh_virtualizable` asks the following field accesses to stay
    /// off the vable path, and this lowering holds no flag that could carry
    /// that.  Answering it as an identity — which is what the suppression did
    /// — leaves every access on the vable path with nothing emitted to say
    /// so, so the statement is refused instead.
    ///
    /// The statement is built as `Stmt::Expr(Expr::Macro, semi)` rather than
    /// parsed from `hint_fresh_virtualizable!(state);`, because syn parses
    /// that spelling as `Stmt::Macro`, whose arm in `lower_stmt_dispatch`
    /// recognises `can_enter_jit` / `jit_loop_header` and answers `None` for
    /// everything else — so it never reaches `lower_vable_hint_suppress` and
    /// the test would pass without exercising anything.  Expression position
    /// is the only spelling that does reach it.
    #[test]
    fn a_fresh_virtualizable_hint_refuses_in_expression_position() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(4);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let mac: syn::ExprMacro = syn::parse_quote! { hint_fresh_virtualizable!(state) };
        let stmt = Stmt::Expr(Expr::Macro(mac), Some(Default::default()));

        assert!(
            lowerer.lower_stmt(&stmt).is_none(),
            "the fresh_virtualizable hint must refuse, not be suppressed"
        );
        assert!(lowerer.statements.is_empty());
    }

    /// The control for the test above: the same expression-position spelling
    /// with `hint_access_directly` is still suppressed, which is what proves
    /// the refusal is about the hint and not about the statement shape.
    #[test]
    fn an_access_directly_hint_is_still_suppressed_in_expression_position() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(4);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let mac: syn::ExprMacro = syn::parse_quote! { hint_access_directly!(state) };
        let stmt = Stmt::Expr(Expr::Macro(mac), Some(Default::default()));

        assert!(
            lowerer.lower_stmt(&stmt).is_some(),
            "the access_directly hint is dropped, as upstream drops it"
        );
        assert!(lowerer.statements.is_empty());
    }

    /// The same hint in value position.  The refusal has to reach the `let`,
    /// because that is the statement the purity test would otherwise score
    /// inert and drop.
    #[test]
    fn a_fresh_virtualizable_hint_in_a_let_refuses() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.vable_var = Some("state".to_string());
        config.vable_input_ref_reg = Some(4);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.install_vable_input_binding();
        let stmt: Stmt = syn::parse_quote! { let fresh = hint_fresh_virtualizable(state); };

        assert!(
            lowerer.lower_stmt(&stmt).is_none(),
            "the hint in value position must refuse the enclosing `let`"
        );
        assert!(!lowerer.bindings.contains_key("fresh"));
    }

    #[test]
    fn a_declared_ref_field_witnesses_its_pointee() {
        let tokens = witness_for("Stack::head");
        for expected in [
            "impl __MajitRefField for * mut Node",
            "impl __MajitRefField for * const Node",
            "impl __MajitRefField for usize",
        ] {
            assert!(
                tokens.contains(expected),
                "the witness must admit `{expected}`; tokens={tokens}"
            );
        }
        assert!(
            tokens.contains("__accept (__s . head)"),
            "the witness must touch the field it names, or it witnesses \
             nothing about the struct; tokens={tokens}"
        );
    }

    /// The other half of the same predicate, stated so the gap is a decision
    /// rather than an oversight: a field named in no map reads into the Int
    /// bank, and stable Rust cannot assert that a type is *not* a pointer, so
    /// no witness is emitted for it.
    #[test]
    fn an_undeclared_field_gets_no_ref_witness() {
        assert_eq!(
            witness_for("Stack::size"),
            "",
            "a key `ref_fields` does not declare must emit nothing; emitting a \
             witness there would reject every legitimate integer field"
        );
    }

    /// `int_fields` `=> f64` is `getfield_gc_f`; an `f64` array field is
    /// `getarrayitem_gc_f` / `setarrayitem_gc_f` (`rewrite_op_getfield` /
    /// `rewrite_op_setarrayitem` pick the kind from the field type).
    #[test]
    fn float_field_and_float_array_lower_to_gc_f() {
        let floatval = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Num),
            field: syn::parse_quote!(floatval),
            int_type: syn::parse_quote!(f64),
        };
        let data = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Col),
            field: syn::parse_quote!(data),
            element_type: syn::parse_quote!(f64),
            header: None,
        };
        let config =
            LowererConfig::inline_helper(&[], &[data], &[floatval], &[], &[], &[], &[], &[]);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "num".to_string(),
            Binding {
                reg: 1,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: Some(syn::parse_quote!(Num)),
            },
        );
        lowerer.bindings.insert(
            "col".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: Some(syn::parse_quote!(Col)),
            },
        );
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let read: syn::Expr = syn::parse_quote!(num.floatval);
        let got = lowerer.lower_value_expr(&read).expect("float field read");
        assert!(matches!(got.kind, BindingKind::Float));
        let store: syn::Stmt = syn::parse_quote!(col.data[i] = num.floatval;);
        assert!(lowerer.lower_stmt(&store).is_some());
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getfield_gc_f"), "{emitted}");
        assert!(emitted.contains("setarrayitem_gc_f"), "{emitted}");
        assert!(emitted.contains("add_raw_float_array_descr"), "{emitted}");
    }

    /// `f64` declared `Elem in Header` is `getarrayitem_gc_f`
    /// (`rewrite_op_getarrayitem`, `getkind(ARRAY.OF)`).
    #[test]
    fn f64_in_header_lowers_to_getarrayitem_gc_f() {
        let data = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Col),
            field: syn::parse_quote!(data),
            element_type: syn::parse_quote!(f64),
            header: Some(syn::parse_quote!(Words)),
        };
        let cap = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Words),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(usize),
        };
        let config = LowererConfig::inline_helper(&[], &[data], &[cap], &[], &[], &[], &[], &[]);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "col".to_string(),
            Binding {
                reg: 1,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: Some(syn::parse_quote!(Col)),
            },
        );
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let read: syn::Expr = syn::parse_quote!(col.data[i]);
        let got = lowerer.lower_value_expr(&read).expect("header float read");
        assert!(matches!(got.kind, BindingKind::Float));
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_gc_f"), "{emitted}");
        assert!(emitted.contains("getarrayitem_gc_f_pure"), "{emitted}");
        assert!(!emitted.contains("getarrayitem_gc_r"), "{emitted}");
        assert!(
            emitted.contains("add_gc_varsize_float_array_descr"),
            "{emitted}"
        );
    }

    /// Headerless immutable float item consults `immutable_array_item_pure`
    /// and can emit `getarrayitem_gc_f_pure`.
    #[test]
    fn headerless_immutable_float_lowers_to_getarrayitem_gc_f_pure() {
        let data = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Col),
            field: syn::parse_quote!(data),
            element_type: syn::parse_quote!(f64),
            header: None,
        };
        let config = LowererConfig::inline_helper(&[], &[data], &[], &[], &[], &[], &[], &[]);
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "col".to_string(),
            Binding {
                reg: 1,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: Some(syn::parse_quote!(Col)),
            },
        );
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let read: syn::Expr = syn::parse_quote!(col.data[i]);
        assert!(lowerer.lower_value_expr(&read).is_some());
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_gc_f_pure"), "{emitted}");
        assert!(emitted.contains("__MAJIT_IMMUTABLE_FIELDS"), "{emitted}");
        assert!(emitted.contains("add_raw_float_array_descr"), "{emitted}");
    }

    /// `symbolic.py` `get_array_token`: a non-pointer header array's
    /// `itemsize` is `sizeof(SUBARRAY.OF)`, and signedness follows the element.
    #[test]
    fn header_itemsize_follows_the_element() {
        let data = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Col),
            field: syn::parse_quote!(data),
            element_type: syn::parse_quote!(u8),
            header: Some(syn::parse_quote!(Bytes)),
        };
        let cap = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Bytes),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(usize),
        };
        let config = LowererConfig::inline_helper(&[], &[data], &[cap], &[], &[], &[], &[], &[]);
        let header: syn::Path = syn::parse_quote!(Bytes);
        let tokens = gc_varsize_descr_tokens(&config, &header).to_string();
        assert!(
            tokens.contains("size_of") && tokens.contains("u8"),
            "{tokens}"
        );
        assert!(
            !tokens.contains("i64"),
            "itemsize must not be a hardcoded i64: {tokens}"
        );
        assert!(tokens.contains("MIN"), "{tokens}");
    }

    /// The length word is only the name `varsize_length_field_name` resolved.
    /// A header that also has `len` keeps that field an ordinary read.
    #[test]
    fn length_member_is_only_the_resolved_field() {
        let items = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Words),
            field: syn::parse_quote!(items),
            element_type: syn::parse_quote!(i64),
            header: Some(syn::parse_quote!(Words)),
        };
        let capacity = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Words),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(usize),
        };
        let len = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Words),
            field: syn::parse_quote!(len),
            int_type: syn::parse_quote!(usize),
        };
        let config =
            LowererConfig::inline_helper(&[], &[items], &[capacity, len], &[], &[], &[], &[], &[]);
        let header: syn::Path = syn::parse_quote!(Words);
        let resolved = varsize_length_field_name(&config, &header);
        assert_eq!(resolved.as_deref(), Some("capacity"));
        assert!(is_varsize_length_member(&config, &header, "capacity"));
        assert!(!is_varsize_length_member(&config, &header, "len"));
    }

    /// `bool` is an int item of size 1, unsigned. Pointer-vs-int uses one
    /// classifier, so the header and the element path agree.
    #[test]
    fn bool_header_element_is_an_unsigned_int_item() {
        let items = crate::jit_interp::ArrayFieldEntry {
            struct_type: syn::parse_quote!(Flags),
            field: syn::parse_quote!(items),
            element_type: syn::parse_quote!(bool),
            header: Some(syn::parse_quote!(Flags)),
        };
        let capacity = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Flags),
            field: syn::parse_quote!(capacity),
            int_type: syn::parse_quote!(usize),
        };
        let config =
            LowererConfig::inline_helper(&[], &[items], &[capacity], &[], &[], &[], &[], &[]);
        let header: syn::Path = syn::parse_quote!(Flags);
        let elem = header_element_path(&config, &header).expect("element path");
        assert!(path_is_int_elem(elem));
        assert!(!header_items_are_pointers(&config, &header));
        let tokens = gc_varsize_descr_tokens(&config, &header).to_string();
        assert!(tokens.contains("bool"), "{tokens}");
        assert!(tokens.contains("size_of"), "{tokens}");
        assert!(
            tokens.contains("MIN") && tokens.contains("< 0"),
            "signedness is the element MIN, which is false for bool: {tokens}"
        );
    }

    /// A plain `[int]` read is the bare `getarrayitem_raw_i` item read.
    /// `rlist.py` `rtype_getitem` uses `dum_nocheck`; `ll_getitem_nonneg`'s
    /// only check is `ll_assert`, which translation removes.
    #[test]
    fn plain_int_array_read_is_bare_getarrayitem_raw_i() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Int));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i]);
        let binding = lowerer
            .lower_state_array_read(&expr)
            .expect("plain [int] read lowers");
        assert_eq!(binding.kind, BindingKind::Int);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_raw_i"), "{emitted}");
        assert!(
            !emitted.contains("note_identity_slot_read"),
            "ptr/len are declared inline-call arguments (`MIFrame.setup_call`), \
             not an implicit identity-slot latch: {emitted}"
        );
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [int] read must not emit a compare or guard: {emitted}"
        );
        assert!(
            !lowerer.op_metadata.iter().any(|m| {
                matches!(m.kind, OpKind::BinopI | OpKind::GotoIfNot)
                    || matches!(m.control, ControlFlowClass::ConditionalGuard)
            }),
            "plain [int] read must not emit a compare or guard: {:?}",
            lowerer
                .op_metadata
                .iter()
                .map(|m| format!("{:?}", m.kind))
                .collect::<Vec<_>>()
        );
    }

    /// A plain `[int]` store is the bare `setarrayitem_raw_i` item store.
    /// `rlist.py` `rtype_setitem` takes `dum_nocheck`; `ll_setitem_nonneg`'s
    /// only check is `ll_assert`, which translation removes.
    #[test]
    fn plain_int_array_write_is_bare_setarrayitem_raw_i() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Int));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] = v);
        lowerer
            .lower_state_array_write(&expr)
            .expect("plain [int] store lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("setarrayitem_raw_i"), "{emitted}");
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [int] store must not emit a compare or guard: {emitted}"
        );
        assert!(
            !lowerer.op_metadata.iter().any(|m| {
                matches!(m.kind, OpKind::BinopI | OpKind::GotoIfNot)
                    || matches!(m.control, ControlFlowClass::ConditionalGuard)
            }),
            "plain [int] store must not emit a compare or guard: {:?}",
            lowerer
                .op_metadata
                .iter()
                .map(|m| format!("{:?}", m.kind))
                .collect::<Vec<_>>()
        );
    }

    /// A plain `[float]` read is the bare `getarrayitem_raw_f` item read.
    /// `jtransform.py` `rewrite_op_getarrayitem` selects the kind from
    /// `getkind(ARRAY.OF)`.
    #[test]
    fn plain_float_array_read_is_bare_getarrayitem_raw_f() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Float));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i]);
        let binding = lowerer
            .lower_state_array_read(&expr)
            .expect("plain [float] read lowers");
        assert_eq!(binding.kind, BindingKind::Float);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_raw_f"), "{emitted}");
        assert!(emitted.contains("add_raw_float_array_descr"), "{emitted}");
        assert!(
            !emitted.contains("getarrayitem_raw_i"),
            "float cells must not lower as int: {emitted}"
        );
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [float] read must not emit a compare or guard: {emitted}"
        );
    }

    /// A plain `[float]` store is the bare `setarrayitem_raw_f` item store.
    /// `jtransform.py` `rewrite_op_setarrayitem` selects the kind from the
    /// stored value's type.
    #[test]
    fn plain_float_array_write_is_bare_setarrayitem_raw_f() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Float));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] = v);
        lowerer
            .lower_state_array_write(&expr)
            .expect("plain [float] store lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("setarrayitem_raw_f"), "{emitted}");
        assert!(emitted.contains("add_raw_float_array_descr"), "{emitted}");
        assert!(
            !emitted.contains("setarrayitem_raw_i"),
            "float cells must not lower as int: {emitted}"
        );
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [float] store must not emit a compare or guard: {emitted}"
        );
    }

    /// `state.cells[i] += v` on a plain `[int]` is one index, a raw get, the
    /// int binop, and a raw set. `jtransform.py` `rewrite_op_getarrayitem` /
    /// `rewrite_op_setarrayitem` raw arms.
    #[test]
    fn plain_int_array_compound_assign_is_get_binop_set_raw_i() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Int));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] += v);
        lowerer
            .lower_state_array_update(&expr)
            .expect("plain [int] compound assign lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_raw_i"), "{emitted}");
        assert!(emitted.contains("setarrayitem_raw_i"), "{emitted}");
        assert!(
            emitted.contains("record_binop_i") && emitted.contains("IntAdd"),
            "{emitted}"
        );
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [int] compound assign must not emit a compare or guard: {emitted}"
        );
        assert!(
            !lowerer
                .op_metadata
                .iter()
                .any(|m| matches!(m.control, ControlFlowClass::ConditionalGuard)),
            "plain [int] compound assign must not emit a guard: {:?}",
            lowerer
                .op_metadata
                .iter()
                .map(|m| format!("{:?}", m.kind))
                .collect::<Vec<_>>()
        );
    }

    /// `state.cells[i] += v` on a plain `[float]` is one index, a raw get,
    /// the float binop, and a raw set.
    #[test]
    fn plain_float_array_compound_assign_is_get_binop_set_raw_f() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Float));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] += v);
        lowerer
            .lower_state_array_update(&expr)
            .expect("plain [float] compound assign lowers");
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("getarrayitem_raw_f"), "{emitted}");
        assert!(emitted.contains("setarrayitem_raw_f"), "{emitted}");
        assert!(emitted.contains("add_raw_float_array_descr"), "{emitted}");
        assert!(
            emitted.contains("record_binop_f") && emitted.contains("FloatAdd"),
            "{emitted}"
        );
        assert!(
            !emitted.contains("getarrayitem_raw_i") && !emitted.contains("setarrayitem_raw_i"),
            "float cells must not lower as int: {emitted}"
        );
        assert!(
            !emitted.contains("goto_if")
                && !emitted.contains("UintLt")
                && !emitted.contains("int_lt"),
            "plain [float] compound assign must not emit a compare or guard: {emitted}"
        );
    }

    /// `opcode_for_assign_binop_f` has no remainder; a kind mismatch is
    /// refused the same way. Both return `None` with no raw get/set.
    #[test]
    fn plain_array_compound_assign_refuses_unsupported_op_and_kind_mismatch() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Float));
        let mut lowerer = Lowerer::new(Some(&config));
        lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] %= v);
        assert!(
            lowerer.lower_state_array_update(&expr).is_none(),
            "float remainder-assign is not in opcode_for_assign_binop_f"
        );
        assert!(
            lowerer.statements.is_empty(),
            "an unsupported op must not emit: {:?}",
            lowerer
                .statements
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        );

        let mut int_config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        int_config
            .state_arrays
            .insert("cells".into(), (0, ValueKind::Int));
        let mut int_lowerer = Lowerer::new(Some(&int_config));
        int_lowerer.bindings.insert(
            "i".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        int_lowerer.bindings.insert(
            "v".to_string(),
            Binding {
                reg: 3,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: Expr = syn::parse_quote!(state.cells[i] += v);
        assert!(
            int_lowerer.lower_state_array_update(&expr).is_none(),
            "int cells plus a float rhs is a kind mismatch"
        );
        let emitted = int_lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            !emitted.contains("getarrayitem_raw") && !emitted.contains("setarrayitem_raw"),
            "a kind mismatch must not emit the raw get/set: {emitted}"
        );
    }
}
