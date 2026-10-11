//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemFn};

use super::{JitInterpConfig, StateFieldKind, VableArrayLayoutDecl, VirtualizableDecl};

fn vable_ir_type(field_type: &syn::Ident) -> TokenStream {
    if field_type == "ref" {
        quote! { majit_ir::Type::Ref }
    } else if field_type == "float" {
        quote! { majit_ir::Type::Float }
    } else {
        quote! { majit_ir::Type::Int }
    }
}

fn vable_array_add_tokens(array: &super::VableArrayDecl) -> TokenStream {
    let aname = array.name.to_string();
    let tp = vable_ir_type(&array.item_type);
    let item_size = quote! { majit_metainterp::virtualizable::item_size_for_type(#tp) };
    match &array.layout {
        VableArrayLayoutDecl::Direct {
            field_offset,
            length_offset,
            items_offset,
        } => {
            let length_offset = length_offset
                .clone()
                .unwrap_or_else(|| syn::parse_quote!(0usize));
            let items_offset = items_offset
                .clone()
                .unwrap_or_else(|| syn::parse_quote!(0usize));
            quote! {
                __info.add_array_field(
                    #aname,
                    #tp,
                    #field_offset,
                    #length_offset,
                    #items_offset,
                    majit_ir::make_array_descr(#items_offset, #item_size, #tp),
                );
            }
        }
        VableArrayLayoutDecl::Embedded {
            field_offset,
            ptr_offset,
            length_offset,
            items_offset,
        } => quote! {
            __info.add_embedded_array_field(
                #aname,
                #tp,
                #field_offset,
                #ptr_offset,
                #length_offset,
                #items_offset,
                majit_ir::make_array_descr(#items_offset, #item_size, #tp),
            );
        },
    }
}

/// `VirtualizableInfo` for a heap object stored in a `ref` state field.
///
/// `virtualizable.py` `VirtualizableInfo.__init__` plus
/// `warmstate.py` `execute_assembler`'s `clear_vable_token`: the object
/// is the storage, so entry and exit only reset the token. Field boxes
/// are read off the object (`read_boxes`) when a compiled entry
/// still expects the expanded input list.
fn heap_frame_virtualizable_methods(
    decl: &VirtualizableDecl,
    struct_path: &syn::Path,
    identity_live_index: usize,
    identity_ref_bank_index: usize,
) -> TokenStream {
    let var_name = &decl.var_name;
    let token_offset = &decl.token_offset;
    let name_str = decl.var_name.to_string();
    let field_adds: Vec<TokenStream> = decl
        .fields
        .iter()
        .map(|f| {
            let fname = f.name.to_string();
            let offset = &f.offset;
            let tp = vable_ir_type(&f.field_type);
            quote! {
                __info.add_field(#fname, #tp, #offset);
            }
        })
        .collect();
    let array_adds: Vec<TokenStream> = decl.arrays.iter().map(vable_array_add_tokens).collect();
    quote! {
        #[allow(non_snake_case)]
        fn __build_virtualizable_info()
        -> Option<::std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
            use majit_metainterp::virtualizable::VirtualizableInfo;
            let mut __info = VirtualizableInfo::new(#token_offset);
            __info.name = #name_str.to_string();
            // Flat inputarg slot of the object (`warmspot.py`
            // `index_of_virtualizable`). `identity_ref_bank_index` is the
            // JitCode ref register the dispatch walk passes as the vable
            // base; `Some` tells the optimizer not to treat inputarg 0 as
            // that object.
            __info.identity_live_index = Some(#identity_live_index);
            __info.identity_ref_bank_index = Some(#identity_ref_bank_index);
            #(#field_adds)*
            #(#array_adds)*
            Some(__info.finalize_arc(
                majit_ir::descr::make_size_descr(::std::mem::size_of::<#struct_path>()),
            ))
        }

        fn virtualizable_heap_ptr(
            &self,
            _meta: &Self::Meta,
            _virtualizable: &str,
            _info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<*mut u8> {
            let __ptr = self.#var_name as *mut u8;
            if __ptr.is_null() {
                None
            } else {
                Some(__ptr)
            }
        }

        fn sync_virtualizable_before_jit(
            &mut self,
            meta: &Self::Meta,
            virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> bool {
            // `warmstate.py` `execute_assembler`: enter with the token clear.
            // A zero token is `TOKEN_NONE` (`virtualizable.py`); resetting it
            // is a no-op until a compiled loop has stored one.
            if let Some(__obj) = self.virtualizable_heap_ptr(meta, virtualizable, info) {
                unsafe { info.reset_vable_token(__obj) };
            }
            true
        }

        fn sync_virtualizable_after_jit(
            &mut self,
            meta: &Self::Meta,
            virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) {
            if let Some(__obj) = self.virtualizable_heap_ptr(meta, virtualizable, info) {
                unsafe { info.reset_vable_token(__obj) };
            }
        }

        fn export_virtualizable_boxes(
            &self,
            meta: &Self::Meta,
            virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<(::std::vec::Vec<i64>, ::std::vec::Vec<usize>)> {
            let __ptr = self.virtualizable_heap_ptr(meta, virtualizable, info)?;
            if !info.can_read_all_array_lengths_from_heap() {
                return None;
            }
            let __lengths = unsafe { info.read_array_lengths_from_heap(__ptr) };
            let __boxes = unsafe { info.read_boxes(__ptr, &__lengths) };
            Some((__boxes, __lengths))
        }
    }
}

/// Generate the JitState types and implementation.
pub fn generate_jit_state(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    generate_state_fields_jit_state(config, func)
}

/// Generate JitState types for state_fields mode (register/tape machines).
///
/// Instead of a storage pool with stacks, individual struct fields are tracked
/// as JIT-managed values. Scalars become single OpRefs, flattened arrays become
/// Vec<OpRef>, and virtualizable arrays (`[int; virt]`) track only a data
/// pointer + length OpRef pair (array stays on heap, accessed via raw memory ops).
#[allow(unused_variables, unused_assignments)]
fn generate_state_fields_jit_state(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let prebuild_fn_name_with_lens = format_ident!(
        "__prebuild_jitcode_liveness_{}_with_array_lens",
        func.sig.ident
    );
    let dispatch_jitcode_fn_name_with_lens =
        format_ident!("__dispatch_jitcode_{}_with_array_lens", func.sig.ident);
    let declare_schema_fn_name_with_lens =
        format_ident!("__declare_jit_schema_{}_with_array_lens", func.sig.ident);
    // Every module-level item this macro emits is suffixed with the annotated
    // function's name, because the expansion lands in the CALLER's module and two
    // machines in one module would otherwise collide (E0428).  These five were
    // fixed names while the three above were already suffixed; the split was not
    // deliberate.  Note the ceiling: unique names let two machines share a module
    // only when their `state` types DIFFER -- two machines over the same state
    // type still conflict on `impl JitState for #state_type` (E0119), which no
    // naming scheme can fix.
    let meta_ty = format_ident!("__JitMeta_{}", func.sig.ident);
    let sym_ty = format_ident!("__JitSym_{}", func.sig.ident);
    let fresh_alloc_fn = format_ident!("__majit_recursive_fresh_alloc_{}", func.sig.ident);
    let fresh_free_fn = format_ident!("__majit_recursive_fresh_free_{}", func.sig.ident);
    let sf = config.state_fields.as_ref().unwrap();

    let is_int_or_float = |ty: &Ident| ty == "int" || ty == "float";
    // Each rejected field is rendered the way it was DECLARED, brackets
    // included. Rendering the element type alone makes a rejected `[float]`
    // read as `regs: float`, which is the scalar form the same message lists
    // as supported — the reader is told the thing they wrote is both.
    let unsupported_fields: Vec<String> = sf
        .fields
        .iter()
        .filter_map(|f| {
            let (rendered, supported) = match &f.kind {
                StateFieldKind::Scalar { ir_type, .. } => {
                    (format!("{}: {}", f.name, ir_type), is_int_or_float(ir_type))
                }
                // A flattened array is int-only: every consumer below reaches
                // its cells through the int register bank.
                StateFieldKind::Array(tp) => (format!("{}: [{}]", f.name, tp), tp == "int"),
                StateFieldKind::VirtArray(tp) => {
                    (format!("{}: [{}; virt]", f.name, tp), is_int_or_float(tp))
                }
                // opaque(T) fields are pass-through; the JIT does not
                // enumerate them as inputargs, so any T is allowed. ref(T) is
                // a ref-typed scalar (usize carrier), also any T.
                StateFieldKind::Opaque(_) | StateFieldKind::Ref(_) | StateFieldKind::Str => {
                    return None;
                }
            };
            (!supported).then_some(rendered)
        })
        .collect();
    if !unsupported_fields.is_empty() {
        let message = format!(
            "state_fields supports int, float, [int], [int; virt], [float; virt], ref(T), opaque(T), and str; unsupported: {}",
            unsupported_fields.join(", ")
        );
        return quote! {
            compile_error!(#message);
        };
    }

    // Separate int scalars, flattened arrays, virtualizable arrays, float scalars,
    // and ref scalars.
    let mut scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(
            |(_, f)| matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "int"),
        )
        .collect();
    // Helper: per-scalar Rust storage type token (`i64` by default, or
    // the explicit `int(<TypePath>)` override). Used to emit `as <type>`
    // casts at the JIT boundary so user struct fields can stay in their
    // natural Rust types (e.g. `selected: usize`, `stacksize: i32`).
    let scalar_rust_type = |kind: &StateFieldKind| -> TokenStream {
        match kind {
            StateFieldKind::Scalar {
                rust_type: Some(p), ..
            } => quote! { #p },
            StateFieldKind::Scalar { ir_type, .. } if ir_type == "float" => quote! { f64 },
            _ => quote! { i64 },
        }
    };
    let arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Array(_)))
        .collect();
    let virt_arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::VirtArray(_)))
        .collect();
    let mut ref_scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Ref(_) | StateFieldKind::Str))
        .collect();
    let mut float_scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "float")
        })
        .collect();
    // virtualizable.py VirtualizableInfo.__init__: scalar fields belong to
    // the same object as its arrays, not to an independent set of JitDriver reds.
    let declared_scalars = scalars.clone();
    let declared_ref_count = ref_scalars.len();
    let declared_float_count = float_scalars.len();
    let vable_scalars: Vec<_> = if virt_arrays.is_empty() {
        Vec::new()
    } else {
        sf.fields
            .iter()
            .filter(|f| {
                matches!(
                    f.kind,
                    StateFieldKind::Scalar { .. } | StateFieldKind::Ref(_)
                )
            })
            .collect()
    };
    if !virt_arrays.is_empty() {
        scalars.clear();
        ref_scalars.clear();
        float_scalars.clear();
    }
    // opaque(T) fields are pass-through carriers the JIT never enumerates as
    // inputargs and never reconstructs.  A fresh recursive-portal frame cannot
    // synthesize an arbitrary `T` generically, so any state shape carrying one
    // is excluded from the fresh-entry helpers below (they fall back to the
    // `None` default and the recursive dispatcher aborts to the interpreter).
    let opaque_fields: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Opaque(_)))
        .collect();

    let num_scalars = scalars.len();
    let num_virt_arrays = virt_arrays.len();
    // `pyjitpl.py reached_loop_header` carries the virtualizable
    // exactly ONCE: it is a single red (`warmspot.py:529-538
    // jd.index_of_virtualizable = jitdriver.reds.index(vname)`), and
    // `virtualizable.py:150-153` reads every `[.. ; virt]` array's length off
    // the live object instead of boxing it.  So a state contributes one
    // identity slot however many virt arrays it declares — a presence flag,
    // not a per-array count.
    let has_vable_identity = num_virt_arrays > 0;
    let num_vable_identity_slots = usize::from(has_vable_identity);
    // Int-bank slots the inline-frame snapshot trim may blank. The
    // alloc_reg floor always skips the identity range; the trim stays
    // gated on `split_dispatch` so a non-split arm's live identity
    // slots are not blanked out of the snapshot. See
    // `int_identity_reserved_end` below.
    let num_reserved_identity_slots = if config.split_dispatch {
        num_scalars
    } else {
        0
    };
    let num_ref_scalars = ref_scalars.len();
    let num_float_scalars = float_scalars.len();
    let portal_greens = super::portal_green_params(config, func);
    let portal_int_greens = portal_greens
        .iter()
        .filter(|(_, _, tag)| matches!(tag, super::green_type_tag::GreenTypeTag::Int))
        .count();
    let portal_ref_greens = portal_greens
        .iter()
        .filter(|(_, _, tag)| {
            matches!(
                tag,
                super::green_type_tag::GreenTypeTag::Ref
                    | super::green_type_tag::GreenTypeTag::Str
                    | super::green_type_tag::GreenTypeTag::Unicode
            )
        })
        .count();
    let portal_float_greens = portal_greens
        .iter()
        .filter(|(_, _, tag)| matches!(tag, super::green_type_tag::GreenTypeTag::Float))
        .count();
    let carried_greens = super::loop_carried_greens(config, func);
    let carried_int_greens = carried_greens
        .iter()
        .filter(|(_, kind)| matches!(kind, super::jitcode_lower::ValueKind::Int))
        .count();
    let carried_ref_greens = carried_greens
        .iter()
        .filter(|(_, kind)| matches!(kind, super::jitcode_lower::ValueKind::Ref))
        .count();
    let carried_float_greens = carried_greens
        .iter()
        .filter(|(_, kind)| matches!(kind, super::jitcode_lower::ValueKind::Float))
        .count();
    // First ref-bank register available for ref-scalar identity slots.
    // `MIFrame::setup_call` packs the dispatch JitCode's ref args densely
    // from r0 (`program` at r0, the virtualizable identity at r1 when
    // present — `with_vable_input_ref_reg(1)` in codegen_trace), and the
    // blackhole re-executes ops reading those argument registers, so the
    // identity slots start past them.  Mirrors
    // `LowererConfig::ref_identity_base`; the vable-presence condition
    // matches the lowerer's `vable_var` synthesis (an explicit
    // `virtualizable` decl or any `[int; virt]` state array).
    let ref_identity_base: usize = 1
        + portal_ref_greens
        + carried_ref_greens
        + usize::from(config.virtualizable_decl.is_some() || num_virt_arrays > 0);
    // A `virtualizable_fields` object that is a `ref` state field, not the
    // state struct. `[.. ; virt]` already makes the state itself the
    // virtualizable; the two do not combine.
    let heap_vable_index: Option<usize> = if num_virt_arrays == 0 {
        config.virtualizable_decl.as_ref().map(|decl| {
            if !arrays.is_empty() {
                panic!(
                    "virtualizable_fields cannot share a state with a fixed-length array; \
                     the object's live index would depend on that array's runtime length"
                );
            }
            let name = decl.var_name.to_string();
            let ref_pos = ref_scalars
                .iter()
                .position(|(_, f)| f.name == name)
                .unwrap_or_else(|| {
                    panic!("virtualizable_fields var `{name}` must be a `ref(_)` state field")
                });
            num_scalars + ref_pos
        })
    } else {
        None
    };
    let carry_vable_boxes = num_virt_arrays >= 1 || heap_vable_index.is_some();
    // First int-bank register available for scalar/array identity slots —
    // the int-bank mirror of `ref_identity_base`. `pc` is i0; portal
    // greens and loop-carried greens follow. An identity slot aliasing
    // one of those inputs would overwrite the green. Mirrors
    // `LowererConfig::int_identity_base`.
    let int_identity_base: usize = 1 + portal_int_greens + carried_int_greens;
    let float_identity_base: usize = portal_float_greens + carried_float_greens;

    let recover_body: TokenStream = if let Some(ref recover_path) = config.recover {
        quote! { self.#recover_path(); }
    } else {
        quote! {}
    };

    // `__JitMeta_<fn>` fields: one `{name}_len: usize` per flattened array
    // Virt arrays do NOT store length in meta: `virtualizable.py:150-153` reads
    // each one off the live object, so it is neither a meta field nor a box.
    let meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: usize, }
        })
        .collect();

    // `__JitSym_<fn>` holds dispatcher bookkeeping, not reds. Plain reds
    // live in the portal frame's identity slots (`MIFrame.setup_call`);
    // vable fields live in `virtualizable_boxes`; the vable identity lives
    // in the ref bank / `virtualizable_boxes[-1]`.
    // Flattened-array lengths are layout (slot count), copied from meta
    // at `create_sym` — `array_elem_slot` / jump-arg collection read them.
    // Virt-array length mirrors feed `recursive_fresh_entry_vable_capacities`.
    let sym_array_len_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: usize, }
        })
        .collect();
    // Per virt array only a plain `i64` length mirror survives: it feeds the
    // fresh-callee capacity, and is NEVER an inputarg
    // (`virtualizable.py VirtualizableInfo` reads the length off the live object).
    let sym_virt_array_fields: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { #len_value_name: i64, }
        })
        .collect();

    // ── JitCodeSym: total_slots ──
    // num_scalars + sum(flattened array lengths). The vable identity is a
    // Ref in the ref bank / `virtualizable_boxes[-1]`, not an int slot.
    let total_slots_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { + self.#len_name }
        })
        .collect();

    // `StateFieldLayout::array_elem_slot`: int_scalar_base + num_scalars
    // + sum of earlier array lengths + elem. Lengths are the layout
    // fields copied from meta at `create_sym`.
    let array_elem_slot_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(array_idx, (_, f))| {
            let len_name = quote::format_ident!("{}_len", f.name);
            let prev: Vec<TokenStream> = arrays[..array_idx]
                .iter()
                .map(|(_, prev)| {
                    let prev_len = quote::format_ident!("{}_len", prev.name);
                    quote! { + self.#prev_len }
                })
                .collect();
            quote! {
                #array_idx => {
                    let __base = #int_identity_base + #num_scalars #(#prev)*;
                    if elem < self.#len_name {
                        Some(__base + elem)
                    } else {
                        None
                    }
                }
            }
        })
        .collect();
    let array_elem_slot_override: TokenStream = if arrays.is_empty() {
        quote! {}
    } else {
        quote! {
            fn array_elem_slot(&self, array_idx: usize, elem: usize) -> Option<usize> {
                match array_idx {
                    #(#array_elem_slot_arms)*
                    _ => None,
                }
            }
        }
    };

    // ── build_meta: capture flattened array lengths ──
    let build_meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: self.#fname.len(), }
        })
        .collect();

    // ── canonical_liveness_slots: array_lens slice expression ──
    // RPython `assembler.py get_liveness_info` extracts per-kind
    // liveness for each `-live-` marker.  In flat-state JIT every slot
    // is permanently live, so the canonical entry is just
    // `[0..total_slots]` of int slots.  The `array_lens` slice fed to
    // `live_slots_for_state_field_jit` enumerates the runtime lengths
    // captured in `__JitMeta_<fn>::<arr>_len` (one per flattened array).
    let canonical_liveness_array_len_refs: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { self.#len_name }
        })
        .collect();

    // ── extract_live: scalars, then flattened array elements, then the vable identity ──
    let extract_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { values.push(self.#fname as i64); }
        })
        .collect();
    let extract_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for elem in &self.#fname {
                    values.push(*elem as i64);
                }
            }
        })
        .collect();
    // One value: the virtualizable identity (`&state` ==
    // `virtualizable_heap_ptr`), NOT any array's data pointer —
    // `vable_getarrayitem_*` reaches every element from this base through the
    // storage each field registered, and all `[.. ; virt]` arrays share it.
    // Emitting it once
    // is `virtualizable.py load_list_of_boxes`; the lengths stay off the red vector
    // (`virtualizable.py:150-153` reads them off the live object).
    let extract_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { values.push(self as *const Self as i64); }
    } else {
        quote! {}
    };
    let debug_scalar_state_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {}\n", #label, self.#fname as i64),
                );
            }
        })
        .collect();
    let debug_array_state_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} len={}\n", #label, self.#fname.len()),
                );
                for (__i, __v) in self.#fname.iter().enumerate() {
                    let _ = ::std::fmt::Write::write_fmt(
                        &mut out,
                        format_args!("    {}[{}] = {}\n", #label, __i, *__v as i64),
                    );
                }
            }
        })
        .collect();
    let debug_virt_array_state_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!(
                        "  {} len={} vable={:#x}\n",
                        #label,
                        self.#fname.len(),
                        self as *const Self as usize,
                    ),
                );
                for (__i, __v) in self.#fname.iter().enumerate() {
                    let _ = ::std::fmt::Write::write_fmt(
                        &mut out,
                        format_args!("    {}[{}] = {}\n", #label, __i, *__v as i64),
                    );
                }
            }
        })
        .collect();
    let debug_ref_scalar_state_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {:#x}\n", #label, self.#fname as usize),
                );
            }
        })
        .collect();
    let debug_scalar_label_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();
    let debug_array_label_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                for __i in 0..self.#fname.len() {
                    labels.push(::std::format!("{}[{}]", #label, __i));
                }
            }
        })
        .collect();
    let debug_vable_identity_label_part: TokenStream = if has_vable_identity {
        quote! { labels.push(::std::string::String::from("<vable>")); }
    } else {
        quote! {}
    };
    let debug_ref_scalar_label_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();

    // Recursive CALL_ASSEMBLER portal entry on the JitCodeSym side
    // A recursive callee runs as its own compiled loop with a fresh frame.
    // `recursive_fresh_entry_reds` allocates a fresh `#state_type` (scalars
    // zeroed = empty frame; arrays re-allocated at the caller's live
    // capacity) and emits its reds in `extract_live` order.  Capacities come
    // from this symbolic state: a fixed array's `{name}_len` layout field,
    // and a virt array's `{name}_len_value` (seeded at
    // `JitState::initialize_sym`).  The whole
    // struct equals `state_fields`, so these inits build a complete fresh
    // `#state_type`.
    let fresh_entry_scalar_inits: Vec<TokenStream> = declared_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: 0, }
        })
        .collect();
    let fresh_entry_array_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #fname: ::std::vec![0i64; self.#len_name], }
        })
        .collect();
    let fresh_entry_virt_array_inits: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            let zero = match &f.kind {
                StateFieldKind::VirtArray(tp) if tp == "float" => quote! { 0.0f64 },
                _ => quote! { 0i64 },
            };
            // Constructed through the backing trait rather than as a `vec![]`,
            // so the field keeps whatever container it was declared with. The
            // target type comes from the field this initializer fills.
            quote! {
                #fname: majit_metainterp::virt_array::VirtArrayBacking::filled(
                    #zero,
                    self.#len_value_name as usize,
                ),
            }
        })
        .collect();
    // Reds in `extract_live` order: int scalars, then flattened fixed-array
    // elements, then the single `&state` identity (Ref) when the state has a
    // virtualizable.  Mirrors `extract_scalar_parts` / `extract_array_parts` /
    // `extract_vable_identity_part` so the fresh reds match the callee loop's
    // input-arg layout and `live_value_types` routing.
    let fresh_entry_scalar_value_pushes: Vec<TokenStream> = scalars
        .iter()
        .map(|_| quote! { __values.push(majit_ir::Value::Int(0)); })
        .collect();
    let fresh_entry_array_value_pushes: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                for _ in 0..self.#len_name {
                    __values.push(majit_ir::Value::Int(0));
                }
            }
        })
        .collect();
    let fresh_entry_vable_identity_push: TokenStream = if has_vable_identity {
        quote! {
            __values.push(majit_ir::Value::Ref(majit_ir::GcRef(__base as usize)));
        }
    } else {
        quote! {}
    };
    // The freshly-boxed `&state` identity feeds the one vable Ref slot; only
    // bound when there is at least one virt array (fixed-array-only reds carry
    // no pointer).
    let fresh_entry_base_let: TokenStream = if has_vable_identity {
        quote! { let __base = &*__fresh as *const #state_type as i64; }
    } else {
        quote! {}
    };
    // Emitted only for state shapes whose whole fresh frame can be synthesized
    // generically: no ref scalars and no opaque(T) carriers (neither has a
    // generic fresh value, and the `#state_type` struct literal below omits
    // opaque fields).  Other shapes fall back to the `JitCodeSym` default
    // (`None`) and the recursive dispatcher aborts to the interpreter.
    let recursive_fresh_entry_reds_override: TokenStream =
        if declared_ref_count == 0 && declared_float_count == 0 && opaque_fields.is_empty() {
            quote! {
                fn recursive_fresh_entry_reds(
                    &self,
                ) -> Option<(Vec<majit_ir::Value>, Box<dyn ::core::any::Any>)> {
                    let __fresh: Box<#state_type> = Box::new(#state_type {
                        #(#fresh_entry_scalar_inits)*
                        #(#fresh_entry_array_inits)*
                        #(#fresh_entry_virt_array_inits)*
                    });
                    #fresh_entry_base_let
                    let mut __values: Vec<majit_ir::Value> = Vec::new();
                    #(#fresh_entry_scalar_value_pushes)*
                    #(#fresh_entry_array_value_pushes)*
                    #fresh_entry_vable_identity_push
                    Some((__values, __fresh as Box<dyn ::core::any::Any>))
                }
            }
        } else {
            quote! {}
        };
    let fresh_entry_vable_capacity_pushes: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { __capacities.push(self.#len_value_name); }
        })
        .collect();
    let recursive_fresh_entry_vable_capacities_override: TokenStream =
        if declared_ref_count == 0 && declared_float_count == 0 && opaque_fields.is_empty() {
            quote! {
                fn recursive_fresh_entry_vable_capacities(&self) -> Option<Vec<i64>> {
                    let mut __capacities = Vec::new();
                    #(#fresh_entry_vable_capacity_pushes)*
                    Some(__capacities)
                }
            }
        } else {
            quote! {}
        };

    // Recursive CALL_ASSEMBLER portal entry for host allocation and release
    // The compiled caller loop cannot `New` a host `#state_type` through the
    // IR, so the recursive dispatcher records a residual call to these host
    // helpers: `alloc` returns a fresh `Box::into_raw`-ed `#state_type`
    // (scalars zeroed, the single virt array sized at the caller's live
    // capacity passed in `__cap`), `free` drops it.  Emitted only for the
    // shape the single-capacity allocator supports: zero ref scalars, no
    // opaque carriers, no fixed arrays, exactly one virt array (the `tl`
    // storage shape).  Other shapes leave `recursive_fresh_alloc_free_targets`
    // at its `None` default so the dispatcher aborts.
    let supports_fresh_alloc = declared_ref_count == 0
        && opaque_fields.is_empty()
        && arrays.is_empty()
        && num_virt_arrays == 1;
    if supports_fresh_alloc && declared_float_count > 0 {
        return quote! {
            compile_error!(
                "state_fields float scalars are not supported with recursive portal fresh allocation yet"
            );
        };
    }
    let recursive_fresh_alloc_free_fns: TokenStream = if supports_fresh_alloc {
        let virt_name = &virt_arrays[0].1.name;
        let virt_zero = match &virt_arrays[0].1.kind {
            StateFieldKind::VirtArray(tp) if tp == "float" => quote! { 0.0f64 },
            _ => quote! { 0i64 },
        };
        quote! {
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn #fresh_alloc_fn(__cap: i64) -> i64 {
                let __fresh: ::std::boxed::Box<#state_type> = ::std::boxed::Box::new(#state_type {
                    #(#fresh_entry_scalar_inits)*
                    // Same backing-trait construction the fresh-reds path uses:
                    // the field keeps whatever container it was declared with,
                    // and the target type comes from the field being filled.
                    #virt_name: majit_metainterp::virt_array::VirtArrayBacking::filled(
                        #virt_zero,
                        __cap as usize,
                    ),
                });
                ::std::boxed::Box::into_raw(__fresh) as i64
            }
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn #fresh_free_fn(__ptr: i64) {
                if __ptr != 0 {
                    unsafe {
                        ::core::mem::drop(::std::boxed::Box::from_raw(__ptr as *mut #state_type));
                    }
                }
            }
        }
    } else {
        quote! {}
    };
    let recursive_fresh_alloc_free_targets_override: TokenStream = if supports_fresh_alloc {
        quote! {
            fn recursive_fresh_alloc_free_targets(&self) -> Option<(*const (), *const ())> {
                Some((
                    #fresh_alloc_fn as usize as *const (),
                    #fresh_free_fn as usize as *const (),
                ))
            }
        }
    } else {
        quote! {}
    };

    // ── create_sym: dispatcher handle only. Plain reds are planted by
    // `MIFrame.setup_call` from `__trace_*` argboxes; this no longer mints
    // InputArgs for them. Flattened-array lengths copy from meta so
    // `array_elem_slot` can name identity registers.
    let create_sym_array_len_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { let #len_name: usize = meta.#len_name; }
        })
        .collect();
    let create_sym_virt_array_inits: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { let #len_value_name = 0i64; }
        })
        .collect();
    let create_sym_array_len_names: Vec<syn::Ident> = arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len", f.name))
        .collect();
    let create_sym_virt_array_len_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len_value", f.name))
        .collect();
    // `create_sym_array_names` remains the field ident list for
    // `state_field_layout` array lengths read off native state, not the
    // sym — `self.#name.len()` there is the live `Vec`.
    let create_sym_array_names: Vec<&syn::Ident> = arrays.iter().map(|(_, f)| &f.name).collect();

    // ── is_compatible: check flattened array lengths match meta ──
    // Virt arrays always compatible (their length is read off the live object).
    let compat_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && self.#fname.len() == meta.#len_name }
        })
        .collect();

    // ── restore: write values back to state fields ──
    // The virtualizable identity slot is skipped (the Vec owns its storage and
    // compiled code already mutated the elements in place).
    // The compiled code writes directly to the heap backing the Vec, so
    // no element-level restore is needed.
    let restore_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                self.#fname = values[__offset] as #rust_ty;
                __offset += 1;
            }
        })
        .collect();
    // Write captured scalar state-field values back into native state.
    // `values[idx]` is the scalar at state-field index `idx`. recover runs
    // afterwards and overwrites the storage-derived caches, so only
    // scalars recover cannot re-derive (a `selected`-style storage index)
    // meaningfully carry through.
    let writeback_from_values_parts: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            let idx_lit = idx;
            quote! {
                self.#fname = values[#idx_lit] as #rust_ty;
            }
        })
        .collect();
    let writeback_live_scalar_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! { #idx => { self.#fname = value as #rust_ty; } }
        })
        .collect();
    let restore_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                let __arr_len = self.#fname.len();
                for i in 0..__arr_len {
                    self.#fname[i] = values[__offset + i];
                }
                __offset += __arr_len;
            }
        })
        .collect();
    // Skip the one identity slot — the virt-array data lives on the heap and
    // was already modified in place by compiled code.
    let restore_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { __offset += 1; }
    } else {
        quote! {}
    };
    // Single-pass close: write the walk-final virt-array element values
    // (captured off the trace-ctx shadow) into native state, one array at a
    // time in declaration order. Mirrors `restore_array_parts`'s per-element
    // copy, but the source is the element-only flat vector
    // (`collect_virtualizable_element_values`), so there are no ptr/len slots to
    // skip. The user's Vec is fixed-capacity (see the reallocation note on
    // `initialize_sym_virt_array_parts`), so the length matches the walk-final
    // element count.
    let writeback_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let assign = match &f.kind {
                StateFieldKind::VirtArray(tp) if tp == "float" => {
                    quote! { self.#fname[i] = f64::from_bits(values[__offset + i] as u64); }
                }
                _ => quote! { self.#fname[i] = values[__offset + i]; },
            };
            quote! {
                let __len = self.#fname.len();
                debug_assert!(
                    values.len() >= __offset + __len,
                    "writeback_virt_array: fewer element values than array length",
                );
                for i in 0..__len {
                    #assign
                }
                __offset += __len;
            }
        })
        .collect();
    // `{varr}_len_value` mirrors the current `<state>.<varr>` length for the
    // fresh-callee capacity. Accurate iff the varray's length does not
    // change during tracing — true for the in-tree examples, whose
    // backings are sized once at construction. A varray the source
    // resizes mid-walk would need a refresh: the capacity is read off
    // this symbolic state by `recursive_fresh_entry_vable_capacities`.
    let initialize_sym_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { sym.#len_value_name = self.#fname.len() as i64; }
        })
        .collect();

    // ── validate_close: flattened array lengths in sym match meta ──
    // Virt arrays always validate (their length is read off the live object).
    let validate_array_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && sym.#len_name == meta.#len_name }
        })
        .collect();

    // ── ref(T) scalars ──
    // Native `usize` carriers (raw GcRef / pointer bits). In the live-value
    // vector they sit after int scalars/arrays and the vable identity;
    // `live_value_types` tags those positions `Type::Ref` so `restore_values`
    // routes them to the ref bank.
    let extract_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { values.push(self.#fname as i64); }
        })
        .collect();
    let restore_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            // `live_value_types` routes the single vable identity `Ref` into
            // the ref bank ahead of the ref scalars, so ref scalar `j` lives at
            // `ref_values[num_vable_identity_slots + j]`, not `ref_values[j]`.
            let slot = num_vable_identity_slots + ref_idx;
            quote! { self.#fname = ref_values[#slot] as usize; }
        })
        .collect();
    // `reached_loop_header` builds one `live_arg_boxes` from the
    // `jit_merge_point` operands (`prepare_list_of_boxes`); the generated
    // identity-slot walk is no longer a second construction of that list.
    let collect_jump_args_with_boxes_method: TokenStream = if carry_vable_boxes {
        quote! {
            fn collect_jump_args_with_boxes(
                _sym: &#sym_ty,
                __boxes: &[(majit_ir::OpRef, majit_ir::Type)],
            ) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                if let Some((__op, _)) = __boxes.last() {
                    args.push(*__op);
                }
                let __elem_count = __boxes.len().saturating_sub(1);
                args.extend(__boxes[..__elem_count].iter().map(|(__op, _)| *__op));
                args
            }
        }
    } else {
        quote! {}
    };
    let writeback_live_ref_scalar_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            quote! { #ref_idx => { self.#fname = value as usize; } }
        })
        .collect();
    // ── float scalars ──
    // Native state stores f64 by default. Live-value / restore bits are the
    // same pattern `MIFrame.registers_f` and `BlackholeInterpreter.registers_f`
    // carry.
    let extract_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            // Widen to f64 before taking bits so the encoding is the 64-bit
            // representation the restore path reads back via
            // `f64::from_bits(_ as u64) as #rust_ty`. A `float(f32)` field's
            // own `to_bits()` yields 32-bit bits, which would round-trip
            // through `f64::from_bits` as a bogus value.
            quote! { values.push((self.#fname as f64).to_bits() as i64); }
        })
        .collect();
    let restore_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                self.#fname = f64::from_bits(float_values[#float_idx] as u64) as #rust_ty;
            }
        })
        .collect();
    let writeback_float_from_values_parts: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            let idx_lit = num_scalars + float_idx;
            quote! {
                self.#fname = f64::from_bits(values[#idx_lit] as u64) as #rust_ty;
            }
        })
        .collect();
    let writeback_live_float_scalar_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                #float_idx => {
                    self.#fname = f64::from_bits(value as u64) as #rust_ty;
                }
            }
        })
        .collect();
    let debug_float_scalar_state_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {}\n", #label, self.#fname as f64),
                );
            }
        })
        .collect();
    let debug_float_scalar_label_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();

    // Optional method overrides — emitted ONLY when ref scalars exist, so
    // interps with none generate a byte-identical token stream (the trait
    // defaults from `JitState` / `JitCodeSym` apply).
    // The virtualizable's value-routing type: one Ref for the identity,
    // whatever the number of `[.. ; virt]` arrays.
    let vable_identity_type_part: TokenStream = if has_vable_identity {
        quote! { types.push(majit_ir::Type::Ref); }
    } else {
        quote! {}
    };
    // Per-array value-routing types: one Int per element.
    let array_type_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for _ in 0..self.#fname.len() {
                    types.push(majit_ir::Type::Int);
                }
            }
        })
        .collect();
    // Writes the typed values straight into the driver's entry buffer, in
    // `extract_live` order, each field as the `Value` its type routes it to.
    // One pass over the state: the word half and the type half are what
    // `extract_live_into` / `live_value_types_into` produce for readers that
    // want them apart, and pairing the two here would fill both buffers and
    // walk them again for every entry, so the pair is emitted directly.
    // Emitted under the same condition as `live_value_types_into`, so a state
    // with only int fields keeps the trait default.
    let extract_live_value_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { out.push(majit_ir::Value::Int(self.#fname as i64)); }
        })
        .collect();
    let extract_live_value_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for elem in &self.#fname {
                    out.push(majit_ir::Value::Int(*elem as i64));
                }
            }
        })
        .collect();
    let extract_live_value_vable_identity_part: TokenStream = if has_vable_identity {
        quote! {
            out.push(majit_ir::Value::Ref(majit_ir::GcRef(self as *const Self as usize)));
        }
    } else {
        quote! {}
    };
    let extract_live_value_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { out.push(majit_ir::Value::Ref(majit_ir::GcRef(self.#fname as usize))); }
        })
        .collect();
    let extract_live_value_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            // Widened to f64 exactly as the word form widens before taking
            // bits, so the two forms carry the same value.
            quote! { out.push(majit_ir::Value::Float(self.#fname as f64)); }
        })
        .collect();
    let extract_live_values_into_override: TokenStream =
        if num_ref_scalars > 0 || num_virt_arrays > 0 || num_float_scalars > 0 {
            quote! {
                fn extract_live_values_into(
                    &self,
                    _meta: &#meta_ty,
                    out: &mut ::std::vec::Vec<majit_ir::Value>,
                    raw: &mut ::std::vec::Vec<i64>,
                    types: &mut ::std::vec::Vec<majit_ir::Type>,
                ) {
                    let _ = (raw, types);
                    #(#extract_live_value_scalar_parts)*
                    #(#extract_live_value_array_parts)*
                    #extract_live_value_vable_identity_part
                    #(#extract_live_value_ref_scalar_parts)*
                    #(#extract_live_value_float_scalar_parts)*
                }
            }
        } else {
            quote! {}
        };
    // No flattened-array length check and the extract override ignores
    // meta, so the steady entry does not probe `compiled_loops`.
    let steady_entry_without_meta = compat_checks.is_empty()
        && (num_ref_scalars > 0 || num_virt_arrays > 0 || num_float_scalars > 0);
    let fill_entry_reds_without_meta_override: TokenStream = if steady_entry_without_meta {
        quote! {
            fn fill_entry_reds_without_meta(
                &self,
                out: &mut ::std::vec::Vec<majit_ir::Value>,
            ) -> bool {
                #(#extract_live_value_scalar_parts)*
                #(#extract_live_value_array_parts)*
                #extract_live_value_vable_identity_part
                #(#extract_live_value_ref_scalar_parts)*
                #(#extract_live_value_float_scalar_parts)*
                true
            }
        }
    } else {
        quote! {}
    };
    // Same words as `fill_entry_reds_without_meta`, as `unspecialize_value`
    // results: Int is the value, Ref is the address, Float is `to_bits()`.
    // Static runs are one `extend_from_slice`; a flattened array's length is
    // read off the state, so those elements stay a loop.
    let raw_int_words: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { self.#fname as i64 }
        })
        .collect();
    let raw_ref_words: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { self.#fname as i64 }
        })
        .collect();
    let raw_float_words: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { (self.#fname as f64).to_bits() as i64 }
        })
        .collect();
    // `compat_checks` is one `&& self.<arr>.len() == meta.<arr>_len` per
    // flattened array, so an empty check list is an empty `arrays` list.
    // The array-push arm under that condition was unreachable.
    let fill_entry_raw_reds_override: TokenStream = if steady_entry_without_meta {
        // A separate `extend_from_slice` of a fixed word list lowers to a
        // handful of stores when this function is outlined. Inlined into
        // `enter_compiled_function_entry` that call becomes
        // `Vec::extend_from_slice`'s memmove. Spell the spare-capacity
        // arm as stores so the inline form keeps the outlined fast path.
        let raw_vable_expr: TokenStream = if has_vable_identity {
            quote! { self as *const Self as i64 }
        } else {
            quote! {}
        };
        let raw_words: Vec<TokenStream> = raw_int_words
            .iter()
            .cloned()
            .chain(has_vable_identity.then(|| raw_vable_expr))
            .chain(raw_ref_words.iter().cloned())
            .chain(raw_float_words.iter().cloned())
            .collect();
        let raw_count = raw_words.len();
        let raw_writes: Vec<TokenStream> = raw_words
            .iter()
            .enumerate()
            .map(|(index, word)| {
                quote! { __dst.add(#index).write(#word); }
            })
            .collect();
        quote! {
            #[inline]
            fn fill_entry_raw_reds(&self, out: &mut ::std::vec::Vec<i64>) -> bool {
                let __len = out.len();
                if out.capacity().wrapping_sub(__len) >= #raw_count {
                    unsafe {
                        let __dst = out.as_mut_ptr().add(__len);
                        #(#raw_writes)*
                        out.set_len(__len + #raw_count);
                    }
                } else {
                    out.extend_from_slice(&[
                        #(#raw_words),*
                    ]);
                }
                true
            }
        }
    } else {
        quote! {}
    };
    let portal_result_type: TokenStream = match super::finish_return_for(&func.sig.output) {
        Some(finish) => {
            let ty = match finish.kind {
                super::FinishReturnKind::Int => quote!(majit_ir::Type::Int),
                super::FinishReturnKind::Float => quote!(majit_ir::Type::Float),
                super::FinishReturnKind::Ref => quote!(majit_ir::Type::Ref),
            };
            quote! {
                const PORTAL_RESULT_TYPE: ::core::option::Option<majit_ir::Type> = ::core::option::Option::Some(#ty);
            }
        }
        None => quote! {},
    };
    let live_value_types_override: TokenStream =
        if num_ref_scalars > 0 || num_virt_arrays > 0 || num_float_scalars > 0 {
            quote! {
                // A bank with no scalars emits `0..0`, which is data, not the
                // `for i in 10..0` typo `reversed_empty_ranges` is aimed at —
                // and that lint is deny-by-default in every consumer of this
                // macro, so a state of arrays only would not compile there.
                #[allow(clippy::reversed_empty_ranges)]
                fn live_value_types_into(
                    &self,
                    _meta: &#meta_ty,
                    types: &mut ::std::vec::Vec<majit_ir::Type>,
                ) {
                    // Value-routing types in `extract_live` order: int scalars,
                    // int array elements, then the ONE virtualizable identity
                    // (Ref), then appended ref scalars (Ref), then appended float
                    // scalars (Float).
                    // The identity slot MUST be Ref so the live `&state` is a Ref
                    // failarg (TAGBOX), which the resume reader decodes through
                    // `decode_ref` in both the vable section and the frame
                    // ref-liveness.
                    for _ in 0..#num_scalars {
                        types.push(majit_ir::Type::Int);
                    }
                    #(#array_type_parts)*
                    #vable_identity_type_part
                    for _ in 0..#num_ref_scalars {
                        types.push(majit_ir::Type::Ref);
                    }
                    for _ in 0..#num_float_scalars {
                        types.push(majit_ir::Type::Float);
                    }
                }

                fn live_value_types(&self, _meta: &#meta_ty) -> Vec<majit_ir::Type> {
                    let mut types: Vec<majit_ir::Type> = Vec::new();
                    self.live_value_types_into(_meta, &mut types);
                    types
                }
            }
        } else {
            quote! {}
        };
    let restore_banked_override: TokenStream = if num_ref_scalars > 0 || num_float_scalars > 0 {
        quote! {
            fn restore_banked3(
                &mut self,
                meta: &#meta_ty,
                int_values: &[i64],
                ref_values: &[i64],
                float_values: &[i64],
            ) {
                // Int scalars/arrays/virt restore from the int bank exactly
                // as `restore`; ref/float scalars restore from their bank by
                // 0-based bank-local index. Float values are raw f64 bits.
                self.restore(meta, int_values);
                #(#restore_ref_scalar_parts)*
                #(#restore_float_scalar_parts)*
            }
        }
    } else {
        quote! {}
    };
    // Emit the virt-array write-back override only for consumers that declare a
    // `[.. ; virt]` array; 0-array interps (tlr, tinyframe, i64env) keep
    // the trait default no-op, leaving their generated impl byte-identical.
    let writeback_virt_array_override: TokenStream = if num_virt_arrays > 0 {
        quote! {
            fn writeback_virt_array_state_fields_from_values(&mut self, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#writeback_virt_array_parts)*
            }
        }
    } else {
        quote! {}
    };
    let float_field_accessor_overrides: TokenStream = if num_float_scalars > 0 {
        quote! {
            fn float_identity_slots_base(&self) -> usize {
                #float_identity_base
            }
            fn float_scalar_slot(&self, field_idx: usize) -> Option<usize> {
                if field_idx < #num_float_scalars {
                    Some(#float_identity_base + field_idx)
                } else {
                    None
                }
            }
        }
    } else {
        quote! {}
    };
    let ref_field_accessor_overrides: TokenStream = if num_ref_scalars > 0 {
        quote! {
            fn ref_scalar_slot(&self, field_idx: usize) -> Option<usize> {
                if field_idx < #num_ref_scalars {
                    Some(#ref_identity_base + field_idx)
                } else {
                    None
                }
            }
        }
    } else {
        quote! {}
    };
    // Always pass `ref_identity_base`: a virt-array state with no ref
    // scalars still has a vable identity at `ref_identity_base - 1`,
    // and `StateFieldLayout::new` leaves `ref_scalar_base` at 0 so
    // `vable_identity_ref_slot` would alias r0 (`program`).
    let state_field_layout_ctor: TokenStream = quote! {
        majit_metainterp::blackhole::StateFieldLayout::with_ref_scalars(
            #num_scalars,
            ::std::vec![#(self.#create_sym_array_names.len()),*],
            #num_vable_identity_slots,
            #num_ref_scalars,
            #ref_identity_base,
            #int_identity_base,
        ).with_float_scalars(#num_float_scalars, #float_identity_base)
    };

    // Naming the virtualizable on the jitdriver static data
    // (`warmspot.py make_virtualizable_infos`) is what makes
    // `compile.py`'s field-reload preamble run, which retires the
    // per-entry re-export of the virtualizable's array elements: the compiled
    // entry reloads them from the virtualizable pointer instead of being handed
    // one entry argument per element.
    //
    // The name arrives through `JitDriver::declare_flat_entry_contract` together
    // with the width of the entry it applies to, because the two halves are only
    // meaningful together: the entry is the flat live-value prefix, not the red
    // count (the whole state is a single red in the merge-point payload), and an
    // index into one picks out a different value than the same index into the
    // other. That prefix is `num_scalars + num_vable_identity_slots +
    // num_ref_scalars + num_float_scalars` with the identity at flat index
    // `num_scalars` — the order `extract_live` emits — valid only while the
    // state declares no fixed arrays, the same restriction `identity_live_index`
    // below carries and for the same reason: a fixed array contributes its
    // runtime length to the prefix, and no constant here can name a length that
    // is read off a state instance.
    //
    // Whether the contract CAN be declared at all is the declared field type's
    // answer, not this expansion's: `compile.py:441-457`'s reconstruction
    // reaches each array's data pointer with a load the trace IR has to be able
    // to express, and a `Vec` embedded by value defeats that — its data pointer
    // is not at a specified offset within it, so no field load portably finds
    // it. `JitDriver::arm_flat_entry_contract` answers that off the vinfo this
    // expansion already installed, and declines rather than declaring, so the
    // width below is stated unconditionally and the structural gate stays where
    // the storage is known. Left unarmed, a state keeps the per-entry re-export
    // and behaves exactly as it did before.
    //
    // SOUNDNESS. Arming puts this driver on
    // `patch_new_loop_to_load_virtualizable_fields`, which BAKES each array's
    // trace-start length into the prologue as a fixed count of `GETARRAYITEM`
    // ops (`compile.py:443`), and nothing re-reads it afterwards. The invariant
    // stated on that helper is that the lengths are a function of the trace's
    // GREENS, so a virtualizable with other lengths keys to a different trace
    // and never reaches this entry. It is the interpreter author's to satisfy —
    // a `[.. ; virt]` array whose length can vary while the greens stay fixed
    // must not be block-backed — and it cannot be checked here: the lengths live
    // on state instances that do not exist at install time.
    let arm_flat_entry_contract: TokenStream =
        if (num_virt_arrays > 0 && arrays.is_empty()) || heap_vable_index.is_some() {
            let entry_len =
                num_scalars + num_vable_identity_slots + num_ref_scalars + num_float_scalars;
            let index_of_virtualizable = heap_vable_index.unwrap_or(num_scalars);
            quote! {
                driver.arm_flat_entry_contract(
                    majit_metainterp::FlatEntryContract {
                        len: #entry_len,
                        index_of_virtualizable: #index_of_virtualizable,
                    },
                );
            }
        } else {
            quote! {}
        };

    // pyjitpl.py `rebuild_state_after_failure`:
    //     if vinfo is not None:
    //         self.virtualizable_boxes = virtualizable_boxes
    // The guard's vable section is decoded by `rebuild_from_resumedata`;
    // rebuild the shadow so the bridge traces with the same virtualizable
    // the parent loop had. Without it `virtualizable_boxes` stays None and
    // `__trace_*` aborts at `standard_virtualizable_jitcode_argbox`.
    let seed_bridge_vable: TokenStream = if num_virt_arrays > 0 || heap_vable_index.is_some() {
        quote! {
            if let Some(__vinfo) = Self::__build_virtualizable_info() {
                let __seeded = majit_metainterp::seed_bridge_virtualizable_boxes(
                    ctx,
                    &__vinfo,
                    rd_virtuals,
                    resume_data,
                    reader,
                    fail_values,
                );
                if majit_metainterp::bridge_debug_enabled() {
                    eprintln!(
                        "[bridgeB] vable seed={} stream={}",
                        __seeded,
                        resume_data.virtualizable_boxes.len(),
                    );
                }
                // A declined seed is not recoverable here and must not pass
                // silently: `virtualizable_boxes` stays unset, so the first
                // vable-shaped op in `__trace_*` has nothing to resolve and the
                // bridge trace aborts — the guard keeps deopting through the
                // blackhole, which is `compile.giveup()`'s outcome
                // (compile.py:27) reached one step later. Surface it on the
                // ordinary log channel so a bridge that never forms is
                // attributable to the seed rather than looking like an
                // unexplained abort.
                if !__seeded && majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][bridgeB] virtualizable seed declined \
                         (vable stream {} entries) — bridge will abort and \
                         the guard deopts through the blackhole",
                        resume_data.virtualizable_boxes.len(),
                    );
                }
            }
        }
    } else {
        quote! {}
    };

    // ── VirtualizableInfo / heap-ptr overrides for `[int; virt]` arrays ──
    // Scalars and arrays share one standard virtualizable. Its field boxes
    // follow the identity red in the resume stream (virtualizable.py read_boxes).
    //
    // Which storage the field registers as is the declared field type's to say,
    // not this expansion's: `register_virt_array_field` resolves it from the
    // container the interpreter author wrote. That is the difference the
    // compiled entry sees — only a field holding a pointer to a block with a
    // fixed payload offset can be reloaded by `compile.py:441-457`, and a `Vec`
    // embedded by value is not one.
    let build_vinfo_override: TokenStream = if num_virt_arrays > 0 {
        let outer_executor_owns_state = !super::has_single_pass_close(func);
        let scalar_field_parts: Vec<_> = vable_scalars
            .iter()
            .map(|f| {
                let name = &f.name;
                let (kind, ty, signed) = match &f.kind {
                    StateFieldKind::Ref(_) => {
                        (quote!(majit_ir::Type::Ref), quote!(usize), quote!(false))
                    }
                    StateFieldKind::Scalar { ir_type, .. } if ir_type == "float" => {
                        let ty = scalar_rust_type(&f.kind);
                        (quote!(majit_ir::Type::Float), ty, quote!(false))
                    }
                    _ => {
                        let ty = scalar_rust_type(&f.kind);
                        (
                            quote!(majit_ir::Type::Int),
                            ty.clone(),
                            quote!(<#ty>::MIN != 0),
                        )
                    }
                };
                quote! {
                    __info.add_field_sized(stringify!(#name), #kind,
                        ::std::mem::offset_of!(#state_type, #name),
                        ::std::mem::size_of::<#ty>(), #signed);
                }
            })
            .collect();
        let scalar_export_parts: Vec<_> = vable_scalars
            .iter()
            .map(|f| {
                let name = &f.name;
                match &f.kind {
                    StateFieldKind::Scalar { ir_type, .. } if ir_type == "float" => {
                        quote!((self.#name as f64).to_bits() as i64)
                    }
                    _ => quote!(self.#name as i64),
                }
            })
            .collect();
        // Per virt array: a registration keyed on the field byte offset.
        let virt_array_field_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .map(|(_, f)| {
                let fname = &f.name;
                let fname_str = f.name.to_string();
                let (item_size, item_type) = match &f.kind {
                    StateFieldKind::VirtArray(tp) if tp == "float" => (
                        quote! { ::std::mem::size_of::<f64>() },
                        quote! { majit_ir::Type::Float },
                    ),
                    _ => (
                        quote! { ::std::mem::size_of::<i64>() },
                        quote! { majit_ir::Type::Int },
                    ),
                };
                quote! {
                    majit_metainterp::virt_array::register_virt_array_field(
                        &mut __info,
                        #fname_str,
                        #item_type,
                        #item_size,
                        ::std::mem::offset_of!(#state_type, #fname),
                        |__s: &#state_type| &__s.#fname,
                    );
                }
            })
            .collect();
        // Field idents in `virt_arrays` order — the same order
        // `register_virt_array_field` registers them, which is the order
        // `VirtualizableInfo::box_types` names their items in. Each pushes its
        // array's length and appends its items to the one box list.
        let virt_array_export_into_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .map(|(_, f)| {
                let fname = &f.name;
                let source = match &f.kind {
                    StateFieldKind::VirtArray(tp) if tp == "float" => {
                        quote! { self.#fname.iter().map(|&__x| __x.to_bits() as i64) }
                    }
                    _ => quote! { self.#fname.iter().map(|&__x| __x as i64) },
                };
                quote! {
                    array_lengths.push(self.#fname.len());
                    boxes.extend(#source);
                }
            })
            .collect();
        // `extract_live` pushes int scalars, then every fixed array's items,
        // then the one identity slot. With no fixed arrays that position is a
        // constant — `num_scalars` — so the identity can be DECLARED the way
        // `warmspot.py:529-538` declares `index_of_virtualizable`, and
        // `initialize_virtualizable` can look it up instead of searching the
        // reds for a matching pointer.
        //
        // A fixed `[int]` array alongside a `[int; virt]` one — which
        // `majit-metainterp/tests/jit_interp_float_state_field.rs`
        // `virt_array_with_float_scalar` declares — makes the position
        // `num_scalars + sum(array lengths)`, and those lengths are the runtime
        // `Vec` lengths this expansion reads back through `meta.<arr>_len`
        // (`create_sym_array_inits`), so no constant can be emitted here — the
        // vinfo itself is built once per driver by the `&self`-less
        // `__build_virtualizable_info`, before any state instance exists.
        // Emitting nothing is therefore correct, but it is NOT harmless on its
        // own: it leaves the identity's position unstated, and both consumers
        // must resolve it instead of assuming one.
        // `MetaInterp::identity_live_position` does resolve it for the runtime
        // path — it pointer-matches `vable_ptr` against the reds, so a wrong or
        // absent declaration is survivable there. The optimizer has no pointer
        // to match against, so with nothing declared it DECLINES to track the
        // virtualizable (`VirtualizableConfig::identity_input_index` is `None`)
        // rather than probing flat slot 0 — an int scalar on this layout, which
        // made every trace abort with VirtualStatesCantMatch. Declining was
        // measured to cost nothing here; see
        // `tests/jit_interp_fixed_array_identity_slot.rs`.
        let identity_live_index_stmt: TokenStream = if arrays.is_empty() {
            quote! { __info.identity_live_index = Some(#num_scalars); }
        } else {
            quote! {}
        };
        quote! {
            #[allow(non_snake_case)]
            fn __build_virtualizable_info()
            -> Option<::std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
                use majit_metainterp::virtualizable::VirtualizableInfo;
                // No `vable_token` field: the stack-local state struct is
                // non-GC and never moved, so the token protocol is inert — the
                // identity value (a `&state` pointer) is recovered straight
                // from the resume snapshot, not via a heap token. The struct's
                // offset 0 is a live user field, so every token read/write must
                // no-op rather than land there.
                let mut __info = VirtualizableInfo::without_vable_token();
                __info.name = "state".to_string();
                __info.outer_executor_owns_state = #outer_executor_owns_state;
                // The dispatch lowering binds the green ref `program` to ref
                // register 0 (it is the base for `program[pc]` reads) and the
                // `&state` virtualizable identity to ref register 1
                // (`vable_input_ref_reg = 1`, jitcode_lower/mod.rs). Tell
                // `initialize_virtualizable` to mint the standard box at that
                // ref-bank index so it matches the traced vable base (the flat
                // `num_green_args + index_of_virtualizable` ordinal would
                // resolve to 0, the slot the green ref occupies).
                __info.identity_ref_bank_index = Some(1);
                #identity_live_index_stmt
                #(#scalar_field_parts)*
                #(#virt_array_field_parts)*
                Some(__info.finalize_arc(
                    majit_ir::descr::make_size_descr(::std::mem::size_of::<#state_type>()),
                ))
            }

            fn virtualizable_heap_ptr(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Option<*mut u8> {
                // The state struct is a stack-allocated mainloop local —
                // stable and non-GC. Use its address as the vable identity
                // heap pointer.
                Some(self as *const Self as *mut u8)
            }

            // warmstate.py `execute_assembler`: all a compiled entry
            // does to its virtualizable is `vinfo.clear_vable_token(virt)` —
            // one store, before the call, and nothing after it. There is no
            // copy in and no copy out, because the virtualizable IS the
            // storage the compiled code reads and writes.
            //
            // The default hooks copy in and copy out instead, which is what a
            // host needs when its interpreter-side storage is a separate
            // mirror of the virtualizable object. This state is not such a
            // host: `virtualizable_heap_ptr` above hands out its own address,
            // and every array field is registered at its byte offset within
            // it, so the boxes the default reads and the boxes it writes back
            // address the same memory. The copy in is discarded outright —
            // nothing here overrides `import_virtualizable_boxes`, whose
            // default drops the boxes it is handed — and the copy out reads
            // each element and writes it back unchanged. Both are no-ops that
            // cost a pass over every array, plus the vectors to hold it, on
            // every entry into compiled code.
            //
            // So keep the token store and drop the copies. The store is itself
            // inert while the info carries no token field, which is the case
            // for a state whose offset 0 is a live user field; it is written
            // rather than omitted so that a state which later declares one
            // gets the upstream behaviour without this having to be revisited.
            fn sync_virtualizable_before_jit(
                &mut self,
                meta: &Self::Meta,
                virtualizable: &str,
                info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> bool {
                if let Some(__obj) = self.virtualizable_heap_ptr(meta, virtualizable, info) {
                    unsafe { info.reset_vable_token(__obj) };
                }
                true
            }

            fn sync_virtualizable_after_jit(
                &mut self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) {
            }

            fn export_virtualizable_boxes(
                &self,
                meta: &Self::Meta,
                virtualizable: &str,
                info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Option<(::std::vec::Vec<i64>, ::std::vec::Vec<usize>)> {
                // warmstate.py:482-511: supply the live virtualizable field
                // values so `extend_compiled_live_values` can grow the entry
                // `live_values` to the compiled loop's full inputarg width.
                // The array elements were seeded as boxes by
                // `initialize_virtualizable` at trace start and carried as
                // loop inputargs; re-entry must re-supply them in
                // `VirtualizableInfo::read_boxes` order (statics, then each
                // array ascending).
                let mut boxes = ::std::vec::Vec::new();
                let mut array_lengths = ::std::vec::Vec::new();
                <Self as majit_metainterp::JitState>::export_virtualizable_boxes_into(
                    self,
                    meta,
                    virtualizable,
                    info,
                    &mut boxes,
                    &mut array_lengths,
                )
                .then_some((boxes, array_lengths))
            }

            // Export into driver-owned buffers, preserving their capacity.
            fn export_virtualizable_boxes_into(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
                boxes: &mut ::std::vec::Vec<i64>,
                array_lengths: &mut ::std::vec::Vec<usize>,
            ) -> bool {
                // A virt-array-only state has no static boxes, so this
                // slice is empty. `extend([])` cannot pick `Extend<i64>`
                // over `Extend<&i64>`; `extend_from_slice` names `&[i64]`.
                boxes.extend_from_slice(&[#(#scalar_export_parts),*]);
                #( #virt_array_export_into_parts )*
                true
            }
        }
    } else if let (Some(decl), Some(index)) = (config.virtualizable_decl.as_ref(), heap_vable_index)
    {
        let struct_path = ref_scalars.iter().find_map(|(_, f)| match &f.kind {
            StateFieldKind::Ref(path) if f.name == decl.var_name => Some(path),
            _ => None,
        });
        let Some(struct_path) = struct_path else {
            panic!(
                "virtualizable_fields var `{}` must be a `ref(_)` state field",
                decl.var_name
            );
        };
        // The vable base register follows the portal's ref greens
        // (`with_vable_input_ref_reg` in codegen_trace). `ref_identity_base`
        // is the first state ref scalar, one past that register.
        let identity_ref_bank_index = ref_identity_base.saturating_sub(1);
        heap_frame_virtualizable_methods(decl, struct_path, index, identity_ref_bank_index)
    } else {
        quote! {}
    };

    quote! {
        /// Compiled loop metadata for state_fields mode: flattened array lengths at trace start.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct #meta_ty {
            #(#meta_fields)*
        }

        impl #meta_ty {
            /// RPython `assembler.py get_liveness_info(insn, kind)`
            /// adapted for flat-state JIT: every state_field slot is
            /// permanently live, so the canonical `(live_i, live_r,
            /// live_f)` triple is `live_i = 0..total_slots` in the int
            /// bank (int scalars, fixed-array elements, virt-array
            /// ptr/len) plus `live_r = 0..num_ref_scalars` for any
            /// `ref(T)` scalars carried in the ref bank.  `live_f` is
            /// always empty (no float state fields).  Used by
            /// `JitCodeBuilder::live` (`assembler.py:148+158`) to
            /// register the canonical entry once per process and emit
            /// a `live/<offset>` prefix on each per-opcode jitcode.
            #[allow(dead_code)]
            fn canonical_liveness_slots(
                &self,
            ) -> (::std::vec::Vec<u8>, ::std::vec::Vec<u8>, ::std::vec::Vec<u8>) {
                let __array_lens: &[usize] = &[#(#canonical_liveness_array_len_refs),*];
                majit_metainterp::live_slots_for_state_field_jit(
                    #num_scalars,
                    __array_lens,
                    #num_vable_identity_slots,
                    #num_ref_scalars,
                    #ref_identity_base,
                    #num_float_scalars,
                    #float_identity_base,
                    #int_identity_base,
                )
            }

            /// RPython `warmspot.py:281-289`'s `make_jitcodes() →
            /// finish_setup(codewriter)` lifecycle reduced to the
            /// canonical-entry slice for state-field JIT
            /// (`pyjitpl.py self.liveness_info = "".join(
            /// asm.all_liveness)`).  Builds a fresh `Assembler`,
            /// registers the canonical
            /// `(live_i, live_r, live_f)` triple via
            /// `Assembler::_encode_liveness` (`assembler.py`),
            /// then publishes the resulting `all_liveness` payload
            /// through `JitDriver::install_canonical_liveness`.
            ///
            /// Caller pattern:
            /// ```ignore
            /// let meta = state.build_meta(0, &program);
            /// meta.install_canonical_liveness(&mut driver);
            /// ```
            /// Must run before the first trace — the
            /// `Arc::get_mut` invariant on `MetaInterp::staticdata`
            /// (`pyjitpl.rs::install_canonical_liveness`) panics
            /// once any tracing setup has cloned the Arc.
            ///
            /// This only installs the canonical liveness entry and
            /// opcode ids.  Consumers whose macro-emitted per-pc
            /// JitCodes can register additional liveness entries via
            /// `JitCodeBuilder::finalize_liveness(__asm)` must build
            /// those JitCodes before the first trace, then call
            /// `JitDriver::sync_liveness_info_from_shared_asm()`.  That
            /// reproduces RPython's order: all `-live-` entries are in
            /// `asm.all_liveness` before `finish_setup` snapshots
            /// `metainterp_sd.liveness_info`.
            #[allow(dead_code)]
            fn install_canonical_liveness(
                &self,
                driver: &mut majit_metainterp::JitDriver<#state_type>,
            ) {
                // RPython `codewriter.py` calls `CallControl.__init__`
                // (`call.py`) before `assemble()` produces the jitcodes
                // that read `jitdriver_sd.index`. Pyre's analog: stamp the
                // descriptor onto the driver before the dispatch JitCode
                // build below reads jdindex via
                // `driver.index().expect(...)`.
                //
                // `ensure_descriptor_registered` mirrors PyPy's `for
                // index, jd in enumerate(jitdrivers_sd): jd.index = index`
                // — when the consumer constructed the driver via
                // `JitDriver::with_descriptor(threshold, jd)`, that jd
                // (carrying `greens`/`reds`/`virtualizable`/result_type
                // info) is registered in place; only when no descriptor
                // was pre-built does an empty stub get registered as a
                // pyre-only fail-soft.  Idempotent: re-entry is a no-op
                // once `driver.index()` returns `Some(_)`.
                //
                // Slice (audit Issue #5) — populate the JitDriver's
                // green / red schema BEFORE
                // `ensure_descriptor_registered` runs, so the
                // descriptor that gets registered carries the real
                // `(name, IR Type)` pairs from the dispatch
                // JitCode body's `BC_JIT_MERGE_POINT` rather than the
                // empty stub.  `green_kind_counts` / `red_kind_counts`
                // then reflect the actual payload partition.
                #declare_schema_fn_name_with_lens(
                    driver,
                    &[#(self.#create_sym_array_len_names),*],
                );
                // `warmspot.py make_virtualizable_infos` names the
                // virtualizable on the jitdriver static data during setup, i.e.
                // before the driver is registered. Order matters here for the
                // same reason: `ensure_descriptor_registered` MOVES the
                // descriptor into the `MetaInterpStaticData` table, so a
                // contract declared after it would land on a descriptor no
                // consumer reads.
                #arm_flat_entry_contract
                driver.ensure_descriptor_registered();
                // Register canonical entry +
                // canonical opcode ids into the driver-shared
                // `Assembler` (cf. `JitDriver::shared_asm`) so per-pc
                // factory calls dedup against the same
                // `all_liveness_positions` and append into the same
                // `all_liveness` byte stream.
                let __shared_asm = driver.shared_asm();
                {
                    let mut __asm = __shared_asm
                        .lock();
                    let (__live_i, __live_r, __live_f) = self.canonical_liveness_slots();
                    // Stage the canonical "all-live" triple for lazy
                    // registration. The first leading-dummy `BC_LIVE`
                    // patched by `JitCodeBuilder::finalize_liveness`
                    // calls `ensure_canonical_liveness_offset`, which
                    // registers the triple at the END of `all_liveness`
                    // (after the per-marker prebuild has populated the
                    // IR-walk-ordered head).  Matches RPython
                    // `assembler.assemble`'s shape: per-marker `-live-`
                    // entries occupy the early offsets; pyre's canonical
                    // entry lands at the tail as a leading-dummy
                    // affordance.
                    __asm.set_canonical_liveness_triple(
                        __live_i,
                        __live_r,
                        __live_f,
                    );
                    // RPython `assembler.py:222 self.insns[key] = opnum`
                    // records every opcode the assembler emits during
                    // `assemble()`.  pyre's macro path skips
                    // `assembler.assemble()` (the per-arm `JitCodeBuilder`
                    // emits BC_* directly), so the canonical state-field
                    // JIT entries are registered explicitly here.  The
                    // downstream `MetaInterpStaticData::
                    // install_canonical_liveness` then calls
                    // `setup_insns(asm.insns())` (`pyjitpl.py`)
                    // to dynamically resolve `op_live` /
                    // `op_catch_exception` / `op_*_return` instead of a
                    // parallel hardcoded `BC_*` seeding block.
                    __asm.register_insn("live/", majit_metainterp::jitcode::insns::BC_LIVE);
                    __asm.register_insn(
                        "catch_exception/L",
                        majit_metainterp::jitcode::insns::BC_CATCH_EXCEPTION,
                    );
                    __asm.register_insn(
                        "rvmprof_code/ii",
                        majit_metainterp::jitcode::insns::BC_RVMPROF_CODE,
                    );
                    __asm.register_insn(
                        "int_return/i",
                        majit_metainterp::jitcode::insns::BC_INT_RETURN,
                    );
                    __asm.register_insn(
                        "ref_return/r",
                        majit_metainterp::jitcode::insns::BC_REF_RETURN,
                    );
                    __asm.register_insn(
                        "float_return/f",
                        majit_metainterp::jitcode::insns::BC_FLOAT_RETURN,
                    );
                    __asm.register_insn(
                        "void_return/",
                        majit_metainterp::jitcode::insns::BC_VOID_RETURN,
                    );
                    // RPython `pyjitpl.py finish_setup` builds every
                    // JitCode and stamps every per-marker `-live-` triple
                    // into `asm.all_liveness` *before* snapshotting
                    // `metainterp_sd.liveness_info`. Pyre's lazy factory
                    // can't eagerly build (pc, op) pairs, so the macro-
                    // generated `__prebuild_jitcode_liveness_*` function
                    // pre-registers each lowered arm's per-marker triples
                    // into the same locked shared assembler. After the
                    // snapshot below, trace-time
                    // `JitCodeBuilder::finalize_liveness` only dedups —
                    // the table never grows past this point (asserted in
                    // `__trace_*`).
                    #prebuild_fn_name_with_lens(
                        &mut __asm,
                        &[#(self.#create_sym_array_len_names),*],
                    );
                    // Build the dispatch JitCode singleton against the
                    // same shared assembler. `__prebuild_jitcode_liveness_*`
                    // registers per-marker triples for both the dispatch
                    // JitCode and every per-arm JitCode, so the
                    // `finalize_liveness` calls inside this factory only
                    // dedup — they do not grow `asm.all_liveness` past the
                    // prebuild snapshot. Mirrors `pyjitpl.py:2264
                    // finish_setup`, where `metainterp_sd.liveness_info`
                    // is snapshotted only after every JitCode has been
                    // built and every `-live-` triple stamped.
                    // Single-phase jdindex resolution (jtransform.py:1704):
                    // `register_descriptor` ran above (line 689 onwards),
                    // unconditionally stamping the index on the driver
                    // before this point. Read it through the now-`Some`
                    // accessor and bake it into the dispatch JitCode body.
                    //
                    // Codex Pre-A.3 review BLOCKER (a) absorption: a fake
                    // `0` index must never end up baked into a registered
                    // JitCode body. With `register_descriptor` ordered
                    // before this read, the `expect()` is a structural
                    // invariant — it can fire only if a future change
                    // accidentally moves the registration after this site.
                    let __jdindex: i64 = driver.index().expect(
                        "register_descriptor must run before install_canonical_liveness — \
                         RPython call.py:46-47 / codewriter.py:23-24 lifecycle invariant"
                    ) as i64;
                    let __dispatch_jc_opt = #dispatch_jitcode_fn_name_with_lens(
                        &mut __asm,
                        __jdindex,
                        &[#(self.#create_sym_array_len_names),*],
                    );
                    // Safety net: ensure the canonical entry has a
                    // registered offset before the snapshot, even if no
                    // per-pc factory has run a `finalize_liveness` yet
                    // to trigger the lazy registration. Subsequent calls
                    // short-circuit via the cached
                    // `canonical_liveness_offset`.
                    let _ = __asm.ensure_canonical_liveness_offset();
                    driver.install_canonical_liveness(&__asm);
                    // PyPy `make_jitcodes()` / `pyjitpl.py finish_setup
                    // finish_setup()` only install completed jitcodes —
                    // there is no path where a body that the codewriter
                    // failed to lower lands as a successfully-installed
                    // singleton.  When `lower_dispatch_body` returned
                    // `None` at proc-macro time, the dispatch builder
                    // returns `None` here; skip
                    // `register_dispatch_jitcode` to match that
                    // lifecycle.  Successful builds (`Some(jc)`) install
                    // unconditionally per PyPy
                    // `pypy/module/pypyjit/interp_jit.py`.
                    if let Some(__dispatch_jc) = __dispatch_jc_opt {
                        driver.register_dispatch_jitcode(__dispatch_jc);
                    }
                }
            }
        }

        #recursive_fresh_alloc_free_fns

        /// Dispatcher handle during tracing. Plain reds live in the portal
        /// frame; vable fields live in `virtualizable_boxes`.
        #[allow(non_camel_case_types)]
        struct #sym_ty {
            #(#sym_array_len_fields)*
            #(#sym_virt_array_fields)*
            loop_header_pc: usize,
            trace_started: bool,
        }

        impl majit_metainterp::JitCodeSym for #sym_ty {
            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)*
            }

            fn loop_carried_boxes(
                &self,
                __boxes: &[(majit_ir::OpRef, majit_ir::Type)],
            ) -> Option<Vec<(majit_ir::OpRef, majit_ir::Type)>> {
                let _ = __boxes;
                None
            }

            #[allow(clippy::reversed_empty_ranges)]
            fn collect_portal_scalar_values(
                &self,
                __frame: &majit_metainterp::MIFrame,
                __ctx: &majit_metainterp::TraceCtx,
            ) -> Vec<i64> {
                let mut values = Vec::new();
                for __k in 0..#num_scalars {
                    let __slot = #int_identity_base + __k;
                    if let Some(__op) = __frame.int_regs.get(__slot).copied().flatten()
                        && let Some(__v) = __ctx.box_bits(__op)
                    {
                        values.push(__v);
                    }
                }
                for __k in 0..#num_float_scalars {
                    let __slot = #float_identity_base + __k;
                    if let Some(__op) = __frame.float_regs.get(__slot).copied().flatten()
                        && let Some(__v) = __ctx.box_bits(__op)
                    {
                        values.push(__v);
                    }
                }
                values
            }

            #[allow(clippy::reversed_empty_ranges)]
            fn collect_portal_ref_scalar_values(
                &self,
                __frame: &majit_metainterp::MIFrame,
                __ctx: &majit_metainterp::TraceCtx,
            ) -> Vec<i64> {
                let mut values = Vec::new();
                for __j in 0..#num_ref_scalars {
                    let __slot = #ref_identity_base + __j;
                    if let Some(__op) = __frame.ref_regs.get(__slot).copied().flatten()
                        && let Some(__v) = __ctx.box_bits(__op)
                    {
                        values.push(__v);
                    }
                }
                values
            }

            fn int_identity_slots_base(&self) -> usize {
                #int_identity_base
            }

            // Mirrors `split_identity_reg_ends`' int end in
            // `jitcode_lower/mod.rs`: the working-register floor stops after
            // the scalars plus the single vable-identity slot, because a virt
            // array's element count is only known from the live object.
            //
            // The snapshot trim keyed on this end blanks the range
            // unconditionally. `split_identity_floor` now always raises a
            // sub-JitCode's `alloc_reg()` past the identity range so a temp
            // cannot clobber a red, but the trim stays gated on
            // `split_dispatch` (`num_reserved_identity_slots`): blanking a
            // non-split arm's live identity slots would drop the reds
            // capture needs. Report an empty range without `split_dispatch`.
            fn int_identity_reserved_end(&self) -> usize {
                #int_identity_base + #num_reserved_identity_slots
            }

            fn loop_header_pc(&self) -> usize {
                self.loop_header_pc
            }

            #array_elem_slot_override
            #ref_field_accessor_overrides
            #float_field_accessor_overrides

            #recursive_fresh_entry_reds_override
            #recursive_fresh_entry_vable_capacities_override

            #recursive_fresh_alloc_free_targets_override
        }

        impl majit_metainterp::JitState for #state_type {
            type Meta = #meta_ty;
            type Sym = #sym_ty;
            type Env = #env_type;

            #portal_result_type

            fn can_trace(&self) -> bool {
                true
            }

            fn build_meta(&self, _header_pc: usize, _program: &#env_type) -> #meta_ty {
                #meta_ty {
                    #(#build_meta_fields)*
                }
            }

            // The buffer-filling form is the primary one and the owning form
            // wraps it: `warmstate.py maybe_compile_and_run` hands
            // `execute_assembler` reds that are already unboxed locals, so the
            // entry path allocates nothing to describe them. The driver keeps
            // these buffers across calls, so a warm entry refills them.
            fn extract_live_into(&self, _meta: &#meta_ty, values: &mut ::std::vec::Vec<i64>) {
                #(#extract_scalar_parts)*
                #(#extract_array_parts)*
                #extract_vable_identity_part
                #(#extract_ref_scalar_parts)*
                #(#extract_float_scalar_parts)*
            }

            fn extract_live(&self, _meta: &#meta_ty) -> Vec<i64> {
                let mut values = Vec::new();
                self.extract_live_into(_meta, &mut values);
                values
            }

            #live_value_types_override

            #extract_live_values_into_override
            #fill_entry_reds_without_meta_override
            #fill_entry_raw_reds_override

            fn create_sym(meta: &#meta_ty, header_pc: usize) -> #sym_ty {
                #(#create_sym_array_len_inits)*
                #(#create_sym_virt_array_inits)*
                #sym_ty {
                    #(#create_sym_array_len_names,)*
                    #(#create_sym_virt_array_len_value_names,)*
                    loop_header_pc: header_pc,
                    trace_started: false,
                }
            }

            fn initialize_sym(&self, sym: &mut #sym_ty, _meta: &#meta_ty) {
                #(#initialize_sym_virt_array_parts)*
            }

            // `create_sym` no longer mints red InputArgs; a bound count of
            // zero is the orthodox shape (`clear_sym_inputarg_bindings` is
            // a matching no-op).
            fn count_bound_sym_inputargs(_sym: &#sym_ty) -> Option<usize> {
                Some(0)
            }

            fn clear_sym_inputarg_bindings(_sym: &mut #sym_ty) {}

            // ── Part A (bridge resume-decode). ──
            //
            // resume.py rebuild_from_resumedata parity for the
            // JitDriver state.  Without this the trait default returns None and
            // `start_bridge_tracing` aborts (jitdriver.rs) so no guard-exit
            // bridge ever forms — a failing loop guard re-enters via
            // ContinueRunningNormally instead of forming a bridge.  Adding it
            // flips `start_bridge_tracing` ok=false→ok=true and bridges form;
            // existing consumers stay byte-identical.
            //
            // NOTE: this is general guard-exit bridge-formation infrastructure.
            // A trace that spins is not by itself evidence of a missing or
            // unseeded bridge: one such spin was instead a peeled loop that
            // constant-folded a red and dropped the matching exit guard,
            // leaving a state-mutating residual to loop forever.
            fn rebuild_from_resumedata(
                _meta: &mut #meta_ty,
                fail_arg_types: &[majit_ir::Type],
                storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
            ) -> Option<majit_metainterp::ResumeDataResult> {
                // resume.py rebuild_from_resumedata:
                //     while not resumereader.done_reading():
                //         jitcode_pos, pc = resumereader.read_jitcode_pos_pc()
                //         jitcode = metainterp.staticdata.jitcodes[jitcode_pos]
                //         f = metainterp.newframe(jitcode); f.setup_resume_at_op(pc)
                //         resumereader.consume_boxes(f.get_current_position_info(), ..)
                //
                // Every section is delimited by ITS OWN jitcode's `-live-`
                // liveness at ITS OWN pc (jitcode.py `enumerate_vars` ->
                // length_i + length_r + length_f); RPython never consumes "the
                // rest of the stream" as one frame.  The writer already honours
                // that contract — `build_state_field_snapshot`
                // (pyjitpl/dispatch.rs) emits one section per MIFrame, outermost
                // to innermost, each stamping its own absolute jitcode index —
                // so a `#[jit_inline]` callee on the frame stack at guard time
                // publishes a multi-frame stream.  Decoding that with the `None`
                // fallback folds the next section's [jitcode_index, pc]
                // header and values into frame 0, so frame 0 stops matching its
                // own liveness and the per-bank register -> sym-slot map in
                // `setup_bridge_sym` is meaningless.
                //
                // `register_dispatch_jitcode` installs the liveness splitter
                // (`install_state_field_fvc`); decode through it exactly like
                // the other compile-time decoders of the same `rd_numb` already
                // do.  Deliberately not `.expect()`ed: a state whose
                // `lower_dispatch_body` failed never calls
                // `register_dispatch_jitcode`, so an absent callback is
                // legitimate and keeps the previous fallback behaviour.
                let storage = storage?;
                let rd_numb = storage.rd_numb().expect("rd_numb");
                let rd_consts = storage.rd_consts().unwrap_or(&[]);
                let __fvc = majit_ir::resumedata::get_frame_value_count_fn();
                let __fvc_ref: ::std::option::Option<&dyn Fn(i32, i32) -> usize> =
                    __fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                let (num_failargs, vable_values, vref_values, frames) =
                    majit_ir::resumedata::rebuild_from_numbering(
                        rd_numb,
                        rd_consts,
                        fail_arg_types,
                        __fvc_ref,
                        storage.rd_virtuals().map_or(0, <[_]>::len),
                    );
                if frames.is_empty() {
                    return None;
                }
                Some(majit_metainterp::ResumeDataResult {
                    frames,
                    virtualizable_boxes: vable_values,
                    virtualref_values: vref_values,
                    storage: Some(storage.clone()),
                    num_failargs,
                    fail_arg_types: fail_arg_types.to_vec(),
                })
            }

            // ── Part B (bridge sym seeding) — resume.py rebuild_from_resumedata/1054 setup_bridge_sym
            //    + consume_boxes parity for the JitDriver state.  Seeds each red
            //    slot's symbolic OpRef + concrete shadow from the guard's decoded
            //    resume frame so the bridge specializes its guards on the real
            //    loop state (without this a guard-exit bridge re-traces
            //    un-seeded). ──
            //
            // Flattened `[int]` cell identity slots are seeded from resume
            // the way scalars are: `bridge_reg_indices` names the live
            // register of each cell (`identity_slot_registers` /
            // `handle_jit_marker__jit_merge_point`), and `consume_boxes`
            // copies those live int registers into the portal frame.
            //
            // `frame.values` is laid out by liveness bank: [int-bank, then
            // ref-bank, then float], greens/loop-invariants decoded as `Const`,
            // live reds as `Box(n)`.  The optimizer renumbers the int bank, so
            // the i-th int Box is NOT necessarily int scalar i — routing by a
            // naive kind-counter mis-binds (e.g. pool_ptr → stacksize slot,
            // deref-crashing on the swapped pointer).  Instead we read the
            // per-bank live REGISTER index of each value from the guard jitcode's
            // liveness (`ctx.bridge_reg_indices()`, stashed by
            // `start_bridge_tracing`) and map register → decl slot via
            // IDENTITY-SLOT MATCHING: each state field is read during tracing
            // by `load_state_field(fi)` (lower_vable.rs) from its FIXED identity
            // register `int_identity_base + fi` (refs: `ref_identity_base + fi`),
            // and that slot is kept live across guards, so the resume frame
            // carries the field's current value at exactly that register.  We
            // therefore locate, for each decl slot k, the frame value whose live
            // register == identity_base + k (NOT a positional/kind-counter zip —
            // the optimizer puts recomputed temps like stacksize's `.size`
            // reload at high working registers, and promotes loop-invariant
            // fields like `selected` to `Const`, so only the identity register
            // reliably names the field).  Box → `input_arg(n)` + `fail_values[n]`;
            // Const → a folded pool constant.  MAJIT_BRIDGE_DEBUG dumps it.
            // `pyjitpl.py rebuild_state_after_failure` →
            // `MIFrame.setup_resume_at_op(pc)`: put the walk back at the
            // guard's own jitcode position so the rest of the opcode it sits
            // inside is traced rather than run where the trace cannot see it.
            // One frame per encoded resume section, outermost first, each with
            // its registers already decoded; nothing here is machine-specific
            // except the `Sym` type, which is why this forwards rather than
            // generating a walk of its own.
            // Empty unless `split_dispatch` raised the sub-JitCodes' alloc
            // floor past the range, which is the only case where a non-root
            // frame's snapshot loses registers to the identity trim.
            fn reserved_int_identity_range() -> Option<(usize, usize)> {
                let (base, end) = (
                    #int_identity_base,
                    #int_identity_base + #num_reserved_identity_slots,
                );
                (end > base).then_some((base, end))
            }

            fn trace_from_guard_resume_position<__R: majit_metainterp::JitCodeRuntime>(
                ctx: &mut majit_metainterp::TraceCtx,
                sym: &mut #sym_ty,
                frames: &[majit_metainterp::GuardResumeFrame],
                outer_program_pc: usize,
                runtime: &__R,
            ) -> Option<majit_metainterp::TraceAction> {
                Some(majit_metainterp::trace_jitcode_at_resume_framestack(
                    ctx,
                    sym,
                    frames,
                    outer_program_pc,
                    runtime,
                ))
            }

            // resume.py _prepare_pendingfields — materialize the guard's virtuals + replay
            // its deferred heap writes as bridge-entry NEW/SETFIELD_GC ops so
            // the compiled bridge observes the heap state the blackhole deopt
            // would rebuild. A push/dup whose node is virtualized-and-elided
            // defers its head-store to rd_pendingfields while the size store
            // commits inline; without this replay the bridge reads size>chain
            // and dereferences a NULL node head. Runs independent of the
            // frame-register seeding in `setup_bridge_sym` (which may decline).
            #[allow(unused_variables)]
            fn prepare_bridge_resume(
                _sym: &mut #sym_ty,
                ctx: &mut majit_metainterp::TraceCtx,
                resume_data: &majit_metainterp::ResumeDataResult,
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                fail_values: &[i64],
                fail_types: &[majit_ir::Type],
                reader: &mut majit_metainterp::BridgeVirtualCache<'_>,
            ) {
                if !majit_metainterp::replay_pending_fields(
                    ctx,
                    resume_data,
                    rd_virtuals,
                    reader,
                ) {
                    ctx.mark_bridge_replay_incomplete();
                }
            }

            // resume.py `consume_vref_and_vable_boxes`: the virtualizable
            // boxes are decoded, and here seeded onto the trace, before any
            // frame's `consume_boxes`.
            #[allow(clippy::reversed_empty_ranges, unused_variables)]
            fn consume_vref_and_vable_boxes(
                _sym: &mut #sym_ty,
                ctx: &mut majit_metainterp::TraceCtx,
                resume_data: &majit_metainterp::ResumeDataResult,
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                fail_values: &[i64],
                fail_types: &[majit_ir::Type],
                reader: &mut majit_metainterp::BridgeVirtualCache<'_>,
            ) -> majit_metainterp::VrefVableBoxes {
                #seed_bridge_vable
                majit_metainterp::VrefVableBoxes::default()
            }

            #[allow(unused_variables)]
            fn setup_bridge_sym(
                _sym: &mut #sym_ty,
                ctx: &mut majit_metainterp::TraceCtx,
                resume_data: &majit_metainterp::ResumeDataResult,
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                fail_values: &[i64],
                fail_types: &[majit_ir::Type],
                reader: &mut majit_metainterp::BridgeVirtualCache<'_>,
                _boxes: &majit_metainterp::VrefVableBoxes,
            ) {
                if majit_metainterp::bridge_diag_enabled() {
                    eprintln!(
                        "[setup_bridge_sym] CALLED frames={} rd_virtuals={}",
                        resume_data.frames.len(),
                        rd_virtuals.map_or(0, |v| v.len()),
                    );
                }
            }

            fn is_compatible(&self, meta: &#meta_ty) -> bool {
                true #(#compat_checks)*
            }

            fn restore(&mut self, _meta: &#meta_ty, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#restore_scalar_parts)*
                #(#restore_array_parts)*
                #restore_vable_identity_part
            }

            #restore_banked_override

            fn recover_after_compiled_run(&mut self) {
                #recover_body
            }

            fn debug_state_fields(&self, _meta: &#meta_ty) -> Option<::std::string::String> {
                let mut out = ::std::string::String::new();
                #(#debug_scalar_state_parts)*
                #(#debug_array_state_parts)*
                #(#debug_virt_array_state_parts)*
                #(#debug_ref_scalar_state_parts)*
                #(#debug_float_scalar_state_parts)*
                Some(out)
            }

            fn debug_state_live_labels(&self, _meta: &#meta_ty) -> Option<::std::vec::Vec<::std::string::String>> {
                let mut labels: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                #(#debug_scalar_label_parts)*
                #(#debug_array_label_parts)*
                #debug_vable_identity_label_part
                #(#debug_ref_scalar_label_parts)*
                #(#debug_float_scalar_label_parts)*
                Some(labels)
            }

            fn collect_scalar_state_field_values(_sym: &Self::Sym) -> Vec<i64> {
                Vec::new()
            }

            fn collect_ref_scalar_state_field_values(_sym: &Self::Sym) -> Vec<i64> {
                Vec::new()
            }

            fn writeback_scalar_state_fields_from_values(&mut self, values: &[i64]) {
                if values.len() < #num_scalars + #num_float_scalars {
                    return;
                }
                #(#writeback_from_values_parts)*
                #(#writeback_float_from_values_parts)*
            }

            fn writeback_live_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_scalar_arms)*
                    _ => {}
                }
            }

            fn writeback_live_ref_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_ref_scalar_arms)*
                    _ => {}
                }
            }

            fn writeback_live_float_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_float_scalar_arms)*
                    _ => {}
                }
            }

            #writeback_virt_array_override

            fn state_field_layout(&self) -> majit_metainterp::blackhole::StateFieldLayout {
                // Flat slot layout for blackhole resume: scalar count is
                // static, each flattened fixed `[int]` array contributes its
                // live length. The vable identity is a Ref, not an int slot.
                // Ref scalars add a parallel 0-based ref-bank count. Mirrors
                // `extract_live` / `live_slots_for_state_field_jit`.
                #state_field_layout_ctor
            }

            #collect_jump_args_with_boxes_method

            fn validate_close(sym: &#sym_ty, meta: &#meta_ty) -> bool {
                true #(#validate_array_checks)*
            }

            // `pyjitpl.py MetaInterp.capture_resumedata` for jitdriver-level
            // guard sites (e.g. `force_finish_trace`'s GuardAlwaysFails).
            // Identity slots already hold the reds; this only walks the
            // live framestack.
            fn populate_frame_for_guard(
                sym: &#sym_ty,
                frames: &mut majit_metainterp::MIFrameStack,
                __op_live: u8,
                __all_liveness: &[u8],
                __virtualizable_boxes: &[majit_ir::OpRef],
                __virtualref_boxes: &[(majit_ir::OpRef, usize)],
            ) -> Option<majit_metainterp::recorder::Snapshot> {
                use majit_metainterp::JitCodeSym as _;
                if frames.frames.is_empty() {
                    return None;
                }
                Some(majit_metainterp::build_state_field_snapshot(
                    frames,
                    __op_live,
                    __all_liveness,
                    false,
                    __virtualizable_boxes,
                    __virtualref_boxes,
                    Some((sym.int_identity_slots_base(), sym.int_identity_reserved_end())),
                ))
            }

            #build_vinfo_override
        }
    }
}
