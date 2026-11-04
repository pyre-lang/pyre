//! Standalone `virtualizable!` proc macro implementation.
//!
//! Generates `VirtualizableInfo` builder, JitState layout helpers, and
//! virtualizable hook functions from a declarative specification.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, LitStr, Path, Token, braced,
    ext::IdentExt,
    parse::{Parse, ParseStream},
};

use crate::jit_interp::{VableArrayDecl, VableArrayLayoutDecl, VableFieldDecl, VirtualizableDecl};

pub(crate) mod codegen;
mod derive;

pub use derive::{expand_meta, expand_state, expand_sym};

// ═══════════════════════════════════════════════════════════════
// Input arg field: a scalar field included in extract_live / jump_args.
// ═══════════════════════════════════════════════════════════════

struct InputArgField {
    /// Field name in the state struct (e.g., `next_instr`).
    name: Ident,
    /// IR type: `Int`, `Ref`, or `Float`.
    ir_type: Ident,
}

// ═══════════════════════════════════════════════════════════════
// Parsed macro configuration.
// ═══════════════════════════════════════════════════════════════

pub(crate) struct VirtualizableMacroInput {
    state_type: Ident,
    decl: VirtualizableDecl,
    heap_ptr_expr: Expr,
    /// Frame pointer field name in the state struct (e.g., `frame`).
    frame_field: Option<Ident>,
    /// Vable scalar fields (= RPython `_virtualizable_` static fields,
    /// `interp_jit.py:25-30`). Read/written via getfield_vable on the
    /// vable heap object; included in extract_live / jump_args.
    inputargs: Vec<InputArgField>,
    /// Extra red inputargs that are NOT vable scalar fields (= RPython
    /// `JitDriver.reds` minus the vable, e.g. `ec` in
    /// `interp_jit.py reds = ['frame', 'ec']`). Pushed to
    /// extract_live / jump_args between `frame` and the vable static
    /// fields, mirroring `pyjitpl.py:2957 redboxes` followed by
    /// `:2964 + virtualizable_boxes`. Restored via plain
    /// `state.set_<name>(...)` accessors on the host-side state struct
    /// (no vable heap deref).
    extra_reds: Vec<InputArgField>,
    /// IR type for array items (default: Ref).
    array_item_type: Option<Ident>,
}

impl Parse for VirtualizableMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut var_name = None;
        let mut token_offset = None;
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        let mut heap_ptr_expr = None;
        let mut frame_field = None;
        let mut inputargs = Vec::new();
        let mut extra_reds = Vec::new();
        let mut array_item_type = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "name" => {
                    let lit: LitStr = input.parse()?;
                    var_name = Some(Ident::new(&lit.value(), lit.span()));
                }
                "heap_ptr" => {
                    heap_ptr_expr = Some(input.parse::<Expr>()?);
                }
                "token_offset" => {
                    token_offset = Some(input.parse::<Path>()?);
                }
                "frame_field" => {
                    frame_field = Some(input.parse::<Ident>()?);
                }
                "inputargs" => {
                    let inner;
                    braced!(inner in input);
                    while !inner.is_empty() {
                        let name: Ident = inner.parse()?;
                        inner.parse::<Token![:]>()?;
                        let ir_type: Ident = inner.parse()?;
                        inputargs.push(InputArgField { name, ir_type });
                        let _ = inner.parse::<Token![,]>();
                    }
                }
                "extra_reds" => {
                    // Non-vable JitDriver red inputargs (e.g. `ec` in
                    // `interp_jit.py reds = ['frame', 'ec']`). Same syntax
                    // as `inputargs`; layout-wise they sit between the frame
                    // pointer and the vable static fields in the canonical
                    // scalar block (`pyjitpl.py:2957 redboxes` then
                    // `:2964 + virtualizable_boxes`).
                    let inner;
                    braced!(inner in input);
                    while !inner.is_empty() {
                        let name: Ident = inner.parse()?;
                        inner.parse::<Token![:]>()?;
                        let ir_type: Ident = inner.parse()?;
                        extra_reds.push(InputArgField { name, ir_type });
                        let _ = inner.parse::<Token![,]>();
                    }
                }
                "array_item_type" => {
                    array_item_type = Some(input.parse::<Ident>()?);
                }
                "fields" => {
                    let inner;
                    braced!(inner in input);
                    while !inner.is_empty() {
                        let name: Ident = inner.parse()?;
                        inner.parse::<Token![:]>()?;
                        let field_type: Ident = inner.call(Ident::parse_any)?;
                        inner.parse::<Token![@]>()?;
                        let offset: Expr = if inner.peek(syn::token::Paren) {
                            let expr_content;
                            syn::parenthesized!(expr_content in inner);
                            expr_content.parse::<Expr>()?
                        } else {
                            inner.parse::<Expr>()?
                        };
                        fields.push(VableFieldDecl {
                            name,
                            field_type,
                            offset,
                        });
                        let _ = inner.parse::<Token![,]>();
                    }
                }
                "arrays" => {
                    let inner;
                    braced!(inner in input);
                    while !inner.is_empty() {
                        let name: Ident = inner.parse()?;
                        inner.parse::<Token![:]>()?;
                        let item_type: Ident = inner.call(Ident::parse_any)?;
                        inner.parse::<Token![@]>()?;
                        let field_offset: Expr = if inner.peek(syn::token::Paren) {
                            let expr_content;
                            syn::parenthesized!(expr_content in inner);
                            expr_content.parse::<Expr>()?
                        } else {
                            let path: Path = inner.parse()?;
                            syn::parse_quote!(#path)
                        };
                        let layout = if inner.peek(syn::token::Brace) {
                            let layout_content;
                            braced!(layout_content in inner);
                            let mut is_embedded = false;
                            let mut ptr_offset = None;
                            let mut length_offset = None;
                            let mut items_offset = None;
                            while !layout_content.is_empty() {
                                let layout_key: Ident = layout_content.parse()?;
                                match layout_key.to_string().as_str() {
                                    "embedded" => {
                                        is_embedded = true;
                                    }
                                    "ptr_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        ptr_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    "length_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        length_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    "items_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        items_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    other => {
                                        return Err(syn::Error::new(
                                            layout_key.span(),
                                            format!(
                                                "unknown virtualizable array layout key: `{other}`"
                                            ),
                                        ));
                                    }
                                }
                                let _ = layout_content.parse::<Token![,]>();
                            }
                            if is_embedded || ptr_offset.is_some() {
                                VableArrayLayoutDecl::Embedded {
                                    field_offset,
                                    ptr_offset: ptr_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `ptr_offset` in embedded array layout",
                                        )
                                    })?,
                                    length_offset: length_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `length_offset` in embedded array layout",
                                        )
                                    })?,
                                    items_offset: items_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `items_offset` in embedded array layout",
                                        )
                                    })?,
                                }
                            } else {
                                VableArrayLayoutDecl::Direct {
                                    field_offset,
                                    length_offset,
                                    items_offset,
                                }
                            }
                        } else {
                            VableArrayLayoutDecl::Direct {
                                field_offset,
                                length_offset: None,
                                items_offset: None,
                            }
                        };
                        arrays.push(VableArrayDecl {
                            name,
                            item_type,
                            layout,
                        });
                        let _ = inner.parse::<Token![,]>();
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown virtualizable parameter: `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }

        let state_type = state_type.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `state` in virtualizable! macro")
        })?;
        let var_name = var_name.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `name` in virtualizable! macro")
        })?;
        let token_offset = token_offset.ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "missing `token_offset` in virtualizable! macro",
            )
        })?;
        let heap_ptr_expr = heap_ptr_expr.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `heap_ptr` in virtualizable! macro")
        })?;

        Ok(VirtualizableMacroInput {
            state_type,
            decl: VirtualizableDecl {
                var_name,
                token_offset,
                fields,
                arrays,
            },
            heap_ptr_expr,
            frame_field,
            inputargs,
            extra_reds,
            array_item_type,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// Code generation
// ═══════════════════════════════════════════════════════════════

fn ir_type_token(ident: &Ident) -> TokenStream {
    match ident.to_string().as_str() {
        "Ref" | "ref" => quote! { majit_ir::Type::Ref },
        "Float" | "float" => quote! { majit_ir::Type::Float },
        _ => quote! { majit_ir::Type::Int },
    }
}

fn ir_value_ctor(ir_type: &Ident, expr: TokenStream) -> TokenStream {
    match ir_type.to_string().as_str() {
        "Ref" | "ref" => quote! { majit_ir::Value::Ref(majit_ir::GcRef(#expr as usize)) },
        "Float" | "float" => quote! { majit_ir::Value::Float(f64::from_bits(#expr as u64)) },
        _ => quote! { majit_ir::Value::Int(#expr as i64) },
    }
}

pub(crate) fn expand(input: VirtualizableMacroInput) -> TokenStream {
    let state_type = &input.state_type;
    let decl = &input.decl;
    let heap_ptr_expr = &input.heap_ptr_expr;
    let vable_name = decl.var_name.to_string();

    // 1. VirtualizableInfo builder
    let info_fn = codegen::generate_vable_info_pub_fn(decl);

    // 2. Field/array spec constants
    let specs = codegen::generate_vable_specs(decl);

    // 3. Virtualizable state hooks (heap_ptr, sync, etc.)
    let hooks = generate_standalone_hooks(state_type, &vable_name, heap_ptr_expr);

    // 4. Layout-based JitState helpers (extract, collect, live_types, etc.)
    let layout_helpers = if !input.inputargs.is_empty() || !input.extra_reds.is_empty() {
        generate_layout_helpers(
            state_type,
            &input.frame_field,
            &input.inputargs,
            &input.extra_reds,
            &input.array_item_type,
        )
    } else {
        quote! {}
    };

    quote! {
        #info_fn
        #specs
        #hooks
        #layout_helpers
    }
}

/// Generate layout-based JitState helper functions.
///
/// Layout: [frame:Ref, extra_red0:U0, ..., inputarg0:T0, inputarg1:T1, ...,
///          locals[]:ItemT, stack[]:ItemT]
///
/// Mirrors `pyjitpl.py:2957 redboxes` (= JitDriver.reds, frame first
/// then extra reds) followed by `:2964 + virtualizable_boxes` (vable
/// static fields then array items).
///
/// `extra_reds` and `inputargs` are concatenated for codegen — both
/// contribute scalar params, SYM_*_IDX consts, restore calls, etc. The
/// split exists at the macro boundary to mirror RPython's
/// `_virtualizable_` (vable static fields, read via getfield_vable) vs
/// `JitDriver.reds` minus the vable (e.g. `ec`, plain host-side scalar)
/// distinction; the generated code treats both uniformly.
fn generate_layout_helpers(
    state_type: &Ident,
    frame_field: &Option<Ident>,
    inputargs: &[InputArgField],
    extra_reds: &[InputArgField],
    array_item_type: &Option<Ident>,
) -> TokenStream {
    let frame_ident = frame_field
        .as_ref()
        .map_or_else(|| format_ident!("frame"), |f| f.clone());
    let frame_setter = format_ident!("set_{}", frame_ident);
    let arr_type = array_item_type
        .as_ref()
        .map_or_else(|| format_ident!("Ref"), |t| t.clone());
    let arr_type_token = ir_type_token(&arr_type);
    // Red block (extra reds) first, then vable static fields — mirrors
    // pyjitpl.py `live_arg_boxes = greenboxes + redboxes` followed by
    // pyjitpl.py:2964 `+ virtualizable_boxes`. JitDriver.reds (e.g.
    // `interp_jit.py reds = ['frame', 'ec']`) come before the
    // virtualizable's static fields in the OpRef stream.
    let all_scalars: Vec<&InputArgField> = extra_reds.iter().chain(inputargs.iter()).collect();
    let num_scalars = 1 + all_scalars.len(); // frame + extra_red + vable static fields
    let num_extra_reds = extra_reds.len(); // non-vable JitDriver red inputargs
    let num_vable_scalars = inputargs.len(); // vable static fields only (excludes frame, extra_reds)

    // ── extract_live_values ──
    // Parameters: state field values + closures for array access
    let scalar_params: Vec<TokenStream> = all_scalars
        .iter()
        .map(|f| {
            let name = &f.name;
            quote! { #name: usize }
        })
        .collect();
    let scalar_value_pushes: Vec<TokenStream> = all_scalars
        .iter()
        .map(|f| {
            let name = &f.name;
            let vc = ir_value_ctor(&f.ir_type, quote! { #name });
            quote! { __vals.push(#vc); }
        })
        .collect();
    let frame_value = ir_value_ctor(&format_ident!("Ref"), quote! { __frame });
    let local_value = ir_value_ctor(&arr_type, quote! { local_at(__i) });
    let stack_value = ir_value_ctor(&arr_type, quote! { stack_at(__i) });

    // ── live_value_types ──
    let scalar_type_pushes: Vec<TokenStream> = {
        let mut v = vec![quote! { __types.push(majit_ir::Type::Ref); }]; // frame
        for f in &all_scalars {
            let tp = ir_type_token(&f.ir_type);
            v.push(quote! { __types.push(#tp); });
        }
        v
    };

    // ── OpRef index constants for create_sym ──
    let sym_idx_consts: Vec<TokenStream> = all_scalars
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let const_name = format_ident!("SYM_{}_IDX", f.name.to_string().to_uppercase());
            let idx = (i + 1) as u32; // frame is 0
            quote! { pub const #const_name: u32 = #idx; }
        })
        .collect();
    let array_base_val = num_scalars as u32;

    // ── is_compatible helper ──
    let compat_params: Vec<TokenStream> = all_scalars
        .iter()
        .map(|f| {
            let state_name = format_ident!("state_{}", f.name);
            let meta_name = format_ident!("meta_{}", f.name);
            quote! { #state_name: usize, #meta_name: usize }
        })
        .collect();
    let compat_checks: Vec<TokenStream> = all_scalars
        .iter()
        .map(|f| {
            let state_name = format_ident!("state_{}", f.name);
            let meta_name = format_ident!("meta_{}", f.name);
            quote! { && #state_name == #meta_name }
        })
        .collect();

    // ── restore_values helper ──
    // Use set_ accessors to write through to heap / host-side state. Vable
    // inputargs route to heap (source of truth, virtualizable.py write_boxes);
    // extra reds route to plain host-side fields on the state struct.
    let restore_scalars: Vec<TokenStream> = all_scalars
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let setter = format_ident!("set_{}", f.name);
            let idx = i + 1; // frame is values[0]
            let tp = &f.ir_type;
            let conv = match tp.to_string().as_str() {
                "Ref" | "ref" => quote! { __value_to_usize(&values[#idx]) },
                "Float" | "float" => {
                    quote! { f64::from_bits(__value_to_usize(&values[#idx]) as u64) as usize }
                }
                _ => quote! { __value_to_usize(&values[#idx]) },
            };
            quote! {
                state.#setter(#conv);
            }
        })
        .collect();
    let restore_scalars_raw: Vec<TokenStream> = all_scalars
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let setter = format_ident!("set_{}", f.name);
            let idx = i + 1;
            quote! {
                state.#setter(raw[#idx] as usize);
            }
        })
        .collect();

    quote! {
        // ── Layout constants ──

        /// OpRef index for the frame pointer in the symbolic state.
        pub const SYM_FRAME_IDX: u32 = 0;
        #(#sym_idx_consts)*
        /// OpRef base index for array slots (locals + stack).
        pub const SYM_ARRAY_BASE: u32 = #array_base_val;
        /// Number of scalar inputargs (frame + declared fields).
        ///
        /// TODO: codegen-time constant equivalent to
        /// `len(VABLEINFO.static_field_descrs) + 1` (frame ptr + N
        /// `_virtualizable_` scalars from `interp_jit.py`). RPython
        /// derives the count dynamically by iterating
        /// `range(len(self.static_field_descrs))` (`virtualizable.py read_boxes`);
        /// pyre crystallises it at proc-macro expansion time so the flat
        /// index arithmetic in trace/state code stays monomorphic.
        ///
        /// **Includes `extra_reds`**: when a JitDriver wires non-vable red
        /// inputargs (e.g. `ec` from `interp_jit.py:67 reds = ['frame', 'ec']`),
        /// they extend this count. Sites that count vable static fields only
        /// (independent of red wiring) must use `NUM_VABLE_SCALARS` instead.
        pub const NUM_SCALAR_INPUTARGS: usize = #num_scalars;

        /// Number of vable static fields (`_virtualizable_` declarations).
        ///
        /// `virtualizable.py read_boxes unroll_static_fields` length, excluding the
        /// frame pointer and any `extra_reds` red inputargs. Stable across
        /// JitDriver red-wiring changes; use this when iterating
        /// `VABLEINFO.static_field_descrs` or computing offsets that pertain
        /// to vable static fields specifically.
        pub const NUM_VABLE_SCALARS: usize = #num_vable_scalars;

        /// Number of non-vable JitDriver red inputargs (e.g. `ec`).
        ///
        /// `interp_jit.py reds = ['frame', 'ec']` declares these alongside
        /// `frame`; they live between the frame pointer and the vable static
        /// fields in the OpRef stream (`pyjitpl.py:2957 redboxes`).
        pub const NUM_EXTRA_REDS: usize = #num_extra_reds;

        /// First OpRef index of the vable static field block.
        ///
        /// Equals `1 + NUM_EXTRA_REDS` — the frame pointer occupies OpRef::from_raw(0),
        /// extra reds occupy OpRef::from_raw(1..1+NUM_EXTRA_REDS), and vable static
        /// fields begin at this index. Use when seeding `virtualizable_boxes`
        /// with vable scalar OpRefs (the start point shifts when extra reds
        /// wire in).
        pub const FIRST_VABLE_SCALAR_IDX: u32 = (1 + #num_extra_reds) as u32;

        // ── extract_live_values ──

        /// Extract live values from state in virtualizable layout order.
        ///
        /// virtualizable.py read_boxes / get_list_of_active_boxes:
        ///   for _, fieldname in unroll_static_fields:
        ///       boxes.append(wrap(cpu, getattr(virtualizable, fieldname)))
        ///   for _, fieldname in unroll_array_fields:
        ///       lst = getattr(virtualizable, fieldname)
        ///       for j in range(len(lst)):
        ///           boxes.append(wrap(cpu, lst[j]))
        ///
        /// Layout: `[frame:Ref, scalars..., array[0..len(lst)]:ItemT]`
        ///
        /// `__array_capacity` is `len(lst)` — the concrete
        /// FixedObjectArray length of the virtualizable's backing
        /// storage, i.e. `nlocals + co_stacksize`. This is the full
        /// `read_boxes()` iteration extent regardless of the live
        /// Python-level stack depth at the merge point.
        ///
        /// `local_at(i)` returns the raw value of array slot `i` (direct index).
        pub fn virt_extract_live_values(
            __frame: usize,
            #(#scalar_params,)*
            __num_locals: usize,
            __array_capacity: usize,
            local_at: impl Fn(usize) -> usize,
            stack_at: impl Fn(usize) -> usize,
        ) -> Vec<majit_ir::Value> {
            let stack_only = __array_capacity.saturating_sub(__num_locals);
            let mut __vals = Vec::with_capacity(#num_scalars + __num_locals + stack_only);
            __vals.push(#frame_value);
            #(#scalar_value_pushes)*
            for __i in 0..__num_locals {
                __vals.push(#local_value);
            }
            for __i in 0..stack_only {
                __vals.push(#stack_value);
            }
            __vals
        }

        // ── live_value_types ──

        /// Return the IR type of each live value in layout order.
        pub fn virt_live_value_types(num_slots: usize) -> Vec<majit_ir::Type> {
            let mut __types = Vec::with_capacity(#num_scalars + num_slots);
            #(#scalar_type_pushes)*
            __types.extend(std::iter::repeat_n(#arr_type_token, num_slots));
            __types
        }

        // ── is_compatible ──

        /// Check if state scalar fields match meta (green key + shape check).
        pub fn virt_is_compatible(
            state_nlocals: usize, meta_nlocals: usize,
            #(#compat_params,)*
        ) -> bool {
            state_nlocals == meta_nlocals
            #(#compat_checks)*
        }

        // ── restore helpers ──

        fn __value_to_usize(v: &majit_ir::Value) -> usize {
            match v {
                majit_ir::Value::Int(i) => *i as usize,
                majit_ir::Value::Ref(r) => r.as_usize(),
                majit_ir::Value::Float(f) => f.to_bits() as usize,
                majit_ir::Value::Void => 0,
            }
        }

        /// Restore scalar fields from a Value slice (virtualizable layout).
        ///
        /// Writes the frame identity through `state.set_frame`, then
        /// `state.<inputarg>...` from `values[0..]`.
        /// Returns the index of the first array slot in `values`.
        pub fn virt_restore_scalars(
            state: &mut #state_type,
            values: &[majit_ir::Value],
        ) -> usize {
            if values.is_empty() {
                return 0;
            }
            state.#frame_setter(__value_to_usize(&values[0]));
            #(#restore_scalars)*
            #num_scalars
        }

        /// Restore scalar fields from a raw i64 slice (virtualizable layout).
        ///
        /// Writes the frame identity through `state.set_frame`, then
        /// `state.<inputarg>...` from `raw[0..]`.
        /// Returns the index of the first array slot in `raw`.
        ///
        /// virtualizable.py write_from_resume_data_partial parity: all entries must be present.
        pub fn virt_restore_scalars_raw(
            state: &mut #state_type,
            raw: &[i64],
        ) -> usize {
            assert!(
                raw.len() >= #num_scalars,
                "virt_restore_scalars_raw: raw.len()={} < num_scalars={}",
                raw.len(), #num_scalars,
            );
            state.#frame_setter(raw[0] as usize);
            #(#restore_scalars_raw)*
            #num_scalars
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Standalone hooks (heap_ptr, sync, etc.) — unchanged from before
// ═══════════════════════════════════════════════════════════════

fn generate_standalone_hooks(
    state_type: &Ident,
    vable_name: &str,
    heap_ptr_closure: &Expr,
) -> TokenStream {
    quote! {
        fn __heap_ptr(state: &#state_type) -> Option<*mut u8> {
            let ptr = (#heap_ptr_closure)(state);
            if ptr.is_null() { None } else { Some(ptr) }
        }

        pub fn virt_heap_ptr(
            state: &#state_type,
            _virtualizable: &str,
        ) -> Option<*mut u8> {
            if _virtualizable != #vable_name {
                return None;
            }
            __heap_ptr(state)
        }

        pub fn virt_array_lengths(
            state: &#state_type,
            _virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<Vec<usize>> {
            if _virtualizable != #vable_name {
                return None;
            }
            let obj_ptr = __heap_ptr(state)?;
            if info.can_read_all_array_lengths_from_heap() {
                Some(unsafe { info.read_array_lengths_from_heap(obj_ptr.cast_const()) })
            } else {
                None
            }
        }

        pub fn virt_export_boxes(
            state: &#state_type,
            _virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<(Vec<i64>, Vec<usize>)> {
            if _virtualizable != #vable_name {
                return None;
            }
            let obj_ptr = __heap_ptr(state)?.cast_const();
            let lengths = if info.can_read_all_array_lengths_from_heap() {
                unsafe { info.read_array_lengths_from_heap(obj_ptr) }
            } else {
                virt_array_lengths(state, _virtualizable, info)?
            };
            let boxes = unsafe { info.read_boxes(obj_ptr, &lengths) };
            Some((boxes, lengths))
        }

    }
}

/// Parse and expand the `virtualizable!` macro input.
pub fn parse_and_expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let config = syn::parse_macro_input!(input as VirtualizableMacroInput);
    expand(config).into()
}
