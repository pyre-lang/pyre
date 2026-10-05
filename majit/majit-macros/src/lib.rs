/// Proc macros for the majit JIT framework.
///
/// rpython/rlib/jit.py decorator equivalents:
/// - #[elidable]: rlib/jit.py — Mark a function as pure (constant-foldable)
/// - #[elidable_promote]: rlib/jit.py — Elidable + auto-promote args
/// - #[dont_look_inside]: rlib/jit.py — Prevent tracing into a function
/// - #[unroll_safe]: rlib/jit.py — Safe to unroll loops
/// - #[loop_invariant]: rlib/jit.py — Loop-invariant function
/// - #[not_in_trace]: rlib/jit.py — Disappears from final assembler
/// - #[always_inline]: objectmodel.py — Backend optimizer must inline
/// - #[not_rpython]: objectmodel.py — Reject from RPython flow graphs
///
/// majit-specific extensions:
/// - #[jit_driver]: Annotate an interpreter's main dispatch loop
/// - #[jit_interp]: Auto-generate trace_instruction and JitState from dispatch
/// - #[jit_inline]: Serialize a helper into a hidden sub-JitCode
/// - #[jit_may_force]: Mark a helper as a may-force call surface
/// - #[jit_loop_invariant]: Alias for #[loop_invariant]
/// - #[jit_module]: Module-level automatic helper discovery
/// - virtualizable!: Standalone virtualizable field declaration
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, Ident, ItemFn, Path, ReturnType, Token, Type, parenthesized, parse::Parse,
    parse::ParseStream, parse_macro_input,
};

mod jit_interp;
mod jit_struct;
mod rffi_expand;
mod virtualizable;

/// `llexternal` (`rffi.py`): funcptr, `ccall_<name>`, and the forwarding wrapper.
#[proc_macro]
pub fn llexternal(input: TokenStream) -> TokenStream {
    match rffi_expand::expand_llexternal(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// `ExternalCompilationInfo` (`cbuild.py`) plus the link directives genc hands
/// to the linker. Non-empty `library_dirs`, `link_extra`,
/// `separate_module_sources`, `separate_module_files`, and `compile_extra`
/// are a compile error.
#[proc_macro]
pub fn external_compilation_info(input: TokenStream) -> TokenStream {
    match rffi_expand::expand_external_compilation_info(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Marker for `call_external_function._call_aroundstate_target_ = funcptr, save_err`.
/// Emits the sibling const `_call_aroundstate_target_<fn>`.
#[proc_macro_attribute]
pub fn call_aroundstate_target(attr: TokenStream, item: TokenStream) -> TokenStream {
    rffi_expand::expand_call_aroundstate_target(attr.into(), item.into()).into()
}

/// `_gctransformer_hint_close_stack_` (`llexternal`'s `call_external_function`).
/// Emits the sibling const `_gctransformer_hint_close_stack_<fn>`.
#[proc_macro_attribute]
pub fn jit_close_stack(_attr: TokenStream, item: TokenStream) -> TokenStream {
    rffi_expand::expand_jit_close_stack(item.into()).into()
}

fn gate_generated_items(
    tokens: proc_macro2::TokenStream,
    condition: &syn::Meta,
) -> proc_macro2::TokenStream {
    let mut file: syn::File = syn::parse2(tokens).expect("generated JIT items");
    for item in &mut file.items {
        let attrs = match item {
            syn::Item::Fn(item) => &mut item.attrs,
            syn::Item::Static(item) => &mut item.attrs,
            syn::Item::Const(item) => &mut item.attrs,
            syn::Item::Impl(item) => &mut item.attrs,
            syn::Item::Struct(item) => &mut item.attrs,
            syn::Item::Enum(item) => &mut item.attrs,
            syn::Item::Type(item) => &mut item.attrs,
            syn::Item::Use(item) => &mut item.attrs,
            syn::Item::Macro(item) => &mut item.attrs,
            syn::Item::Mod(item) => &mut item.attrs,
            _ => unreachable!("unexpected generated JIT item"),
        };
        attrs.push(syn::parse_quote!(#[cfg(#condition)]));
    }
    quote!(#file)
}

struct JitInlineArgs {
    trace_cfg: Option<syn::Meta>,
    calls: Vec<jit_interp::CallEntry>,
    ref_params: Vec<(Ident, Path)>,
    ref_fields: Vec<jit_interp::RefFieldEntry>,
    array_fields: Vec<jit_interp::ArrayFieldEntry>,
    int_fields: Vec<jit_interp::IntFieldEntry>,
    native_int_binops: Vec<(Path, Ident)>,
    native_tag_small: Vec<Path>,
    native_identity: Vec<Path>,
    struct_allocs: Vec<(Path, Path)>,
    headerless_structs: Vec<Path>,
    inlined_prefix: Vec<jit_interp::InlinedPrefixEntry>,
}

impl Parse for JitInlineArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut trace_cfg = None;
        let mut calls: Vec<jit_interp::CallEntry> = Vec::new();
        let mut ref_params: Vec<(Ident, Path)> = Vec::new();
        let mut ref_fields: Vec<jit_interp::RefFieldEntry> = Vec::new();
        let mut int_fields: Vec<jit_interp::IntFieldEntry> = Vec::new();
        let mut native_int_binops: Vec<(Path, Ident)> = Vec::new();
        let mut native_tag_small: Vec<Path> = Vec::new();
        let mut native_identity: Vec<Path> = Vec::new();
        let mut struct_allocs: Vec<(Path, Path)> = Vec::new();
        let mut headerless_structs: Vec<Path> = Vec::new();
        let mut inlined_prefix: Vec<jit_interp::InlinedPrefixEntry> = Vec::new();
        let mut array_fields: Vec<jit_interp::ArrayFieldEntry> = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "trace_cfg" => {
                    let content;
                    parenthesized!(content in input);
                    trace_cfg = Some(content.parse()?);
                }
                "calls" => {
                    let content;
                    syn::braced!(content in input);
                    while !content.is_empty() {
                        let func: Path = content.parse()?;
                        let policy = if content.peek(Token![=>]) {
                            content.parse::<Token![=>]>()?;
                            let kind: Ident = content.parse()?;
                            Some(jit_interp::parse_call_policy_kind(&kind).ok_or_else(|| {
                                syn::Error::new(
                                    kind.span(),
                                    "#[jit_inline(calls = { ... })] supports residual/may_force/release_gil/loopinvariant call policies for void/int and wrapped int/ref/float helpers, plus inline_int/inline_ref/inline_float",
                                )
                            })?)
                        } else {
                            None
                        };
                        calls.push(jit_interp::CallEntry { path: func, policy });
                        let _ = content.parse::<Token![,]>();
                    }
                }
                "ref_params" => {
                    let content;
                    syn::braced!(content in input);
                    while !content.is_empty() {
                        let name: Ident = content.parse()?;
                        content.parse::<Token![:]>()?;
                        content.parse::<Token![ref]>()?;
                        let inner;
                        parenthesized!(inner in content);
                        let struct_type: Path = inner.parse()?;
                        ref_params.push((name, struct_type));
                        let _ = content.parse::<Token![,]>();
                    }
                }
                "ref_fields" => {
                    ref_fields = jit_interp::parse_ref_fields_map(input)?;
                }
                "array_fields" => {
                    array_fields = jit_interp::parse_array_fields_map(input)?;
                }
                "int_fields" => {
                    int_fields = jit_interp::parse_int_fields_map(input)?;
                }
                "helpers" => {
                    let content;
                    syn::bracketed!(content in input);
                    let paths: syn::punctuated::Punctuated<Path, Token![,]> =
                        content.parse_terminated(Path::parse, Token![,])?;
                    calls.extend(paths.into_iter().map(|p| jit_interp::CallEntry {
                        path: p,
                        policy: None,
                    }));
                }
                "native_int_binops" => {
                    native_int_binops = jit_interp::parse_native_int_binops_map(input)?;
                }
                "native_tag_small" => {
                    native_tag_small = jit_interp::parse_native_tag_small_list(input)?;
                }
                "native_identity" => {
                    native_identity = jit_interp::parse_path_set(input)?;
                }
                "struct_allocs" => {
                    struct_allocs = jit_interp::parse_call_returns_map(input)?;
                }
                "headerless_structs" => {
                    headerless_structs = jit_interp::parse_path_set(input)?;
                }
                "inlined_prefix" => {
                    inlined_prefix = jit_interp::parse_inlined_prefix_map(input)?;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_inline parameter: `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }
        jit_interp::reject_overlapping_call_vocabularies(
            &calls,
            &native_int_binops,
            &native_tag_small,
            &native_identity,
        )?;

        Ok(Self {
            trace_cfg,
            calls,
            ref_params,
            ref_fields,
            array_fields,
            int_fields,
            native_int_binops,
            native_tag_small,
            native_identity,
            struct_allocs,
            headerless_structs,
            inlined_prefix,
        })
    }
}

fn rewrite_jit_inline_ref_param_fields(
    block: &syn::Block,
    ref_params: &[(Ident, Path)],
    ref_fields: &[jit_interp::RefFieldEntry],
    array_fields: &[jit_interp::ArrayFieldEntry],
    int_fields: &[jit_interp::IntFieldEntry],
    struct_allocs: &[(Path, Path)],
    inlined_prefix: &[jit_interp::InlinedPrefixEntry],
) -> syn::Block {
    use std::collections::HashMap;
    use syn::visit_mut::VisitMut;

    struct InlineRefFieldRewriter {
        local_ref_types: HashMap<String, syn::Path>,
        field_pointees: HashMap<String, syn::Path>,
        // `array_fields` entries: "StructLast::field" -> element type. The
        // field holds the buffer BASE POINTER, so an indexed access derefs
        // through it rather than reading the field itself.
        array_field_elems: HashMap<String, syn::Path>,
        // `ElementType in Header`: element 0 is `Header.items`, not `*field`.
        array_headers: HashMap<String, syn::Path>,
        struct_allocs: HashMap<Vec<String>, syn::Path>,
        // Outer struct segments -> (base field, base path); empty unless a
        // caller declares a leading substructure.
        inlined_prefix: HashMap<Vec<String>, (String, syn::Path)>,
        // Every `"StructLast::field"` this block's vocabularies declare.
        declared_field_keys: std::collections::HashSet<String>,
    }

    impl InlineRefFieldRewriter {
        fn local_ref_struct_of_base(&self, expr: &syn::Expr) -> Option<syn::Path> {
            let syn::Expr::Path(path) = expr else {
                return None;
            };
            let ident = path.path.get_ident()?;
            self.local_ref_types.get(&ident.to_string()).cloned()
        }

        fn field_pointee(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let struct_last = struct_path.segments.last()?.ident.to_string();
            let key = format!("{}::{}", struct_last, field_name);
            self.field_pointees.get(&key).cloned()
        }

        // The struct that declares `field_name`, so the emitted cast names
        // the type it really lives on.  The JIT lowerer redirects the same
        // way; a disagreement leaves this walk dereferencing a struct with
        // no such field.
        fn declaring_struct(&self, struct_path: &syn::Path, field_name: &str) -> syn::Path {
            jit_interp::declaring_struct_of(
                &self.inlined_prefix,
                |path, field| {
                    path.segments.last().is_some_and(|last| {
                        self.declared_field_keys
                            .contains(&format!("{}::{}", last.ident, field))
                    })
                },
                struct_path,
                field_name,
            )
        }

        fn array_field_key(struct_path: &syn::Path, field_name: &str) -> Option<String> {
            let struct_last = struct_path.segments.last()?.ident.to_string();
            Some(format!("{}::{}", struct_last, field_name))
        }

        fn array_field_elem(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let key = Self::array_field_key(struct_path, field_name)?;
            self.array_field_elems.get(&key).cloned()
        }

        fn array_header(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let key = Self::array_field_key(struct_path, field_name)?;
            self.array_headers.get(&key).cloned()
        }

        fn record_ref_field_local(&mut self, local: &syn::Local) {
            let Some(init) = &local.init else {
                return;
            };
            let syn::Expr::Field(field) = &*init.expr else {
                return;
            };
            let Some(struct_path) = self.local_ref_struct_of_base(&field.base) else {
                return;
            };
            let syn::Member::Named(member) = &field.member else {
                return;
            };
            let struct_path = self.declaring_struct(&struct_path, &member.to_string());
            let Some(pointee) = self.field_pointee(&struct_path, &member.to_string()) else {
                return;
            };
            if let syn::Pat::Ident(pat_ident) = &local.pat {
                self.local_ref_types
                    .insert(pat_ident.ident.to_string(), pointee);
            }
        }

        // `let col = p as *mut Struct` keeps the pointee the array rewrite
        // indexes. `jtransform.py` `rewrite_op_cast_pointer` is `same_as`;
        // the concrete body still has to name the struct to dereference it.
        fn record_pointer_cast_local(&mut self, local: &syn::Local) {
            let Some(init) = &local.init else {
                return;
            };
            let syn::Expr::Cast(cast) = &*init.expr else {
                return;
            };
            let syn::Type::Ptr(ptr) = &*cast.ty else {
                return;
            };
            let syn::Type::Path(path) = ptr.elem.as_ref() else {
                return;
            };
            let Some(last) = path.path.segments.last() else {
                return;
            };
            if !last
                .ident
                .to_string()
                .starts_with(|c: char| c.is_ascii_uppercase())
            {
                return;
            }
            let syn::Pat::Ident(pat_ident) = &local.pat else {
                return;
            };
            self.local_ref_types
                .insert(pat_ident.ident.to_string(), path.path.clone());
        }
    }

    impl VisitMut for InlineRefFieldRewriter {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // Rewrite struct literal inits on the concrete path:
            // `let x = StructType { f0: v0, f1: v1 }` where StructType is
            // in `struct_allocs` → `let x = allocator_func(v0, v1)`. This
            // must run before `record_ref_field_local`/the default visitor
            // descend. The allocator returns `*mut Struct`, so `x` is that
            // pointer: record the pointee or a later `x.field` / `x.field[i]`
            // cannot see the layout.
            if let syn::Stmt::Local(local) = stmt
                && let Some(init) = &mut local.init
                && let syn::Expr::Struct(s) = &*init.expr
            {
                let segs: Vec<String> = s
                    .path
                    .segments
                    .iter()
                    .map(|seg| seg.ident.to_string())
                    .collect();
                if let Some(alloc_func) = self.struct_allocs.get(&segs).cloned() {
                    let field_args: Vec<syn::Expr> =
                        s.fields.iter().map(|f| f.expr.clone()).collect();
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        self.local_ref_types
                            .insert(pat_ident.ident.to_string(), s.path.clone());
                    }
                    *init.expr = syn::parse_quote! {
                        #alloc_func(#(#field_args),*)
                    };
                }
            }
            if let syn::Stmt::Local(local) = stmt {
                self.record_ref_field_local(local);
                self.record_pointer_cast_local(local);
            }
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }

        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            // Nested `struct_allocs` literals (`items: CelItemsBlock { .. }`
            // inside `W_ListObject { .. }`) rewrite to the concrete
            // allocator too. Visit fields first so an inner literal is
            // already a call when it becomes an argument.
            if let syn::Expr::Struct(struct_expr) = expr {
                let segs: Vec<String> = struct_expr
                    .path
                    .segments
                    .iter()
                    .map(|seg| seg.ident.to_string())
                    .collect();
                let alloc_func = self.struct_allocs.get(&segs).cloned();
                syn::visit_mut::visit_expr_mut(self, expr);
                if let Some(alloc_func) = alloc_func
                    && let syn::Expr::Struct(struct_expr) = expr
                {
                    let field_args: Vec<syn::Expr> = struct_expr
                        .fields
                        .iter()
                        .map(|field| field.expr.clone())
                        .collect();
                    *expr = syn::parse_quote! {
                        #alloc_func(#(#field_args),*)
                    };
                }
                return;
            }
            // Array element WRITE: `<base>.<array_field>[<idx>] = <rhs>`.
            // Must precede the plain-field arms: the field itself holds the
            // buffer BASE POINTER, so letting the default visitor rewrite
            // `<base>.<array_field>` on its own would leave a raw pointer
            // being indexed with `[]`, which does not compile.
            if let syn::Expr::Assign(assign) = expr
                && let syn::Expr::Index(index_expr) = &*assign.left
                && let syn::Expr::Field(field) = &*index_expr.expr
                && let Some(binding_path) = self.local_ref_struct_of_base(&field.base)
                && let syn::Member::Named(member_id) = &field.member
                // Resolve against the struct that DECLARES the field, the way
                // `match_array_field_base` does on the JIT side. Looking the
                // element type up under the binding's own spelling misses a
                // field an embedded base declares, and the plain-field arm
                // below then rewrites `<base>.<field>` to the buffer pointer
                // and leaves the `[]` on it, which does not compile.
                && let struct_path = self.declaring_struct(&binding_path, &member_id.to_string())
                && self
                    .array_field_elem(&struct_path, &member_id.to_string())
                    .is_some()
            {
                let base = (*field.base).clone();
                let member = field.member.clone();
                let member_name = member_id.to_string();
                let header = self.array_header(&struct_path, &member_name);
                let element = self
                    .array_field_elem(&struct_path, &member_name)
                    .expect("array field element");
                let mut idx = (*index_expr.index).clone();
                let mut rhs = (*assign.right).clone();
                self.visit_expr_mut(&mut idx);
                self.visit_expr_mut(&mut rhs);
                // The assigned value is evaluated before the assignee place,
                // so an index or RHS with side effects must not be reordered:
                // `base.data[next_index()] = compute_value()` runs
                // `compute_value()` first. A header array's field addresses
                // the block, and element 0 is `header.items` — the same
                // concrete shape `jit_interp` emits for `ElementType in Header`.
                *expr = if let Some(header) = header {
                    syn::parse_quote! {
                        {
                            let __majit_arr_val = #rhs;
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                let __majit_arr_block = core::mem::transmute::<
                                    _,
                                    *mut #header,
                                >(
                                    (*(__majit_arr_obj as *const #struct_path)).#member,
                                );
                                let __majit_arr_items = (__majit_arr_block as *mut u8).add(
                                    core::mem::offset_of!(#header, items),
                                ) as *mut #element;
                                *__majit_arr_items.add(__majit_arr_idx as usize) = __majit_arr_val;
                            }
                        }
                    }
                } else {
                    syn::parse_quote! {
                        {
                            let __majit_arr_val = #rhs;
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                *((*(__majit_arr_obj as *mut #struct_path))
                                    .#member
                                    .add(__majit_arr_idx as usize)) = __majit_arr_val;
                            }
                        }
                    }
                };
                return;
            }

            // Array element READ: `<base>.<array_field>[<idx>]`.
            if let syn::Expr::Index(index_expr) = expr
                && let syn::Expr::Field(field) = &*index_expr.expr
                && let Some(binding_path) = self.local_ref_struct_of_base(&field.base)
                && let syn::Member::Named(member_id) = &field.member
                && let struct_path = self.declaring_struct(&binding_path, &member_id.to_string())
                && self
                    .array_field_elem(&struct_path, &member_id.to_string())
                    .is_some()
            {
                let base = (*field.base).clone();
                let member = field.member.clone();
                let member_name = member_id.to_string();
                let header = self.array_header(&struct_path, &member_name);
                let element = self
                    .array_field_elem(&struct_path, &member_name)
                    .expect("array field element");
                let mut idx = (*index_expr.index).clone();
                self.visit_expr_mut(&mut idx);
                *expr = if let Some(header) = header {
                    syn::parse_quote! {
                        {
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                let __majit_arr_block = core::mem::transmute::<
                                    _,
                                    *mut #header,
                                >(
                                    (*(__majit_arr_obj as *const #struct_path)).#member,
                                );
                                let __majit_arr_items = (__majit_arr_block as *mut u8).add(
                                    core::mem::offset_of!(#header, items),
                                ) as *mut #element;
                                *__majit_arr_items.add(__majit_arr_idx as usize)
                            }
                        }
                    }
                } else {
                    syn::parse_quote! {
                        {
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                *((*(__majit_arr_obj as *const #struct_path))
                                    .#member
                                    .add(__majit_arr_idx as usize))
                            }
                        }
                    }
                };
                return;
            }

            if let syn::Expr::Assign(assign) = expr
                && let syn::Expr::Field(lhs) = &*assign.left
                && let Some(struct_path) = self.local_ref_struct_of_base(&lhs.base)
            {
                let base = (*lhs.base).clone();
                let member = lhs.member.clone();
                let member_name = match &lhs.member {
                    syn::Member::Named(id) => id.to_string(),
                    _ => String::new(),
                };
                let struct_path = self.declaring_struct(&struct_path, &member_name);
                let mut rhs = (*assign.right).clone();
                self.visit_expr_mut(&mut rhs);
                if let Some(pointee) = self.field_pointee(&struct_path, &member_name) {
                    *expr = syn::parse_quote! {
                        unsafe { (*(#base as *mut #struct_path)).#member = #rhs as *mut #pointee }
                    };
                } else {
                    *expr = syn::parse_quote! {
                        unsafe { (*(#base as *mut #struct_path)).#member = #rhs }
                    };
                }
                return;
            }

            if let syn::Expr::Field(field) = expr
                && let Some(struct_path) = self.local_ref_struct_of_base(&field.base)
            {
                let base = (*field.base).clone();
                let member = field.member.clone();
                let member_name = match &field.member {
                    syn::Member::Named(id) => id.to_string(),
                    _ => String::new(),
                };
                let struct_path = self.declaring_struct(&struct_path, &member_name);
                if self.field_pointee(&struct_path, &member_name).is_some() {
                    *expr = syn::parse_quote! {
                        {
                            let __majit_getfield_obj = #base;
                            unsafe {
                                (*(__majit_getfield_obj as *const #struct_path)).#member as usize
                            }
                        }
                    };
                } else {
                    *expr = syn::parse_quote! {
                        {
                            let __majit_getfield_obj = #base;
                            unsafe {
                                (*(__majit_getfield_obj as *const #struct_path)).#member
                            }
                        }
                    };
                }
                return;
            }

            syn::visit_mut::visit_expr_mut(self, expr);
        }
    }

    let field_pointees = ref_fields
        .iter()
        .map(|entry| {
            let struct_last = entry
                .struct_type
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            (
                format!("{}::{}", struct_last, entry.field),
                entry.pointee_type.clone(),
            )
        })
        .collect();
    let struct_allocs_map: HashMap<Vec<String>, syn::Path> = struct_allocs
        .iter()
        .map(|(struct_path, alloc_func)| {
            let segs: Vec<String> = struct_path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            (segs, alloc_func.clone())
        })
        .collect();
    let mut rewriter = InlineRefFieldRewriter {
        inlined_prefix: jit_interp::inlined_prefix_index(inlined_prefix),
        declared_field_keys: jit_interp::declared_field_keys(ref_fields, int_fields, array_fields),
        local_ref_types: ref_params
            .iter()
            .map(|(name, struct_type)| (name.to_string(), struct_type.clone()))
            .collect(),
        field_pointees,
        array_field_elems: array_fields
            .iter()
            .map(|entry| {
                let struct_last = entry
                    .struct_type
                    .segments
                    .last()
                    .map(|seg| seg.ident.to_string())
                    .unwrap_or_default();
                (
                    format!("{}::{}", struct_last, entry.field),
                    entry.element_type.clone(),
                )
            })
            .collect(),
        array_headers: array_fields
            .iter()
            .filter_map(|entry| {
                let header = entry.header.clone()?;
                let struct_last = entry.struct_type.segments.last()?.ident.to_string();
                Some((format!("{}::{}", struct_last, entry.field), header))
            })
            .collect(),
        struct_allocs: struct_allocs_map,
    };
    let mut block = block.clone();
    // `struct_allocs` rewrites `let x = Struct { .. }` even when the helper
    // declares no ref parameter. Gating the visit on ref params left that
    // concrete allocator unapplied.
    if !rewriter.local_ref_types.is_empty() || !rewriter.struct_allocs.is_empty() {
        rewriter.visit_block_mut(&mut block);
    }
    block
}

fn helper_policy_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_policy_{}", last.ident))
}

fn helper_call_target_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_target_{}", last.ident))
}

/// Emit an RPython attribute-named `pub const` next to the wrapper so
/// `rg <attribute>_<NAME>` finds the parity counterpart in both pyre
/// and PyPy.  RPython source citations:
///
/// * `_elidable_function_` — `rlib/jit.py` `@elidable`,
///   `@elidable_promote()`.  pyre's `_cannot_raise` / `_or_memerror`
///   variants are codewriter-derived effect classes that all start
///   from the same `_elidable_function_` attribute upstream
///   (`call.py` 3-way pick on `_canraise(op)`).
/// * `_jit_look_inside_` — `rlib/jit.py` `@dont_look_inside`
///   (`= False`); `:148` `@look_inside` (`= True`).
/// * `_jit_loop_invariant_` — `rlib/jit.py` `@loop_invariant`.
/// * `_jit_unroll_safe_` — `rlib/jit.py` `@unroll_safe`.
///
/// `_call_aroundstate_target_` (`rffi.py`) is emitted by
/// `#[call_aroundstate_target]` on `llexternal`'s `call_external_function`,
/// because it carries a 2-tuple `(funcptr, save_err)` rather than a bool.
///
/// Returns `None` for attributes with no RPython attribute counterpart
/// (e.g. `jit_may_force` — `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` is
/// analyzer-derived from `virtualizable_analyzer.analyze()`, not from
/// a wrapper attribute).
///
/// `elidable_cannot_raise` / `elidable_or_memerror` additionally emit a
/// `_jit_elidable_cannot_raise_` / `_jit_elidable_or_memerror_` marker
/// alongside `_elidable_function_`, so the ullbc hint harvester
/// (`front::llbc_hints`) can recover the strengthened-effect sub-flag
/// the policy byte collapses to `UNSUPPORTED` for ref/float-return
/// helpers. `dont_look_inside_cannot_raise` similarly emits
/// `_jit_cannot_raise_` alongside `_jit_look_inside_` so graph-pipeline
/// residual calls retain the declared exception effect.
///
/// Every function carries its markers as body-local consts, which Charon
/// promotes under the function's own path
/// (`<fn>::_jit_look_inside_<fn>`); `llbc_hints::marker_path_to_fn_path`
/// attaches such a marker to its parent function.  A body-local item takes
/// no generics from the function around it, so a method of a generic impl
/// keeps its marker in a monomorphized extraction, which instantiates only
/// the method and drops an associated const of the impl.  A body-local
/// const is also legal in a trait impl, where a foreign associated const is
/// not.
///
/// A function without a receiver additionally gets the sibling const
/// (`rpython_attribute_const_for`), so `rg _jit_look_inside_` still finds
/// the parity counterpart next to it.
fn rpython_attribute_markers(attr_name: &str) -> Option<&'static [(&'static str, bool)]> {
    Some(match attr_name {
        "elidable" | "jit_elidable" => &[("_elidable_function_", true)],
        "elidable_cannot_raise" => &[
            ("_elidable_function_", true),
            ("_jit_elidable_cannot_raise_", true),
        ],
        "elidable_or_memerror" => &[
            ("_elidable_function_", true),
            ("_jit_elidable_or_memerror_", true),
        ],
        "dont_look_inside" => &[("_jit_look_inside_", false)],
        "dont_look_inside_cannot_raise" => {
            &[("_jit_look_inside_", false), ("_jit_cannot_raise_", true)]
        }
        "look_inside" => &[("_jit_look_inside_", true)],
        "jit_loop_invariant" => &[
            ("_jit_loop_invariant_", true),
            // rlib/jit.py loop_invariant calls dont_look_inside(func):
            // the attribute implies the callee is opaque to the tracer.
            ("_jit_look_inside_", false),
        ],
        "unroll_safe" => &[("_jit_unroll_safe_", true)],
        _ => return None,
    })
}

/// The module-level sibling marker consts of a function without a
/// receiver.  `None` for a method, whose markers are body-local only
/// (`rpython_attribute_body_markers`).
fn rpython_attribute_const_for(
    attr_name: &str,
    sig: &syn::Signature,
    vis: &syn::Visibility,
) -> Option<proc_macro2::TokenStream> {
    if sig.receiver().is_some() {
        return None;
    }
    let fn_ident = &sig.ident;
    let consts = rpython_attribute_markers(attr_name)?
        .iter()
        .map(|(prefix, value)| {
            let const_name = format_ident!("{}{}", prefix, fn_ident);
            quote! {
                #[doc(hidden)]
                #[allow(non_upper_case_globals)]
                #vis const #const_name: bool = #value;
            }
        });
    Some(quote! { #(#consts)* })
}

/// The body-local marker consts of `sig`'s function, spliced at the head
/// of its body.
fn rpython_attribute_body_markers(
    attr_name: &str,
    sig: &syn::Signature,
) -> proc_macro2::TokenStream {
    let fn_ident = &sig.ident;
    let consts = rpython_attribute_markers(attr_name)
        .unwrap_or_default()
        .iter()
        .map(|(prefix, value)| {
            let const_name = format_ident!("{}{}", prefix, fn_ident);
            quote! {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #const_name: bool = #value;
            }
        });
    quote! { #(#consts)* }
}

fn primitive_type_ident(ty: &Type) -> Option<&Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    Some(&type_path.path.segments.last()?.ident)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HelperCallKind {
    Void,
    Int,
    Ref,
    Float,
    Unsupported,
}

fn is_gc_ref_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Path(type_path)
            if type_path.qself.is_none()
                && type_path
                    .path
                    .segments
                    .last()
                    .map(|segment| segment.ident == "GcRef")
                    .unwrap_or(false)
    )
}

fn is_raw_pointer_type(ty: &Type) -> bool {
    // A thin raw pointer is one ABI word. Named aliases (`PyObjectRef` =
    // `*mut PyObject`, `GCREF` = `*mut GCREFOpaque` / `llmemory.GCREF`) are
    // `Type::Path`, so a `Type::Ptr` match alone would skip them and
    // `emit_helper_call_target_fn` would refuse the word-ABI adapter
    // (`prepare_list_ref_store` had to spell `*mut PyObject` for that
    // reason).
    matches!(ty, Type::Ptr(_)) || is_pointer_alias(ty)
}

/// Path aliases of a thin pointer. The proc-macro sees tokens, not the
/// expanded type, so these last-idents are the same one-word case as
/// `Type::Ptr`.
fn is_pointer_alias(ty: &Type) -> bool {
    path_type_last_ident(ty).is_some_and(|id| id == "PyObjectRef" || id == "GCREF")
}

/// `Option<T>` whose `T` is one Ref-word pointer (`PyObjectRef`, `*mut X`,
/// `*const X` with a thin pointee). `None` is the null word and `Some(p)` is
/// `p`. `Option<i64>`, `Option<bool>`, `Option<&T>`, `Option<GcRef>` and an
/// `Option` of a fat pointer stay ordinary returns.
fn nullable_pointer_option(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    if type_path.qself.is_some() {
        return false;
    }
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    if segment.ident != "Option" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return false;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    });
    let Some(inner) = types.next() else {
        return false;
    };
    if types.next().is_some() {
        return false;
    }
    is_raw_pointer_type(inner) && !is_fat_pointer_arg(inner)
}

fn path_type_last_ident(ty: &Type) -> Option<&Ident> {
    match ty {
        Type::Path(type_path) if type_path.qself.is_none() => {
            type_path.path.segments.last().map(|segment| &segment.ident)
        }
        _ => None,
    }
}

/// The spellings whose pointee is unsized, so a reference to one is two words.
///
/// A residual-call argument is a single ABI word, which is what
/// [`helper_arg_from_i64`] rebuilds the reference from.  An unsized pointee
/// needs an address *and* a length or vtable, so it cannot survive that round
/// trip and the helper must fall through to `HelperCallKind::Unsupported`
/// instead.  Sizedness is a type-level property and this runs on syntax, so the
/// unsized spellings are named: the two structural ones, and the named types
/// this workspace actually passes by reference.  A pointee missing from here
/// fails to compile at its call site with E0606 rather than lowering wrongly.
///
/// A pair-slice parameter is still wide here: `&[T]` or `&mut [T]` stays a
/// two-word Rust reference.  When `T` is one machine word ([`pair_slice_param`]),
/// the trampoline accepts the pointer word and the length word as two adjacent
/// `i64`s and rebuilds the slice, so [`trampoline_skip_reason`] does not report
/// [`HelperFnAddrSkip::FatPointerArg`] for that parameter.  An object-pointer
/// slice (`&[PyObjectRef]`, `&[*mut PyObject]`) is not [`pair_slice_param`].
/// A shared `&str` or `&Wtf8` (any path whose last segment is `Wtf8`) is one
/// `Ref` word rebuilt by [`rstr_arg_from_i64`], so it is not a fat-pointer
/// argument either.  `&mut str`, `&Path`, `&[u8]`, an object-pointer slice,
/// a raw pointer to `str`, `Wtf8`, or a slice, and `&dyn Trait` stay
/// fat-pointer arguments.
fn is_wide_pointee(ty: &Type) -> bool {
    if matches!(ty, Type::Slice(_) | Type::TraitObject(_)) {
        return true;
    }
    let named = primitive_type_ident(ty).or_else(|| path_type_last_ident(ty));
    matches!(named, Some(ident) if ident == "str" || ident == "Wtf8" || ident == "Path")
}

fn is_reference_type(ty: &Type) -> bool {
    matches!(ty, Type::Reference(reference) if !is_wide_pointee(&reference.elem))
}

/// Shared `&str` or `&Wtf8` (last path segment `str` or `Wtf8`).
///
/// In JIT code that value is one `Ref` word: a pointer to an rstr `STR`.
/// `&mut str`, `&mut Wtf8`, and raw pointers stay fat.
fn is_shared_rstr_ref(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    if reference.mutability.is_some() {
        return false;
    }
    let named =
        primitive_type_ident(&reference.elem).or_else(|| path_type_last_ident(&reference.elem));
    matches!(named, Some(ident) if ident == "str" || ident == "Wtf8")
}

fn rstr_arg_from_i64(arg_ident: &Ident, elem: &Type) -> proc_macro2::TokenStream {
    let payload = quote! { ::majit_ir::helper_fnaddr::rstr_payload(#arg_ident) };
    let named = primitive_type_ident(elem).or_else(|| path_type_last_ident(elem));
    if named.is_some_and(|ident| ident == "str") {
        quote! { unsafe { ::core::str::from_utf8_unchecked(#payload) } }
    } else {
        quote! { unsafe { <#elem>::from_bytes_unchecked(#payload) } }
    }
}

fn is_unit_type(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(tuple) if tuple.elems.is_empty())
}

/// The item type of a `Vec<T>` whose items are one machine word: a raw
/// pointer, a `GcRef`, or a word-sized integer or `f64`.  The lowering holds
/// such a `Vec` as the address of its raw three-word header (`rrustvec.rs`
/// `RustVecRepr`, kind `int`).
fn vec_one_word_item(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    let item = types.next()?;
    if types.next().is_some() {
        return None;
    }
    let one_word = is_raw_pointer_type(item)
        || is_gc_ref_type(item)
        || primitive_type_ident(item).is_some_and(|ident| {
            matches!(
                ident.to_string().as_str(),
                "usize" | "isize" | "u64" | "i64" | "f64"
            )
        });
    one_word.then_some(item)
}

fn helper_call_kind_for_type(ty: &Type) -> HelperCallKind {
    if is_unit_type(ty) {
        return HelperCallKind::Void;
    }
    if vec_one_word_item(ty).is_some() {
        return HelperCallKind::Int;
    }
    if is_gc_ref_type(ty)
        || is_raw_pointer_type(ty)
        || is_reference_type(ty)
        || is_shared_rstr_ref(ty)
        || nullable_pointer_option(ty)
    {
        return HelperCallKind::Ref;
    }
    match primitive_type_ident(ty)
        .map(|ident| ident.to_string())
        .as_deref()
    {
        Some(
            "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
            | "bool",
        ) => HelperCallKind::Int,
        Some("f64") => HelperCallKind::Float,
        _ => HelperCallKind::Unsupported,
    }
}

fn helper_call_kind_for_return(output: &ReturnType) -> HelperCallKind {
    match output {
        ReturnType::Default => HelperCallKind::Void,
        ReturnType::Type(_, ty) => helper_call_kind_for_type(ty),
    }
}

/// Rebuild one ABI word as the helper's Rust argument.
///
/// A pair-slice parameter (`&[T]` / `&mut [T]` of a one-word item, see
/// [`pair_slice_param`]) is not rebuilt here.  It occupies two `i64` words,
/// and [`emit_helper_call_target_fn`] turns `__majit_arg_{i}` plus the
/// adjacent `__majit_arg_{i}_len` back into the slice: length 0 is `&[]` or
/// `&mut []`, and a positive length is `from_raw_parts` / `from_raw_parts_mut`.
fn helper_arg_from_i64(arg_ident: &Ident, ty: &Type) -> Option<proc_macro2::TokenStream> {
    if is_gc_ref_type(ty) {
        return Some(quote! { #ty((#arg_ident) as usize) });
    }
    if is_raw_pointer_type(ty) {
        return Some(quote! { ((#arg_ident) as usize) as #ty });
    }
    if is_shared_rstr_ref(ty)
        && let Type::Reference(reference) = ty
    {
        return Some(rstr_arg_from_i64(arg_ident, &reference.elem));
    }
    if is_reference_type(ty)
        && let Type::Reference(reference) = ty
    {
        let elem = &reference.elem;
        return if reference.mutability.is_some() {
            Some(quote! { unsafe { &mut *((#arg_ident) as usize as *mut #elem) } })
        } else {
            Some(quote! { unsafe { &*((#arg_ident) as usize as *const #elem) } })
        };
    }
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            Some(quote! { (#arg_ident) as #ty })
        }
        "i64" => Some(quote! { #arg_ident }),
        "bool" => Some(quote! { (#arg_ident) != 0 }),
        "f64" => Some(quote! { f64::from_bits((#arg_ident) as u64) }),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HelperFnAddrSkip {
    Generic,
    FatPointerArg,
    ResultReturn,
    MethodReceiver,
    Other,
}

impl HelperFnAddrSkip {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::FatPointerArg => "fat-pointer arg",
            Self::ResultReturn => "Result return",
            Self::MethodReceiver => "method receiver",
            Self::Other => "other",
        }
    }
}

fn is_fat_pointer_arg(ty: &Type) -> bool {
    match ty {
        Type::Reference(reference) => is_wide_pointee(&reference.elem),
        Type::Ptr(ptr) => is_wide_pointee(&ptr.elem),
        _ => false,
    }
}

/// `&[T]` / `&mut [T]` whose item is one machine word and whose lowering is
/// the `(ptr, len)` pair.
///
/// The item is one of: a path whose last segment is `GcRef`, a thin raw
/// pointer `*const X` / `*mut X` (or a one-word pointer alias), or
/// `i64` / `u64` / `isize` / `usize`. `PyObjectRef` and `*mut PyObject`
/// / `*const PyObject` are the length-prefixed object array, one word,
/// so they are not pair items.
/// Returns the item type and whether the slice is mutable.
fn pair_slice_param(ty: &Type) -> Option<(&Type, bool)> {
    let Type::Reference(reference) = ty else {
        return None;
    };
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return None;
    };
    let item = slice.elem.as_ref();
    is_pair_slice_item(item).then_some((item, reference.mutability.is_some()))
}

/// `PyObjectRef` or `*mut PyObject` / `*const PyObject`.
fn is_object_pointer_item(ty: &Type) -> bool {
    if path_type_last_ident(ty).is_some_and(|id| id == "PyObjectRef") {
        return true;
    }
    let Type::Ptr(ptr) = ty else {
        return false;
    };
    path_type_last_ident(&ptr.elem).is_some_and(|id| id == "PyObject")
}

fn is_pair_slice_item(ty: &Type) -> bool {
    if is_object_pointer_item(ty) {
        return false;
    }
    if is_gc_ref_type(ty) {
        return true;
    }
    if is_raw_pointer_type(ty) {
        return !is_fat_pointer_arg(ty);
    }
    primitive_type_ident(ty).is_some_and(|ident| {
        matches!(
            ident.to_string().as_str(),
            "i64" | "u64" | "isize" | "usize"
        )
    })
}

fn pair_slice_expr(
    ptr: &Ident,
    len: &Ident,
    item_ty: &Type,
    is_mut: bool,
) -> proc_macro2::TokenStream {
    let (empty, raw) = if is_mut {
        (
            quote! { &mut [] },
            quote! { std::slice::from_raw_parts_mut(#ptr as usize as *mut #item_ty, #len as usize) },
        )
    } else {
        (
            quote! { &[] },
            quote! { std::slice::from_raw_parts(#ptr as usize as *const #item_ty, #len as usize) },
        )
    };
    quote! {
        if #len == 0 {
            #empty
        } else {
            unsafe { #raw }
        }
    }
}

/// ABI words of `inputs`.  A pair-slice parameter is two words (pointer, length).
fn helper_abi_words(inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>) -> u8 {
    inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Typed(pat_type) if pair_slice_param(&pat_type.ty).is_some() => 2u8,
            _ => 1u8,
        })
        .sum()
}

fn result_ok_and_err(ty: &Type) -> Option<(&Type, &Type)> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last = type_path.path.segments.last()?;
    // `pyre-interpreter error.rs`: `pub type PyResult = Result<PyObjectRef,
    // PyError>`. The alias carries no generic arguments to read the two
    // types from, so spell them here.
    if last.ident == "PyResult" && matches!(last.arguments, syn::PathArguments::None) {
        let ok: &'static Type = Box::leak(Box::new(
            syn::parse_str("PyObjectRef").expect("PyResult ok type"),
        ));
        let err: &'static Type = Box::leak(Box::new(syn::parse_quote!(PyError)));
        return Some((ok, err));
    }
    if last.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    });
    Some((types.next()?, types.next()?))
}

/// Last ident of the exception carrier `tyref_is_result_of_carrier` matches
/// against `ErrorCarrierSpec.carrier_path` (ends in `PyError`).
/// `dont_look_inside_return_token` uses that same gate before projecting
/// `FUNC.RESULT` to the Ok payload. Other `Result` error types stay real
/// ADTs, so a payload trampoline would disagree.
fn is_project_error_type(ty: &Type) -> bool {
    path_type_last_ident(ty).is_some_and(|id| id == "PyError")
}

fn result_exc_payload(output: &ReturnType) -> Option<&Type> {
    let ReturnType::Type(_, ty) = output else {
        return None;
    };
    let (ok, err) = result_ok_and_err(ty)?;
    is_project_error_type(err).then_some(ok)
}

fn payload_is_projectable(ok_ty: &Type) -> bool {
    match helper_call_kind_for_type(ok_ty) {
        HelperCallKind::Unsupported => false,
        HelperCallKind::Void => true,
        HelperCallKind::Int | HelperCallKind::Ref | HelperCallKind::Float => {
            helper_return_to_i64(quote! { __probe }, ok_ty).is_some()
                || helper_call_kind_for_type(ok_ty) == HelperCallKind::Float
        }
    }
}

fn wrap_result_exc_call(
    inner: proc_macro2::TokenStream,
    return_kind: HelperCallKind,
    payload_ty: &Type,
) -> Option<proc_macro2::TokenStream> {
    match return_kind {
        HelperCallKind::Void => Some(quote! {
            match #inner {
                ::core::result::Result::Ok(_) => {}
                ::core::result::Result::Err(__majit_err) => {
                    ::majit_ir::helper_fnaddr::ResidualError::publish_residual(__majit_err);
                }
            }
        }),
        HelperCallKind::Float => Some(quote! {
            match #inner {
                ::core::result::Result::Ok(__majit_ok) => __majit_ok,
                ::core::result::Result::Err(__majit_err) => {
                    ::majit_ir::helper_fnaddr::ResidualError::publish_residual(__majit_err);
                    0.0
                }
            }
        }),
        HelperCallKind::Int | HelperCallKind::Ref => {
            let converted = helper_return_to_i64(quote! { __majit_ok }, payload_ty)?;
            Some(quote! {
                match #inner {
                    ::core::result::Result::Ok(__majit_ok) => #converted,
                    ::core::result::Result::Err(__majit_err) => {
                        ::majit_ir::helper_fnaddr::ResidualError::publish_residual(__majit_err);
                        0
                    }
                }
            })
        }
        HelperCallKind::Unsupported => None,
    }
}

/// Why a helper gets no residual-call trampoline, or `None` when one is emitted.
///
/// A pair-slice parameter ([`pair_slice_param`]) is not
/// [`HelperFnAddrSkip::FatPointerArg`]: its pointer and length are two `i64`
/// words.  A shared `&str` or `&Wtf8` is one `Ref` word, same as
/// `PyObjectRef` / `GcRef`.  `&[PyObjectRef]` and `&[*mut PyObject]` stay
/// fat-pointer arguments: that slice is one length-prefixed object-array
/// word, and `from_raw_parts` would not rebuild it.  Every other fat
/// pointer (`&mut str`, `&Path`, `&[u8]`, `&[u32]`, `&[f64]`, `*const str`,
/// `*const [T]`, `&dyn Trait`, a slice of any other item) still skips as
/// `FatPointerArg`.  A return of `&str` or `&Wtf8` still skips.
fn trampoline_skip_reason(
    func: &ItemFn,
    attr_name: &str,
    word_enums: &[Path],
) -> Option<HelperFnAddrSkip> {
    if !func.sig.generics.params.is_empty() {
        return Some(HelperFnAddrSkip::Generic);
    }
    for arg in &func.sig.inputs {
        if let FnArg::Typed(pat_type) = arg
            && pair_slice_param(&pat_type.ty).is_none()
            && !is_shared_rstr_ref(&pat_type.ty)
            && is_fat_pointer_arg(&pat_type.ty)
        {
            return Some(HelperFnAddrSkip::FatPointerArg);
        }
    }
    let result_payload = if let ReturnType::Type(_, ty) = &func.sig.output {
        if let Some((ok_ty, err_ty)) = result_ok_and_err(ty) {
            if attr_name == "dont_look_inside_cannot_raise" || !is_project_error_type(err_ty) {
                return Some(HelperFnAddrSkip::ResultReturn);
            }
            if is_fat_pointer_arg(ok_ty) {
                return Some(HelperFnAddrSkip::Other);
            }
            if !payload_is_projectable(ok_ty) {
                return Some(HelperFnAddrSkip::ResultReturn);
            }
            Some(ok_ty)
        } else {
            None
        }
    } else {
        None
    };
    if let ReturnType::Type(_, ty) = &func.sig.output
        && result_payload.is_none()
        && is_fat_pointer_arg(ty)
    {
        return Some(HelperFnAddrSkip::Other);
    }
    if func.sig.receiver().is_some() {
        return Some(HelperFnAddrSkip::MethodReceiver);
    }
    for arg in &func.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Some(HelperFnAddrSkip::Other);
        };
        if pair_slice_param(&pat_type.ty).is_some() {
            continue;
        }
        if helper_arg_from_i64(&format_ident!("__probe"), &pat_type.ty).is_none()
            && !is_word_enum_arg(&pat_type.ty, word_enums)
        {
            return Some(HelperFnAddrSkip::Other);
        }
    }
    if result_payload.is_some() {
        return None;
    }
    match helper_call_kind_for_return(&func.sig.output) {
        HelperCallKind::Unsupported => Some(HelperFnAddrSkip::Other),
        HelperCallKind::Void => None,
        HelperCallKind::Int | HelperCallKind::Ref | HelperCallKind::Float => {
            let ReturnType::Type(_, ty) = &func.sig.output else {
                return Some(HelperFnAddrSkip::Other);
            };
            if helper_return_to_i64(quote! { __probe }, ty).is_none() {
                Some(HelperFnAddrSkip::Other)
            } else {
                None
            }
        }
    }
}

fn report_helper_fnaddr_skip(name: &Ident, reason: HelperFnAddrSkip) {
    if std::env::var("MAJIT_HELPER_FNADDR_SKIP").is_ok() {
        eprintln!("[MAJIT_HELPER_FNADDR_SKIP] {}: {name}", reason.as_str());
    }
}

/// A unique side-effect so LLVM MergeFunctions cannot fold two residual-call
/// targets that would otherwise have identical bodies. `drain_list_append`
/// keeps a forwarding call for the same reason: one published address must
/// name one function (`registered_paths_sharing_an_address_are_alias_spellings`).
///
/// Expands to [`majit_ir::icf_identity`]: empty `nomem` asm whose comment
/// carries a unique const immediate (native) or a volatile read of that
/// immediate (wasm32), never `black_box` of a pointer.
fn icf_identity_tokens(name: &Ident) -> proc_macro2::TokenStream {
    quote! {
        ::majit_ir::icf_identity!(::core::concat!(
            ::core::module_path!(),
            "::",
            stringify!(#name),
        ));
    }
}

fn emit_helper_fnaddr_registration(
    path_name: &Ident,
    trampoline_ident: &Ident,
    arity: u8,
) -> proc_macro2::TokenStream {
    let static_name = format_ident!("__MAJIT_HELPER_FNADDR_{path_name}");
    let row = quote! {
        ::majit_ir::helper_fnaddr::HelperFnAddr::new(
            ::core::concat!(::core::module_path!(), "::", stringify!(#path_name)),
            #trampoline_ident as *const (),
            #arity,
        )
    };
    quote! {
        #[cfg(not(target_arch = "wasm32"))]
        {
            #[::majit_ir::linkme::distributed_slice(::majit_ir::helper_fnaddr::HELPER_FNADDRS)]
            #[linkme(crate = ::majit_ir::linkme)]
            #[allow(non_upper_case_globals, unused)]
            static #static_name: ::majit_ir::helper_fnaddr::HelperFnAddr = #row;
        }
    }
}

fn emit_helper_fnaddr_registrations(
    path_names: &[&Ident],
    trampoline_ident: &Ident,
    arity: u8,
) -> proc_macro2::TokenStream {
    let rows = path_names
        .iter()
        .map(|path_name| emit_helper_fnaddr_registration(path_name, trampoline_ident, arity));
    quote! { #(#rows)* }
}

fn emit_helper_fnaddr_ctor(
    path_name: &Ident,
    trampoline_ident: &Ident,
    arity: u8,
) -> proc_macro2::TokenStream {
    let ctor_name = format_ident!("__majit_register_helper_fnaddr_{path_name}");
    let register = quote! {
        ::majit_ir::helper_fnaddr::register(
            ::core::concat!(::core::module_path!(), "::", stringify!(#path_name)),
            #trampoline_ident as *const (),
            #arity,
        );
    };
    quote! {
        #[cfg(target_arch = "wasm32")]
        #[::ctor::ctor(unsafe)]
        fn #ctor_name() {
            #register
        }
    }
}

fn emit_helper_fnaddr_ctors(
    path_names: &[&Ident],
    trampoline_ident: &Ident,
    arity: u8,
) -> proc_macro2::TokenStream {
    let ctors = path_names
        .iter()
        .map(|path_name| emit_helper_fnaddr_ctor(path_name, trampoline_ident, arity));
    quote! { #(#ctors)* }
}

fn emit_prebuilt_class_static_registration(name: &Ident) -> proc_macro2::TokenStream {
    let static_name = format_ident!("__MAJIT_PREBUILT_CLASS_STATIC_{name}");
    quote! {
        #[cfg(not(target_arch = "wasm32"))]
        #[::majit_ir::linkme::distributed_slice(::majit_ir::helper_fnaddr::PREBUILT_CLASS_STATICS)]
        #[linkme(crate = ::majit_ir::linkme)]
        #[allow(non_upper_case_globals)]
        static #static_name: ::majit_ir::helper_fnaddr::PrebuiltStaticAddr =
            ::majit_ir::helper_fnaddr::PrebuiltStaticAddr::new(
                ::core::concat!(::core::module_path!(), "::", stringify!(#name)),
                &#name as *const _ as *const (),
            );
    }
}

fn emit_prebuilt_class_static_ctor(name: &Ident) -> proc_macro2::TokenStream {
    let ctor_name = format_ident!("__majit_register_prebuilt_class_static_{name}");
    quote! {
        #[cfg(target_arch = "wasm32")]
        #[::ctor::ctor(unsafe)]
        #[allow(non_snake_case)]
        fn #ctor_name() {
            ::majit_ir::helper_fnaddr::register_prebuilt_class_static(
                ::core::concat!(::core::module_path!(), "::", stringify!(#name)),
                &#name as *const _ as *const (),
            );
        }
    }
}

/// Publish a `static` into the JIT address tables.
///
/// An `Atomic*` static gets a nullary getter. That getter answers the
/// residual read the front end emits for a static no address table resolves:
/// arity 0, registered under the static's own path, returning the `Relaxed`
/// load the front end folds. A `PyType` static is a prebuilt class-singleton
/// address.
#[proc_macro_attribute]
pub fn prebuilt_static(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: proc_macro2::TokenStream = item.into();
    let parsed = match syn::parse2::<syn::ItemStatic>(item.clone()) {
        Ok(parsed) => parsed,
        Err(_) => {
            return quote! {
                compile_error!("prebuilt_static: expected a static");
                #item
            }
            .into();
        }
    };
    if matches!(parsed.mutability, syn::StaticMutability::Mut(_)) {
        return quote! {
            compile_error!("prebuilt_static: static mut is not supported");
            #item
        }
        .into();
    }
    let name = &parsed.ident;
    let Some(last) = path_type_last_ident(&parsed.ty) else {
        return quote! {
            compile_error!("prebuilt_static: unsupported static type");
            #item
        }
        .into();
    };
    let last_name = last.to_string();
    if last_name == "PyType" {
        let registration = emit_prebuilt_class_static_registration(name);
        let ctor = emit_prebuilt_class_static_ctor(name);
        return quote! {
            #item
            #registration
            #ctor
        }
        .into();
    }
    if last_name.starts_with("Atomic") {
        let getter = format_ident!("__majit_prebuilt_static_get_{name}");
        let loaded = quote! { #name.load(::core::sync::atomic::Ordering::Relaxed) };
        let value = if last_name == "AtomicPtr" {
            quote! { #loaded as usize as i64 }
        } else {
            quote! { #loaded as i64 }
        };
        let registration = emit_helper_fnaddr_registration(name, &getter, 0);
        let ctor = emit_helper_fnaddr_ctor(name, &getter, 0);
        return quote! {
            #item
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn #getter() -> i64 {
                #registration
                #value
            }
            #ctor
        }
        .into();
    }
    quote! {
        compile_error!("prebuilt_static: unsupported static type");
        #item
    }
    .into()
}

fn helper_return_to_i64(
    value: proc_macro2::TokenStream,
    ty: &Type,
) -> Option<proc_macro2::TokenStream> {
    // A by-value `Vec` result is moved into a header the lowering owns:
    // `raw_malloc_varsize_char` of the three header words, which
    // `ll_vec_free_*` releases together with the buffer.
    if vec_one_word_item(ty).is_some() {
        return Some(quote! {{
            let __majit_vec: #ty = #value;
            let __majit_header = ::majit_rlib::lltypesystem::rffi::raw_malloc_varsize_char(
                ::core::mem::size_of::<#ty>(),
            ) as *mut #ty;
            unsafe { __majit_header.write(__majit_vec) };
            __majit_header as usize as i64
        }});
    }
    if is_gc_ref_type(ty) {
        return Some(quote! { (#value).0 as i64 });
    }
    if is_raw_pointer_type(ty) {
        return Some(quote! { (#value) as usize as i64 });
    }
    // Same word as a plain `PyObjectRef` return: `Some(p)` is that pointer,
    // `None` is the null word.
    if nullable_pointer_option(ty) {
        return Some(quote! {
            match #value {
                ::core::option::Option::Some(__majit_ptr) => (__majit_ptr as usize) as i64,
                ::core::option::Option::None => 0,
            }
        });
    }
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "u8" | "u16" | "u32" | "u64" | "usize" | "bool" => {
            Some(quote! { (#value) as i64 })
        }
        "i64" => Some(quote! { #value }),
        "isize" => Some(quote! { (#value) as i64 }),
        "f64" => Some(quote! { f64::to_bits(#value) as i64 }),
        _ => None,
    }
}

fn is_word_enum_arg(ty: &Type, word_enums: &[Path]) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    if type_path.qself.is_some() {
        return false;
    }
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    word_enums.iter().any(|path| {
        path.segments
            .last()
            .is_some_and(|listed| listed.ident == segment.ident)
    })
}

fn emit_helper_call_target_fn(
    func: &ItemFn,
    register_fnaddr: bool,
    register_as: Option<&Ident>,
    attr_name: &str,
    word_enums: &[Path],
    for_native_entry: bool,
) -> syn::Result<Option<(Ident, Ident, proc_macro2::TokenStream)>> {
    if let Some(reason) = trampoline_skip_reason(func, attr_name, word_enums) {
        if register_fnaddr {
            report_helper_fnaddr_skip(&func.sig.ident, reason);
        }
        return Ok(None);
    }

    let trace_target_name = helper_call_target_fn_name(&Path::from(func.sig.ident.clone()))?;
    let concrete_target_name = format_ident!("{}_concrete", trace_target_name);
    let mut wrapper_params = Vec::new();
    let mut converted_args = Vec::new();
    let mut fnaddr_params = Vec::new();
    let mut fnaddr_args = Vec::new();
    let mut has_float_arg = false;
    for (index, arg) in func.sig.inputs.iter().enumerate() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Ok(None);
        };
        let arg_ident = format_ident!("__majit_arg_{index}");
        if let Some((item_ty, is_mut)) = pair_slice_param(&pat_type.ty) {
            let len_ident = format_ident!("__majit_arg_{index}_len");
            wrapper_params.push(quote! { #arg_ident: i64 });
            wrapper_params.push(quote! { #len_ident: i64 });
            let rebuilt = pair_slice_expr(&arg_ident, &len_ident, item_ty, is_mut);
            converted_args.push(rebuilt.clone());
            fnaddr_params.push(quote! { #arg_ident: i64 });
            fnaddr_params.push(quote! { #len_ident: i64 });
            fnaddr_args.push(rebuilt);
            continue;
        }
        wrapper_params.push(quote! { #arg_ident: i64 });
        let word_enum = is_word_enum_arg(&pat_type.ty, word_enums);
        let converted = if word_enum {
            let ty = &pat_type.ty;
            quote! {
                <#ty as ::majit_ir::helper_fnaddr::FieldlessEnumArg>::from_discriminant(#arg_ident)
            }
        } else if let Some(converted) = helper_arg_from_i64(&arg_ident, &pat_type.ty) {
            converted
        } else {
            return Ok(None);
        };
        converted_args.push(converted.clone());
        if !word_enum && helper_call_kind_for_type(&pat_type.ty) == HelperCallKind::Float {
            has_float_arg = true;
            fnaddr_params.push(quote! { #arg_ident: f64 });
            fnaddr_args.push(quote! { #arg_ident });
        } else {
            fnaddr_params.push(quote! { #arg_ident: i64 });
            fnaddr_args.push(converted);
        }
    }

    // Wrapper visibility follows the user fn so external integration
    // tests can use the macro-emitted `extern "C"` ABI trampoline as
    // the actual trace function pointer (PyPy `getfunctionptr` parity
    // verification).  `#[doc(hidden)]` + `__majit_call_target_*`
    // naming keeps it off the user-facing surface.
    let vis = &func.vis;
    let helper_name = &func.sig.ident;
    // Residual CALL_PURE targets this helper (`jit.py elidable_promote`:
    // `_orig_func_unlikely_name` is `func`, `getfunctionptr(graph)` of that
    // graph). `register_as` is the promoting wrapper's ident; publish both
    // so a lookup of either spelling finds this trampoline.
    let fnaddr_path_names: Vec<&Ident> = match register_as {
        Some(alias) if alias != helper_name => vec![helper_name, alias],
        _ => vec![helper_name],
    };
    // An `unsafe fn` helper must be called inside an `unsafe` block from the
    // generated `extern "C"` trampoline; a safe helper is called bare (an
    // `unsafe` wrapper there would be an unused-unsafe warning).
    let mut call_expr = if func.sig.unsafety.is_some() {
        quote! { unsafe { #helper_name(#(#converted_args),*) } }
    } else {
        quote! { #helper_name(#(#converted_args),*) }
    };
    let mut fnaddr_call_expr = if func.sig.unsafety.is_some() {
        quote! { unsafe { #helper_name(#(#fnaddr_args),*) } }
    } else {
        quote! { #helper_name(#(#fnaddr_args),*) }
    };
    let result_payload = result_exc_payload(&func.sig.output);
    let return_kind = if let Some(ok_ty) = result_payload {
        helper_call_kind_for_type(ok_ty)
    } else {
        helper_call_kind_for_return(&func.sig.output)
    };
    if let Some(ok_ty) = result_payload {
        let Some(wrapped) = wrap_result_exc_call(call_expr, return_kind, ok_ty) else {
            return Ok(None);
        };
        call_expr = wrapped;
        let Some(fnaddr_wrapped) = wrap_result_exc_call(fnaddr_call_expr, return_kind, ok_ty)
        else {
            return Ok(None);
        };
        fnaddr_call_expr = fnaddr_wrapped;
    }
    let arity = helper_abi_words(&func.sig.inputs);
    let shim_registration = if register_fnaddr && !has_float_arg {
        emit_helper_fnaddr_registrations(&fnaddr_path_names, &trace_target_name, arity)
    } else {
        quote! {}
    };
    let abi_return_ty = result_payload.or_else(|| match &func.sig.output {
        ReturnType::Type(_, ty) => Some(ty.as_ref()),
        ReturnType::Default => None,
    });
    let trace_identity = icf_identity_tokens(&trace_target_name);
    let concrete_identity = icf_identity_tokens(&concrete_target_name);
    let wrapper = match return_kind {
        HelperCallKind::Void => quote! {
            #[doc(hidden)]
            #[allow(non_snake_case)]
            #vis extern "C" fn #trace_target_name(#(#wrapper_params),*) {
                #shim_registration
                #trace_identity
                #call_expr;
            }
        },
        HelperCallKind::Int | HelperCallKind::Ref => {
            let converted_return = if result_payload.is_some() {
                call_expr.clone()
            } else {
                let Some(ty) = abi_return_ty else {
                    return Ok(None);
                };
                let Some(converted) = helper_return_to_i64(call_expr.clone(), ty) else {
                    return Ok(None);
                };
                converted
            };
            quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #trace_target_name(#(#wrapper_params),*) -> i64 {
                    #shim_registration
                    #trace_identity
                    #converted_return
                }
            }
        }
        HelperCallKind::Float => {
            let Some(ty) = abi_return_ty else {
                return Ok(None);
            };
            let float_wrapper = quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #trace_target_name(#(#wrapper_params),*) -> f64 {
                    #shim_registration
                    #trace_identity
                    #call_expr
                }
            };
            let Some(concrete_return) = helper_return_to_i64(call_expr.clone(), ty) else {
                return Ok(None);
            };
            let concrete_wrapper = quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #concrete_target_name(#(#wrapper_params),*) -> i64 {
                    #concrete_identity
                    #concrete_return
                }
            };
            quote! {
                #float_wrapper
                #concrete_wrapper
            }
        }
        HelperCallKind::Unsupported => return Ok(None),
    };

    let registered = if register_fnaddr && has_float_arg {
        let fnaddr_name = format_ident!("__majit_fnaddr_target_{helper_name}");
        let registration =
            emit_helper_fnaddr_registrations(&fnaddr_path_names, &fnaddr_name, arity);
        let ctor = emit_helper_fnaddr_ctors(&fnaddr_path_names, &fnaddr_name, arity);
        let fnaddr_identity = icf_identity_tokens(&fnaddr_name);
        let float_abi = match return_kind {
            HelperCallKind::Void => quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #fnaddr_name(#(#fnaddr_params),*) {
                    #registration
                    #fnaddr_identity
                    #fnaddr_call_expr;
                }
            },
            HelperCallKind::Float => quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #fnaddr_name(#(#fnaddr_params),*) -> f64 {
                    #registration
                    #fnaddr_identity
                    #fnaddr_call_expr
                }
            },
            HelperCallKind::Int | HelperCallKind::Ref => {
                let converted_return = if result_payload.is_some() {
                    fnaddr_call_expr.clone()
                } else {
                    let Some(ty) = abi_return_ty else {
                        return Ok(None);
                    };
                    let Some(converted) = helper_return_to_i64(fnaddr_call_expr.clone(), ty) else {
                        return Ok(None);
                    };
                    converted
                };
                quote! {
                    #[doc(hidden)]
                    #[allow(non_snake_case)]
                    #vis extern "C" fn #fnaddr_name(#(#fnaddr_params),*) -> i64 {
                        #registration
                        #fnaddr_identity
                        #converted_return
                    }
                }
            }
            HelperCallKind::Unsupported => return Ok(None),
        };
        quote! {
            #float_abi
            #ctor
        }
    } else if register_fnaddr {
        emit_helper_fnaddr_ctors(&fnaddr_path_names, &trace_target_name, arity)
    } else {
        quote! {}
    };

    // `bh_call_*` passes a real `f64` where `arg_classes` says `'f'`. The
    // widening shim takes `i64` and reconstructs the float inside, so a
    // native entry has to be this declaration-order ABI instead.
    let (entry_name, native_abi) = if for_native_entry && has_float_arg {
        let native_name = format_ident!("__majit_native_entry_{helper_name}");
        let native_fn = match return_kind {
            HelperCallKind::Void => quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #native_name(#(#fnaddr_params),*) {
                    #fnaddr_call_expr;
                }
            },
            HelperCallKind::Float => quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                #vis extern "C" fn #native_name(#(#fnaddr_params),*) -> f64 {
                    #fnaddr_call_expr
                }
            },
            HelperCallKind::Int | HelperCallKind::Ref => {
                let converted_return = if result_payload.is_some() {
                    fnaddr_call_expr.clone()
                } else {
                    let Some(ty) = abi_return_ty else {
                        return Ok(None);
                    };
                    let Some(converted) = helper_return_to_i64(fnaddr_call_expr.clone(), ty) else {
                        return Ok(None);
                    };
                    converted
                };
                quote! {
                    #[doc(hidden)]
                    #[allow(non_snake_case)]
                    #vis extern "C" fn #native_name(#(#fnaddr_params),*) -> i64 {
                        #converted_return
                    }
                }
            }
            HelperCallKind::Unsupported => return Ok(None),
        };
        (native_name, native_fn)
    } else {
        (trace_target_name.clone(), quote! {})
    };

    let wrapper = quote! {
        #wrapper
        #registered
        #native_abi
    };

    let concrete_name = if matches!(return_kind, HelperCallKind::Float) {
        concrete_target_name
    } else {
        trace_target_name.clone()
    };
    Ok(Some((entry_name, concrete_name, wrapper)))
}

fn helper_policy_tokens_for_fn(
    func: &ItemFn,
    attr_name: &str,
    trace_target_name: Option<&Ident>,
    concrete_target_name: Option<&Ident>,
) -> syn::Result<proc_macro2::TokenStream> {
    let unsupported_byte = jit_interp::call_policy_byte::UNSUPPORTED;
    let unsupported = quote! {
        (#unsupported_byte, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), 0i32)
    };
    let (Some(trace_target_name), Some(concrete_target_name)) =
        (trace_target_name, concrete_target_name)
    else {
        return Ok(unsupported);
    };
    let trace_addr = quote! { #trace_target_name as *const () };
    let concrete_addr = quote! { #concrete_target_name as *const () };
    // 6-tuple: (policy, inline_builder, trace_target, concrete_target, prebuild, save_err).
    // `prebuild` is the per-helper liveness prebuild fn pointer or null
    // for non-Inline helpers (these have no per-marker triples to register).
    // Only `#[jit_inline]` emits a real prebuild fn; every
    // other helper attribute that flows through here advertises null and
    // the parent `#[jit_interp]` lowerer's inferred-policy site
    // (`jitcode_lower.rs::CallPolicySpec::Infer`) skips the call.
    // The trailing `save_err` is `0i32` (`RFFI_ERR_NONE`). A GIL-releasing
    // external carries `save_err` on `EffectInfo.call_release_gil_target`,
    // filled by `getcalldescr` from `_call_aroundstate_target_`.
    use jit_interp::call_policy_byte::{
        INT_DONT_LOOK_INSIDE, INT_DONT_LOOK_INSIDE_CANNOT_RAISE, INT_ELIDABLE,
        INT_ELIDABLE_CANNOT_RAISE, INT_ELIDABLE_OR_MEMERROR, INT_LOOP_INVARIANT, INT_MAY_FORCE,
        REF_DONT_LOOK_INSIDE, REF_DONT_LOOK_INSIDE_CANNOT_RAISE, REF_ELIDABLE,
        REF_ELIDABLE_CANNOT_RAISE, REF_ELIDABLE_OR_MEMERROR, REF_LOOP_INVARIANT, REF_MAY_FORCE,
        UNSUPPORTED, VOID_DONT_LOOK_INSIDE, VOID_DONT_LOOK_INSIDE_CANNOT_RAISE,
        VOID_LOOP_INVARIANT, VOID_MAY_FORCE,
    };
    match helper_call_kind_for_return(&func.sig.output) {
        HelperCallKind::Void => Ok(match attr_name {
            "dont_look_inside" => quote! {
                (#VOID_DONT_LOOK_INSIDE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            // `call.py getcalldescr`'s non-elidable `else` branch —
            // EF_CANNOT_RAISE for void-return helpers.  Same dispatch
            // surface as `dont_look_inside` (residual call), but the
            // recording walker uses `cannot_raise_effect_info()` so no
            // trailing `-live-` is required.
            "dont_look_inside_cannot_raise" => quote! {
                (#VOID_DONT_LOOK_INSIDE_CANNOT_RAISE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_may_force" => quote! {
                (#VOID_MAY_FORCE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_loop_invariant" => quote! {
                (#VOID_LOOP_INVARIANT, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            _ => unsupported,
        }),
        HelperCallKind::Int => Ok(match attr_name {
            "elidable" => quote! {
                (#INT_ELIDABLE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            // call.py elidable && _canraise(op) == False — EF_ELIDABLE_CANNOT_RAISE.
            "elidable_cannot_raise" => quote! {
                (#INT_ELIDABLE_CANNOT_RAISE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            // call.py elidable && _canraise(op) == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
            "elidable_or_memerror" => quote! {
                (#INT_ELIDABLE_OR_MEMERROR, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "dont_look_inside" => quote! {
                (#INT_DONT_LOOK_INSIDE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            // `call.py:303` non-elidable EF_CANNOT_RAISE for int-return helpers.
            "dont_look_inside_cannot_raise" => quote! {
                (#INT_DONT_LOOK_INSIDE_CANNOT_RAISE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_may_force" => quote! {
                (#INT_MAY_FORCE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_loop_invariant" => quote! {
                (#INT_LOOP_INVARIANT, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            _ => unsupported,
        }),
        HelperCallKind::Ref => Ok(match attr_name {
            // `lower_call_*` cannot recover a static ref-return
            // `BindingKind` from these bytes and therefore still
            // dispatches the residual_call builder family from the
            // explicit-policy match.  However, the cond_call /
            // record_known_result lowerers know the binding kind
            // from the leading argument, so for them the policy byte
            // identifies both the `EffectInfoSlot` and the PyPy
            // getcalldescr checks that must run before registering the
            // descr (result-kind match, forces/release-gil rejection,
            // loop-invariant no-args assertion).
            "elidable" => quote! {
                (#REF_ELIDABLE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "elidable_cannot_raise" => quote! {
                (#REF_ELIDABLE_CANNOT_RAISE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "elidable_or_memerror" => quote! {
                (#REF_ELIDABLE_OR_MEMERROR, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_loop_invariant" => quote! {
                (#REF_LOOP_INVARIANT, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "dont_look_inside" => quote! {
                (#REF_DONT_LOOK_INSIDE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            // `call.py:303` non-elidable EF_CANNOT_RAISE for ref-return helpers.
            // Closes the audit's Item 4 parity-divergence (ref dont_look_inside
            // mapping to CanRaise).  Existing `dont_look_inside` (REF_DONT_LOOK_INSIDE)
            // stays CanRaise (conservative default for residuals whose annotation
            // is unknown); REF_DONT_LOOK_INSIDE_CANNOT_RAISE is the explicit
            // cannot-raise opt-in.
            "dont_look_inside_cannot_raise" => quote! {
                (#REF_DONT_LOOK_INSIDE_CANNOT_RAISE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            "jit_may_force" => quote! {
                (#REF_MAY_FORCE, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            _ => unsupported,
        }),
        HelperCallKind::Float => Ok(match attr_name {
            // Inferred value-call lowering cannot model a static float
            // result bank. Explicit wrapped float policies consume these
            // targets directly. `save_err` on the tuple stays `0i32`.
            "elidable"
            | "elidable_cannot_raise"
            | "elidable_or_memerror"
            | "dont_look_inside"
            | "dont_look_inside_cannot_raise"
            | "jit_may_force"
            | "jit_loop_invariant" => quote! {
                (#UNSUPPORTED, std::ptr::null(), #trace_addr, #concrete_addr, std::ptr::null(), 0i32)
            },
            _ => unsupported,
        }),
        HelperCallKind::Unsupported => Ok(unsupported),
    }
}

fn emit_helper_policy_fn(
    path: &Path,
    vis: &syn::Visibility,
    body: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let helper_name = helper_policy_fn_name(path)?;
    // `__majit_call_policy_*` visibility follows the user fn so
    // external integration tests can read the returned tuple's
    // trace_target / concrete_target function pointers for PyPy
    // `getfunctionptr` parity verification.  The arity is deliberately
    // not restated here: the signature below is the only statement of it
    // that cannot go stale, and this comment read "4-tuple" while the
    // signature returned six — an arity read off prose rather than off
    // the type is how a caller ends up destructuring the wrong shape.
    // The trailing `i32` is `0i32` (`RFFI_ERR_NONE`). `save_err` for a
    // GIL-releasing external lives on `EffectInfo.call_release_gil_target`.
    Ok(quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        #vis fn #helper_name() -> (u8, *const (), *const (), *const (), *const (), i32) {
            #body
        }
    })
}

/// Parsed contents of `#[jit_driver(greens = [...], reds = [...])]`.
struct JitDriverArgs {
    greens: Vec<Ident>,
    reds: Vec<Ident>,
    virtualizable: Option<Ident>,
}

/// Parse a bracketed list of identifiers: `[a, b, c]`.
fn parse_ident_list(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    syn::bracketed!(content in input);
    let idents = content.parse_terminated(Ident::parse, Token![,])?;
    Ok(idents.into_iter().collect())
}

impl Parse for JitDriverArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut greens = None;
        let mut reds = None;
        let mut virtualizable = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "greens" => {
                    if greens.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `greens`"));
                    }
                    greens = Some(parse_ident_list(input)?);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_ident_list(input)?);
                }
                "virtualizable" => {
                    if virtualizable.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `virtualizable`"));
                    }
                    virtualizable = Some(input.parse::<Ident>()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_driver parameter: `{other}`"),
                    ));
                }
            }

            // Consume optional trailing comma between greens and reds
            let _ = input.parse::<Token![,]>();
        }

        let greens = greens
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `greens`"))?;
        let reds =
            reds.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `reds`"))?;

        Ok(JitDriverArgs {
            greens,
            reds,
            virtualizable,
        })
    }
}

/// Mark a struct as a JIT driver configuration.
///
/// Usage:
/// ```ignore
/// #[majit::jit_driver(
///     greens = [next_instr, pycode],
///     reds = [frame, ec],
/// )]
/// struct MyJitDriver;
/// ```
///
/// Generates an `impl` block with associated constants describing the green
/// and red variable names, their counts, and the total number of JIT
/// variables.
#[proc_macro_attribute]
pub fn jit_driver(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JitDriverArgs);

    let input: syn::DeriveInput = match syn::parse(item) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let struct_name = &input.ident;
    let virtualizable = args.virtualizable.clone();

    let green_strs: Vec<String> = args.greens.iter().map(|id| id.to_string()).collect();
    let red_strs: Vec<String> = args.reds.iter().map(|id| id.to_string()).collect();
    let greens_joined = green_strs.join(", ");
    let reds_joined = red_strs.join(", ");

    let num_greens = green_strs.len();
    let num_reds = red_strs.len();
    let num_vars = num_greens + num_reds;

    let mut seen = std::collections::HashSet::new();
    for green in &args.greens {
        if !seen.insert(green.to_string()) {
            return syn::Error::new(green.span(), "duplicate variable in `greens`")
                .to_compile_error()
                .into();
        }
    }
    for red in &args.reds {
        if !seen.insert(red.to_string()) {
            return syn::Error::new(red.span(), "green/red variables must be distinct")
                .to_compile_error()
                .into();
        }
    }
    if let Some(virtualizable) = &virtualizable
        && !args.reds.iter().any(|red| red == virtualizable)
    {
        return syn::Error::new(
            virtualizable.span(),
            "`virtualizable` must name one of the red variables",
        )
        .to_compile_error()
        .into();
    }

    let doc = format!("JIT driver: greens=[{greens_joined}], reds=[{reds_joined}]");

    let attrs = &input.attrs;
    let vis = &input.vis;
    let generics = &input.generics;
    let data = &input.data;

    // Re-emit the struct with doc annotation, then add the impl block.
    let struct_token = match data {
        syn::Data::Struct(s) => {
            let fields = &s.fields;
            let semi = &s.semi_token;
            quote! {
                #(#attrs)*
                #[doc = #doc]
                #vis struct #struct_name #generics #fields #semi
            }
        }
        _ => {
            return syn::Error::new_spanned(&input, "jit_driver can only be applied to structs")
                .to_compile_error()
                .into();
        }
    };

    let virtualizable_value = if let Some(virtualizable) = &virtualizable {
        quote! { Some(stringify!(#virtualizable)) }
    } else {
        quote! { None }
    };

    let expanded = quote! {
        #struct_token

        impl #generics #struct_name #generics {
            /// Green variable names.
            pub const GREENS: &'static [&'static str] = &[#(#green_strs),*];
            /// Red variable names.
            pub const REDS: &'static [&'static str] = &[#(#red_strs),*];
            /// Total number of JIT variables.
            pub const NUM_VARS: usize = #num_vars;
            /// Number of green variables.
            pub const NUM_GREENS: usize = #num_greens;
            /// Number of red variables.
            pub const NUM_REDS: usize = #num_reds;
            /// Name of the virtualizable red variable, if any.
            pub const VIRTUALIZABLE: Option<&'static str> = #virtualizable_value;

            pub fn descriptor(
                green_types: &[majit_ir::Type],
                red_types: &[majit_ir::Type],
            ) -> Result<majit_metainterp::JitDriverStaticData, &'static str> {
                if green_types.len() != Self::NUM_GREENS {
                    return Err("wrong number of green variable types");
                }
                if red_types.len() != Self::NUM_REDS {
                    return Err("wrong number of red variable types");
                }

                let greens = Self::GREENS
                    .iter()
                    .zip(green_types.iter().copied())
                    .map(|(name, tp)| (*name, tp))
                    .collect::<Vec<_>>();
                let reds = Self::REDS
                    .iter()
                    .zip(red_types.iter().copied())
                    .map(|(name, tp)| (*name, tp))
                    .collect::<Vec<_>>();
                let descriptor = majit_metainterp::JitDriverStaticData::with_virtualizable(
                    greens,
                    reds,
                    Self::VIRTUALIZABLE,
                );
                if let Some(virtualizable) = descriptor.virtualizable() {
                    if virtualizable.tp != majit_ir::Type::Ref {
                        return Err("virtualizable red must have Ref type");
                    }
                }
                Ok(descriptor)
            }

            pub fn green_key(values: &[i64]) -> Result<majit_ir::GreenKey, &'static str> {
                if values.len() != Self::NUM_GREENS {
                    return Err("wrong number of green key values");
                }
                Ok(majit_ir::GreenKey::new(values.to_vec()))
            }
        }

        impl #generics majit_metainterp::DeclarativeJitDriver for #struct_name #generics {
            const GREENS: &'static [&'static str] = <Self>::GREENS;
            const REDS: &'static [&'static str] = <Self>::REDS;
            const NUM_VARS: usize = <Self>::NUM_VARS;
            const NUM_GREENS: usize = <Self>::NUM_GREENS;
            const NUM_REDS: usize = <Self>::NUM_REDS;
            const VIRTUALIZABLE: Option<&'static str> = <Self>::VIRTUALIZABLE;

            fn descriptor(
                green_types: &[majit_ir::Type],
                red_types: &[majit_ir::Type],
            ) -> Result<majit_metainterp::JitDriverStaticData, &'static str> {
                <Self>::descriptor(green_types, red_types)
            }

            fn green_key(values: &[i64]) -> Result<majit_ir::GreenKey, &'static str> {
                <Self>::green_key(values)
            }
        }
    };

    expanded.into()
}

/// Mark a function as elidable (pure / constant-foldable).
///
/// The JIT can eliminate calls to this function when all arguments are constants.
/// `rlib/jit.py elidable` sets `_elidable_function_ = True` and nothing else;
/// the flag travels here as the marker const `rpython_attribute_const_for`
/// emits, which `front/llbc_hints.rs` harvests from the extracted LLBC, and the
/// constant fold reaches the separate `__majit_call_target_*` trampoline. None
/// of that is a property of this function's codegen.
///
/// The backend inliner it leaves free is nonetheless a much narrower one than
/// LLVM's, and that is why `#[inline(never)]` stays below. `auto_inlining`
/// stops a graph at `weight >= threshold` (`backendopt/inline.py`) with
/// `weight = 0.9999 * measure_median_execution_cost(graph) +
/// static_instruction_count(graph)` (`:539 inlining_heuristic`, a hard reject
/// at `count >= 200`), counting roughly one per lltype operation
/// (`:479 OP_WEIGHTS`).  A straight-line graph is its own median path, so that
/// weight is about twice its op count and the default
/// `DEFL_INLINE_THRESHOLD = 32.4` buys on the order of sixteen operations —
/// described at
/// `config/translationoption.py` as "just enough to inline
/// add__Int_Int() and just small enough to prevent inlining of some rlist
/// functions". Elidable bodies are not that size: of the 126 non-test sites
/// this expansion covers, 54 are in `descroperation.rs` and 25 in
/// `longobject.rs`. Dropping the attribute hands them to a cost model that
/// accepts them, which is not what leaving upstream's inliner free would have
/// done.
///
/// Measurement offers nothing to weigh against that. Throughput was flat
/// (1.0070 / 1.0000 on the shadow-stack probes, 1.0000 on a pure integer
/// control), while the bodies inlining into the eval loop cost frame size:
/// reachable Python recursion depth fell 167700 to 159700, read out through
/// the SP-based `stack_check.rs current_sp()`.
///
/// `#[elidable]` is the conservative `EF_ELIDABLE_CAN_RAISE` form
/// (`rpython/jit/codewriter/effectinfo.py:21`), matching `call.py:297
/// getcalldescr` where `_canraise(op) == True`.  Use
/// `#[elidable_cannot_raise]` / `#[elidable_or_memerror]` for the
/// other two branches of `call.py`'s 3-way pick.
#[proc_macro_attribute]
pub fn elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_elidable_attribute(item, "elidable")
}

/// Deprecated alias for `#[elidable]`.
///
/// `rlib/jit.py`:
/// ```python
/// def purefunction(*args, **kwargs):
///     """Deprecated, use elidable instead."""
///     return elidable(*args, **kwargs)
/// ```
///
/// Pyre's alias forwards to `expand_elidable_attribute` with the
/// canonical `"elidable"` attr_name so the emitted `_elidable_function_
/// <NAME>` const + policy fn match the `@elidable` path verbatim.
#[proc_macro_attribute]
pub fn purefunction(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_elidable_attribute(item, "elidable")
}

/// `#[elidable_cannot_raise]` — `call.py getcalldescr`'s
/// `else` branch (`_canraise(op) == False`).  Maps to
/// `EF_ELIDABLE_CANNOT_RAISE` (`effectinfo.py:17`).  The canonical
/// walker (`pyjitpl.py do_residual_call`) records `CALL_PURE_*`
/// without the trailing `GUARD_NO_EXCEPTION` because
/// `effectinfo.check_can_raise(False)` (`effectinfo.py`) is false
/// for `extraeffect == 0`.
#[proc_macro_attribute]
pub fn elidable_cannot_raise(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_elidable_attribute(item, "elidable_cannot_raise")
}

/// `#[elidable_or_memerror]` — `call.py getcalldescr`'s
/// `cr == "mem"` branch.  Maps to `EF_ELIDABLE_OR_MEMORYERROR`
/// (`effectinfo.py:20`).  Same dispatch as `#[elidable]` (the
/// trailing `GUARD_NO_EXCEPTION` is recorded — `check_can_raise(False)`
/// is true for `extraeffect == 3`) but distinguishes memory-only
/// failure modes for the optimizer.
#[proc_macro_attribute]
pub fn elidable_or_memerror(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_elidable_attribute(item, "elidable_or_memerror")
}

fn expand_elidable_attribute(item: TokenStream, attr_name: &str) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func, true, None, attr_name, &[], false) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        vis,
        match helper_policy_tokens_for_fn(
            &func,
            attr_name,
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };
    let rpython_attribute_const = rpython_attribute_const_for(attr_name, sig, vis);
    let body_markers = rpython_attribute_body_markers(attr_name, sig);
    let identity = icf_identity_tokens(&sig.ident);

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_ELIDABLE: bool = true;
            #body_markers
            #identity
            #block
        }

        #call_target_fn
        #policy_fn
        #rpython_attribute_const
    };

    expanded.into()
}

/// Mark a function as opaque to the tracer.
///
/// The JIT will not trace into this function; it will be called as a black box.
/// `rlib/jit.py @dont_look_inside` — sets `_jit_look_inside_ = False`
/// (line 139) and nothing else.  This expansion carries that flag as the
/// `_jit_look_inside_` marker const `rpython_attribute_const_for` emits next
/// to the function; `front/llbc_hints.rs` harvests it out of the extracted LLBC
/// and `front/mir.rs` turns it into the residual-call decision.  The policy is
/// therefore a property of the marker, not of the function's codegen.
#[proc_macro_attribute]
pub fn dont_look_inside(attr: TokenStream, item: TokenStream) -> TokenStream {
    let word_enums = match parse_word_enums(attr) {
        Ok(word_enums) => word_enums,
        Err(err) => return err,
    };
    expand_dont_look_inside_attribute(item, "dont_look_inside", &word_enums)
}

/// Make sure the JIT traces inside the decorated function, even if
/// the rest of the module is not visible to the JIT.
///
/// `rlib/jit.py @look_inside` — sets `_jit_look_inside_ =
/// True` (line 148).  The RPython body also issues a deprecation
/// warning (line 147); pyre omits the warning because Rust callers
/// pick the attribute at compile time rather than at import time.
///
/// Unlike `#[dont_look_inside]`, this attribute does NOT emit a
/// call-target wrapper or policy fn — it's a tracing override, not a
/// residual-call surface declaration.
#[proc_macro_attribute]
pub fn look_inside(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let rpython_attribute_const = rpython_attribute_const_for("look_inside", sig, vis);
    let body_markers = rpython_attribute_body_markers("look_inside", sig);

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_LOOK_INSIDE: bool = true;
            #body_markers
            #block
        }

        #rpython_attribute_const
    };

    expanded.into()
}

/// `#[dont_look_inside_cannot_raise]` — non-elidable opaque helper that
/// the user statically guarantees does not raise.  Maps to
/// `EF_CANNOT_RAISE` (`call.py getcalldescr`'s non-elidable `else`
/// branch), so the recording walker skips the trailing `-live-` marker
/// and the produced calldescr's `EffectInfo` matches PyPy's
/// `cannot_raise_effect_info()`.
///
/// Use when `#[dont_look_inside]` is parity-conservative: the function
/// is opaque to the tracer (RPython annotation analysis would mark it
/// as `EF_CANNOT_RAISE`) while pyre's conservative analyzer cannot prove
/// that result. This attribute records the user's exception-effect
/// assertion for both macro-generated and graph-pipeline JitCode.
#[proc_macro_attribute]
pub fn dont_look_inside_cannot_raise(attr: TokenStream, item: TokenStream) -> TokenStream {
    let word_enums = match parse_word_enums(attr) {
        Ok(word_enums) => word_enums,
        Err(err) => return err,
    };
    expand_dont_look_inside_attribute(item, "dont_look_inside_cannot_raise", &word_enums)
}

/// `word_enums(CompareOp, CallMode)` lists fieldless enums a residual call
/// passes as their discriminant word.
struct WordEnumsArg(Vec<Path>);

impl Parse for WordEnumsArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self(Vec::new()));
        }
        let name: Ident = input.parse()?;
        if name != "word_enums" {
            return Err(syn::Error::new(name.span(), "expected word_enums(...)"));
        }
        let content;
        parenthesized!(content in input);
        let paths = syn::punctuated::Punctuated::<Path, Token![,]>::parse_terminated(&content)?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after word_enums"));
        }
        Ok(Self(paths.into_iter().collect()))
    }
}

fn parse_word_enums(attr: TokenStream) -> Result<Vec<Path>, TokenStream> {
    match syn::parse::<WordEnumsArg>(attr) {
        Ok(parsed) => Ok(parsed.0),
        Err(err) => Err(err.to_compile_error().into()),
    }
}

fn expand_dont_look_inside_attribute(
    item: TokenStream,
    attr_name: &str,
    word_enums: &[Path],
) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func, true, None, attr_name, word_enums, false) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        vis,
        match helper_policy_tokens_for_fn(
            &func,
            attr_name,
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };
    let rpython_attribute_const = rpython_attribute_const_for(attr_name, sig, vis);
    let body_markers = rpython_attribute_body_markers(attr_name, sig);
    let identity = icf_identity_tokens(&sig.ident);

    // No-op inline-prebuild fn.  A residual (opaque) helper has no inline
    // body, hence no per-marker `-live-` triples to pre-register.  But a
    // value-position call to this helper under `auto_calls` takes the
    // `CallPolicySpec::Infer` arm in `lower_call_value`, which
    // unconditionally queues `inline_prebuild_path(func)(__asm)` into the
    // parent's `__prebuild_jitcode_liveness_*` (the path is always
    // constructible).  Emit a no-op `__majit_inline_jitcode_<name>_prebuild`
    // so that reference resolves and registers zero triples — matching the
    // "no triples to register" intent documented at
    // `jitcode_lower::lower_value::lower_call_value`'s Infer arm.  Mirrors
    // the `#[jit_inline]` prebuild fn shape (`lib.rs` jit_inline expansion),
    // with an empty body in place of the inline helper's triples.
    //
    // The assembler parameter is generic rather than the concrete
    // `majit_metainterp::Assembler`: unlike `#[jit_inline]`/`#[jit_interp]`
    // (used only inside the JIT crate), `#[dont_look_inside]` annotates
    // functions in crates that have no `majit_metainterp` dependency at all
    // (e.g. `pyre-object`), so naming the type here fails to resolve. The fn
    // is only ever CALLED from the parent `__prebuild_jitcode_liveness_*`
    // (codegen_trace.rs) where `__asm: &mut Assembler`, so the type param is
    // inferred to `Assembler` at that single JIT-crate call site.
    let prebuild_name = format_ident!("__majit_inline_jitcode_{}_prebuild", sig.ident);

    // No `#[inline(never)]`: the tracer's view of this function does not depend
    // on how the host backend codegens it.  `@dont_look_inside` sets one flag
    // (`rlib/jit.py _jit_look_inside_ = False`) and leaves the C backend's
    // inliner free to inline the body; the decision is read off the
    // `_jit_look_inside_` marker const below, which `front/llbc_hints.rs`
    // harvests from the extracted LLBC.  That LLBC cannot be reshaped by an
    // inlining attribute either — Charon disables the MIR optimizations
    // (`charon cargo --mir`: "Charon disables all the optimizations it can"),
    // so no `dont_look_inside` body is folded into a traced caller regardless.
    // Residual call targets are likewise unaffected: they resolve through the
    // function-item coercions `pyre-interpreter/src/jit_fnaddr.rs` writes by
    // hand, not through a linker symbol. [`majit_ir::icf_identity`] still
    // sits in the body so LLVM MergeFunctions cannot fold two published
    // helpers that would otherwise be byte-identical (`drain_list_append`).
    //
    // `#[elidable]` keeps its `#[inline(never)]` and this family drops it, and
    // the split follows upstream's inlining budget rather than the attribute.
    // `DEFL_INLINE_THRESHOLD = 32.4` (`config/translationoption.py`) is
    // "just enough to inline add__Int_Int()" — on the order of sixteen lltype
    // operations, since `inline.py inlining_heuristic` charges a
    // straight-line graph both its static count and its median execution cost.
    // This family sits inside that budget —
    // `gc_roots::shadow_stack_len` is one call, `pyopcode::label_arg_to_usize`
    // one field read — so leaving them to the host inliner is what upstream's
    // free inliner would also have done.  The elidable bodies do not, which is
    // why the same edit measured worse there.
    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OPAQUE: bool = true;
            #body_markers
            #identity
            #block
        }

        #[doc(hidden)]
        #[allow(non_snake_case, unused_variables)]
        #vis fn #prebuild_name<__MajitAsm>(__asm: &mut __MajitAsm) {}

        #call_target_fn
        #policy_fn
        #rpython_attribute_const
    };

    expanded.into()
}

fn expand_call_surface_attr(attr_name: &str, marker_name: &str, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let marker = format_ident!("{marker_name}");
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func, false, None, attr_name, &[], false) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        vis,
        match helper_policy_tokens_for_fn(
            &func,
            attr_name,
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let rpython_attribute_const = rpython_attribute_const_for(attr_name, sig, vis);
    let body_markers = rpython_attribute_body_markers(attr_name, sig);
    let identity = icf_identity_tokens(&sig.ident);

    // `jit_may_force` and `loop_invariant` keep `#[inline(never)]` for the
    // reason recorded on [`elidable`]: their bodies are past the inlining
    // budget the free RPython inliner would have applied, and dropping it
    // measured flat to worse.
    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const #marker: bool = true;
            #body_markers
            #identity
            #block
        }

        #call_target_fn
        #policy_fn
        #rpython_attribute_const
    };

    expanded.into()
}

/// Mark a function as a may-force call surface.
#[proc_macro_attribute]
pub fn jit_may_force(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_may_force", "_MAJIT_MAY_FORCE", item)
}

/// Mark a function as a loop-invariant call surface.
#[proc_macro_attribute]
pub fn jit_loop_invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_loop_invariant", "_MAJIT_LOOP_INVARIANT", item)
}

/// Mark a function as loop-invariant.
///
/// RPython name parity alias for `#[jit_loop_invariant]`.
///
/// rlib/jit.py — `@loop_invariant`: describes a function with no argument
/// that returns an object that is always the same in a loop.
/// Implies `@dont_look_inside`.
#[proc_macro_attribute]
pub fn loop_invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_loop_invariant", "_MAJIT_LOOP_INVARIANT", item)
}

/// Mark struct fields whose value never mutates after construction.
///
/// RPython parity: `_immutable_fields_ = ['field_a', 'field_b']` on the
/// class body (`rpython/rlib/jit.py` -> `rpython/rtyper/rclass.py`).
/// The annotator pipes the list through to `cpu.fielddescrof()` which
/// stores `is_pure=True` on the field descr, allowing the JIT to fold
/// reads of those fields into constants once the receiver is known.
///
/// Usage:
/// ```ignore
/// #[majit_macros::jit_immutable_fields("pools", "size?", "digits[*]")]
/// pub struct Storage {
///     pub pools: [*mut Stack; STORAGE_COUNT],
///     ...
/// }
/// ```
///
/// Entries carry the `rclass.py _parse_field_list` suffixes
/// (`?` quasi-immutable, `[*]` immutable array); a bare identifier is
/// accepted as shorthand for a plain immutable field.
///
/// The struct definition itself is left untouched.  Like the other
/// `majit_macros` attributes, the attribute is consumed at expansion
/// time and does not survive in Charon's `attr_info`, so the macro
/// leaves a `#[doc(hidden)]` marker const
/// `_immutable_fields_<Struct>: &str` next to the struct holding the
/// comma-joined entry list.  Charon extracts it into `global_decls`,
/// and `majit-translate::front::llbc_hints::
/// harvest_immutable_fields_from_llbcs` reads it back into
/// `SemanticProgram.immutable_fields` — the analog of RPython's
/// translator reading `cls._immutable_fields_` off the class object.
///
/// The proc-macro front end has no such extraction step, so the same
/// list is published a second way: an inherent associated const
/// `__MAJIT_IMMUTABLE_FIELDS`, shadowing the empty default on
/// `majit_metainterp::MajitImmutableFields`.  `jitcode_lower` reads it
/// at each `register_struct_layout` emit site.  Both publications carry
/// the entry list verbatim and neither splits it, so the suffix grammar
/// stays in one place (`majit_translate::model::ImmutableRank::parse`).
#[proc_macro_attribute]
pub fn jit_immutable_fields(attr: TokenStream, item: TokenStream) -> TokenStream {
    let entries = parse_macro_input!(attr as ImmutableFieldList);
    let item_struct = parse_macro_input!(item as syn::ItemStruct);
    let vis = &item_struct.vis;
    let const_name = format_ident!("_immutable_fields_{}", item_struct.ident);
    // `rclass.py _parse_field_list` splits the declared list into
    // (name, rank) pairs; the marker carries the list verbatim and the
    // harvester does the splitting, so the suffix grammar stays in one
    // place (`majit_translate::model::ImmutableRank::parse`).
    let joined = entries.0.join(",");
    // The same declaration, published a second way for the proc-macro front
    // end.  That path has no Charon extraction to harvest the marker const, so
    // it reads the ranks straight off the type at the layout emit site;
    // `majit_metainterp::MajitImmutableFields` supplies the empty default this
    // shadows, which is what lets a site name a struct that declared nothing.
    let ident = &item_struct.ident;
    let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();
    quote! {
        #item_struct

        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        #vis const #const_name: &'static str = #joined;

        impl #impl_generics #ident #ty_generics #where_clause {
            #[doc(hidden)]
            #[allow(non_upper_case_globals, dead_code)]
            pub const __MAJIT_IMMUTABLE_FIELDS: &'static str = #joined;
        }
    }
    .into()
}

/// `_immutable_fields_` entry list as written in the attribute —
/// either string literals (`"size?"`, `"digits[*]"`) or bare
/// identifiers for plain immutable fields.
struct ImmutableFieldList(Vec<String>);

impl Parse for ImmutableFieldList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut entries = Vec::new();
        while !input.is_empty() {
            if input.peek(syn::LitStr) {
                entries.push(input.parse::<syn::LitStr>()?.value());
            } else {
                entries.push(input.parse::<Ident>()?.to_string());
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        if entries.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[jit_immutable_fields] needs at least one field entry",
            ));
        }
        Ok(Self(entries))
    }
}

/// Mark a method (or free function) as elidable / pure.
///
/// RPython parity: `@jit.elidable` (`rpython/rlib/jit.py`). The JIT
/// can fold calls to this function once all arguments are constant.
///
/// Companion to the existing `#[elidable]` attribute, with two
/// differences:
///   1. Works on `ImplItemFn` as well as free functions — the existing
///      `#[elidable]` parses as `ItemFn` and rejects `impl` methods.
///   2. Pure pass-through, so it does not synthesize trampolines /
///      helper policy tokens. Methods on `&self` / `&mut self` are
///      called by the codewriter via `CallTarget::method` path
///      resolution, which doesn't need the trampoline that free
///      functions get.
///
/// `#[jit_elidable]` emits the same hidden `_elidable_function_<NAME>`
/// const as `#[elidable]` (see `rpython_attribute_const_for`); the
/// ullbc hint harvester (`majit-translate::front::llbc_hints`) maps that
/// const to the `"elidable"` function hint, which `mark_elidable`
/// consumes downstream.
#[proc_macro_attribute]
pub fn jit_elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_ts = proc_macro2::TokenStream::from(item);
    if let Ok(func) = syn::parse2::<ItemFn>(item_ts.clone()) {
        // `syn::ItemFn` accepts a `self` receiver even though such a function
        // is only legal inside an impl, so receiver methods arrive through
        // this branch before the `ImplItemFn` fallback below. They cannot emit
        // a sibling associated const in trait impls; leave a body-local marker
        // that Charon promotes under the method path instead.
        if func.sig.receiver().is_some() {
            let attrs = &func.attrs;
            let vis = &func.vis;
            let sig = &func.sig;
            let block = &func.block;
            let body_markers = rpython_attribute_body_markers("jit_elidable", sig);
            return quote! {
                #(#attrs)*
                #vis #sig {
                    #body_markers
                    #block
                }
            }
            .into();
        }
        let attrs = &func.attrs;
        let vis = &func.vis;
        let sig = &func.sig;
        let block = &func.block;
        let body_markers = rpython_attribute_body_markers("jit_elidable", sig);
        let rpython_attribute_const = rpython_attribute_const_for("jit_elidable", sig, vis);
        return quote! {
            #(#attrs)*
            #vis #sig {
                #body_markers
                #block
            }
            #rpython_attribute_const
        }
        .into();
    }
    if let Ok(method) = syn::parse2::<syn::ImplItemFn>(item_ts.clone()) {
        // An impl method cannot emit the module-level siblings used by
        // `expand_elidable_attribute`: trait impls reject extra associated
        // items, and the residual-call trampoline cannot live inside an impl.
        // Keep the method ABI untouched and leave the RPython attribute as a
        // body-local const instead. Charon promotes local consts to
        // `global_decls` under the method path; `front::llbc_hints` recognizes
        // that nested spelling and attaches `elidable` to the parent method.
        let attrs = &method.attrs;
        let vis = &method.vis;
        let defaultness = &method.defaultness;
        let sig = &method.sig;
        let block = &method.block;
        let body_markers = rpython_attribute_body_markers("jit_elidable", sig);
        return quote! {
            #(#attrs)*
            #vis #defaultness #sig {
                #body_markers
                #block
            }
        }
        .into();
    }
    syn::Error::new_spanned(
        item_ts,
        "#[jit_elidable] supports free functions and impl methods",
    )
    .to_compile_error()
    .into()
}

/// Force backend-optimizer inlining, matching
/// `rpython.rlib.objectmodel.always_inline` / a function's
/// `_always_inline_ = True` attribute.
///
/// This is a pure metadata decorator: it preserves the Rust ABI and emits a
/// body-local marker so it works for inherent and trait methods as well as
/// free functions. Charon promotes the marker under the owning function path;
/// `front::llbc_hints` restores the `always_inline` hint and the flowspace
/// adapter places it on `graph.func._always_inline_`.
#[proc_macro_attribute]
pub fn always_inline(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_ts = proc_macro2::TokenStream::from(item);
    if let Ok(func) = syn::parse2::<ItemFn>(item_ts.clone()) {
        let attrs = &func.attrs;
        let vis = &func.vis;
        let sig = &func.sig;
        let block = &func.block;
        let marker = format_ident!("_always_inline_{}", sig.ident);
        return quote! {
            #(#attrs)*
            #vis #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    if let Ok(method) = syn::parse2::<syn::ImplItemFn>(item_ts.clone()) {
        let attrs = &method.attrs;
        let vis = &method.vis;
        let defaultness = &method.defaultness;
        let sig = &method.sig;
        let block = &method.block;
        let marker = format_ident!("_always_inline_{}", sig.ident);
        return quote! {
            #(#attrs)*
            #vis #defaultness #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    syn::Error::new_spanned(
        item_ts,
        "#[always_inline] supports free functions and impl methods",
    )
    .to_compile_error()
    .into()
}

/// Request best-effort backend-optimizer inlining, matching a function's
/// `_always_inline_ = 'try'` attribute.
///
/// RPython treats this value as truthy when selecting zero-cost inline
/// candidates, but unlike literal `True` it does not turn a failed inline into
/// `CannotInline` (`translator/backendopt/inline.py`).
#[proc_macro_attribute]
pub fn always_inline_try(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_ts = proc_macro2::TokenStream::from(item);
    if let Ok(func) = syn::parse2::<ItemFn>(item_ts.clone()) {
        let attrs = &func.attrs;
        let vis = &func.vis;
        let sig = &func.sig;
        let block = &func.block;
        let marker = format_ident!("_always_inline_try_{}", sig.ident);
        return quote! {
            #(#attrs)*
            #vis #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    if let Ok(method) = syn::parse2::<syn::ImplItemFn>(item_ts.clone()) {
        let attrs = &method.attrs;
        let vis = &method.vis;
        let defaultness = &method.defaultness;
        let sig = &method.sig;
        let block = &method.block;
        let marker = format_ident!("_always_inline_try_{}", sig.ident);
        return quote! {
            #(#attrs)*
            #vis #defaultness #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    syn::Error::new_spanned(
        item_ts,
        "#[always_inline_try] supports free functions and impl methods",
    )
    .to_compile_error()
    .into()
}

/// Mark a host-only function as ineligible for RPython flow-graph building.
///
/// RPython parity: `rpython.rlib.objectmodel.not_rpython` sets
/// `func._not_rpython_ = True`; `flowspace/objspace.py:21-22`
/// `assert_rpythonic()` rejects the function before constructing a graph.
///
/// This is metadata-only and preserves the native Rust ABI. A body-local
/// `_not_rpython_<NAME>` marker works for both free functions and impl methods;
/// Charon promotes it into `global_decls`, `front::llbc_hints` restores the
/// marker, and the MIR frontend excludes the function before lowering its
/// body.
#[proc_macro_attribute]
pub fn not_rpython(_attr: TokenStream, item: TokenStream) -> TokenStream {
    host_function_marker(item, "_not_rpython_", "not_rpython")
}

/// `objectmodel.py specialize.memo`: attach the source callable's
/// `_annspecialcase_ = 'specialize:memo'` attribute without changing its ABI.
/// The translator must supply a native host evaluator and finite prebuilt
/// arguments; this marker does not residualize or execute the function.
#[proc_macro_attribute]
pub fn specialize_memo(_attr: TokenStream, item: TokenStream) -> TokenStream {
    host_function_marker(item, "_annspecialcase_memo_", "specialize_memo")
}

/// `objectmodel.py specialize.call_location`: attach
/// `_annspecialcase_ = 'specialize:call_location'` without changing ABI.
/// One specialized graph per call site (`FunctionDesc.cachedgraph(op)`).
#[proc_macro_attribute]
pub fn specialize_call_location(_attr: TokenStream, item: TokenStream) -> TokenStream {
    host_function_marker(
        item,
        "_annspecialcase_call_location_",
        "specialize_call_location",
    )
}

fn host_function_marker(item: TokenStream, prefix: &str, attribute: &str) -> TokenStream {
    let item_ts = proc_macro2::TokenStream::from(item);
    if let Ok(func) = syn::parse2::<ItemFn>(item_ts.clone()) {
        let attrs = &func.attrs;
        let vis = &func.vis;
        let sig = &func.sig;
        let block = &func.block;
        let marker = format_ident!("{}{}", prefix, sig.ident);
        return quote! {
            #(#attrs)*
            #vis #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    if let Ok(method) = syn::parse2::<syn::ImplItemFn>(item_ts.clone()) {
        let attrs = &method.attrs;
        let vis = &method.vis;
        let defaultness = &method.defaultness;
        let sig = &method.sig;
        let block = &method.block;
        let marker = format_ident!("{}{}", prefix, sig.ident);
        return quote! {
            #(#attrs)*
            #vis #defaultness #sig {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #marker: bool = true;
                #block
            }
        }
        .into();
    }
    syn::Error::new_spanned(
        item_ts,
        format!("#[{attribute}] supports free functions and impl methods"),
    )
    .to_compile_error()
    .into()
}

/// JIT can safely unroll loops in this function and this will
/// not lead to code explosion.
///
/// rlib/jit.py — `@unroll_safe`.
/// Cannot be combined with `#[elidable]` or `#[dont_look_inside]`.
#[proc_macro_attribute]
pub fn unroll_safe(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    // RPython attribute-name parity: `rlib/jit.py func._jit_unroll_safe_
    // = True`.  `front/llbc_hints.rs` harvests `_jit_unroll_safe_<NAME>` and
    // nothing else; the marker this used to leave in the body was spelled
    // `_MAJIT_UNROLL_SAFE`, carrying neither the prefix nor the function
    // name, so it was never read — and a method got no sibling const, so a
    // method got no hint at all.  `@unroll_safe` decorates methods upstream
    // (`pyframe.py` `fast2locals`, `peekvalues`, `dropvalues`), which is
    // exactly the shape that was being dropped.
    //
    // The body-local spelling is what carries it, as for `jit_elidable`: a
    // trait impl rejects a foreign associated const, and an attribute macro
    // cannot tell an inherent impl from a trait impl.  Charon promotes a
    // body-local const under the method's own path, and
    // `llbc_hints::marker_path_to_fn_path` already resolves that spelling.
    // A free function additionally gets the module-level sibling, so
    // `rg _jit_unroll_safe_` still finds the parity counterpart there.
    let body_markers = rpython_attribute_body_markers("unroll_safe", sig);
    let unroll_safe_const = rpython_attribute_const_for("unroll_safe", sig, vis);

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #body_markers
            #block
        }

        #unroll_safe_const
    };

    expanded.into()
}

/// A decorator for a function with no return value.  It makes the
/// function call disappear from the jit traces. It is still called in
/// interpreted mode, and by the jit tracing and blackholing, but not
/// by the final assembler.
///
/// rlib/jit.py — `@not_in_trace`.
#[proc_macro_attribute]
pub fn not_in_trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    // RPython attribute-name parity: `rlib/jit.py func.oopspec =
    // "jit.not_in_trace()"`.  Emit a module-level `pub const
    // oopspec_<NAME>: &'static str` next to the wrapper so `rg oopspec_`
    // finds the parity counterpart in both pyre and PyPy.  Skip for
    // methods (`self`-receiver) — see `rpython_attribute_const_for`'s
    // receiver guard for the same reasoning.
    let oopspec_const = if sig.receiver().is_none() {
        let const_name = format_ident!("oopspec_{}", sig.ident);
        Some(quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals)]
            #vis const #const_name: &'static str = "jit.not_in_trace()";
        })
    } else {
        None
    };

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_NOT_IN_TRACE: bool = true;
            #block
        }

        #oopspec_const
    };

    expanded.into()
}

/// A decorator that promotes all arguments and then calls the supplied
/// elidable function.
///
/// rlib/jit.py — `@elidable_promote(promote_args='all')`.
///
/// Deprecated alias for `#[elidable_promote]`.
///
/// `rlib/jit.py`:
/// ```python
/// def purefunction_promote(*args, **kwargs):
///     """Deprecated, use elidable_promote instead."""
///     return elidable_promote(*args, **kwargs)
/// ```
#[proc_macro_attribute]
pub fn purefunction_promote(attr: TokenStream, item: TokenStream) -> TokenStream {
    elidable_promote(attr, item)
}

/// The decorated name **becomes** the promoting wrapper (RPython parity).
/// The original elidable body is renamed to a hidden `_orig_<name>_unlikely_name`.
///
/// Usage:
///   `#[elidable_promote]` — promote all arguments (default)
///   `#[elidable_promote(promote_args = "0,2")]` — promote args at indices 0, 2
#[proc_macro_attribute]
pub fn elidable_promote(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Parse promote_args from attribute
    let promote_args_str = if attr.is_empty() {
        "all".to_string()
    } else {
        let config = parse_macro_input!(attr as ElidablePromoteArgs);
        config.promote_args
    };

    let arg_count = func.sig.inputs.len();
    let promote_indices: Vec<usize> = if promote_args_str == "all" {
        (0..arg_count).collect()
    } else {
        promote_args_str
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let fn_name = &sig.ident;
    // rlib/jit.py:196 — _orig_func_unlikely_name
    let orig_name = format_ident!("_orig_{}_unlikely_name", fn_name);
    let output = &sig.output;

    // Collect param info (rlib/jit.py _get_args — includes self if method)
    let mut param_names: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut full_params: Vec<syn::FnArg> = Vec::new();
    let mut named_args: Vec<Ident> = Vec::new();
    let has_self = matches!(sig.inputs.first(), Some(FnArg::Receiver(_)));
    for arg in &sig.inputs {
        full_params.push(arg.clone());
        match arg {
            FnArg::Receiver(_) => {
                param_names.push(quote! { self });
            }
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = pat_ident.ident.clone();
                    param_names.push(quote! { #name });
                    named_args.push(name);
                }
            }
        }
    }

    // rlib/jit.py — promote each selected arg with both hints.
    // promote_indices index into full arg list (including self), matching _get_args.
    let promote_stmts: Vec<_> = promote_indices
        .iter()
        .filter_map(|&idx| {
            if has_self && idx == 0 {
                // rlib/jit.py — promote self by identity (guard_value on pointer)
                Some(quote! {
                    let _ = majit_ir::jit::promote(self as *const _ as usize);
                })
            } else {
                let named_idx = if has_self { idx - 1 } else { idx };
                named_args.get(named_idx).map(|name| {
                    quote! { let #name = majit_ir::jit::promote(#name); }
                })
            }
        })
        .collect();

    let call_args: Vec<_> = param_names.clone();
    let orig_call = if has_self {
        let method_args = call_args.iter().skip(1);
        quote! { self.#orig_name(#(#method_args),*) }
    } else {
        quote! { #orig_name(#(#call_args),*) }
    };

    // Build call_target/policy for the ORIGINAL elidable function
    let orig_sig = syn::Signature {
        ident: orig_name.clone(),
        ..sig.clone()
    };
    let orig_func = syn::ItemFn {
        sig: orig_sig,
        ..func.clone()
    };
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&orig_func, true, Some(fn_name), "elidable", &[], false) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_path = Path::from(orig_name.clone());
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        vis,
        match helper_policy_tokens_for_fn(
            &orig_func,
            "elidable",
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    // `rlib/jit.py elidable(func)` — the ORIGINAL `func` is what
    // receives `_elidable_function_ = True`; `result` (the returned
    // wrapper, see jit.py) does NOT carry the attribute.  In
    // pyre's layout `func` becomes the hidden `_orig_<NAME>_unlikely_
    // name` and the wrapper takes the decorated name, so the const
    // lives on the renamed original.  Emitted at the user-facing `vis`
    // (Rust adaptation: the renamed original is private `fn`, but
    // callers need to read the const through the wrapper's module).
    let orig_elidable_const = if !has_self {
        let const_name = format_ident!("_elidable_function_{}", orig_name);
        Some(quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals)]
            #vis const #const_name: bool = true;
        })
    } else {
        None
    };

    let orig_body_markers = rpython_attribute_body_markers("elidable", &orig_func.sig);

    let expanded = quote! {
        // rlib/jit.py — elidable(func); original body hidden
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_snake_case, non_upper_case_globals)]
        fn #orig_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_ELIDABLE: bool = true;
            #orig_body_markers
            #block
        }

        #call_target_fn
        #policy_fn

        // rlib/jit.py:188-200 — the decorated name IS the promoting wrapper
        #(#attrs)*
        #vis fn #fn_name(#(#full_params),*) #output {
            #(#promote_stmts)*
            #orig_call
        }

        #orig_elidable_const
    };

    expanded.into()
}

/// Parse helper for `#[elidable_promote(promote_args = "...")]`.
struct ElidablePromoteArgs {
    promote_args: String,
}

impl Parse for ElidablePromoteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        if key != "promote_args" {
            return Err(syn::Error::new(key.span(), "expected `promote_args`"));
        }
        input.parse::<Token![=]>()?;
        let value: syn::LitStr = input.parse()?;
        Ok(Self {
            promote_args: value.value(),
        })
    }
}

/// The JIT compiler won't look inside this decorated function,
/// but instead during translation, rewrites it according to the handler in
/// the codewriter/jtransform.
///
/// rlib/jit.py — `@oopspec(spec)`.
///
/// Usage: `#[oopspec("jit.isconstant(value)")]`
///
/// The spec string is stored as a hidden constant for the codewriter to discover.
#[proc_macro_attribute]
pub fn oopspec(attr: TokenStream, item: TokenStream) -> TokenStream {
    let spec: syn::LitStr = parse_macro_input!(attr as syn::LitStr);
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let spec_value = spec.value();
    // RPython attribute-name parity: `rlib/jit.py func.oopspec = spec`.
    // `harvest_hints_from_llbcs` keys a global whose leaf starts with
    // `oopspec_`. A free function can carry that const beside the item.
    // An inherent method cannot: a trait impl rejects a foreign associated
    // item (`rpython_attribute_const_for`). Charon promotes a body-local
    // const under `<method>::oopspec_<method>`, and `marker_path_to_fn_path`
    // binds that parent when it is the function path. Emit the body-local
    // const for every item, and keep the module-level const for free
    // functions so the existing external name stays visible.
    let body_name = format_ident!("oopspec_{}", sig.ident);
    // The function stays an ordinary callable one: a call jtransform does not
    // rewrite from its oopspec is a residual call to its address
    // (`getfunctionptr`).
    // A policy attribute stacked under this one emits the call target itself.
    let policy_attr_follows = func.attrs.iter().any(|attr| {
        let segments = &attr.path().segments;
        segments
            .first()
            .is_some_and(|seg| seg.ident == "majit_macros")
            || segments.last().is_some_and(|seg| {
                matches!(
                    seg.ident.to_string().as_str(),
                    "dont_look_inside"
                        | "dont_look_inside_cannot_raise"
                        | "elidable"
                        | "elidable_cannot_raise"
                        | "elidable_or_memerror"
                        | "purefunction"
                        | "look_inside_iff"
                )
            })
    });
    // ... or one already expanded above it left its marker const in the body.
    let policy_attr_expanded = func.block.stmts.iter().any(|stmt| {
        matches!(stmt, syn::Stmt::Item(syn::Item::Const(item))
            if item.ident.to_string().starts_with("_MAJIT_"))
    });
    let call_target_fn = if policy_attr_follows || policy_attr_expanded {
        None
    } else {
        match emit_helper_call_target_fn(&func, true, None, "oopspec", &[], false) {
            Ok(Some((_, _, tokens))) => Some(tokens),
            Ok(None) => None,
            Err(err) => return err.to_compile_error().into(),
        }
    };
    let oopspec_const = if sig.receiver().is_none() {
        let const_name = body_name.clone();
        Some(quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals)]
            #vis const #const_name: &'static str = #spec_value;
        })
    } else {
        None
    };
    let identity = icf_identity_tokens(&sig.ident);

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OOPSPEC: &str = #spec_value;
            #[doc(hidden)]
            #[allow(non_upper_case_globals, dead_code)]
            const #body_name: &'static str = #spec_value;
            #identity
            #block
        }

        #call_target_fn
        #oopspec_const
    };

    expanded.into()
}

/// Look inside (including unrolling loops) the target function, if and only if
/// `predicate(args)` returns true.
///
/// rlib/jit.py — `@look_inside_iff(predicate)`.
///
/// Generates three functions:
/// 1. `_orig_<name>` — original body, marked `#[unroll_safe]` (hidden)
/// 2. `<name>_trampoline` — `@dont_look_inside` wrapper calling _orig (hidden)
/// 3. `<name>` — dispatch wrapper (the public name):
///    `if !we_are_jitted() || predicate(args) { _orig(args) } else { trampoline(args) }`
///
/// Usage: `#[look_inside_iff(my_predicate)]`
/// where `my_predicate` has the same signature as the decorated function returning bool.
#[proc_macro_attribute]
pub fn look_inside_iff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let predicate_path: Path = parse_macro_input!(attr as Path);
    let func = parse_macro_input!(item as ItemFn);

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let fn_name = &sig.ident;
    let unsafety = &sig.unsafety;
    // rlib/jit.py — func = unroll_safe(func)
    let orig_name = format_ident!("_orig_{}", fn_name);
    // rlib/jit.py — trampoline.__name__ = func.__name__ + "_trampoline"
    let trampoline_name = format_ident!("{}_trampoline", fn_name);
    let output = &sig.output;

    // Collect parameter patterns and call argument expressions.
    // rlib/jit.py args = `_get_args(func)` — includes `self` if method.
    let mut full_params: Vec<syn::FnArg> = Vec::new();
    let mut call_args: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut has_receiver = false;
    for arg in &sig.inputs {
        full_params.push(arg.clone());
        match arg {
            FnArg::Receiver(_) => {
                // Forward `self` as-is (works for &self, &mut self, self, Box<Self>).
                has_receiver = true;
                call_args.push(quote! { self });
            }
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = &pat_ident.ident;
                    call_args.push(quote! { #name });
                }
            }
        }
    }
    // Sibling `_orig_*` / trampoline items sit in the same `impl` when
    // the decorated item is a method (`argument.py unpack`). A bare
    // `_orig_unpack(self)` looks up a free function; `Self::` is the
    // inherent-method call.
    let orig_call = if has_receiver {
        quote! { Self::#orig_name(#(#call_args),*) }
    } else {
        quote! { #orig_name(#(#call_args),*) }
    };
    let trampoline_call = if has_receiver {
        quote! { Self::#trampoline_name(#(#call_args),*) }
    } else {
        quote! { #trampoline_name(#(#call_args),*) }
    };

    // rlib/jit.py — func = unroll_safe(func); trampoline = dont_look_inside.
    // `front/llbc_hints` harvests these sibling / body-local markers the
    // same way `#[unroll_safe]` / `#[dont_look_inside]` do.
    let orig_unroll_marker = format_ident!("_jit_unroll_safe_{}", orig_name);
    let trampoline_opaque_marker = format_ident!("_jit_look_inside_{}", trampoline_name);

    // Residual CondCall (rlist.py `_ll_list_resize_ge` →
    // `jit.conditional_call(_ll_list_resize_hint_really, ...)`) needs the
    // same word-ABI entry `#[dont_look_inside]` emits. The public name is
    // the dispatch wrapper; the adapter calls that, matching
    // `getfunctionptr` of the decorated function. It is published to
    // `HELPER_FNADDRS` like every other residual entry.
    let call_target_fn =
        match emit_helper_call_target_fn(&func, true, None, "look_inside_iff", &[], false) {
            Ok(Some((_, _, tokens))) => Some(tokens),
            Ok(None) => None,
            Err(err) => return err.to_compile_error().into(),
        };

    let expanded = quote! {
        // rlib/jit.py — func = unroll_safe(func)
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #unsafety fn #orig_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(non_upper_case_globals, dead_code)]
            const #orig_unroll_marker: bool = true;
            #block
        }

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        const #orig_unroll_marker: bool = true;

        // rlib/jit.py — @dont_look_inside def trampoline(...): return func(...)
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #unsafety fn #trampoline_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(non_upper_case_globals, dead_code)]
            const #trampoline_opaque_marker: bool = false;
            #orig_call
        }

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        const #trampoline_opaque_marker: bool = false;

        // rlib/jit.py f — the decorated name becomes the dispatch wrapper
        // def f(*args):
        //     if not we_are_jitted() or predicate(*args):
        //         return func(*args)
        //     else:
        //         return trampoline(*args)
        #(#attrs)*
        #vis #unsafety fn #fn_name(#(#full_params),*) #output {
            if !majit_rlib::jit::we_are_jitted() || #predicate_path(#(#call_args),*) {
                #orig_call
            } else {
                #trampoline_call
            }
        }

        #call_target_fn
    };

    expanded.into()
}

/// Serialize a helper into a hidden `JitCode` builder.
///
/// This is the proc-macro side of RPython's `codewriter.py` helper serialization:
/// the original function stays callable by the interpreter, and the macro also
/// emits a hidden `__majit_inline_jitcode_*()` function that `#[jit_interp]`
/// can use when a call policy maps the helper to
/// `inline_int`/`inline_ref`/`inline_float`/`inline_void`.
///
/// Supports Int (i64/isize), Ref (usize/pointer), Float (f64), and void return
/// types. Parameters must be Int, Ref, or Float.
#[proc_macro_attribute]
pub fn jit_inline(attr: TokenStream, item: TokenStream) -> TokenStream {
    use jit_interp::jitcode_lower::InlineReturnKind;

    let args = parse_macro_input!(attr as JitInlineArgs);
    let func = parse_macro_input!(item as ItemFn);
    let helper = match jit_interp::jitcode_lower::generate_inline_helper_jitcode_with_calls(
        &func,
        &args.calls,
        &args.ref_params,
        &args.ref_fields,
        &args.array_fields,
        &args.int_fields,
        &args.native_int_binops,
        &args.native_tag_small,
        &args.native_identity,
        &args.headerless_structs,
        &args.inlined_prefix,
    ) {
        Ok(Some(lowered)) => lowered,
        Ok(None) => {
            return syn::Error::new_spanned(
                &func.block,
                "#[jit_inline] could not lower this helper into JitCode",
            )
            .to_compile_error()
            .into();
        }
        Err(err) => return err.to_compile_error().into(),
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = rewrite_jit_inline_ref_param_fields(
        &func.block,
        &args.ref_params,
        &args.ref_fields,
        &args.array_fields,
        &args.int_fields,
        &args.struct_allocs,
        &args.inlined_prefix,
    );
    let helper_with_asm_name = format_ident!("__majit_inline_jitcode_{}_with_asm", sig.ident);
    let helper_shared_name = format_ident!("__majit_inline_jitcode_{}_shared", sig.ident);
    let helper_id_name = format_ident!("__majit_inline_jitcode_{}_id", sig.ident);
    let helper_prebuild_name = format_ident!("__majit_inline_jitcode_{}_prebuild", sig.ident);
    let policy_name = format_ident!("__majit_call_policy_{}", sig.ident);
    let helper_source_name = sig.ident.to_string();
    let helper_body = helper.body;
    let helper_liveness_prebuild = helper.liveness_prebuild;
    let return_reg = helper.return_reg;
    let helper_return = match helper.return_kind {
        InlineReturnKind::Int => quote! { __builder.int_return(#return_reg); },
        InlineReturnKind::Ref => quote! { __builder.ref_return(#return_reg); },
        InlineReturnKind::Float => quote! { __builder.float_return(#return_reg); },
        InlineReturnKind::Void => quote! { __builder.void_return(); },
    };

    // RPython jtransform.py: rewrite_call() bakes the result kind into the
    // emitted opname. Our inferred helper-policy surface can only model that
    // parity for int-return inline helpers; ref/float inline helpers must go
    // through explicit `inline_ref` / `inline_float` policies.
    let inferred_policy_code: u8 = match helper.return_kind {
        InlineReturnKind::Int => jit_interp::call_policy_byte::INT_INLINE,
        InlineReturnKind::Ref | InlineReturnKind::Float | InlineReturnKind::Void => {
            jit_interp::call_policy_byte::UNSUPPORTED
        }
    };
    let inferred_inline_builder = match helper.return_kind {
        InlineReturnKind::Int => quote! { #helper_shared_name as *const () },
        InlineReturnKind::Ref | InlineReturnKind::Float | InlineReturnKind::Void => {
            quote! { std::ptr::null() }
        }
    };

    // Ensure the right register file for each parameter
    let ensure_param_regs = {
        let mut stmts = Vec::new();
        let (count_i, count_r, count_f) =
            match jit_interp::jitcode_lower::inline_helper_param_counts(&func) {
                Ok(counts) => counts,
                Err(err) => return err.to_compile_error().into(),
            };
        if count_i != 0 {
            stmts.push(quote! { __builder.ensure_i_regs(#count_i); });
        }
        if count_r != 0 {
            stmts.push(quote! { __builder.ensure_r_regs(#count_r); });
        }
        if count_f != 0 {
            stmts.push(quote! { __builder.ensure_f_regs(#count_f); });
        }
        stmts
    };

    // `call.py:167-169` `JitCode(name, fnaddr, calldescr)`: the helper's own
    // translated function is the entry `bhimpl_inline_call_*`
    // (`blackhole.py:1278-1319`) calls instead of interpreting the body. The
    // Rust fn re-emitted above IS that function, so the address of an
    // `extern "C"` trampoline over it is the same program the jitcode encodes;
    // `emit_helper_call_target_fn` already builds one for the residual-call
    // policies, with the widened `-> i64` result `bh_call_i_dispatch` reads.
    //
    // A helper it declines — a generic, or a parameter type the trampoline
    // cannot carry — is left at `fnaddr = 0`, which is the byte-interpreted
    // path this expansion had before.
    let native_entry = match emit_helper_call_target_fn(&func, false, None, "jit_inline", &[], true)
    {
        Ok(Some((trace_target, _concrete, wrapper))) => {
            let arg_classes = match jit_interp::jitcode_lower::inline_helper_arg_classes(&func) {
                Ok(classes) => classes,
                Err(err) => return err.to_compile_error().into(),
            };
            let result_class =
                jit_interp::jitcode_lower::inline_return_kind_class(helper.return_kind);
            Some((wrapper, trace_target, arg_classes, result_class))
        }
        Ok(None) => None,
        Err(err) => return err.to_compile_error().into(),
    };
    let (native_entry_fn, set_native_entry) = match &native_entry {
        // The returned target matches `arg_classes`: integer arguments use
        // the widening shim, and a `'f'` argument uses the real `f64` entry
        // (`__majit_native_entry_*`). `bh_call_f` reads an `f64` return.
        Some((wrapper, target, arg_classes, result_class)) => (
            quote! { #wrapper },
            quote! {
                __builder.set_native_entry(
                    #target as *const () as usize as i64,
                    #arg_classes,
                    #result_class,
                );
            },
        ),
        None => (quote! {}, quote! {}),
    };

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #block
        }

        #native_entry_fn

        // Inline helper jitcodes register
        // per-marker liveness triples through the caller-supplied
        // `Assembler`.  The production caller threads the driver-shared
        // `Assembler` (see `JitDriver::shared_asm`) so all jitcodes —
        // top-level per-pc bodies and inline helpers — share the same
        // `all_liveness` byte stream and dedup against the same cache.
        // RPython parity: `rpython/jit/codewriter/assembler.py` is a
        // single object that assembles every JitCode in the program
        // (`call.py get_jitcode_calldescr`), so liveness offsets are always relative
        // to one shared table.
        #[doc(hidden)]
        #vis fn #helper_with_asm_name(
            __asm: &mut majit_metainterp::Assembler,
        ) -> majit_metainterp::JitCode {
            let mut __builder = majit_metainterp::JitCodeBuilder::new();
            // `jitcode.py JitCode.__init__ self.name = name` — every jitcode upstream is
            // named at construction. The builder defaults the field to the
            // empty string, and the diagnostics that print it (the bytecode
            // encoder's register/const ceiling audit among them) then identify
            // nothing: a declined helper reports `""` and the reader has no way
            // to tell which of a consumer's dozens it was.
            __builder.set_name(#helper_source_name);
            #set_native_entry
            #(#ensure_param_regs)*
            #helper_body
            #helper_return
            __builder.finalize_liveness(__asm);
            __builder.finish()
        }

        // RPython `pyjitpl.py finish_setup` order: pre-register
        // the helper's per-marker `-live-` triples into the driver-
        // shared `Assembler` at install time so the trace-time
        // `__builder.finalize_liveness(__asm)` above only dedups
        // (does not grow `all_liveness` past the snapshot taken by
        // `JitDriver::install_canonical_liveness`). Called from each
        // parent `__prebuild_jitcode_liveness_*` that statically
        // resolves an inline-call to this helper (see
        // `jitcode_lower::inline_prebuild_path`).
        #[allow(non_snake_case, unused_variables, unused_mut)]
        #[doc(hidden)]
        #vis fn #helper_prebuild_name(
            __asm: &mut majit_metainterp::Assembler,
        ) {
            // `call.py CallControl.unfinished_graphs`: this walk descends into
            // every statically resolved callee's prebuild, so a helper that
            // reaches itself would recurse with nothing to stop it and no
            // `JitCodeBuilder` in scope to decline.  Visiting once is also
            // semantically free — the body only registers liveness triples,
            // which `all_liveness_positions` already dedups.
            if !__asm.enter_inline_prebuild(&#helper_id_name as *const _ as usize) {
                return;
            }
            #helper_liveness_prebuild
        }

        /// The helper's identity, as an address.
        ///
        /// `call.py CallControl.jitcodes` keys its cache by the graph object;
        /// there is no graph object here, so a per-helper static stands in.
        /// It carries a value so it cannot be merged with another helper's
        /// marker by identical-code folding, which would silently make two
        /// helpers share one jitcode.
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis static #helper_id_name: std::sync::atomic::AtomicU8 =
            std::sync::atomic::AtomicU8::new(0);

        /// `call.py CallControl.get_jitcode` — cache first, then mint the
        /// identity BEFORE assembling the body.
        ///
        /// That order is the whole point.  Upstream registers an empty JitCode
        /// under the graph and fills it afterwards, so a graph that calls
        /// itself links to the object it is already registered under.  Here the
        /// identity comes from `Arc::new_cyclic`, which supplies a `Weak` to the
        /// allocation before the value exists: the body assembles inside that
        /// closure, and a self-call re-entering this function finds the
        /// under-construction slot and takes a back edge instead of assembling
        /// a second copy.
        ///
        /// Nothing inside the closure may upgrade that `Weak` — it does not
        /// resolve until the value is published.
        #[doc(hidden)]
        #[allow(non_snake_case)]
        #vis fn #helper_shared_name(
            __asm: &mut majit_metainterp::Assembler,
        ) -> majit_metainterp::jitcode::InlineJitCodeRef {
            let __key = &#helper_id_name as *const _ as usize;
            if let Some(__hit) = majit_metainterp::jitcode::lookup_inline_jitcode(__asm, __key) {
                return __hit;
            }
            let __arc = std::sync::Arc::new_cyclic(
                |__weak: &std::sync::Weak<majit_metainterp::JitCode>| {
                    majit_metainterp::jitcode::begin_inline_jitcode(__asm, __key, __weak.clone());
                    #helper_with_asm_name(__asm)
                },
            );
            majit_metainterp::jitcode::finish_inline_jitcode(__asm, __key, __arc.clone());
            majit_metainterp::jitcode::InlineJitCodeRef::Strong(__arc)
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        #vis fn #policy_name() -> (u8, *const (), *const (), *const (), *const (), i32) {
            (
                #inferred_policy_code,
                #inferred_inline_builder,
                std::ptr::null(),
                std::ptr::null(),
                #helper_prebuild_name as *const (),
                0i32,
            )
        }
    };

    // rlib/jit.py hints leave ordinary interpreter execution intact without
    // JIT translation. Keep the concrete helper in both builds; only its
    // generated JitCode/ABI metadata requires the consumer's JIT feature.
    if let Some(condition) = args.trace_cfg {
        let mut file: syn::File = syn::parse2(expanded).expect("generated inline helper items");
        for item in file.items.iter_mut().skip(1) {
            let attrs = match item {
                syn::Item::Fn(item) => &mut item.attrs,
                syn::Item::Static(item) => &mut item.attrs,
                _ => unreachable!("unexpected inline helper metadata item"),
            };
            attrs.push(syn::parse_quote!(#[cfg(#condition)]));
        }
        // `#[jit_module]` always emits `__majit_helper_trace_fnaddrs`,
        // which calls `__majit_call_policy_<helper>`. Keep a stub when
        // tracing is compiled out so a non-JIT build still resolves.
        let unsupported = jit_interp::call_policy_byte::UNSUPPORTED;
        quote! {
            #file
            #[cfg(not(#condition))]
            #[doc(hidden)]
            #[allow(non_snake_case)]
            #vis fn #policy_name() -> (u8, *const (), *const (), *const (), *const (), i32) {
                (
                    #unsupported,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0i32,
                )
            }
        }
        .into()
    } else {
        expanded.into()
    }
}

/// Auto-generate trace_instruction and JitState from an interpreter's dispatch loop.
///
/// This is the Rust equivalent of RPython's meta-tracing: the proc macro analyzes
/// the interpreter's opcode dispatch match and generates the tracing code automatically.
///
/// The interpreter author writes ONLY the dispatch loop.
/// The macro generates:
/// - `trace_instruction()` function (IR recording for each opcode)
/// - `JitState` impl with Meta/Sym types
/// - Replaces `jit_merge_point!()` / `can_enter_jit!()` markers with JitDriver calls
///
/// # Example
///
/// ```ignore
/// #[jit_interp(
///     state = InterpState,
///     env = Program,
///     storage = {
///         pool: state.storage,
///         selector: state.selected,
///         untraceable: [VAL_QUEUE, VAL_PORT],
///         scan: find_used_storages,
///     },
///     io_shims = {
///         interp_io::write_number => jit_write_number,
///         interp_io::write_utf8 => jit_write_utf8,
///     },
///     // optional: infer direct helper calls from sidecar metadata
///     // non-int result helpers still need explicit `calls = { ... => ... }`
///     auto_calls = true,
///     calls = {
///         helper_compute,
///         helper_opaque,
///         helper_sink,
///         helper_inline,
///         // explicit overrides are still allowed:
///         helper_force_residual => residual_int,
///     },
/// )]
/// pub fn mainloop_jit(program: &Program) -> i64 {
///     // ... setup ...
///     while pc < program.size {
///         jit_merge_point!();
///         match op {
///             OP_ADD => state.storage.get_mut(state.selected).add(),
///             // ...
///         }
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn jit_interp(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_macro_input!(attr as jit_interp::JitInterpConfig);
    let func = parse_macro_input!(item as ItemFn);
    jit_interp::transform_jit_interp(config, func).into()
}

/// Register a Rust struct with `majit_ir::descr::GcCache`.
///
/// RPython parity: descr.py `get_size_descr` + descr.py
/// `get_field_descr`. RPython's translator auto-discovers `lltype.Struct`
/// fields and emits FieldDescr/SizeDescr; this macro performs the same
/// auto-discovery for Rust structs using `offset_of!` / `size_of`.
///
/// Generated inherent methods:
/// - `__majit_type_id() -> u64`
/// - `__majit_register_descrs(&mut GcCache) -> DescrRef`
/// - `const __MAJIT_FIELD_NAMES: &'static [&'static str]`
///
/// Only named-field structs are supported in this skeleton. Field types
/// are classified by a simple heuristic (integer/float primitives vs Ref).
#[proc_macro_attribute]
pub fn jit_struct(attr: TokenStream, item: TokenStream) -> TokenStream {
    jit_struct::expand(attr.into(), item.into()).into()
}

/// JIT attribute names recognized by `#[jit_module]` for automatic helper discovery.
const JIT_HELPER_ATTRS: &[&str] = &[
    "jit_inline",
    "elidable",
    // call.py _canraise(op) 3-way pick on the elidable branch.
    "elidable_cannot_raise",
    "elidable_or_memerror",
    "elidable_promote",
    // ImplItemFn-friendly pass-through variant.  The
    // `#[elidable]` family emits a module-level trampoline, so attaching
    // it inside an `impl` block fails with `not found in this scope`.
    // `#[jit_elidable]` flows the hint without a trampoline, so it can
    // be attached to a method — but discovery requires registering it
    // in this list.  It emits the same `_elidable_function_<NAME>` const
    // as `#[elidable]`, which the ullbc hint harvester
    // (`front::llbc_hints`) maps to the "elidable" hint.
    "jit_elidable",
    "dont_look_inside",
    "dont_look_inside_cannot_raise",
    "look_inside",
    "unroll_safe",
    "loop_invariant",
    "not_in_trace",
    "look_inside_iff",
    "oopspec",
    "jit_may_force",
    "jit_loop_invariant",
    // `rlib/jit.py` — `@purefunction` is a deprecated alias for
    // `@elidable`; `@purefunction_promote` (`jit.py`) likewise
    // for `@elidable_promote`.  Listed here so `#[jit_module]` discovery
    // recognises the alias decorators when callers use the deprecated
    // name.
    "purefunction",
    "purefunction_promote",
];

/// Check if a syn attribute path matches one of the JIT helper attributes.
fn jit_attr_name(attr: &syn::Attribute) -> Option<String> {
    let path = attr.path();
    // Match both bare `elidable` and qualified `majit_macros::elidable`
    let last_segment = path.segments.last()?;
    let name = last_segment.ident.to_string();
    if JIT_HELPER_ATTRS.contains(&name.as_str()) {
        Some(name)
    } else {
        None
    }
}

/// Discovered helper entry: function name and its JIT attribute.
///
/// `impl_type_segments` is `Some(vec)` for inherent / trait-impl methods
/// discovered inside an `impl` block, carrying the type-path segments
/// exactly as written at the `impl` header (e.g. `[a, Foo]` for
/// `impl a::Foo { ... }`). Segments are extracted from the type path
/// (`syn::Type::Path`) so that downstream code can render the
/// `impl_type` as a joined string matching the canonical spelling
/// `CallControl::register_macro_impl_helper_trace_fnaddr` qualifies.
/// RPython parity:
/// `getfunctionptr(graph)`
/// (call.py) does not distinguish free fns from methods; pyre
/// keys methods by the `[impl_type_joined, method]` 2-segment CallPath
/// (lib.rs), so the macro emits exactly that.
struct DiscoveredHelper {
    fn_name: Ident,
    attr_name: String,
    /// `None` for free fns, `Some(segments)` for impl methods.
    impl_type_segments: Option<Vec<Ident>>,
    /// `Some(segments)` for trait impls (`impl Trait for Type { fn m }`),
    /// carrying the trait path segments. Emitted in the fnaddr cast as
    /// `<Type as Trait>::method` to disambiguate when a type has
    /// multiple inherent/trait methods named `method`. `None` for
    /// inherent impls (plain `<Type>::method`) and free fns.
    trait_type_segments: Option<Vec<Ident>>,
}

/// Extract the identifier sequence from a `syn::Type::Path`, e.g.
/// `a::b::Foo` → `[a, b, Foo]`. Returns `None` for non-path types
/// (trait objects, references, fn pointers, generics on the outer
/// path, …) — those cases are not expressible as a canonical
/// `self_ty_root` string in the parser either, so we skip them.
fn impl_type_path_segments(ty: &syn::Type) -> Option<Vec<Ident>> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    // Strip generic arguments from each segment and join the
    // identifiers — the canonical `self_ty_root` shape. Match it here
    // by taking `ident` only.
    Some(
        type_path
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.clone())
            .collect(),
    )
}

/// Scan a module's items for functions annotated with JIT helper attributes.
///
/// Walks both top-level `Item::Fn` and inherent / trait-impl methods
/// inside `Item::Impl` blocks. Instance methods are NOT skipped — Rust
/// allows `S::f as fn(&S)`, `S::g as fn(&mut S)`, `S::h as fn(S)` to
/// coerce to plain function pointers (verified with rustc), and RPython
/// upstream treats `getfunctionptr(graph)` uniformly across free fns and
/// methods (`call.py get_jitcode_calldescr`).
fn discover_helpers(items: &[syn::Item]) -> Vec<DiscoveredHelper> {
    let mut discovered = Vec::new();
    for item in items {
        match item {
            syn::Item::Fn(func) => {
                for attr in &func.attrs {
                    if let Some(attr_name) = jit_attr_name(attr) {
                        discovered.push(DiscoveredHelper {
                            fn_name: func.sig.ident.clone(),
                            attr_name,
                            impl_type_segments: None,
                            trait_type_segments: None,
                        });
                        // Only record the first JIT attribute per function
                        break;
                    }
                }
            }
            // RPython parity: `impl Type { fn helper(...) }` and
            // `impl Trait for Type { fn helper(...) }` both lower to a
            // `getfunctionptr(graph)` whose canonical CallPath is
            // `[impl_type_joined, helper]`.
            syn::Item::Impl(item_impl) => {
                let Some(impl_segs) = impl_type_path_segments(&item_impl.self_ty) else {
                    continue;
                };
                // `impl Trait for Type { ... }` — carry the trait path
                // so the fnaddr cast can disambiguate via
                // `<Type as Trait>::method`. RPython `getfunctionptr(graph)`
                // uses graph identity directly so no such aliasing exists
                // upstream; in Rust a bare `<Type>::method` cast is
                // ambiguous when the Type carries multiple trait methods
                // with the same name (or a name-colliding inherent).
                let trait_segs: Option<Vec<Ident>> =
                    item_impl.trait_.as_ref().and_then(|(_, path, _)| {
                        Some(
                            path.segments
                                .iter()
                                .map(|seg| seg.ident.clone())
                                .collect::<Vec<_>>(),
                        )
                        .filter(|segs: &Vec<Ident>| !segs.is_empty())
                    });
                for impl_item in &item_impl.items {
                    let syn::ImplItem::Fn(method) = impl_item else {
                        continue;
                    };
                    for attr in &method.attrs {
                        if let Some(attr_name) = jit_attr_name(attr) {
                            discovered.push(DiscoveredHelper {
                                fn_name: method.sig.ident.clone(),
                                attr_name,
                                impl_type_segments: Some(impl_segs.clone()),
                                trait_type_segments: trait_segs.clone(),
                            });
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    discovered
}

/// Module-level automatic helper discovery for JIT-annotated functions.
///
/// Place `#[jit_module]` on a `mod` block containing JIT-annotated functions:
/// `#[elidable]`, `#[elidable_promote]`, `#[dont_look_inside]`,
/// `#[unroll_safe]`, `#[loop_invariant]`, `#[not_in_trace]`,
/// `#[look_inside_iff]`, `#[oopspec]`, `#[jit_inline]`,
/// `#[jit_may_force]`, `#[jit_loop_invariant]`.
/// The macro scans all items and generates a hidden registry constant
/// listing discovered helpers and their attributes.
///
/// # Example
///
/// ```ignore
/// #[jit_module]
/// mod my_interp {
///     #[jit_inline]
///     fn helper_add(a: i64, b: i64) -> i64 { a + b }
///
///     #[elidable]
///     fn lookup(key: i64) -> i64 { /* ... */ }
///
///     #[dont_look_inside]
///     fn opaque(x: i64) -> i64 { /* ... */ }
///
///     fn not_jit_relevant() { /* ignored */ }
/// }
///
/// // After expansion, `my_interp` contains:
/// // const __MAJIT_DISCOVERED_HELPERS: &[&str] = &["helper_add", "lookup", "opaque"];
/// // const __MAJIT_HELPER_POLICIES: &[(&str, &str)] = &[
/// //     ("helper_add", "jit_inline"),
/// //     ("lookup", "elidable"),
/// //     ("opaque", "dont_look_inside"),
/// // ];
/// ```
#[proc_macro_attribute]
pub fn jit_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut module = parse_macro_input!(item as syn::ItemMod);

    let Some((brace, ref items)) = module.content else {
        return syn::Error::new_spanned(
            &module.ident,
            "#[jit_module] requires an inline module body (not `mod foo;`)",
        )
        .to_compile_error()
        .into();
    };

    let discovered = discover_helpers(items);

    // Per-helper tokens: free fns route through the existing
    // `__majit_call_policy_*` trampoline indirection; impl methods take
    // their direct `<Type>::method` address since the policy macro family
    // does not yet decorate impl items.  Both shapes feed the same
    // `__majit_helper_trace_fnaddrs()` table that the codewriter reads
    // through `register_macro_helper_trace_fnaddr`.
    // For impl methods we emit the `impl_type_joined` string by
    // concat!-ing the individual `stringify!(ident)` tokens per segment;
    // `stringify!` on individual idents produces clean
    // whitespace-free identifier strings (unlike `stringify!(a::Foo)`
    // which expands with spaces around `::`).  The resulting joined
    // form is what `CallControl::register_macro_impl_helper_trace_fnaddr`
    // qualifies into the canonical `[impl_type_joined, method]` CallPath.
    let helper_name_lits: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .map(|h| {
            let fn_name = &h.fn_name;
            match &h.impl_type_segments {
                None => quote! { stringify!(#fn_name) },
                Some(segs) => {
                    let parts = segs_joined_with_method(segs, fn_name);
                    quote! { #parts }
                }
            }
        })
        .collect();
    let helper_attr_lits: Vec<&str> = discovered.iter().map(|h| h.attr_name.as_str()).collect();
    let helper_path_lits: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .map(|h| {
            let fn_name = &h.fn_name;
            match &h.impl_type_segments {
                None => quote! {
                    concat!(module_path!(), "::", stringify!(#fn_name))
                },
                Some(segs) => {
                    let parts = segs_joined_with_method(segs, fn_name);
                    quote! {
                        concat!(module_path!(), "::", #parts)
                    }
                }
            }
        })
        .collect();
    let helper_addr_exprs: Vec<proc_macro2::TokenStream> =
        discovered.iter().map(impl_addr_expr).collect();

    // Structured impl-method registry:
    //   `(module_path_with_crate, impl_type_as_written, method, fnaddr)`.
    // The codewriter consumes this through
    // `CallControl::register_macro_impl_helper_trace_fnaddr` which applies
    // its module-prefix-qualification rule to decide whether to prepend
    // the module prefix before registering the canonical 2-segment
    // CallPath `[impl_type_joined, method]` (lib.rs).
    let impl_entries: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .filter_map(|h| {
            let segs = h.impl_type_segments.as_ref()?;
            let fn_name = &h.fn_name;
            let impl_type_as_written = segs_joined(segs);
            let addr = impl_addr_expr(h);
            Some(quote! {
                (
                    module_path!(),
                    #impl_type_as_written,
                    stringify!(#fn_name),
                    #addr,
                )
            })
        })
        .collect();

    let registry_names = quote! {
        /// Hidden registry of automatically discovered JIT helpers.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub const __MAJIT_DISCOVERED_HELPERS: &[&str] = &[
            #(#helper_name_lits),*
        ];
    };

    let registry_policies = quote! {
        /// Hidden registry mapping each discovered helper to its JIT attribute.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub const __MAJIT_HELPER_POLICIES: &[(&str, &str)] = &[
            #((#helper_name_lits, #helper_attr_lits)),*
        ];
    };

    let registry_trace_fnaddrs = quote! {
        /// Hidden registry mapping each discovered helper to the compiled
        /// trace-call surface address used by `majit-codewriter`'s
        /// `getfunctionptr(graph)` parity path.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub fn __majit_helper_trace_fnaddrs() -> ::std::vec::Vec<(&'static str, i64)> {
            ::std::vec![
                #((#helper_path_lits, #helper_addr_exprs)),*
            ]
        }
    };

    let registry_impl_trace_fnaddrs = quote! {
        /// Hidden registry mapping each discovered impl-method helper to
        /// its `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`
        /// 4-tuple. The codewriter consumes this through
        /// `CallControl::register_macro_impl_helper_trace_fnaddr`, which
        /// applies its module-prefix-qualification rule to decide whether
        /// to prepend the module prefix before storing the canonical
        /// 2-segment CallPath `[impl_type_joined, method]` — same shape
        /// used for `self_ty_root`-keyed methods (lib.rs).
        #[doc(hidden)]
        #[allow(dead_code)]
        pub fn __majit_helper_impl_trace_fnaddrs()
            -> ::std::vec::Vec<(&'static str, &'static str, &'static str, i64)>
        {
            ::std::vec![
                #(#impl_entries),*
            ]
        }
    };

    // Inject the registry constants into the module body
    let mut new_items = items.clone();
    new_items.push(syn::parse2(registry_names).expect("failed to parse registry_names"));
    new_items.push(syn::parse2(registry_policies).expect("failed to parse registry_policies"));
    new_items
        .push(syn::parse2(registry_trace_fnaddrs).expect("failed to parse registry_trace_fnaddrs"));
    new_items.push(
        syn::parse2(registry_impl_trace_fnaddrs)
            .expect("failed to parse registry_impl_trace_fnaddrs"),
    );
    module.content = Some((brace, new_items));

    quote! { #module }.into()
}

/// Helper attributes that are pure pass-throughs and therefore do not
/// emit a `__majit_call_policy_<name>()` trampoline.  When `#[jit_module]`
/// discovers a free fn carrying one of these, the registry must take the
/// direct function address instead of the policy-fn indirection — the
/// indirection symbol simply does not exist for these.
///
/// Impl methods always take the direct address regardless (`impl_addr_expr`
/// only enters this branch on the free-fn path).
fn attr_is_passthrough(attr_name: &str) -> bool {
    matches!(
        attr_name,
        // `#[jit_elidable]` — ImplItemFn-friendly pass-through.
        "jit_elidable"
        // `#[unroll_safe]`.
        | "unroll_safe"
        // `#[not_in_trace]`.
        | "not_in_trace"
        // `#[oopspec(...)]` — body untouched, only `_MAJIT_OOPSPEC`
        // marker constant added.
        | "oopspec"
        // `#[look_inside_iff(...)]` — emits dispatch wrapper around
        // `_orig_<name>` / `<name>_trampoline`; the public name keeps its body
        // but no policy fn is generated.
        | "look_inside_iff"
    )
}

/// Emit the runtime address expression for a `DiscoveredHelper`:
///
/// * Free fn carrying a policy-emitting attribute: route through
///   `__majit_call_policy_<name>()`.
/// * Free fn carrying a pass-through attribute (`#[jit_elidable]`,
///   `#[unroll_safe]`, `#[not_in_trace]`, `#[oopspec]`,
///   `#[look_inside_iff]`): take the direct fn address — the policy
///   symbol is not emitted by these macros and routing through it would
///   fail with `not found in this scope` at expansion time.
/// * Inherent impl (`impl Type { ... }`): `<Type>::method as *const ()`.
/// * Trait impl (`impl Trait for Type { ... }`):
///   `<Type as Trait>::method as *const ()` — fully-qualified trait
///   method syntax, so the cast is unambiguous when the type carries
///   multiple name-colliding inherent/trait methods. RPython's
///   `getfunctionptr(graph)` uses graph identity directly so upstream
///   never needs this disambiguation.
fn impl_addr_expr(h: &DiscoveredHelper) -> proc_macro2::TokenStream {
    let fn_name = &h.fn_name;
    let Some(impl_segs) = h.impl_type_segments.as_ref() else {
        if attr_is_passthrough(&h.attr_name) {
            return quote! {
                #fn_name as *const () as usize as i64
            };
        }
        let policy_fn = format_ident!("__majit_call_policy_{}", fn_name);
        return quote! {
            {
                let (_, _inline_builder, __trace_target, __concrete_target, _prebuild, _save_err) = #policy_fn();
                let __trace_target = if __trace_target.is_null() {
                    if __concrete_target.is_null() {
                        #fn_name as *const ()
                    } else {
                        __concrete_target
                    }
                } else {
                    __trace_target
                };
                __trace_target as usize as i64
            }
        };
    };
    let ty_path = path_tokens_from_segs(impl_segs);
    match h.trait_type_segments.as_ref() {
        Some(trait_segs) => {
            let trait_path = path_tokens_from_segs(trait_segs);
            quote! {
                <#ty_path as #trait_path>::#fn_name as *const () as usize as i64
            }
        }
        None => quote! {
            <#ty_path>::#fn_name as *const () as usize as i64
        },
    }
}

/// Build a `concat!(stringify!(s1), "::", stringify!(s2), …)` token
/// stream that renders the impl type's joined canonical form (e.g.
/// `"a::Foo"`). Uses `stringify!` per ident to avoid the whitespace
/// artefacts that `stringify!(a::Foo)` produces.
fn segs_joined(segs: &[Ident]) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            parts.push(quote! { "::" });
        }
        parts.push(quote! { stringify!(#seg) });
    }
    quote! { concat!(#(#parts),*) }
}

/// `segs_joined` followed by `"::"` and the method name — used for the
/// full `Type::method` rendering inside `__MAJIT_DISCOVERED_HELPERS` /
/// `__MAJIT_HELPER_POLICIES` and the path slot of
/// `__majit_helper_trace_fnaddrs`.
fn segs_joined_with_method(segs: &[Ident], method: &Ident) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for seg in segs {
        parts.push(quote! { stringify!(#seg) });
        parts.push(quote! { "::" });
    }
    parts.push(quote! { stringify!(#method) });
    quote! { concat!(#(#parts),*) }
}

/// Reconstruct the `syn::Path` tokens that `<#path>::method` can use as
/// a generic-path context. Just re-emits the ident sequence
/// `s1::s2::…::sN`.
fn path_tokens_from_segs(segs: &[Ident]) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            parts.push(quote! { :: });
        }
        parts.push(quote! { #seg });
    }
    quote! { #(#parts)* }
}

/// Standalone virtualizable field declaration macro.
///
/// Generates `VirtualizableInfo` builder, field spec constants, and
/// JitState hook helper functions from a declarative specification.
///
/// # Example
///
/// ```ignore
/// majit_macros::virtualizable! {
///     state = MyState,
///     name = "frame",
///     heap_ptr = |s: &MyState| s.frame_ptr(),
///     token_offset = VABLE_TOKEN_OFFSET,
///
///     fields = {
///         next_instr: int @ NEXT_INSTR_OFFSET,
///         code: ref @ CODE_OFFSET,
///     },
///
///     arrays = {
///         stack: ref @ STACK_OFFSET {
///             embedded,
///             ptr_offset: PTR_OFFSET,
///             length_offset: LEN_OFFSET,
///             items_offset: 0,
///         },
///     },
/// }
/// ```
#[proc_macro]
pub fn virtualizable(input: TokenStream) -> TokenStream {
    virtualizable::parse_and_expand(input)
}

/// Derive macro for virtualizable symbolic state structs.
///
/// Recognizes `#[vable(...)]` attributes on fields:
/// - `#[vable(frame)]` — frame pointer OpRef
/// - `#[vable(field)]` — static virtualizable field OpRef
/// - `#[vable(array_base)]` — array base index
/// - `#[vable(locals)]` — symbolic locals `Vec<OpRef>`
/// - `#[vable(stack)]` — symbolic stack `Vec<OpRef>`
/// - `#[vable(local_types)]` / `#[vable(stack_types)]` — type vectors
/// - `#[vable(nlocals)]` / `#[vable(valuestackdepth)]` — shape fields
///
/// Generates: `flush_vable_fields`, `vable_field_oprefs`,
/// `init_vable_indices`, `vable_collect_jump_args`,
/// `vable_collect_typed_jump_args`.
#[proc_macro_derive(VirtualizableSym, attributes(vable))]
pub fn derive_virtualizable_sym(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_sym(input).into()
}

/// Derive macro for virtualizable meta structs.
///
/// Recognizes `#[vable(...)]` attributes on fields:
/// - `#[vable(num_locals)]` — number of locals
/// - `#[vable(valuestackdepth)]` — value stack depth
/// - `#[vable(slot_types)]` — slot type vector
///
/// Generates: `vable_stack_only_depth`, `vable_update_vsd_from_len`.
#[proc_macro_derive(VirtualizableMeta, attributes(vable))]
pub fn derive_virtualizable_meta(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_meta(input).into()
}

/// Derive macro for virtualizable interpreter state structs.
///
/// Recognizes `#[vable(...)]` attributes:
/// - `#[vable(frame)]` — frame pointer field (usize)
/// - `#[vable(static_field = N)]` — state-backed VirtualizableInfo field at index N
///
/// Generates: `virt_export_static_boxes`, `virt_import_static_boxes`,
/// `virt_export_all`.
#[proc_macro_derive(VirtualizableState, attributes(vable))]
pub fn derive_virtualizable_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_state(input).into()
}

/// Fieldless enum passed to a residual call as its discriminant word.
#[proc_macro_derive(FieldlessEnumArg)]
pub fn derive_fieldless_enum_arg(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    match expand_fieldless_enum_arg(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_fieldless_enum_arg(input: &syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let syn::Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "FieldlessEnumArg requires a fieldless enum",
        ));
    };
    let name = &input.ident;
    let mut arms = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, syn::Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "FieldlessEnumArg requires every variant to be a unit variant",
            ));
        }
        let variant_ident = &variant.ident;
        arms.push(quote! {
            x if x == #name::#variant_ident as i64 => #name::#variant_ident
        });
    }
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        unsafe impl #impl_generics ::majit_ir::helper_fnaddr::FieldlessEnumArg for #name #ty_generics #where_clause {
            fn from_discriminant(d: i64) -> Self {
                match d {
                    #(#arms,)*
                    _ => panic!("invalid discriminant"),
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_fn(src: &str) -> ItemFn {
        syn::parse_str(src).expect("function parses")
    }

    fn skip(src: &str) -> Option<HelperFnAddrSkip> {
        trampoline_skip_reason(&parse_fn(src), "dont_look_inside", &[])
    }

    #[test]
    fn trampoline_skip_reason_classifies_each_listed_cause() {
        assert_eq!(
            skip("fn f<T>(x: T) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::Generic)
        );
        assert_eq!(skip("fn f(s: &str) -> i64 { 0 }"), None);
        assert_eq!(
            skip("fn f(s: &mut str) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(s: *const str) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(skip("fn f(o: PyObjectRef, name: &Wtf8) -> i64 { 0 }"), None);
        assert_eq!(
            skip("fn f(o: PyObjectRef, name: &pyre_object::Wtf8) -> i64 { 0 }"),
            None
        );
        assert_eq!(
            skip("fn f(o: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> { None }"),
            None
        );
        assert_eq!(
            skip("fn f() -> Option<i64> { None }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(s: &str) -> &str { s }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(name: &str, path: &Path) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(xs: &[u8]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(xs: &[PyObjectRef]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(xs: &[*mut PyObject]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(skip("fn f(xs: &[*mut u8]) -> i64 { 0 }"), None);
        assert_eq!(skip("fn f(xs: &[GCREF]) -> i64 { 0 }"), None);
        assert_eq!(skip("fn f(item: GCREF) -> GCREF { item }"), None);
        assert_eq!(
            skip("fn f(l: &mut Vec<GCREF>, index: usize, item: GCREF) {}"),
            None
        );
        assert_eq!(skip("fn f() -> Vec<GCREF> { Vec::new() }"), None);
        assert_eq!(skip("fn f(xs: &mut [i64]) -> i64 { 0 }"), None);
        assert_eq!(
            skip("fn f(p: *const [PyObjectRef]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(p: *const [u8]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(p: *mut [u8]) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f(p: *const dyn Trait) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::FatPointerArg)
        );
        assert_eq!(
            skip("fn f() -> *const [u8] { loop {} }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(x: i64) -> Result<&[u8], PyError> { Ok(&[]) }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(x: i64) -> Result<i64, ()> { Ok(x) }"),
            Some(HelperFnAddrSkip::ResultReturn)
        );
        assert_eq!(
            skip("fn f(x: i64) -> Result<Vec<i64>, PyError> { Ok(Vec::new()) }"),
            None
        );
        assert_eq!(
            skip("fn f(x: i64) -> Result<i64, RBigIntError> { Ok(x) }"),
            Some(HelperFnAddrSkip::ResultReturn)
        );
        assert_eq!(
            trampoline_skip_reason(
                &parse_fn("fn f(x: i64) -> Result<i64, PyError> { Ok(x) }"),
                "dont_look_inside_cannot_raise",
                &[],
            ),
            Some(HelperFnAddrSkip::ResultReturn)
        );
        assert_eq!(
            skip("fn f(&self, x: i64) -> i64 { x }"),
            Some(HelperFnAddrSkip::MethodReceiver)
        );
        assert_eq!(
            skip("fn f(x: String) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(op: CompareOp) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(
            skip("fn f(x: (i64, i64)) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::Other)
        );
        assert_eq!(skip("fn f(x: i64) -> i64 { x }"), None);
        assert_eq!(skip("fn f(a: f64, b: f64) -> f64 { a }"), None);
        assert_eq!(skip("fn f(x: i64) -> Result<i64, PyError> { Ok(x) }"), None);
        assert_eq!(skip("fn f() -> Result<(), PyError> { Ok(()) }"), None);
        assert_eq!(
            skip("fn f(x: i64) -> Result<bool, crate::PyError> { Ok(true) }"),
            None
        );
        assert_eq!(
            skip("fn f(a: &BigInt, b: &BigInt) -> Result<f64, PyError> { Ok(0.0) }"),
            None
        );
    }

    #[test]
    fn gcref_pointer_alias_emits_a_word_abi_trampoline() {
        let func =
            parse_fn("fn ll_vec_setitem_fast_r(l: &mut Vec<GCREF>, index: usize, item: GCREF) {}");
        assert_eq!(trampoline_skip_reason(&func, "dont_look_inside", &[]), None);
        let (_, _, tokens) =
            emit_helper_call_target_fn(&func, true, None, "dont_look_inside", &[], false)
                .expect("emit")
                .expect("trampoline");
        let text = tokens.to_string();
        assert!(
            text.contains(
                "fn __majit_call_target_ll_vec_setitem_fast_r (__majit_arg_0 : i64 , __majit_arg_1 : i64 , __majit_arg_2 : i64)"
            ),
            "{text}"
        );
        assert!(text.contains("as * const () , 3"), "{text}");

        let ret =
            parse_fn("fn ll_vec_newlist_hint_r(lengthhint: usize) -> Vec<GCREF> { Vec::new() }");
        assert_eq!(trampoline_skip_reason(&ret, "dont_look_inside", &[]), None);
        let (_, _, ret_tokens) =
            emit_helper_call_target_fn(&ret, true, None, "dont_look_inside", &[], false)
                .expect("emit")
                .expect("trampoline");
        let ret_text = ret_tokens.to_string();
        assert!(
            ret_text.contains(
                "fn __majit_call_target_ll_vec_newlist_hint_r (__majit_arg_0 : i64) -> i64"
            ),
            "{ret_text}"
        );
        assert!(ret_text.contains("raw_malloc_varsize_char"), "{ret_text}");
    }

    #[test]
    fn pair_slice_trampoline_emits_ptr_len_and_arity_three() {
        let func = parse_fn("fn f(o: PyObjectRef, xs: &[*mut u8]) -> i64 { 0 }");
        let (_, _, tokens) =
            emit_helper_call_target_fn(&func, true, None, "dont_look_inside", &[], false)
                .expect("emit")
                .expect("trampoline");
        let text = tokens.to_string();
        assert!(
            text.contains(
                "fn __majit_call_target_f (__majit_arg_0 : i64 , __majit_arg_1 : i64 , __majit_arg_1_len : i64) -> i64"
            ),
            "{text}"
        );
        assert!(text.contains("__majit_arg_1_len"), "{text}");
        assert!(text.contains("as * const () , 3"), "{text}");
        assert!(text.contains("std :: slice :: from_raw_parts"), "{text}");
        assert!(text.contains("& []"), "{text}");
    }

    #[test]
    fn shared_rstr_ref_trampoline_is_one_ref_word() {
        let func = parse_fn("fn f(o: PyObjectRef, name: &Wtf8) -> i64 { 0 }");
        let (_, _, tokens) =
            emit_helper_call_target_fn(&func, true, None, "dont_look_inside", &[], false)
                .expect("emit")
                .expect("trampoline");
        let text = tokens.to_string();
        assert!(
            text.contains(
                "fn __majit_call_target_f (__majit_arg_0 : i64 , __majit_arg_1 : i64) -> i64"
            ),
            "{text}"
        );
        assert!(text.contains("as * const () , 2"), "{text}");
        assert!(text.contains("< Wtf8 > :: from_bytes_unchecked"), "{text}");
        assert!(
            text.contains(":: majit_ir :: helper_fnaddr :: rstr_payload"),
            "{text}"
        );

        let str_func = parse_fn("fn g(s: &str) -> i64 { 0 }");
        let (_, _, str_tokens) =
            emit_helper_call_target_fn(&str_func, true, None, "dont_look_inside", &[], false)
                .expect("emit")
                .expect("trampoline");
        let str_text = str_tokens.to_string();
        assert!(
            str_text.contains(":: core :: str :: from_utf8_unchecked"),
            "{str_text}"
        );
        assert!(str_text.contains("as * const () , 1"), "{str_text}");
    }

    #[test]
    fn word_enums_argument_emits_a_trampoline_for_the_listed_enum() {
        let func = parse_fn("fn f(op: CompareOp) -> i64 { 0 }");
        let words: WordEnumsArg = syn::parse_str("word_enums(CompareOp)").unwrap();
        assert_eq!(
            trampoline_skip_reason(&func, "dont_look_inside", &words.0),
            None
        );
        assert_eq!(
            skip("fn f(op: CompareOp) -> i64 { 0 }"),
            Some(HelperFnAddrSkip::Other)
        );
        let (_, _, tokens) =
            emit_helper_call_target_fn(&func, true, None, "dont_look_inside", &words.0, false)
                .expect("emit")
                .expect("trampoline");
        assert!(
            tokens.to_string().contains("from_discriminant"),
            "{}",
            tokens
        );
    }

    #[test]
    fn elidable_promote_fnaddr_path_is_the_orig_ident() {
        let orig = parse_fn("fn _orig_leaf_unlikely_name(x: i64) -> i64 { x }");
        let wrapper: Ident = syn::parse_str("leaf").unwrap();
        let (_, _, tokens) =
            emit_helper_call_target_fn(&orig, true, Some(&wrapper), "elidable", &[], false)
                .expect("emit")
                .expect("trampoline");
        let text = tokens.to_string();
        assert!(
            text.contains("stringify ! (_orig_leaf_unlikely_name)"),
            "{text}"
        );
        assert!(text.contains("stringify ! (leaf)"), "{text}");
    }
}
