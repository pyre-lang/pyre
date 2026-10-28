//! Expansion of `llexternal` and `external_compilation_info`.
//!
//! `llexternal` (`rffi.py`) builds the funcptr, then `call_external_function`,
//! then the forwarding wrapper. The derivations below follow that function's
//! order: `calling_conv`, callback scan, `releasegil` → `invoke_around_handlers`,
//! `random_effects_on_gcobjs`, then `assert not (elidable_function and
//! random_effects_on_gcobjs)`.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Expr, Ident, LitBool, LitInt, LitStr, Meta, Token, Type, Visibility, bracketed,
    parse::Parse, parse::ParseStream,
};

pub fn expand_llexternal(input: TokenStream) -> syn::Result<TokenStream> {
    let parsed: LlexternalInput = syn::parse2(input)?;
    Ok(parsed.expand())
}

pub fn expand_external_compilation_info(input: TokenStream) -> syn::Result<TokenStream> {
    let parsed: EciInput = syn::parse2(input)?;
    parsed.expand()
}

struct LlexternalInput {
    link_attrs: Vec<LinkAttr>,
    vis: Visibility,
    name: Ident,
    c_name: String,
    args: Vec<Type>,
    result: Type,
    sandboxsafe: bool,
    releasegil: Tri,
    nowrapper: bool,
    calling_conv: Option<String>,
    elidable: bool,
    macro_path: Option<syn::Path>,
    random_effects: Tri,
    save_err: Expr,
    natural_arity: i64,
    compilation_info: Option<Expr>,
}

#[derive(Clone, Copy)]
enum Tri {
    Auto,
    Bool(bool),
}

struct LinkAttr {
    attr: Attribute,
    /// `None` is an unconditional `#[link_name]`. `Some` is the `cfg_attr` predicate.
    cfg_pred: Option<TokenStream>,
}

impl Parse for LlexternalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut link_attrs = Vec::new();
        for attr in input.call(Attribute::parse_outer)? {
            link_attrs.push(parse_link_attr(attr)?);
        }
        let vis: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let c_name: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let args_content;
        bracketed!(args_content in input);
        let args: Punctuated<Type, Token![,]> =
            args_content.parse_terminated(Type::parse, Token![,])?;
        input.parse::<Token![,]>()?;
        let result: Type = input.parse()?;

        let mut sandboxsafe = false;
        let mut releasegil = Tri::Auto;
        let mut nowrapper = false;
        let mut calling_conv = None;
        let mut elidable = false;
        let mut macro_path = None;
        let mut random_effects = Tri::Auto;
        let mut save_err = None;
        let mut natural_arity = -1i64;
        let mut compilation_info = None;

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "sandboxsafe" => sandboxsafe = input.parse::<LitBool>()?.value(),
                "releasegil" => releasegil = parse_tri(input, &key)?,
                "_nowrapper" => nowrapper = input.parse::<LitBool>()?.value(),
                "calling_conv" => {
                    calling_conv = Some(input.parse::<LitStr>()?.value());
                }
                "elidable_function" => elidable = input.parse::<LitBool>()?.value(),
                "macro" => macro_path = Some(input.parse()?),
                "random_effects_on_gcobjs" => random_effects = parse_tri(input, &key)?,
                "save_err" => save_err = Some(input.parse()?),
                "natural_arity" => natural_arity = parse_signed_int(input)?,
                "compilation_info" => compilation_info = Some(input.parse()?),
                "_callable" => {
                    return Err(syn::Error::new(
                        key.span(),
                        "`_callable` is the ll2ctypes stand-in; pass `macro = <path>` for a Rust funcptr",
                    ));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown llexternal kwarg `{other}`"),
                    ));
                }
            }
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after llexternal arguments"));
        }
        if let Some(conv) = &calling_conv
            && conv != "c"
            && conv != "unknown"
            && conv != "win"
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "calling_conv must be \"c\", \"unknown\", or \"win\"",
            ));
        }
        if macro_path.is_some() && !link_attrs.is_empty() {
            return Err(syn::Error::new_spanned(
                &link_attrs[0].attr,
                "link_name attributes apply to the generated extern funcptr; remove them when `macro` replaces that funcptr",
            ));
        }
        Ok(LlexternalInput {
            link_attrs,
            vis,
            name,
            c_name: c_name.value(),
            args: args.into_iter().collect(),
            result,
            sandboxsafe,
            releasegil,
            nowrapper,
            calling_conv,
            elidable,
            macro_path,
            random_effects,
            save_err: save_err
                .unwrap_or_else(|| syn::parse_quote!(::majit_rlib::rffi::RFFI_ERR_NONE)),
            natural_arity,
            compilation_info,
        })
    }
}

fn parse_link_attr(attr: Attribute) -> syn::Result<LinkAttr> {
    if attr.path().is_ident("link_name") {
        match &attr.meta {
            Meta::NameValue(name_value) => {
                require_link_name_str(&name_value.value)?;
                return Ok(LinkAttr {
                    attr,
                    cfg_pred: None,
                });
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    &attr,
                    "llexternal accepts `#[link_name = \"...\"]`, not another `link_name` form",
                ));
            }
        }
    }
    if attr.path().is_ident("cfg_attr") {
        let Meta::List(list) = &attr.meta else {
            return Err(syn::Error::new_spanned(
                &attr,
                "llexternal accepts `#[cfg_attr(<pred>, link_name = \"...\")]`",
            ));
        };
        let nested: Punctuated<Meta, Token![,]> =
            list.parse_args_with(Punctuated::parse_terminated)?;
        let mut metas = nested.into_iter();
        let Some(pred) = metas.next() else {
            return Err(syn::Error::new_spanned(
                &attr,
                "llexternal `#[cfg_attr]` needs a predicate and `link_name = \"...\"`",
            ));
        };
        let mut saw_link_name = false;
        for meta in metas {
            match &meta {
                Meta::NameValue(name_value) if name_value.path.is_ident("link_name") => {
                    require_link_name_str(&name_value.value)?;
                    saw_link_name = true;
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "llexternal `#[cfg_attr]` may only set `link_name = \"...\"`",
                    ));
                }
            }
        }
        if !saw_link_name {
            return Err(syn::Error::new_spanned(
                &attr,
                "llexternal `#[cfg_attr]` needs `link_name = \"...\"`",
            ));
        }
        return Ok(LinkAttr {
            attr,
            cfg_pred: Some(quote! { #pred }),
        });
    }
    Err(syn::Error::new_spanned(
        attr,
        "llexternal accepts only `#[link_name = \"...\"]` and `#[cfg_attr(<pred>, link_name = \"...\")]`",
    ))
}

fn require_link_name_str(expr: &Expr) -> syn::Result<()> {
    match expr {
        Expr::Lit(lit) if matches!(lit.lit, syn::Lit::Str(_)) => Ok(()),
        _ => Err(syn::Error::new_spanned(
            expr,
            "link_name value must be a string literal",
        )),
    }
}

fn parse_tri(input: ParseStream, key: &Ident) -> syn::Result<Tri> {
    if input.peek(LitBool) {
        return Ok(Tri::Bool(input.parse::<LitBool>()?.value()));
    }
    let ident: Ident = input.parse()?;
    if ident == "auto" {
        Ok(Tri::Auto)
    } else {
        Err(syn::Error::new(
            ident.span(),
            format!("`{key}` is `auto`, `true`, or `false`"),
        ))
    }
}

fn parse_signed_int(input: ParseStream) -> syn::Result<i64> {
    let neg = if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        true
    } else {
        false
    };
    let lit: LitInt = input.parse()?;
    let value = lit.base10_parse::<i64>()?;
    Ok(if neg { -value } else { value })
}

fn cfg_gate(pred: &TokenStream, items: TokenStream) -> TokenStream {
    if items.is_empty() {
        return items;
    }
    let file: syn::File = match syn::parse2(items) {
        Ok(file) => file,
        Err(err) => return err.to_compile_error(),
    };
    let mut out = TokenStream::new();
    for item in file.items {
        out.extend(quote! {
            #[cfg(#pred)]
            #item
        });
    }
    out
}

fn type_is_fn(ty: &Type) -> bool {
    match ty {
        Type::BareFn(_) => true,
        Type::Ptr(ptr) => type_is_fn(&ptr.elem),
        Type::Reference(reference) => type_is_fn(&reference.elem),
        Type::Paren(paren) => type_is_fn(&paren.elem),
        Type::Group(group) => type_is_fn(&group.elem),
        _ => false,
    }
}

/// `rffi.py` `RFFI_*` integer values. A path whose last segment is one of
/// these names resolves to that integer; the emitted `assert!(lit == expr)`
/// fails the build if the const drifts from this table.
fn rffi_flag(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int) => int.base10_parse().ok(),
            _ => None,
        },
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            rffi_flag(&unary.expr).map(|v| -v)
        }
        Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitOr(_)) => {
            Some(rffi_flag(&binary.left)? | rffi_flag(&binary.right)?)
        }
        Expr::Path(path) => {
            let name = path.path.segments.last()?.ident.to_string();
            Some(match name.as_str() {
                "RFFI_SAVE_ERRNO" => 1,
                "RFFI_READSAVED_ERRNO" => 2,
                "RFFI_ZERO_ERRNO_BEFORE" => 4,
                "RFFI_FULL_ERRNO" => 1 | 2,
                "RFFI_FULL_ERRNO_ZERO" => 1 | 4,
                "RFFI_SAVE_LASTERROR" => 8,
                "RFFI_READSAVED_LASTERROR" => 16,
                "RFFI_SAVE_WSALASTERROR" => 32,
                "RFFI_FULL_LASTERROR" => 8 | 16,
                "RFFI_ERR_NONE" => 0,
                "RFFI_ERR_ALL" => (1 | 2) | (8 | 16),
                "RFFI_ALT_ERRNO" => 64,
                _ => return None,
            })
        }
        Expr::Group(group) => rffi_flag(&group.expr),
        Expr::Paren(paren) => rffi_flag(&paren.expr),
        _ => None,
    }
}

impl LlexternalInput {
    fn expand(&self) -> TokenStream {
        if self.natural_arity < -1 || self.natural_arity > self.args.len() as i64 {
            return syn::Error::new(
                Span::call_site(),
                "natural_arity is -1, or the number of fixed parameters before `...`",
            )
            .to_compile_error();
        }
        let has_callback = self.args.iter().any(type_is_fn);
        let invoke = match self.releasegil {
            Tri::Bool(value) => value,
            Tri::Auto => !self.sandboxsafe && !self.nowrapper,
        };
        // rffi.py llexternal: random_effects_on_gcobjs='auto' becomes
        // invoke_around_handlers or has_callback, decided here because every
        // input is a literal.
        let random_effects = match self.random_effects {
            Tri::Bool(value) => value,
            Tri::Auto => invoke || has_callback,
        };
        let save_err_value = rffi_flag(&self.save_err);
        let save_err_nonzero = save_err_value.unwrap_or(1) != 0;
        let register_funcptr = invoke && self.macro_path.is_none() && self.natural_arity == -1;

        let name = &self.name;
        let deriv = self.derivation_consts(has_callback, invoke, random_effects);
        let eci = self.compilation_info.as_ref().map(|eci| {
            // One module declares several externals (`fcntl_int`, `fcntl_str`,
            // …) that share an ECI. A fixed const name collides; the name is
            // the declaration's, the value is still that ECI.
            let info_name = format_ident!("_COMPILATION_INFO_{}", name);
            quote! {
                #[allow(dead_code, non_upper_case_globals)]
                const #info_name: ::majit_rlib::rffi::ExternalCompilationInfo = #eci;
            }
        });

        let real = match self.expand_real(invoke, save_err_value, save_err_nonzero) {
            Ok(tokens) => {
                // The funcptr-valued marker names the graphless leaf the
                // real wrapper calls. The stub has no `__rffi_fp_*` extern
                // (`expand_stub`), so this rides with `expand_real`.
                let marker = self.random_effects_funcptr_marker(random_effects);
                quote! {
                    #tokens
                    #marker
                }
            }
            Err(tokens) => return tokens,
        };
        // `sandboxsafe = true` keeps one shape in every feature combination so
        // LLBC extracted under one feature set still matches the rustc body
        // (`rffi.py` `_safe_not_sandboxed`). `_nowrapper=True` returns the C
        // funcptr (`rffi.py` `llexternal`); a Rust stub body is Transparent in
        // LLBC and `CallControl.getcalldescr` then disagrees with the host_env
        // caller. Keep that opaque `extern` shape everywhere too.
        if self.sandboxsafe || self.nowrapper {
            return quote! {
                #deriv
                #eci
                #real
            };
        }
        let stub = self.expand_stub(register_funcptr);
        let (stub_cfg, real_cfg) = self.variant_cfgs();
        let real = cfg_gate(&real_cfg, real);
        let stub = cfg_gate(&stub_cfg, stub);
        quote! {
            #deriv
            #eci
            #real
            #stub
        }
    }

    /// Non-sandboxsafe only. Stub under `any(not(feature = "host_env"),
    /// feature = "sandbox")`; real under `all(feature = "host_env",
    /// not(feature = "sandbox"))` (`rffi.py` `_safe_not_sandboxed`,
    /// `node.py` `db.sandbox`). Evaluated in the invoking crate.
    fn variant_cfgs(&self) -> (TokenStream, TokenStream) {
        (
            quote! { any(not(feature = "host_env"), feature = "sandbox") },
            quote! { all(feature = "host_env", not(feature = "sandbox")) },
        )
    }

    fn expand_real(
        &self,
        invoke: bool,
        save_err_value: Option<i64>,
        save_err_nonzero: bool,
    ) -> Result<TokenStream, TokenStream> {
        if self.nowrapper {
            let funcptr = self.funcptr_item(true);
            let save_err = &self.save_err;
            return Ok(quote! {
                const _: () = assert!(#save_err == ::majit_rlib::rffi::RFFI_ERR_NONE);
                #funcptr
            });
        }

        let name = &self.name;
        let funcptr = self.funcptr_item(false);
        let call_name = format_ident!("ccall_{}", name);
        let (params, arg_names) = self.params();
        let result = &self.result;
        let call_expr = self.call_expr(&arg_names);
        let save_err = &self.save_err;
        // `rffi.py llexternal` closes the wrapper over `save_err`, so its
        // graph reads a Constant, not a global.
        let (save_err_const, save_err_assert) = match save_err_value {
            Some(lit) => (
                quote! { #lit },
                quote! { const _: () = assert!(#lit == #save_err); },
            ),
            None => (quote! { #save_err }, quote! {}),
        };

        let around_attr = if invoke && self.macro_path.is_none() && self.natural_arity == -1 {
            let Some(lit) = save_err_value else {
                return Err(syn::Error::new_spanned(
                        &self.save_err,
                        "save_err must be an integer literal or an RFFI_* const so call_aroundstate_target can take an integer literal",
                    )
                    .to_compile_error());
            };
            let fp = self.funcptr_path();
            let abi = self.extern_abi();
            quote! { #[::majit_macros::call_aroundstate_target(funcptr = #fp, save_err = #lit, abi = #abi)] }
        } else {
            quote! {}
        };

        let body = if invoke {
            quote! {
                // `rgil.release` / `rgil.acquire`: `before_external_block` also leaves the STW RUNNING census.
                let __guard = ::majit_gc::gc_sync::before_external_block();
                if #save_err_const != 0 {
                    ::majit_rlib::rposix::_errno_before(#save_err_const);
                }
                let __res = unsafe { #call_expr };
                if #save_err_const != 0 {
                    ::majit_rlib::rposix::_errno_after(#save_err_const);
                }
                drop(__guard);
                __res
            }
        } else {
            quote! {
                if #save_err_const != 0 {
                    ::majit_rlib::rposix::_errno_before(#save_err_const);
                }
                let __res = unsafe { #call_expr };
                if #save_err_const != 0 {
                    ::majit_rlib::rposix::_errno_after(#save_err_const);
                }
                __res
            }
        };

        // The around-handler body is one shape on every target. The
        // non-around body depends on `calling_conv`: the default is `"c"`
        // except on windows when a wrapper exists, where `llexternal` uses
        // `"unknown"` and forces `need_wrapper`.
        let direct_on_c = !invoke && !self.macro_path.is_some() && !save_err_nonzero;
        let specified_not_c = self.calling_conv.as_deref().is_some_and(|c| c != "c");
        let need_wrapper =
            !invoke && (self.macro_path.is_some() || save_err_nonzero || specified_not_c);

        if invoke || need_wrapper {
            let header = if invoke {
                // `jit_close_stack` is innermost so it sees the function;
                // `call_aroundstate_target` then sees that function plus the
                // close-stack marker and adds its own sibling const.
                quote! {
                    #[inline(never)]
                    #around_attr
                    #[::majit_macros::jit_close_stack]
                }
            } else {
                quote! { #[::majit_macros::dont_look_inside] }
            };
            let vis = &self.vis;
            let funcptr_fnaddr = if invoke && self.macro_path.is_none() && self.natural_arity == -1
            {
                self.funcptr_fnaddr_registration()
            } else {
                quote! {}
            };
            // The residual callee is `ccall_<name>` (`call_external_function`).
            // `jit_trace_fnaddrs` must carry that path or the funcbox stays a
            // symbolic hash and the tracer cannot run the wrapper.
            let ccall_fnaddr = self.ccall_fnaddr_registration();
            let identity = crate::icf_identity_tokens(&call_name, &syn::Generics::default());
            Ok(quote! {
                #save_err_assert
                #funcptr
                #funcptr_fnaddr
                #ccall_fnaddr
                #header
                #vis unsafe fn #call_name(#(#params),*) -> #result {
                    #identity
                    #body
                }
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_name(#(#arg_names),*) }
                }
            })
        } else if direct_on_c && self.calling_conv.is_none() {
            // unix: calling_conv "c" and no errno → call the funcptr.
            // windows: calling_conv "unknown" → wrapper (`need_wrapper`).
            // `dont_look_inside` is not applied under `cfg(windows)`: a proc-macro
            // attribute is expanded before cfg-elimination and would emit the
            // wrapper on every target.
            let vis = &self.vis;
            Ok(quote! {
                #funcptr
                #[cfg(not(windows))]
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
                #[cfg(windows)]
                #[inline(never)]
                #vis unsafe fn #call_name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
                #[cfg(windows)]
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_name(#(#arg_names),*) }
                }
            })
        } else {
            let vis = &self.vis;
            Ok(quote! {
                #funcptr
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
            })
        }
    }

    /// `rsandbox.py` `get_sandbox_stub`: rustc path panics; residual `ccall_`
    /// publishes and returns the zero of `R`. `fnname` is `fnobj._name`.
    /// `_nowrapper=True` real expansion is `extern <abi> { fn name }`; the stub
    /// uses that ABI so `CallControl.getcalldescr` sees one `FUNC.ARGS`.
    fn expand_stub(&self, register_funcptr: bool) -> TokenStream {
        let vis = &self.vis;
        let name = &self.name;
        let call_name = format_ident!("ccall_{}", name);
        let (params, _arg_names) = self.params();
        let result = &self.result;
        let fnname = &self.c_name;
        let save_err = &self.save_err;
        let look_name = format_ident!("_jit_look_inside_{}", name);
        let look_call = format_ident!("_jit_look_inside_{}", call_name);
        let ccall_fnaddr = self.ccall_fnaddr_registration();
        let funcptr_fnaddr = if register_funcptr {
            self.funcptr_fnaddr_registration_targeting(&call_name)
        } else {
            quote! {}
        };
        let abi = if self.nowrapper && self.macro_path.is_none() {
            let abi = self.extern_abi();
            quote! { extern #abi }
        } else {
            quote! {}
        };
        quote! {
            #[allow(unused_variables)]
            #vis unsafe #abi fn #name(#(#params),*) -> #result {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #look_name: bool = false;
                let _ = #save_err;
                ::majit_rlib::rffi::sandbox_stub_panic(#fnname)
            }
            #[allow(unused_variables)]
            #vis unsafe fn #call_name(#(#params),*) -> #result {
                #[doc(hidden)]
                #[allow(non_upper_case_globals, dead_code)]
                const #look_call: bool = false;
                ::majit_rlib::rffi::sandbox_stub_publish(#fnname);
                unsafe { ::core::mem::zeroed() }
            }
            #ccall_fnaddr
            #funcptr_fnaddr
        }
    }

    fn derivation_consts(
        &self,
        has_callback: bool,
        invoke: bool,
        random_effects: bool,
    ) -> TokenStream {
        let name = &self.name;
        let sandbox = format_ident!("_SANDBOXSAFE_{}", name);
        let nowrap = format_ident!("_NOWRAPPER_{}", name);
        let conv = format_ident!("_CALLING_CONV_{}", name);
        let callback = format_ident!("_HAS_CALLBACK_{}", name);
        let around = format_ident!("_INVOKE_AROUND_HANDLERS_{}", name);
        let elidable = format_ident!("_ELIDABLE_FUNCTION_{}", name);
        let effects = format_ident!("_RANDOM_EFFECTS_ON_GCOBJS_{}", name);
        let sandbox_v = self.sandboxsafe;
        let nowrap_v = self.nowrapper;
        let elidable_v = self.elidable;
        let callback_v = has_callback;
        let conv_expr = if let Some(conv) = &self.calling_conv {
            quote! { #conv }
        } else if self.nowrapper {
            quote! { "c" }
        } else {
            quote! {
                {
                    #[cfg(windows)]
                    { "unknown" }
                    #[cfg(not(windows))]
                    { "c" }
                }
            }
        };
        let effects_marker = self.random_effects_presence_marker(random_effects, &effects);
        quote! {
            #[allow(dead_code, non_upper_case_globals)]
            const #sandbox: bool = #sandbox_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #nowrap: bool = #nowrap_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #conv: &str = #conv_expr;
            #[allow(dead_code, non_upper_case_globals)]
            const #callback: bool = #callback_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #around: bool = {
                let _ = #conv;
                #invoke
            };
            #[allow(dead_code, non_upper_case_globals)]
            const #elidable: bool = #elidable_v;
            #effects_marker
            const _: () = assert!(!(#elidable && #effects));
        }
    }

    /// `_RANDOM_EFFECTS_ON_GCOBJS_{name}` from the same bool that gates the
    /// funcptr marker. The marker itself is emitted with the real wrapper.
    fn random_effects_presence_marker(&self, random_effects: bool, effects: &Ident) -> TokenStream {
        quote! {
            #[allow(dead_code, non_upper_case_globals)]
            const #effects: bool = #random_effects;
        }
    }

    /// Graphless leaf the wrapper calls: `__rffi_fp_{name}`, `{name}` when
    /// `_nowrapper`, the `macro = <path>` target when `macro` is given.
    fn graphless_leaf(&self) -> TokenStream {
        if let Some(path) = &self.macro_path {
            quote! { #path }
        } else if self.nowrapper {
            let name = &self.name;
            quote! { #name }
        } else {
            let fp = format_ident!("__rffi_fp_{}", self.name);
            quote! { #fp }
        }
    }

    /// `rffi.py llexternal` stores `random_effects_on_gcobjs` on the graphless
    /// C functionptr. Named after the llexternal `{name}`; the initializer is
    /// that leaf as a raw pointer so a variadic or `macro =` target typechecks.
    fn random_effects_funcptr_marker(&self, random_effects: bool) -> TokenStream {
        if !random_effects {
            return quote! {};
        }
        let vis = &self.vis;
        let marker = format_ident!("_random_effects_on_gcobjs_{}", self.name);
        let leaf = self.graphless_leaf();
        quote! {
            #[doc(hidden)]
            #[allow(non_upper_case_globals)]
            #vis const #marker: *const () = #leaf as *const ();
        }
    }

    fn params(&self) -> (Vec<TokenStream>, Vec<Ident>) {
        self.args
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let ident = format_ident!("a{i}");
                (quote! { #ident: #ty }, ident)
            })
            .unzip()
    }

    /// Publish the extern funcptr on `HELPER_FNADDRS`, the slice
    /// `jit_trace_fnaddrs` folds in (`merge_macro_helper_fnaddrs`). The path
    /// is `module_path!()::__rffi_fp_<name>`, which
    /// `register_macro_helper_trace_fnaddr` binds under the crate-stripped
    /// spelling `harvest_hints_from_llbcs` records for the funcptr.
    fn funcptr_fnaddr_registration(&self) -> TokenStream {
        let fp = format_ident!("__rffi_fp_{}", self.name);
        self.helper_fnaddr_registration(
            &fp,
            &fp,
            &format_ident!("__RFFI_FP_FNADDR_{}", self.name),
            &format_ident!("__rffi_register_fnaddr_{}", self.name),
        )
    }

    /// Same lookup path as [`Self::funcptr_fnaddr_registration`], pointing at
    /// the publishing `ccall_` so a host-extracted residual never panics.
    fn funcptr_fnaddr_registration_targeting(&self, target: &Ident) -> TokenStream {
        let fp = format_ident!("__rffi_fp_{}", self.name);
        self.helper_fnaddr_registration(
            &fp,
            target,
            &format_ident!("__RFFI_FP_FNADDR_{}", self.name),
            &format_ident!("__rffi_register_fnaddr_{}", self.name),
        )
    }

    /// Publish `ccall_<name>` the same way. The path is
    /// `module_path!()::ccall_<name>`, the residual callee
    /// `register_macro_helper_trace_fnaddr` binds.
    fn ccall_fnaddr_registration(&self) -> TokenStream {
        let call_name = format_ident!("ccall_{}", self.name);
        self.helper_fnaddr_registration(
            &call_name,
            &call_name,
            &format_ident!("__RFFI_CCALL_FNADDR_{}", self.name),
            &format_ident!("__rffi_register_ccall_fnaddr_{}", self.name),
        )
    }

    fn helper_fnaddr_registration(
        &self,
        path_func: &Ident,
        target_func: &Ident,
        static_name: &Ident,
        ctor_name: &Ident,
    ) -> TokenStream {
        let arity = u8::try_from(self.args.len()).unwrap_or(u8::MAX);
        quote! {
            #[cfg(not(target_arch = "wasm32"))]
            #[::majit_ir::linkme::distributed_slice(::majit_ir::helper_fnaddr::HELPER_FNADDRS)]
            #[linkme(crate = ::majit_ir::linkme)]
            #[allow(non_upper_case_globals)]
            static #static_name: ::majit_ir::helper_fnaddr::HelperFnAddr =
                ::majit_ir::helper_fnaddr::HelperFnAddr::new(
                    ::core::concat!(::core::module_path!(), "::", stringify!(#path_func)),
                    #target_func as *const (),
                    #arity,
                );
            #[cfg(target_arch = "wasm32")]
            #[::ctor::ctor(unsafe)]
            fn #ctor_name() {
                ::majit_ir::helper_fnaddr::register(
                    ::core::concat!(::core::module_path!(), "::", stringify!(#path_func)),
                    #target_func as *const (),
                    #arity,
                );
            }
        }
    }

    fn funcptr_path(&self) -> TokenStream {
        if let Some(path) = &self.macro_path {
            quote! { #path }
        } else {
            let fp = format_ident!("__rffi_fp_{}", self.name);
            quote! { #fp }
        }
    }

    fn call_expr(&self, arg_names: &[Ident]) -> TokenStream {
        let path = self.funcptr_path();
        quote! { #path(#(#arg_names),*) }
    }

    fn funcptr_item(&self, nowrapper: bool) -> TokenStream {
        if let Some(path) = &self.macro_path {
            let name = &self.name;
            let vis = &self.vis;
            return if nowrapper {
                quote! { #vis use #path as #name; }
            } else {
                quote! {}
            };
        }
        let result = &self.result;
        let rust_name = if nowrapper {
            self.name.clone()
        } else {
            format_ident!("__rffi_fp_{}", self.name)
        };
        let (fixed, variadic) = self.extern_params();
        let self_vis = &self.vis;
        let vis = if nowrapper {
            quote! { #self_vis }
        } else {
            quote! {}
        };
        let decl = if variadic && fixed.is_empty() {
            quote! { #vis fn #rust_name(...) -> #result; }
        } else if variadic {
            quote! { #vis fn #rust_name(#(#fixed),*, ...) -> #result; }
        } else {
            quote! { #vis fn #rust_name(#(#fixed),*) -> #result; }
        };
        let link_attrs = self.extern_link_attrs();
        let abi = self.extern_abi();
        quote! {
            unsafe extern #abi {
                #link_attrs
                #decl
            }
        }
    }

    /// The ABI string of the funcptr declaration. `calling_conv="win"` is
    /// `__stdcall` (`ll2ctypes` picks `ctypes.WinDLL`, `genc` emits
    /// `__stdcall`), which Rust spells `"system"`; every other convention is
    /// the C one.
    fn extern_abi(&self) -> LitStr {
        let abi = if self.calling_conv.as_deref() == Some("win") {
            "system"
        } else {
            "C"
        };
        LitStr::new(abi, Span::call_site())
    }

    /// An unconditional `#[link_name]` replaces `c_name`. Each `cfg_attr`
    /// link name is copied through, and `c_name` applies only where none of
    /// those predicates hold, so two `link_name` attributes are never both active.
    fn extern_link_attrs(&self) -> TokenStream {
        let c_name = &self.c_name;
        if self.link_attrs.is_empty() {
            return quote! { #[link_name = #c_name] };
        }
        let user = self.link_attrs.iter().map(|attr| &attr.attr);
        let unconditional = self.link_attrs.iter().any(|attr| attr.cfg_pred.is_none());
        if unconditional {
            quote! { #(#user)* }
        } else {
            let preds = self
                .link_attrs
                .iter()
                .filter_map(|attr| attr.cfg_pred.as_ref());
            quote! {
                #(#user)*
                #[cfg_attr(not(any(#(#preds),*)), link_name = #c_name)]
            }
        }
    }

    fn extern_params(&self) -> (Vec<TokenStream>, bool) {
        let variadic = self.natural_arity != -1;
        let fixed_n = if variadic {
            usize::try_from(self.natural_arity).unwrap_or(self.args.len())
        } else {
            self.args.len()
        };
        let n = fixed_n.min(self.args.len());
        let fixed = if variadic {
            self.args
                .iter()
                .take(n)
                .enumerate()
                .map(|(i, ty)| {
                    let ident = format_ident!("a{i}");
                    quote! { #ident: #ty }
                })
                .collect()
        } else {
            self.params().0
        };
        (fixed, variadic)
    }
}

const FORBIDDEN_NONEMPTY: &[&str] = &[
    "library_dirs",
    "link_extra",
    "separate_module_sources",
    "separate_module_files",
    "compile_extra",
];

struct EciInput {
    vis: Visibility,
    name: Ident,
    fields: Vec<(Ident, EciValue)>,
}

enum EciValue {
    Strs(Vec<LitStr>),
    Bool(bool),
}

impl Parse for EciInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vis = input.parse()?;
        input.parse::<Token![const]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let body;
        syn::braced!(body in input);
        let mut fields = Vec::new();
        while !body.is_empty() {
            let key: Ident = body.parse()?;
            body.parse::<Token![:]>()?;
            let value = if body.peek(LitBool) {
                EciValue::Bool(body.parse::<LitBool>()?.value())
            } else {
                let list;
                syn::bracketed!(list in body);
                let mut strs = Vec::new();
                while !list.is_empty() {
                    strs.push(list.parse::<LitStr>()?);
                    if list.is_empty() {
                        break;
                    }
                    list.parse::<Token![,]>()?;
                }
                EciValue::Strs(strs)
            };
            fields.push((key, value));
            let _ = body.parse::<Token![,]>();
        }
        let _ = input.parse::<Token![;]>();
        Ok(EciInput { vis, name, fields })
    }
}

impl EciInput {
    fn expand(&self) -> syn::Result<TokenStream> {
        let mut strings: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut use_cpp = false;
        for (key, value) in &self.fields {
            let name = key.to_string();
            match value {
                EciValue::Bool(value) => {
                    if name != "use_cpp_linker" {
                        return Err(syn::Error::new(key.span(), "only use_cpp_linker is a bool"));
                    }
                    use_cpp = *value;
                }
                EciValue::Strs(strs) => {
                    if !ECI_FIELDS.contains(&name.as_str()) {
                        return Err(syn::Error::new(
                            key.span(),
                            format!("unknown ExternalCompilationInfo field `{name}`"),
                        ));
                    }
                    let values: Vec<String> = strs.iter().map(LitStr::value).collect();
                    if FORBIDDEN_NONEMPTY.contains(&name.as_str()) && !values.is_empty() {
                        let message = format!(
                            "`{name}` needs a C build step or a Cargo search path; it must be empty"
                        );
                        return Ok(syn::Error::new(key.span(), message).to_compile_error());
                    }
                    strings.insert(name, values);
                }
            }
        }
        let vis = &self.vis;
        let name = &self.name;
        let field_inits = ECI_FIELDS.iter().map(|field| {
            let ident = Ident::new(field, Span::call_site());
            let values = strings.get(*field).cloned().unwrap_or_default();
            quote! { #ident: &[#(#values),*] }
        });
        let mut links = TokenStream::new();
        for lib in strings.get("libraries").into_iter().flatten() {
            links.extend(quote! {
                #[link(name = #lib)]
                unsafe extern "C" {}
            });
        }
        for framework in strings.get("frameworks").into_iter().flatten() {
            links.extend(quote! {
                #[link(name = #framework, kind = "framework")]
                unsafe extern "C" {}
            });
        }
        Ok(quote! {
            #vis const #name: ::majit_rlib::rffi::ExternalCompilationInfo =
                ::majit_rlib::rffi::ExternalCompilationInfo {
                    #(#field_inits,)*
                    use_cpp_linker: #use_cpp,
                };
            #links
        })
    }
}

pub fn expand_jit_close_stack(item: TokenStream) -> TokenStream {
    match close_stack_marker(item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

fn item_cfg_attrs(func: &syn::ItemFn) -> Vec<&syn::Attribute> {
    func.attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg"))
        .collect()
}

fn close_stack_marker(item: TokenStream) -> syn::Result<TokenStream> {
    let file = syn::parse2::<syn::File>(item)?;
    let func = file.items.iter().find_map(|item| match item {
        syn::Item::Fn(func) => Some(func),
        _ => None,
    });
    let Some(func) = func else {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[jit_close_stack] supports free functions",
        ));
    };
    let vis = &func.vis;
    let cfg_attrs = item_cfg_attrs(func);
    let marker = format_ident!("_gctransformer_hint_close_stack_{}", func.sig.ident);
    Ok(quote! {
        #file
        #(#cfg_attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        #vis const #marker: bool = true;
    })
}

pub fn expand_call_aroundstate_target(attr: TokenStream, item: TokenStream) -> TokenStream {
    match aroundstate_marker(attr, item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

struct AroundStateAttr {
    funcptr: syn::Path,
    save_err: LitInt,
    abi: Option<LitStr>,
}

impl Parse for AroundStateAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut funcptr = None;
        let mut save_err = None;
        let mut abi = None;
        while !input.is_empty() {
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "funcptr" => funcptr = Some(input.parse()?),
                "save_err" => save_err = Some(input.parse()?),
                "abi" => abi = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown call_aroundstate_target argument `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }
        Ok(AroundStateAttr {
            funcptr: funcptr.ok_or_else(|| input.error("missing funcptr"))?,
            save_err: save_err.ok_or_else(|| input.error("missing save_err"))?,
            abi,
        })
    }
}

fn aroundstate_marker(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: AroundStateAttr = syn::parse2(attr)?;
    let file = syn::parse2::<syn::File>(item)?;
    let func = file.items.iter().find_map(|item| match item {
        syn::Item::Fn(func) => Some(func),
        _ => None,
    });
    let Some(func) = func else {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[call_aroundstate_target] supports free functions",
        ));
    };
    let vis = &func.vis;
    let marker = format_ident!("_call_aroundstate_target_{}", func.sig.ident);
    let arg_tys = func.sig.inputs.iter().map(|arg| match arg {
        syn::FnArg::Typed(pat) => Ok(&pat.ty),
        syn::FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
            receiver,
            "#[call_aroundstate_target] supports free functions",
        )),
    });
    let mut tys = Vec::new();
    for ty in arg_tys {
        tys.push(ty?);
    }
    let ret = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };
    let funcptr = &args.funcptr;
    let save_err = &args.save_err;
    let abi = args
        .abi
        .unwrap_or_else(|| LitStr::new("C", Span::call_site()));
    let cfg_attrs = item_cfg_attrs(func);
    Ok(quote! {
        #file
        #(#cfg_attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        #vis const #marker: (unsafe extern #abi fn(#(#tys),*) -> #ret, i64) =
            (#funcptr, #save_err);
    })
}

const ECI_FIELDS: &[&str] = &[
    "pre_include_bits",
    "includes",
    "include_dirs",
    "post_include_bits",
    "libraries",
    "library_dirs",
    "separate_module_sources",
    "separate_module_files",
    "compile_extra",
    "link_extra",
    "frameworks",
    "link_files",
    "testonly_libraries",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(input: &str) -> String {
        expand_llexternal(input.parse().unwrap())
            .unwrap()
            .to_string()
    }

    #[test]
    fn releasegil_emits_funcptr_valued_marker() {
        let out = expand(r#"c_foo = "foo", [i32], i32"#);
        assert!(out.contains("_random_effects_on_gcobjs_c_foo"), "{out}");
        assert!(out.contains("__rffi_fp_c_foo as * const ()"), "{out}");
        assert!(out.contains("_RANDOM_EFFECTS_ON_GCOBJS_c_foo"), "{out}");
        let no_gil = expand(
            r#"c_memcpy = "memcpy", [*mut u8, *const u8, usize], *mut u8, releasegil = false"#,
        );
        assert!(!no_gil.contains("_random_effects_on_gcobjs_"), "{no_gil}");
        let foreign = expand(r#"c_stat = "stat", [i32], i32, macro = libc::stat"#);
        assert!(
            foreign.contains("_random_effects_on_gcobjs_c_stat"),
            "{foreign}"
        );
        assert!(foreign.contains("libc :: stat as * const ()"), "{foreign}");
    }
}
