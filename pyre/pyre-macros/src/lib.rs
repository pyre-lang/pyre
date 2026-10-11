//! `pyre-macros` — attribute proc macros that put the PyPy
//! `@unwrap_spec`-style argument-conversion plumbing behind the same
//! single Rust attribute the function definition already needs.
//!
//! ```ignore
//! #[pyre_function]
//! fn stack_effect(opcode: i64) -> i64 {
//!     // typed body: takes a real i64, returns a real i64.
//!     0
//! }
//! ```
//!
//! emits a wrapper that the `py_module! { functions: { ... } }` table
//! can reference by name:
//!
//! ```ignore
//! fn stack_effect(args: &[pyre_object::PyObjectRef])
//!     -> Result<pyre_object::PyObjectRef, ::pyre_interpreter::PyError>
//! {
//!     let opcode: i64 = pyre_interpreter::baseobjspace::gateway_int_w(args[0])?;
//!     Ok(pyre_object::w_int_new(__stack_effect_user(opcode)))
//! }
//!
//! fn __stack_effect_user(opcode: i64) -> i64 { 0 }
//! ```
//!
//! Supported parameter types (per-position):
//! * `i64` / `i32` / `u32` / `usize` — `gateway_int_w` + cast.
//! * `f64` — `float_w`.
//! * `bool` — `is_true`.
//! * `&str` — `str_utf8_w` (a lone surrogate raises `UnicodeEncodeError`).
//! * `pyre_object::PyObjectRef` — passthrough (`args[i]`).
//! * `&[pyre_object::PyObjectRef]` — passthrough of the whole slice (varargs).
//!
//! Supported return types:
//! * `i64` / `i32` / `u32` / `usize` — `w_int_new`.
//! * `f64` — `w_float_new`.
//! * `bool` — `w_bool_from`.
//! * `String` — `w_str_new_managed`.
//! * `pyre_object::PyObjectRef` — passthrough.
//! * `Result<T, ::pyre_interpreter::PyError>` — `?`-propagated, then `T` wrapped.
//! * `()` — `w_none()`.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Fields, FnArg, ImplItem, ItemFn, ItemImpl, ItemStruct, Pat, PatType, ReturnType, Type,
    parse_macro_input, parse_quote, spanned::Spanned,
};

#[proc_macro_attribute]
pub fn pyre_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand_pyre_function(func) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_pyre_function(func: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let vis = &func.vis;
    let user_name = func.sig.ident.clone();
    let user_attrs = &func.attrs;
    let user_body = &func.block;
    let user_sig = &func.sig;
    let inner_name = format_ident!("__{}_impl", user_name);

    // Build typed inner fn — verbatim original body, just renamed.
    let mut inner_sig = user_sig.clone();
    inner_sig.ident = inner_name.clone();

    // Generate unwrap statements for each parameter, and collect the
    // parameter-name / required tables so the wrapper can resolve keyword
    // arguments by name (PyPy gateway `Signature` + `_match_signature`).
    let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut reread_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut call_args = Vec::<proc_macro2::TokenStream>::new();
    let mut param_names = Vec::<String>::new();
    let mut param_required = Vec::<bool>::new();
    let mut param_positional = Vec::<bool>::new();
    let mut has_varargs = false;
    let mut has_kw_markers = false;
    let mut kwonly_tail = false;
    let user_name_str = user_name.to_string();
    let reads: Vec<ArgRead> = user_sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pt) => Some(arg_read(&pt.ty)),
            FnArg::Receiver(_) => None,
        })
        .collect();
    let rooted = args_need_roots(&reads, false);
    for (idx, arg) in user_sig.inputs.iter().enumerate() {
        let pat_type = match arg {
            FnArg::Typed(pt) => pt,
            FnArg::Receiver(r) => {
                return Err(syn::Error::new(
                    r.span(),
                    "#[pyre_function] cannot wrap methods (no `self` arg)",
                ));
            }
        };
        if is_varargs_param(&pat_type.ty) {
            has_varargs = true;
        }
        let is_kwonly = pat_type.attrs.iter().any(|a| a.path().is_ident("kwonly"));
        let is_kwargs = pat_type.attrs.iter().any(|a| a.path().is_ident("kwargs"));
        if is_kwonly {
            kwonly_tail = true;
        }
        if is_kwonly || is_kwargs {
            has_kw_markers = true;
        }
        param_names.push(param_name(idx, pat_type));
        // Optional iff it has a `#[default(...)]` or is `Option<T>`.
        param_required
            .push(arg_default(pat_type)?.is_none() && option_inner(&pat_type.ty).is_none());
        param_positional.push(!kwonly_tail && !is_kwargs);
        let (unwrap, reread, ident) =
            unwrap_arg(idx, pat_type, Some((&user_name_str, idx + 1)), rooted)?;
        unwrap_stmts.push(unwrap);
        reread_stmts.extend(reread);
        call_args.push(quote! { #ident });
    }

    reject_duplicate_param_names(&param_names, user_name.span())?;

    // Keyword-binding preamble: when the call carried a `__pyre_kw__`
    // dict, rebind positional+keyword args into a resolved scope keyed by
    // parameter name; otherwise the positional fast path runs unchanged.
    // Skipped for varargs fns — `&[PyObjectRef]` consumes the whole slice
    // (including any trailing kwargs dict), which the resolved-scope shape
    // cannot express.
    let kwargs_preamble = if has_varargs {
        quote! {}
    } else {
        let name_lits = param_names.iter().map(|n| quote! { #n });
        let req_lits = param_required.iter().map(|b| quote! { #b });
        let fn_name_str = user_name.to_string();
        quote! {
            let __pyre_has_kwargs = ::pyre_interpreter::builtins::has_builtin_kwargs(args);
            // `bind_kwargs_to_signature` pads `args` out to the full
            // parameter count with PY_NULL, so keyword-only slots and absent
            // optionals would otherwise be counted as positional.  PY_NULL is
            // never a real argument, so the leading non-null run is the true
            // positional count on both the bound and the raw path.  The
            // marker, when present, is the last slot and is not part of that
            // run; capping there is `split_builtin_kwargs`'s prefix without
            // building the sub-slice.
            let __pyre_positional_count =
                ::pyre_interpreter::builtins::builtin_positional_count(args, __pyre_has_kwargs);
            const __PYRE_PARAM_NAMES: &[&str] = &[ #(#name_lits),* ];
            const __PYRE_PARAM_REQUIRED: &[bool] = &[ #(#req_lits),* ];
            let __pyre_bound_args;
            let args: &[::pyre_object::PyObjectRef] =
                if __pyre_has_kwargs {
                    __pyre_bound_args = ::pyre_interpreter::builtins::bind_builtin_kwargs(
                        args,
                        __PYRE_PARAM_NAMES,
                        __PYRE_PARAM_REQUIRED,
                        #fn_name_str,
                    )?;
                    &__pyre_bound_args
                } else {
                    args
                };
        }
    };

    // `Arguments.parse_obj` rejects a positional count that does not fit the
    // signature before `BuiltinCode.funcrun` enters the unwrapped body.  The
    // unwrap statements below index their slots directly, so the count is
    // checked here — a missing required slot would otherwise read past the
    // end of `args`.  A raw whole-args slice takes any count.
    // A signature with kw-only / **kwargs markers is matched by the gateway
    // before this wrapper runs.  The resulting scope contains every
    // positional and kw-only slot, so `args.len()` is no longer the number
    // of positionals supplied by the caller.  PyPy's
    // `BuiltinCode.funcrun_obj` likewise runs the activation directly after
    // `args.parse_obj(...)`; do not perform a second positional-count check
    // on that already-bound scope.
    let count_preamble = if has_varargs || has_kw_markers {
        quote! {}
    } else {
        let total = param_positional
            .iter()
            .filter(|&&positional| positional)
            .count();
        let required = param_required
            .iter()
            .zip(param_positional.iter())
            .filter(|(required, positional)| **required && **positional)
            .count();
        let fn_name = user_name.to_string();
        let too_few = arity_message(&fn_name, required, total, false);
        let too_many = arity_message(&fn_name, required, total, true);
        let too_many_guard = if has_kw_markers {
            // `gateway.py:BuiltinCode.funcrun` first resolves a Signature
            // into one slot per positional / keyword-only parameter (plus
            // the **kwargs tail).  Such a bound scope has
            // `__PYRE_PARAM_NAMES.len()` entries even though only `total`
            // of them are positional.  Do not count those trailing slots as
            // supplied positionals; the Signature binder has already
            // rejected genuine positional overflow before entering here.
            quote! {
                if __pyre_positional_count > #total as i64
                    && args.len() != __PYRE_PARAM_NAMES.len()
                {
                    return ::std::result::Result::Err(::pyre_interpreter::PyError::type_error(#too_many));
                }
            }
        } else {
            quote! {
                if __pyre_positional_count > #total as i64 {
                    return ::std::result::Result::Err(::pyre_interpreter::PyError::type_error(#too_many));
                }
            }
        };
        quote! {
            if __pyre_positional_count < #required as i64 {
                return ::std::result::Result::Err(::pyre_interpreter::PyError::type_error(#too_few));
            }
            #too_many_guard
        }
    };

    let call_inner = quote! { #inner_name( #(#call_args),* ) };
    let body = wrap_return(&user_sig.output, call_inner)?;

    // Strip `#[default(expr)]` arg attributes + rewrite typed-receiver
    // aliases (PyTuple → PyObjectRef) so rustc accepts the emitted inner
    // fn.  The wrapper above has already consumed both transforms.
    let mut stripped = inner_sig.clone();
    for arg in stripped.inputs.iter_mut() {
        if let FnArg::Typed(pt) = arg {
            pt.attrs.retain(|a| {
                !a.path().is_ident("default")
                    && !a.path().is_ident("posonly")
                    && !a.path().is_ident("kwonly")
                    && !a.path().is_ident("kwargs")
            });
        }
    }
    rewrite_alias_args(&mut stripped);
    let inner_fn = quote! {
        #(#user_attrs)*
        #[allow(non_snake_case)]
        #[inline]
        #stripped #user_body
    };

    let (roots_open, roots_close) = arg_roots_bracket(rooted);
    let wrapper = quote! {
        #[allow(non_snake_case)]
        #vis fn #user_name(
            args: &[::pyre_object::PyObjectRef],
        ) -> ::std::result::Result<::pyre_object::PyObjectRef, ::pyre_interpreter::PyError> {
            #kwargs_preamble
            #count_preamble
            #roots_open
            #(#unwrap_stmts)*
            #(#reread_stmts)*
            #roots_close
            #body
        }
    };

    // Companion `<name>_pyre_sig()` — derives a keyword-aware `Signature`
    // from the typed parameter names so the registration path can bind
    // keyword / kw-only / `**kwargs`-style calls by name.  A `#[kwonly]`
    // marker starts the keyword-only tail at that parameter; a `#[kwargs]`
    // marker designates the `**kwargs` catch-all dict.
    //
    // A bare `&[PyObjectRef]` parameter is the raw whole-args-slice
    // passthrough ABI (the wrapper hands it the entire `args` slice), which
    // is incompatible with by-name binding — such a builtin gets no
    // `Signature` (`None`) and keeps the raw positional fast path.
    let sig_fn_name = format_ident!("{}_pyre_sig", user_name);
    let mut sig_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut posonly_marked = false;
    let mut kwonly_marked = false;
    let mut raw_slice = false;
    for arg in user_sig.inputs.iter() {
        let FnArg::Typed(pt) = arg else { continue };
        let name_lit = match &*pt.pat {
            Pat::Ident(pi) => python_param_keyword(&pi.ident.to_string()),
            _ => continue,
        };
        if pt.attrs.iter().any(|a| a.path().is_ident("kwargs")) {
            sig_stmts.push(quote! { __b.kwargname = ::std::option::Option::Some(#name_lit); });
            continue;
        }
        // Only a `&[PyObjectRef]` slice is the raw whole-args passthrough
        // that suppresses the signature.  A `&[u8]` (or other element type)
        // is a positioned bytes-like parameter and keeps by-name binding.
        let is_slice = match unwrap_type_group(&pt.ty) {
            Type::Reference(r) => match unwrap_type_group(&r.elem) {
                Type::Slice(s) => type_is_py_object_ref(unwrap_type_group(&s.elem)),
                _ => false,
            },
            _ => false,
        };
        if is_slice {
            raw_slice = true;
            continue;
        }
        if !posonly_marked && pt.attrs.iter().any(|a| a.path().is_ident("posonly")) {
            sig_stmts.push(quote! { __b.marker_posonly(); });
            posonly_marked = true;
        }
        if !kwonly_marked && pt.attrs.iter().any(|a| a.path().is_ident("kwonly")) {
            sig_stmts.push(quote! { __b.marker_kwonly(); });
            kwonly_marked = true;
        }
        sig_stmts.push(quote! { __b.append(#name_lit); });
    }
    let sig_body = if raw_slice {
        quote! { ::std::option::Option::None }
    } else {
        quote! {
            let mut __b = ::pyre_interpreter::SignatureBuilder::default();
            #(#sig_stmts)*
            ::std::option::Option::Some(__b.signature())
        }
    };
    let sig_fn = quote! {
        #[allow(dead_code, non_snake_case)]
        #vis fn #sig_fn_name() -> ::std::option::Option<::pyre_interpreter::Signature> {
            #sig_body
        }
    };

    // Companion `<name>_pyre_arity()` — the `fast_natural_arity` this
    // builtin may declare.  `BuiltinCode.fast_natural_arity` names the one
    // argument count at which the body may be entered with its slots filled
    // directly, so it exists only when every parameter is required: an
    // optional parameter (`#[default(...)]` or `Option<T>`), a keyword-only
    // or `**kwargs` marker and a raw whole-args slice all admit more than one
    // count, and get `HOPELESS`.
    let arity_fn_name = format_ident!("{}_pyre_arity", user_name);
    let natural_arity =
        if has_varargs || has_kw_markers || param_required.iter().any(|required| !required) {
            quote! { ::pyre_interpreter::HOPELESS }
        } else {
            let n = param_positional
                .iter()
                .filter(|&&positional| positional)
                .count() as u16;
            quote! { #n }
        };
    let arity_fn = quote! {
        #[allow(dead_code, non_snake_case)]
        #vis fn #arity_fn_name() -> u16 {
            #natural_arity
        }
    };

    Ok(quote! {
        #inner_fn
        #wrapper
        #sig_fn
        #arity_fn
    })
}

/// How a wrapper parameter consumes its `args` slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgRead {
    /// Binds the slot's object itself (`PyObjectRef`, `Option<PyObjectRef>`,
    /// a type-checked passthrough alias such as `PyTuple`).
    Ref,
    /// Runs a gateway conversion (`int_w`, `text_w`, `fsencode_w`, ...),
    /// which can call back into Python and collect.
    Convert,
    /// Takes the whole `args` slice.
    WholeSlice,
}

fn is_passthrough_alias(name: &str) -> bool {
    matches!(
        name,
        "PyTuple"
            | "PyList"
            | "PyDict"
            | "PyStr"
            | "PyBytes"
            | "PyByteArray"
            | "PyInt"
            | "PyFloat"
            | "PyBool"
            | "PySet"
            | "PyFrozenSet"
    )
}

fn arg_read(ty: &Type) -> ArgRead {
    let ty = unwrap_type_group(ty);
    if is_varargs_param(ty) {
        return ArgRead::WholeSlice;
    }
    if type_is_py_object_ref(ty) {
        return ArgRead::Ref;
    }
    if let Some(inner) = option_inner(ty) {
        return arg_read(inner);
    }
    if let Type::Path(p) = ty
        && let Some(seg) = p.path.segments.last()
        && is_passthrough_alias(&seg.ident.to_string())
    {
        return ArgRead::Ref;
    }
    ArgRead::Convert
}

/// Whether the wrapper must publish `args` on the shadow stack while it
/// unwraps them: `interp2app`'s activation reads every argument out of
/// `scope_w`, a GC-visible list, so a conversion that collects never leaves
/// an earlier or later argument stale.  The `args` slice here is a native
/// copy, so any conversion followed by another read of a slot, or preceded
/// by a bound object (the receiver included), needs the slots rooted.
fn args_need_roots(reads: &[ArgRead], leading_ref: bool) -> bool {
    reads.iter().enumerate().any(|(p, r)| {
        *r == ArgRead::Convert
            && (leading_ref
                || reads[..p].contains(&ArgRead::Ref)
                || reads[p + 1..].iter().any(|q| *q != ArgRead::WholeSlice))
    })
}

/// The shadow-stack bracket around a rooted wrapper's unwraps.  It closes
/// before the call: the callee roots its own locals.
fn arg_roots_bracket(rooted: bool) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    if rooted {
        (
            quote! {
                let __pyre_arg_roots = ::pyre_object::gc_roots::push_roots();
                let __pyre_arg_base = __pyre_arg_roots.pin_roots(args);
            },
            quote! { ::std::mem::drop(__pyre_arg_roots); },
        )
    } else {
        (quote! {}, quote! {})
    }
}

/// The value read of `args[idx]`: the shadow-stack slot when the wrapper
/// rooted `args`, the native slice otherwise.
fn arg_slot(idx: usize, rooted: bool) -> proc_macro2::TokenStream {
    if rooted {
        quote! { __pyre_arg_roots.get(__pyre_arg_base + #idx) }
    } else {
        quote! { args[#idx] }
    }
}

/// Generate `let <ident>: <T> = <unwrap-from-args[idx]>;`.
///
/// `#[default(expr)]` on an arg substitutes `expr` whenever
/// `args.len() <= idx`.  Mirrors PyPy `@unwrap_spec(w_x=WrappedDefault(v))`
/// — when the caller omits a positional, the wrapper synthesises a value
/// in the user's typed coordinate space, not in `PyObjectRef` space.
///
/// With `rooted`, the second result re-reads a bound object from its slot
/// once every conversion has run.  A `#[default]` object is not in that
/// slice: the binding pins the word it produced, and the re-read loads the pin.
fn unwrap_arg(
    idx: usize,
    pt: &PatType,
    missing_ctx: Option<(&str, usize)>,
    rooted: bool,
) -> syn::Result<(
    proc_macro2::TokenStream,
    Option<proc_macro2::TokenStream>,
    syn::Ident,
)> {
    let ident = match &*pt.pat {
        Pat::Ident(pi) => pi.ident.clone(),
        _ => format_ident!("__pyre_arg{}", idx),
    };
    let ty = &*pt.ty;

    let slot = arg_slot(idx, rooted);
    let unwrap = unwrap_expr(ty, &slot, idx)?;
    let default = arg_default(pt)?;
    let read = arg_read(ty);
    // A defaulted object is not an `args` slot.  Pin that word before the
    // later conversions collect, and reload it afterwards.
    let pin_defaulted_ref = rooted && default.is_some() && read == ArgRead::Ref;
    let (pin_stmt, reread) = if pin_defaulted_ref {
        let slot_ident = format_ident!("__pyre_{}_slot", ident);
        let is_option = option_inner(unwrap_type_group(ty)).is_some();
        let (presence, word, reread_value) = if is_option {
            let some_ident = format_ident!("__pyre_{}_some", ident);
            (
                quote! { let #some_ident = #ident.is_some(); },
                quote! { #ident.unwrap_or(::pyre_object::PY_NULL) },
                quote! {
                    if #some_ident {
                        ::std::option::Option::Some(__pyre_arg_roots.get(#slot_ident))
                    } else {
                        ::std::option::Option::None
                    }
                },
            )
        } else {
            (
                quote! {},
                quote! { #ident },
                quote! { __pyre_arg_roots.get(#slot_ident) },
            )
        };
        (
            quote! {
                #presence
                let #slot_ident = __pyre_arg_roots.pin_roots(&[#word]);
            },
            Some(quote! { let #ident = #reread_value; }),
        )
    } else {
        let reread = match read {
            ArgRead::Ref if rooted => {
                let value = if option_inner(unwrap_type_group(ty)).is_some() {
                    quote! { ::std::option::Option::Some(#slot) }
                } else {
                    quote! { #slot }
                };
                Some(quote! {
                    let #ident = if #idx < args.len() && !args[#idx].is_null() { #value } else { #ident };
                })
            }
            // The whole slice is the native copy; rebuild it from the slots.
            ArgRead::WholeSlice if rooted => Some(quote! {
                let mut __pyre_live_args: ::std::vec::Vec<::pyre_object::PyObjectRef> =
                    ::std::vec::Vec::with_capacity(args.len());
                let mut __pyre_live_i = 0usize;
                while __pyre_live_i < args.len() {
                    __pyre_live_args.push(__pyre_arg_roots.get(__pyre_arg_base + __pyre_live_i));
                    __pyre_live_i += 1;
                }
                let #ident: &[::pyre_object::PyObjectRef] = &__pyre_live_args;
            }),
            _ => None,
        };
        (quote! {}, reread)
    };
    // A `&[PyObjectRef]` whole-slice parameter binds the entire `args`
    // slice — it has no per-slot index to bounds-check.  Other slice
    // element types (e.g. `&[u8]`) are positioned params indexing
    // `args[idx]`, so they keep the missing-argument bounds guard.
    let is_whole_slice = if let Type::Reference(r) = unwrap_type_group(ty) {
        if let Type::Slice(s) = unwrap_type_group(&r.elem) {
            type_is_py_object_ref(unwrap_type_group(&s.elem))
        } else {
            false
        }
    } else {
        false
    };
    let argname = ident.to_string();
    // An `Option<T>` parameter carries its own absence: `unwrap_expr` yields
    // `None` for an out-of-range or PY_NULL slot, so the slot is optional and
    // gets no missing-argument guard.
    let is_optional = option_inner(ty).is_some();
    let expr = match default {
        // `bind_kwargs_to_signature` pads `args` to the parameter count
        // with PY_NULL, so the default only applies when the slot is both
        // present and non-null.
        Some(default) => quote! {
            if #idx < args.len() && !args[#idx].is_null() { #unwrap } else { #default }
        },
        None if is_whole_slice || is_optional => unwrap,
        None => {
            // getargs.c `missing required argument` for a required slot left
            // unfilled.  The signature-bearing `#[pyre_function]` path reports
            // the function name and 1-based position (`{name}() missing
            // required argument '{arg}' (pos {n})`); other callers keep the
            // bare form.
            let missing_err = match missing_ctx {
                Some((fname, pos)) => quote! {
                    format!(
                        "{}() missing required argument '{}' (pos {})",
                        #fname, #argname, #pos
                    )
                },
                None => quote! { format!("missing required argument: '{}'", #argname) },
            };
            quote! {
                {
                    if #idx >= args.len() || args[#idx].is_null() {
                        return ::std::result::Result::Err(
                            ::pyre_interpreter::PyError::type_error(#missing_err)
                        );
                    }
                    #unwrap
                }
            }
        }
    };
    // Typed-receiver aliases erase to the binding type declared in
    // `typed_alias` (PyObjectRef for passthrough, Vec<u8> for PyPath).
    // The inner fn signature is rewritten in parallel by
    // `rewrite_alias_args` so user code sees the same type.
    let binding_ty = if let Type::Path(p) = unwrap_type_group(ty) {
        if let Some(seg) = p.path.segments.last() {
            typed_alias_binding_ty(&seg.ident.to_string()).unwrap_or_else(|| quote! { #ty })
        } else {
            quote! { #ty }
        }
    } else {
        quote! { #ty }
    };
    Ok((
        quote! {
            let #ident: #binding_ty = #expr;
            #pin_stmt
        },
        reread,
        ident,
    ))
}

/// Substitute typed-receiver aliases in a fn signature with the rust
/// type the alias resolves to.  Applied to inner fns (`#[pyre_function]`)
/// and impl-block methods (`#[pyre_methods]`) so the user-visible alias
/// name (`PyTuple` / `PyPath`) is erased to a real Rust type before the
/// body compiles.
fn rewrite_alias_args(sig: &mut syn::Signature) {
    for arg in sig.inputs.iter_mut() {
        let FnArg::Typed(pt) = arg else { continue };
        let ty = unwrap_type_group(&pt.ty);
        let Type::Path(p) = ty else { continue };
        let Some(seg) = p.path.segments.last() else {
            continue;
        };
        if let Some(binding_ty) = typed_alias_binding_ty(&seg.ident.to_string()) {
            *pt.ty = syn::parse2(binding_ty).expect("alias binding ty parses");
        }
    }
}

/// Extract the inner expression from `#[default(expr)]` on a fn arg.
fn arg_default(pt: &PatType) -> syn::Result<Option<proc_macro2::TokenStream>> {
    for a in pt.attrs.iter() {
        if !a.path().is_ident("default") {
            continue;
        }
        let expr: syn::Expr = a.parse_args().map_err(|e| {
            syn::Error::new(
                a.span(),
                format!("#[default(...)]: expected a single expression — {e}"),
            )
        })?;
        return Ok(Some(quote! { #expr }));
    }
    Ok(None)
}

/// Resolve a typed-receiver alias name to its `(binding_type,
/// unwrap_expr_template)` pair.  Two flavours of alias coexist:
///
/// * Passthrough-with-typecheck (`PyTuple`, `PyList`, …) — mirrors PyPy
///   `@unwrap_spec(w_x=W_TupleObject)` which `gateway.py`
///   lowers to `space.interp_w(W_TupleObject, scope_w[i])`.  Binding
///   type stays `PyObjectRef`; the wrapper just emits an `is_X` check
///   and bails out with a TypeError on mismatch.
///
/// * Convert-to-rust (`PyPath`) — mirrors PyPy
///   `@unwrap_spec(path='fsencode')` (`gateway.py:visit_fsencode` line
///   365) which lowers to `space.fsencode_w(...)`.  Binding type is the
///   Rust value the conversion produces (`Vec<u8>` for PyPath); the
///   inner-fn signature is rewritten the same way and the body sees the
///   converted Rust value directly.
///
/// `__a` and `__idx` are the wrapper-local names the template must
/// bind/refer to so the caller can splice the chosen index in.
fn typed_alias(
    name: &str,
    slot: &proc_macro2::TokenStream,
    idx: usize,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let passthrough = |check: proc_macro2::TokenStream| {
        (
            quote! { ::pyre_object::PyObjectRef },
            quote! {
                {
                    let __a = #slot;
                    if !unsafe { #check(__a) } {
                        return ::std::result::Result::Err(
                            ::pyre_interpreter::PyError::type_error(format!(
                                "argument {} must be {}", #idx, #name
                            )),
                        );
                    }
                    __a
                }
            },
        )
    };
    Some(match name {
        "PyTuple" => passthrough(quote! { ::pyre_object::is_tuple }),
        "PyList" => passthrough(quote! { ::pyre_object::is_list }),
        "PyDict" => passthrough(quote! { ::pyre_object::is_dict }),
        "PyStr" => passthrough(quote! { ::pyre_object::is_str }),
        "PyBytes" => passthrough(quote! { ::pyre_object::is_bytes }),
        "PyByteArray" => passthrough(quote! { ::pyre_object::is_bytearray }),
        "PyInt" => passthrough(quote! { ::pyre_object::is_int }),
        "PyFloat" => passthrough(quote! { ::pyre_object::is_float }),
        "PyBool" => passthrough(quote! { ::pyre_object::is_bool }),
        "PySet" => passthrough(quote! { ::pyre_object::is_set }),
        "PyFrozenSet" => passthrough(quote! { ::pyre_object::is_frozenset }),
        "PyPath" => (
            quote! { ::std::vec::Vec<u8> },
            quote! { ::pyre_interpreter::gateway::fsencode_bytes_w(#slot)? },
        ),
        "PyIndex" => (
            // Mirrors PyPy `space.getindex_w(w_obj, None)`: consults
            // `__index__` per PEP 357 and returns the underlying i64,
            // clamping to i64::MAX / i64::MIN on overflow.  Raises
            // TypeError when the object has no `__index__` and is not
            // already int-like.
            quote! { i64 },
            quote! { ::pyre_interpreter::baseobjspace::getindex_w(#slot)? },
        ),
        "PyIndexInt" => (
            // CPython 3.14 Argument Clinic `int` parameters that accept the
            // index protocol but must still reject values outside pyre's
            // machine-int range.  Unlike `PyIndex`, this must not clamp a
            // huge bigint to i64::{MIN,MAX}.
            quote! { i64 },
            quote! {
                ::pyre_interpreter::baseobjspace::index_int_w_preserve_negative(#slot)?
            },
        ),
        "PyIndexCInt" => (
            // The same parameters narrowed to a C int, as an `int` parameter
            // whose callee compares against a small fixed set is.  `PyCInt`
            // does not serve: its converter reaches the value through
            // `__int__` first, which the index protocol does not.
            quote! { i32 },
            quote! { ::pyre_interpreter::baseobjspace::index_c_int_w(#slot)? },
        ),
        // Integer aliases — route through the `space.gateway_nonnegint_w`
        // / `space.c_*_w` converters in baseobjspace.rs so the range / sign
        // checks and their exception messages live in one place.
        "PyNonNegInt" => (
            quote! { i64 },
            quote! { ::pyre_interpreter::baseobjspace::gateway_nonnegint_w(#slot)? },
        ),
        "PyCInt" => (
            quote! { i32 },
            quote! { ::pyre_interpreter::baseobjspace::c_int_w(#slot)? },
        ),
        "PyCUInt" => (
            quote! { u32 },
            quote! { ::pyre_interpreter::baseobjspace::c_uint_w(#slot)? },
        ),
        "PyCShort" => (
            quote! { i16 },
            quote! { ::pyre_interpreter::baseobjspace::c_short_w(#slot)? },
        ),
        "PyCUShort" => (
            quote! { u16 },
            quote! { ::pyre_interpreter::baseobjspace::c_ushort_w(#slot)? },
        ),
        "PyCUidT" => (
            quote! { u32 },
            quote! { ::pyre_interpreter::baseobjspace::c_uid_t_w(#slot)? },
        ),
        "PyTruncatedInt" => (
            quote! { i64 },
            quote! { ::pyre_interpreter::baseobjspace::truncatedint_w(#slot)? },
        ),
        "PyText0" => (
            quote! { &'static str },
            quote! { ::pyre_interpreter::baseobjspace::text0_w(#slot)? },
        ),
        "PyBytes0" => (
            // space.bytes0_w — bytes_w plus a rejection of embedded NUL.
            quote! { &'static [u8] },
            quote! {
                {
                    if !unsafe { ::pyre_object::bytesobject::is_bytes_like(#slot) } {
                        return ::std::result::Result::Err(
                            ::pyre_interpreter::PyError::type_error(
                                format!("argument {} must be bytes-like", #idx)
                            )
                        );
                    }
                    let __b = unsafe { ::pyre_object::bytesobject::bytes_like_data(#slot) };
                    if __b.contains(&0) {
                        return ::std::result::Result::Err(
                            ::pyre_interpreter::PyError::value_error(
                                "embedded null byte".to_string()
                            )
                        );
                    }
                    __b
                }
            },
        ),
        "PyPathOrNone" => (
            // space.fsencode_or_none_w — None passes through, otherwise
            // fsencode_bytes_w (same conversion as the `PyPath` alias).
            quote! { ::std::option::Option<::std::vec::Vec<u8>> },
            quote! {
                if unsafe { ::pyre_object::is_none(#slot) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(::pyre_interpreter::gateway::fsencode_bytes_w(#slot)?)
                }
            },
        ),
        // space.realunicode_w / space.utf8_w — a str argument as unicode
        // text; both route through baseobjspace and return the borrowed
        // utf8 buffer, differing only in the TypeError message they raise.
        "PyUnicode" => (
            quote! { &'static str },
            quote! { ::pyre_interpreter::baseobjspace::realunicode_w(#slot)? },
        ),
        "PyUtf8" => (
            quote! { &'static str },
            quote! { ::pyre_interpreter::baseobjspace::utf8_w(#slot)? },
        ),
        "PyTextOrNone" => (
            // space.text_or_none_w — None passes through, otherwise text_w.
            quote! { ::std::option::Option<&'static str> },
            quote! {
                if #slot.is_null() || unsafe { ::pyre_object::is_none(#slot) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(::pyre_interpreter::baseobjspace::text_w(#slot)?)
                }
            },
        ),
        "PyText0OrNone" => (
            // space.text0_or_none_w — None passes through, otherwise text0_w.
            quote! { ::std::option::Option<&'static str> },
            quote! {
                if #slot.is_null() || unsafe { ::pyre_object::is_none(#slot) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(::pyre_interpreter::baseobjspace::text0_w(#slot)?)
                }
            },
        ),
        "PyBufferStr" => (
            quote! { ::std::vec::Vec<u8> },
            quote! { ::pyre_interpreter::baseobjspace::charbuf_w(#slot)? },
        ),
        "PyCNonNegInt" => (
            quote! { i32 },
            quote! { ::pyre_interpreter::baseobjspace::c_nonnegint_w(#slot)? },
        ),
        _ => return None,
    })
}

/// Helper used by `rewrite_alias_args` — looks up just the binding
/// type produced by `typed_alias`.  The dummy idx `0` is fine because
/// the returned binding type never references `idx`.
fn typed_alias_binding_ty(name: &str) -> Option<proc_macro2::TokenStream> {
    typed_alias(name, &quote! { args[0] }, 0).map(|(ty, _)| ty)
}

fn unwrap_expr(
    ty: &Type,
    slot: &proc_macro2::TokenStream,
    idx: usize,
) -> syn::Result<proc_macro2::TokenStream> {
    let ty = unwrap_type_group(ty);
    // Typed-receiver aliases — `state: PyTuple` becomes a typecheck +
    // PyObjectRef binding; `path: PyPath` becomes an fsencode_bytes_w call +
    // `Vec<u8>` binding.  Inner fn signatures get rewritten elsewhere so
    // the body's parameter type matches the binding type the alias
    // resolves to.
    if let Type::Path(p) = ty
        && let Some(seg) = p.path.segments.last()
        && let Some((_, expr)) = typed_alias(&seg.ident.to_string(), slot, idx)
    {
        return Ok(expr);
    }
    // `&[PyObjectRef]` — pass the whole slice (varargs).
    // `&[u8]`        — bytes-like (bytes + bytearray) → `bytes_like_data`,
    //                  with a runtime type check that returns a TypeError
    //                  on non-bytes-like input.
    if let Type::Reference(r) = ty {
        if let Type::Slice(s) = &*r.elem {
            let elem = unwrap_type_group(&s.elem);
            if type_is_py_object_ref(elem) {
                return Ok(quote! { args });
            }
            if let Type::Path(p) = elem
                && path_is_ident(&p.path, "u8")
            {
                return Ok(quote! {
                    {
                        if !unsafe { ::pyre_object::bytesobject::is_bytes_like(#slot) } {
                            return ::std::result::Result::Err(
                                ::pyre_interpreter::PyError::type_error(
                                    format!("argument {} must be bytes-like", #idx)
                                )
                            );
                        }
                        unsafe { ::pyre_object::bytesobject::bytes_like_data(#slot) }
                    }
                });
            }
        }
        // `&str` — borrow the utf-8 view.  A lone surrogate has no such view,
        // so it is reported as the `UnicodeEncodeError` the strict utf-8
        // encoder raises rather than demanding one.
        if let Type::Path(p) = &*r.elem
            && path_is_ident(&p.path, "str")
        {
            return Ok(quote! {
                ::pyre_interpreter::baseobjspace::str_utf8_w(#slot)?
            });
        }
    }

    if let Type::Path(p) = ty {
        if type_is_py_object_ref(ty) {
            return Ok(quote! { #slot });
        }
        // `Option<T>` — present when the slot is in range and not the
        // `PY_NULL` "argument omitted" marker `bind_builtin_kwargs` writes
        // for an absent optional after keyword resolution.  Mirrors PyPy
        // `@unwrap_spec(s=W_Root)` with `def f(self, space, s=None)`.
        if let Some(inner) = option_inner(ty) {
            let inner_unwrap = unwrap_expr(inner, slot, idx)?;
            return Ok(quote! {
                if #idx < args.len() && !#slot.is_null() { Some(#inner_unwrap) } else { None }
            });
        }
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            // The scalar unwraps are the `@unwrap_spec` conversions, not raw
            // payload reads: `int` is `space.gateway_int_w`, `float` is
            // `space.float_w`, `bool` is `space.is_true`.  Reading the payload
            // by layout instead would take an arbitrary object's first words
            // as the value — `msvcrt.SetErrorMode("x")` handed a str's
            // interior to the Win32 call and answered with it — where every
            // one of these owes a `TypeError` for an argument that is not the
            // type it declared.
            match name.as_str() {
                "i64" => {
                    return Ok(quote! {
                        ::pyre_interpreter::baseobjspace::gateway_int_w(#slot)?
                    });
                }
                "i32" | "u32" | "usize" | "isize" | "u16" | "i16" | "u8" | "i8" => {
                    // Parenthesised so the cast composes inside `if/else`
                    // / `match` arms when a `#[default(...)]` wrapper sits
                    // around the unwrap.  Without the parens,
                    // `if ... { unsafe{} as u32 } else { ... }` fails to
                    // parse because `as` doesn't accept a block-form LHS.
                    //
                    // The cast is the truncation `@unwrap_spec(x=int)` leaves
                    // to the body; a parameter that owes the narrower type's
                    // own `OverflowError` names `PyCInt` / `PyCNonNegInt`
                    // instead.
                    return Ok(quote! {
                        (::pyre_interpreter::baseobjspace::gateway_int_w(#slot)? as #ty)
                    });
                }
                "f64" => {
                    return Ok(quote! {
                        ::pyre_interpreter::baseobjspace::float_w(#slot)?
                    });
                }
                "bool" => {
                    return Ok(quote! {
                        ::pyre_interpreter::baseobjspace::is_true(#slot)?
                    });
                }
                _ => {}
            }
        }
    }

    Err(syn::Error::new(
        ty.span(),
        "#[pyre_function]: unsupported argument type — \
             add a mapping in pyre-macros/src/lib.rs::unwrap_expr"
            .to_string(),
    ))
}

/// Derive the Python keyword name from a Rust parameter identifier.  A single
/// trailing underscore is the convention for dodging a Rust keyword clash
/// (`final_`, `type_`), and the Python name drops it.  Dunder names end in
/// `__` and are left intact.
fn python_keyword_name(ident: &str) -> String {
    if ident.ends_with('_') && !ident.ends_with("__") {
        ident[..ident.len() - 1].to_string()
    } else {
        ident.to_string()
    }
}

/// The Python keyword a `w_`-prefixed wrapped-object parameter binds under.
/// A bare `w_` has no name left after stripping, so it keeps the identifier
/// rather than binding under the empty keyword.
fn python_param_keyword(ident: &str) -> String {
    let visible = match ident.strip_prefix("w_") {
        Some(stripped) if !stripped.is_empty() => stripped,
        _ => ident,
    };
    python_keyword_name(visible)
}

/// Reject two parameters that would bind under the same Python keyword.
/// `w_obj` and `obj` both render as `obj`, so by-name binding would silently
/// resolve every `obj=` to the first slot and the rendered signature would
/// name the keyword twice.
fn reject_duplicate_param_names(names: &[String], span: proc_macro2::Span) -> syn::Result<()> {
    for (i, name) in names.iter().enumerate() {
        if names[..i].contains(name) {
            return Err(syn::Error::new(
                span,
                format!(
                    "two parameters bind under the Python keyword `{name}`; \
                     rename one (a `w_` prefix is stripped from the keyword)"
                ),
            ));
        }
    }
    Ok(())
}

/// The keyword name a parameter binds under — its identifier, or a
/// synthetic positional-only placeholder for a non-ident pattern (which
/// no real keyword can match).
fn param_name(idx: usize, pt: &PatType) -> String {
    match &*pt.pat {
        Pat::Ident(pi) => python_param_keyword(&pi.ident.to_string()),
        _ => format!("__pyre_positional_{idx}"),
    }
}

/// The `TypeError` text for a positional count that does not fit a builtin
/// taking `required`..`total` arguments, as a `format!` over `args.len()`.
///
/// A signature with no optional slot names one exact count; one with an
/// optional tail names the bound the call missed, so `too_many` picks
/// between them.
fn arity_message(
    name: &str,
    required: usize,
    total: usize,
    too_many: bool,
) -> proc_macro2::TokenStream {
    if required != total {
        let bound = if too_many { total } else { required };
        let plural = if bound == 1 { "" } else { "s" };
        let at = if too_many { "at most" } else { "at least" };
        let text = format!("{name} expected {at} {bound} argument{plural}, got {{}}");
        return quote! { format!(#text, __pyre_positional_count) };
    }
    match total {
        0 => {
            let text = format!("{name}() takes no arguments ({{}} given)");
            quote! { format!(#text, __pyre_positional_count) }
        }
        1 => {
            let text = format!("{name}() takes exactly one argument ({{}} given)");
            quote! { format!(#text, __pyre_positional_count) }
        }
        n => {
            let text = format!("{name} expected {n} arguments, got {{}}");
            quote! { format!(#text, __pyre_positional_count) }
        }
    }
}

/// `true` for a `&[PyObjectRef]` varargs parameter (binds the whole arg
/// slice).
fn is_varargs_param(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    if let Type::Reference(r) = ty
        && let Type::Slice(s) = &*r.elem
    {
        return type_is_py_object_ref(unwrap_type_group(&s.elem));
    }
    false
}

/// `Option<T>` → `Some(&T)`; anything else → None.
fn option_inner(ty: &Type) -> Option<&Type> {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let first = args.args.iter().next()?;
    let syn::GenericArgument::Type(t) = first else {
        return None;
    };
    Some(t)
}

fn wrap_return(
    ret: &ReturnType,
    call_inner: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let ty = match ret {
        ReturnType::Default => {
            return Ok(quote! { #call_inner; Ok(::pyre_object::w_none()) });
        }
        ReturnType::Type(_, t) => &**t,
    };

    // `Result<T, ::pyre_interpreter::PyError>` — propagate via `?`, then wrap T.
    if let Some(inner) = result_pyerror_inner(ty) {
        let wrap = wrap_value_expr(inner, quote! { __pyre_v })?;
        return Ok(quote! {
            let __pyre_v = #call_inner ?;
            Ok(#wrap)
        });
    }

    let wrap = wrap_value_expr(ty, call_inner)?;
    Ok(quote! { Ok(#wrap) })
}

/// Wrap a Rust value expression of type `ty` into a `PyObjectRef`.
fn wrap_value_expr(
    ty: &Type,
    value: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let ty = unwrap_type_group(ty);
    if type_is_py_object_ref(ty) {
        return Ok(value);
    }
    if let Type::Path(p) = ty
        && let Some(seg) = p.path.segments.last()
    {
        let name = seg.ident.to_string();
        match name.as_str() {
            "i64" => return Ok(quote! { ::pyre_object::w_int_new(#value) }),
            "i32" | "u32" | "usize" | "isize" | "u16" | "i16" | "u8" | "i8" => {
                return Ok(quote! { ::pyre_object::w_int_new((#value) as i64) });
            }
            "f64" => return Ok(quote! { ::pyre_object::w_float_new(#value) }),
            "bool" => return Ok(quote! { ::pyre_object::w_bool_from(#value) }),
            "String" => return Ok(quote! { ::pyre_object::w_str_new_managed(&#value) }),
            // A method whose text may hold a lone surrogate — a `__repr__`
            // naming a filename, say — returns the WTF-8 buffer itself
            // rather than a `String` it cannot spell.
            "Wtf8Buf" => return Ok(quote! { ::pyre_object::w_str_from_wtf8_managed(#value) }),
            _ => {}
        }
        // `Vec<T>` — bytes / list-of-X.
        //   * `Vec<u8>`                        → bytes via w_bytes_from_bytes
        //   * `Vec<PyObjectRef>`               → list passthrough
        //   * `Vec<i64> / <i32> / <f64> / <String> / <bool>` →
        //     wrap each element via `PywrapKind` then `w_list_new`.
        // Mirrors PyPy `space.newlist([space.newint(x) for x in vec])`
        // where the interp2app auto-wraps a Rust `[W_Root]` return
        // through `space.newlist`.
        if seg.ident == "Vec"
            && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
        {
            let inner = unwrap_type_group(inner);
            if let Type::Path(ip) = inner {
                if path_is_ident(&ip.path, "u8") {
                    return Ok(quote! {
                        ::pyre_object::bytesobject::w_bytes_from_bytes(&#value)
                    });
                }
                if type_is_py_object_ref(inner) {
                    return Ok(quote! { ::pyre_object::w_list_new(#value) });
                }
                if let Some(last) = ip.path.segments.last() {
                    let nm = last.ident.to_string();
                    if matches!(
                        nm.as_str(),
                        "i64"
                            | "i32"
                            | "u32"
                            | "usize"
                            | "isize"
                            | "u16"
                            | "i16"
                            | "i8"
                            | "f64"
                            | "bool"
                            | "String"
                    ) {
                        return Ok(quote! {
                            ::pyre_object::w_list_new(
                                (#value).into_iter()
                                    .map(<_ as ::pyre_interpreter::PywrapKind>::into_py)
                                    .collect()
                            )
                        });
                    }
                }
            }
        }
    }
    if let Type::Tuple(t) = ty
        && t.elems.is_empty()
    {
        return Ok(quote! { { #value; ::pyre_object::w_none() } });
    }
    use quote::ToTokens;
    Err(syn::Error::new(
        ty.span(),
        format!(
            "#[pyre_function]: unsupported return type `{}` — \
             add a mapping in pyre-macros/src/lib.rs::wrap_value_expr",
            ty.to_token_stream()
        ),
    ))
}

/// Strip `Type::Group` wrappers that `macro_rules!` `:ty` capture adds
/// around emitted type fragments — without this, `wrap_value_expr` and
/// friends never recognize the inner `Type::Path` for `String` /
/// `PyObjectRef` / `i64` / etc. when the fn definition arrives through
/// `py_module!`'s `inline_functions:` arm.
fn unwrap_type_group(ty: &Type) -> &Type {
    let mut t = ty;
    while let Type::Group(g) = t {
        t = &g.elem;
    }
    t
}

fn type_is_py_object_ref(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return false };
    let segs: Vec<_> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    matches!(
        segs.as_slice(),
        [s] if s == "PyObjectRef"
    ) || matches!(
        segs.as_slice(),
        [a, b] if a == "pyre_object" && b == "PyObjectRef"
    ) || matches!(
        segs.as_slice(),
        [_, a, b] if a == "pyre_object" && b == "PyObjectRef"
    )
}

fn path_is_ident(p: &syn::Path, name: &str) -> bool {
    p.segments.len() == 1 && p.segments[0].ident == name
}

/// If `ty` is `Result<T, ::pyre_interpreter::PyError>` (or `PyError` short form), return `&T`.
fn result_pyerror_inner(ty: &Type) -> Option<&Type> {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut iter = args.args.iter();
    let first = iter.next()?;
    let second = iter.next()?;
    let syn::GenericArgument::Type(ok_ty) = first else {
        return None;
    };
    let syn::GenericArgument::Type(err_ty) = second else {
        return None;
    };
    if !type_is_pyerror(err_ty) {
        return None;
    }
    Some(ok_ty)
}

fn type_is_pyerror(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return false };
    let segs: Vec<_> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    matches!(segs.last().map(String::as_str), Some("PyError"))
}

// Avoid "unused" warnings on parse_quote import in the trimmed build.
#[allow(dead_code)]
fn _unused() {
    let _: syn::Type = parse_quote!(i64);
}

// ──────────────────────────────────────────────────────────────────────
// `#[pyre_class("python.name", type_id = N)]` — PyPy `class W_X(W_Root)`
// equivalent.
//
// Generates the typed-payload boilerplate every existing `W_X` struct
// writes by hand (see `pyre/pyre-object/src/descriptor.rs`):
// ──────────────────────────────────────────────────────────────────────
//
// User source:
//   #[pyre_class("_random.Random", type_id = 53)]
//   pub struct W_Random {
//       state: u64,
//   }
//
// Emitted:
//   pub static RANDOM_TYPE: ::pyre_object::PyType =
//       ::pyre_object::pyobject::new_pytype("_random.Random");
//
//   #[repr(C)]
//   pub struct W_Random {
//       pub ob: ::pyre_object::PyObject,   // <- macro-prepended header
//       pub state: u64,
//   }
//
//   pub const W_RANDOM_GC_TYPE_ID: u32 = 53;
//   pub const W_RANDOM_OBJECT_SIZE: usize = ::std::mem::size_of::<W_Random>();
//   pub const W_RANDOM_GC_PTR_OFFSETS: [usize; 0] = [];
//
//   impl ::pyre_object::lltype::GcType for W_Random {
//       const TYPE_ID: u32 = W_RANDOM_GC_TYPE_ID;
//       const SIZE: usize = W_RANDOM_OBJECT_SIZE;
//   }
//
//   impl W_Random {
//       pub unsafe fn from_obj(obj: ::pyre_object::PyObjectRef)
//           -> ::std::option::Option<&'static mut Self>
//       {
//           if unsafe { ::pyre_object::py_type_check(obj, &RANDOM_TYPE) } {
//               Some(unsafe { &mut *(obj as *mut Self) })
//           } else { None }
//       }
//   }
//
// `PTR_OFFSETS` auto-derived from the user's struct: every field whose
// type is `PyObjectRef` becomes one entry via `std::mem::offset_of!`.
// Primitive fields (`u64` / `i32` / etc.) are skipped because the GC
// doesn't need to trace them.
//
// `type_id = N` is required (manual): the GC's `pytype_to_tid` table
// asserts a contiguous monotonic sequence at JIT-init in
// `pyre/pyre-jit/src/eval.rs:1335-1352`.  Reserve a slot, register
// it in eval.rs, and pass the same number here.
//
// The PyType static name is derived from the struct name (snake-case
// uppercased + `_TYPE` suffix): `W_Random` → `RANDOM_TYPE`.
// `GetSetProperty` → `GETSETPROPERTY_TYPE`.  Override the suffix
// path is not yet supported — pick struct names whose derived static
// matches the import path callers expect.

#[proc_macro_attribute]
pub fn pyre_class(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as PyreClassAttrs);
    let st = parse_macro_input!(item as ItemStruct);
    match expand_pyre_class(attrs, st) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct PyreClassAttrs {
    name: syn::LitStr,
    /// Optional explicit `type_id = N`.  When `Some`, the macro emits
    /// the legacy `pub const W_X_GC_TYPE_ID: u32 = N;` alias and the
    /// runtime cell starts pre-initialized to `N` so the JIT driver
    /// can drift-check it.  When `None`, no const is emitted and the
    /// cell starts at `TypeIdCell::UNASSIGNED`; the JIT driver writes
    /// the next available tid back into it.
    type_id: Option<syn::LitInt>,
    /// Optional override for the upper-case suffix used in derived
    /// static / const names.  Defaults to `strip_prefix("W_")` over
    /// the struct ident, but legacy classes like `W_Super` (which
    /// historically shipped as `SUPER_TYPE` and `W_SUPER_GC_TYPE_ID`,
    /// not `SUPEROBJECT_TYPE`) need to opt into the shorter form.
    static_name: Option<syn::LitStr>,
    /// Optional override for *just* the PyType static identifier.  Most
    /// classes accept the `{static_name}_TYPE` default
    /// (`SUPER_TYPE`, `RANDOM_TYPE`); a few legacy classes ship the
    /// PyType under a name unrelated to the GC consts (e.g.
    /// `GETSET_DESCRIPTOR_TYPE` vs `W_GETSET_PROPERTY_GC_TYPE_ID`).
    /// Specifying this lets the GC consts retain one prefix while the
    /// PyType keeps its historical name.
    pytype_static: Option<syn::LitStr>,
    /// Optional name of the `_getusercls` typeptr static
    /// (`typedef.py` `_getusercls`). When set, the PyType is built with
    /// `new_pytype_with_user_subclass` and `from_obj` accepts that
    /// typeptr as well as this class's own: a user-subclass layout
    /// starts with the base payload. Example:
    /// `user_subclass = "ARRAY_USER_TYPE"`.
    user_subclass: Option<syn::LitStr>,
    /// Emit `typedef.py` `_getusercls`: `SUser` appends `MapdictStorageMixin`
    /// (`map` then `storage`), `P_USER_TYPE` is that layout's typeptr, and
    /// `allocate_instance` (`objspace.py`) picks it when `w_subtype` is a
    /// real subclass. Mutually exclusive with `user_subclass`.
    user_layout: bool,
    /// CPython 3.14 constructs this module type with `PyType_From*Spec`.
    /// This is a public flag projection only; the PyPy TypeDef remains a
    /// builtin internally.
    cpython_heaptype: bool,
    /// CPython permits mutation of this heap type's namespace.  Implies
    /// `cpython_heaptype`; absent this marker, heap extension types default to
    /// IMMUTABLETYPE, matching the dominant 3.14 shape.
    cpython_mutable: bool,
    /// Append (or reuse) the `make_weakref_descr` `_lifeline_` field and
    /// publish its byte offset on this class's PyType. Does not set the
    /// app-level weakrefable flag; `#[pyre_methods(weakrefable)]` does that.
    weakrefable: bool,
}

impl syn::parse::Parse for PyreClassAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // `"name.path"[, type_id = N][, static_name = "PREFIX"][, user_subclass = "IDENT"][, user_layout][, weakrefable]`
        let name: syn::LitStr = input.parse()?;
        let mut type_id: Option<syn::LitInt> = None;
        let mut static_name: Option<syn::LitStr> = None;
        let mut pytype_static: Option<syn::LitStr> = None;
        let mut user_subclass: Option<syn::LitStr> = None;
        let mut user_layout = false;
        let mut cpython_heaptype = false;
        let mut cpython_mutable = false;
        let mut weakrefable = false;
        while !input.is_empty() {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "cpython_heaptype" => {
                    cpython_heaptype = true;
                    continue;
                }
                "cpython_mutable" => {
                    cpython_heaptype = true;
                    cpython_mutable = true;
                    continue;
                }
                "user_layout" => {
                    user_layout = true;
                    continue;
                }
                "weakrefable" => {
                    weakrefable = true;
                    continue;
                }
                _ => {}
            }
            input.parse::<syn::Token![=]>()?;
            match key.to_string().as_str() {
                "type_id" => type_id = Some(input.parse()?),
                "static_name" => static_name = Some(input.parse()?),
                "pytype_static" => pytype_static = Some(input.parse()?),
                "user_subclass" => user_subclass = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `#[pyre_class]` key `{other}` — \
                             expected `type_id` / `static_name` / `pytype_static` / \
                             `user_subclass` / `user_layout` / `weakrefable` / \
                             `cpython_heaptype` / `cpython_mutable`",
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            name,
            type_id,
            static_name,
            pytype_static,
            user_subclass,
            user_layout,
            cpython_heaptype,
            cpython_mutable,
            weakrefable,
        })
    }
}

fn expand_pyre_class(
    attrs: PyreClassAttrs,
    mut st: ItemStruct,
) -> syn::Result<proc_macro2::TokenStream> {
    let st_name = st.ident.clone();
    let st_vis = st.vis.clone();
    let name_lit = attrs.name;
    let cpython_heaptype = attrs.cpython_heaptype;
    let cpython_immutabletype = !attrs.cpython_mutable;
    let (has_map, has_storage) = {
        let Fields::Named(ref named) = st.fields else {
            return Err(syn::Error::new(
                st.span(),
                "#[pyre_class] requires a struct with named fields",
            ));
        };
        let has_map = named
            .named
            .iter()
            .any(|f| f.ident.as_ref().is_some_and(|i| i == "map"));
        let has_storage = named
            .named
            .iter()
            .any(|f| f.ident.as_ref().is_some_and(|i| i == "storage"));
        (has_map, has_storage)
    };
    let has_mapdict_mixin = has_map && has_storage;
    let user_layout = attrs.user_layout;
    let weakrefable = attrs.weakrefable;
    if user_layout && attrs.user_subclass.is_some() {
        return Err(syn::Error::new(
            st.span(),
            "`user_layout` cannot be combined with `user_subclass`",
        ));
    }
    if user_layout && (has_map || has_storage) {
        return Err(syn::Error::new(
            st.span(),
            "`user_layout` rejects a struct that already declares `map` or `storage`",
        ));
    }

    // Derive static names from the struct name.
    //   W_Random          -> RANDOM_TYPE, W_RANDOM_GC_TYPE_ID,
    //                       W_RANDOM_OBJECT_SIZE, W_RANDOM_GC_PTR_OFFSETS
    //   GetSetProperty  -> GETSETPROPERTY_TYPE, W_GETSETPROPERTY_GC_TYPE_ID, …
    let st_str = st_name.to_string();
    let suffix = attrs
        .static_name
        .as_ref()
        .map(|s| s.value())
        .unwrap_or_else(|| st_str.strip_prefix("W_").unwrap_or(&st_str).to_uppercase());
    let pytype_static = match attrs.pytype_static.as_ref() {
        Some(s) => format_ident!("{}", s.value()),
        None => format_ident!("{}_TYPE", suffix),
    };
    let gc_type_id_const = format_ident!("W_{}_GC_TYPE_ID", suffix);
    let gc_type_id_cell = format_ident!("W_{}_GC_TYPE_ID_CELL", suffix);
    let object_size_const = format_ident!("W_{}_OBJECT_SIZE", suffix);
    let ptr_offsets_const = format_ident!("W_{}_GC_PTR_OFFSETS", suffix);
    let descriptor_static = format_ident!("W_{}_PYRE_CLASS_DESCRIPTOR", suffix);
    let descriptor_slice_elem = format_ident!("W_{}_PYRE_CLASS_DESCRIPTOR_SLICE", suffix);
    let descriptor_ctor_fn = format_ident!("__register_w_{}_pyre_class_descriptor", suffix);
    let user_struct_name = format_ident!("{}User", st_name);
    let user_pytype_static = {
        let base = pytype_static.to_string();
        let name = match base.strip_suffix("_TYPE") {
            Some(stem) => format!("{stem}_USER_TYPE"),
            None => format!("{base}_USER_TYPE"),
        };
        format_ident!("{name}")
    };
    let user_gc_type_id_cell = format_ident!("W_{}_USER_GC_TYPE_ID_CELL", suffix);
    let user_descriptor_static = format_ident!("W_{}_USER_PYRE_CLASS_DESCRIPTOR", suffix);
    let user_descriptor_slice_elem = format_ident!("W_{}_USER_PYRE_CLASS_DESCRIPTOR_SLICE", suffix);
    let user_descriptor_ctor_fn =
        format_ident!("__register_w_{}_user_pyre_class_descriptor", suffix);
    // `user_layout` builds the `_getusercls` typeptr itself and feeds it
    // through the same `from_obj` / `new_pytype_with_user_subclass` path as
    // an explicit `user_subclass` ident.
    let user_subclass = if user_layout {
        Some(user_pytype_static.clone())
    } else {
        attrs
            .user_subclass
            .as_ref()
            .map(|s| format_ident!("{}", s.value()))
    };

    // When the user declared `type_id = N` we pre-initialize the cell
    // to `N` and additionally emit the legacy `pub const W_X_GC_TYPE_ID:
    // u32 = N;` alias so existing consumers (pyre-jit-trace,
    // drift-detection unit tests) keep compiling.  When the user
    // omitted `type_id`, the cell starts unassigned and the legacy
    // const is not emitted — callers must read the cell at runtime via
    // `<W_X as GcType>::type_id()` (which itself becomes `cell.get()`).
    let mapdict_adjacency_assert = if has_mapdict_mixin {
        quote! {
            const _: () = {
                assert!(
                    ::std::mem::offset_of!(#st_name, storage)
                        == ::std::mem::offset_of!(#st_name, map)
                            + ::std::mem::size_of::<usize>()
                );
            };
        }
    } else {
        quote! {}
    };
    let from_obj_type_check = match &user_subclass {
        Some(ident) => quote! {
            unsafe {
                ::pyre_object::py_type_check(obj, &#pytype_static)
                    || ::pyre_object::py_type_check(obj, &#ident)
            }
        },
        None => quote! {
            unsafe { ::pyre_object::py_type_check(obj, &#pytype_static) }
        },
    };

    let (cell_init, legacy_const) = match attrs.type_id.as_ref() {
        Some(n) => (
            quote! { ::pyre_object::lltype::TypeIdCell::with(#n) },
            quote! { #st_vis const #gc_type_id_const: u32 = #n; },
        ),
        None => (
            quote! { ::pyre_object::lltype::TypeIdCell::auto() },
            quote! {},
        ),
    };

    // Enforce `#[repr(C)]` and prepend the PyObject header.
    let already_repr_c = st.attrs.iter().any(|a| {
        a.path().is_ident("repr")
            && a.parse_args::<syn::Ident>()
                .map(|i| i == "C")
                .unwrap_or(false)
    });
    if !already_repr_c {
        st.attrs.push(parse_quote!(#[repr(C)]));
    }

    // Prepend `pub ob: PyObject` if not already present.
    let Fields::Named(ref mut named) = st.fields else {
        return Err(syn::Error::new(
            st.span(),
            "#[pyre_class] requires a struct with named fields",
        ));
    };
    let has_ob = named
        .named
        .iter()
        .any(|f| f.ident.as_ref().map(|i| i == "ob").unwrap_or(false));
    if !has_ob {
        use syn::parse::Parser;
        let ob_field: syn::Field = syn::Field::parse_named
            .parse2(quote! { pub ob: ::pyre_object::PyObject })
            .expect("parse ob field");
        named.named.insert(0, ob_field);
    }

    // `typedef.py make_weakref_descr` stores `_lifeline_` on the instance.
    // Reuse `lifeline` or `w_weakreflifeline` when the struct already has
    // one; otherwise append `lifeline`. This attribute publishes the
    // offset only — `#[pyre_methods(weakrefable)]` owns the app-level flag.
    let weakref_field: Option<syn::Ident> = if weakrefable {
        let existing = named.named.iter().find_map(|f| {
            f.ident.as_ref().and_then(|ident| {
                if ident == "lifeline" || ident == "w_weakreflifeline" {
                    Some(ident.clone())
                } else {
                    None
                }
            })
        });
        if let Some(ident) = existing {
            Some(ident)
        } else {
            use syn::parse::Parser;
            let lifeline_field: syn::Field = syn::Field::parse_named
                .parse2(quote! { pub lifeline: ::pyre_object::PyObjectRef })
                .expect("parse lifeline field");
            named.named.push(lifeline_field);
            Some(syn::Ident::new("lifeline", proc_macro2::Span::call_site()))
        }
    } else {
        None
    };
    // `user_subclass` names the `_getusercls` typeptr (`typedef.py`
    // `_getusercls`) and wins over the mapdict-mixin constructor. The
    // user typeptr publishes the same lifeline offset only when the
    // field exists: `base` is a prefix, so `offset_of!(Base, field)`
    // addresses it on a subclass instance too.
    let pytype_init = match (&user_subclass, &weakref_field) {
        (Some(ident), Some(field)) => quote! {
            ::pyre_object::pyobject::new_pytype_with_user_subclass_and_weakref(
                #name_lit,
                &#ident,
                ::std::mem::offset_of!(#st_name, #field),
            )
        },
        (Some(ident), None) => quote! {
            ::pyre_object::pyobject::new_pytype_with_user_subclass(#name_lit, &#ident)
        },
        (None, Some(field)) if has_mapdict_mixin => quote! {
            ::pyre_object::pyobject::new_pytype_with_mapdict_and_weakref(
                #name_lit,
                ::std::mem::offset_of!(#st_name, map),
                ::std::mem::offset_of!(#st_name, #field),
            )
        },
        (None, None) if has_mapdict_mixin => quote! {
            ::pyre_object::pyobject::new_pytype_with_mapdict_mixin(
                #name_lit,
                ::std::mem::offset_of!(#st_name, map),
            )
        },
        (None, Some(field)) => quote! {
            ::pyre_object::pyobject::new_pytype_with_weakref(
                #name_lit,
                ::std::mem::offset_of!(#st_name, #field),
            )
        },
        (None, None) => quote! {
            ::pyre_object::pyobject::new_pytype(#name_lit)
        },
    };
    let user_pytype_init = match &weakref_field {
        Some(field) => quote! {
            ::pyre_object::pyobject::new_user_pytype_with_lifeline(
                #name_lit,
                &#pytype_static,
                ::std::mem::offset_of!(#user_struct_name, map),
                ::std::mem::offset_of!(#st_name, #field),
            )
        },
        None => quote! {
            ::pyre_object::pyobject::new_user_pytype(
                #name_lit,
                &#pytype_static,
                ::std::mem::offset_of!(#user_struct_name, map),
            )
        },
    };

    // Collect `PyObjectRef` fields' offsets for GC tracing.  `ob` is the
    // `PyObject` header; its `w_class` word is the instance -> class edge
    // and is listed explicitly.  A `#[pyre_class]` instance created for a
    // Python subclass carries the (managed) heap type there, and the
    // collector traces nothing but the offsets registered for the
    // instance's own type id — a parent tid contributes none — so leaving
    // it out let the subclass type be swept while instances were live.
    // `ob_type` stays out: it always points at the `static PyType`.
    let mut ptr_field_idents: Vec<syn::Ident> = Vec::new();
    for f in named.named.iter() {
        let Some(ident) = f.ident.clone() else {
            continue;
        };
        if ident == "ob" {
            continue;
        }
        if type_is_py_object_ref(&f.ty) {
            ptr_field_idents.push(ident);
        }
    }
    let ptr_offsets_len = ptr_field_idents.len() + 1;
    let ptr_offsets_inits: Vec<proc_macro2::TokenStream> =
        std::iter::once(quote! { ::std::mem::offset_of!(#st_name, ob.w_class) })
            .chain(
                ptr_field_idents
                    .iter()
                    .map(|i| quote! { ::std::mem::offset_of!(#st_name, #i) }),
            )
            .collect();

    let payload_restores: Vec<proc_macro2::TokenStream> = ptr_field_idents
        .iter()
        .enumerate()
        .map(|(i, ident)| {
            let idx = i + 1usize;
            quote! {
                payload.#ident = ::pyre_object::gc_roots::shadow_stack_get(__base + #idx);
            }
        })
        .collect();
    let payload_binding = if ptr_field_idents.is_empty() {
        quote! { payload }
    } else {
        quote! { mut payload }
    };
    let (user_layout_items, allocate_instance_fn) = if user_layout {
        (
            quote! {
                /// `typedef.py` `_getusercls`: the base payload plus
                /// `MapdictStorageMixin` (`map` followed by `storage`).
                #[repr(C)]
                pub struct #user_struct_name {
                    pub base: #st_name,
                    pub map: usize,
                    pub storage: *mut ::pyre_object::object_array::ItemsBlock,
                }

                const _: () = {
                    assert!(
                        ::std::mem::offset_of!(#user_struct_name, storage)
                            == ::std::mem::offset_of!(#user_struct_name, map)
                                + ::std::mem::size_of::<usize>()
                    );
                };

                #st_vis static #user_pytype_static: ::pyre_object::PyType =
                    #user_pytype_init;

                #st_vis static #user_gc_type_id_cell: ::pyre_object::lltype::TypeIdCell =
                    ::pyre_object::lltype::TypeIdCell::auto();

                impl ::pyre_object::lltype::GcType for #user_struct_name {
                    #[inline]
                    fn type_id() -> u32 {
                        #user_gc_type_id_cell.get()
                    }
                    const SIZE: usize = ::std::mem::size_of::<#user_struct_name>();
                }

                #st_vis static #user_descriptor_static: ::pyre_object::lltype::PyreClassDescriptor =
                    ::pyre_object::lltype::PyreClassDescriptor {
                        pytype_ptr: &#user_pytype_static as *const ::pyre_object::PyType,
                        gc_type_id: &#user_gc_type_id_cell,
                        object_size: ::std::mem::size_of::<#user_struct_name>(),
                        ptr_offsets: &#ptr_offsets_const,
                        pyname: #name_lit,
                        pytype_path: ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            ::core::stringify!(#user_pytype_static),
                        ),
                        struct_path: ::core::concat!(
                            ::core::module_path!(),
                            "::",
                            ::core::stringify!(#user_struct_name),
                        ),
                        mapdict_user_layout: true,
                    };

                #[cfg(not(target_arch = "wasm32"))]
                #[::linkme::distributed_slice(::pyre_object::lltype::PYRE_CLASS_DESCRIPTORS)]
                static #user_descriptor_slice_elem: &'static ::pyre_object::lltype::PyreClassDescriptor =
                    &#user_descriptor_static;

                #[cfg(target_arch = "wasm32")]
                #[::ctor::ctor(unsafe)]
                fn #user_descriptor_ctor_fn() {
                    ::pyre_object::lltype::register_class_descriptor(&#user_descriptor_static);
                }
            },
            quote! {
                /// `objspace.py` `allocate_instance`. An exact instance (null
                /// `w_subtype`, or the builtin type object) is the base layout.
                /// A user subclass is `typedef.py` `_getusercls`: `SUser` with
                /// mapdict `map` / `storage`, `ob_type` the user typeptr, and
                /// `w_class` equal to `w_subtype`. The finalizer enqueue runs
                /// when that class has `__del__`.
                #[allow(dead_code)]
                pub fn allocate_instance(
                    #payload_binding: Self,
                    w_subtype: ::pyre_object::PyObjectRef,
                ) -> ::pyre_object::PyObjectRef {
                    if w_subtype.is_null()
                        || ::std::ptr::eq(
                            w_subtype,
                            ::pyre_object::pyobject::get_instantiate(&#pytype_static),
                        )
                    {
                        // `malloc_fixedsize`: born young (see `allocate`), so a
                        // `super(w_type, w_type)` built for `__init_subclass__`
                        // dies with its class instead of pinning it to the
                        // next major from the remembered set.
                        let obj = Self::allocate(payload);
                        // objspace.py `allocate_instance`: hasuserdel types
                        // enqueue the fresh instance after the malloc.
                        ::pyre_object::gc_hook::maybe_register_finalizer(obj);
                        return obj;
                    }
                    let _roots = ::pyre_object::gc_roots::push_roots();
                    let __refs = [w_subtype, #(payload.#ptr_field_idents),*];
                    let __base = ::pyre_object::gc_roots::pin_roots(&__refs);
                    let type_id = match <#user_struct_name as ::pyre_object::lltype::GcType>::type_id() {
                        ::pyre_object::lltype::TypeIdCell::UNASSIGNED => 0,
                        id => id,
                    };
                    let raw = ::pyre_object::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(
                        type_id,
                        ::std::mem::size_of::<#user_struct_name>(),
                    );
                    let w_subtype = ::pyre_object::gc_roots::shadow_stack_get(__base);
                    #(#payload_restores)*
                    let user = #user_struct_name {
                        base: Self {
                            ob: ::pyre_object::PyObject {
                                ob_type: &#user_pytype_static as *const ::pyre_object::PyType,
                                w_class: w_subtype,
                            },
                            ..payload
                        },
                        map: 0,
                        storage: ::std::ptr::null_mut(),
                    };
                    let obj = if raw.is_null() {
                        ::pyre_object::lltype::malloc_typed(user)
                            as ::pyre_object::PyObjectRef
                    } else {
                        unsafe {
                            ::std::ptr::write(raw as *mut #user_struct_name, user);
                            ::pyre_object::gc_hook::try_gc_write_barrier_managed(raw);
                            raw as ::pyre_object::PyObjectRef
                        }
                    };
                    ::pyre_object::gc_hook::maybe_register_finalizer(obj);
                    obj
                }
            },
        )
    } else {
        (quote! {}, quote! {})
    };

    Ok(quote! {
        #st

        #mapdict_adjacency_assert

        #user_layout_items

        #st_vis static #pytype_static: ::pyre_object::PyType =
            #pytype_init;

        /// Runtime-resolved GC tid for this class.  Initialized either
        /// to the explicit `type_id = N` from the attribute (drift-
        /// checked at JIT init) or to `TypeIdCell::UNASSIGNED` for
        /// auto-mode (then stamped by the JIT driver).
        #st_vis static #gc_type_id_cell: ::pyre_object::lltype::TypeIdCell =
            #cell_init;
        #legacy_const
        #st_vis const #object_size_const: usize = ::std::mem::size_of::<#st_name>();
        #st_vis const #ptr_offsets_const: [usize; #ptr_offsets_len] = [
            #(#ptr_offsets_inits),*
        ];

        impl ::pyre_object::lltype::GcType for #st_name {
            #[inline]
            fn type_id() -> u32 {
                #gc_type_id_cell.get()
            }
            const SIZE: usize = #object_size_const;
        }

        /// Compile-time descriptor consumed by `pyre/pyre-jit/src/eval.rs`'s
        /// GC registration loop.  Aggregates the four constants above into
        /// a single `Sync` static the JIT driver iterates over without
        /// per-type knowledge.
        #st_vis static #descriptor_static: ::pyre_object::lltype::PyreClassDescriptor =
            ::pyre_object::lltype::PyreClassDescriptor {
                pytype_ptr: &#pytype_static as *const ::pyre_object::PyType,
                gc_type_id: &#gc_type_id_cell,
                object_size: #object_size_const,
                ptr_offsets: &#ptr_offsets_const,
                pyname: #name_lit,
                pytype_path: ::core::concat!(
                    ::core::module_path!(),
                    "::",
                    ::core::stringify!(#pytype_static),
                ),
                struct_path: ::core::concat!(
                    ::core::module_path!(),
                    "::",
                    ::core::stringify!(#st_name),
                ),
                mapdict_user_layout: false,
            };

        /// Registration of this class's descriptor into the whole-program
        /// registry, so the JIT driver's GC-root completeness oracle and
        /// the `PyType`-address binder see every `#[pyre_class]` type
        /// without a hand-maintained list.  `distributed_slice` rejects
        /// `wasm32`, which registers from a constructor instead.
        #[cfg(not(target_arch = "wasm32"))]
        #[::linkme::distributed_slice(::pyre_object::lltype::PYRE_CLASS_DESCRIPTORS)]
        static #descriptor_slice_elem: &'static ::pyre_object::lltype::PyreClassDescriptor =
            &#descriptor_static;

        #[cfg(target_arch = "wasm32")]
        #[::ctor::ctor(unsafe)]
        fn #descriptor_ctor_fn() {
            ::pyre_object::lltype::register_class_descriptor(&#descriptor_static);
        }

        impl ::pyre_object::lltype::PyreClassPyTypeOf for #st_name {
            const PYTYPE: *const ::pyre_object::PyType =
                &#pytype_static as *const ::pyre_object::PyType;
            const DESCRIPTOR: &'static ::pyre_object::lltype::PyreClassDescriptor =
                &#descriptor_static;
            const PYNAME: &'static str = #name_lit;
            const CPYTHON_HEAPTYPE: bool = #cpython_heaptype;
            const CPYTHON_IMMUTABLETYPE: bool = #cpython_immutabletype;
            const HAS_MAPDICT_MIXIN: bool = #has_mapdict_mixin;
        }

        impl #st_name {
            /// Borrow `obj` as `&mut Self` after verifying its
            /// `ob_type` matches this class's static `PyType`.
            /// Returns `None` if `obj` is the wrong type — callers
            /// must convert that to a Python `TypeError`.
            #[allow(dead_code)]
            #[inline]
            #[expect(
                clippy::not_unsafe_ptr_arg_deref,
                reason = "PyObjectRef is pyre's GC-rooted VM handle; this generated object-space downcast checks the runtime type before dereferencing and intentionally provides the safe boundary used by Python operations"
            )]
            pub fn from_obj(obj: ::pyre_object::PyObjectRef)
                -> ::std::option::Option<&'static mut Self>
            {
                if #from_obj_type_check {
                    ::std::option::Option::Some(unsafe { &mut *(obj as *mut Self) })
                } else {
                    ::std::option::Option::None
                }
            }

            /// Allocate a fresh instance via
            /// `lltype::malloc_typed_young_nonmoving`, stamping the PyObject
            /// header so the collector and dispatcher can identify it.
            /// `payload` carries the user-defined fields; the `ob` header is
            /// filled in by this fn.
            ///
            /// PyPy's translated `gct_fv_gc_malloc` uses the nursery because
            /// the framework GC transform roots every live GC pointer around a
            /// collecting operation.  Pyre's Rust interpreter has not yet
            /// completed that transform for arbitrary Rust locals, so a
            /// `#[pyre_class]` value can outlive this call in an unrewritable
            /// raw local (a range iterator held by a compiled loop is the
            /// smallest example).  Keep these payloads non-moving until that
            /// transform exists — `external_malloc(..., alloc_young=True)`,
            /// so the block is still young and the first minor collection
            /// that does not reach it frees it, as `malloc_fixedsize` would.
            /// A birth in the old generation instead kept every such object,
            /// and everything it pointed at, alive until the next major: a
            /// `copy_for_type` descriptor held its class that long.
            #[allow(dead_code)]
            pub fn allocate(payload: Self) -> ::pyre_object::PyObjectRef {
                let _roots = ::pyre_object::gc_roots::push_roots();
                let full = Self {
                    ob: ::pyre_object::PyObject {
                        ob_type: &#pytype_static as *const ::pyre_object::PyType,
                        w_class: ::pyre_object::pyobject::get_instantiate(&#pytype_static),
                    },
                    ..payload
                };
                ::pyre_object::lltype::malloc_typed_young_nonmoving(full) as ::pyre_object::PyObjectRef
            }

            /// Stable-address variant of [`Self::allocate`]: allocates via
            /// the non-moving `malloc_typed_stable`, so the object never
            /// relocates.  Required for a self-mutating payload whose methods
            /// re-derive `self` from a raw `PyObjectRef` across an allocating
            /// call — a movable object would leave that raw pointer stale.
            #[allow(dead_code)]
            pub fn allocate_stable(payload: Self) -> ::pyre_object::PyObjectRef {
                let _roots = ::pyre_object::gc_roots::push_roots();
                let full = Self {
                    ob: ::pyre_object::PyObject {
                        ob_type: &#pytype_static as *const ::pyre_object::PyType,
                        w_class: ::pyre_object::pyobject::get_instantiate(&#pytype_static),
                    },
                    ..payload
                };
                ::pyre_object::lltype::malloc_typed_stable(full) as ::pyre_object::PyObjectRef
            }

            #allocate_instance_fn
        }
    })
}

// ──────────────────────────────────────────────────────────────────────
// `#[pyre_methods]` — PyPy `TypeDef("...", method=interp2app(W_X.m))`
// equivalent attached to an `impl W_X { ... }` block.
//
// User source:
//   #[pyre_methods]
//   impl W_Random {
//       fn __init__(&mut self, seed: Option<i64>) {
//           self.state = seed.unwrap_or(DEFAULT) as u64;
//       }
//       fn random(&mut self) -> f64 { ... }
//   }
//
// Emitted: one `args: &[PyObjectRef]` wrapper per method (downcasts
// `args[0]` to `&mut Self` via `from_obj`, unwraps the rest through
// the same `unwrap_expr` machinery `#[pyre_function]` uses, calls the
// typed method, wraps the return through `wrap_return`) plus
// `pub fn type_object()` that consumes `<Self as PyreClassPyTypeOf>::
// {PYNAME, PYTYPE}` to drive `make_builtin_type_with_layout` and
// `set_instantiate` exactly like `py_class_typed!` does.
// ──────────────────────────────────────────────────────────────────────

#[proc_macro_attribute]
pub fn pyre_methods(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as PyreMethodsAttrs);
    let imp = parse_macro_input!(item as ItemImpl);
    match expand_pyre_methods(attrs, imp) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `#[pyre_methods(doc = "...", weakrefable, unhashable)]` —
/// declarative slot keys propagated into the `type_object()`
/// registration closure.  Mirrors `TypeDef(..., __doc__=..., __weakref__=
/// make_weakref_descr(W_X), __hash__=None)` in
/// `pypy/interpreter/typedef.py`.
#[derive(Default)]
struct PyreMethodsAttrs {
    doc: Option<syn::LitStr>,
    /// TypeDef `_text_signature_` — the constructor signature exposed by
    /// `type.__text_signature__`.
    text_signature: Option<syn::LitStr>,
    weakrefable: bool,
    unhashable: bool,
    /// Optional base class expression — the `__base` of the TypeDef.
    /// Mirrors `TypeDef("X", __base=W_Base.typedef)` in
    /// `pypy/interpreter/typedef.py`.  Defaults to `w_object()` (the
    /// `object` base) when omitted.
    base: Option<syn::Expr>,
}

impl syn::parse::Parse for PyreMethodsAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut out = PyreMethodsAttrs::default();
        if input.is_empty() {
            return Ok(out);
        }
        loop {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "doc" => {
                    input.parse::<syn::Token![=]>()?;
                    out.doc = Some(input.parse()?);
                }
                "_text_signature_" => {
                    input.parse::<syn::Token![=]>()?;
                    out.text_signature = Some(input.parse()?);
                }
                "weakrefable" => out.weakrefable = true,
                "unhashable" => out.unhashable = true,
                "base" => {
                    input.parse::<syn::Token![=]>()?;
                    out.base = Some(input.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `#[pyre_methods]` key `{other}` — expected \
                             `doc = \"...\"` / `_text_signature_ = \"...\"` / \
                             `weakrefable` / `unhashable` / `base = <expr>`",
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
        }
        Ok(out)
    }
}

// PyPy `interp_attrproperty('state', cls=W_X, wrapfn=newint)`
// (`typedef.py`) is the read-only-field exposure helper.  The
// `#[getter]` / `#[setter]` machinery below covers the exact same
// ground in one line per slot — `#[getter] fn state(&self) -> i64
// { self.state as i64 }` — without a separate alias.  Documented
// here so future reviewers don't reintroduce `interp_attrproperty`
// as its own macro: there is no syntactic gap left to close.

#[derive(Clone)]
enum MethodKind {
    Instance,
    Static,
    Class,
    // `(py_name, doc)` — doc is the optional `doc="…"` arg shared by all
    // three accessors, mirroring `GetSetProperty(fget, fset, fdel, doc=)`.
    Getter(String, Option<String>),
    Setter(String, Option<String>),
    Deleter(String, Option<String>),
}

/// One `#[getter(...)]` / `#[setter(...)]` / `#[deleter(...)]` argument:
/// either a positional `"name"` (or `name = "…"`) or `doc = "…"`.
enum GetSetArg {
    Name(String),
    Doc(String),
}

impl syn::parse::Parse for GetSetArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(syn::LitStr) {
            let s: syn::LitStr = input.parse()?;
            return Ok(GetSetArg::Name(s.value()));
        }
        let key: syn::Ident = input.parse()?;
        let _: syn::Token![=] = input.parse()?;
        let val: syn::LitStr = input.parse()?;
        match key.to_string().as_str() {
            "name" => Ok(GetSetArg::Name(val.value())),
            "doc" => Ok(GetSetArg::Doc(val.value())),
            _ => Err(syn::Error::new(key.span(), "expected `name` or `doc`")),
        }
    }
}

/// Parse a `#[getter]` / `#[setter]` / `#[deleter]` attribute into its
/// optional `(name, doc)`.  Accepts `#[x]`, `#[x("name")]`,
/// `#[x(doc = "…")]`, and `#[x("name", doc = "…")]`.
fn parse_getset_attr(a: &syn::Attribute) -> syn::Result<(Option<String>, Option<String>)> {
    use syn::Meta;
    match &a.meta {
        Meta::Path(_) => Ok((None, None)),
        Meta::List(_) => {
            let args = a.parse_args_with(
                syn::punctuated::Punctuated::<GetSetArg, syn::Token![,]>::parse_terminated,
            )?;
            let mut name = None;
            let mut doc = None;
            for arg in args {
                match arg {
                    GetSetArg::Name(s) => name = Some(s),
                    GetSetArg::Doc(s) => doc = Some(s),
                }
            }
            Ok((name, doc))
        }
        Meta::NameValue(_) => Err(syn::Error::new(
            a.span(),
            "expected `#[attr]`, `#[attr(\"name\")]`, or `#[attr(doc = \"…\")]`",
        )),
    }
}

/// Inspect `#[staticmethod]` / `#[classmethod]` / `#[getter]` /
/// `#[setter]` markers on an impl-fn.  Mutually exclusive; the
/// unannotated default is `Instance`.
///
/// `#[getter]` / `#[setter]` accept an optional `(py_name)` arg.  When
/// omitted, the py-name is derived from the rust fn name (setters strip
/// a leading `set_` to pair with their getter).  A setter `set_traceback`
/// therefore pairs with getter `__traceback__` (and `set_CHUNK_SIZE` with
/// `_CHUNK_SIZE`) rather than requiring a rust name like `set___traceback__`.
/// Mirrors PyPy `name = GetSetProperty(W_X.descr_get_name, W_X.descr_set_name)`
/// where both descr handlers share the python-visible `name`.
fn classify_method(
    m: &syn::ImplItemFn,
    getter_names: &std::collections::HashSet<String>,
) -> syn::Result<MethodKind> {
    let mut kind = MethodKind::Instance;
    let mut seen = false;
    for a in m.attrs.iter() {
        let new_kind = if a.path().is_ident("staticmethod") {
            Some(MethodKind::Static)
        } else if a.path().is_ident("classmethod") {
            Some(MethodKind::Class)
        } else if a.path().is_ident("getter") {
            let (name, doc) = parse_getset_attr(a)?;
            let py_name = name.unwrap_or_else(|| m.sig.ident.to_string());
            Some(MethodKind::Getter(py_name, doc))
        } else if a.path().is_ident("setter") {
            let (name, doc) = parse_getset_attr(a)?;
            let py_name =
                derive_setdel_py_name("set_", &m.sig.ident.to_string(), name, getter_names);
            Some(MethodKind::Setter(py_name, doc))
        } else if a.path().is_ident("deleter") {
            let (name, doc) = parse_getset_attr(a)?;
            let py_name =
                derive_setdel_py_name("del_", &m.sig.ident.to_string(), name, getter_names);
            Some(MethodKind::Deleter(py_name, doc))
        } else {
            None
        };
        let Some(new_kind) = new_kind else { continue };
        if seen {
            return Err(syn::Error::new(
                a.span(),
                "#[pyre_methods]: at most one of \
                 `#[classmethod]` / `#[staticmethod]` / `#[getter]` / `#[setter]` / `#[deleter]`",
            ));
        }
        kind = new_kind;
        seen = true;
    }
    Ok(kind)
}

/// One python-visible GetSetProperty being assembled across `#[getter]` /
/// `#[setter]` / `#[deleter]` arms that share a py-name.  Mirrors
/// `GetSetProperty(fget, fset, fdel, doc=)`.
#[derive(Default, Clone)]
struct PropEntry {
    name: String,
    fget: Option<syn::Ident>,
    fset: Option<syn::Ident>,
    fdel: Option<syn::Ident>,
    doc: Option<String>,
}

/// Which accessor a `record_property` call contributes.
enum Accessor {
    Get,
    Set,
    Del,
}

/// Insert one accessor wrapper into the `properties` Vec, merging with an
/// existing entry on the same py-name and rejecting duplicate accessors.
/// A `doc` provided by any accessor is adopted; conflicting docs error.
fn record_property(
    properties: &mut Vec<PropEntry>,
    name: String,
    accessor: Accessor,
    wrapper: syn::Ident,
    doc: Option<String>,
    m: &syn::ImplItemFn,
) -> syn::Result<()> {
    let entry = match properties.iter_mut().find(|e| e.name == name) {
        Some(e) => e,
        None => {
            properties.push(PropEntry {
                name: name.clone(),
                ..PropEntry::default()
            });
            properties.last_mut().unwrap()
        }
    };
    let (slot, label) = match accessor {
        Accessor::Get => (&mut entry.fget, "getter"),
        Accessor::Set => (&mut entry.fset, "setter"),
        Accessor::Del => (&mut entry.fdel, "deleter"),
    };
    if slot.is_some() {
        return Err(syn::Error::new(
            m.sig.span(),
            format!("#[{label}]: property `{name}` already has a {label}"),
        ));
    }
    *slot = Some(wrapper);
    if let Some(doc) = doc {
        match &entry.doc {
            Some(existing) if *existing != doc => {
                return Err(syn::Error::new(
                    m.sig.span(),
                    format!("property `{name}` has conflicting `doc` values"),
                ));
            }
            _ => entry.doc = Some(doc),
        }
    }
    Ok(())
}

/// `set_traceback` → `__traceback__` when that getter exists, else
/// `traceback`.  Same for `set_CHUNK_SIZE` / `_CHUNK_SIZE` and deleters.
fn derive_setdel_py_name(
    prefix: &str,
    fn_name: &str,
    explicit: Option<String>,
    getter_names: &std::collections::HashSet<String>,
) -> String {
    if let Some(name) = explicit {
        return name;
    }
    let rest = fn_name.strip_prefix(prefix).unwrap_or(fn_name);
    if getter_names.contains(rest) {
        return rest.to_owned();
    }
    let underscored = format!("_{rest}");
    if getter_names.contains(&underscored) {
        return underscored;
    }
    let dunder = format!("__{rest}__");
    if getter_names.contains(&dunder) {
        return dunder;
    }
    rest.to_owned()
}

fn getter_py_name(m: &syn::ImplItemFn) -> Option<String> {
    for a in m.attrs.iter() {
        if !a.path().is_ident("getter") {
            continue;
        }
        let (name, _) = parse_getset_attr(a).ok()?;
        return Some(name.unwrap_or_else(|| m.sig.ident.to_string()));
    }
    None
}

fn expand_pyre_methods(
    attrs: PyreMethodsAttrs,
    mut imp: ItemImpl,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some((_, path, _)) = &imp.trait_ {
        return Err(syn::Error::new(
            path.span(),
            "#[pyre_methods] must annotate an inherent impl, not a trait impl",
        ));
    }
    let self_ty = (*imp.self_ty).clone();
    let getter_names: std::collections::HashSet<String> = imp
        .items
        .iter()
        .filter_map(|item| {
            let ImplItem::Fn(m) = item else {
                return None;
            };
            getter_py_name(m)
        })
        .collect();

    // Auto-synthesize `__new__` when the user wrote `__init__` but no
    // `__new__`.  Mirrors PyPy `TypeDef` behavior where a class without
    // an explicit `__new__` inherits `object.__new__` which allocates a
    // zero-initialized instance.  Requires the user struct to implement
    // `Default` (typically via `#[derive(Default)]` — `PyObject` itself
    // derives `Default` so the auto-derive resolves).
    let has_init = imp
        .items
        .iter()
        .any(|i| matches!(i, ImplItem::Fn(f) if f.sig.ident == "__init__"));
    let has_new = imp
        .items
        .iter()
        .any(|i| matches!(i, ImplItem::Fn(f) if f.sig.ident == "__new__"));
    if has_init && !has_new {
        let synth: ImplItem = parse_quote! {
            #[staticmethod]
            fn __new__(
                _cls: ::pyre_object::PyObjectRef,
                _args: &[::pyre_object::PyObjectRef],
            ) -> ::pyre_object::PyObjectRef {
                <#self_ty>::allocate(<#self_ty as ::std::default::Default>::default())
            }
        };
        imp.items.push(synth);
    }

    // Collect (python_name, wrapper_ident) per method as we rewrite the
    // impl block in-place: every `fn name(&self|&mut self, …)` keeps
    // its typed body untouched so users can call it directly from Rust,
    // and we synthesise a sibling `pub fn name__majit_wrapper(args)` as
    // a free-fn inside an attached `mod _pyre_wrappers_<Self>` module.
    let mut wrappers = Vec::<proc_macro2::TokenStream>::new();
    let mut registrations = Vec::<proc_macro2::TokenStream>::new();
    // `(py_name, fget_wrapper, fset_wrapper)` accumulated across the
    // method-loop.  Each `#[getter]` and `#[setter]` arm writes one
    // slot; after the loop we emit one `w_getset_property_new` per
    // distinct py_name.  Mirrors PyPy `name = GetSetProperty(fget=,
    // fset=)` where both handlers share the python-visible name.
    let mut properties: Vec<PropEntry> = Vec::new();

    for item in imp.items.iter() {
        let ImplItem::Fn(m) = item else { continue };
        if m.sig.asyncness.is_some()
            || m.sig.constness.is_some()
            || m.sig.unsafety.is_some()
            || !m.sig.generics.params.is_empty()
        {
            return Err(syn::Error::new(
                m.sig.span(),
                "#[pyre_methods]: async/const/unsafe/generic methods not supported",
            ));
        }
        let mname = &m.sig.ident;
        let wrapper_name = format_ident!("__majit_wrap_{}", mname);
        let wrapper_target_name = format_ident!("__majit_builtin_wrapper_target_{}", mname);
        let kind = classify_method(m, &getter_names)?;

        // Build per-kind wrapper preamble (self extraction) + call form,
        // and pick the registration constructor.
        //
        // PyPy `gateway.py`: `interp2app(W_X.fn, as_classmethod=True)` /
        // `as_staticmethod=True` decide whether the descriptor binds
        // `cls` / nothing instead of `self`.  Here the descriptor wrap is
        // applied at `make_builtin_function` time via `w_classmethod_new`
        // / `w_staticmethod_new`; everything else (arg unwrap, return
        // wrap) reuses the regular machinery.
        //
        // Getter / Setter share the Instance preamble — they both bind
        // `self` and look identical to a 0-arg / 1-arg method to the
        // wrapper machinery.  The distinction surfaces only at
        // registration time, where instead of `dict_storage_store(...,
        // make_builtin_function(...))` they participate in a deferred
        // `w_getset_property_new(fget=, fset=)` build keyed by py_name.
        let has_receiver = matches!(
            kind,
            MethodKind::Instance
                | MethodKind::Getter(..)
                | MethodKind::Setter(..)
                | MethodKind::Deleter(..)
        );
        let reads: Vec<ArgRead> = m
            .sig
            .inputs
            .iter()
            .filter_map(|arg| match arg {
                FnArg::Typed(pt) => Some(arg_read(&pt.ty)),
                FnArg::Receiver(_) => None,
            })
            .collect();
        let rooted = args_need_roots(&reads, has_receiver);
        let mut inputs = m.sig.inputs.iter().peekable();
        let (recv_check, preamble, call_target, first_arg_idx) = match &kind {
            MethodKind::Instance
            | MethodKind::Getter(..)
            | MethodKind::Setter(..)
            | MethodKind::Deleter(..) => {
                let recv = match inputs.next() {
                    Some(FnArg::Receiver(r)) => r,
                    _ => {
                        return Err(syn::Error::new(
                            m.sig.span(),
                            "#[pyre_methods]: instance method needs `&self` or `&mut self` \
                             — add `#[staticmethod]` / `#[classmethod]` to opt out",
                        ));
                    }
                };
                // Method dispatch passes `(self,)`+args; GetSetProperty
                // dispatch passes `(descriptor_self, w_obj,…)` so the
                // actual `self` slides one slot right.  Mirrors
                // `typedef.py` fget_unwrap_spec.
                let self_idx: usize = match &kind {
                    MethodKind::Getter(..) | MethodKind::Setter(..) | MethodKind::Deleter(..) => 1,
                    _ => 0,
                };
                let bind_self = if recv.mutability.is_some() {
                    quote! { s }
                } else {
                    // `from_obj` returns `Option<&mut Self>` even for
                    // `&self` callers — reborrow as `&Self` so the
                    // method's signature matches without a separate
                    // `from_obj_ref` API. Bind the reborrow in the `Some`
                    // arm: routing it through `Option::map` would add a
                    // closure aggregate and `call_once` to every wrapper.
                    quote! { &*s }
                };
                let needed = self_idx + 1;
                let recv_slot = arg_slot(self_idx, rooted);
                // `descr_check` names the descriptor as Python sees it, and
                // a `#[setter]` / `#[deleter]` reaches the type under the
                // property's name rather than the `set_` / `del_` prefixed
                // Rust one: `__traceback__`, not `set_traceback`.
                let descr_name = match &kind {
                    MethodKind::Getter(py_name, _)
                    | MethodKind::Setter(py_name, _)
                    | MethodKind::Deleter(py_name, _) => py_name.clone(),
                    _ => mname.to_string(),
                };
                // `method_check_args` runs `descr_check` against the owning
                // type, which the wrapper knows as `PYNAME`.  The layout test
                // here is the one that answers for a type carrying no
                // `method_owner` row, so it words the refusal the same way
                // rather than inventing its own.
                //
                // Split in two: `descr_check` precedes the clinic's own
                // argument parsing, so `BytesIO.seek(1, 2, 3, 4)` names the
                // receiver rather than the count.  The reference is bound
                // only after the unwraps, which can collect, so the test runs
                // twice and the second one never fails.
                // Written without an `Option` combinator: a closure here
                // becomes an aggregate and a `call_once` in every wrapper.
                let recv_check = quote! {
                    if args.len() < #needed || args[#self_idx].is_null() {
                        return ::pyre_interpreter::gateway::receiver_mismatch(
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                            #descr_name,
                            ::std::option::Option::None,
                        );
                    }
                    if <#self_ty>::from_obj(args[#self_idx]).is_none() {
                        return ::pyre_interpreter::gateway::receiver_mismatch(
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                            #descr_name,
                            ::std::option::Option::Some(args[#self_idx]),
                        );
                    }
                };
                let preamble = quote! {
                    if args.len() < #needed {
                        return ::pyre_interpreter::gateway::receiver_mismatch(
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                            #descr_name,
                            ::std::option::Option::None,
                        );
                    }
                    let __pyre_recv = #recv_slot;
                    let __pyre_self = match <#self_ty>::from_obj(__pyre_recv) {
                        ::std::option::Option::Some(s) => #bind_self,
                        ::std::option::Option::None => {
                            return ::pyre_interpreter::gateway::receiver_mismatch(
                                <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                                #descr_name,
                                ::std::option::Option::Some(__pyre_recv),
                            );
                        }
                    };
                };
                (
                    recv_check,
                    preamble,
                    quote! { __pyre_self.#mname },
                    self_idx + 1,
                )
            }
            MethodKind::Static => {
                if matches!(inputs.peek(), Some(FnArg::Receiver(_))) {
                    return Err(syn::Error::new(
                        m.sig.span(),
                        "#[staticmethod]: must not take `self` / `&self` / `&mut self`",
                    ));
                }
                (quote! {}, quote! {}, quote! { <#self_ty>::#mname }, 0)
            }
            MethodKind::Class => {
                if matches!(inputs.peek(), Some(FnArg::Receiver(_))) {
                    return Err(syn::Error::new(
                        m.sig.span(),
                        "#[classmethod]: first arg must be a typed `cls: PyObjectRef`, \
                         not `&self` / `&mut self`",
                    ));
                }
                (quote! {}, quote! {}, quote! { <#self_ty>::#mname }, 0)
            }
        };

        let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
        let mut reread_stmts = Vec::<proc_macro2::TokenStream>::new();
        let mut call_args = Vec::<proc_macro2::TokenStream>::new();
        let mut param_names = Vec::<String>::new();
        let mut param_required = Vec::<bool>::new();
        let mut param_positional = Vec::<bool>::new();
        // `SignatureBuilder` statements mirroring `#[pyre_function]`
        // (see `expand_pyre_function`): each parameter appends its name, a
        // `#[kwonly]` param emits `marker_kwonly()` at the tail's start, and a
        // `#[kwargs]` param records the `**kwargs` name instead of appending.
        let mut sig_stmts = Vec::<proc_macro2::TokenStream>::new();
        let mut has_varargs = false;
        let mut kwonly_tail = false;
        let mut kwonly_start = None;
        let mut kwonly_marked = false;
        for (offset, arg) in inputs.enumerate() {
            let FnArg::Typed(pt) = arg else {
                return Err(syn::Error::new(
                    arg.span(),
                    "#[pyre_methods]: unexpected receiver mid-signature",
                ));
            };
            let arg_idx = offset + first_arg_idx;
            if is_varargs_param(&pt.ty) {
                has_varargs = true;
            }
            let is_kwonly = pt.attrs.iter().any(|a| a.path().is_ident("kwonly"));
            let is_kwargs = pt.attrs.iter().any(|a| a.path().is_ident("kwargs"));
            if is_kwonly {
                kwonly_tail = true;
                // The first kwonly offset splits the rendered signature text.
                if kwonly_start.is_none() {
                    kwonly_start = Some(offset);
                }
            }
            let pname = param_name(offset, pt);
            // A bare `&[PyObjectRef]` (raw whole-args passthrough) suppresses
            // the signature entirely (`method_sig_body` below), so it
            // contributes no builder statement.
            if !is_varargs_param(&pt.ty) {
                if is_kwargs {
                    sig_stmts.push(quote! { __b.kwargname = ::std::option::Option::Some(#pname); });
                } else {
                    if is_kwonly && !kwonly_marked {
                        sig_stmts.push(quote! { __b.marker_kwonly(); });
                        kwonly_marked = true;
                    }
                    sig_stmts.push(quote! { __b.append(#pname); });
                }
            }
            param_names.push(pname);
            // Optional iff it has a `#[default(...)]` or is `Option<T>`.
            param_required.push(arg_default(pt)?.is_none() && option_inner(&pt.ty).is_none());
            param_positional.push(!kwonly_tail && !is_kwargs);
            let (stmt, reread, ident) = unwrap_arg(arg_idx, pt, None, rooted)?;
            unwrap_stmts.push(stmt);
            reread_stmts.extend(reread);
            call_args.push(quote! { #ident });
        }

        reject_duplicate_param_names(&param_names, m.sig.span())?;

        // Keyword-binding preamble (mirrors `#[pyre_function]`): when the
        // call carried a trailing `__pyre_kw__` dict, rebind positional +
        // keyword args into a resolved scope keyed by parameter name before
        // the receiver/arg unwraps run; otherwise the positional fast path
        // is untouched. Instance methods prepend a `self` slot — filled
        // positionally — so the receiver index lines up with the bound
        // scope; static/class methods already carry every slot (including
        // `cls`) as a typed parameter. Property getters/setters/deleters use
        // the descriptor calling convention and varargs methods consume the
        // whole slice, so both opt out.
        let kwargs_preamble = {
            let is_property = matches!(
                kind,
                MethodKind::Getter(..) | MethodKind::Setter(..) | MethodKind::Deleter(..)
            );
            if has_varargs || is_property {
                quote! {}
            } else if param_names.is_empty() {
                let receiver_slots = usize::from(matches!(kind, MethodKind::Instance));
                let expected_total = receiver_slots;
                let fn_name = mname.to_string();
                let arity_name = if mname == "__init__" {
                    quote! { <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME }
                } else {
                    quote! { #fn_name }
                };
                quote! {
                    // A method with no user-visible parameters has no keyword
                    // slots to bind.  Its valid-call fast path is the exact
                    // total arity; only a mismatch enters the cold classifier
                    // that distinguishes CALL_KW's marker from surplus
                    // positional arguments.
                    if args.len() != #expected_total {
                        return ::pyre_interpreter::gateway::method_noarg_failure(
                            args,
                            #arity_name,
                            #receiver_slots as i64,
                        );
                    }
                    // PyPy `argument.py:_match_signature` carries argument
                    // counts as RPython Signed.  Keep this generated gateway
                    // local in that type; `usize` belongs only at Rust slice
                    // indexing boundaries.
                    let __pyre_positional_count = args.len() as i64;
                }
            } else {
                quote! {
                    // Every `#[pyre_methods]` shim registers with a
                    // `Signature` (see `method_sig` below), so the call path
                    // (`call::bind_kwargs_to_signature`) has already resolved
                    // keywords into positional PY_NULL-padded slots before the
                    // wrapper runs — there is no trailing `__pyre_kw__` marker
                    // dict to peel.  PY_NULL is never a real argument, so the
                    // leading non-null run is the true positional count on both
                    // the keyword-bound and the raw positional path.
                    let __pyre_positional_count = ::pyre_interpreter::builtins::leading_non_null_count(args);
                }
            }
        };

        // PyPy's `interp2app` gateway validates the positional count before
        // entering the typed interp-level method.  `make_builtin_function`
        // stores no signature contract of its own, so the generated wrapper
        // must reject surplus positional arguments instead of silently
        // ignoring everything beyond the final typed parameter.
        let arity_preamble = {
            let is_property = matches!(
                kind,
                MethodKind::Getter(..) | MethodKind::Setter(..) | MethodKind::Deleter(..)
            );
            let has_keyword_slots = param_positional.iter().any(|positional| !positional);
            if has_varargs || is_property || has_keyword_slots {
                // See `expand_pyre_function`: once a slot can be filled by
                // keyword, the resolved scope no longer distinguishes it from
                // a positional one, and `bind_kwargs_to_signature` has already
                // applied the bound.
                quote! {}
            } else {
                let receiver_slots = usize::from(matches!(kind, MethodKind::Instance));
                let visible_max = param_positional
                    .iter()
                    .filter(|&&positional| positional)
                    .count();
                let visible_required = param_required
                    .iter()
                    .zip(param_positional.iter())
                    .filter(|(required, positional)| **required && **positional)
                    .count();
                // `Arguments.parse_obj` fills every parameter slot before the
                // body runs, so a keyword-only parameter widens `args` past
                // the positional count.  The positional bound stays in the
                // message, but the surplus guard compares against the full
                // slot count so a keyword-only slot is not mistaken for a
                // surplus positional.
                let expected_total = receiver_slots + param_positional.len();
                let fn_name = mname.to_string();
                // CPython's Argument Clinic reports constructor arity under
                // the concrete builtin type, not the implementation method's
                // bare `__init__` name. `interp2app` carries that owner in its
                // gateway metadata; use the pyre-class descriptor's same
                // compile-time owner here.
                let arity_name = if mname == "__init__" {
                    quote! { <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME }
                } else {
                    quote! { #fn_name }
                };
                let expected = if visible_required == visible_max {
                    match visible_max {
                        0 => "no arguments".to_string(),
                        1 => "exactly one argument".to_string(),
                        n => format!("exactly {n} arguments"),
                    }
                } else {
                    format!(
                        "at most {visible_max} argument{}",
                        if visible_max == 1 { "" } else { "s" }
                    )
                };
                // A missing required slot is rejected the same way: the
                // unwrap statements below index their slots directly, so the
                // body would otherwise read past the end of `args`.
                let expected_min = receiver_slots + visible_required;
                let too_few = if visible_required == visible_max {
                    quote! {
                        ::pyre_interpreter::gateway::method_arity_failure(
                            #arity_name,
                            #expected,
                            __pyre_user_positional_count,
                        )
                    }
                } else {
                    quote! {
                        ::pyre_interpreter::gateway::method_min_arity_failure(
                            #arity_name,
                            #visible_required as i64,
                            __pyre_user_positional_count,
                        )
                    }
                };
                quote! {
                    // PyPy's `input_argcount`/`upfront` calculation is
                    // ordinary Signed arithmetic.  Spell out the lower
                    // bound instead of introducing Rust's integer-method
                    // call into every generated gateway graph.
                    let __pyre_user_positional_count =
                        if __pyre_positional_count > #receiver_slots as i64 {
                            __pyre_positional_count - #receiver_slots as i64
                        } else {
                            0i64
                        };
                    if __pyre_positional_count < #expected_min as i64 {
                        return #too_few;
                    }
                    if __pyre_positional_count > #expected_total as i64 {
                        return ::pyre_interpreter::gateway::method_arity_failure(
                            #arity_name,
                            #expected,
                            __pyre_user_positional_count,
                        );
                    }
                }
            }
        };

        let call_inner = quote! { #call_target( #(#call_args),* ) };
        let body = wrap_return(&m.sig.output, call_inner)?;

        // `__new__` subclass override — PyPy `space.allocate_instance(W_X,
        // w_subtype)` semantics adapted for our static-PyType layout.
        // The user's body allocates via `W_X::allocate(...)` which stamps
        // the static `X_TYPE`; if the caller passed a subclass cls
        // (`class MyR(_random.Random)`), re-point `(*obj).w_class = cls`
        // so `type(obj)` / `isinstance(obj, MyR)` see the subclass while
        // the Rust-side `from_obj` (which keys on the static `ob_type`)
        // still resolves to W_X.  Mirrors `typedef.rs:int_descr_new`
        // (line 905-925) which already applies the same fix-up to
        // builtin int/float subclasses.
        let is_new = mname == "__new__" && matches!(kind, MethodKind::Static);
        let body = if is_new {
            quote! {
                // `space.allocate_instance(W_X, w_subtype)` keeps the subtype
                // live across allocation and initializer work.  The flat ABI
                // argument slice is not itself a moving-GC root, so pin and
                // reload it before stamping the result.
                let __pyre_new_roots = ::pyre_object::gc_roots::push_roots();
                let __pyre_cls_slot = __pyre_new_roots.pin_roots(&[__pyre_cls_input]);
                let __pyre_obj: ::pyre_object::PyObjectRef = match { #body } {
                    ::std::result::Result::Ok(o) => o,
                    ::std::result::Result::Err(e) => return ::std::result::Result::Err(e),
                };
                if !__pyre_obj.is_null() {
                    let __pyre_cls = __pyre_new_roots.get(__pyre_cls_slot);
                    if !__pyre_cls.is_null() && unsafe { ::pyre_object::is_type(__pyre_cls) } {
                        let __pyre_static_tp = ::pyre_interpreter::typedef::gettypefor(
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
                        );
                        let __pyre_same_tp = match __pyre_static_tp {
                            ::std::option::Option::Some(t) => __pyre_cls == t.as_ptr(),
                            ::std::option::Option::None => false,
                        };
                        if !__pyre_same_tp {
                            // `ob.w_class` is a traced edge of every
                            // `#[pyre_class]` layout, and an old-gen payload
                            // can take a young heap type here, so the store
                            // joins the remembered set.
                            unsafe { (*__pyre_obj).w_class = __pyre_cls; }
                            ::pyre_object::gc_hook::try_gc_write_barrier(
                                __pyre_obj as ::pyre_object::gc_hook::GCREF,
                            );
                        }
                    }
                }
                ::std::result::Result::Ok(__pyre_obj)
            }
        } else {
            body
        };

        // `__new__` reads `cls` before the unwrap bracket closes: after the
        // unwraps, `args` is stale wherever they could collect.
        let cls_read = if is_new {
            let cls_slot = arg_slot(0, rooted);
            quote! {
                let __pyre_cls_input = if args.is_empty() {
                    ::pyre_object::PY_NULL
                } else {
                    #cls_slot
                };
            }
        } else {
            quote! {}
        };
        let py_name = mname.to_string();
        let (roots_open, roots_close) = arg_roots_bracket(rooted);
        wrappers.push(quote! {
            #[allow(non_snake_case)]
            pub fn #wrapper_name(
                args: &[::pyre_object::PyObjectRef],
            ) -> ::std::result::Result<::pyre_object::PyObjectRef, ::pyre_interpreter::PyError> {
                #recv_check
                #kwargs_preamble
                #arity_preamble
                #roots_open
                #(#unwrap_stmts)*
                #(#reread_stmts)*
                // Gateway coercions may invoke Python (`__index__`, path
                // conversion, etc.) and therefore collect.  Perform them
                // before borrowing the typed receiver, matching interp2app's
                // unwrap-before-call boundary.
                #preamble
                #cls_read
                #roots_close
                #body
            }
        });
        wrappers.push(quote! {
            #[cfg(not(target_arch = "wasm32"))]
            #[::linkme::distributed_slice(::pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
            #[allow(non_upper_case_globals)]
            static #wrapper_target_name: ::pyre_interpreter::gateway::BuiltinWrapperDescriptor =
                ::pyre_interpreter::gateway::BuiltinWrapperDescriptor {
                    path: concat!(module_path!(), "::", stringify!(#wrapper_name)),
                    func: #wrapper_name,
                };

            // `distributed_slice` rejects wasm32. Register from a constructor
            // the way `#[pyre_class]` registers `PYRE_CLASS_DESCRIPTORS`.
            #[cfg(target_arch = "wasm32")]
            #[::ctor::ctor(unsafe)]
            fn #wrapper_target_name() {
                ::pyre_interpreter::gateway::register_builtin_wrapper_descriptor(
                    concat!(module_path!(), "::", stringify!(#wrapper_name)),
                    #wrapper_name,
                );
            }
        });
        // The argument `Signature` this method binds keywords against.  An
        // instance method prepends `self` as a positional-only slot (filled by
        // the receiver, never by keyword), mirroring interp2app's `self`
        // handling (`UnwrapSpec_Check.visit__W_Root`, gateway.py); a
        // raw whole-args slice has no by-name binding and carries `None`.
        // `#[staticmethod]` / `#[classmethod]` name every slot (including
        // `cls`) as a typed parameter, so they append no synthetic receiver.
        let sig_receiver = if matches!(kind, MethodKind::Instance) {
            quote! {
                __b.append("self");
                __b.marker_posonly();
            }
        } else {
            quote! {}
        };
        let method_sig_body = if has_varargs {
            quote! { ::std::option::Option::None }
        } else {
            quote! {
                let mut __b = ::pyre_interpreter::SignatureBuilder::default();
                #sig_receiver
                #(#sig_stmts)*
                ::std::option::Option::Some(__b.signature())
            }
        };
        let method_sig = quote! {{ #method_sig_body }};

        // gateway.py `_generate_text_signature`: a regular
        // interp2app instance method whose arguments are all required and
        // non-variadic has a lossless generated signature. Defaults need
        // their Python repr, so those continue to require an explicit
        // declaration rather than guessing from Rust tokens.
        let raw_fn = if matches!(kind, MethodKind::Instance)
            && !has_varargs
            && kwonly_start.is_none()
            && param_required.iter().all(|required| *required)
        {
            let mut parts = vec!["$self".to_string()];
            parts.extend(param_names.iter().cloned());
            parts.push("/".to_string());
            let text_signature = format!("({})", parts.join(", "));
            quote! {
                ::pyre_interpreter::gateway::make_builtin_function_with_text_signature_and_sig(
                    #py_name,
                    #wrapper_name,
                    #text_signature,
                    #method_sig,
                )
            }
        } else {
            quote! {
                ::pyre_interpreter::make_builtin_function_maybe_sig(
                    #py_name,
                    #wrapper_name,
                    #method_sig,
                )
            }
        };
        match &kind {
            MethodKind::Instance => {
                registrations.push(quote! {{
                        let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
                        let __pyre_built = #raw_fn;
                        let __pyre_built_slot = ::pyre_object::gc_roots::shadow_stack_len();
                        let _ = ::pyre_object::gc_roots::pin_root(__pyre_built);
                        unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    #py_name,
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_built_slot),
                ) };
                    }});
            }
            // `__new__` is the one static entry that is not a `tp_methods`
            // entry: `add_tp_new_wrapper` stores the carrier itself, so the
            // namespace holds a `builtin_function_or_method` bound to the
            // owning type rather than a `staticmethod` around one.
            MethodKind::Static if py_name == "__new__" => {
                registrations.push(quote! {{
                    let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
                    let __pyre_built = ::pyre_interpreter::typedef::make_new_descr_maybe_sig(#wrapper_name, #method_sig);
                    let __pyre_built_slot = ::pyre_object::gc_roots::shadow_stack_len();
                    let _ = ::pyre_object::gc_roots::pin_root(__pyre_built);
                    unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                #py_name,
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_built_slot),
            ) };
                }});
            }
            MethodKind::Static => {
                registrations.push(quote! {{
                        let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
                        let __pyre_built = #raw_fn;
                        let __pyre_built_slot = ::pyre_object::gc_roots::shadow_stack_len();
                        let _ = ::pyre_object::gc_roots::pin_root(__pyre_built);
                        let __pyre_wrapped = ::pyre_object::w_staticmethod_new(
                            ::pyre_object::gc_roots::shadow_stack_get(__pyre_built_slot),
                        );
                        let __pyre_wrapped_slot = ::pyre_object::gc_roots::shadow_stack_len();
                        let _ = ::pyre_object::gc_roots::pin_root(__pyre_wrapped);
                        unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    #py_name,
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_wrapped_slot),
                ) };
                    }});
            }
            MethodKind::Class => {
                registrations.push(quote! {{
                        let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
                        let __pyre_built = #raw_fn;
                        let __pyre_built_slot = ::pyre_object::gc_roots::shadow_stack_len();
                        let _ = ::pyre_object::gc_roots::pin_root(__pyre_built);
                        let __pyre_wrapped = ::pyre_object::w_classmethod_new(
                            ::pyre_object::gc_roots::shadow_stack_get(__pyre_built_slot),
                        );
                        let __pyre_wrapped_slot = ::pyre_object::gc_roots::shadow_stack_len();
                        let _ = ::pyre_object::gc_roots::pin_root(__pyre_wrapped);
                        unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    #py_name,
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_wrapped_slot),
                ) };
                    }});
            }
            MethodKind::Getter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Get,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
            MethodKind::Setter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Set,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
            MethodKind::Deleter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Del,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
        }
    }

    // Property emission: one `w_getset_property_new` per distinct
    // py-name.  Slots not provided fall back to `PY_NULL` (matching
    // PyPy `GetSetProperty(fget=W_X.descr_get_X, fset=None, fdel=None)`
    // when only a getter is declared).  A `doc="…"` provided on any
    // accessor populates the `doc` slot; otherwise `PY_NULL`.
    for prop in &properties {
        let prop_name = &prop.name;
        let accessor_slot = |slot: &Option<syn::Ident>, idx: u32| match slot {
            Some(id) => quote! {{
                let __pyre_acc = ::pyre_interpreter::make_builtin_function(#prop_name, #id);
                let __pyre_acc_slot = ::pyre_object::gc_roots::shadow_stack_len();
                let _ = ::pyre_object::gc_roots::pin_root(__pyre_acc);
                let _ = #idx;
                __pyre_acc_slot
            }},
            None => quote! { ::pyre_object::gc_roots::shadow_stack_len() },
        };
        // `None` accessors are `PY_NULL` and must not read a slot. The index
        // keeps each `Some` arm's locals in its own block.
        let fget_some = prop.fget.is_some();
        let fset_some = prop.fset.is_some();
        let fdel_some = prop.fdel.is_some();
        let doc_some = prop.doc.is_some();
        let fget_slot_expr = accessor_slot(&prop.fget, 0);
        let fset_slot_expr = accessor_slot(&prop.fset, 1);
        let fdel_slot_expr = accessor_slot(&prop.fdel, 2);
        let doc_expr = match &prop.doc {
            Some(doc) => quote! {{
                let __pyre_doc = ::pyre_object::w_str_new(#doc);
                let __pyre_doc_slot = ::pyre_object::gc_roots::shadow_stack_len();
                let _ = ::pyre_object::gc_roots::pin_root(__pyre_doc);
                __pyre_doc_slot
            }},
            None => quote! { 0usize },
        };
        registrations.push(quote! {{
            let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
            let __pyre_fget_slot = #fget_slot_expr;
            let __pyre_fset_slot = #fset_slot_expr;
            let __pyre_fdel_slot = #fdel_slot_expr;
            let __pyre_doc_slot = #doc_expr;
            let __pyre_name = ::pyre_object::w_str_new(#prop_name);
            let __pyre_name_slot = ::pyre_object::gc_roots::shadow_stack_len();
            let _ = ::pyre_object::gc_roots::pin_root(__pyre_name);
            let __pyre_prop = ::pyre_object::typedef::w_getset_property_new(
                if #fget_some {
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_fget_slot)
                } else {
                    ::pyre_object::PY_NULL
                },
                if #fset_some {
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_fset_slot)
                } else {
                    ::pyre_object::PY_NULL
                },
                if #fdel_some {
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_fdel_slot)
                } else {
                    ::pyre_object::PY_NULL
                },
                if #doc_some {
                    ::pyre_object::gc_roots::shadow_stack_get(__pyre_doc_slot)
                } else {
                    ::pyre_object::PY_NULL
                },
                ::pyre_object::PY_NULL,
                false,
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_name_slot),
            );
            let __pyre_prop_slot = ::pyre_object::gc_roots::shadow_stack_len();
            let _ = ::pyre_object::gc_roots::pin_root(__pyre_prop);
            unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                #prop_name,
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_prop_slot),
            ) };
        }});
    }

    // Append declarative-slot registrations after method entries.  Order
    // matches PyPy `TypeDef("X", method=..., __doc__=..., __weakref__=...,
    // __hash__=None)`: methods first, slots last.  `make_weakref_descr`
    // returns the canonical descriptor; the typedef install pass
    // (`make_builtin_type_with_layout`) sees `__weakref__` in `ns` and
    // flips the `weakrefable` bit on the new type via
    // `w_type_set_weakrefable`.
    if let Some(lit) = attrs.doc.as_ref() {
        registrations.push(quote! {{
            let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
            let __pyre_doc = ::pyre_object::w_str_new(#lit);
            let __pyre_doc_slot = ::pyre_object::gc_roots::shadow_stack_len();
            let _ = ::pyre_object::gc_roots::pin_root(__pyre_doc);
            unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "__doc__",
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_doc_slot),
            ) };
        }});
    }
    if attrs.weakrefable {
        registrations.push(quote! {{
            let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
            let __pyre_weak = ::pyre_interpreter::typedef::make_weakref_descr(::pyre_object::PY_NULL);
            let __pyre_weak_slot = ::pyre_object::gc_roots::shadow_stack_len();
            let _ = ::pyre_object::gc_roots::pin_root(__pyre_weak);
            unsafe { ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "__weakref__",
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_weak_slot),
            ) };
        }});
    }
    if attrs.unhashable {
        registrations.push(quote! {
                unsafe { {
            let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
            let __pyre_value_now_1 = ::pyre_object::w_none();
            let __pyre_value_slot_1 = ::pyre_object::gc_roots::shadow_stack_len();
            let _ = ::pyre_object::gc_roots::pin_root(__pyre_value_now_1);
            ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "__hash__",
                ::pyre_object::gc_roots::shadow_stack_get(__pyre_value_slot_1),
            )
        } };
            });
    }

    // Strip marker attrs the user placed.  `#[pyre_method]` /
    // `#[pyre_property]` are future-compat no-ops today; `#[classmethod]`
    // / `#[staticmethod]` / `#[default(...)]` were already consumed by
    // the wrapper-generation pass above and would otherwise leak into
    // the emitted impl block and confuse rustc.
    for item in imp.items.iter_mut() {
        let ImplItem::Fn(m) = item else { continue };
        m.attrs.retain(|a| {
            !(a.path().is_ident("pyre_method")
                || a.path().is_ident("pyre_property")
                || a.path().is_ident("classmethod")
                || a.path().is_ident("staticmethod")
                || a.path().is_ident("getter")
                || a.path().is_ident("setter")
                || a.path().is_ident("deleter"))
        });
        for arg in m.sig.inputs.iter_mut() {
            if let FnArg::Typed(pt) = arg {
                pt.attrs.retain(|a| {
                    !a.path().is_ident("default")
                        && !a.path().is_ident("kwonly")
                        && !a.path().is_ident("kwargs")
                });
            }
        }
        rewrite_alias_args(&mut m.sig);
    }

    // Base class for the TypeDef.  `#[pyre_methods(base = <expr>)]`
    // supplies the parent type; absent it, the class inherits `object`.
    let base_expr = match &attrs.base {
        Some(e) => quote! { #e },
        None => quote! { ::pyre_interpreter::typedef::w_object() },
    };
    let set_type_text_signature = match &attrs.text_signature {
        Some(signature) => quote! {
            unsafe { ::pyre_object::w_type_set_text_signature(tp, #signature) };
        },
        None => quote! {},
    };

    let type_object_fn = quote! {
        pub fn type_object() -> ::pyre_object::PyObjectRef {
            // PyPy TypeDef objects are interpreter/process-owned, never
            // execution-context or thread-local.  Store the immortal pointer
            // as usize so the process-global cache remains Sync without
            // duplicating Python type identity per thread.
            static CELL: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
            *CELL.get_or_init(|| {
                    let tp = ::pyre_interpreter::typedef::make_builtin_type_with_layout(
                        <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                        |ns| {
                            let _root_scope = ::pyre_object::gc_roots::push_roots();
                            let ns_slot = ::pyre_object::gc_roots::shadow_stack_len();
                            let _ = ::pyre_object::gc_roots::pin_root(ns);
                            #(#registrations)*
                        },
                        #base_expr,
                        <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
                    );
                    // [3.14-spec] The declaration records CPython's public
                    // owner/mutability axes while the PyPy TypeDef produced
                    // above remains structurally builtin.
                    unsafe {
                        ::pyre_object::w_type_set_cpython_type_flags(
                            tp,
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::CPYTHON_HEAPTYPE,
                            !<#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::CPYTHON_HEAPTYPE,
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::CPYTHON_IMMUTABLETYPE,
                        );
                    }
                    ::pyre_object::pyobject::set_instantiate(
                        unsafe {
                            &*<#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE
                        },
                        tp,
                    );
                    #set_type_text_signature
                    tp as usize
                }) as ::pyre_object::PyObjectRef
        }

        // Publish this accessor's residual-call address.  `type_object` is
        // `dont_look_inside` (the JIT never lifts the `OnceLock` body — its
        // `CELL` read is unliftable host plumbing), so the codewriter emits a
        // residual call the runtime resolves through `jit_trace_fnaddrs`.
        ::pyre_object::register_type_object_fnaddr!();
    };

    Ok(quote! {
        #imp

        #(#wrappers)*

        #type_object_fn
    })
}
