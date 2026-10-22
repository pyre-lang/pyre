//! `rposix.py` `replace_os_function` / `rtime.py` `replace_time_function`.
//!
//! Under `feature = "sandbox"` the wrapper body is
//! `rsandbox.py` `make_sandbox_trampoline`; otherwise the original body stays.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, LitStr, Pat, parse_macro_input};

pub fn expand_replace_os_function(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_replace_function(attr, item, "ll_os.ll_os_")
}

pub fn expand_replace_time_function(attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_replace_function(attr, item, "ll_time.ll_time_")
}

fn expand_replace_function(attr: TokenStream, item: TokenStream, prefix: &str) -> TokenStream {
    let name = parse_macro_input!(attr as LitStr);
    let func = parse_macro_input!(item as ItemFn);
    let wire_name = format!("{}{}", prefix, name.value());

    let mut arg_idents = Vec::new();
    for input in &func.sig.inputs {
        match input {
            FnArg::Receiver(recv) => {
                return syn::Error::new_spanned(
                    recv,
                    "replace_os_function / replace_time_function apply to free functions",
                )
                .to_compile_error()
                .into();
            }
            FnArg::Typed(pat) => {
                let ident = match &*pat.pat {
                    Pat::Ident(p) => p.ident.clone(),
                    other => {
                        return syn::Error::new_spanned(
                            other,
                            "replace_os_function / replace_time_function need identifier arguments",
                        )
                        .to_compile_error()
                        .into();
                    }
                };
                arg_idents.push(ident);
            }
        }
    }

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let inputs = &func.sig.inputs;
    let request_fn = format_ident!("{}_sandbox_request_bytes", func.sig.ident);

    quote! {
        #[cfg(not(feature = "sandbox"))]
        #func

        #(#attrs)*
        #[cfg(feature = "sandbox")]
        #vis #sig {
            ::majit_rlib::rsandbox::run_trampoline(
                #wire_name,
                &[#( ::majit_rlib::rmarshal::SandboxArg::to_marshal(&#arg_idents) ),*],
            )
        }

        #[cfg(all(test, unix))]
        pub(crate) fn #request_fn(#inputs) -> Vec<u8> {
            ::majit_rlib::rsandbox::encode_request(
                #wire_name,
                &[#( ::majit_rlib::rmarshal::SandboxArg::to_marshal(&#arg_idents) ),*],
            )
        }
    }
    .into()
}
