//! `external_compilation_info!` input grammar and the `llexternal!` head
//! `compile_crate_ecis` reads for `macro` wrappers.

use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Ident, LitBool, LitStr, Token, Type, Visibility, bracketed};

use crate::{_ATTRIBUTES, ExternalCompilationInfo};

pub struct EciInput {
    pub vis: Visibility,
    pub name: Ident,
    pub fields: Vec<(Ident, EciValue)>,
}

pub enum EciValue {
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
    pub fn to_eci(&self) -> syn::Result<ExternalCompilationInfo> {
        let mut eci = ExternalCompilationInfo::default();
        for (key, value) in &self.fields {
            let name = key.to_string();
            match value {
                EciValue::Bool(value) => {
                    if name != "use_cpp_linker" {
                        return Err(syn::Error::new(key.span(), "only use_cpp_linker is a bool"));
                    }
                    eci.use_cpp_linker = *value;
                }
                EciValue::Strs(strs) => {
                    if !_ATTRIBUTES.contains(&name.as_str()) {
                        return Err(syn::Error::new(
                            key.span(),
                            format!("unknown ExternalCompilationInfo field `{name}`"),
                        ));
                    }
                    let values: Vec<String> = strs.iter().map(LitStr::value).collect();
                    set_strings(&mut eci, &name, values);
                }
            }
        }
        Ok(eci)
    }
}

/// The `llexternal!` tokens `compile_crate_ecis` reads: attributes (discarded),
/// visibility, name, c_name, argument types, result type, then `key = expr`
/// pairs. Only `macro` and `compilation_info` are kept.
pub struct LlexternalHead {
    pub name: Ident,
    pub c_name: String,
    pub args: Vec<Type>,
    pub result: Type,
    pub macro_: Option<Expr>,
    pub compilation_info: Option<Expr>,
}

impl Parse for LlexternalHead {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _attrs = input.call(Attribute::parse_outer)?;
        let _vis: Visibility = input.parse()?;
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

        let mut macro_ = None;
        let mut compilation_info = None;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;
            match key.to_string().as_str() {
                "macro" => macro_ = Some(value),
                "compilation_info" => compilation_info = Some(value),
                _ => {}
            }
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after llexternal arguments"));
        }
        Ok(LlexternalHead {
            name,
            c_name: c_name.value(),
            args: args.into_iter().collect(),
            result,
            macro_,
            compilation_info,
        })
    }
}

fn set_strings(eci: &mut ExternalCompilationInfo, name: &str, values: Vec<String>) {
    match name {
        "pre_include_bits" => eci.pre_include_bits = values,
        "includes" => eci.includes = values,
        "include_dirs" => eci.include_dirs = values,
        "post_include_bits" => eci.post_include_bits = values,
        "libraries" => eci.libraries = values,
        "library_dirs" => eci.library_dirs = values,
        "separate_module_sources" => eci.separate_module_sources = values,
        "separate_module_files" => eci.separate_module_files = values,
        "compile_extra" => eci.compile_extra = values,
        "link_extra" => eci.link_extra = values,
        "frameworks" => eci.frameworks = values,
        "link_files" => eci.link_files = values,
        "testonly_libraries" => eci.testonly_libraries = values,
        other => panic!("unknown ExternalCompilationInfo field `{other}`"),
    }
}
