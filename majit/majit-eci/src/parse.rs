//! `external_compilation_info!` input grammar and the `llexternal!` /
//! `cextern_variable!` / `copaque_ptr!` heads `compile_crate_ecis` reads,
//! plus the `llexternal_use_eci!` head the proc macro shares.

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

/// The `llexternal!` tokens `compile_crate_ecis` reads: outer attributes,
/// visibility, name, c_name, argument types, result type, then `key = expr`
/// pairs. Only `macro` and `compilation_info` are kept; `attrs` is the outer
/// attribute list so a `cfg` on the declaration can skip it.
pub struct LlexternalHead {
    pub attrs: Vec<Attribute>,
    pub name: Ident,
    pub c_name: String,
    pub args: Vec<Type>,
    pub result: Type,
    pub macro_: Option<Expr>,
    pub compilation_info: Option<Expr>,
}

impl Parse for LlexternalHead {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
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
            attrs,
            name,
            c_name: c_name.value(),
            args: args.into_iter().collect(),
            result,
            macro_,
            compilation_info,
        })
    }
}

/// The `cextern_variable!` tokens `compile_crate_ecis` and the proc macro
/// share: outer attributes, getter, optional setter, C name, type, then
/// `key = expr` pairs (`rffi.CExternVariable`).
pub struct CExternVariableHead {
    pub attrs: Vec<Attribute>,
    pub getter: (Visibility, Ident),
    pub setter: Option<(Visibility, Ident)>,
    pub name: String,
    pub ty: Type,
    pub compilation_info: Expr,
    pub sandboxsafe: Option<Expr>,
    pub nowrapper: Option<Expr>,
    pub c_type: Option<String>,
    pub getter_only: bool,
    pub declare_as_extern: Option<bool>,
}

impl Parse for CExternVariableHead {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let getter_vis: Visibility = input.parse()?;
        let getter: Ident = input.parse()?;
        let setter = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let setter_vis: Visibility = input.parse()?;
            let setter: Ident = input.parse()?;
            Some((setter_vis, setter))
        } else {
            None
        };
        input.parse::<Token![=]>()?;
        let name: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let ty: Type = input.parse()?;

        let mut compilation_info = None;
        let mut sandboxsafe = None;
        let mut nowrapper = None;
        let mut c_type = None;
        let mut getter_only = false;
        let mut declare_as_extern = None;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "compilation_info" => compilation_info = Some(input.parse()?),
                "sandboxsafe" => sandboxsafe = Some(input.parse()?),
                "_nowrapper" => nowrapper = Some(input.parse()?),
                "c_type" => c_type = Some(input.parse::<LitStr>()?.value()),
                "getter_only" => getter_only = input.parse::<LitBool>()?.value(),
                "declare_as_extern" => declare_as_extern = Some(input.parse::<LitBool>()?.value()),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown cextern_variable kwarg `{other}`"),
                    ));
                }
            }
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after cextern_variable arguments"));
        }
        let compilation_info = compilation_info.ok_or_else(|| {
            syn::Error::new(getter.span(), "cextern_variable requires compilation_info")
        })?;
        if setter.is_some() && getter_only {
            return Err(syn::Error::new(
                getter.span(),
                "cextern_variable with two names cannot set getter_only = true",
            ));
        }
        if setter.is_none() && !getter_only {
            return Err(syn::Error::new(
                getter.span(),
                "cextern_variable with one name requires getter_only = true",
            ));
        }
        Ok(CExternVariableHead {
            attrs,
            getter: (getter_vis, getter),
            setter,
            name: name.value(),
            ty,
            compilation_info,
            sandboxsafe,
            nowrapper,
            c_type,
            getter_only,
            declare_as_extern,
        })
    }
}

/// The `copaque_ptr!` tokens `compile_crate_ecis` and the proc macro share:
/// outer attributes, visibility, name, c_name, then `key = expr` pairs
/// (`rffi.COpaquePtr`). `typedef` / `c_pointer_typedef` is rejected.
pub struct COpaquePtrHead {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub name: Ident,
    pub c_name: String,
    pub compilation_info: Option<Expr>,
}

impl COpaquePtrHead {
    /// Per-declaration injective key: escaped `c_name`, `__`, then the Rust
    /// name (`COpaque.lazy_getsize` is one cache per declaration).
    pub fn size_key(&self) -> String {
        copaque_size_key(&self.c_name, &self.name.to_string())
    }
}

/// `$OUT_DIR/majit_copaque/<key>.rs` stem: escaped `c_name` + `__` + `rust_name`.
/// Escaping keeps `[A-Za-z0-9]` and writes every other byte (including `_`) as
/// `_` plus two lowercase hex digits.
pub fn copaque_size_key(c_name: &str, rust_name: &str) -> String {
    let mut out = String::new();
    for b in c_name.bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(char::from(b));
        } else {
            out.push('_');
            out.push(char::from(hex_digit(b >> 4)));
            out.push(char::from(hex_digit(b & 0x0f)));
        }
    }
    out.push_str("__");
    out.push_str(rust_name);
    out
}

fn hex_digit(n: u8) -> u8 {
    if n < 10 { b'0' + n } else { b'a' + (n - 10) }
}

impl Parse for COpaquePtrHead {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let c_name: LitStr = input.parse()?;

        let mut compilation_info = None;
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "compilation_info" => compilation_info = Some(input.parse()?),
                "typedef" => {
                    return Err(syn::Error::new(
                        key.span(),
                        "copaque_ptr does not support `typedef`",
                    ));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown copaque_ptr kwarg `{other}`"),
                    ));
                }
            }
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after copaque_ptr arguments"));
        }
        Ok(COpaquePtrHead {
            attrs,
            vis,
            name,
            c_name: c_name.value(),
            compilation_info,
        })
    }
}

/// The `llexternal_use_eci!` tokens the proc macro parses: outer attributes,
/// visibility, optional `fn`, name, then the ECI (`rffi.llexternal_use_eci`).
pub struct LlexternalUseEciHead {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub name: Ident,
    pub compilation_info: Expr,
}

impl Parse for LlexternalUseEciHead {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        if input.peek(Token![fn]) {
            input.parse::<Token![fn]>()?;
        }
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let compilation_info: Expr = input.parse()?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after llexternal_use_eci arguments"));
        }
        Ok(LlexternalUseEciHead {
            attrs,
            vis,
            name,
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
