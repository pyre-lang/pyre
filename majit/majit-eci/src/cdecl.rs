//! `database.py gettype` type templates (`@` marks the declarator) and `support.py cdecl`.

use syn::Type;

/// Struct tags `gettype` spells as `struct <name>`. libc's Rust types
/// carry no `c_name`/typedef hint, so this table stands in for the `Struct`
/// hints `database.py gettype` reads.
const STRUCT_TAGS: &[&str] = &[
    "stat",
    "dirent",
    "timeval",
    "timespec",
    "pollfd",
    "sigaction",
    "tm",
    "sockaddr",
    "sockaddr_in",
    "sockaddr_in6",
    "sockaddr_un",
    "addrinfo",
    "msghdr",
    "iovec",
    "rlimit",
    "rusage",
    "statvfs",
    "utsname",
    "termios",
    "passwd",
    "group",
];

/// `support.py cdecl`: insert `cname` at each `@` in a `gettype` template.
pub fn cdecl(ctype: &str, cname: &str) -> String {
    ctype
        .replace("(@)", "@")
        .replace('@', cname)
        .trim()
        .to_string()
}

/// A path `gettype` resolved in the crate: an alias to expand, a path into
/// another crate, or a crate-local struct.
pub enum Resolved {
    Alias(syn::Type, Vec<String>),
    Extern(Vec<String>),
    LocalStruct(String),
}

/// `LowLevelDatabase.gettype` type template for `ty`. `@` marks the declarator.
pub fn gettype(
    ty: &syn::Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
) -> Result<String, String> {
    match ty {
        Type::Tuple(tuple) if tuple.elems.is_empty() => Ok("void @".into()),
        Type::Ptr(ptr) => {
            let inner = gettype(&ptr.elem, module_path, resolve)?;
            if ptr.const_token.is_some() {
                if !inner.contains('*') && !inner.contains('(') {
                    Ok(format!("const {}", inner.replace('@', "*@")))
                } else {
                    Ok(inner.replace('@', "const *@"))
                }
            } else {
                Ok(inner.replace('@', "*@"))
            }
        }
        Type::BareFn(bare) => {
            if bare.variadic.is_some() {
                return Err(format!(
                    "unsupported type for generate_macro_wrapper: {}",
                    tokens(ty)
                ));
            }
            let result = match &bare.output {
                syn::ReturnType::Default => "void @".to_string(),
                syn::ReturnType::Type(_, out) => gettype(out, module_path, resolve)?,
            };
            let params = if bare.inputs.is_empty() {
                "void".to_string()
            } else {
                let mut parts = Vec::new();
                for input in &bare.inputs {
                    let t = gettype(&input.ty, module_path, resolve)?;
                    parts.push(cdecl(&t, ""));
                }
                parts.join(", ")
            };
            Ok(result.replace('@', &format!("(*@)({params})")))
        }
        Type::Path(type_path) if type_path.qself.is_none() => {
            if let Some(inner) = option_of_bare_fn(&type_path.path) {
                return gettype(inner, module_path, resolve);
            }
            let segments = &type_path.path.segments;
            if segments.len() == 1 {
                let name = segments[0].ident.to_string();
                if matches!(
                    name.as_str(),
                    "i8" | "u8"
                        | "i16"
                        | "u16"
                        | "i32"
                        | "u32"
                        | "i64"
                        | "u64"
                        | "isize"
                        | "usize"
                        | "f32"
                        | "f64"
                ) {
                    return Ok(format!("{} @", last_segment_c_type(&name)));
                }
            }
            match resolve(&type_path.path, module_path) {
                Some(Resolved::Alias(target, alias_module)) => {
                    gettype(&target, &alias_module, resolve)
                }
                Some(Resolved::Extern(p)) => match p.last() {
                    Some(last) => Ok(format!("{} @", last_segment_c_type(last))),
                    None => Err(format!(
                        "cannot resolve type `{}` for generate_macro_wrapper",
                        tokens(ty)
                    )),
                },
                Some(Resolved::LocalStruct(n)) => {
                    // `database.py gettype` spells an `lltype.Struct` as
                    // `struct <c_name>` unless it has a `typedef` hint; a crate
                    // `#[repr(C)]` struct is such a struct.
                    Ok(format!("struct {n} @"))
                }
                None => Err(format!(
                    "cannot resolve type `{}` for generate_macro_wrapper",
                    tokens(ty)
                )),
            }
        }
        Type::Paren(paren) => gettype(&paren.elem, module_path, resolve),
        Type::Group(group) => gettype(&group.elem, module_path, resolve),
        _ => Err(format!(
            "unsupported type for generate_macro_wrapper: {}",
            tokens(ty)
        )),
    }
}

fn option_of_bare_fn(path: &syn::Path) -> Option<&Type> {
    let last = path.segments.last()?;
    if last.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    let syn::GenericArgument::Type(inner) = &args.args[0] else {
        return None;
    };
    let mut t = inner;
    loop {
        match t {
            Type::Paren(paren) => t = &paren.elem,
            Type::Group(group) => t = &group.elem,
            Type::BareFn(_) => return Some(inner),
            _ => return None,
        }
    }
}

fn last_segment_c_type(name: &str) -> String {
    match name {
        "i8" => "signed char".into(),
        "u8" => "unsigned char".into(),
        "i16" => "short".into(),
        "u16" => "unsigned short".into(),
        "i32" => "int".into(),
        "u32" => "unsigned int".into(),
        "i64" => "long long".into(),
        "u64" => "unsigned long long".into(),
        "isize" => "Signed".into(),
        "usize" => "Unsigned".into(),
        "f32" => "float".into(),
        "f64" => "double".into(),
        "c_char" => "char".into(),
        "c_schar" => "signed char".into(),
        "c_uchar" => "unsigned char".into(),
        "c_short" => "short".into(),
        "c_ushort" => "unsigned short".into(),
        "c_int" => "int".into(),
        "c_uint" => "unsigned int".into(),
        "c_long" => "long".into(),
        "c_ulong" => "unsigned long".into(),
        "c_longlong" => "long long".into(),
        "c_ulonglong" => "unsigned long long".into(),
        "c_float" => "float".into(),
        "c_double" => "double".into(),
        "c_void" => "void".into(),
        tag if STRUCT_TAGS.contains(&tag) => format!("struct {tag}"),
        other => other.to_string(),
    }
}

fn tokens(ty: &Type) -> String {
    quote::quote!(#ty).to_string()
}
