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

/// Peel `Type::Paren`/`Type::Group` and follow `Resolved::Alias` to the type
/// the name stands for, with the module path that name lives in.
fn resolve_target(
    ty: &Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
) -> Result<(Type, Vec<String>), String> {
    let mut current = ty.clone();
    let mut module_path = module_path.to_vec();
    let mut hops = 0;
    loop {
        current = match current {
            Type::Paren(paren) => *paren.elem,
            Type::Group(group) => *group.elem,
            Type::Path(type_path) if type_path.qself.is_none() => {
                match resolve(&type_path.path, &module_path) {
                    Some(Resolved::Alias(target, alias_module)) => {
                        hops += 1;
                        if hops > 64 {
                            return Err("alias cycle".into());
                        }
                        module_path = alias_module;
                        target
                    }
                    _ => return Ok((Type::Path(type_path), module_path)),
                }
            }
            other => return Ok((other, module_path)),
        };
    }
}

/// `FuncType.__init__`: function arguments can only be primitive or pointer.
pub fn functype_arg(
    ty: &Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
) -> Result<String, String> {
    functype_slot(
        ty,
        module_path,
        resolve,
        "function arguments can only be primitive or pointer",
    )
}

/// `FuncType.__init__`: function result can only be primitive or pointer.
pub fn functype_result(
    ty: &Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
) -> Result<String, String> {
    functype_slot(
        ty,
        module_path,
        resolve,
        "function result can only be primitive or pointer",
    )
}

fn functype_slot(
    ty: &Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
    container_msg: &str,
) -> Result<String, String> {
    let (target, target_module) = resolve_target(ty, module_path, resolve)?;
    match &target {
        Type::Array(_) => return Err(container_msg.into()),
        Type::Path(type_path) if type_path.qself.is_none() => {
            if matches!(
                resolve(&type_path.path, &target_module),
                Some(Resolved::LocalStruct(_))
            ) {
                return Err(container_msg.into());
            }
        }
        _ => {}
    }
    gettype(ty, module_path, resolve)
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
            // `database.py gettype` on a `Ptr`: the node of the resolved
            // target `T.TO` decides; `FixedSizeArrayDefNode.getptrtype`
            // returns a pointer to the item.
            let (elem, elem_module) = resolve_target(&ptr.elem, module_path, resolve)?;
            let inner = match &elem {
                Type::Array(arr) => {
                    fixed_array_length(arr, ty)?;
                    gettype(&arr.elem, &elem_module, resolve)?
                }
                _ => gettype(&ptr.elem, module_path, resolve)?,
            };
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
                syn::ReturnType::Type(_, out) => functype_result(out, module_path, resolve)?,
            };
            let params = if bare.inputs.is_empty() {
                "void".to_string()
            } else {
                let mut parts = Vec::new();
                for input in &bare.inputs {
                    let t = functype_arg(&input.ty, module_path, resolve)?;
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
        // `FixedSizeArrayDefNode.fulltypename`
        Type::Array(arr) => {
            let n = fixed_array_length(arr, ty)?;
            Ok(gettype(&arr.elem, module_path, resolve)?.replace('@', &format!("(@)[{n}]")))
        }
        _ => Err(format!(
            "unsupported type for generate_macro_wrapper: {}",
            tokens(ty)
        )),
    }
}

fn unwrap_expr(expr: &syn::Expr) -> &syn::Expr {
    let mut e = expr;
    loop {
        match e {
            syn::Expr::Paren(paren) => e = &paren.expr,
            syn::Expr::Group(group) => e = &group.expr,
            other => return other,
        }
    }
}

fn fixed_array_length(arr: &syn::TypeArray, ty: &Type) -> Result<usize, String> {
    let unsupported = || {
        format!(
            "unsupported array length for generate_macro_wrapper: {}",
            tokens(ty)
        )
    };
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(n),
        ..
    }) = unwrap_expr(&arr.len)
    {
        n.base10_parse::<usize>().map_err(|_| unsupported())
    } else {
        Err(unsupported())
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
