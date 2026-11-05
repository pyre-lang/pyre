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
    /// A platform typedef whose class `rffi_platform` measured.
    Int(crate::rffi_platform::IntType),
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
    let mut visited: Vec<(Vec<String>, String)> = Vec::new();
    loop {
        current = match current {
            Type::Paren(paren) => *paren.elem,
            Type::Group(group) => *group.elem,
            Type::Path(type_path) if type_path.qself.is_none() => {
                match resolve(&type_path.path, &module_path) {
                    Some(Resolved::Alias(target, alias_module)) => {
                        let path = &type_path.path;
                        let key = (module_path.clone(), quote::quote!(#path).to_string());
                        if visited.contains(&key) {
                            return Err(format!("alias cycle: {}", key.1));
                        }
                        visited.push(key);
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

/// `FuncType.__init__`: function result can only be primitive or pointer.
pub fn functype_result(
    ty: &Type,
    module_path: &[String],
    resolve: &dyn Fn(&syn::Path, &[String]) -> Option<Resolved>,
) -> Result<String, String> {
    let container_msg = "function result can only be primitive or pointer";
    let (target, target_module) = resolve_target(ty, module_path, resolve)?;
    match &target {
        Type::Array(_) => return Err(container_msg.into()),
        Type::Path(type_path) if type_path.qself.is_none() => {
            match resolve(&type_path.path, &target_module) {
                Some(Resolved::LocalStruct(_)) => return Err(container_msg.into()),
                Some(Resolved::Extern(p))
                    if p.last()
                        .is_some_and(|last| STRUCT_TAGS.contains(&last.as_str())) =>
                {
                    return Err(container_msg.into());
                }
                _ => {}
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
                // `FixedSizeArrayDefNode.getptrtype` does not use the length.
                Type::Array(arr) => gettype(&arr.elem, &elem_module, resolve)?,
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
            let mut parts = Vec::new();
            for input in &bare.inputs {
                // There are external C functions eating raw structures, not pointers.
                // `LowLevelDatabase.gettype` FuncType: skip `Void` (the resolved type).
                let t = gettype(&input.ty, module_path, resolve)?;
                if t == "void @" {
                    continue;
                }
                parts.push(cdecl(&t, ""));
            }
            let params = if parts.is_empty() {
                "void".to_string()
            } else {
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
                    return Ok(last_segment_c_type(&name));
                }
            }
            match resolve(&type_path.path, module_path) {
                Some(Resolved::Alias(target, alias_module)) => {
                    gettype(&target, &alias_module, resolve)
                }
                Some(Resolved::Int(t)) => Ok(int_template(t.signed, t.bits).to_string()),
                Some(Resolved::Extern(p)) => match p.last() {
                    Some(last) => {
                        let spelled = if p.len() == 3 && p[0] == "majit_rlib" && p[1] == "rffi" {
                            rffi_type_template(last)
                        } else {
                            None
                        };
                        Ok(spelled.map_or_else(|| last_segment_c_type(last), str::to_string))
                    }
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

pub(crate) fn option_of_bare_fn(path: &syn::Path) -> Option<&Type> {
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

/// `LONG_BIT`: the width of `lltype.Signed`, the machine word.
fn long_bit() -> u32 {
    std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
        .ok()
        .and_then(|w| w.parse().ok())
        .unwrap_or(usize::BITS)
}

/// Width of C `long`: 32 on Windows (`CARGO_CFG_TARGET_OS` == "windows", fallback
/// `cfg!(windows)`) or when `long_bit()` is 32; otherwise 64.
fn c_long_bit() -> u32 {
    let windows = match std::env::var("CARGO_CFG_TARGET_OS") {
        Ok(os) => os == "windows",
        Err(_) => cfg!(windows),
    };
    if windows || long_bit() == 32 { 32 } else { 64 }
}

/// `PrimitiveType` for an integer class. `rarithmetic.build_int` and
/// `lltype.build_number` identify an integer type by (sign, bits);
/// `Platform._make_type` and `rffi.setup` build that class; the `LONG_BIT`-wide
/// class is `Signed`/`Unsigned` (`PrimitiveType`: `Signed: 'Signed @'`,
/// `Unsigned: 'size_t @'`), and every other class takes the first
/// `define_c_primitive` registered for it (`if ll_type in PrimitiveName: return`).
fn int_template(signed: bool, bits: u32) -> &'static str {
    if bits == long_bit() {
        if signed { "Signed @" } else { "size_t @" }
    } else {
        match (signed, bits) {
            (true, 8) => "signed char @",
            (false, 8) => "unsigned char @",
            (true, 16) => "short @",
            (false, 16) => "unsigned short @",
            (true, 32) => "int @",
            (false, 32) => "unsigned int @",
            (true, 64) => "long long @",
            (false, 64) => "unsigned long long @",
            _ => panic!("int_template({signed}, {bits})"),
        }
    }
}

/// `rffi.py` `TYPES`/`setup` and `CCHARP`: C type templates for the
/// `majit_rlib::rffi` aliases. Integer names go through `int_template`;
/// `rffi.CHAR` is `lltype.Char` (`PrimitiveType` `Char: 'char @'`).
fn rffi_type_template(name: &str) -> Option<&'static str> {
    Some(match name {
        "CHAR" => "char @",
        "UCHAR" => int_template(false, 8),
        "SHORT" => int_template(true, 16),
        "USHORT" => int_template(false, 16),
        "INT" => int_template(true, 32),
        "UINT" => int_template(false, 32),
        "LONG" => int_template(true, c_long_bit()),
        "ULONG" => int_template(false, c_long_bit()),
        "LONGLONG" => int_template(true, 64),
        "ULONGLONG" => int_template(false, 64),
        "SIZE_T" => int_template(false, long_bit()),
        "SSIZE_T" => int_template(true, long_bit()),
        "SIGNED" => int_template(true, long_bit()),
        "UNSIGNED" => int_template(false, long_bit()),
        "DOUBLE" => "double @",
        "FLOAT" => "float @",
        "VOIDP" => "void *@",
        "CONST_VOIDP" => "const void *@",
        "CCHARP" => "char *@",
        "CONST_CCHARP" => "const char *@",
        "CCHARPP" => "char **@",
        "CWCHARP" => "wchar_t *@",
        "CWCHARPP" => "wchar_t **@",
        _ => return None,
    })
}

/// C type template for a Rust or libc last segment. Integers go through
/// `int_template` (`rarithmetic.build_int` / `lltype.build_number` class);
/// `c_char` is `lltype.Char`.
fn last_segment_c_type(name: &str) -> String {
    match name {
        "i8" | "c_schar" => int_template(true, 8).to_string(),
        "u8" | "c_uchar" => int_template(false, 8).to_string(),
        "i16" | "c_short" => int_template(true, 16).to_string(),
        "u16" | "c_ushort" => int_template(false, 16).to_string(),
        "i32" | "c_int" => int_template(true, 32).to_string(),
        "u32" | "c_uint" => int_template(false, 32).to_string(),
        "i64" | "c_longlong" => int_template(true, 64).to_string(),
        "u64" | "c_ulonglong" => int_template(false, 64).to_string(),
        "isize" | "ssize_t" | "intptr_t" => int_template(true, long_bit()).to_string(),
        "usize" | "size_t" | "uintptr_t" => int_template(false, long_bit()).to_string(),
        "c_long" => int_template(true, c_long_bit()).to_string(),
        "c_ulong" => int_template(false, c_long_bit()).to_string(),
        "c_char" => "char @".into(),
        "f32" | "c_float" => "float @".into(),
        "f64" | "c_double" => "double @".into(),
        "c_void" => "void @".into(),
        tag if STRUCT_TAGS.contains(&tag) => format!("struct {tag} @"),
        other => format!("{other} @"),
    }
}

fn tokens(ty: &Type) -> String {
    quote::quote!(#ty).to_string()
}
