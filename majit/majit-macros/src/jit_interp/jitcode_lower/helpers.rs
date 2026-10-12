use super::*;

// Loop control detection
/// Check if a block contains break or continue at the top level (not nested in inner loops).
pub(super) fn block_has_loop_control(block: &Block) -> bool {
    block.stmts.iter().any(stmt_has_loop_control)
}

pub(super) fn stmt_has_loop_control(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_has_loop_control(expr),
        _ => false,
    }
}

pub(super) fn expr_has_loop_control(expr: &Expr) -> bool {
    match expr {
        Expr::Break(_) | Expr::Continue(_) => true,
        Expr::If(expr_if) => {
            block_has_loop_control(&expr_if.then_branch)
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, e)| expr_has_loop_control(e))
        }
        Expr::Block(block) => block_has_loop_control(&block.block),
        Expr::Match(m) => m.arms.iter().any(|arm| expr_has_loop_control(&arm.body)),
        // Don't recurse into nested loops — they have their own break/continue scope
        Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_) => false,
        _ => false,
    }
}

// Helper functions
/// Extract the get_mut argument from a pool.get_mut(arg) expression.
pub(super) fn extract_stmts(expr: &Expr) -> Vec<Stmt> {
    match expr {
        Expr::Block(block) => block.block.stmts.clone(),
        _ => vec![Stmt::Expr(expr.clone(), None)],
    }
}

/// The integer value of a literal appearing as a dispatch-arm pattern.
///
/// `b'>'` and `'>'` are integer constants that happen to be spelled as
/// characters: a frontend whose opcodes are characters (Brainfuck's `b'>'`,
/// `b'['`, `b']'`) dispatches on exactly the `u8` an `OP_*: u8` constant would
/// produce. Accepting only `Lit::Int` here made every such arm fail extraction,
/// and the caller drops an arm it cannot extract *before* any abort stub is
/// built — so the arm registered nowhere, `degraded_dispatch_arms()` stayed
/// empty, and an interpreter written entirely in byte literals installed a
/// dispatch with no arms at all. Its whole optimized body was the `Finish()`
/// the default label emits.
fn pat_lit_int_value(lit: &Lit) -> Option<i64> {
    match lit {
        Lit::Int(int_lit) => int_lit.base10_parse::<i64>().ok(),
        Lit::Byte(byte_lit) => Some(i64::from(byte_lit.value())),
        // `char` is a Unicode scalar value; go through `u32` so the cast is the
        // code point rather than a sign-extension of anything.
        Lit::Char(char_lit) => Some(i64::from(char_lit.value() as u32)),
        _ => None,
    }
}

/// Extract pattern values as token expressions for use in generated code.
///
/// Unlike `extract_pat_literals`, this accepts constant paths (`Pat::Path`)
/// that cannot be evaluated at proc-macro time. Returns each value as a
/// `TokenStream` expression (`#path as i64` or `#lit as i64`) that is valid
/// in generated Rust code where the constant is in scope.
///
/// pyopcode.py+ if/elif dispatch over opcode constants (e.g. `OP_NOP`,
/// `OP_INC_A`) that are defined as symbolic constants, not inline literals.
pub(super) fn extract_pat_value_tokens(pat: &Pat) -> Option<Vec<TokenStream>> {
    match pat {
        Pat::Lit(expr_lit) => {
            let val = pat_lit_int_value(&expr_lit.lit)?;
            Some(vec![quote! { #val as i64 }])
        }
        Pat::Path(pp) => {
            let path = &pp.path;
            Some(vec![quote! { #path as i64 }])
        }
        // Bare identifier in a match arm (e.g. `OP_NOP`): syn 2 parses
        // unqualified constants the same as binding patterns. Emit
        // `#ident as i64`; the Rust compiler resolves whether it is a
        // constant or a binding at compile time. The caller (`lower_dispatch_chain`)
        // skips `Pat::Wild` and delegates catch-all arms via the default
        // label, so a bare ident reaching this branch is always a constant.
        Pat::Ident(pi) if pi.subpat.is_none() && pi.mutability.is_none() && pi.by_ref.is_none() => {
            let ident = &pi.ident;
            Some(vec![quote! { #ident as i64 }])
        }
        Pat::Or(pat_or) => {
            let mut tokens = Vec::new();
            for case in &pat_or.cases {
                tokens.extend(extract_pat_value_tokens(case)?);
            }
            Some(tokens)
        }
        _ => None,
    }
}

/// True when a match arm's pattern binds a name rather than naming a constant.
///
/// syn parses `OP_NOP => ..` and `other => ..` as the same `Pat::Ident`, and
/// nothing at proc-macro time can resolve which one it is. Rust's own
/// convention decides it: a binding is `snake_case`, so a pattern whose first
/// character is a lower-case ASCII letter is one and everything else names a
/// constant.
///
/// The two mistakes are not symmetric, which is why the ambiguous cases resolve
/// toward "constant". Reading a constant as a binding makes the arm the
/// catch-all: its guard is never emitted and the jitcode computes one arm for
/// every discriminant, while the concrete path — real Rust — stays right, so
/// only a warm-versus-cold answer can see it. Reading a binding as a constant
/// emits `#name as i64` into the generated builder, which does not compile.
///
/// This is the single answer for every arm classifier here: the dispatch
/// chain, `lower_match_stmt` and `lower_match_value` all consult it.
pub(super) fn is_lowercase_binding_pat(pat: &Pat) -> bool {
    let Pat::Ident(pi) = pat else {
        return false;
    };
    if pi.subpat.is_some() || pi.mutability.is_some() || pi.by_ref.is_some() {
        return false;
    }
    pi.ident
        .to_string()
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_lowercase())
}

/// Case emitters for `switch_dispatch`.
///
/// Literal/path/or arms become direct `cases.push((key, label))` statements.
/// Range endpoints are emitted as generated-code expressions because const
/// paths cannot be evaluated inside the proc macro; the user crate resolves
/// them while building the JitCode.
pub(super) fn extract_pat_switch_case_tokens(
    pat: &Pat,
    arm_label: &syn::Ident,
) -> Option<Vec<TokenStream>> {
    match pat {
        Pat::Lit(_) | Pat::Path(_) | Pat::Ident(_) => {
            let values = extract_pat_value_tokens(pat)?;
            Some(
                values
                    .into_iter()
                    .map(|value| quote! { __switch_cases.push(((#value), #arm_label)); })
                    .collect(),
            )
        }
        Pat::Or(pat_or) => {
            let mut tokens = Vec::new();
            for case in &pat_or.cases {
                tokens.extend(extract_pat_switch_case_tokens(case, arm_label)?);
            }
            Some(tokens)
        }
        Pat::Range(range) => {
            let start = range.start.as_ref()?;
            let end = range.end.as_ref()?;
            Some(vec![quote! {
                for __v in ((#start as i64)..=(#end as i64)) {
                    __switch_cases.push((__v, #arm_label));
                }
            }])
        }
        _ => None,
    }
}

pub(super) fn int_arg_regs(bindings: &[Binding]) -> Option<Vec<u16>> {
    bindings
        .iter()
        .map(|binding| match binding.kind {
            BindingKind::Int => Some(binding.reg),
            BindingKind::Ref | BindingKind::Float => None,
        })
        .collect()
}

pub(super) fn inline_int_arg_pairs(bindings: &[Binding]) -> Vec<(u16, u16)> {
    let mut next_idx = 0u16;
    bindings
        .iter()
        .filter_map(|binding| match binding.kind {
            BindingKind::Int => {
                let reg = binding.reg;
                let idx = next_idx;
                next_idx = next_idx.saturating_add(1);
                Some((reg, idx))
            }
            BindingKind::Ref | BindingKind::Float => None,
        })
        .collect()
}

/// Plain `[int]` ptr then len identity slots as `(parent, callee)` pairs.
///
/// `MIFrame.setup_call` / `fill_registers` and `bhimpl_inline_call_ir_v`
/// give the callee only the declared argument boxes. `getarrayitem_raw_i`
/// reads the ptr register of the current JitCode frame, so that register
/// must be a declared int argument. Callee register equals the identity
/// slot the sub-JitCode already reserves (`split_identity_reg_ends`).
pub(super) fn plain_array_identity_int_pairs(config: Option<&LowererConfig>) -> Vec<(u16, u16)> {
    let Some(config) = config else {
        return Vec::new();
    };
    if config.state_arrays.is_empty() {
        return Vec::new();
    }
    let int_base = config.int_identity_base();
    let mut offset = config.plain_int_scalar_count();
    let mut pairs = Vec::new();
    for _ in config.state_array_names_in_order() {
        let ptr = int_base.saturating_add(offset);
        pairs.push((ptr, ptr));
        offset = offset.saturating_add(1);
        let len = int_base.saturating_add(offset);
        pairs.push((len, len));
        offset = offset.saturating_add(1);
    }
    pairs
}

/// `inline_call_grouped` requires a positional I-list (`callee_dst == index`).
/// Array identity slots sit at `int_identity_base + scalar_count + 2*k`, so
/// holes below them are filled with `(k, k)`. Those registers are the packed
/// caller-local prefix or the scalar identity slots, not extra caller-locals
/// (`assert_no_split_identity_alias`).
pub(super) fn extend_int_args_with_array_identity(
    existing: Vec<(u16, u16)>,
    identity: &[(u16, u16)],
) -> Vec<(u16, u16)> {
    if identity.is_empty() {
        return existing;
    }
    let max_callee = existing
        .iter()
        .map(|&(_, callee)| callee)
        .chain(identity.iter().map(|&(_, callee)| callee))
        .max()
        .unwrap_or(0);
    let mut parent_at: Vec<Option<u16>> = vec![None; max_callee as usize + 1];
    for (parent, callee) in existing {
        parent_at[callee as usize] = Some(parent);
    }
    for &(parent, callee) in identity {
        parent_at[callee as usize] = Some(parent);
    }
    parent_at
        .into_iter()
        .enumerate()
        .map(|(i, parent)| (parent.unwrap_or(i as u16), i as u16))
        .collect()
}

pub(super) fn quote_inline_arg_pairs(pairs: &[(u16, u16)]) -> TokenStream {
    let args = pairs.iter().map(|&(parent, callee)| {
        quote! { (#parent as u16, #callee as u16) }
    });
    quote! { &[#(#args),*] }
}

/// Caller registers an inline_call reads, including plain `[int]` identity
/// slots passed as declared int arguments.
pub(super) fn inline_call_read_regs(
    bindings: &[Binding],
    identity_int_pairs: &[(u16, u16)],
) -> Vec<Register> {
    let mut regs: Vec<Register> = bindings.iter().map(Register::from_binding).collect();
    for &(parent, _) in identity_int_pairs {
        let reg = Register::int(parent);
        if !regs.contains(&reg) {
            regs.push(reg);
        }
    }
    regs
}

pub(super) fn inline_ref_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let mut next_idx = 0u16;
    let args = bindings.iter().filter_map(|binding| match binding.kind {
        BindingKind::Ref => {
            let reg = binding.reg;
            let idx = next_idx;
            next_idx = next_idx.saturating_add(1);
            Some(quote! { (#reg, #idx) })
        }
        BindingKind::Int | BindingKind::Float => None,
    });
    quote! { &[#(#args),*] }
}

pub(super) fn inline_float_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let mut next_idx = 0u16;
    let args = bindings.iter().filter_map(|binding| match binding.kind {
        BindingKind::Float => {
            let reg = binding.reg;
            let idx = next_idx;
            next_idx = next_idx.saturating_add(1);
            Some(quote! { (#reg, #idx) })
        }
        BindingKind::Int | BindingKind::Ref => None,
    });
    quote! { &[#(#args),*] }
}

/// Returns `(call_match, post_live)` so the caller can register the
/// `BC_INLINE_CALL` and the trailing `BC_LIVE` marker as two distinct
/// `OpMeta` entries (RPython `jtransform.py:480-481` emits inline_call
/// followed by `-live-`).
///
/// `identity_int_pairs` are `(slot, slot)` for each plain `[int]` ptr/len.
/// `MIFrame.setup_call` plants only declared arguments; a callee
/// `getarrayitem_raw_i` reads the current frame's ptr register.
pub(super) fn inline_call_tokens(
    bindings: &[Binding],
    result_reg: u16,
    identity_int_pairs: &[(u16, u16)],
) -> (TokenStream, TokenStream) {
    let args_i_pairs =
        extend_int_args_with_array_identity(inline_int_arg_pairs(bindings), identity_int_pairs);
    let args_i = quote_inline_arg_pairs(&args_i_pairs);
    let args_r = inline_ref_arg_tokens(bindings);
    let args_f = inline_float_arg_tokens(bindings);
    let has_int_args = !args_i_pairs.is_empty();
    let has_float_args = bindings
        .iter()
        .any(|binding| matches!(binding.kind, BindingKind::Float));

    let call_i = if has_float_args {
        quote! {
            __builder.inline_call_irf_i(
                __sub_idx,
                #args_i,
                #args_r,
                #args_f,
                Some(#result_reg),
            );
        }
    } else if has_int_args {
        quote! {
            __builder.inline_call_ir_i(
                __sub_idx,
                #args_i,
                #args_r,
                Some(#result_reg),
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_i(
                __sub_idx,
                #args_r,
                Some(#result_reg),
            );
        }
    };
    let call_r = if has_float_args {
        quote! {
            __builder.inline_call_irf_r(
                __sub_idx,
                #args_i,
                #args_r,
                #args_f,
                Some(#result_reg),
            );
        }
    } else if has_int_args {
        quote! {
            __builder.inline_call_ir_r(
                __sub_idx,
                #args_i,
                #args_r,
                Some(#result_reg),
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_r(
                __sub_idx,
                #args_r,
                Some(#result_reg),
            );
        }
    };
    let call_f = quote! {
        __builder.inline_call_irf_f(
            __sub_idx,
            #args_i,
            #args_r,
            #args_f,
            Some(#result_reg),
        );
    };

    let call_match = quote! {
        match __sub_return_kind {
            majit_metainterp::JitArgKind::Int => {
                #call_i
            }
            majit_metainterp::JitArgKind::Ref => {
                #call_r
            }
            majit_metainterp::JitArgKind::Float => {
                #call_f
            }
        }
    };
    // RPython jtransform.py:480-481 emits inline_call_* followed
    // immediately by -live-.  Parent-frame resume snapshots rely on
    // BC_INLINE_CALL leaving frame.pc at this post-call LIVE marker
    // before opencoder.py:create_snapshot calls
    // get_list_of_active_boxes(in_a_call=True).
    let post_live = quote! { let _ = __builder.live_placeholder(); };
    (call_match, post_live)
}

pub(super) fn inline_call_tokens_void(
    bindings: &[Binding],
    identity_int_pairs: &[(u16, u16)],
) -> (TokenStream, TokenStream) {
    let args_i_pairs =
        extend_int_args_with_array_identity(inline_int_arg_pairs(bindings), identity_int_pairs);
    let args_i = quote_inline_arg_pairs(&args_i_pairs);
    let args_r = inline_ref_arg_tokens(bindings);
    let args_f = inline_float_arg_tokens(bindings);
    let has_int_args = !args_i_pairs.is_empty();
    let has_float_args = bindings
        .iter()
        .any(|binding| matches!(binding.kind, BindingKind::Float));

    let call = if has_float_args {
        quote! {
            __builder.inline_call_irf_v(
                __sub_idx,
                #args_i,
                #args_r,
                #args_f,
                None,
            );
        }
    } else if has_int_args {
        quote! {
            __builder.inline_call_ir_v(
                __sub_idx,
                #args_i,
                #args_r,
                None,
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_v(
                __sub_idx,
                #args_r,
                None,
            );
        }
    };
    let post_live = quote! { let _ = __builder.live_placeholder(); };
    (call, post_live)
}

pub(super) fn typed_call_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let args = bindings.iter().map(|binding| {
        let reg = binding.reg;
        match binding.kind {
            BindingKind::Int => quote! { majit_metainterp::JitCallArg::int(#reg) },
            BindingKind::Ref => quote! { majit_metainterp::JitCallArg::reference(#reg) },
            BindingKind::Float => quote! { majit_metainterp::JitCallArg::float(#reg) },
        }
    });
    quote! { &[#(#args),*] }
}

pub(super) fn is_supported_int_cast(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            type_path.path.is_ident("i64")
                || type_path.path.is_ident("Val")
                || type_path.path.is_ident("u64")
                || type_path.path.is_ident("isize")
                || type_path.path.is_ident("usize")
                || type_path.path.is_ident("i32")
                || type_path.path.is_ident("u32")
                || type_path.path.is_ident("i16")
                || type_path.path.is_ident("u16")
                || type_path.path.is_ident("i8")
                || type_path.path.is_ident("u8")
                || type_path.path.is_ident("bool")
        }
        _ => false,
    }
}

pub(super) fn is_supported_ref_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("usize"),
        Type::Ptr(_) => true,
        _ => false,
    }
}

/// Whether an `as <ty>` target is a full machine word, i.e. a cast that neither
/// truncates nor extends the 64-bit value. Only these are safe targets for a
/// `<float> as <ty>` cast: Rust saturates at the TARGET type's bounds, which a
/// float->i64 followed by an int narrowing would not reproduce.
pub(super) fn is_word_width_int(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            type_path.path.is_ident("i64")
                || type_path.path.is_ident("u64")
                || type_path.path.is_ident("isize")
                || type_path.path.is_ident("usize")
        }
        _ => false,
    }
}

pub(super) fn type_is_unsigned_int(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            type_path.path.is_ident("u8")
                || type_path.path.is_ident("u16")
                || type_path.path.is_ident("u32")
                || type_path.path.is_ident("u64")
                || type_path.path.is_ident("usize")
        }
        _ => false,
    }
}

/// A primitive integer's `from_le_bytes` / `from_be_bytes` associated function.
///
/// Only the bare spelling (`u16::from_le_bytes`) and the explicit primitive
/// paths (`core::primitive::u16::…`, `std::primitive::u16::…`) match. A
/// namespaced user type whose last segment is a primitive name (`wire::u16::…`)
/// is declined.
pub(super) struct PrimitiveFromEndianBytes {
    pub ty_name: &'static str,
    pub little_endian: bool,
    pub unsigned: bool,
    pub byte_width: usize,
    pub acc_ty_name: &'static str,
}

/// Path-match a primitive `from_le_bytes` / `from_be_bytes` associated function.
pub(super) fn primitive_from_endian_bytes_path(func: &Expr) -> Option<PrimitiveFromEndianBytes> {
    let segments = canonical_expr_segments(func)?;
    let n_seg = segments.len();
    let ty_idx = match n_seg {
        2 => 0,
        4 if matches!(
            (segments[0].as_str(), segments[1].as_str()),
            ("core", "primitive") | ("std", "primitive")
        ) =>
        {
            2
        }
        _ => return None,
    };
    let little_endian = match segments[n_seg - 1].as_str() {
        "from_le_bytes" => true,
        "from_be_bytes" => false,
        _ => return None,
    };
    let (ty_name, unsigned, byte_width, acc_ty_name) = match segments[ty_idx].as_str() {
        "i16" => ("i16", false, 2usize, "u16"),
        "u16" => ("u16", true, 2, "u16"),
        "i32" => ("i32", false, 4, "u32"),
        "u32" => ("u32", true, 4, "u32"),
        "i64" => ("i64", false, 8, "u64"),
        "u64" => ("u64", true, 8, "u64"),
        // `isize`/`usize` are deliberately absent: their width belongs to
        // the TARGET, and this expansion runs on the host, so the host's
        // `size_of` would be the wrong number to check the element count
        // against on any cross build.
        _ => return None,
    };
    Some(PrimitiveFromEndianBytes {
        ty_name,
        little_endian,
        unsigned,
        byte_width,
        acc_ty_name,
    })
}

/// Whether a source expression is already an unsigned integer value.
/// Covers `x as u64` / `(x as u32)` so `as f64` can pick
/// `cast_uint_to_float` the way `rewrite_op_force_cast` uses
/// `rffi.size_and_sign`. A primitive unsigned `from_le_bytes` /
/// `from_be_bytes` call is unsigned too: signedness is read off the
/// original expression, one level above the desugared `((chain) as u64)`.
pub(super) fn expr_is_unsigned_int(expr: &Expr) -> bool {
    match expr {
        Expr::Paren(paren) => expr_is_unsigned_int(&paren.expr),
        Expr::Group(group) => expr_is_unsigned_int(&group.expr),
        Expr::Cast(cast) => type_is_unsigned_int(&cast.ty),
        Expr::Call(call) => {
            primitive_from_endian_bytes_path(&call.func).is_some_and(|info| info.unsigned)
        }
        _ => false,
    }
}

pub(super) fn type_is_raw_pointer(ty: &Type) -> bool {
    matches!(ty, Type::Ptr(_))
}

/// A call whose every argument is a literal. Its address is fixed when the
/// jitcode is built (`jtransform.py` constant pointer, `rewrite_op_cast_pointer`).
pub(super) fn expr_is_literal_call(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    call.args.iter().all(|arg| matches!(arg, Expr::Lit(_)))
}

/// `jtransform.py` `_rewrite_equality`'s `not arg.value` test for a
/// pointer: the operand is still a source expression, so the null
/// spellings RPython would have as `Constant(nullptr)` are recognised
/// here. `core::ptr::null` / `std::ptr::null_mut` / `ptr::null`
/// (optional turbofish) and `0 as *mut T` / `0 as *const T`.
pub(super) fn expr_is_null_ptr(expr: &Expr) -> bool {
    match expr {
        Expr::Paren(paren) => expr_is_null_ptr(&paren.expr),
        Expr::Group(group) => expr_is_null_ptr(&group.expr),
        Expr::Call(call) => call_is_null_ptr(call),
        Expr::Cast(cast) => {
            int_literal_value(&cast.expr) == Some(0) && type_is_raw_pointer(&cast.ty)
        }
        _ => false,
    }
}

/// `core::ptr::null` / `null_mut`, `std::ptr::null` / `null_mut`, and
/// `ptr::null` / `null_mut`. Any other path is an ordinary call.
/// Optional turbofish, no arguments.
pub(crate) fn call_is_null_ptr(call: &syn::ExprCall) -> bool {
    if !call.args.is_empty() {
        return false;
    }
    let Expr::Path(path) = &*call.func else {
        return false;
    };
    let owned: Vec<String> = path
        .path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    let segments: Vec<&str> = owned.iter().map(String::as_str).collect();
    matches!(
        segments.as_slice(),
        ["core", "ptr", "null"]
            | ["core", "ptr", "null_mut"]
            | ["std", "ptr", "null"]
            | ["std", "ptr", "null_mut"]
            | ["ptr", "null"]
            | ["ptr", "null_mut"]
    )
}

/// `x.is_null()` — the Rust spelling of RPython `ptr_iszero`.
pub(super) fn expr_is_ptr_is_null_method(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Paren(paren) => expr_is_ptr_is_null_method(&paren.expr),
        Expr::Group(group) => expr_is_ptr_is_null_method(&group.expr),
        Expr::MethodCall(call) if call.method == "is_null" && call.args.is_empty() => {
            Some(&call.receiver)
        }
        _ => None,
    }
}

pub(super) fn is_supported_float_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("f64"),
        _ => false,
    }
}

pub(crate) fn classify_param_type(ty: &Type) -> Option<InlineReturnKind> {
    // `usize`/pointer params are Ref in the inline-helper ABI. `usize` also
    // satisfies `is_supported_int_cast` (for `as` casts), so the Ref check
    // must precede the Int check or `usize` would misclassify as Int.
    if is_supported_ref_type(ty) {
        Some(InlineReturnKind::Ref)
    } else if is_supported_int_cast(ty) {
        Some(InlineReturnKind::Int)
    } else if is_supported_float_type(ty) {
        Some(InlineReturnKind::Float)
    } else {
        None
    }
}

pub(super) fn extract_bool_branch_values(expr_if: &ExprIf) -> Option<(i64, i64)> {
    let then_value = extract_block_tail_int(&expr_if.then_branch)?;
    let (_, else_expr) = expr_if.else_branch.as_ref()?;
    let else_value = extract_branch_int(else_expr)?;
    Some((then_value, else_value))
}

pub(super) fn extract_block_tail_int(block: &Block) -> Option<i64> {
    match block.stmts.as_slice() {
        [Stmt::Expr(expr, None)] => extract_branch_int(expr),
        _ => None,
    }
}

pub(super) fn extract_branch_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(int_lit),
            ..
        }) => int_lit.base10_parse::<i64>().ok(),
        Expr::Paren(ExprParen { expr, .. }) => extract_branch_int(expr),
        Expr::Block(block) => extract_block_tail_int(&block.block),
        _ => None,
    }
}

/// The memoizing entry point `#[jit_inline]` generates beside `_with_asm`.
///
/// `_with_asm` assembles unconditionally; this one is `CallControl.get_jitcode`
/// — it consults the driver-shared cache first and registers the helper's
/// identity before assembling, so a helper that calls itself resolves to the
/// jitcode it is already being built under instead of recursing.
pub(super) fn inline_shared_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}_shared", last.ident);
    Some(path)
}

/// The callee's return kind, as a token.
///
/// It used to be read back off the assembled jitcode's trailing return opcode.
/// A recursive call has no assembled jitcode to read — the callee is the caller,
/// still under construction — and the kind is a static property of the emit site
/// anyway, so it is spelled rather than recovered.
pub(super) fn jit_arg_kind_tokens(kind: BindingKind) -> TokenStream {
    match kind {
        BindingKind::Int => quote! { majit_metainterp::JitArgKind::Int },
        BindingKind::Ref => quote! { majit_metainterp::JitArgKind::Ref },
        BindingKind::Float => quote! { majit_metainterp::JitArgKind::Float },
    }
}

/// Construct the path of the per-helper liveness prebuild fn that
/// `#[jit_inline]` generates alongside `_with_asm`. The parent
/// `#[jit_interp]` calls this from its
/// `__prebuild_jitcode_liveness_*` so the helper's per-marker
/// triples land in `asm.all_liveness` before
/// `metainterp_sd.liveness_info` snapshot, matching RPython
/// `pyjitpl.py finish_setup` order.
pub(super) fn inline_prebuild_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}_prebuild", last.ident);
    Some(path)
}

pub(super) fn binding_kind_for_inline_policy(
    kind: crate::jit_interp::CallPolicyKind,
) -> Option<BindingKind> {
    match kind {
        crate::jit_interp::CallPolicyKind::InlineInt
        | crate::jit_interp::CallPolicyKind::InlinePipelineInt => Some(BindingKind::Int),
        crate::jit_interp::CallPolicyKind::InlineRef
        | crate::jit_interp::CallPolicyKind::InlinePipelineRef => Some(BindingKind::Ref),
        crate::jit_interp::CallPolicyKind::InlineFloat
        | crate::jit_interp::CallPolicyKind::InlinePipelineFloat => Some(BindingKind::Float),
        _ => None,
    }
}

pub(crate) fn helper_policy_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_call_policy_{}", last.ident);
    Some(path)
}

/// Emit the int-binop recording call for `dst = lhs <op> rhs`.
///
/// Rust `/` and `%` are C-truncating, so they record `support.py`
/// `_ll_2_int_floordiv` / `_ll_2_int_mod`. Those two are in
/// `inline_calls_to`: the assembler emits the inlined body (`int.py_div`
/// / `int.py_mod` plus the truncation adjustment), not a residual call.
/// `int.py_div` / `int.py_mod` alone are `_handle_int_special` for
/// Python-floor `//` and `%`.
pub(super) fn binop_i_emit_tokens(
    dst: u16,
    opcode: &Ident,
    lhs: u16,
    rhs: u16,
) -> proc_macro2::TokenStream {
    match opcode.to_string().as_str() {
        "IntFloorDiv" => quote! { __builder.record_int_floordiv(#dst, #lhs, #rhs); },
        "IntMod" => quote! { __builder.record_int_mod(#dst, #lhs, #rhs); },
        _ => quote! { __builder.record_binop_i(#dst, majit_ir::OpCode::#opcode, #lhs, #rhs); },
    }
}

/// [`binop_i_emit_tokens`] whose right operand is a `Constant`, which
/// `assembler.py` `emit_const` encodes in the operation itself. `None` for
/// `/` and `%`: their inlined `support.py` bodies read the divisor from a
/// register.
pub(super) fn binop_i_const_emit_tokens(
    dst: u16,
    opcode: &Ident,
    lhs: u16,
    rhs: &proc_macro2::TokenStream,
) -> Option<proc_macro2::TokenStream> {
    binop_i_accepts_constant(opcode).then(
        || quote! { __builder.record_binop_i_const(#dst, majit_ir::OpCode::#opcode, #lhs, #rhs); },
    )
}

/// Whether [`binop_i_const_emit_tokens`] can encode `opcode` with a
/// `Constant` right operand.
pub(super) fn binop_i_accepts_constant(opcode: &Ident) -> bool {
    !matches!(opcode.to_string().as_str(), "IntFloorDiv" | "IntMod")
}

pub(super) fn binop_f_emit_tokens(
    dst: u16,
    opcode: &Ident,
    lhs: u16,
    rhs: u16,
) -> proc_macro2::TokenStream {
    quote! { __builder.record_binop_f(#dst, majit_ir::OpCode::#opcode, #lhs, #rhs); }
}

/// jtransform.py `_rewrite_symmetric`'s binding list, in this layer's operator
/// domain: the symmetric arithmetic and bitwise operators plus every
/// comparison (upstream binds `int_lt` .. `uint_ge` and `float_lt` ..
/// `float_ge`, and reaches the same function again as `_rewrite_equality`'s
/// fallthrough for `int_eq` / `int_ne` / `ptr_eq` / `ptr_ne`).  `-`, the
/// divisions and the shifts are absent for the same reason they are absent
/// upstream: they are not symmetric.
pub(super) fn binop_is_symmetric(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add(_)
            | BinOp::Mul(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::BitXor(_)
            | BinOp::And(_)
            | BinOp::Or(_)
            | BinOp::Eq(_)
            | BinOp::Ne(_)
            | BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Gt(_)
            | BinOp::Ge(_)
    )
}

/// jtransform.py `_rewrite_symmetric`'s `reversename` table: swapping the
/// operands of an ordered comparison mirrors it.  Everything else keeps its
/// operator, which is upstream's `.get(op.opname, op.opname)` default.
pub(super) fn mirrored_compare_binop(op: &BinOp) -> BinOp {
    match op {
        BinOp::Lt(_) => BinOp::Gt(Default::default()),
        BinOp::Le(_) => BinOp::Ge(Default::default()),
        BinOp::Gt(_) => BinOp::Lt(Default::default()),
        BinOp::Ge(_) => BinOp::Le(Default::default()),
        other => *other,
    }
}

/// `isinstance(arg, Constant)` plus the `arg.value` `_rewrite_equality` tests,
/// answered at the layer this lowering runs in: the operand is still a source
/// literal, not yet a flow-graph `Constant`.  A `bool` literal counts because
/// its lowered register holds `0` / `1`, which is the same `lltype.Bool` the
/// upstream test sees.
pub(super) fn int_literal_value(expr: &syn::Expr) -> Option<i64> {
    match expr {
        syn::Expr::Group(group) => int_literal_value(&group.expr),
        syn::Expr::Paren(paren) => int_literal_value(&paren.expr),
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(value) => value.base10_parse::<i64>().ok(),
            syn::Lit::Bool(value) => Some(i64::from(value.value)),
            _ => None,
        },
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            int_literal_value(&unary.expr).and_then(i64::checked_neg)
        }
        _ => None,
    }
}

pub(super) fn opcode_for_binop(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::Add(_) => "IntAdd",
        BinOp::Sub(_) => "IntSub",
        BinOp::Mul(_) => "IntMul",
        BinOp::Div(_) => "IntFloorDiv",
        BinOp::Rem(_) => "IntMod",
        BinOp::BitAnd(_) => "IntAnd",
        BinOp::BitOr(_) => "IntOr",
        BinOp::BitXor(_) => "IntXor",
        // Logical `&&`/`||` on boolean operands (Rust requires `bool`, whose
        // lowered register holds 0/1) is bit-for-bit the same as `&`/`|`.
        // M4 audit: every macro-lowered `&&`/`||` operand is pure/safe to
        // evaluate unconditionally; no audited `#[jit_interp]` body depends on
        // short-circuiting (only closure-literal `|| {}` tokens appear in the
        // current annotated bodies).  A green operand folds so only the taken
        // branch survives.
        BinOp::And(_) => "IntAnd",
        BinOp::Or(_) => "IntOr",
        BinOp::Shl(_) => "IntLshift",
        BinOp::Shr(_) => "IntRshift",
        BinOp::Eq(_) => "IntEq",
        BinOp::Ne(_) => "IntNe",
        BinOp::Lt(_) => "IntLt",
        BinOp::Le(_) => "IntLe",
        BinOp::Gt(_) => "IntGt",
        BinOp::Ge(_) => "IntGe",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

pub(super) fn opcode_for_binop_f(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::Add(_) => "FloatAdd",
        BinOp::Sub(_) => "FloatSub",
        BinOp::Mul(_) => "FloatMul",
        BinOp::Div(_) => "FloatTrueDiv",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

/// Float comparison operators. Both operands are float; the result is an
/// int (`0`/`1`) recorded via `record_compare_f` (`float_lt/ff>i` etc.).
pub(super) fn opcode_for_compare_f(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::Lt(_) => "FloatLt",
        BinOp::Le(_) => "FloatLe",
        BinOp::Eq(_) => "FloatEq",
        BinOp::Ne(_) => "FloatNe",
        BinOp::Gt(_) => "FloatGt",
        BinOp::Ge(_) => "FloatGe",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

pub(super) fn opcode_for_assign_binop(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::AddAssign(_) => "IntAdd",
        BinOp::SubAssign(_) => "IntSub",
        BinOp::MulAssign(_) => "IntMul",
        BinOp::DivAssign(_) => "IntFloorDiv",
        BinOp::RemAssign(_) => "IntMod",
        BinOp::BitAndAssign(_) => "IntAnd",
        BinOp::BitOrAssign(_) => "IntOr",
        BinOp::BitXorAssign(_) => "IntXor",
        BinOp::ShlAssign(_) => "IntLshift",
        BinOp::ShrAssign(_) => "IntRshift",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

pub(super) fn opcode_for_assign_binop_f(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::AddAssign(_) => "FloatAdd",
        BinOp::SubAssign(_) => "FloatSub",
        BinOp::MulAssign(_) => "FloatMul",
        BinOp::DivAssign(_) => "FloatTrueDiv",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

/// Reject a raw Ref residual target whose return is not one pointer word.
///
/// `CallBuilderX86.load_result`, `CallBuilder64.load_result`,
/// `CallBuilderARM64.load_result`, and `llmodel.py bh_call_r` read that
/// word. `Option<*mut T>` is two words — a raw pointer has no niche — and
/// fails this assert. The closure is monomorphized so the check runs, and
/// it is not called.
pub(super) fn assert_ref_result_one_word(func: &impl ToTokens, arity: usize) -> TokenStream {
    let args: Vec<TokenStream> = (0..arity)
        .map(|_| quote! { __majit_ref_word_arg() })
        .collect();
    quote! {
        {
            fn __majit_ref_word_arg<T>() -> T {
                unsafe { ::core::mem::MaybeUninit::<T>::uninit().assume_init() }
            }
            fn __majit_assert_ref_result_one_word<R, F>(_: F)
            where
                F: FnOnce() -> R,
            {
                const {
                    assert!(
                        ::core::mem::size_of::<R>() == ::core::mem::size_of::<usize>(),
                        "a Ref residual call target must return one pointer word"
                    );
                }
            }
            __majit_assert_ref_result_one_word(|| {
                #[allow(unused_unsafe)]
                unsafe { #func(#(#args),*) }
            });
        }
    }
}

/// The address of a shim that returns `func`'s result as a machine word.
///
/// Registered addresses are dispatched through
/// `majit_backend::call_stub::bh_call_i_dispatch`, which transmutes them to
/// `extern "C" fn(..) -> i64` and reads the whole return register. A callee
/// whose result is narrower than a word leaves the bits above it undefined —
/// see [`majit_ir::CallResultWord`] for what that costs — so the register is
/// not readable as an `i64` unless the callee is known to fill it. Registering
/// a shim rather than the callee itself makes that true of every target.
///
/// The shim's parameters are the machine words a compiled call passes, not the
/// types `func` declares: a non-float argument goes in spelled `i64` and is
/// narrowed at the boundary by [`majit_ir::CallArgWord`], whose target type
/// inference reads off `func`'s own signature. Where a pointer is as wide as a
/// word that spelling is the same function it always was; where it is narrower
/// it is the difference between a call and a type error, since a `usize` or a
/// `&T` parameter is then not the word the call holds. A float argument keeps
/// its hole: it rides a float register, and a word is not what the call passes
/// for it. `arg_kinds` is one entry per parameter, the receiver of a method
/// path included.
///
/// `extern`-free is also all this can be. A non-capturing closure coerces to
/// `fn(..)` and not to `extern "C" fn(..)`, and an `extern "C"` fn item has to
/// spell parameter types that only inference knows here — so the shim carries
/// the Rust ABI, as the callee it stands in for does. The C ABI that both the
/// dispatch and the compiled call name is answered where a helper's signature
/// is in scope instead: `#[elidable]` and its siblings emit an
/// `extern "C" fn(..) -> i64` call-target wrapper and register that. A
/// `calls = { .. }` entry naming a function that carries no such attribute has
/// no wrapper to reach for.
pub(super) fn word_result_addr_tokens(
    func: &impl ToTokens,
    arg_kinds: &[BindingKind],
) -> TokenStream {
    let params: Vec<Ident> = (0..arg_kinds.len())
        .map(|i| format_ident!("__majit_a{i}"))
        .collect();
    let holes: Vec<TokenStream> = arg_kinds
        .iter()
        .map(|kind| match kind {
            BindingKind::Float => quote! { _ },
            _ => quote! { i64 },
        })
        .collect();
    let args: Vec<TokenStream> = arg_kinds
        .iter()
        .zip(&params)
        .map(|(kind, param)| match kind {
            BindingKind::Float => quote! { #param },
            _ => quote! { unsafe { majit_ir::CallArgWord::from_call_word(#param) } },
        })
        .collect();
    quote! {
        {
            let __majit_word_shim: fn(#(#holes),*) -> i64 = |#(#params),*| {
                majit_ir::CallResultWord::into_call_word(#func(#(#args),*))
            };
            __majit_word_shim as *const ()
        }
    }
}

/// [`word_result_addr_tokens`] for a target whose result kind is `kind`, and
/// `None` for a kind that has no narrow result to widen: a ref result is a
/// pointer, which already fills the register, and a void result is never read.
pub(super) fn word_result_addr_for_kind(
    kind: BindingKind,
    func: &impl ToTokens,
    arg_kinds: &[BindingKind],
) -> Option<TokenStream> {
    matches!(kind, BindingKind::Int).then(|| word_result_addr_tokens(func, arg_kinds))
}

/// [`word_result_addr_tokens`] for a target the trace records with a `()`
/// result.
///
/// Void needs its own because what `word_result_addr_tokens` widens is the
/// *result*, and there is none here — but the parameters still have to be the
/// machine words a call carries them in. A target declaring a pointer
/// parameter is spelled narrower than that wherever a pointer is narrower than
/// a word, and an indirect call that type-checks its callee will not accept it;
/// the shim is the function that call names instead.
///
/// Emitted only where the two spellings can differ. Where a pointer already
/// fills a word the target IS its own shim, and standing one in front of it
/// would add a frame to every call and hand every consumer keyed on a callee's
/// address a stand-in instead.
pub(super) fn word_void_addr_tokens(
    func: &impl ToTokens,
    arg_kinds: &[BindingKind],
) -> TokenStream {
    let params: Vec<Ident> = (0..arg_kinds.len())
        .map(|i| format_ident!("__majit_a{i}"))
        .collect();
    let holes: Vec<TokenStream> = arg_kinds
        .iter()
        .map(|kind| match kind {
            BindingKind::Float => quote! { _ },
            _ => quote! { i64 },
        })
        .collect();
    let args: Vec<TokenStream> = arg_kinds
        .iter()
        .zip(&params)
        .map(|(kind, param)| match kind {
            BindingKind::Float => quote! { #param },
            _ => quote! { unsafe { majit_ir::CallArgWord::from_call_word(#param) } },
        })
        .collect();
    quote! {
        {
            #[cfg(target_arch = "wasm32")]
            let __majit_void_target = {
                let __majit_word_shim: fn(#(#holes),*) = |#(#params),*| {
                    #func(#(#args),*);
                };
                __majit_word_shim as *const ()
            };
            #[cfg(not(target_arch = "wasm32"))]
            let __majit_void_target = #func as *const ();
            __majit_void_target
        }
    }
}
