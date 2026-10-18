use super::*;

pub struct GeneratedJitCodeBody {
    pub body: TokenStream,
    pub liveness_prebuild: TokenStream,
    /// Slice (audit Issue #5) — green schema in declaration order,
    /// each pair is `(name, green_type_token)` where the token
    /// resolves to a `majit_ir::GreenType` variant at the install
    /// site (Int / Ref / Float / Void / Str / Unicode).  The
    /// dispatch path passes this to
    /// `JitDriver::declare_schema_typed` so
    /// `JitDriverStaticData::green_args_spec` reports STR/UNICODE
    /// where the user tagged `: str` / `: unicode` instead of
    /// collapsing to `GreenType::Ref`.  Per-arm bodies leave it
    /// empty.
    pub green_schema: Vec<(String, TokenStream)>,
    /// Red schema — `(name, ir_type_token)` resolving to a
    /// `majit_ir::Type` (Int / Ref / Float / Void).  Reds carry no
    /// upstream lltype subtype distinction (no `equal_whatever`
    /// dispatch on runtime args).
    pub red_schema: Vec<(String, TokenStream)>,
}

/// Failure from a per-arm sub-JitCode body (`split_dispatch` / infer /
/// `inline_pipeline_*`). A degrade is the historical abort-stub path; an
/// identity-slot access must fail expansion instead.
#[derive(Debug)]
pub(crate) enum SubJitcodeBodyError {
    Degrade(String),
    /// The arm would emit `load_state_field*` / `store_state_field*` for
    /// this state field. `MIFrame.setup_call` / `bhimpl_inline_call_*`
    /// copy declared args only, so a callee cannot reach portal identity
    /// reds.
    IdentitySlot {
        field: String,
        span: proc_macro2::Span,
    },
}

impl SubJitcodeBodyError {
    /// Expansion diagnostic for an identity-slot access inside a
    /// sub-JitCode arm. `warmspot.py` `make_virtualizable_infos` has one
    /// virtualizable per jitdriver, so leftover state scalars stay portal
    /// reds.
    pub(crate) fn identity_slot_message(arm: &str, field: &str) -> String {
        format!(
            "sub-JitCode arm `{arm}` emits load/store_state_field* for state field `{field}`; \
             a sub-JitCode cannot reach identity-slot state reds \
             (`MIFrame.setup_call` copies declared args only; `bhimpl_inline_call_*`). \
             With `virtualizable_fields` declared the state struct is not the virtualizable \
             (`warmspot.py` `make_virtualizable_infos` has one virtualizable per jitdriver), \
             so `{field}` must be a field of the declared virtualizable object \
             (or the arm must not be split)"
        )
    }
}

fn sub_jitcode_lower_error(lowerer: &mut Lowerer<'_>) -> SubJitcodeBodyError {
    SubJitcodeBodyError::Degrade(lowerer.take_body_failure_reason())
}

#[allow(dead_code)]
pub fn try_generate_jitcode_body(body: &Expr) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, None).map(|p| p.body)
}

#[allow(dead_code)]
pub fn try_generate_jitcode_body_with_config(
    config: &LowererConfig,
    body: &Expr,
) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, Some(config)).map(|p| p.body)
}

/// Per-caller-local layout descriptor produced by
/// [`try_generate_jitcode_body_parts_with_caller_bindings`].
///
/// The dispatch arm parent emit needs three things to construct the
/// `inline_call_<types>_v(__sub_idx, args_i, args_r, args_f)` call:
/// - the parent's reg (read from caller's `Binding`, or a portal identity
///   slot for a packed plain-array `{ptr, len}` / vable ref),
/// - the callee's packed argument register (assigned per-bank by the
///   sub-Lowerer pre-bind pass),
/// - the bank (Int / Ref / Float) so the (parent, callee) pair lands in
///   the matching `args_<kind>` vector.
#[allow(private_interfaces)]
#[derive(Clone, Debug)]
pub(crate) struct CallerLocalLayout {
    #[allow(dead_code)]
    pub name: String,
    pub parent_reg: u16,
    pub callee_reg: u16,
    pub kind: BindingKind,
}

/// Dispatch arm caller-local plumbing.
///
/// Dispatch-arm lowering entry that pre-binds a list of caller-locals as
/// portal-input bindings on the sub-Lowerer before lowering the body.
/// The caller (dispatch arm emit at `lower_dispatch_chain`)
/// collects them via
/// [`collect_arm_caller_locals`] and threads the same list through
/// `inline_call_<types>_v` as `(parent_reg, callee_reg)` pairs.
///
/// Layout convention (mirrors the portal-input pre-bind in
/// `lower_dispatch_body`):
/// - per-bank packed regs starting at 0 (first Int → int_reg 0, second
///   Int → int_reg 1, …; same for Ref and Float independently);
/// - flat `next_reg` counter advanced past the highest per-bank slot
///   so subsequent `alloc_reg()` calls inside the body lowering do not
///   collide with the pre-bound caller-locals.
///
/// `pyopcode.py` / `jtransform.py` `handle_regular_call`: PyPy's dispatch
/// inline_call passes `(opcode, oparg, ...)` as call args; the callee
/// jitcode receives them via its own argument registers
/// (`MIFrame.setup_call` / `bhimpl_inline_call_ir_v`). Pyre's sub-frame
/// uses the same `int_regs[]` / `ref_regs[]` arrays per kind, so per-bank
/// packing is the orthodox layout.
/// Layout-side helper extracted from
/// [`try_generate_jitcode_body_parts_with_caller_bindings`] so the
/// per-bank packing rule is testable without needing a body that
/// lowers cleanly.  Returns the layout descriptors plus the
/// worst-case `next_reg` advance that the caller must apply so
/// subsequent `alloc_reg()` cannot collide.
#[allow(private_interfaces)]
pub(crate) fn assign_caller_local_layout(
    caller_locals: &[(String, Binding)],
) -> (Vec<CallerLocalLayout>, u16) {
    let mut next_int = 0u16;
    let mut next_ref = 0u16;
    let mut next_float = 0u16;
    let mut layout = Vec::with_capacity(caller_locals.len());
    for (name, parent_binding) in caller_locals {
        let callee_reg = match parent_binding.kind {
            BindingKind::Int => {
                let r = next_int;
                next_int = next_int.saturating_add(1);
                r
            }
            BindingKind::Ref => {
                let r = next_ref;
                next_ref = next_ref.saturating_add(1);
                r
            }
            BindingKind::Float => {
                let r = next_float;
                next_float = next_float.saturating_add(1);
                r
            }
        };
        layout.push(CallerLocalLayout {
            name: name.clone(),
            parent_reg: parent_binding.reg,
            callee_reg,
            kind: parent_binding.kind,
        });
    }
    let max_pre_bound = next_int.max(next_ref).max(next_float);
    (layout, max_pre_bound)
}

fn layout_bank_end(layout: &[CallerLocalLayout], kind: BindingKind) -> u16 {
    layout
        .iter()
        .filter(|entry| entry.kind == kind)
        .map(|entry| entry.callee_reg.saturating_add(1))
        .max()
        .unwrap_or(0)
}

/// Append plain-array `{ptr, len}` words and the state virtualizable ref as
/// ordinary declared arguments, packed in the same per-bank numbering
/// [`assign_caller_local_layout`] uses.
///
/// `array_names` are the plain `[int]`/`[float]` fields whose
/// `state.<name>[...]` the arm body reads or writes. Parent registers stay
/// the portal identity slots (`int_identity_base` / `vable_input_ref_reg`);
/// callee registers continue densely after the caller-locals.
/// `Transformer.handle_regular_call` / `MIFrame.setup_call` copy only
/// those declared boxes into the callee's own argument registers.
fn extend_layout_with_callee_declared_args(
    layout: &mut Vec<CallerLocalLayout>,
    config: Option<&LowererConfig>,
    array_names: &[&str],
) {
    let Some(config) = config else {
        return;
    };
    let next_int = layout_bank_end(layout, BindingKind::Int);
    let array_pairs = plain_array_identity_int_pairs(Some(config), next_int, array_names);
    let mut pair_idx = 0usize;
    for name in array_names {
        let (ptr_parent, ptr_callee) = array_pairs[pair_idx];
        pair_idx += 1;
        let (len_parent, len_callee) = array_pairs[pair_idx];
        pair_idx += 1;
        layout.push(CallerLocalLayout {
            name: plain_array_ptr_binding_name(name),
            parent_reg: ptr_parent,
            callee_reg: ptr_callee,
            kind: BindingKind::Int,
        });
        layout.push(CallerLocalLayout {
            name: plain_array_len_binding_name(name),
            parent_reg: len_parent,
            callee_reg: len_callee,
            kind: BindingKind::Int,
        });
    }
    let (Some(vable_var), Some(parent_reg)) =
        (config.vable_var.as_ref(), config.vable_input_ref_reg)
    else {
        return;
    };
    if layout.iter().any(|entry| entry.name == *vable_var) {
        return;
    }
    layout.push(CallerLocalLayout {
        name: vable_var.clone(),
        parent_reg,
        callee_reg: layout_bank_end(layout, BindingKind::Ref),
        kind: BindingKind::Ref,
    });
}

/// Pre-bind caller-locals plus packed array `{ptr, len}` / vable-ref
/// arguments on a sub-JitCode lowerer. Array `{ptr, len}` words are only
/// those whose `state.<name>[...]` `body_stmts` reads or writes.
///
/// `next_reg` is the packed declared-arg floor, not the portal identity
/// range: temps must not alias the callee's own argument registers, and
/// they must not be forced up to `int_identity_base()`.
pub(super) fn bind_callee_declared_args(
    lowerer: &mut Lowerer<'_>,
    caller_locals: &[(String, Binding)],
    config: Option<&LowererConfig>,
    body_stmts: &[Stmt],
) -> Vec<CallerLocalLayout> {
    let used = lowerer.body_used_state_array_names(body_stmts);
    let used_names: Vec<&str> = used.iter().map(String::as_str).collect();
    let (mut layout, _) = assign_caller_local_layout(caller_locals);
    extend_layout_with_callee_declared_args(&mut layout, config, &used_names);
    for entry in &layout {
        lowerer.bindings.insert(
            entry.name.clone(),
            Binding {
                reg: entry.callee_reg,
                kind: entry.kind,
                depends_on_stack: false,
                struct_type: None,
            },
        );
    }
    let max_pre_bound = layout
        .iter()
        .map(|entry| entry.callee_reg.saturating_add(1))
        .max()
        .unwrap_or(0);
    lowerer.next_reg = max_pre_bound;
    layout
}

#[allow(private_interfaces)]
pub(crate) fn try_generate_jitcode_body_parts_with_caller_bindings(
    body: &Expr,
    config: Option<&LowererConfig>,
    caller_locals: &[(String, Binding)],
) -> Result<(GeneratedJitCodeBody, Vec<CallerLocalLayout>), SubJitcodeBodyError> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return Err(SubJitcodeBodyError::Degrade(
            "arm body has no statements to lower".to_string(),
        ));
    }

    let mut lowerer = Lowerer::new(config);
    // Sole dispatch-arm-body lowerer entry — `lower_stmt`'s `Stmt::Macro`
    // recognition for `can_enter_jit!()` emits
    // `__builder.loop_header(__jdindex);` which references the
    // `__jdindex: i64` parameter of the enclosing `__dispatch_jitcode_<fn>`.
    // The remaining public `try_generate_jitcode_body*` helpers are legacy
    // test/API shims and do not set this flag, so they still reject
    // `can_enter_jit!()` bodies instead of emitting a loop header without
    // a driver index.
    lowerer.in_dispatch_arm_body = true;

    // Packed declared args in the callee's own numbering
    // (`MIFrame.setup_call` / `bhimpl_inline_call_ir_v`). Temps start
    // after those registers; they are not reserved up to the portal
    // identity range.
    let layout = bind_callee_declared_args(&mut lowerer, caller_locals, config, &stmts);

    for (i, stmt) in stmts.iter().enumerate() {
        if lowerer.lower_stmt(stmt).is_none() {
            // The body is already doomed. Keep walking the REMAINING statements
            // for their reasons only, discarding whatever they lower, so
            // `reason` enumerates the arm's blockers instead of naming whichever
            // check happened to fire first. Nothing downstream sees the partial
            // state: we return Err below either way.
            //
            // This reports one reason per offending STATEMENT, never two for
            // one statement — each refusal site still returns from `lower_stmt`
            // at its first hit. An arm whose blockers all sit inside a single
            // statement (dualtape's `b']'` is one `if`/`else`) still reports
            // exactly one.
            for rest in &stmts[i + 1..] {
                let _ = lowerer.lower_stmt(rest);
            }
            return Err(sub_jitcode_lower_error(&mut lowerer));
        }
    }
    // Completed body with load/store_state_field*: assembler.rs
    // `assert_sub_jitcode_has_no_identity_slots` would trip at construct
    // time. Fail expansion at the field (`MIFrame.setup_call`).
    if let Some((field, span)) = lowerer.sub_jitcode_identity_slot.take() {
        return Err(SubJitcodeBodyError::IdentitySlot { field, span });
    }

    // flatten.py GraphFlattener.make_return emits void_return for a void
    // return block. Reaching the end of the bytecode is not a frame return.
    lowerer.emit_op(
        OpMeta::terminal(Vec::new()),
        quote! { __builder.void_return(); },
    );
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body_with_caller_bindings", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Ok((
        GeneratedJitCodeBody {
            body: quote! {
                #(#statements)*
            },
            liveness_prebuild,
            green_schema: Vec::new(),
            red_schema: Vec::new(),
        },
        layout,
    ))
}

/// pc-returning variant of
/// [`try_generate_jitcode_body_parts_with_caller_bindings`] for `split_dispatch`
/// pure forward-advancing arms.  The body's straight-line `work` statements are
/// lowered as usual; the trailing `pc += increment` is NOT lowered (on the
/// non-pinned sub-JitCode path it is inert and dropped — the dispatch loop owns
/// the pc register).  Instead, an explicit `BC_INT_RETURN(pc + increment)` is
/// emitted so the paired `inline_call_<types>_i` writes the advanced pc back
/// into the caller's green pc register (`next_instr = self.OPCODE(...)`).
#[allow(private_interfaces)]
pub(crate) fn try_generate_jitcode_pc_return_body_with_caller_bindings(
    body: &Expr,
    config: Option<&LowererConfig>,
    caller_locals: &[(String, Binding)],
    increment: i64,
) -> Result<(GeneratedJitCodeBody, Vec<CallerLocalLayout>), SubJitcodeBodyError> {
    let stmts = extract_stmts(body);
    // The predicate (`arm_is_pure_pc_advance`) guarantees a trailing `pc += N`,
    // which is replaced by the explicit pc-return below.  Lower only the work.
    let Some((_pc_advance, work)) = stmts.split_last() else {
        return Err(SubJitcodeBodyError::Degrade(
            "arm body has no statements to lower".to_string(),
        ));
    };

    let mut lowerer = Lowerer::new(config);
    lowerer.in_dispatch_arm_body = true;

    let layout = bind_callee_declared_args(&mut lowerer, caller_locals, config, work);

    for (i, stmt) in work.iter().enumerate() {
        if lowerer.lower_stmt(stmt).is_none() {
            // Same accumulation as the parts entry above: enumerate the arm's
            // blockers rather than reporting whichever check fired first.
            //
            // `work` is `stmts.split_last()`, so the trailing pc-advance is
            // NOT in this loop and contributes no reason here. That is correct
            // for this entry point — reaching it at all means the advance is
            // the arm's final statement and IS the channel out, which is
            // exactly the case the green-write refusal does not fire on.
            //
            // No crate in the tree reaches this loop: it is selected by
            // `pc_return_increment`, which is `Some` only under
            // `split_dispatch = true`, and a tracked-file sweep finds zero
            // crates setting it (positive control: the same sweep reads 3
            // example files for `jit_interp`). Patched for consistency, not for
            // yield — the measured yield is all on the parts entry above.
            for rest in &work[i + 1..] {
                let _ = lowerer.lower_stmt(rest);
            }
            return Err(sub_jitcode_lower_error(&mut lowerer));
        }
    }
    // Same completed-body check as `try_generate_jitcode_body_parts_with_caller_bindings`.
    if let Some((field, span)) = lowerer.sub_jitcode_identity_slot.take() {
        return Err(SubJitcodeBodyError::IdentitySlot { field, span });
    }

    // `pc` is collected as a caller-local (the trailing `pc += N` references it),
    // so it is pre-bound at its callee reg; the work statements only read it, so
    // the binding still holds the incoming pc.  Return pc + increment.
    let Some(pc_binding) = lowerer.bindings.get("pc") else {
        return Err(SubJitcodeBodyError::Degrade(
            "arm body has no `pc` binding for the pc-return writeback".to_string(),
        ));
    };
    let pc_reg = pc_binding.reg;
    let ret_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(pc_reg)],
            vec![Register::int(ret_reg)],
        ),
        quote! {
            __builder.record_binop_i_const(
                #ret_reg as u16,
                majit_ir::OpCode::IntAdd,
                #pc_reg as u16,
                #increment as i64,
            );
        },
    );
    lowerer.emit_op(
        OpMeta::terminal(vec![Register::int(ret_reg)]),
        quote! { __builder.int_return(#ret_reg as u16); },
    );

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_pc_return_body", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Ok((
        GeneratedJitCodeBody {
            body: quote! {
                #(#statements)*
            },
            liveness_prebuild,
            green_schema: Vec::new(),
            red_schema: Vec::new(),
        },
        layout,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
)]
pub(crate) fn generate_inline_helper_jitcode_with_calls(
    func: &ItemFn,
    calls: &[crate::jit_interp::CallEntry],
    ref_params: &[(Ident, syn::Path)],
    ref_fields: &[crate::jit_interp::RefFieldEntry],
    array_fields: &[crate::jit_interp::ArrayFieldEntry],
    int_fields: &[crate::jit_interp::IntFieldEntry],
    native_int_binops: &[(Path, Ident)],
    native_tag_small: &[Path],
    native_identity: &[Path],
    headerless_structs: &[Path],
    inlined_prefix: &[crate::jit_interp::InlinedPrefixEntry],
) -> syn::Result<Option<InlineHelperJitCode>> {
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "#[jit_inline] does not support generic helper functions yet",
        ));
    }

    let return_kind = match &func.sig.output {
        ReturnType::Default => InlineReturnKind::Void,
        ReturnType::Type(_, return_ty) => classify_param_type(return_ty).ok_or_else(|| {
            syn::Error::new_spanned(
                return_ty,
                "#[jit_inline] supports i64/isize (Int), usize/pointer (Ref), f64 (Float), or void return types",
            )
        })?,
    };

    let call_policies = calls
        .iter()
        .map(|entry| {
            let spec = match entry.policy {
                Some(kind) => CallPolicySpec::Explicit(kind),
                None => CallPolicySpec::Infer,
            };
            (canonical_path_segments(&entry.path), spec)
        })
        .collect();
    let ref_param_structs: HashMap<String, syn::Path> = ref_params
        .iter()
        .map(|(name, struct_type)| (name.to_string(), struct_type.clone()))
        .collect();
    let inline_config = if ref_param_structs.is_empty()
        && ref_fields.is_empty()
        && array_fields.is_empty()
        && int_fields.is_empty()
        && native_int_binops.is_empty()
        && native_tag_small.is_empty()
        && native_identity.is_empty()
        && headerless_structs.is_empty()
        && inlined_prefix.is_empty()
    {
        None
    } else {
        Some(LowererConfig::inline_helper(
            ref_fields,
            array_fields,
            int_fields,
            native_int_binops,
            native_tag_small,
            native_identity,
            headerless_structs,
            inlined_prefix,
        ))
    };
    let mut lowerer = Lowerer::new_with_call_policies(
        inline_config.as_ref(),
        call_policies,
        InferenceFailureMode::Panic,
    );
    let param_layout = inline_helper_param_layout(func)?;
    let mut max_reg = 0u16;
    let mut seen_ref_params = HashSet::new();
    for (arg, (param_kind, reg)) in func.sig.inputs.iter().zip(param_layout) {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "#[jit_inline] parameters must be simple identifiers",
            ));
        };
        let binding_kind = match param_kind {
            InlineReturnKind::Int => BindingKind::Int,
            InlineReturnKind::Ref => BindingKind::Ref,
            InlineReturnKind::Float => BindingKind::Float,
            InlineReturnKind::Void => unreachable!("helper parameters cannot be void"),
        };
        let param_name = pat_ident.ident.to_string();
        let declared_struct_type = ref_param_structs.get(&param_name).cloned();
        if declared_struct_type.is_some() && !matches!(binding_kind, BindingKind::Ref) {
            return Err(syn::Error::new_spanned(
                &pat_type.ty,
                "#[jit_inline(ref_params = { ... })] may only declare usize/pointer ref parameters",
            ));
        }
        if declared_struct_type.is_some() {
            seen_ref_params.insert(param_name.clone());
        }
        max_reg = max_reg.max(reg.saturating_add(1));
        lowerer.bindings.insert(
            param_name,
            Binding {
                reg,
                kind: binding_kind,
                depends_on_stack: false,
                struct_type: declared_struct_type,
            },
        );
    }
    for (name, _) in ref_params {
        if !seen_ref_params.contains(&name.to_string()) {
            return Err(syn::Error::new_spanned(
                name,
                "#[jit_inline(ref_params = { ... })] names a parameter that does not exist",
            ));
        }
    }
    lowerer.next_reg = max_reg;

    let return_reg = if matches!(return_kind, InlineReturnKind::Void) {
        for stmt in &func.block.stmts {
            if lowerer.lower_stmt(stmt).is_none() {
                return Ok(None);
            }
        }
        0
    } else {
        let Some(binding) = lowerer.lower_block_value(&func.block) else {
            return Ok(None);
        };
        binding.reg
    };

    let helper_name = func.sig.ident.to_string();
    let (reserved_i, reserved_r, reserved_f) = inline_helper_param_counts(func)?;
    let return_register = match return_kind {
        InlineReturnKind::Int => Some(Register::int(return_reg)),
        InlineReturnKind::Ref => Some(Register::ref_(return_reg)),
        InlineReturnKind::Float => Some(Register::float(return_reg)),
        InlineReturnKind::Void => None,
    };
    let (_, compact_return) = super::regalloc::compact_registers(
        &mut lowerer,
        super::regalloc::RegisterCounts {
            ints: reserved_i,
            refs: reserved_r,
            floats: reserved_f,
        },
        return_register,
    );
    let return_reg = compact_return.map_or(return_reg, |reg| u16::from(reg.index));
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness(&helper_name, &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    // The body has lowered, so the consulted-key set is final. Reported per
    // HELPER, not per machine: each `#[jit_inline]` carries its own
    // `int_fields` / `ref_fields`, and a key one helper consults says nothing
    // about the helper next to it.
    //
    // This surface is where the population lives. A `#[jit_interp]` machine
    // declares a handful of keys; a consumer's helpers repeat theirs at every
    // site, and a key that matches nothing at one site matches nothing at
    // dozens.
    let unconsulted_declarations = match inline_config.as_ref() {
        Some(config) => {
            let consulted = config.consulted_field_keys.borrow();
            let mut keys: Vec<&String> = config
                .int_fields
                .keys()
                .chain(config.ref_fields.keys())
                .filter(|key| !consulted.contains(*key))
                .collect();
            keys.sort();
            keys.dedup();
            let records = keys.into_iter().map(|key| {
                quote! {
                    majit_metainterp::record_unconsulted_field_declaration(#helper_name, #key);
                }
            });
            quote! { #(#records)* }
        }
        None => quote! {},
    };
    let statements = lowerer.statements;
    Ok(Some(InlineHelperJitCode {
        body: quote! {
            #unconsulted_declarations
            #(#statements)*
        },
        return_reg,
        return_kind,
        liveness_prebuild,
    }))
}

pub(crate) fn inline_helper_param_layout(
    func: &ItemFn,
) -> syn::Result<Vec<(InlineReturnKind, u16)>> {
    let mut next_i = 0u16;
    let mut next_r = 0u16;
    let mut next_f = 0u16;
    let mut layout = Vec::with_capacity(func.sig.inputs.len());
    for arg in &func.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let param_kind = classify_param_type(&pat_type.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &pat_type.ty,
                "#[jit_inline] parameters must use i64/isize (Int), usize/pointer (Ref), or f64 (Float)",
            )
        })?;
        let reg = match param_kind {
            InlineReturnKind::Int => {
                let reg = next_i;
                next_i = next_i.saturating_add(1);
                reg
            }
            InlineReturnKind::Ref => {
                let reg = next_r;
                next_r = next_r.saturating_add(1);
                reg
            }
            InlineReturnKind::Float => {
                let reg = next_f;
                next_f = next_f.saturating_add(1);
                reg
            }
            InlineReturnKind::Void => unreachable!("helper parameters cannot be void"),
        };
        layout.push((param_kind, reg));
    }
    Ok(layout)
}

/// The `arg_classes` half of the calldescr `JitCode(name, fnaddr, calldescr)`
/// (`call.py:167-169`) needs to describe the helper's own entry point: one char
/// per parameter in source order, which is the order `descr.py create_call_stub`
/// places them in and the order the blackhole's `collect_call_args` reads them
/// back out of the per-kind lists.
///
/// The result half is not here because the caller already holds it — it is the
/// `return_kind` [`generate_inline_helper_jitcode_with_calls`] classified off
/// the same signature, and the jitcode's own typed return is emitted from it.
pub(crate) fn inline_helper_arg_classes(func: &ItemFn) -> syn::Result<String> {
    Ok(inline_helper_param_layout(func)?
        .into_iter()
        .map(|(kind, _)| inline_return_kind_class(kind))
        .collect())
}

/// The `descr.py` argument/result class char for an inline-helper kind.
pub(crate) fn inline_return_kind_class(kind: InlineReturnKind) -> char {
    match kind {
        InlineReturnKind::Int => 'i',
        InlineReturnKind::Ref => 'r',
        InlineReturnKind::Float => 'f',
        InlineReturnKind::Void => 'v',
    }
}

pub(crate) fn inline_helper_param_counts(func: &ItemFn) -> syn::Result<(u16, u16, u16)> {
    let layout = inline_helper_param_layout(func)?;
    let mut count_i = 0u16;
    let mut count_r = 0u16;
    let mut count_f = 0u16;
    for (kind, _) in layout {
        match kind {
            InlineReturnKind::Int => count_i = count_i.saturating_add(1),
            InlineReturnKind::Ref => count_r = count_r.saturating_add(1),
            InlineReturnKind::Float => count_f = count_f.saturating_add(1),
            InlineReturnKind::Void => {}
        }
    }
    Ok((count_i, count_r, count_f))
}

fn try_generate_jitcode_body_inner(
    body: &Expr,
    config: Option<&LowererConfig>,
) -> Option<GeneratedJitCodeBody> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    // RPython `compute_liveness(ssarepr) → remove_repeated_live(ssarepr)
    // → assemble()` (codewriter.py call order). `annotate_live_markers_
    // with_liveness` materialises the per-marker fixed-point alive set
    // onto each `LiveMarker.reads` so the repeated-live pass and the
    // emit-time triple rewrite both consume the same ssarepr-mutated
    // shape `liveness.py` produces.
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some(GeneratedJitCodeBody {
        body: quote! {
            #(#statements)*
        },
        liveness_prebuild,
        green_schema: Vec::new(),
        red_schema: Vec::new(),
    })
}

/// A.3.6.1 (`jtransform.py` `promote_greens`): bind body-local `let` stmts that
/// appear in the dispatch while-body BEFORE the `jit_merge_point!()`
/// macro stmt, so that consumer-declared
/// `#[jit_interp(greens = [<body-local>])]` (say `greens = [ok]` with
/// `let ok = program.get_req_size(pc) <= ...`)
/// flow through `resolve_greens` / `emit_promote_greens` without panic.
///
/// TODO: RPython has no equivalent two-pass walker.
/// Its annotator-driven flowgraph SpaceOperation-lowers every stmt before
/// `jtransform.handle_jit_marker__jit_merge_point` fires, so `Variable`
/// records exist for body-locals at merge-point rewrite time. Pyre's
/// proc-macro lowerer walks `syn::Block` AST directly with limited
/// expression-lowering coverage, requiring this explicit pre-pass.
///
/// Returns `Some(())` once the `jit_merge_point!()` macro stmt is reached
/// (or the while-body ends without one). Eligibility for binding is
/// delegated to `Lowerer::lower_local`, which in turn delegates to
/// `lower_value_expr`; non-`let` stmts and `let`s with unsupported RHS
/// shapes are skipped silently here (the existing `lower_pre_dispatch_stmts`
/// post-merge-point walker / dispatch-body emit is responsible for
/// diagnostics on its own pass).
pub(super) fn bind_pre_merge_point_stmts(
    lowerer: &mut Lowerer,
    func_block: &syn::Block,
) -> Option<()> {
    let loop_body = if let Some(dispatch_match) = find_dispatch_match(func_block) {
        find_dispatch_loop_body(func_block, dispatch_match)?
    } else {
        portal_loop_body(func_block)?
    };
    for stmt in &loop_body.stmts {
        if is_jit_merge_point_macro(stmt) {
            break;
        }
        if let syn::Stmt::Local(local) = stmt {
            // A side-effecting `let` the lowerer cannot reproduce must
            // not stay in the interpreter-only prefix: the compiled
            // back-edge would omit it. Fail the body so the portal
            // stays in the interpreter rather than compiling a loop
            // that drops the mutation. A side-effect-free match
            // (`jit_interp_dispatch_match_choice` `InLoopState`) is
            // prefix-only and can be skipped — `jit_merge_point` is
            // the compiled header.
            if lowerer.lower_local(local).is_none()
                && crate::jit_interp::codegen_trace::local_init_has_mutating_effect(local)
            {
                return None;
            }
        }
    }
    Some(())
}
