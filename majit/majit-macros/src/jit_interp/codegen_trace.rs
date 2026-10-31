#![allow(dead_code)]
//! Generate `JitCode` builders and the generic `__trace_*` wrapper.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Block, Expr, ExprMatch, ItemFn, Stmt};

use super::JitInterpConfig;
use super::StateFieldKind;
use super::classify::classify_arms;
use super::jitcode_lower::{
    self, LowererConfig, ValueKind, is_can_enter_jit_macro, is_jit_merge_point_macro,
};

/// `pyjitpl.py MetaInterp.initialize_state_from_start` →
/// `MIFrame.setup_call(original_boxes)`: after greens, the dispatch
/// JitCode's remaining arguments are the plain reds (int scalars,
/// ptr then len per `[int]`, ref scalars, float scalars).
/// `jtransform.py` `rewrite_op_getarrayitem` raw arm emits
/// `getarrayitem_raw_<kind>` for a raw (non-GC) array (`_i` or `_f`
/// from `getkind(ARRAY.OF)`); a Rust slice is that `{ptr, len}` view,
/// so both words are INT identity reds regardless of element kind.
/// Vable-field scalars stay in
/// `initialize_virtualizable` / `virtualizable_boxes` and are not
/// pushed here.
fn plain_red_argbox_pushes(config: &JitInterpConfig, has_infer_call_arm: bool) -> TokenStream {
    let Some(sf) = config.state_fields.as_ref() else {
        return quote! {};
    };
    let skip_vable_scalars = super::state_is_vable_identity(
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
        config.split_dispatch,
        has_infer_call_arm,
    );
    let int_scalars: Vec<&super::StateFieldDecl> = if skip_vable_scalars {
        Vec::new()
    } else {
        sf.fields
            .iter()
            .filter(
                |f| matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "int"),
            )
            .collect()
    };
    let arrays: Vec<&super::StateFieldDecl> = sf
        .fields
        .iter()
        .filter(|f| matches!(f.kind, StateFieldKind::Array(_)))
        .collect();
    let ref_scalars: Vec<&super::StateFieldDecl> = if skip_vable_scalars {
        Vec::new()
    } else {
        sf.fields
            .iter()
            .filter(|f| matches!(f.kind, StateFieldKind::Ref(_) | StateFieldKind::Str))
            .collect()
    };
    // `warmspot.py` `make_virtualizable_infos`: the heap virtualizable is
    // one red at `jitdriver.reds.index(vname)`. Push it with the other
    // `ref(_)` / `str` fields, in declaration order, and give that slot
    // the identity box (`pyjitpl.py` `_nonstandard_virtualizable` step 3:
    // a bridge's fail args are not the loop's reds, so the walk's vable
    // base must be `virtualizable_boxes[-1]`).
    let heap_vable_name: Option<String> = if skip_vable_scalars {
        None
    } else {
        config
            .virtualizable_decl
            .as_ref()
            .map(|decl| decl.var_name.to_string())
    };
    let float_scalars: Vec<&super::StateFieldDecl> = if skip_vable_scalars {
        Vec::new()
    } else {
        sf.fields
            .iter()
            .filter(
                |f| matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "float"),
            )
            .collect()
    };
    if int_scalars.is_empty()
        && arrays.is_empty()
        && ref_scalars.is_empty()
        && float_scalars.is_empty()
    {
        return quote! {};
    }
    let int_pushes: Vec<TokenStream> = int_scalars
        .iter()
        .map(|f| {
            let fname = &f.name;
            quote! {
                {
                    let __bits = state.#fname as i64;
                    let __box = majit_ir::OpRef::input_arg_int(__red_offset);
                    __red_offset = __red_offset.wrapping_add(1);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Int,
                        __box,
                        __bits,
                    ));
                }
            }
        })
        .collect();
    let array_pushes: Vec<TokenStream> = arrays
        .iter()
        .map(|f| {
            let fname = &f.name;
            quote! {
                {
                    // JIT stores through this address (`setarrayitem_raw_*`), so
                    // it is derived from a mutable borrow.
                    let __bits = state.#fname.as_mut_ptr() as i64;
                    let __box = majit_ir::OpRef::input_arg_int(__red_offset);
                    __red_offset = __red_offset.wrapping_add(1);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Int,
                        __box,
                        __bits,
                    ));
                }
                {
                    let __bits = state.#fname.len() as i64;
                    let __box = majit_ir::OpRef::input_arg_int(__red_offset);
                    __red_offset = __red_offset.wrapping_add(1);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Int,
                        __box,
                        __bits,
                    ));
                }
            }
        })
        .collect();
    let ref_pushes: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|f| {
            if heap_vable_name.as_ref().is_some_and(|n| f.name == *n) {
                heap_vable_identity_argbox_push()
            } else {
                let fname = &f.name;
                quote! {
                    {
                        let __bits = state.#fname as i64;
                        let __box = majit_ir::OpRef::input_arg_ref(__red_offset);
                        __red_offset = __red_offset.wrapping_add(1);
                        __jitcode_args.push((
                            majit_metainterp::JitArgKind::Ref,
                            __box,
                            __bits,
                        ));
                    }
                }
            }
        })
        .collect();
    let float_pushes: Vec<TokenStream> = float_scalars
        .iter()
        .map(|f| {
            let fname = &f.name;
            quote! {
                {
                    let __bits = (state.#fname as f64).to_bits() as i64;
                    let __box = majit_ir::OpRef::input_arg_float(__red_offset);
                    __red_offset = __red_offset.wrapping_add(1);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Float,
                        __box,
                        __bits,
                    ));
                }
            }
        })
        .collect();
    quote! {
        let mut __red_offset: u32 = 0;
        #(#int_pushes)*
        #(#array_pushes)*
        #(#ref_pushes)*
        #(#float_pushes)*
        let _ = __red_offset;
    }
}

/// One argbox for a `virtualizable_fields` heap object.
///
/// `pyjitpl.py` `_nonstandard_virtualizable` step 3: the walk's vable
/// base is standard only when it IS `virtualizable_boxes[-1]`. A heap
/// frame's entry index is not that box on a bridge (fail args are not
/// the loop's reds), so this slot carries the identity box itself. It
/// is the object's one red (`warmspot.py` `make_virtualizable_infos`
/// `index_of_virtualizable`), occupying `__red_offset` so later ref
/// scalars mint distinct `input_arg_ref` boxes.
fn heap_vable_identity_argbox_push() -> TokenStream {
    quote! {
        {
            let __vable_argbox = match (
                __ctx.standard_virtualizable_box(),
                __ctx.standard_virtualizable_concrete(),
            ) {
                (Some(__op), Some(majit_ir::Value::Ref(__r))) => (
                    majit_metainterp::JitArgKind::Ref,
                    __op,
                    __r.as_usize() as i64,
                ),
                (Some(__op), Some(majit_ir::Value::Int(__bits))) => (
                    majit_metainterp::JitArgKind::Ref,
                    __op,
                    __bits,
                ),
                _ => {
                    let Some(__fallback) = __ctx.standard_virtualizable_jitcode_argbox() else {
                        return TraceAction::Abort;
                    };
                    __fallback
                }
            };
            __jitcode_args.push(__vable_argbox);
            __red_offset = __red_offset.wrapping_add(1);
        }
    }
}

pub fn generate_trace_fn(
    config: &JitInterpConfig,
    func: &ItemFn,
    has_infer_call_arm: bool,
) -> TokenStream {
    let fn_name = &func.sig.ident;
    let trace_fn_name = format_ident!("__trace_{}", fn_name);
    let prebuild_fn_name = format_ident!("__prebuild_jitcode_liveness_{}", fn_name);
    let dispatch_jitcode_fn_name = format_ident!("__dispatch_jitcode_{}", fn_name);
    let dispatch_jitcode_name = fn_name.to_string();
    let declare_schema_fn_name = format_ident!("__declare_jit_schema_{}", fn_name);
    // Must match `codegen_state.rs`'s spelling: the symbolic-state struct is one
    // module-level item shared by both emitters, suffixed so two machines can
    // live in one module.
    let sym_ty = format_ident!("__JitSym_{}", fn_name);

    // A portal loop with no opcode `match` after `jit_merge_point` is the
    // marked.py shape: `while i < len { can_enter; merge_point; body }`.
    // `warmspot.rewrite_can_enter_jit` inserts the header tick when the
    // source has no `can_enter_jit` of its own. Keep the existing error
    // only when there is neither a dispatch match nor a portal loop.
    let match_expr = find_dispatch_match(&func.block);
    let classified = match match_expr {
        Some(match_expr) => classify_arms(&match_expr.arms),
        None if portal_loop_body(&func.block).is_some() => {
            if let Some(stmt) = first_unsupported_pre_merge_stmt(&func.block) {
                return syn::Error::new_spanned(
                    stmt,
                    "matchless portal loop runs this statement before \
                     jit_merge_point, but the compiled back-edge does not; \
                     move it after the merge point or bind it with let",
                )
                .to_compile_error();
            }
            Vec::new()
        }
        None => {
            return syn::Error::new_spanned(
                func,
                "could not find a portal loop or opcode dispatch match",
            )
            .to_compile_error();
        }
    };

    let portal_greens = super::portal_green_params(config, func);
    let portal_green_layout: Vec<(String, ValueKind)> = portal_greens
        .iter()
        .map(|(name, _, tag)| {
            let kind = match tag {
                super::green_type_tag::GreenTypeTag::Int => ValueKind::Int,
                super::green_type_tag::GreenTypeTag::Ref
                | super::green_type_tag::GreenTypeTag::Str
                | super::green_type_tag::GreenTypeTag::Unicode => ValueKind::Ref,
                super::green_type_tag::GreenTypeTag::Float => ValueKind::Float,
            };
            (name.to_string(), kind)
        })
        .collect();

    let carried_greens = super::loop_carried_greens(config, func);
    let lowerer_config = LowererConfig::new(
        &config.io_shims,
        &config.calls,
        config.auto_calls,
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
        &config.greens,
        &config.green_type_tags,
        &portal_green_layout,
        &config.reds,
        &config.state_type,
        &config.env_type,
        &config.residual_writes,
        &config.pool_arrays,
        &config.ref_fields,
        &config.array_fields,
        &config.int_fields,
        &config.call_returns,
        &config.headerless_structs,
        &config.inlined_prefix,
        &config.native_int_binops,
        &config.native_tag_small,
        &config.native_identity,
        config.split_dispatch,
        has_infer_call_arm,
        config.switch_dispatch,
    )
    .with_loop_carried_greens(
        carried_greens
            .iter()
            .map(|(name, kind)| (name.to_string(), *kind))
            .collect(),
    );

    let env_type = &config.env_type;
    let state_type = &config.state_type;
    let red_argbox_pushes = plain_red_argbox_pushes(config, has_infer_call_arm);

    // Dispatch JitCode singleton produced by lower_dispatch_body.
    // `__trace_*` invokes it; the install pipeline registers it as the
    // driver-shared singleton. The prebuild step splices the
    // dispatch JitCode's per-marker liveness prebuild into
    // `__prebuild_jitcode_liveness_*` alongside the per-arm prebuilds, so the
    // driver-shared `Assembler` already holds every triple the dispatch
    // factory will emit — preserving the no-growth invariant asserted in
    // `__trace_*` below for the dispatch JitCode build path.
    let (_, portal_ref_args, _) = lowerer_config.portal_input_kind_counts();
    // Heap virtualizable: the object's one red sits at its declared
    // position among the ref state fields (`warmspot.py`
    // `make_virtualizable_infos` `index_of_virtualizable`). `[.. ; virt]`
    // still extra-pushes the state identity at `portal_ref_args`.
    let vable_input_reg =
        portal_ref_args.saturating_add(lowerer_config.heap_vable_ref_pos().unwrap_or(0));
    let dispatch_lowerer_config = lowerer_config.with_vable_input_ref_reg(vable_input_reg);
    let (
        dispatch_body,
        dispatch_prebuild,
        dispatch_lower_ok,
        dispatch_green_schema,
        dispatch_red_schema,
    ) = match jitcode_lower::lower_dispatch_body(
        &dispatch_lowerer_config,
        &func.block,
        &classified,
        &func.sig.output,
    ) {
        Some(generated) => (
            generated.body,
            generated.liveness_prebuild,
            true,
            generated.green_schema,
            generated.red_schema,
        ),
        None => (quote! {}, quote! {}, false, Vec::new(), Vec::new()),
    };
    // Slice (audit Issue #5) — split the (name, type) tuples into
    // separate name + type token vectors so the macro splat below
    // can interleave them as `(#name, #type)` without losing
    // ordering.  Per-pair iteration via tuple destructuring inside
    // the splat is not supported by `quote!`.
    let dispatch_green_schema_names: Vec<&str> = dispatch_green_schema
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let dispatch_green_schema_types: Vec<&proc_macro2::TokenStream> =
        dispatch_green_schema.iter().map(|(_, t)| t).collect();
    let dispatch_red_schema_names: Vec<&str> = dispatch_red_schema
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let dispatch_red_schema_types: Vec<&proc_macro2::TokenStream> =
        dispatch_red_schema.iter().map(|(_, t)| t).collect();
    let (identity_i_scalars, identity_r_scalars, identity_f_scalars) =
        dispatch_lowerer_config.identity_scalar_red_kind_counts();

    // The identity `label_at` closure is constructed by
    // `generate_merge_wrapper` and passed into `__trace_*` as a
    // `ClosureRuntimeWithResolver`.  The recursive-call seams live at
    // the merge wrapper layer because they borrow
    // `MetaInterp::warm_state` via the
    // `with_trace_ctx_and_token_resolver` split-borrow helper.
    // (Legacy identity closure `|pc| pc` retained at the
    // call site there.)
    // A state-as-vable identity (`[.. ; virt]`, or a sub-JitCode arm
    // (`split_dispatch` / infer / `inline_pipeline_*`) with scalar/`ref(_)`/
    // `str` fields) seeds a standard-virtualizable identity: the OpRef of
    // `virtualizable_boxes[-1]` must reach the extra-pushed ref register so
    // `getfield_vable_*` / `getarrayitem_vable_*` resolve the same box that
    // `init_virtualizable_boxes` minted (identity-equal per
    // `begin_nonstandard_virtualizable` Step 3). A `virtualizable_fields`
    // heap object is already a named red among the ref scalars and is
    // pushed once in `plain_red_argbox_pushes`.
    let state_has_vable_identity = super::state_is_vable_identity(
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
        config.split_dispatch,
        has_infer_call_arm,
    );
    // `[.. ; virt]` keeps the extra identity argbox: its identity slot
    // and `virtualizable_boxes[-1]` are the same inputarg. A heap
    // virtualizable occupies its declared ref slot instead.
    let push_virtualizable_argbox = if state_has_vable_identity {
        quote! {
            let Some(__vable_argbox) = __ctx.standard_virtualizable_jitcode_argbox() else {
                return TraceAction::Abort;
            };
            __jitcode_args.push(__vable_argbox);
        }
    } else {
        quote! {}
    };
    let portal_green_param_decls: Vec<_> = portal_greens
        .iter()
        .map(|(name, ty, _)| quote! { #name: #ty, })
        .collect();
    let carried_param_decls: Vec<_> = carried_greens
        .iter()
        .map(|(name, _)| {
            quote! { #name: impl ::core::marker::Copy + majit_ir::GreenAsI64, }
        })
        .collect();
    let carried_arg_pushes: Vec<_> = carried_greens
        .iter()
        .map(|(name, kind)| match kind {
            ValueKind::Int => quote! {
                {
                    let __carried_bits =
                        <_ as majit_ir::GreenAsI64>::__green_repr(#name).0;
                    let __carried_box = __ctx.const_int(__carried_bits);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Int,
                        __carried_box,
                        __carried_bits,
                    ));
                }
            },
            ValueKind::Ref => quote! {
                {
                    let __carried_bits =
                        <_ as majit_ir::GreenAsI64>::__green_repr(#name).0;
                    let __carried_box = __ctx.const_ref(__carried_bits);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Ref,
                        __carried_box,
                        __carried_bits,
                    ));
                }
            },
            ValueKind::Float => quote! {
                {
                    let __carried_bits =
                        <_ as majit_ir::GreenAsI64>::__green_repr(#name).0;
                    let __carried_box = __ctx.const_float(__carried_bits);
                    __jitcode_args.push((
                        majit_metainterp::JitArgKind::Float,
                        __carried_box,
                        __carried_bits,
                    ));
                }
            },
        })
        .collect();
    let portal_green_arg_pushes: Vec<_> = portal_greens
        .iter()
        .map(|(name, _, tag)| match tag {
            super::green_type_tag::GreenTypeTag::Int => quote! {
                let __green_bits = #name as i64;
                let __green_box = __ctx.const_int(__green_bits);
                __jitcode_args.push((
                    majit_metainterp::JitArgKind::Int,
                    __green_box,
                    __green_bits,
                ));
            },
            super::green_type_tag::GreenTypeTag::Ref
            | super::green_type_tag::GreenTypeTag::Str
            | super::green_type_tag::GreenTypeTag::Unicode => quote! {
                let __green_bits = #name as i64;
                let __green_box = __ctx.const_ref(__green_bits);
                __jitcode_args.push((
                    majit_metainterp::JitArgKind::Ref,
                    __green_box,
                    __green_bits,
                ));
            },
            super::green_type_tag::GreenTypeTag::Float => quote! {
                let __green_bits = (#name as f64).to_bits() as i64;
                let __green_box = __ctx.const_float(__green_bits);
                __jitcode_args.push((
                    majit_metainterp::JitArgKind::Float,
                    __green_box,
                    __green_bits,
                ));
            },
        })
        .collect();

    let trace_fn_body = quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #trace_fn_name<__R: majit_metainterp::JitCodeRuntime>(
            __ctx: &mut majit_metainterp::TraceCtx,
            __sym: &mut #sym_ty,
            state: &mut #state_type,
            program: &#env_type,
            pc: usize,
            #(#portal_green_param_decls)*
            #(#carried_param_decls)*
            // Caller passes a `ClosureRuntimeWithResolver` carrying
            // `label_at` and the recursive-call seams. CALL_ASSEMBLER
            // descrs carry a `Weak<JitCellToken>`; the token number is
            // log-only.
            __runtime: &__R,
            // Dispatch JitCode singleton cloned from JitDriver by the
            // caller before the mutable borrow. `None` only when
            // `lower_dispatch_body` returned `None` at proc-macro time
            // (`dispatch_lower_ok == false`) and `register_dispatch_jitcode`
            // was therefore skipped at install — trace aborts permanently
            // in that case, matching `pypy/module/pypyjit/interp_jit.py:82-94`
            // dispatch() which only invokes the singleton dispatch JitCode.
            __dispatch_jitcode_arg: Option<&::std::sync::Arc<majit_metainterp::JitCode>>,
        ) -> majit_metainterp::TraceAction {
            use majit_metainterp::TraceAction;

            let Some(__dispatch_arc) = __dispatch_jitcode_arg else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] no dispatch JitCode registered at pc={} \
                         (lower_dispatch_body returned None at proc-macro time)",
                        pc,
                    );
                }
                return TraceAction::AbortPermanent;
            };
            let __jitcode: majit_metainterp::JitCode = (**__dispatch_arc).clone();

            // The metainterp walk is the sole executor: it runs each residual
            // function-pointer call (BC_CALL_INT / BC_RESIDUAL_CALL_VOID etc.)
            // for real and records the matching IR call op. The IR call op runs
            // at compiled-trace runtime.
            //
            // Dispatch JitCode reads `program` and `pc` as caller-provided
            // IR arguments (matching the (Ref, Int) prefix of `reds` declared
            // by `__declare_jit_schema_*` and consumed by
            // `trace_jitcode_with_args`).
            let mut __jitcode_args: ::std::vec::Vec<(
                majit_metainterp::JitArgKind,
                majit_ir::OpRef,
                i64,
            )> = ::std::vec::Vec::new();
            let __program_bits = program as *const #env_type as *const () as usize as i64;
            let __program_box = __ctx.const_ref(__program_bits);
            __jitcode_args.push((
                majit_metainterp::JitArgKind::Ref,
                __program_box,
                __program_bits,
            ));
            let __pc_bits = pc as i64;
            let __pc_box = __ctx.const_int(__pc_bits);
            __jitcode_args.push((
                majit_metainterp::JitArgKind::Int,
                __pc_box,
                __pc_bits,
            ));
            #(#portal_green_arg_pushes)*
            #(#carried_arg_pushes)*
            #push_virtualizable_argbox
            #red_argbox_pushes
            let __result = majit_metainterp::trace_jitcode_with_args_and_runtime(
                __ctx,
                __sym,
                &__jitcode,
                pc,
                __runtime,
                &__jitcode_args,
            );
            if majit_metainterp::majit_log_enabled() && !matches!(__result, TraceAction::Continue) {
                eprintln!(
                    "[jit] trace action at pc={} -> {:?}",
                    pc,
                    __result
                );
            }
            __result
        }
    };

    quote! {
        /// Dispatch JitCode singleton builder.
        ///
        /// Builds the entire dispatch loop body (jit_merge_point + pre-dispatch
        /// ops + opcode fetch + dispatch chain + per-arm INLINE_CALL + loop
        /// close) as a single JitCode. `__trace_*` invokes this; the install
        /// pipeline registers it via `JitDriver::register_dispatch_jitcode`.
        ///
        /// Returns `Option<JitCode>`: `Some(jc)` when `lower_dispatch_body`
        /// succeeded at proc-macro time, `None` when the body shape was
        /// rejected (e.g. unrecognised inner control flow).  PyPy's
        /// `make_jitcodes()` / `pyjitpl.py finish_setup()` only install
        /// completed jitcodes — there is no "empty body installed as
        /// success" path.  The install pipeline at `codegen_state.rs`
        /// `if let Some(jc) = ... { register }` matches that lifecycle
        /// by skipping `register_dispatch_jitcode` when this returns
        /// `None`.
        #[allow(non_snake_case, unused_variables, unused_mut)]
        #[doc(hidden)]
        pub fn #dispatch_jitcode_fn_name(
            __asm: &mut majit_metainterp::Assembler,
            // jtransform.py:1704 portal_jd.index threaded as runtime param.
            __jdindex: i64,
        ) -> Option<majit_metainterp::JitCode> {
            if !#dispatch_lower_ok {
                // `lower_dispatch_body` rejected the body at proc-macro
                // time; surface as None so the install pipeline skips
                // `register_dispatch_jitcode` per PyPy parity.
                return None;
            }
            let mut __builder = majit_metainterp::JitCodeBuilder::new();
            // `jitcode.py JitCode.__init__ self.name = name`.  This is the root JitCode of
            // the machine, so it names the dispatch function itself; its arms'
            // sub-JitCodes name the arm they came from.
            __builder.set_name(#dispatch_jitcode_name);
            let _live_offset_patch = __builder.live_placeholder();
            #dispatch_body
            __builder.finalize_liveness(__asm);
            Some(__builder.finish())
        }

        /// Pre-register the dispatch JitCode's per-marker liveness
        /// triples into the driver-shared `Assembler`, mirroring RPython
        /// `pyjitpl.py finish_setup`'s "all `-live-` entries land
        /// in `asm.all_liveness` before the snapshot" invariant.
        /// Invoked from `__JitMeta_<fn>::install_canonical_liveness` exactly
        /// once at install time, before
        /// `JitDriver::install_canonical_liveness` snapshots
        /// `metainterp_sd.liveness_info`.
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #prebuild_fn_name(__asm: &mut majit_metainterp::Assembler) {
            #dispatch_prebuild
        }

        /// Slice (audit Issue #5) — declare the dispatch JitCode's
        /// `(name, GreenType)` green schema and `(name, IR Type)` red
        /// schema on the JitDriver so `JitDriverStaticData::
        /// green_args_spec` reports STR/UNICODE subtypes and
        /// `green_kind_counts` / `red_kind_counts` reflect the real
        /// payload of `BC_JIT_MERGE_POINT`.  RPython
        /// `warmspot.py:663-665` derives the same `_green_args_spec`
        /// from the `JIT_ENTER_FUNCTYPE` signature; pyre derives it
        /// from the `lowerer.bindings` BindingKind plus
        /// `green_type_tags` (the `: str` / `: unicode` declarations)
        /// at `lower_dispatch_body` time.  No-op when the dispatch
        /// body failed to lower (the schema vectors are then empty).
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #declare_schema_fn_name<S: majit_metainterp::JitState>(
            __driver: &mut majit_metainterp::JitDriver<S>,
        ) {
            let __greens: ::std::vec::Vec<(&str, majit_ir::GreenType)> = vec![
                #( (#dispatch_green_schema_names, #dispatch_green_schema_types) ),*
            ];
            let __reds: ::std::vec::Vec<(&str, majit_ir::Type)> = vec![
                #( (#dispatch_red_schema_names, #dispatch_red_schema_types) ),*
            ];
            // Named portal reds only. Identity slots (plain scalars and
            // ptr then len per `[int]`) are marker operands
            // (`resolve_reds`) and `-live-` entries, not
            // `JitDriverStaticData::vars` — `portal_calldescr` is the
            // original function (`warmspot.py`
            // `cpu.calldescrof(portal_runner)`).
            __driver.declare_schema_typed(__greens, __reds);
            // Same source `resolve_reds` uses: `identity_slot_registers`.
            // `install_canonical_liveness` declares this before building
            // the dispatch JitCode; `register_dispatch_jitcode` is the
            // only reader (`merge_point_identity_reds`).
            __driver.declare_merge_point_identity_reds((
                #identity_i_scalars,
                #identity_r_scalars,
                #identity_f_scalars,
            ));
        }

        #trace_fn_body
    }
}

pub(crate) fn find_dispatch_match(block: &syn::Block) -> Option<&syn::ExprMatch> {
    // The dispatch match is the one in the portal loop's body, after the merge
    // point. That is the shape the rest of this machinery already walks —
    // `find_dispatch_loop_body`, `lower_pre_dispatch_stmts` and
    // `bind_pre_merge_point_stmts` all locate the same loop — so search there
    // rather than over the whole function.
    //
    // The rule used to be "the match with the most arms, anywhere", on the
    // reasoning that a dispatch has many opcode arms and a setup match only a
    // few. Arm count is not a property of being the dispatch: a five-arm
    // `let x = match ..` in front of a three-opcode dispatch takes its place,
    // and then `classify_arms` reads that match's arms as the opcodes while
    // the two pre-dispatch walkers, which find their loop by the dispatch
    // match it contains, find no loop and silently lower nothing. The portal
    // compiles a loop that runs none of the interpreter and answers with it.
    if let Some(body) = portal_loop_body(block) {
        let mut after_merge_point = Vec::new();
        let mut seen_merge_point = false;
        for stmt in &body.stmts {
            if is_jit_merge_point_macro(stmt) {
                seen_merge_point = true;
            } else if seen_merge_point {
                collect_matches_in_stmt(stmt, &mut after_merge_point);
            }
        }
        // Only a match dominated by an opcode fetch is the dispatch.
        // A matchless portal may contain an ordinary local `match`; treating
        // that as dispatch skips `lower_matchless_portal_body` and emits an
        // empty JitCode body.
        return after_merge_point
            .into_iter()
            .filter(|m| match_is_opcode_dispatch(body, m))
            .max_by_key(|m| m.arms.len());
    }

    // No portal loop this reads. Fall back to the old search so a shape
    // outside the rule above keeps whatever it did before.
    let mut all = Vec::new();
    collect_all_matches(block, &mut all);
    all.into_iter().max_by_key(|m| m.arms.len())
}

/// A match is the opcode dispatch only when an opcode-fetch binding
/// (`let op = program[pc]` / `program.get_op(pc)` / `insn_op(program, pc)`)
/// dominates it, or the scrutinee is itself `program[pc]`. A data index
/// (`program[state.pos]`) is not the portal PC and stays on the matchless path.
fn match_is_opcode_dispatch(loop_body: &syn::Block, candidate: &syn::ExprMatch) -> bool {
    let mut names = Vec::new();
    let mut seen_merge_point = false;
    for stmt in &loop_body.stmts {
        if is_jit_merge_point_macro(stmt) {
            seen_merge_point = true;
            continue;
        }
        if !seen_merge_point {
            continue;
        }
        if let Some(name) = opcode_fetch_binding_name(stmt) {
            names.push(name);
        }
        let mut matches = Vec::new();
        collect_matches_in_stmt(stmt, &mut matches);
        if matches.iter().any(|m| std::ptr::eq(*m, candidate)) {
            return match_scrutinee_is_opcode_fetch(candidate, &names);
        }
    }
    false
}

fn opcode_fetch_binding_name(stmt: &syn::Stmt) -> Option<String> {
    let syn::Stmt::Local(local) = stmt else {
        return None;
    };
    let init = local.init.as_ref()?;
    let init_expr = match init.expr.as_ref() {
        syn::Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    let fetches_program = match init_expr {
        syn::Expr::Index(idx) => index_is_program_at_pc(idx),
        // `get_op(pc)` and `insn_op(program, pc)` are the opcode fetch.
        // `program.len()` / `get_operand` / `program[state.pos]` /
        // `insn_a(program, pc)` are ordinary values; treating them as
        // the dispatch would send a local match through
        // `lower_dispatch_chain` and drop the body.
        syn::Expr::MethodCall(mc) => method_is_get_op_at_pc(mc),
        syn::Expr::Call(call) => call_is_insn_op_at_pc(call),
        _ => false,
    };
    if !fetches_program {
        return None;
    }
    match &local.pat {
        syn::Pat::Ident(id) => Some(id.ident.to_string()),
        syn::Pat::Type(ty) => match ty.pat.as_ref() {
            syn::Pat::Ident(id) => Some(id.ident.to_string()),
            _ => None,
        },
        _ => None,
    }
}

fn match_scrutinee_is_opcode_fetch(m: &syn::ExprMatch, names: &[String]) -> bool {
    let expr = match m.expr.as_ref() {
        syn::Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    match expr {
        syn::Expr::Path(p) => p
            .path
            .get_ident()
            .is_some_and(|id| names.iter().any(|n| n == &id.to_string())),
        syn::Expr::Index(idx) => index_is_program_at_pc(idx),
        _ => false,
    }
}

pub(crate) fn unwrap_cast(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Cast(c) => unwrap_cast(&c.expr),
        other => other,
    }
}

fn index_is_program_at_pc(idx: &syn::ExprIndex) -> bool {
    expr_is_ident(&idx.expr, "program") && expr_is_ident(unwrap_cast(&idx.index), "pc")
}

fn method_is_get_op_at_pc(mc: &syn::ExprMethodCall) -> bool {
    expr_is_ident(&mc.receiver, "program")
        && mc.method == "get_op"
        && mc
            .args
            .first()
            .is_some_and(|a| expr_is_ident(unwrap_cast(a), "pc"))
}

/// The opcode-fetch call `insn_op(program, pc)`. Not `insn_a` / `insn_b`.
/// Both arguments must be the bare identifiers. A cast on either one
/// (`insn_op(program as _, pc as _)`) is a different fetch: lowering it
/// as the bare call would drop a narrowing conversion and make the
/// traced opcode disagree with the interpreter.
pub(crate) fn call_is_insn_op_at_pc(call: &syn::ExprCall) -> bool {
    let is_insn_op = match call.func.as_ref() {
        syn::Expr::Path(p) => p.path.segments.last().is_some_and(|s| s.ident == "insn_op"),
        _ => false,
    };
    is_insn_op
        && call.args.len() == 2
        && expr_is_ident(&call.args[0], "program")
        && expr_is_ident(&call.args[1], "pc")
}

fn expr_is_ident(expr: &syn::Expr, name: &str) -> bool {
    matches!(expr, syn::Expr::Path(p) if p.path.is_ident(name))
}

/// Statements the matchless compiled body will not run: everything before
/// `jit_merge_point` except `let` bindings the lowerer can reproduce and
/// the interpreter-only `can_enter_jit!` tick (`rewrite_can_enter_jit`).
fn first_unsupported_pre_merge_stmt(func_block: &syn::Block) -> Option<&syn::Stmt> {
    let body = portal_loop_body(func_block)?;
    for stmt in &body.stmts {
        if is_jit_merge_point_macro(stmt) {
            return None;
        }
        if is_can_enter_jit_macro(stmt) {
            continue;
        }
        if let syn::Stmt::Local(local) = stmt {
            if local_init_has_side_effect(local) {
                return Some(stmt);
            }
            continue;
        }
        return Some(stmt);
    }
    None
}

/// A pre-merge `let` whose initializer mutates (an assignment or
/// assign-op, including one nested in a block) or calls out (a
/// qualified `crate::tick()` the lowerer emits as a compile-time
/// constant) runs in the interpreter but is omitted or run once from
/// the compiled back-edge. Reject those so the portal fails at compile
/// time. Tracing hints (`promote` / `assert_not_none` /
/// `record_exact_class`) have no user-visible effect and stay allowed.
pub(crate) fn local_init_has_side_effect(local: &syn::Local) -> bool {
    let Some(init) = &local.init else {
        return false;
    };
    expr_has_pre_merge_side_effect(&init.expr)
}

/// Dispatch portals skip an unlowerable prefix `let` so the machine still
/// installs (`jit_interp_macro_init_is_opaque` `pick!`). Matchless portals
/// have no opcode arms to degrade, so [`local_init_has_side_effect`] still
/// compile-errors on unrecognized macros.
pub(crate) fn local_init_has_mutating_effect(local: &syn::Local) -> bool {
    let Some(init) = &local.init else {
        return false;
    };
    expr_has_mutating_effect(&init.expr)
}

fn expr_is_tracing_hint_call(func: &syn::Expr) -> bool {
    is_promote_call_path(func)
        || is_assert_not_none_call_path(func)
        || is_record_exact_class_call_path(func)
}

fn expr_is_tracing_hint_macro(mac: &syn::ExprMacro) -> bool {
    use super::jitcode_lower::classify_virtualizable_hint_syn_path;
    classify_virtualizable_hint_syn_path(&mac.mac.path).is_some()
        || path_ends_with(&mac.mac.path, "promote")
        || path_ends_with(&mac.mac.path, "assert_not_none")
        || path_ends_with(&mac.mac.path, "record_exact_class")
}

fn path_ends_with(path: &syn::Path, name: &str) -> bool {
    path.segments.last().is_some_and(|seg| seg.ident == name)
}

fn expr_has_mutating_effect(expr: &syn::Expr) -> bool {
    expr_has_pre_merge_effect(expr, false)
}

fn expr_has_pre_merge_side_effect(expr: &syn::Expr) -> bool {
    expr_has_pre_merge_effect(expr, true)
}

fn expr_has_pre_merge_effect(expr: &syn::Expr, reject_macros: bool) -> bool {
    use syn::visit::Visit;
    struct Finder {
        hit: bool,
        reject_macros: bool,
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_expr_assign(&mut self, _: &'ast syn::ExprAssign) {
            self.hit = true;
        }
        fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
            // A null constant is not a call (`jtransform.py` `Constant(nullptr)`).
            if !crate::jit_interp::jitcode_lower::call_is_null_ptr(node)
                && !expr_is_tracing_hint_call(&node.func)
            {
                self.hit = true;
            }
            syn::visit::visit_expr_call(self, node);
        }
        fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
            self.hit = true;
            syn::visit::visit_expr_method_call(self, node);
        }
        fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
            // syn::visit treats a macro body as opaque, so `tick!(state)`
            // would otherwise look side-effect-free. Matchless portals
            // reject unrecognized expression macros; tracing hints stay
            // allowed. Dispatch bind_pre_merge skips them so `pick!`
            // still installs the machine.
            if self.reject_macros && !expr_is_tracing_hint_macro(node) {
                self.hit = true;
            }
            syn::visit::visit_expr_macro(self, node);
        }
        fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
            self.hit = true;
        }
        fn visit_expr_break(&mut self, _: &'ast syn::ExprBreak) {
            self.hit = true;
        }
        fn visit_expr_continue(&mut self, _: &'ast syn::ExprContinue) {
            self.hit = true;
        }
        fn visit_expr_try(&mut self, _: &'ast syn::ExprTry) {
            self.hit = true;
        }
        fn visit_expr_yield(&mut self, _: &'ast syn::ExprYield) {
            self.hit = true;
        }
        fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
            // `[1, 2][state.pos]` can panic; lower_local cannot
            // reproduce a non-const index. The compiled back-edge
            // would drop it.
            self.hit = true;
            syn::visit::visit_expr_index(self, node);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {
            // The closure body does not run at the merge point.
        }
        fn visit_expr_async(&mut self, _: &'ast syn::ExprAsync) {}
        fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
            if matches!(
                node.op,
                syn::BinOp::AddAssign(_)
                    | syn::BinOp::SubAssign(_)
                    | syn::BinOp::MulAssign(_)
                    | syn::BinOp::DivAssign(_)
                    | syn::BinOp::RemAssign(_)
                    | syn::BinOp::BitXorAssign(_)
                    | syn::BinOp::BitAndAssign(_)
                    | syn::BinOp::BitOrAssign(_)
                    | syn::BinOp::ShlAssign(_)
                    | syn::BinOp::ShrAssign(_)
            ) {
                self.hit = true;
            }
            syn::visit::visit_expr_binary(self, node);
        }
    }
    let mut finder = Finder {
        hit: false,
        reject_macros,
    };
    finder.visit_expr(expr);
    finder.hit
}

/// The portal loop: a `while`/`loop` statement of the function body whose own
/// statements open a merge point. `find_dispatch_loop_body` accepts the same
/// statement positions, recognising the loop by the dispatch match inside it
/// instead — which is why it cannot be the one to choose that match.
pub(crate) fn portal_loop_body(func_block: &syn::Block) -> Option<&syn::Block> {
    find_portal_loop(func_block).map(|found| found.body)
}

/// A top-level `while`/`loop` whose body contains `jit_merge_point!`.
pub(crate) struct PortalLoop<'a> {
    pub index: usize,
    pub body: &'a syn::Block,
    pub while_cond: Option<&'a syn::Expr>,
}

pub(crate) fn find_portal_loop(func_block: &syn::Block) -> Option<PortalLoop<'_>> {
    func_block
        .stmts
        .iter()
        .enumerate()
        .find_map(|(index, stmt)| {
            let Stmt::Expr(expr, _) = stmt else {
                return None;
            };
            let (body, while_cond) = match expr {
                Expr::While(while_expr) => (&while_expr.body, Some(&*while_expr.cond)),
                Expr::Loop(loop_expr) => (&loop_expr.body, None),
                _ => return None,
            };
            body.stmts
                .iter()
                .any(is_jit_merge_point_macro)
                .then_some(PortalLoop {
                    index,
                    body,
                    while_cond,
                })
        })
}

fn collect_all_matches<'a>(block: &'a syn::Block, out: &mut Vec<&'a syn::ExprMatch>) {
    for stmt in &block.stmts {
        collect_matches_in_stmt(stmt, out);
    }
}

fn collect_matches_in_stmt<'a>(stmt: &'a syn::Stmt, out: &mut Vec<&'a syn::ExprMatch>) {
    match stmt {
        syn::Stmt::Expr(expr, _) => collect_matches_in_expr(expr, out),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                collect_matches_in_expr(&init.expr, out);
            }
        }
        _ => {}
    }
}

fn collect_matches_in_expr<'a>(expr: &'a syn::Expr, out: &mut Vec<&'a syn::ExprMatch>) {
    match expr {
        syn::Expr::Match(m) => out.push(m),
        syn::Expr::While(w) => collect_all_matches(&w.body, out),
        syn::Expr::Loop(l) => collect_all_matches(&l.body, out),
        syn::Expr::Block(b) => collect_all_matches(&b.block, out),
        syn::Expr::If(i) => {
            collect_all_matches(&i.then_branch, out);
            if let Some((_, else_expr)) = &i.else_branch {
                collect_matches_in_expr(else_expr, out);
            }
        }
        _ => {}
    }
}

fn find_match_in_stmt(stmt: &syn::Stmt) -> Option<&syn::ExprMatch> {
    match stmt {
        syn::Stmt::Expr(expr, _) => find_match_in_expr(expr),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                find_match_in_expr(&init.expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn find_match_in_expr(expr: &syn::Expr) -> Option<&syn::ExprMatch> {
    match expr {
        syn::Expr::Match(m) => Some(m),
        syn::Expr::While(w) => {
            for stmt in &w.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Loop(l) => {
            for stmt in &l.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Block(b) => {
            for stmt in &b.block.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::If(i) => {
            for stmt in &i.then_branch.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            if let Some((_, else_expr)) = &i.else_branch {
                return find_match_in_expr(else_expr);
            }
            None
        }
        _ => None,
    }
}

/// `true` iff `stmt`'s expression subtree contains `target_match`.
pub(crate) fn stmt_contains_match(stmt: &Stmt, target_match: &ExprMatch) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_contains_match(expr, target_match),
        Stmt::Local(local) => local
            .init
            .as_ref()
            .map(|init| expr_contains_match(&init.expr, target_match))
            .unwrap_or(false),
        _ => false,
    }
}

pub(crate) fn block_contains_match(block: &Block, target_match: &ExprMatch) -> bool {
    block
        .stmts
        .iter()
        .any(|s| stmt_contains_match(s, target_match))
}

fn expr_contains_match(expr: &Expr, target_match: &ExprMatch) -> bool {
    match expr {
        Expr::Match(m) => std::ptr::eq(m, target_match),
        Expr::While(w) => block_contains_match(&w.body, target_match),
        Expr::Loop(l) => block_contains_match(&l.body, target_match),
        Expr::Block(b) => block_contains_match(&b.block, target_match),
        Expr::If(i) => {
            block_contains_match(&i.then_branch, target_match)
                || i.else_branch
                    .as_ref()
                    .map(|(_, e)| expr_contains_match(e, target_match))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Check if a call expression's function path is a promote call.
///
/// Matches: `promote`, `hint_promote`, `jit::promote`,
/// `majit_metainterp::jit::promote`.
pub(crate) fn is_promote_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "promote" || name == "hint_promote",
        [ns, name] => name == "promote" && ns == "jit",
        [_, ns, name] => name == "promote" && ns == "jit",
        _ => false,
    }
}

/// Check if a call expression's function path is a `jit::assert_not_none`
/// call.
///
/// Matches: `assert_not_none`, `jit::assert_not_none`,
/// `majit_metainterp::jit::assert_not_none`.  Mirrors RPython
/// `rtyper/debug.py ll_assert_not_none` recognition.
pub(crate) fn is_assert_not_none_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "assert_not_none",
        [ns, name] => name == "assert_not_none" && ns == "jit",
        [_, ns, name] => name == "assert_not_none" && ns == "jit",
        _ => false,
    }
}

/// Check if a call expression's function path is a `jit::record_exact_class`
/// call.
///
/// Matches: `record_exact_class`, `jit::record_exact_class`,
/// `majit_metainterp::jit::record_exact_class`.  Mirrors RPython
/// `rlib/jit.py jit.record_exact_class` recognition.
pub(crate) fn is_record_exact_class_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "record_exact_class",
        [ns, name] => name == "record_exact_class" && ns == "jit",
        [_, ns, name] => name == "record_exact_class" && ns == "jit",
        _ => false,
    }
}

#[cfg(test)]
mod find_dispatch_match_tests {
    use super::*;

    fn fn_block(src: &str) -> syn::Block {
        let item: syn::ItemFn = syn::parse_str(&format!("fn f() {{ {src} }}")).unwrap();
        *item.block
    }

    #[test]
    fn matchless_portal_loop_is_not_a_dispatch() {
        let block = fn_block(
            "while pos < len {
                jit_merge_point!(driver, program, pc; state);
                pos = pos + 1;
            }",
        );
        assert!(find_dispatch_match(&block).is_none());
        assert!(portal_loop_body(&block).is_some());
    }

    #[test]
    fn opcode_fetch_match_is_the_dispatch() {
        let cases = [
            ("index", "let opcode = program[pc];"),
            ("get_op", "let opcode = program.get_op(pc);"),
        ];
        for (name, fetch) in cases {
            let block = fn_block(&format!(
                "while pc < program.len() {{
                    jit_merge_point!(driver, program, pc; state);
                    {fetch}
                    match opcode {{
                        0 => {{}},
                        1 => {{}},
                        _ => break,
                    }}
                }}"
            ));
            let found = find_dispatch_match(&block).unwrap_or_else(|| panic!("case {name}"));
            assert_eq!(found.arms.len(), 3, "case {name}");
        }
    }

    #[test]
    fn insn_op_call_match_is_the_dispatch() {
        let block = fn_block(
            "loop {
                jit_merge_point!(driver, program, pc; state);
                let opcode = insn_op(program, pc);
                let next = match opcode {
                    0 => 1,
                    1 => 2,
                    _ => break,
                };
            }",
        );
        let found = find_dispatch_match(&block).expect("dispatch match");
        assert_eq!(found.arms.len(), 3);
    }

    #[test]
    fn insn_a_call_is_not_the_opcode_fetch() {
        let block = fn_block(
            "loop {
                jit_merge_point!(driver, program, pc; state);
                let a = insn_a(program, pc);
                match a {
                    0 => {},
                    1 => {},
                    _ => break,
                }
            }",
        );
        assert!(find_dispatch_match(&block).is_none());
    }

    #[test]
    fn a_setup_match_before_the_loop_does_not_become_the_dispatch() {
        let block = fn_block(
            "let label = match sel { 0 => \"a\", 1 => \"b\", 2 => \"c\", 3 => \"d\", _ => \"e\" };
            while pc < program.len() {
                jit_merge_point!(driver, program, pc; state);
                let opcode = program[pc];
                match opcode {
                    0 => {},
                    _ => break,
                }
            }",
        );
        let found = find_dispatch_match(&block).expect("dispatch match");
        assert_eq!(found.arms.len(), 2);
    }

    #[test]
    fn a_non_opcode_match_is_not_the_dispatch() {
        let cases = [
            (
                "local",
                "match acc {
                    0 => {},
                    1 => {},
                    _ => {},
                }",
            ),
            (
                "program_len",
                "let n = program.len();
                match n {
                    0 => {},
                    1 => {},
                    _ => {},
                }",
            ),
            (
                "program_data_index",
                "match program[pos] {
                    0 => {},
                    1 => {},
                    _ => {},
                }",
            ),
            (
                "let_program_data_index",
                "let byte = program[pos];
                match byte {
                    0 => {},
                    1 => {},
                    _ => {},
                }",
            ),
        ];
        for (name, matched) in cases {
            let block = fn_block(&format!(
                "while pos < len {{
                    jit_merge_point!(driver, program, pc; state);
                    {matched}
                    pos = pos + 1;
                }}"
            ));
            assert!(find_dispatch_match(&block).is_none(), "case {name}");
            assert!(portal_loop_body(&block).is_some(), "case {name}");
        }
    }

    #[test]
    fn match_on_program_index_is_the_dispatch() {
        let block = fn_block(
            "while pc < program.len() {
                jit_merge_point!(driver, program, pc; state);
                match program[pc] {
                    0 => {},
                    _ => break,
                }
            }",
        );
        let found = find_dispatch_match(&block).expect("dispatch match");
        assert_eq!(found.arms.len(), 2);
    }

    #[test]
    fn can_enter_before_merge_is_an_allowed_matchless_prefix() {
        let block = fn_block(
            "while pos < len {
                can_enter_jit!(driver, 0usize, &mut state, program, || {});
                jit_merge_point!(driver, program, pc; state);
                pos = pos + 1;
            }",
        );
        assert!(first_unsupported_pre_merge_stmt(&block).is_none());
    }

    #[test]
    fn assignment_before_merge_is_an_unsupported_matchless_prefix() {
        let block = fn_block(
            "while pos < len {
                pos = pos + 1;
                jit_merge_point!(driver, program, pc; state);
            }",
        );
        assert!(first_unsupported_pre_merge_stmt(&block).is_some());
    }

    #[test]
    fn a_plain_let_before_merge_is_allowed() {
        let block = fn_block(
            "while pos < len {
                let n = 0;
                jit_merge_point!(driver, program, pc; state);
                pos = pos + 1;
            }",
        );
        assert!(first_unsupported_pre_merge_stmt(&block).is_none());
    }

    #[test]
    fn an_effectful_let_before_merge_is_unsupported() {
        let cases = [
            ("mutating", "let ignored = { state.acc += 1; 0 };"),
            ("qualified_call", "let ignored = crate::tick();"),
            ("macro_call", "let ignored = tick!(state);"),
            ("return", "let ignored = if stop { return 7 } else { 0 };"),
            ("indexed", "let ignored = [1, 2][state.pos];"),
        ];
        for (name, stmt) in cases {
            let block = fn_block(&format!(
                "while pos < len {{
                    {stmt}
                    jit_merge_point!(driver, program, pc; state);
                    pos = pos + 1;
                }}"
            ));
            assert!(
                first_unsupported_pre_merge_stmt(&block).is_some(),
                "case {name}"
            );
        }
    }

    #[test]
    fn a_closure_in_a_pre_merge_let_is_allowed() {
        let block = fn_block(
            "while pos < len {
                let ignored = || return 7;
                jit_merge_point!(driver, program, pc; state);
                pos = pos + 1;
            }",
        );
        assert!(first_unsupported_pre_merge_stmt(&block).is_none());
    }
}
