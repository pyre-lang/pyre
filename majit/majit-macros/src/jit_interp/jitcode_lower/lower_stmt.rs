use super::lower_value::struct_type_id_tokens;
use super::*;

/// Joins the refusals accumulated into one `DegradedDispatchArm::reason`.
///
/// Cross-crate contract. `majit_metainterp::REFUSAL_SEPARATOR` must hold the
/// same bytes — a proc-macro crate cannot export a value to its runtime, so the
/// two literals are mirrored rather than shared.
///
/// The drift detector is `both_blockers_are_reported` in
/// `majit-metainterp/tests/jit_interp_degraded_arm_accumulates_refusals.rs`. It
/// takes a two-blocker arm's reason from the registry — minted with THIS
/// literal — and splits it with the runtime's, so a change to either side reads
/// one member where it expects two. It has to be that fixture and not the
/// literal corpus in `degraded_arm_refusal_kind.rs`: a recorded string frozen
/// into a `const` was minted by whatever this literal said on the day it was
/// copied, so it goes stale silently when this side changes.
pub(super) const REFUSAL_SEPARATOR: &str = " || ";

impl<'c> Lowerer<'c> {
    /// If `func` is a registered `residual_writes` mutator, return the
    /// `residual_write_effect_info(...)` expression naming the written fields
    /// and arrays, so the residual call records a write-set `EffectInfo` that
    /// invalidates cached getfields and cached array elements.  `None` for a
    /// plain residual call (empty write-set).  `can_raise` carries the
    /// call policy's extra-effect into the write-set `EffectInfo` so a
    /// `ResidualVoidCannotRaise` mutator keeps `CannotRaise` instead of
    /// silently widening to `CanRaise`.
    ///
    /// A `field` declaration names the field itself; a `field[]` declaration
    /// names the ELEMENTS of the array `field` points at.  The two are separate
    /// caches in `OptHeap` — invalidating the base field hands the next element
    /// load a fresh base, which only helps when the base actually changes — so a
    /// residual that mutates a buffer in place has to say `field[]`.
    pub(super) fn residual_write_effect_info_tokens(
        &self,
        func: &Expr,
        can_raise: bool,
    ) -> Option<TokenStream> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(func)?;
        let writes: Vec<_> = config
            .residual_writes
            .iter()
            .filter(|(segments, _, _, _)| *segments == func_segments)
            .collect();
        let mut layouts: Vec<(syn::Path, Vec<TokenStream>)> = Vec::new();
        // `(key, element_path)` for the `field[]` declarations, deduplicated by
        // key: naming one array from two helpers of the same residual would
        // otherwise repeat its descr in the write set.
        let mut arrays: Vec<(String, &syn::Path)> = Vec::new();
        for (_, path, field, writes_elements) in writes {
            // A write set names the descriptors an opaque call may invalidate,
            // and a descriptor belongs to the struct that declares the field.
            // Resolving the entry the same way an access site does is what
            // makes the two name one descriptor: entries written against
            // different structs that share an inlined base collapse onto it,
            // and the grouping below merges them.
            let path = config.declaring_struct(path, &field.to_string());
            let struct_last = path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_default();
            let key = format!("{}::{}", struct_last, field);
            if *writes_elements {
                let Some((_, _, element_path)) = config.array_fields.get(&key) else {
                    return Some(
                        syn::Error::new(
                            field.span(),
                            format!(
                                "jit_interp: residual_writes `{key}[]` declares a write to the \
                                 elements of an array field, but `{key}` is not declared in \
                                 `array_fields`, so there is no element type to mint the written \
                                 array descr from"
                            ),
                        )
                        .to_compile_error(),
                    );
                };
                if !arrays.iter().any(|(seen, _)| *seen == key) {
                    arrays.push((key, element_path));
                }
                continue;
            }
            let fields = if let Some((_, fields)) = layouts.iter_mut().find(|(p, _)| *p == path) {
                fields
            } else {
                layouts.push((path.clone(), Vec::new()));
                &mut layouts.last_mut().unwrap().1
            };
            // Same declared width the getfield/setfield lowering registers, so
            // this write-EI rebuild mints the field descr the reads resolve to
            // rather than a machine-word twin of it.
            //
            // A pointer field stays a pointer field whichever map declares it.
            // `ref_fields` names the pointee of a single-object link; an array
            // declaration names the element type of a buffer the field points
            // at.  Both read back through `getfield_gc_r`, and the array base's
            // own layout registration describes the field as a pointer word.
            // Asking only `ref_fields` here therefore describes the SAME member
            // as an eight-byte signed integer at a second producer, and
            // `get_field_descr` resolves that disagreement by keeping whichever
            // producer reached its `(struct, fieldname)` slot first — so the
            // read's descr and the write set's are the same object only by
            // registration order.
            let member = syn::Member::Named(field.clone());
            let (is_ref, __fsize, __fsigned, __fcheck) = match config.array_fields.get(&key) {
                Some((_, _, element_path)) => (
                    true,
                    quote! { ::core::mem::size_of::<usize>() },
                    quote! { false },
                    // The array base's witness, and it claims the same thing:
                    // the element's WIDTH, not its name.  Reading through the
                    // field rather than naming a pointer type accepts
                    // `*const T` as well as `*mut T`, and `transmute` accepts a
                    // `#[repr(transparent)]` wrapper over the declared type —
                    // which has the declared type's size and, by the definition
                    // of `repr(transparent)`, its representation.  A
                    // declaration that drifted in width (`=> u16` over a
                    // `*mut u8`) still fails to compile; one that keeps the
                    // width and flips the sign does not, the same gap the base
                    // witness documents.
                    quote! {
                        const _: fn(&#path) = |__s| {
                            let _: #element_path =
                                unsafe { ::core::mem::transmute(*__s.#field) };
                        };
                    },
                ),
                None => {
                    let (size, signed, check) =
                        super::lower_vable::field_scalar_tokens(config, &key, &path, &member);
                    (config.ref_fields.contains_key(&key), size, signed, check)
                }
            };
            // `descr.py` `get_type_flag`: an `f64` field is the float flag,
            // so the write-set descr matches the getfield descr.
            let is_float =
                config.array_fields.get(&key).is_none() && config.float_fields.contains(&key);
            fields.push(quote! {
                {
                    #__fcheck
                    (
                        ::core::mem::offset_of!(#path, #field),
                        #is_ref,
                        stringify!(#field),
                        #__fsize,
                        #__fsigned,
                        #is_float,
                    )
                }
            });
        }
        let layouts: Vec<_> = layouts
            .iter()
            .map(|(struct_path, fields)| {
                // The struct's own gc-kind — the same id the getfield/setfield
                // lowering uses, so the write-EI rebuilds the SAME parent
                // SizeDescr identity.
                let gc_managed = config.struct_gc_kind_is_managed(struct_path);
                let tid = struct_type_id_tokens(struct_path, gc_managed);
                // An inlined base's fields, exactly as the access sites list
                // them. This layout is a second producer of the same struct's
                // spec, and `index_in_parent` comes from the positions in it:
                // a write set that named only the struct's own fields would
                // number the first of them zero, the slot the base's first
                // field holds for the same object, and the two would share a
                // cache entry.
                let (prefix_fields, prefix_witness) =
                    config.prefix_field_entries_tokens(struct_path);
                quote! {
                    {
                        #prefix_witness
                        (
                            ::core::mem::size_of::<#struct_path>(),
                            #tid,
                            #gc_managed,
                            &[#(#prefix_fields,)* #(#fields),*],
                        )
                    }
                }
            })
            .collect();
        let arrays: Vec<_> = arrays
            .iter()
            .map(|(_, element_path)| {
                // `writes_array_descr_by_shape` keys on `is_item_signed`, so
                // this has to derive it exactly as the element read and write
                // do (`add_raw_int_array_descr_signed` in
                // `lower_ref_binding_array_read` / `_write`).  A literal here
                // mints a shape no unsigned-element read ever produces, and
                // the declaration then silently stops invalidating the load it
                // names — the optimizer keeps serving the element it cached
                // from before the mutating call.
                quote! {
                    (
                        ::core::mem::size_of::<#element_path>(),
                        (<#element_path>::MIN as i128) < 0,
                    )
                }
            })
            .collect();
        Some(quote! {
            // Each layout carries the struct's own gc-kind, matching the
            // getfield/setfield lowering so the write-EI rebuilds the SAME
            // parent SizeDescr identity the getfield reads back.
            majit_metainterp::residual_write_effect_info(
                &[#(#layouts),*],
                &[#(#arrays),*],
                #can_raise,
            )
        })
    }

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt) -> Option<()> {
        // A statement that lowers contributes no blockers. An inner attempt can
        // refuse and stash its reason before an outer strategy succeeds on the
        // same statement — `lower_local` failing into `lower_stmt_fallback`'s
        // inert arm is the shape — and reporting that stash would name a
        // statement that played no part in the refusal. Drop back to the depth
        // this call started at, so only genuinely unlowered statements carry
        // reasons upward.
        let carried_on_entry = self.nested_failure_reasons.len();
        let lowered = self.lower_stmt_dispatch(stmt);
        if lowered.is_some() {
            self.nested_failure_reasons.truncate(carried_on_entry);
        }
        lowered
    }

    fn lower_stmt_dispatch(&mut self, stmt: &Stmt) -> Option<()> {
        // Consume the one-shot tail marker (see `Lowerer::inline_arm_tail_stmt`).
        // Taking it here rather than reading it is what confines the in-arm
        // `return` lowering to statement-tail position: every nested
        // `lower_stmt` — an `if` body, a loop body — observes `false`.
        let is_arm_tail = std::mem::take(&mut self.inline_arm_tail_stmt);
        match stmt {
            Stmt::Local(local) => {
                if let Some(()) = self.lower_local(local) {
                    return Some(());
                }
                self.lower_stmt_fallback(stmt, "local")
            }
            Stmt::Expr(expr, _) => {
                if let Expr::Return(ret) = expr {
                    return self.lower_return_stmt(ret, is_arm_tail);
                }
                if let Expr::Continue(cont) = expr {
                    // The only `continue` this lowering can honour is the one
                    // that means the dispatch back-edge, because the jump it
                    // emits targets `dispatch_loop_label` and there is no other
                    // label in scope here.  Two spellings reach this arm naming
                    // a different loop, and both used to be answered with that
                    // same jump -- or, with no dispatch label at all, with a
                    // bare `Some(())` that emitted nothing and dropped the
                    // transfer:
                    //
                    //   * a `continue` inside a loop written in the arm body.
                    //     Its direct and `if`-branch spellings are taken by
                    //     `lower_loop_stmt` / `lower_loop_if` with that loop's
                    //     own labels, so what arrives here came through a
                    //     `match` arm or a nested block.  `lower_loop_body_block`
                    //     clears the dispatch label for exactly this reason, so
                    //     the `None` below is what such a `continue` sees.
                    //   * a labelled `continue 'l`, which crosses nested loops
                    //     and names a loop this lowering has no label for.
                    //
                    // Both are refused instead of mistargeted.  The refusal
                    // declines the whole body, so the arm keeps running in the
                    // interpreter rather than jumping to the wrong head or
                    // falling through with the transfer gone.
                    if cont.label.is_none()
                        && let Some(label) = self.dispatch_loop_label.clone()
                    {
                        self.emit_jump(&label);
                        return Some(());
                    }
                    // `stmt_contains_loop_control` fires on a bare `continue`,
                    // so the fallback refuses and records the reason.
                    return self.lower_stmt_fallback(stmt, "expr");
                }
                if let Some(()) = self.lower_expr_stmt(expr) {
                    return Some(());
                }
                self.lower_stmt_fallback(stmt, "expr")
            }
            Stmt::Macro(stmt_macro) => {
                // jtransform.py handle_jit_marker__loop_header —
                // a `can_enter_jit!()` call at the user's source-level
                // back-edge (interp_jit.py inside `jump_absolute`'s
                // backward-jump branch) lowers to `loop_header(jd.index)`
                // at the SAME source position.  Per-arm emission at the
                // dispatch JitCode level (post-INLINE_CALL) would over-
                // emit on every arm execution including forward-jump
                // path; emitting at the call site inside the arm body
                // sub-JitCode makes the LH op execute only when control
                // reaches the conditional that contains can_enter_jit!.
                //
                // Only fire when this Lowerer is producing the dispatch
                // arm body sub-JitCode (where the surrounding
                // `__dispatch_jitcode_<fn>` provides `__jdindex` in
                // scope).  For the per-arm trace JitCode path (whose
                // surrounding fn has no `__jdindex`) the recognition
                // falls through to `None` and the body lowering aborts
                // — pyre's per-arm trace JitCode is a
                // TODO: not present in RPython, so
                // omitting `loop_header` there is consistent with
                // upstream's single-JitCode model.
                // Allow can_enter_jit! in the dispatch JitCode body
                // (both at the dispatch level and inside arm sub-JitCodes).
                // __jdindex is in scope in both contexts.
                let path_str = stmt_macro
                    .mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "can_enter_jit"
                    || path_str.ends_with("::can_enter_jit")
                    || path_str == "jit_loop_header"
                    || path_str.ends_with("::jit_loop_header")
                {
                    self.emit_op(
                        OpMeta::linear(OpKind::LoopHeader, vec![], vec![]),
                        quote! {
                            // jtransform.py:1716 c_index = Constant(jd.index, ...);
                            // __jdindex is the runtime parameter of the
                            // enclosing `__dispatch_jitcode_<fn>(__asm,
                            // __jdindex: i64)` and remains in scope of
                            // the arm body sub-builder block.
                            __builder.loop_header(__jdindex);
                        },
                    );
                    return Some(());
                }
                None
            }
            Stmt::Item(_) => None,
        }
    }

    /// Lower a `return` statement to its typed return terminator —
    /// `int_return` / `ref_return` / `float_return` by operand kind, or
    /// `void_return` for a bare `return;`.
    ///
    /// **Language-gap adaptation, not a parity fix.**  `interp_jit.py:95-100`
    /// is upstream's single return point: the portal funnels every exit from
    /// the bytecode loop through that one `return`, because its opcode
    /// implementations raise `Return` rather than returning in place.  That
    /// shape never produces a `return` inside a dispatch arm, so there is no
    /// upstream construct to mirror here.  A Rust interpreter instead
    /// idiomatically returns straight out of a `match` arm, and the terminator
    /// emitted here is what that spelling corresponds to — the same single
    /// return point upstream arrives at by unwinding.
    ///
    /// Accepted **only** in statement-tail position of an inline dispatch arm
    /// body (`is_arm_tail`, from `Lowerer::inline_arm_tail_stmt`).  A `return`
    /// anywhere else is rejected with `None`, which rolls the arm back to the
    /// sub-JitCode / abort path.  Rejecting is mandatory rather than tidy: this
    /// site exists because lowering a `return`'s operand while dropping its
    /// control transfer let the arm fall through to the dispatch back-edge and
    /// re-enter the loop at the terminal pc, so a walk could only ever end by
    /// closing a loop.  Reproducing that one level down — for a `return` nested
    /// in an `if` — would reintroduce exactly the defect.
    fn lower_return_stmt(&mut self, ret: &syn::ExprReturn, is_arm_tail: bool) -> Option<()> {
        if !is_arm_tail {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt rejected `return`: only an inline dispatch \
                     arm body's tail statement can carry a return terminator, so this \
                     one's control transfer cannot be lowered here: {}",
                    quote!(#ret)
                );
            }
            return None;
        }
        // A bare `return;` has no operand and lowers to `void_return`.  An
        // operand that fails to lower must NOT fall back to `void_return` — it
        // would return the wrong value — so `?` rejects and the caller rolls
        // back whatever ops the partial operand lowering emitted.
        let binding = match ret.expr.as_deref() {
            Some(expr) => Some(self.lower_value_expr(expr)?),
            None => None,
        };
        let (reads, emitter) = super::dispatch::typed_return_terminator(binding);
        self.emit_op(OpMeta::terminal(reads), emitter);
        Some(())
    }

    /// Last resort for a statement no lowering arm accepted, in the
    /// state-field dispatch body (`config` present).
    ///
    /// A statement with no observable effect — it neither writes jit
    /// state nor touches storage/heap/user locals — is dropped from the
    /// jitcode (e.g. `pc += 1`: the dispatch loop manages the pc
    /// register itself).
    ///
    /// Anything else (state-field writes, residual side effects,
    /// bindings later statements may observe) cannot be expressed and
    /// fails as unsupported (`None`). The codewriter lowers a graph op
    /// exactly or rejects it — `jtransform.py` `rewrite_operation` raises
    /// for operations it cannot transform — rather than emitting a
    /// runtime abort for part of a body. Returning `None` lets the caller
    /// degrade cleanly: `try_inline_dispatch_arm` rolls back the partial
    /// emission and the sub-JitCode entry returns `None` too, so the arm
    /// runs in the interpreter instead of compiling to a trace that
    /// aborts mid-record.
    fn lower_stmt_fallback(&mut self, stmt: &Stmt, what: &str) -> Option<()> {
        self.config?;
        // A statement containing a `return` is never inert, however little jit
        // state it touches.  `if flag { return 0; }` writes no state, reads no
        // storage and calls nothing, so the purity test below would classify it
        // inert and drop it — silently deleting the control transfer and
        // letting the arm fall through to the dispatch back-edge.  That is the
        // exact defect `lower_return_stmt` exists to fix, so it is refused here
        // rather than reproduced one level down.  Only a `return` in arm-body
        // tail position lowers; it never reaches this fallback.
        if stmt_contains_return(stmt) {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt rejected ({what}): statement encloses a \
                     `return` that cannot be lowered in place: {}",
                    quote!(#stmt)
                );
            }
            self.record_body_failure("encloses a `return` that cannot be lowered in place", stmt);
            return None;
        }
        // `break` and `continue` are control transfers for exactly the same
        // reason `return` is, and the purity test below cannot see either:
        // `expr_modifies_jit_state` reports `false` for both, so a statement
        // whose only effect is `if cond { continue; }` writes no state, touches
        // no storage and calls nothing — it is scored inert, dropped, and the
        // arm falls through to the dispatch back-edge with the transfer gone.
        //
        // The observable symptom is one extra loop iteration.  A terminal arm
        // spelled `{ store; break }` is an `Expr::Block`, so
        // `classify_arm_body` does not reach `ArmPattern::Halt` —
        // `is_break_expr` requires the body to be exactly `break` — and the arm
        // is lowered, putting its tail `break` on this path.  That is the
        // defect, not a hypothetical.
        //
        // Measured coverage, `MAJIT_MACRO_DEBUG` over all 13 examples: this
        // guard fires exactly four times — tiny2, tiny3, braininterp and
        // dualtape, once each — and zero times in `examples/cel` or
        // `examples/tl`. Those two crates do not witness this guard.
        // Rebuilding with the guard disabled (confirmed by zero firings)
        // leaves every example's output unchanged, because at each of the four
        // sites the same arm body also yields `unsupported` — `if target <= pc`
        // in tiny2/tiny3, `find_matching_open` in braininterp/dualtape — which
        // refuses the arm on its own.  The guard is therefore defensive: it
        // covers a body reaching this path with no co-occurring refusal, a
        // shape the current corpus does not contain.
        if stmt_contains_loop_control(stmt) {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt rejected ({what}): statement encloses a \
                     `break`/`continue` that cannot be lowered in place: {}",
                    quote!(#stmt)
                );
            }
            self.record_body_failure(
                "encloses a `break`/`continue` that cannot be lowered in place",
                stmt,
            );
            return None;
        }
        // `hint_fresh_virtualizable` is the fourth member, and the purity test
        // is blind to it in the same way: the hint reads nothing, writes
        // nothing and calls nothing, so a statement whose only content is the
        // hint is scored inert and dropped.  Dropping it is what the previous
        // lowering did on purpose -- and that is the defect, because the hint
        // is not inert.  Upstream records `fresh_virtualizable` in
        // `Transformer.vable_flags` and reads it back in `is_vable_getfield` /
        // `is_vable_setfield`, which then answer `False` so the following field
        // accesses stay off the `getfield_vable_*` / `setfield_vable_*` path.
        // This lowering has no carrier for that flag, so it cannot honour the
        // hint; suppressing it leaves every following access on the vable path,
        // which is the opposite of what was asked and is invisible in the
        // emitted jitcode.  Refuse the statement rather than answer it wrongly.
        //
        // `hint_access_directly` is deliberately not covered: upstream drops it
        // at the same site, and the flag it records is consulted for
        // `fresh_virtualizable` alone.
        if stmt_contains_fresh_virtualizable_hint(stmt) {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt rejected ({what}): statement encloses a \
                     `hint_fresh_virtualizable` this lowering has no flag to carry: {}",
                    quote!(#stmt)
                );
            }
            self.record_body_failure(
                "encloses a `hint_fresh_virtualizable` this lowering has no flag to carry",
                stmt,
            );
            return None;
        }
        // A write to a green is the third member of the family above, and the
        // purity test is blind to it for the same reason it is blind to
        // `break`: `stmt_modifies_jit_state` scores writes to `state.*`, and a
        // green is a caller local, not state.  So `pc += 1` writes no state,
        // references no storage and calls nothing — scored inert, dropped, and
        // the advance is gone from the arm's jitcode.
        //
        // Its symptom is worse than the other two.  `break`/`continue` cost one
        // extra loop iteration; a dropped green-pc advance costs nothing
        // visible at all until the walk resumes: the arm emits no
        // `BC_INT_ADD`, the green pc advances only by the dispatch prologue's
        // read of the opcode byte, and a multi-byte instruction therefore
        // resumes on its own operand — which the interpreter decodes as the
        // next opcode.  The trace is well-formed, records, compiles, and
        // computes a wrong answer.
        //
        // The sub-JitCode arm path is where this lands, because a green write
        // has exactly one channel out of a sub-JitCode: the `BC_INT_RETURN`
        // that `try_generate_jitcode_pc_return_body_with_caller_bindings`
        // emits for an arm whose advance is its *final* statement.  An arm
        // that spells the advance mid-body (`let r = program[pc]; pc += 1;
        // residual(r);`) fails `arm_is_pure_pc_advance`, gets the void
        // `inline_call` instead, and has no channel at all.
        //
        // Measured coverage, `MAJIT_MACRO_DEBUG` per crate over all 13
        // examples (160 dispatch arms): three firings — `tl::ROLL`,
        // `tlc::ROLL` and `tlr::ALLOCATE`, one each, and none anywhere else.
        // All three are inert today, because each arm ALSO yields
        // `unsupported` for a co-occurring statement —
        // `storage_roll(state.stack.as_mut_ptr() as usize, ...)`,
        // `tlc_roll(...)` and `state.regs = vec![0; n]` — so each degrades to
        // an abort stub on its own and its dropped advance never runs.
        // Confirmed at runtime, not inferred from the census: `MAJIT_LOG=1
        // cargo test -p tl|tlc|tlr --features cranelift` records exactly `TlState::ROLL`,
        // `TlcState::ROLL` and `TlrState::ALLOCATE` as degraded.
        //
        // So this guard is defensive, like the `break` one above — but it
        // brakes a change already in flight rather than a hypothesis.  The two
        // `ROLL` arms are inert only because the residual's
        // `state.stack.as_mut_ptr()` argument has no lowering; give it one and
        // both become real sub-JitCodes whose advance is dropped, which is a
        // trace that records, compiles, and returns a wrong answer.
        if let Some(green) = self
            .config
            .map(green_idents)
            .and_then(|greens| stmt_writes_green(stmt, &greens))
        {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt rejected ({what}): statement writes the green \
                     `{green}`, which this path cannot carry back to the caller: {}",
                    quote!(#stmt)
                );
            }
            // The reason says its own scope out loud.  Lowering genuinely stops
            // at this statement, and a green advance is typically statement 2 of
            // 3, so this refusal is the arm's OUTERMOST blocker rather than its
            // only one.  `record_body_failure` accumulates and the caller keeps
            // walking the remaining statements for their reasons, so the ones
            // behind the stop (`storage_roll(…)`, `tlc_roll(…)`,
            // `state.regs = vec![0; n]`) are reported after it instead of being
            // displaced by it — a reader watching `MAJIT_LOG` for their own
            // blocker to disappear would otherwise read this refusal alone as
            // progress.
            self.record_body_failure(
                "writes a green this lowering path cannot carry back to the caller \
                 (lowering stopped at this statement; any further blockers follow)",
                stmt,
            );
            return None;
        }
        // Drop only genuinely inert statements: no jit-state write, no
        // storage/user-local reference, AND no call.  A residual call
        // (e.g. `record_event();` or an unrolled `for _ in 0..4 {
        // side_effect(); }` body) that no lowering arm consumed cannot be
        // proven pure, so silently dropping it would delete a side effect
        // the interpreter performs.  Returning `None` aborts lowering so
        // the arm runs interpreted instead — jtransform.py rejects ops it
        // cannot transform rather than deleting them.
        let inert = !self.stmt_modifies_jit_state(stmt)
            && !self.stmt_touches_storage(stmt)
            && !self.stmt_contains_call(stmt);
        if inert {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt silent-skip ({what}): {}",
                    quote!(#stmt)
                );
            }
            return Some(());
        }
        if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
            eprintln!(
                "[majit-macro] lower_stmt unsupported ({what}): {}",
                quote!(#stmt)
            );
        }
        self.record_body_failure("has a statement the lowerer cannot express", stmt);
        None
    }

    /// Record why this body's lowering refused; the first writer keeps the head.
    ///
    /// The first entry wins because the failure that matters most is the one
    /// that stopped lowering. It is no longer true that later statements go
    /// unreached: the caller deliberately keeps walking them for their reasons,
    /// discarding whatever they lower, so the rest of the string enumerates the
    /// blockers behind the stop.
    ///
    /// Reasons do NOT cross a nested lowerer. `lower_control` and
    /// `lower_value` build a child `Lowerer` with its own
    /// `body_failure_reason: None` and merge back `next_reg`, `next_label`,
    /// `statements` and `op_metadata` — never this field. A blocker found
    /// inside a nested block is therefore recorded here and then dropped with
    /// the child, which is why an arm whose inner statement is the interesting
    /// one (braininterp's and dualtape's `b']'`, whose real blocker is
    /// `find_matching_open(program, pc)`) reports only its outer refusal.
    /// Measured from the built artifacts, not inferred:
    /// `strings target/debug/deps/<crate>-<hash> | rg 'arm body '` shows those
    /// two crates with a single-member reason. That filter is only trustworthy
    /// while the spelling below stays newline-free: `strings(1)` ends a run at a
    /// newline, so a reason carrying one is read back as two literals, and the
    /// second — having lost the `arm body ` prefix — is invisible to the census
    /// this comment cites.
    ///
    /// The statement spelling is carried because the classification alone does
    /// not say which lowering rule is missing — "cannot express" reads the same
    /// for `find_matching_open(program, pc)` (needs a slice argument) and for
    /// `state.stack[n] = v` (needs a computed-index store), and those are
    /// different pieces of work. Whitespace runs are collapsed to single spaces
    /// BEFORE the truncation, so a whole `if` block cannot bake a multi-line
    /// literal into the binary and the length bound counts the characters that
    /// actually reach it.
    ///
    /// The truncation marker is ASCII `...` on purpose: a `…` splits the
    /// literal in `strings(1)` output, which is how these reasons get read out
    /// of a built artifact.
    /// Refusals accumulate; the FIRST one stays the head of the string.
    ///
    /// The head is byte-identical to what this produced before accumulation
    /// existed, which is what lets every example crate's `.contains()` snippet
    /// assertion and every landed `RefusalKind` pin keep matching untouched.
    ///
    /// The separator is ASCII and spaced for the same reason the truncation
    /// marker is ASCII `...`: these strings are read out of a built artifact
    /// with `strings(1)`, and a multi-byte separator splits the literal there.
    pub(super) fn record_body_failure(&mut self, what: &str, stmt: &Stmt) {
        const MAX_SPELLING: usize = 80;
        // A braced group stringifies across several lines, and `strings(1)`
        // ends a run at a newline: an un-normalised spelling reaches the
        // artifact as two independent literals whose second half has lost the
        // `arm body ` prefix every reader of these reasons matches on. Collapse
        // before truncating, so the bound counts the characters that survive
        // into the binary rather than characters the reader will never see.
        let mut spelling = quote!(#stmt)
            .to_string()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if spelling.chars().count() > MAX_SPELLING {
            spelling = spelling.chars().take(MAX_SPELLING).collect::<String>() + "...";
        }
        let entry = format!("arm body {what}: {spelling}");
        match &mut self.body_failure_reason {
            // Deduplicated, and not cosmetically: the nested lowerers re-visit
            // statements, so without this one blocker can be recorded twice and
            // a count-based reader would take the repeat for a second
            // mechanism.
            Some(existing) => {
                if !existing.split(REFUSAL_SEPARATOR).any(|seen| seen == entry) {
                    existing.push_str(REFUSAL_SEPARATOR);
                    existing.push_str(&entry);
                }
            }
            None => self.body_failure_reason = Some(entry),
        }
    }

    pub(super) fn lower_local(&mut self, local: &Local) -> Option<()> {
        // `let x: i32 = 1;` parses as a `Pat::Type` wrapping the very
        // `Pat::Ident` that `let x = 1;` produces. The annotation carries no
        // lowering information — the binding kind comes from the initialiser
        // via `lower_value_expr` either way — so unwrap it and lower the
        // annotated spelling exactly like the bare one.
        let mut pat = &local.pat;
        while let Pat::Type(pat_type) = pat {
            pat = &*pat_type.pat;
        }
        let Pat::Ident(pat_ident) = pat else {
            return None;
        };
        let init = local.init.as_ref()?;

        // Try normal lowering
        if let Some(binding) = self.lower_value_expr(&init.expr) {
            // When a stack pop is lowered to a JitCode register, also emit a
            // Rust `let` binding so that subsequent un-lowered code (e.g.,
            // complex expressions referencing the variable) can still compile.
            // The value is 0 — only the JitCode register carries the real
            // runtime value, but this prevents "cannot find value" errors.
            if binding.depends_on_stack {
                let ident = &pat_ident.ident;
                self.emit_aux(quote! { let #ident: i64 = 0; });
            }
            self.bindings.insert(pat_ident.ident.to_string(), binding);
            return Some(());
        }

        // Config-aware: runtime constant (expression not touching storage).
        //
        // Slice ε.3 fail-closed: ALSO refuse this fallback when the init
        // expression references any name already bound in `self.bindings`.
        // The fallback emits the original `let X = <init_expr>;` line as
        // verbatim Rust into the surrounding `__builder` block scope, then
        // a `__builder.load_const_i_value(reg, X as i64)`.  That contract
        // assumes `init_expr` is a true compile-time constant whose
        // identifiers (if any) are Rust types / `const` items / module
        // paths — NOT JIT-level bindings (`program` Ref / `pc` Int /
        // arm-pattern bound names) which are not in scope at the
        // surrounding Rust scope.  Without this guard, dispatch arm
        // sub-JitCode bodies that contain unrecognised method calls on
        // a parent binding (`program.get_operand(pc - 1)` with no
        // `Program::get_operand` call policy registered, say) would
        // emit verbatim Rust referencing `program`/`pc` in the
        // `__sub_builder` block — failing to compile.  Returning `None`
        // here triggers the dispatch arm's `None` branch which substitutes
        // an `abort_permanent()` sub-JitCode (see `lower_dispatch_chain`).
        if self.config.is_some()
            && !self.expr_touches_storage(&init.expr)
            && !self.expr_references_any_binding(&init.expr)
        {
            let reg = self.alloc_reg();
            let ident = &pat_ident.ident;
            let init_expr = &init.expr;
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                quote! {
                    let #ident = #init_expr;
                    __builder.load_const_i_value(#reg, #ident as i64);
                },
            );
            self.bindings.insert(
                ident.to_string(),
                Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                },
            );
            return Some(());
        }

        None
    }

    /// Walk `expr` and return `true` if any single-segment `Expr::Path`
    /// references a name bound in `self.bindings`, **excluding** names
    /// shadowed by an inner `let` in the same expression.  Mirrors the
    /// recognition core of `collect_arm_caller_locals` but stops at the
    /// first match — used as a fail-closed gate inside `lower_local`'s
    /// runtime-constant fallback.
    ///
    /// Scope tracking: PyPy `flowspace` produces distinct flowgraph
    /// variables per lexical scope.  Pyre's probe approximates this by
    /// pushing a fresh scope frame on entering an `ExprBlock` and
    /// popping on exit; `let X = ...` inside the block adds X to the
    /// innermost frame.  An ident is "locally bound" if any frame in
    /// the stack contains it, so the inner `let pc = 42; pc + 1` shape
    /// correctly suppresses the outer `pc` parent-binding match.
    fn expr_references_any_binding(&self, expr: &Expr) -> bool {
        use syn::visit::Visit;
        struct BindingProbe<'a> {
            bindings: &'a HashMap<String, Binding>,
            hit: bool,
            /// Stack of per-block local-binding sets (innermost on top).
            scope_stack: Vec<HashSet<String>>,
        }
        impl BindingProbe<'_> {
            fn is_locally_bound(&self, name: &str) -> bool {
                self.scope_stack.iter().any(|s| s.contains(name))
            }
        }
        impl<'ast> Visit<'ast> for BindingProbe<'_> {
            fn visit_expr_path(&mut self, p: &'ast ExprPath) {
                if self.hit || p.qself.is_some() || p.path.segments.len() != 1 {
                    return;
                }
                let seg = &p.path.segments[0];
                if !seg.arguments.is_none() {
                    return;
                }
                let name = seg.ident.to_string();
                if self.is_locally_bound(&name) {
                    return;
                }
                if self.bindings.contains_key(&name) {
                    self.hit = true;
                }
            }
            fn visit_expr_field(&mut self, ef: &'ast syn::ExprField) {
                self.visit_expr(&ef.base);
            }
            fn visit_expr_method_call(&mut self, mc: &'ast ExprMethodCall) {
                self.visit_expr(&mc.receiver);
                for arg in &mc.args {
                    self.visit_expr(arg);
                }
            }
            fn visit_block(&mut self, b: &'ast Block) {
                // Cover every `Block` traversal — explicit `{ ... }`
                // (`ExprBlock`'s default impl forwards here), if/else
                // branches (`ExprIf::then_branch` / `else_branch`),
                // while / loop / for bodies — not just the explicit
                // block expression form.  Each lexical block pushes a
                // fresh scope frame so inner `let X = ...` shadows the
                // parent binding inside that block only.
                self.scope_stack.push(HashSet::new());
                for stmt in &b.stmts {
                    self.visit_stmt(stmt);
                    if self.hit {
                        break;
                    }
                }
                self.scope_stack.pop();
            }
            fn visit_expr_match(&mut self, em: &'ast ExprMatch) {
                self.visit_expr(&em.expr);
                for arm in &em.arms {
                    if self.hit {
                        break;
                    }
                    // Each match arm introduces a scope: pattern-bound
                    // names shadow outer bindings inside the arm body.
                    // Mirrors `flowspace`'s SpaceOperation scope per
                    // match arm.
                    let mut arm_scope = HashSet::new();
                    collect_pat_bound_idents(&arm.pat, &mut arm_scope);
                    self.scope_stack.push(arm_scope);
                    if let Some((_, guard)) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                    self.scope_stack.pop();
                }
            }
            fn visit_local(&mut self, local: &'ast Local) {
                // Visit init RHS BEFORE adding the bound name so the
                // init expression's references are still probed against
                // outer scope (`let X = X + 1` at scope entry uses the
                // outer X for the RHS).
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    if let Some((_, diverge)) = &init.diverge {
                        self.visit_expr(diverge);
                    }
                }
                // All bindings produced by the pattern enter the
                // innermost scope frame.  `let (a, b) = ...;`,
                // `let Foo { x } = ...;`, `let A(y) | B(y) = ...;`
                // — each pattern shape contributes its bound names.
                // Mirrors `flowspace`'s SpaceOperation per
                // pattern-extraction step.
                if let Some(top) = self.scope_stack.last_mut() {
                    collect_pat_bound_idents(&local.pat, top);
                }
            }
        }
        let mut probe = BindingProbe {
            bindings: &self.bindings,
            hit: false,
            // Seed with one root frame so `visit_local` inside the
            // top-level expression (no enclosing block) can still
            // record bindings.
            scope_stack: vec![HashSet::new()],
        };
        probe.visit_expr(expr);
        probe.hit
    }

    /// RPython jtransform.py:923 `_rewrite_op_setfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name = value` and emits vable_setfield JitCode.
    pub(super) fn lower_conditional_call(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = condition, args[1] = func path, args[2..] = function arguments
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("conditional_call does not support more than 4 arguments");
        }
        let cond_binding = self.lower_value_expr(args[0])?;
        let cond_reg = cond_binding.reg;
        // RPython make_three_lists: tag each arg with its kind (int/ref).
        let mut typed_arg_tokens = Vec::new();
        // cond_reg is Int per the conditional_call argcode prefix.
        let mut arg_regs: Vec<Register> = vec![Register::int(cond_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                // jtransform.py:1668: float → raise Exception
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // `conditional_call!` always lowers to a void residual_call.
        // Default to `ResidualVoidWrapped` for `Infer` so the
        // analyzer-absent CanRaise slot is the lowering's static slot;
        // the runtime helper-policy lookup overrides this for callees
        // whose flavor turns out otherwise.
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call!",
            crate::jit_interp::CallPolicyKind::ResidualVoidWrapped,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("conditional_call! helper policy {policy:?} has no direct-call result kind");
        };
        if result_kind != CallResultKind::Void {
            panic!("conditional_call! requires a void-return helper policy, got {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call!");
        // `call.py getcalldescr`:
        //   if loopinvariant:
        //       assert not NON_VOID_ARGS, ("arguments not supported for "
        //                                  "loop-invariant function!")
        // The canonical `call_loopinvariant_*_canonical_via_target`
        // builders enforce the same invariant via `arg_regs.is_empty()`
        // (`jitcode/assembler.rs`), but the cond_call helper
        // dispatch routes through `conditional_call_ir_v_typed_args`
        // which doesn't share that assert. Mirror the check here so
        // a `conditional_call!(cond, loop_invariant_helper, arg)`
        // panics at expansion time instead of silently registering a
        // bytecode shape RPython would reject at calldescr build.  In
        // `Infer` mode the slot is decided at runtime from `__policy`,
        // so the static check only fires when the macro-time default
        // resolves to LoopInvariant — explicit policy paths preserve
        // the original eager assert.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_policy_check(func_args.is_empty()))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
            None,
            func_args.len(),
        );
        self.emit_op(
            OpMeta::linear(OpKind::Call, arg_regs, vec![]),
            quote! {
                #register_target
                __builder.conditional_call_ir_v_typed_args(__fn_idx, #cond_reg, &[#(#typed_arg_tokens),*]);
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)` for the selected calldescr.
        // In inferred mode the physical BC_LIVE is guarded by the same
        // helper-policy byte that selects the calldescr slot, preserving
        // PyPy's cannot-raise / loop-invariant no-marker shape.
        if is_inferred {
            let condition = inferred_policy_live_condition(func_path, &[1]);
            self.emit_op(
                OpMeta::live_marker_if(condition),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    /// RPython jtransform.py — `rewrite_op_jit_conditional_call_value`.
    ///
    /// Recognizes `conditional_call_elidable!(value, func, args...)` and emits
    /// the canonical `conditional_call_value_ir_{i,r}` builder entrypoint.
    pub(super) fn lower_conditional_call_elidable(&mut self, expr: &Expr) -> Option<Binding> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call_elidable" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        let value_binding = self.lower_value_expr(args[0])?;
        let value_reg = value_binding.reg;
        // jtransform.py:1668: value itself must not be float
        if matches!(value_binding.kind, BindingKind::Float) {
            panic!("Conditional call does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        // value_reg is Int or Ref per the conditional_call_value_ir_{i|r} arm.
        let value_kind = value_binding.kind;
        let mut arg_regs: Vec<Register> = vec![Register::new(value_kind, value_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        let result_reg = self.alloc_reg();
        // RPython jtransform.py rewrite_op_jit_conditional_call_value — conditional_call_value_ir_{i|r}
        let builder_call = match value_kind {
            BindingKind::Ref => quote! {
                __builder.conditional_call_value_ir_r_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
            _ => quote! {
                __builder.conditional_call_value_ir_i_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
        };
        // `conditional_call_elidable!` is the elidable cache helper; per
        // `rlib/jit.py` the callee need not be `@elidable` but
        // the cond_call_value op itself caches the result.  Default to
        // `Elidable*Wrapped` based on the leading value-kind so an
        // inferred policy still classifies as elidable.
        let inferred_default = match value_kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call_elidable!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!(
                "conditional_call_elidable! helper policy {policy:?} has no direct-call result kind"
            );
        };
        if !call_result_matches_binding(result_kind, value_kind) {
            panic!(
                "conditional_call_elidable! value/result kind mismatch for helper policy {policy:?}"
            );
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call_elidable!");
        // `call.py getcalldescr`'s loop-invariant non-void-args
        // assert (see plain `conditional_call!` lowerer for the citation).
        // `conditional_call_elidable!` accepts non-elidable cache-computing
        // helpers per `rlib/jit.py`, so a `LoopInvariant` slot is
        // legal in principle and must enforce the same args-empty rule.
        // Static check applies only to explicit-policy paths; `Infer`
        // resolves slot at runtime from the `__policy` byte.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call_elidable!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_value_policy_check(
                value_kind,
                func_args.is_empty(),
            ))
        } else {
            None
        };
        // The shim stands in for the callee, whose parameters are `func_args`
        // alone: `arg_regs` carries the leading value register the cond_call op
        // reads and the callee never sees.
        let cond_call_arg_kinds: Vec<BindingKind> = arg_regs[1..].iter().map(|r| r.kind).collect();
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
            word_result_addr_for_kind(value_kind, func_path, &cond_call_arg_kinds),
            cond_call_arg_kinds.len(),
        );
        self.emit_op(
            OpMeta::linear(
                OpKind::Call,
                arg_regs,
                vec![Register::new(value_kind, result_reg)],
            ),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)`.  `conditional_call_elidable`
        // still accepts non-elidable cache-computing helpers per
        // `rlib/jit.py`; their explicit policy maps to
        // `EffectInfoSlot::CanRaise` and therefore keeps the marker.
        // `Infer` resolves slot at runtime; guard the physical marker with
        // the same can-raise policy cases instead of emitting a redundant
        // PyPy-invisible marker.
        if is_inferred {
            let can_raise_codes: &[u8] = match value_kind {
                BindingKind::Int => &[INT_DONT_LOOK_INSIDE, INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR, REF_DONT_LOOK_INSIDE],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(Binding {
            reg: result_reg,
            kind: value_kind,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py `handle_recursive_call` — recognises
    /// `recursive_portal_call!(driver, green0, green1, ...)` and emits the
    /// `recursive_call_int` opcode (pyjitpl.py _opimpl_recursive_call `opimpl_recursive_call`
    /// → BC_RECURSIVE_CALL_INT).
    ///
    /// The greens are lowered in jitdriver declaration order and each is
    /// tagged with its register-bank kind, so the dispatcher reads it from
    /// the matching bank (an int `pc` green from the int bank, a ref
    /// `program` green from the ref bank) and hashes the key against
    /// `green_args_spec`.  The callee's fresh reds are built by the
    /// dispatcher from `recursive_fresh_entry_reds`, so no caller→callee
    /// argument moves are recorded here (`args = &[]`).  `jd_index` is the
    /// runtime `__jdindex` parameter of `__dispatch_jitcode_*`
    /// (jtransform.py `portal_jd.index`, threaded exactly like
    /// `jit_merge_point` / `loop_header`), not a literal 0: when a consumer
    /// also installs the propagate-descr placeholder driver via
    /// `ensure_default_driver_sd`, the real portal driver registers at slot
    /// 1+, so a hardcoded 0 would read the placeholder's empty
    /// `green_args_spec`.
    pub(super) fn lower_recursive_portal_call(&mut self, expr: &Expr) -> Option<Binding> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "recursive_portal_call" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        // `driver`, then one expression per green in declaration order.
        if args.len() < 2 {
            panic!("recursive_portal_call! requires a driver and at least one green");
        }
        let green_exprs = &args[1..];
        let mut green_bindings = Vec::with_capacity(green_exprs.len());
        for g in green_exprs {
            green_bindings.push(self.lower_value_expr(g)?);
        }
        let green_reads: Vec<Register> =
            green_bindings.iter().map(Register::from_binding).collect();
        // jtransform.py `handle_recursive_call` opens with `promote_greens`:
        // every green Variable is pinned by a `-live-` + `<kind>_guard_value`
        // pair before the call is emitted.  The dispatcher picks the callee by
        // hashing the green key it reads out of these registers, so a green the
        // trace never guarded lets a recorded loop replay with a different key
        // and enter the portal at a pc it was not compiled for.  The merge
        // point promotes its greens for the same reason
        // (`dispatch::emit_promote_greens`); this is the other `promote_greens`
        // call site.  Upstream skips Constants and Voids; every green here is a
        // register a value expression just produced, so none is skipped.
        for binding in &green_bindings {
            self.emit_live_and_guard_value(binding.kind, binding.reg);
        }
        let green_tokens: Vec<proc_macro2::TokenStream> = green_bindings
            .iter()
            .map(|b| {
                let reg = b.reg;
                let kind = match b.kind {
                    BindingKind::Int => quote! { majit_metainterp::jitcode::JitArgKind::Int },
                    BindingKind::Ref => quote! { majit_metainterp::jitcode::JitArgKind::Ref },
                    BindingKind::Float => quote! { majit_metainterp::jitcode::JitArgKind::Float },
                };
                quote! { (#kind, #reg) }
            })
            .collect();
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(OpKind::Call, green_reads, vec![Register::int(result_reg)]),
            quote! {
                __builder.recursive_call_int(__jdindex as u16, #result_reg, &[#(#green_tokens),*], &[]);
            },
        );
        // jtransform.py:533 — `recursive_call_*` is always followed by `-live-`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py — `rewrite_op_jit_record_known_result`.
    ///
    /// Recognizes `record_known_result!(result, func, args...)` and emits
    /// the canonical `record_known_result_{i,r}_ir_v` builder entrypoint.
    pub(super) fn lower_record_known_result(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "record_known_result" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = known result, args[1] = func path, args[2..] = function arguments
        let result_binding = self.lower_value_expr(args[0])?;
        let result_reg = result_binding.reg;
        // jtransform.py:293-295: float → raise Exception
        if matches!(result_binding.kind, BindingKind::Float) {
            panic!("record_known_result does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        let mut arg_regs: Vec<Register> = Vec::new();
        for arg in &args[2..] {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("record_known_result does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // RPython jtransform.py:302-307 — record_known_result_{i|r}
        let builder_call = match result_binding.kind {
            BindingKind::Ref => quote! {
                __builder.record_known_result_r_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
            _ => quote! {
                __builder.record_known_result_i_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
        };
        // RPython pyjitpl.py:413-419 passes the known result box as
        // `prepend_box=resbox`; record_known_result reads that box and
        // produces no result (`_v` suffix).
        // `record_known_result!` requires an elidable callee — the
        // `slot.is_elidable()` assert below catches non-elidable
        // policies.  Default `Infer` to `Elidable*Wrapped` so the
        // assert succeeds when the helper is registered through the
        // wrapped policy path.
        let inferred_default = match result_binding.kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "record_known_result!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("record_known_result! helper policy {policy:?} has no direct-call result kind");
        };
        if !call_result_matches_binding(result_kind, result_binding.kind) {
            panic!("record_known_result! result kind mismatch for helper policy {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "record_known_result!");
        if !slot.is_elidable() {
            panic!("record_known_result! requires an elidable helper policy, got {policy:?}");
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_record_known_result_policy_check(
                result_binding.kind,
            ))
        } else {
            None
        };
        let known_result_arg_kinds: Vec<BindingKind> = arg_regs.iter().map(|r| r.kind).collect();
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
            word_result_addr_for_kind(result_binding.kind, func_path, &known_result_arg_kinds),
            known_result_arg_kinds.len(),
        );
        let result_typed = Register::new(result_binding.kind, result_reg);
        let mut reads = Vec::with_capacity(arg_regs.len() + 1);
        reads.push(result_typed);
        reads.extend(arg_regs);
        self.emit_op(
            OpMeta::linear(OpKind::RecordKnownResult, reads, Vec::new()),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:311-312`: append `-live-` exactly when the
        // elidable calldescr can raise.  In inferred mode, guard the
        // physical marker on the elidable-can-raise / memoryerror policy
        // bytes instead of emitting one for elidable_cannot_raise.
        if is_inferred {
            let can_raise_codes: &[u8] = match result_binding.kind {
                BindingKind::Int => &[INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    /// Lower the inside of `unsafe { ... }` as ordinary statements.
    fn lower_unsafe_block(&mut self, block: &syn::Block) -> Option<()> {
        for stmt in &block.stmts {
            self.lower_stmt(stmt)?;
        }
        Some(())
    }

    fn lower_expr_stmt(&mut self, expr: &Expr) -> Option<()> {
        // Green-pc inline dispatch: a `pc += N` / `pc = target` write inside
        // an inlined arm body must land in pc's pinned register (reg0), not
        // SSA-rebind the `pc` name (`lower_local_reassign`) nor be dropped as
        // non-jit-state.  Runs first so it intercepts the pc-write before the
        // generic reassign / drop path; a no-op when `!self.pc_pinned`.
        if let Some(()) = self.lower_pc_pinned_write(expr) {
            return Some(());
        }
        // An `unsafe` block is the same statements as its body. The
        // interpreter still type-checks the block; the trace does not
        // grow an op for the brace.
        if let Expr::Unsafe(inner) = expr {
            return self.lower_unsafe_block(&inner.block);
        }
        // jtransform.py rewrite_op_hint — `hint(x, promote=True)` in
        // statement context.  Routes both `x = promote(arg)` (plain local
        // re-assignment, no state-write to trigger
        // `lower_state_field_write`'s RHS recursion) and bare
        // `promote(x);` through `lower_promote_call`, which emits the
        // `-live-` + `<kind>_guard_value` pair.  Without this site the
        // statement-form promote would silently no-op when the
        // config-aware fall-through later observes `stmt_modifies_jit_
        // state(stmt) == false`.
        if let Some(()) = self.lower_promote_stmt(expr) {
            return Some(());
        }
        // pyjitpl.py opimpl_assert_not_none — statement-form
        // `jit::assert_not_none(x);` (discard return value).
        if let Some(()) = self.lower_assert_not_none_stmt(expr) {
            return Some(());
        }
        // pyjitpl.py opimpl_record_exact_class — statement-form
        // `jit::record_exact_class(value, cls);` (no return value).
        if let Some(()) = self.lower_record_exact_class_stmt(expr) {
            return Some(());
        }
        // State field writes (register/tape machines).
        if let Some(()) = self.lower_state_field_update(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_field_write(expr) {
            return Some(());
        }
        // `state.<vable>.<field> =` / `state.<vable>.<array>[i] =` before the
        // heap setfield / setarrayitem rewrites. `jtransform.py`
        // `setfield_vable_*` / `setarrayitem_vable_*`.
        if let Some(()) = self.lower_vable_field_write(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_vable_array_update(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_vable_array_write(expr) {
            return Some(());
        }
        // Field write-through a `ref(T)` state scalar:
        // `state.<ref>.<member> = <int expr>` → setfield_gc_i (invalidates the
        // matching cached getfield). After lower_state_field_write (which only
        // matches `state.<scalar> = ...` whose LHS base is `state`).
        if let Some(()) = self.lower_state_ref_field_setfield(expr) {
            return Some(());
        }
        // Array element write on a local ref binding whose field is declared
        // in `array_fields`: `<binding>.<field>[<idx>] = <expr>` →
        // getfield_gc_r for the buffer base, then setarrayitem_gc_i.
        if let Some(()) = self.lower_ref_binding_array_write(expr) {
            return Some(());
        }
        // Field write on a local ref binding with known struct type:
        // `<binding>.<field> = <expr>` → setfield_gc_i/setfield_gc_r.
        if let Some(()) = self.lower_ref_binding_setfield(expr) {
            return Some(());
        }
        // Raw native-memory store: `majit_raw_store_{i,u}{8,16,32,64}(base,
        // ea, val);` →
        // raw_store_i (jtransform.py rewrite_op_raw_store).  Runs
        // before the residual-call path so the store lowers to an inline
        // side-effecting IR op instead of an opaque helper call.
        if let Some(()) = self.lower_raw_store_stmt(expr) {
            return Some(());
        }
        // Plain `[int]`/`[float]` compound assign then store. Update before
        // write, matching the vable array pair. `jtransform.py`
        // `rewrite_op_getarrayitem` / `rewrite_op_setarrayitem` raw arms.
        if let Some(()) = self.lower_state_array_update(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_array_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:650 — hint_force_virtualizable rewrite.
        if let Some(()) = self.lower_vable_force(expr) {
            return Some(());
        }
        // RPython jtransform.py:655 — access_directly/fresh_virtualizable suppression.
        if let Some(()) = self.lower_vable_hint_suppress(expr) {
            return Some(());
        }
        // RPython jtransform.py rewrite_op_jit_conditional_call — conditional_call!(condition, func, args...)
        if let Some(()) = self.lower_conditional_call(expr) {
            return Some(());
        }
        // RPython jtransform.py rewrite_op_jit_record_known_result — record_known_result!(result, func, args...)
        if let Some(()) = self.lower_record_known_result(expr) {
            return Some(());
        }
        // Loop-carried local updates have to write their existing physical
        // register; rebinding would leave an already-emitted loop header
        // reading the pre-update register.
        if let Some(()) = self.lower_local_update(expr) {
            return Some(());
        }
        // Local variable reassignment: `pc = expr` or `stackok = expr`.
        // Rebinds an already-known local to a freshly-lowered RHS value.
        if let Some(()) = self.lower_local_reassign(expr) {
            return Some(());
        }

        if let Expr::If(expr_if) = expr {
            return self.lower_if_stmt(expr_if);
        }

        if let Expr::Match(expr_match) = expr {
            return self.lower_match_stmt(expr_match);
        }

        if let Expr::While(expr_while) = expr {
            return self.lower_while_loop(expr_while);
        }

        if let Expr::Loop(expr_loop) = expr {
            return self.lower_loop_expr(expr_loop);
        }

        if let Expr::ForLoop(expr_for) = expr {
            return self.lower_for_loop(expr_for);
        }

        if let Some(()) = self.lower_config_call_stmt(expr) {
            return Some(());
        }

        // Config-aware patterns
        if self.config.is_some()
            && let Some(()) = self.lower_io_call_stmt(expr)
        {
            return Some(());
        }

        None
    }

    pub(super) fn lower_config_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        // RPython `Transformer.rewrite_call` sees a method receiver as the
        // first ordinary argument.  Normalize Rust method syntax through the
        // same call-policy path as value-position calls instead of silently
        // dropping a void/result-discarding method statement.
        let normalized;
        let call = match expr {
            Expr::Call(call) => call,
            Expr::MethodCall(method_call) => {
                normalized = self.normalize_method_call(method_call)?;
                &normalized
            }
            _ => return None,
        };
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding);
        }
        let func = &call.func;
        let arg_kinds: Vec<BindingKind> = arg_bindings.iter().map(|b| b.kind).collect();
        let word_result_addr = word_result_addr_tokens(func, &arg_kinds);
        // A void residual is recorded with a `()` result, so its target is
        // spelled words-in-nothing-out rather than through the result-widening
        // shim above.
        let word_void_addr = word_void_addr_tokens(func, &arg_kinds);
        // jtransform.py:467-471 / 480-482: `-live-` follows the call, it does
        // not precede it.  Decide here whether the explicit arm below needs a
        // trailing marker; the inferred arm emits its own runtime-conditional
        // one, so it reports `false` and is excluded.
        let post_live_after_call = match &policy {
            CallPolicySpec::Explicit(kind) => explicit_call_emits_post_live(*kind),
            CallPolicySpec::Infer => false,
        };
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualVoid
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise => {
                    let cannot_raise = matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise,
                    );
                    let write_ei = self.residual_write_effect_info_tokens(func, !cannot_raise);
                    // The register literals must be spelled inside the
                    // `__builder` call itself: `regalloc`'s rewriter recolors
                    // only the arguments of a non-aux builder method call, so
                    // arguments bound to a local first would keep the register
                    // numbers the lowerer handed out before coloring.
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = if let Some(write_ei) = write_ei {
                        // Declared field mutator: residual with a write-set
                        // naming the mutated field so the optimizer invalidates
                        // its cached `getfield_gc_i`, preserving the policy's
                        // can-raise / cannot-raise extra-effect.
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #write_ei,
                            );
                        }
                    } else if cannot_raise {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! {
                            __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args);
                        }
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr_void(#word_void_addr);
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::InlineVoid => {
                    let shared_path = inline_shared_path(&call.func)?;
                    let prebuild_path = inline_prebuild_path(&call.func)?;
                    let (inline_call, post_live) = inline_call_tokens_void(&arg_bindings);
                    let __arg_regs = inline_call_read_regs(&arg_bindings);
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(OpKind::InlineCall, __arg_regs, vec![]),
                        quote! {
                            let __sub_edge = #shared_path(__asm);
                            let __sub_idx = __builder.add_sub_jitcode_ref(__sub_edge);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                crate::jit_interp::CallPolicyKind::InlinePipelineVoid => {
                    let pipeline_name = match &*call.func {
                        syn::Expr::Path(ep) => ep
                            .path
                            .segments
                            .last()
                            .map(|segment| segment.ident.to_string()),
                        _ => None,
                    }?;
                    let (inline_call, post_live) = inline_call_tokens_void(&arg_bindings);
                    let arg_regs = inline_call_read_regs(&arg_bindings);
                    self.inline_liveness_prebuild.push(quote! {
                        __majit_pipeline_liveness_prebuild(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(OpKind::InlineCall, arg_regs, vec![]),
                        quote! {
                            let __sub_jitcode = __majit_pipeline_jitcode(#pipeline_name);
                            let __sub_idx = __builder.add_sub_jitcode_arc(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                // A result-returning inline helper whose result the statement
                // discards. The sub-jitcode still ends in a typed return
                // opcode, so it needs a destination even though nothing reads
                // it; `alloc_reg` mints one, the same way the discarded
                // `ResidualInt` family below does. Everything else is the
                // value-position lowering verbatim.
                //
                // Without this arm the call reaches `_ => return None` and the
                // statement is dropped: `explicit_call_emits_post_live` already
                // answers for these kinds, so the accounting says the policy is
                // handled while the lowering says it is not, and the effect the
                // helper performs leaves the trace with no diagnostic. The
                // workaround is to bind the result to `let _x = ...`, which is
                // a source change the declaration does not ask for.
                crate::jit_interp::CallPolicyKind::InlinePipelineInt
                | crate::jit_interp::CallPolicyKind::InlinePipelineRef
                | crate::jit_interp::CallPolicyKind::InlinePipelineFloat => {
                    let result_kind = binding_kind_for_inline_policy(kind)
                        .expect("the arm's own patterns are the pipeline result policies");
                    let throwaway_reg = self.alloc_reg();
                    let pipeline_name = match &*call.func {
                        syn::Expr::Path(ep) => ep
                            .path
                            .segments
                            .last()
                            .map(|segment| segment.ident.to_string()),
                        _ => None,
                    }?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, throwaway_reg);
                    let arg_regs = inline_call_read_regs(&arg_bindings);
                    self.inline_liveness_prebuild.push(quote! {
                        __majit_pipeline_liveness_prebuild(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            arg_regs,
                            vec![Register::new(result_kind, throwaway_reg)],
                        ),
                        quote! {
                            let __sub_jitcode = __majit_pipeline_jitcode(#pipeline_name);
                            let __sub_idx = __builder.add_sub_jitcode_arc(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                crate::jit_interp::CallPolicyKind::InlineInt
                | crate::jit_interp::CallPolicyKind::InlineRef
                | crate::jit_interp::CallPolicyKind::InlineFloat => {
                    let result_kind = binding_kind_for_inline_policy(kind)
                        .expect("the arm's own patterns are the inline result policies");
                    let throwaway_reg = self.alloc_reg();
                    let shared_path = inline_shared_path(&call.func)?;
                    let sub_return_kind = jit_arg_kind_tokens(result_kind);
                    let prebuild_path = inline_prebuild_path(&call.func)?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, throwaway_reg);
                    let __arg_regs = inline_call_read_regs(&arg_bindings);
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, throwaway_reg)],
                        ),
                        quote! {
                            let __sub_edge = #shared_path(__asm);
                            let __sub_return_kind = #sub_return_kind;
                            let __sub_idx = __builder.add_sub_jitcode_ref(__sub_edge);
                            #inline_call
                        },
                    );
                    self.emit_post_inline_live_marker(post_live);
                }
                crate::jit_interp::CallPolicyKind::MayForceVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr_void(#word_void_addr);
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr_void(#word_void_addr);
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::LoopInvariantVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr_void(#word_void_addr);
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr_void(#word_void_addr);
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                // Stmt-form variants of result-returning policies discard
                // the value but still need the IR call op recorded so the
                // compiled trace runs the side effect (a `stack_pop(...)`
                // whose result is discarded still has to pop in compiled
                // code).
                // Allocate a throwaway destination register; never read it.
                //
                // RPython jtransform.py `handle_residual_call` lowers
                // every direct_call to a residual_call regardless of result
                // usage; majit's CallPolicyKind enum captures the effect
                // distinction (Residual / MayForce / LoopInvariant / Elidable)
                // so the dispatched bytecode
                // varies per policy here.  Wrapped variants stay deferred
                // — wrapper closure plumbing is shared with the void path
                // and not exercised by current `#[jit_interp]` users.
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let throwaway_reg = self.alloc_reg();
                    let canonical_call = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            quote! { residual_call_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceInt => {
                            quote! { call_may_force_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                            quote! { call_loopinvariant_int_canonical_via_target }
                        }
                        _ => unreachable!(),
                    };
                    // A declared `residual_writes` mutator routes through the
                    // `_with_effect_info` int variant carrying the field
                    // write-set; only the residual policy qualifies (may-force /
                    // loop-invariant carry their own effects).
                    let write_ei = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            self.residual_write_effect_info_tokens(func, true)
                        }
                        _ => None,
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #throwaway_reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #throwaway_reg,
                                );
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                Register::ints(&arg_regs),
                                vec![Register::int(throwaway_reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                                #call_invocation
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(__fn_idx, #typed_args, #throwaway_reg);
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                __arg_regs,
                                vec![Register::int(throwaway_reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                                #call_invocation
                            },
                        );
                    }
                }
                // `call.py getcalldescr` non-elidable EF_CANNOT_RAISE
                // for int residuals.  Dispatches via the
                // `_with_effect_info(cannot_raise_effect_info())` builder
                // method so the recorded calldescr's `EffectInfo`
                // matches PyPy's `cannot_raise_effect_info()`.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #throwaway_reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // Pure flows through
                    // the canonical `BC_RESIDUAL_CALL_*_I` family with the
                    // calldescr's `extra_info` set per `call.py
                    // _canraise(op)`'s 3-way pick.  The walker
                    // (`pyjitpl/dispatch.rs`) reads
                    // `effectinfo.check_is_elidable()` and routes through
                    // `record_result_of_call_pure` mirroring
                    // `pyjitpl.py`; the trailing
                    // `GUARD_NO_EXCEPTION` is gated on
                    // `effectinfo.check_can_raise(False)` so cannot-raise
                    // elidable callees skip it.
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ResidualVoidWrapped
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    // `call.py getcalldescr`: descr's `EffectInfo`
                    // differs by the analyzer's `_canraise` result, but the
                    // residual_call dispatch family is the same.
                    let call_stmt = if matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped,
                    ) {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! { __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args); }
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::MayForceVoidWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::MayForceVoidWrapped => {
                            quote! { __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                            quote! { __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                // Non-wrapped Ref statement-form (result discarded).
                // Like ResidualVoid but emits a ref-return residual call
                // whose result is discarded.
                crate::jit_interp::CallPolicyKind::ResidualRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // The register literals must be spelled inside the
                    // `__builder` call itself: `regalloc`'s rewriter recolors
                    // only the arguments of a non-aux builder method call, so
                    // arguments bound to a local first would keep the register
                    // numbers the lowerer handed out before coloring.
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(throwaway_reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg);
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::AllocRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(throwaway_reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #throwaway_reg, majit_metainterp::can_raise_effect_info());
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::StrConcat => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let ref_word = assert_ref_result_one_word(func, arg_bindings.len());
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(throwaway_reg)]),
                        quote! {
                            #ref_word
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #throwaway_reg, majit_metainterp::str_concat_effect_info());
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::NurseryAllocRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // The register literals must be spelled inside the
                    // `__builder` call itself: `regalloc`'s rewriter recolors
                    // only the arguments of a non-aux builder method call, so
                    // arguments bound to a local first would keep the register
                    // numbers the lowerer handed out before coloring.
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(throwaway_reg)]),
                        quote! {
                            let __fn_idx = __builder.add_word_abi_fn_ptr(#word_result_addr);
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, #typed_args, #throwaway_reg, majit_metainterp::nursery_alloc_effect_info());
                        },
                    );
                }
                // Wrapped Int / Ref / Float statement-form: result discarded,
                // but the residual_call must still execute the side effect on
                // the compiled trace.  RPython jtransform.py handle_residual_call
                // handle_residual_call lowers every direct_call regardless of
                // result usage; the wrapped policy adds the trace_target /
                // concrete_target tuple resolution shared with the void
                // wrapped variants above.  Throwaway destination register is
                // allocated (per-bank slot picked by JitCodeBuilder when the
                // typed call dispatches) and never read.
                crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceFloatWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    // Result bank — pick from the wrapped policy variant family.
                    let result_kind = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            BindingKind::Int
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            BindingKind::Ref
                        }
                        _ => BindingKind::Float,
                    };
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        _ => unreachable!(),
                    };
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, throwaway_reg)],
                        ),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_word_abi_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ConcreteOnlyVoid => {
                    // No IR ops emitted on the JIT path. The concrete
                    // RefFieldRewriter path calls the function normally.
                }
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                let __slot_tokens = CondCallEffectSlot::slot_from_policy_tokens();
                self.emit_op(
                    OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                    quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                        let __trace_target = if __trace_target.is_null() {
                            // Statement Infer calls only the void policy bytes
                            // in the match below, so this raw address is not a
                            // Ref call target (`llmodel.py bh_call_r`). Any
                            // other byte panics before the call.
                            #func as *const ()
                        } else {
                            __trace_target
                        };
                        let __concrete_target = if __concrete_target.is_null() {
                            __trace_target
                        } else {
                            __concrete_target
                        };
                        let __fn_idx = __builder.add_call_target_with_save_err(
                            __trace_target,
                            __concrete_target,
                            #__slot_tokens,
                            __save_err,
                        );
                        match __policy {
                            #VOID_DONT_LOOK_INSIDE => {
                                __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // `call.py:303` non-elidable EF_CANNOT_RAISE for void.
                            #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                __builder.residual_call_void_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                            #VOID_MAY_FORCE => {
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            #VOID_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // The `_ =>` arm is a runtime invariant violation
                            // (helper policy companion fn returned an
                            // unrecognized byte), NOT a recoverable lower-time
                            // inference failure, so it panics regardless of the
                            // outer Lowerer's `InferenceFailureMode`. Earlier
                            // versions routed this through
                            // `inference_failure_tokens` which emits
                            // `return None;` in `ReturnNone` mode — wrong for
                            // dispatch-body wrappers that return `JitCode`,
                            // not `Option<_>`, surfaced as a type-check error
                            // when a `dont_look_inside` helper is called from
                            // a dispatch JitCode body (A.2.5).
                            other => panic!(
                                "inferred void-call policy returned unrecognized byte {other}; \
                                 expected one of 1 (residual), 9 (may_force), \
                                 17 (loopinvariant)"
                            ),
                        }
                    },
                );
                // jtransform.py:467-471 — trailing `-live-` gated on the
                // runtime policy byte's can-raise codes (void residual /
                // may-force); LOOP_INVARIANT and the
                // CANNOT_RAISE void surface skip it.
                self.emit_op(
                    OpMeta::live_marker_if(inferred_policy_live_condition(
                        func,
                        &[VOID_DONT_LOOK_INSIDE, VOID_MAY_FORCE],
                    )),
                    quote! { let _ = __builder.live_placeholder(); },
                );
            }
        }
        // jtransform.py:467-471 / 480-482 — trailing `-live-` for the explicit
        // residual / elidable / may-force arms (computed above).
        if post_live_after_call {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    /// Lower a raw native-memory store intrinsic
    /// `majit_raw_store_{i,u}{8,16,32,64}(base, ea, val)` to a `raw_store_i`
    /// op (the write-side analogue of `lower_raw_load_call`).
    ///
    /// RPython parity: an `rffi.raw_storage_setitem(base, offset, value)`
    /// in the interpreter source is rewritten by
    /// `jtransform.py rewrite_op_raw_store` to
    /// `raw_store_i(base, offset, value, arraydescrof(CArray(T)))`, where `T`
    /// is the STORED VALUE's own type — so the access width and signedness
    /// come off the descr and any width upstream can store, this can.  The
    /// three operands are all int-kind (raw address, byte offset, stored
    /// value); the op has no result.
    fn lower_raw_store_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let segments = canonical_expr_segments(&call.func)?;
        let (item_size, is_signed) = match segments.last().map(String::as_str)? {
            "majit_raw_store_i8" => (1usize, true),
            "majit_raw_store_u8" => (1usize, false),
            "majit_raw_store_i16" => (2usize, true),
            "majit_raw_store_u16" => (2usize, false),
            "majit_raw_store_i32" => (4usize, true),
            "majit_raw_store_u32" => (4usize, false),
            "majit_raw_store_i64" => (8usize, true),
            // At 8 bytes the signedness cannot change which bits are written,
            // but the intrinsic is documented as supported, so accept it for
            // parity — as the load side does.
            "majit_raw_store_u64" => (8usize, false),
            _ => return None,
        };
        if call.args.len() != 3 {
            return None;
        }
        let base = self.lower_value_expr(&call.args[0])?;
        let ea = self.lower_value_expr(&call.args[1])?;
        let val = self.lower_value_expr(&call.args[2])?;
        let (base_reg, ea_reg, val_reg) = (base.reg, ea.reg, val.reg);
        self.emit_op(
            OpMeta::linear(
                OpKind::RawStore,
                vec![
                    Register::int(base_reg),
                    Register::int(ea_reg),
                    Register::int(val_reg),
                ],
                vec![],
            ),
            quote! {
                let __raw_descr = __builder.add_raw_int_array_descr_signed(#item_size, #is_signed);
                __builder.raw_store_i(
                    #base_reg as u16,
                    #ea_reg as u16,
                    #val_reg as u16,
                    __raw_descr,
                );
            },
        );
        Some(())
    }

    /// Lower I/O call: `<io>::write_number(r, writer)` → `residual_call_void(shim, r)`
    fn lower_io_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;

        for (io_path, shim) in &config.io_shims {
            if func_segments == *io_path {
                let arg = unwrap_ref_expr(call.args.first()?);
                let binding = self.lower_value_expr(arg)?;
                let reg = binding.reg;
                self.emit_op(
                    OpMeta::linear(OpKind::Call, vec![Register::int(reg)], vec![]),
                    quote! {
                        let __fn_idx = __builder.add_fn_ptr(#shim as *const ());
                        __builder.residual_call_void_canonical_via_target(
                            __fn_idx,
                            &[majit_metainterp::JitCallArg::int(#reg)],
                        );
                    },
                );
                // jtransform.py:467-471 — the void shim is a may-raise
                // residual call, so `-live-` follows it.
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                return Some(());
            }
        }

        None
    }
}

/// Idents naming this machine's greens.
///
/// `LowererConfig::greens` holds the merge point's green *expressions*.  Only a
/// plain-ident green names a caller local that a statement can assign to, so a
/// field or index spelling is skipped rather than flattened to a bare name that
/// would collide with an unrelated local of that name.
pub(super) fn green_idents(config: &LowererConfig) -> Vec<String> {
    config
        .greens
        .iter()
        .filter_map(|green| match green {
            Expr::Path(p) if p.qself.is_none() => p.path.get_ident().map(|id| id.to_string()),
            _ => None,
        })
        .collect()
}

/// Name of the green `stmt` assigns to, if it assigns to one.
///
/// Covers both spellings the dispatch loops use: the compound
/// `pc += N` (syn 2 parses it as `Expr::Binary` with an assigning `BinOp`) and
/// the plain `pc = <expr>`.  Only the assignment *target* counts — a green read
/// on the right-hand side is what every arm does and is not a write.
///
/// Closure bodies and nested items are skipped for the reason
/// [`stmt_contains_return`] skips them: a name bound there is a different
/// binding that merely shares a spelling.
fn stmt_writes_green(stmt: &Stmt, greens: &[String]) -> Option<String> {
    use syn::visit::Visit;
    struct Probe<'g> {
        greens: &'g [String],
        hit: Option<String>,
    }
    impl Probe<'_> {
        fn record(&mut self, target: &Expr) {
            if self.hit.is_some() {
                return;
            }
            let Expr::Path(p) = target else { return };
            if p.qself.is_some() {
                return;
            }
            let Some(id) = p.path.get_ident() else { return };
            let name = id.to_string();
            if self.greens.contains(&name) {
                self.hit = Some(name);
            }
        }
    }
    impl<'ast> Visit<'ast> for Probe<'_> {
        fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
            self.record(&node.left);
            syn::visit::visit_expr_assign(self, node);
        }
        fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
            if opcode_for_assign_binop(&node.op).is_some() {
                self.record(&node.left);
            }
            syn::visit::visit_expr_binary(self, node);
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }
    let mut probe = Probe { greens, hit: None };
    probe.visit_stmt(stmt);
    probe.hit
}

/// `true` if `stmt` encloses a `return` belonging to the interpreter function
/// being lowered.
///
/// Closure bodies and nested items are deliberately NOT descended into: a
/// `return` inside `|| { return 1; }` or an inner `fn` exits *that* body, not
/// the dispatch arm, so it is neither lowerable here nor a reason to refuse the
/// enclosing statement.
fn stmt_contains_return(stmt: &Stmt) -> bool {
    use syn::visit::Visit;
    struct Probe {
        hit: bool,
    }
    impl<'ast> Visit<'ast> for Probe {
        fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
            self.hit = true;
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }
    let mut probe = Probe { hit: false };
    probe.visit_stmt(stmt);
    probe.hit
}

/// Whether `stmt` encloses a `break` or `continue` that targets the dispatch
/// loop being lowered — the loop-control twin of `stmt_contains_return`.
///
/// `break`/`continue` bind to the *innermost* enclosing loop, so an unlabelled
/// one inside a nested `loop`/`while`/`for` written in the arm body exits that
/// inner loop and never reaches the dispatch back-edge.  Descending into those
/// bodies would refuse statements that are perfectly safe to drop, so the probe
/// tracks loop depth and only fires at depth 0 — the same scoping rule
/// `expr_has_loop_control` documents.  A *labelled* `break 'l` / `continue 'l`
/// can cross a nested loop, so it counts at any depth.
///
/// `expr_has_loop_control` is not reused here because it only inspects
/// `Stmt::Expr`: a control transfer in an initializer (`let x = if c { break }
/// else { 1 };`) is invisible to it, and this guard has to see it.  Closures and
/// nested items are skipped for the same reason `stmt_contains_return` skips
/// them — a `break` inside them belongs to a different body.
fn stmt_contains_loop_control(stmt: &Stmt) -> bool {
    use syn::visit::Visit;
    struct Probe {
        hit: bool,
        loop_depth: u32,
    }
    impl Probe {
        fn record(&mut self, labelled: bool) {
            if labelled || self.loop_depth == 0 {
                self.hit = true;
            }
        }
    }
    impl<'ast> Visit<'ast> for Probe {
        fn visit_expr_break(&mut self, node: &'ast syn::ExprBreak) {
            self.record(node.label.is_some());
            syn::visit::visit_expr_break(self, node);
        }
        fn visit_expr_continue(&mut self, node: &'ast syn::ExprContinue) {
            self.record(node.label.is_some());
        }
        fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
            self.loop_depth += 1;
            syn::visit::visit_expr_loop(self, node);
            self.loop_depth -= 1;
        }
        fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
            self.loop_depth += 1;
            syn::visit::visit_expr_while(self, node);
            self.loop_depth -= 1;
        }
        fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
            self.loop_depth += 1;
            syn::visit::visit_expr_for_loop(self, node);
            self.loop_depth -= 1;
        }
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}
        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }
    let mut probe = Probe {
        hit: false,
        loop_depth: 0,
    };
    probe.visit_stmt(stmt);
    probe.hit
}

/// Whether `stmt` encloses a `hint_fresh_virtualizable` in either spelling.
///
/// The hint reaches the lowerer two ways: as a call
/// (`lower_vable_hint_identity_call` matches `Expr::Call`) and as a macro
/// (`lower_vable_hint_suppress` matches `Expr::Macro`).  Both are refused, so
/// the probe has to see both -- and it has to see them anywhere in the
/// statement, because the call form appears in value position
/// (`let f = hint_fresh_virtualizable(frame);`) where the statement that gets
/// refused is the `let`.
///
/// Closures and nested items are not skipped the way `stmt_contains_return`
/// skips them: a hint inside a closure still names this body's virtualizable.
fn stmt_contains_fresh_virtualizable_hint(stmt: &Stmt) -> bool {
    use syn::visit::Visit;
    struct Probe {
        hit: bool,
    }
    impl<'ast> Visit<'ast> for Probe {
        fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
            if let Expr::Path(p) = &*node.func
                && classify_virtualizable_hint_syn_path(&p.path)
                    == Some(VirtualizableHintKind::FreshVirtualizable)
            {
                self.hit = true;
            }
            syn::visit::visit_expr_call(self, node);
        }
        fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
            if classify_virtualizable_hint_syn_path(&node.mac.path)
                == Some(VirtualizableHintKind::FreshVirtualizable)
            {
                self.hit = true;
            }
            syn::visit::visit_expr_macro(self, node);
        }
        fn visit_stmt_macro(&mut self, node: &'ast syn::StmtMacro) {
            if classify_virtualizable_hint_syn_path(&node.mac.path)
                == Some(VirtualizableHintKind::FreshVirtualizable)
            {
                self.hit = true;
            }
            syn::visit::visit_stmt_macro(self, node);
        }
    }
    let mut probe = Probe { hit: false };
    probe.visit_stmt(stmt);
    probe.hit
}

#[cfg(test)]
mod continue_target_tests {
    use super::*;

    const DISPATCH_LABEL: &str = "__l_dispatch";

    fn dispatch_lowerer() -> Lowerer<'static> {
        let mut lowerer = Lowerer::new(None);
        lowerer.dispatch_loop_label = Some(syn::Ident::new(
            DISPATCH_LABEL,
            proc_macro2::Span::call_site(),
        ));
        lowerer
    }

    fn emitted(lowerer: &Lowerer<'_>) -> String {
        lowerer
            .statements
            .iter()
            .map(|tokens| tokens.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The spelling this lowering exists for: a `continue` in a dispatch arm
    /// body is the interpreter's back-edge and lowers to a jump to the loop
    /// head the dispatch JitCode marked.
    #[test]
    fn an_arm_body_continue_lowers_to_the_dispatch_back_edge() {
        let mut lowerer = dispatch_lowerer();
        let stmt: Stmt = syn::parse_quote! { continue; };

        assert!(lowerer.lower_stmt(&stmt).is_some());
        let body = emitted(&lowerer);
        assert!(
            body.contains(DISPATCH_LABEL),
            "an arm-body `continue` must jump to the dispatch head:\n{body}"
        );
    }

    /// With no dispatch label there is no loop head to jump to, so the old
    /// `Some(())` emitted nothing and reported success — the transfer was
    /// silently deleted and the caller ran on to the next statement.
    #[test]
    fn a_continue_with_no_dispatch_label_refuses_instead_of_vanishing() {
        let mut lowerer = Lowerer::new(None);
        let stmt: Stmt = syn::parse_quote! { continue; };

        assert!(
            lowerer.lower_stmt(&stmt).is_none(),
            "a `continue` with no loop head to name must refuse"
        );
        assert!(lowerer.statements.is_empty());
        assert!(lowerer.op_metadata.is_empty());
    }

    /// A labelled `continue` crosses nested loops to a loop this lowering
    /// holds no label for; the dispatch head is not it.
    #[test]
    fn a_labelled_continue_refuses() {
        let mut lowerer = dispatch_lowerer();
        let stmt: Stmt = syn::parse_quote! { continue 'outer; };

        assert!(
            lowerer.lower_stmt(&stmt).is_none(),
            "`continue 'outer` names a loop the dispatch label is not"
        );
        assert!(lowerer.statements.is_empty());
        assert!(lowerer.op_metadata.is_empty());
    }
}

#[cfg(test)]
mod residual_write_float_tests {
    use super::*;

    /// `descr.py` `get_type_flag`: an `f64` residual-write field is
    /// `is_float: true`, matching the getfield descr.
    #[test]
    fn residual_write_float_field_sets_is_float() {
        let floatval = crate::jit_interp::IntFieldEntry {
            struct_type: syn::parse_quote!(Num),
            field: syn::parse_quote!(floatval),
            int_type: syn::parse_quote!(f64),
        };
        let mut config =
            LowererConfig::inline_helper(&[], &[], &[floatval], &[], &[], &[], &[], &[]);
        config.residual_writes.push((
            vec!["bump".to_string()],
            syn::parse_quote!(Num),
            syn::parse_quote!(floatval),
            false,
        ));
        let lowerer = Lowerer::new(Some(&config));
        let func: syn::Expr = syn::parse_quote!(bump);
        let tokens = lowerer
            .residual_write_effect_info_tokens(&func, true)
            .expect("write set")
            .to_string();
        assert!(
            tokens.contains("true"),
            "f64 write-set field must carry is_float: {tokens}"
        );
        assert!(tokens.contains("floatval"), "{tokens}");
    }
}
