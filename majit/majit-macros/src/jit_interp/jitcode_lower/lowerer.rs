use super::lower_stmt::REFUSAL_SEPARATOR;
use super::*;

pub(super) struct Lowerer<'c> {
    pub(super) bindings: HashMap<String, Binding>,
    pub(super) statements: Vec<TokenStream>,
    /// Per-op metadata, parallel to `statements`. Populated as B.2.A.ii
    /// migrates each emit site through `emit_op`. Read by the backward
    /// walker (B.2.B). Currently sparse — only `LiveMarker` sites land.
    #[allow(dead_code)]
    pub(super) op_metadata: Vec<OpMeta>,
    pub(super) next_reg: u16,
    pub(super) next_label: u16,
    pub(super) config: Option<&'c LowererConfig>,
    pub(super) call_policies: Vec<(Vec<String>, CallPolicySpec)>,
    pub(super) inference_failure_mode: InferenceFailureMode,
    pub(super) auto_calls: bool,
    /// Prebuild tokens carried up from nested inline-helper lowerings.
    /// These get merged into the parent body's
    /// `liveness_prebuild_tokens` output so the helper's per-marker
    /// triples land in `__prebuild_jitcode_liveness_*` alongside the
    /// outer arm's triples.
    #[allow(dead_code)]
    pub(super) inline_liveness_prebuild: Vec<TokenStream>,
    /// A.2.3a fail-closed install gate signal. Set when
    /// `lower_pre_dispatch_stmts` encounters a pre-dispatch construct
    /// whose structural shape cannot be safely lowered to dispatch
    /// JitCode (currently only inner `Expr::While` that fails the
    /// EXTENDED_ARG-shape recognizer). When `Some`,
    /// `lower_dispatch_body` returns `None` so the dispatch
    /// JitCode body is empty and the runtime install gate at
    /// `generate_state_fields_jit_state` refuses to register the singleton —
    /// matching the Pre-A.2.3 codex (gpt-5.5, 2026-05-05) "fail-closed"
    /// requirement that an unrecognized inner while must NOT silently
    /// pass the existing `BC_GETARRAYITEM_GC_I`-presence gate.
    pub(super) dispatch_tainted_reason: Option<&'static str>,
    /// Why this body's lowering refused, recorded by the FIRST site that
    /// returns `None` so the cause survives the `?` that discards it.
    ///
    /// Without it every refusal reaches `record_degraded_dispatch_arm` as the
    /// single string "arm body could not be lowered to a sub-JitCode", which
    /// covers at least three unrelated causes — an unsupported statement, an
    /// enclosed `return`, and an enclosed `break`/`continue` — so an install
    /// -time reader cannot tell which arm needs which lowering rule.
    pub(super) body_failure_reason: Option<String>,
    /// Diagnoses carried up from child lowerers that refused, held back until
    /// [`Lowerer::take_body_failure_reason`] so they can never take the head.
    ///
    /// A child lowers a nested block *before* this lowerer reaches its own
    /// guard for the statement containing it, so merging a child's reason at
    /// the failure site would make the innermost blocker the head. The head is
    /// contractually the OUTERMOST refusal — every landed `.contains()` pin and
    /// every `refusal_kind` assertion reads it — so child reasons are stashed
    /// here and appended behind it instead.
    ///
    /// Dropped when the statement lowers anyway: a failed inner attempt that
    /// another strategy recovers from is not a blocker, and reporting it would
    /// name a statement that played no part in the refusal.
    pub(super) nested_failure_reasons: Vec<String>,
    /// Name of the LHS variable that received the opcode-fetch
    /// result, set by `try_lower_opcode_fetch_stmt` when it recognises
    /// `let <name> = program[<idx>]` (or the method-call form
    /// `program.get_op(<idx>)`).  `lower_dispatch_chain` uses this name
    /// to look up the opcode reg in `bindings` so the dispatch chain
    /// emits regardless of the consumer's chosen variable name.
    /// Falls back to the literal `"opcode"` (PyPy `pyopcode.py`
    /// canonical name) when unset, preserving existing fixtures.
    pub(super) opcode_var_name: Option<String>,
    /// `true` when this Lowerer is producing the arm body sub-JitCode
    /// inside the dispatch JitCode (`__dispatch_jitcode_<fn>(__asm,
    /// __jdindex: i64)` — `__jdindex` is in scope here).  `false` when
    /// producing the per-arm trace JitCode (`#jitcode_fn_name(__asm,
    /// program, pc, __op)` — `__jdindex` is NOT in scope, so the
    /// `Stmt::Macro` recognition for `can_enter_jit!()` must NOT emit
    /// `__builder.loop_header(__jdindex);` to avoid a
    /// "cannot find value `__jdindex`" compile error in the consumer's
    /// macro expansion).  Set by
    /// `try_generate_jitcode_body_parts_with_caller_bindings` (the sole
    /// dispatch-arm-body lowerer entry).  Pyre's per-arm trace JitCode
    /// is a TODO not present in RPython, so omitting
    /// `loop_header` there is consistent with upstream's single-JitCode
    /// model where `loop_header` lives only in the dispatch-equivalent
    /// JitCode.
    pub(super) in_dispatch_arm_body: bool,
    /// First identity-slot field a sub-JitCode body tried to
    /// `load/store_state_field*`. A fully-lowered body fails expansion
    /// at that access. `MIFrame.setup_call` / `bhimpl_inline_call_*`
    /// copy declared args only.
    pub(super) sub_jitcode_identity_slot: Option<(String, proc_macro2::Span)>,
    /// Label for the dispatch loop start (BC_JIT_MERGE_POINT).
    /// Set by `lower_dispatch_body` so that `continue` statements
    /// in pre-dispatch match arms can emit BC_GOTO to restart the
    /// dispatch iteration (RPython `pc = target; continue` parity).
    pub(super) dispatch_loop_label: Option<Ident>,
    /// `true` while lowering a dispatch arm body INLINE into the dispatch
    /// JitCode (green-pc gated inline path, `lower_dispatch_chain`).  When
    /// set, `pc`-writes (`pc += N`, `pc = target`) are pinned to pc's
    /// register (reg0) via `lower_pc_pinned_write` instead of the default
    /// SSA rebind / drop — the dispatch merge point reads pc from reg0, so
    /// an inline arm's pc update must land there to advance the loop.
    /// PyPy `tl.py` `greens=['pc','code']`: pc is a green threaded through
    /// the portal, advanced in place by each opcode.  Always `false` for
    /// the sub-JitCode arm path (BC_INLINE_CALL copies args caller→callee
    /// only, so a sub-JitCode pc-write cannot reach the dispatch reg0).
    pub(super) pc_pinned: bool,
    /// Name → register for bindings that existed in the enclosing lowerer.
    ///
    /// This lowerer's bindings are not copied back (`lower_branch_expr`,
    /// `lower_loop_body`). `flowcontext.py` `FlowContext.mergeblock` /
    /// `framestate.py` `FrameState.union` give that join one variable, and
    /// `flatten.py` `GraphFlattener.insert_renamings` copies each predecessor
    /// into it. An assignment whose current register is still one of these
    /// moves into it. A straight-line lowerer leaves the map empty and
    /// rebinds. A `let` that shadows the name is absent here and rebinds too.
    pub(super) join_merge: HashMap<String, u16>,
    /// One-shot marker for "the statement `lower_stmt` is about to lower is
    /// the inline dispatch arm body's FINAL statement".  Set per-statement by
    /// `try_inline_dispatch_arm` and **consumed** (`mem::take`) at the top of
    /// `lower_stmt`, so it is observable for exactly one statement.
    ///
    /// Only a `return` in that position lowers to a typed return terminator.
    /// The take-on-entry is what bounds the scope: `lower_control` lowers an
    /// `if` / loop body's statements through `self.lower_stmt` on this same
    /// `Lowerer`, so a nested statement always observes `false` and a `return`
    /// there is rejected outright rather than having its operand lowered and
    /// its control transfer dropped.  Deliberately NOT copied into the nested
    /// `Lowerer`s built by `lower_control` / `lower_value` (unlike
    /// `pc_pinned`) for the same reason.
    pub(super) inline_arm_tail_stmt: bool,
}

/// Everything a failed lowering has to put back.
///
/// A lowerer that emits before it can know whether it will succeed —
/// every construct whose branch target must exist before its body is
/// lowered — owes the caller an unchanged stream when it refuses. What it
/// leaves behind otherwise is a `__builder.new_label()` with no
/// `mark_label`, and the jitcode assembler can only report that at the
/// end, as a label referenced by a `goto` that names no position.
pub(super) struct LowerSnapshot {
    statements: usize,
    op_metadata: usize,
    /// The prebuild accumulator is a second emission stream, so it is
    /// truncated with the first. An inline call pushes its helper's
    /// `-live-` prebuild call here and its op into `statements`, and the two
    /// are only meaningful together: a prebuild left behind by a refused
    /// lowering registers liveness offsets for a jitcode the parent no
    /// longer calls.
    inline_liveness_prebuild: usize,
    next_reg: u16,
    next_label: u16,
    bindings: HashMap<String, Binding>,
}

impl<'c> Lowerer<'c> {
    pub(super) fn new(config: Option<&'c LowererConfig>) -> Self {
        let call_policies = config.map(|cfg| cfg.calls.clone()).unwrap_or_default();
        Self::new_with_call_policies(config, call_policies, InferenceFailureMode::ReturnNone)
    }

    pub(super) fn new_with_call_policies(
        config: Option<&'c LowererConfig>,
        call_policies: Vec<(Vec<String>, CallPolicySpec)>,
        inference_failure_mode: InferenceFailureMode,
    ) -> Self {
        let mut this = Self {
            bindings: HashMap::new(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: 0,
            next_label: 0,
            config,
            call_policies,
            inference_failure_mode,
            auto_calls: config.map(|cfg| cfg.auto_calls).unwrap_or(false),
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            body_failure_reason: None,
            nested_failure_reasons: Vec::new(),
            opcode_var_name: None,
            in_dispatch_arm_body: false,
            sub_jitcode_identity_slot: None,
            dispatch_loop_label: None,
            pc_pinned: false,
            join_merge: HashMap::new(),
            inline_arm_tail_stmt: false,
        };
        this.install_vable_input_binding();
        this
    }

    /// Registers a nested lowerer must keep writing. The caller's bindings
    /// do not come back from `lower_branch_expr` or `lower_loop_body`.
    pub(super) fn enclosing_join_merge(&self) -> HashMap<String, u16> {
        self.bindings
            .iter()
            .map(|(name, binding)| (name.clone(), binding.reg))
            .collect()
    }

    /// Consume the recorded refusal reason, falling back to the historical
    /// generic string when a refusal site has not been taught to record one.
    ///
    /// The fallback is deliberately the old wording: a site that still reports
    /// it is one this change has not reached, and keeping the exact string
    /// makes those sites greppable rather than disguising them as classified.
    pub(super) fn take_body_failure_reason(&mut self) -> String {
        let mut reasons: Vec<String> = Vec::new();
        // The parent's own refusals first, so the head is unchanged by
        // propagation: a child reason can extend the string, never re-head it.
        if let Some(existing) = self.body_failure_reason.take() {
            reasons.extend(existing.split(REFUSAL_SEPARATOR).map(str::to_string));
        }
        for carried in std::mem::take(&mut self.nested_failure_reasons) {
            reasons.extend(carried.split(REFUSAL_SEPARATOR).map(str::to_string));
        }
        // Same dedup rule the single-entry path uses, applied across the join:
        // a child re-visits statements the parent also walked, so without this
        // one blocker can appear twice and a count-based reader would take the
        // repeat for a second mechanism.
        let mut seen: Vec<&str> = Vec::new();
        let mut joined = String::new();
        for reason in &reasons {
            if seen.contains(&reason.as_str()) {
                continue;
            }
            seen.push(reason);
            if !joined.is_empty() {
                joined.push_str(REFUSAL_SEPARATOR);
            }
            joined.push_str(reason);
        }
        if joined.is_empty() {
            // Deliberately the old wording — see this function's doc.
            "arm body could not be lowered to a sub-JitCode".to_string()
        } else {
            joined
        }
    }

    /// Carry a refused child lowerer's diagnosis into this one.
    ///
    /// Call at every site that discards a child after it returned `None`. The
    /// `?` there propagates the failure faithfully and drops the reason with
    /// the child, which is the whole defect: control flow stays correct while
    /// the diagnosis is lost, so the arm reports only its outermost refusal and
    /// the statement that actually needs a lowering rule is never named.
    ///
    /// Transitive on purpose — a child's own carried reasons come up too, so
    /// depth does not decide whether a blocker is reportable.
    pub(super) fn absorb_nested_failure(&mut self, nested: &mut Lowerer<'_>) {
        if let Some(reason) = nested.body_failure_reason.take() {
            self.nested_failure_reasons.push(reason);
        }
        self.nested_failure_reasons
            .append(&mut nested.nested_failure_reasons);
        self.absorb_nested_identity_slot(nested);
    }

    /// Copy a child's recorded identity-slot access. Called when the child
    /// returns (success or failure): an `if`/`loop` body that fully lowers
    /// still named a portal red the callee cannot reach (`MIFrame.setup_call`).
    pub(super) fn absorb_nested_identity_slot(&mut self, nested: &mut Lowerer<'_>) {
        if self.sub_jitcode_identity_slot.is_none() {
            self.sub_jitcode_identity_slot = nested.sub_jitcode_identity_slot.take();
        }
    }

    /// Record the first identity-slot field a sub-JitCode body would
    /// `load/store_state_field*`. The emit continues so a body that otherwise
    /// lowers fails expansion after the statement loop; an arm that degrades
    /// for another reason keeps its abort stub.
    ///
    /// `MIFrame.setup_call` / `bhimpl_inline_call_*` copy declared args
    /// only. `warmspot.py` `make_virtualizable_infos` has one
    /// virtualizable per jitdriver, so leftover state scalars stay portal
    /// identity reds a callee cannot reach.
    pub(super) fn note_sub_jitcode_identity_slot(&mut self, field: &str, span: proc_macro2::Span) {
        if !self.in_dispatch_arm_body {
            return;
        }
        if self.sub_jitcode_identity_slot.is_none() {
            self.sub_jitcode_identity_slot = Some((field.to_string(), span));
        }
    }

    pub(super) fn install_vable_input_binding(&mut self) {
        let Some(config) = self.config else {
            return;
        };
        let (Some(vable_var), Some(vable_reg)) =
            (config.vable_var.as_ref(), config.vable_input_ref_reg)
        else {
            return;
        };
        self.bindings.insert(
            vable_var.clone(),
            Binding {
                reg: vable_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        self.next_reg = self.next_reg.max(vable_reg.saturating_add(1));
    }

    pub(super) fn vable_base_reg(&self) -> Option<u16> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let binding = self.bindings.get(vable_var)?;
        match binding.kind {
            BindingKind::Ref => Some(binding.reg),
            _ => None,
        }
    }

    pub(super) fn alloc_reg(&mut self) -> u16 {
        let reg = self.next_reg;
        self.next_reg = self.next_reg.saturating_add(1);
        reg
    }

    /// Allocate the ident for a new label.
    ///
    /// The counter is macro-side bookkeeping only — the runtime label comes
    /// from the `__builder.new_label()` statement `emit_aux` pushes beside the
    /// ident — so a gap in it costs nothing, right up to `u16::MAX`, where
    /// `saturating_add` stops moving and every further allocation hands back
    /// the same ident: labels then alias instead of the counter panicking.
    ///
    /// Rolling `next_label` back is sound only where the statements are
    /// truncated with it. The ident is already bound by the emitted
    /// `let __jit_label_N = __builder.new_label();`, so restoring the counter
    /// while that statement stands re-issues a live ident: the second `let`
    /// shadows the first, uses written before it bind to one runtime label and
    /// uses after it to another, and the first can be created without ever
    /// being marked. Every restore site pairs the two writes for that reason.
    ///
    /// A lowerer that fails below its first `alloc_label` with the labels
    /// already consumed therefore leaves the counter alone; it is put back by
    /// `try_inline_dispatch_arm`, which discards the statement stream in the
    /// same block.
    pub(super) fn snapshot(&self) -> LowerSnapshot {
        LowerSnapshot {
            statements: self.statements.len(),
            op_metadata: self.op_metadata.len(),
            inline_liveness_prebuild: self.inline_liveness_prebuild.len(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            bindings: self.bindings.clone(),
        }
    }

    /// Truncate the statements and roll the label counter back together, the
    /// pairing [`Self::alloc_label`] requires.
    pub(super) fn restore(&mut self, snapshot: LowerSnapshot) {
        self.statements.truncate(snapshot.statements);
        self.op_metadata.truncate(snapshot.op_metadata);
        self.inline_liveness_prebuild
            .truncate(snapshot.inline_liveness_prebuild);
        self.next_reg = snapshot.next_reg;
        self.next_label = snapshot.next_label;
        self.bindings = snapshot.bindings;
    }

    /// Run a lowering that emits before it can fail, and leave the stream
    /// untouched if it refuses.
    ///
    /// The stream is what is put back. The refusal-reason channels
    /// (`body_failure_reason`, `nested_failure_reasons`) are deliberately not:
    /// they are the diagnosis, and a refused attempt is often the only place
    /// the deepest blocker is ever named. [`Self::take_body_failure_reason`]
    /// joins them behind the reporting lowerer's own reason and dedups, so a
    /// reason carried out of a discarded attempt can add a clause but can
    /// never re-head the message — which is the same rule
    /// [`Self::absorb_nested_failure`] exists to enforce.
    pub(super) fn transactional<T>(
        &mut self,
        lower: impl FnOnce(&mut Self) -> Option<T>,
    ) -> Option<T> {
        let snapshot = self.snapshot();
        let lowered = lower(self);
        if lowered.is_none() {
            self.restore(snapshot);
        }
        lowered
    }

    pub(super) fn alloc_label(&mut self) -> syn::Ident {
        let label = self.next_label;
        self.next_label = self.next_label.saturating_add(1);
        format_ident!("__jit_label_{label}")
    }

    /// Emit an op token plus its parallel `OpMeta` entry, keeping
    /// `statements` and `op_metadata` index-aligned for the backward
    /// liveness walker.
    pub(super) fn emit_op(&mut self, meta: OpMeta, tokens: TokenStream) {
        self.statements.push(tokens);
        self.op_metadata.push(meta);
    }

    /// `-live-` after `inline_call_*` (`jtransform.py handle_regular_call`).
    /// Force-alives the portal identity prefix so
    /// `get_list_of_active_boxes(in_a_call=True)` captures those reds on
    /// the caller frame.
    pub(super) fn emit_post_inline_live_marker(&mut self, tokens: TokenStream) {
        // Arm sub-JitCodes address portal identity through `frames[0]`;
        // those slots are not this frame's registers, so they must not
        // appear in this jitcode's `-live-` (`get_list_of_active_boxes`
        // would read uninitialized arm-frame slots). The parent dispatch
        // INLINE_CALL's trailing `-live-` keeps them on the portal frame.
        let reads = if self.in_dispatch_arm_body {
            Vec::new()
        } else {
            self.config
                .map(|c| c.identity_slot_registers())
                .unwrap_or_default()
        };
        self.emit_op(OpMeta::live_marker_with(reads, Vec::new()), tokens);
    }

    /// Identity-slot register listed as a `load_state_field*` use, or
    /// empty when this body is an arm sub-JitCode (the slot lives on the
    /// portal frame).
    pub(super) fn identity_use(&self, slot: Register) -> Vec<Register> {
        if self.in_dispatch_arm_body {
            Vec::new()
        } else {
            vec![slot]
        }
    }

    /// `jtransform.py promote_greens` — the `-live-` + `<kind>_guard_value`
    /// pair that pins one green to a constant.
    ///
    /// Both of upstream's `promote_greens` call sites emit exactly this shape:
    /// `handle_jit_marker__jit_merge_point` before `jit_merge_point`, and
    /// `handle_recursive_call` before `recursive_call_<kind>`.  The `-live-`
    /// comes first so the codewriter's per-marker liveness analysis records the
    /// alive set at the guard.
    pub(super) fn emit_live_and_guard_value(&mut self, kind: BindingKind, reg: u16) {
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let (read, guard) = match kind {
            BindingKind::Int => (
                Register::int(reg),
                quote! { __builder.int_guard_value(#reg); },
            ),
            BindingKind::Ref => (
                Register::ref_(reg),
                quote! { __builder.ref_guard_value(#reg); },
            ),
            BindingKind::Float => (
                Register::float(reg),
                quote! { __builder.float_guard_value(#reg); },
            ),
        };
        self.emit_op(
            OpMeta::linear(OpKind::GuardValue, vec![read], vec![]),
            guard,
        );
    }

    pub(super) fn append_lowered_sequence(&mut self, lowered: LoweredSequence) {
        debug_assert_eq!(
            lowered.statements.len(),
            lowered.op_metadata.len(),
            "RPython ssarepr.insns parity requires branch statements and metadata to append together"
        );
        self.statements.extend(lowered.statements);
        self.op_metadata.extend(lowered.op_metadata);
    }

    pub(super) fn emit_label_def(&mut self, label: &Ident) {
        self.emit_op(
            OpMeta::label_def(label.clone()),
            quote! { __builder.mark_label(#label); },
        );
    }

    pub(super) fn emit_jump(&mut self, target: &Ident) {
        self.emit_op(
            OpMeta::jump(target.clone()),
            quote! { __builder.jump(#target); },
        );
    }

    pub(super) fn emit_conditional_guard(&mut self, cond_reg: u16, target: &Ident) {
        self.emit_conditional_guard_negatable(cond_reg, target, false, false, BindingKind::Int);
    }

    /// `optimize_goto_if_not` on `int_eq(disc, const)`: fall through when
    /// they match, jump to `miss` otherwise. Zero is `_rewrite_equality`
    /// → `goto_if_not_int_is_zero`.
    pub(super) fn emit_fused_int_eq_miss(
        &mut self,
        disc_reg: u16,
        value_tok: &TokenStream,
        miss: &Ident,
    ) {
        // `flatten.insert_exits` puts `-live-` immediately before the
        // fused `goto_if_not_*`. The constant stays in the branch
        // (`assembler.py` `emit_const`), so nothing is loaded ahead of it.
        let is_zero = value_tok.to_string().replace(' ', "") == "0";
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        if !is_zero {
            self.emit_op(
                OpMeta::conditional_guard(Register::int(disc_reg), miss.clone()),
                quote! {
                    __builder.goto_if_not_int_const(majit_ir::OpCode::IntEq, #disc_reg, #value_tok, #miss);
                },
            );
        } else {
            self.emit_op(
                OpMeta::conditional_guard(Register::int(disc_reg), miss.clone()),
                quote! { __builder.goto_if_not_int_is_zero(#disc_reg, #miss); },
            );
        }
    }

    /// The branch RPython writes as `goto_if_not_<opname>`: fall through
    /// when the condition holds, take `target` otherwise.
    ///
    /// `negated` selects the `int_is_zero` form, which is the shape `!cond`
    /// reaches flatten as. `jtransform.py` renames `bool_not` to
    /// `int_is_zero`, and `optimize_goto_if_not` folds that operation into
    /// the block's exitswitch rather than materialising its result, so the
    /// negation costs a different branch opname and nothing else.
    ///
    /// `int_is_true` is the integer `!= 0` rewrite. A Rust `if`/`while`
    /// condition is otherwise `bool`, so flatten emits plain `goto_if_not`.
    ///
    /// A Ref binding is the `_rewrite_equality` fold of `ptr_eq` /
    /// `ptr_ne` against null (`ptr_iszero` / `ptr_nonzero`); flatten
    /// writes `goto_if_not_ptr_iszero/rL` / `goto_if_not_ptr_nonzero/rL`.
    pub(super) fn emit_conditional_guard_negatable(
        &mut self,
        cond_reg: u16,
        target: &Ident,
        negated: bool,
        int_is_true: bool,
        kind: BindingKind,
    ) {
        let (branch, register) = if matches!(kind, BindingKind::Ref) {
            let branch = if negated {
                format_ident!("goto_if_not_ptr_iszero")
            } else {
                format_ident!("goto_if_not_ptr_nonzero")
            };
            (branch, Register::ref_(cond_reg))
        } else {
            let branch = if negated {
                format_ident!("goto_if_not_int_is_zero")
            } else if int_is_true {
                format_ident!("goto_if_not_int_is_true")
            } else {
                format_ident!("goto_if_not")
            };
            (branch, Register::int(cond_reg))
        };
        // `assembler.py write_insn` takes a `Register` operand's argcode
        // from `x.kind[0]`; encode the bank into the metadata `Register`
        // so the liveness walker keeps a Ref truth-test under Ref.
        self.emit_op(
            OpMeta::conditional_guard(register, target.clone()),
            quote! { __builder.#branch(#cond_reg, #target); },
        );
    }

    /// Emit the exitswitch shape chosen by `jtransform.py`'s
    /// `optimize_goto_if_not`.
    pub(super) fn emit_lowered_condition_guard(
        &mut self,
        condition: &LoweredCondition,
        target: &Ident,
    ) {
        match condition {
            LoweredCondition::Value {
                binding,
                negated,
                int_is_true,
            } => {
                self.emit_conditional_guard_negatable(
                    binding.reg,
                    target,
                    *negated,
                    *int_is_true,
                    binding.kind,
                );
            }
            LoweredCondition::Compare { lhs, rhs, branch } => {
                let lhs_reg = Register::new(lhs.kind, lhs.reg);
                let rhs_reg = Register::new(rhs.kind, rhs.reg);
                let lhs_index = lhs.reg;
                let rhs_index = rhs.reg;
                self.emit_op(
                    OpMeta::conditional_guard_compare(lhs_reg, rhs_reg, target.clone()),
                    quote! { __builder.#branch(#lhs_index, #rhs_index, #target); },
                );
            }
            LoweredCondition::CompareConst { lhs, value, opcode } => {
                let lhs_index = lhs.reg;
                self.emit_op(
                    OpMeta::conditional_guard(Register::int(lhs_index), target.clone()),
                    quote! {
                        __builder.goto_if_not_int_const(majit_ir::OpCode::#opcode, #lhs_index, #value, #target);
                    },
                );
            }
        }
    }

    /// Emit a builder-side aux statement (no BC_* op, no def/use).
    pub(super) fn emit_aux(&mut self, tokens: TokenStream) {
        self.emit_op(OpMeta::aux(), tokens);
    }

    pub(super) fn inference_failure_tokens(&self, message: &str) -> TokenStream {
        match self.inference_failure_mode {
            InferenceFailureMode::ReturnNone => quote! { return None; },
            InferenceFailureMode::Panic => {
                let message = message.to_string();
                quote! { panic!(#message); }
            }
        }
    }

    pub(super) fn resolve_call_policy(&self, func: &Expr) -> Option<CallPolicySpec> {
        let auto_calls =
            self.auto_calls || matches!(self.inference_failure_mode, InferenceFailureMode::Panic);
        resolve_call_policy_for_probe(func, &self.call_policies, auto_calls)
    }

    /// Resolve the cond_call / record_known_result helper policy for
    /// `func`, falling back to `inferred_default` when the helper has a
    /// `helper_policy_path` but no explicit `calls={{ helper => ... }}`
    /// entry — RPython's `getcalldescr` (`call.py`) derives
    /// `extraeffect` from the call graph regardless of any user
    /// annotation, so a missing explicit policy must not crash.
    ///
    /// Returns `(kind, is_inferred)`.  The `kind` is the
    /// `CallPolicyKind` to drive expansion-time decisions (result-kind
    /// dispatch, wrapped-vs-direct registration shape, the
    /// `record_known_result!` elidable assert).  When `is_inferred` is
    /// true, the runtime `__policy` byte from the helper's
    /// `_jit_helper_policy` accessor reflects the actual analyzer
    /// outcome, and the registration code below picks the matching
    /// `EffectInfoSlot` at runtime instead of trusting the static
    /// default.
    ///
    /// Panics only when no helper-policy path exists at all — in that
    /// case the macro literally cannot register the function pointer.
    pub(super) fn cond_call_policy_or_inferred_default(
        &self,
        func: &Expr,
        macro_name: &str,
        inferred_default: crate::jit_interp::CallPolicyKind,
    ) -> (crate::jit_interp::CallPolicyKind, bool) {
        match self.resolve_call_policy(func) {
            Some(CallPolicySpec::Explicit(kind)) => (kind, false),
            Some(CallPolicySpec::Infer) => (inferred_default, true),
            None => {
                panic!(
                    "{macro_name} cannot resolve a helper policy for the callee — \
                     no `calls={{ helper => ... }}` entry and no `_jit_helper_policy` \
                     accessor on the function path"
                );
            }
        }
    }

    pub(super) fn cond_call_slot_for_policy(
        &self,
        kind: crate::jit_interp::CallPolicyKind,
        macro_name: &str,
    ) -> CondCallEffectSlot {
        call_policy_effect_slot(kind).unwrap_or_else(|| {
            panic!(
                "{macro_name} cannot lower helper policy {kind:?}: RPython \
                 jtransform.py:1677 rejects conditional_call / record_known_result \
                 callees whose calldescr forces virtuals or uses release-gil, and \
                 inline helpers do not have a direct-call calldescr"
            )
        })
    }

    pub(super) fn call_target_registration_tokens(
        &self,
        func: &Expr,
        kind: crate::jit_interp::CallPolicyKind,
        slot: CondCallEffectSlot,
        is_inferred: bool,
        inferred_policy_check: Option<TokenStream>,
        word_result_addr: Option<TokenStream>,
        func_arity: usize,
    ) -> TokenStream {
        let static_slot_token = slot.token();
        // For `Infer` mode, the helper's `_jit_helper_policy` byte is
        // the macro-time stand-in for RPython's `_canraise` /
        // `_elidable_function_` / `_jit_loop_invariant_` analyzers
        // (`call.py getcalldescr`).  Map it to the matching
        // `EffectInfoSlot` at runtime so an auto-discovered
        // `#[elidable_cannot_raise]` helper used without an explicit
        // `calls = { ... }` entry still registers an
        // `ElidableCannotRaise` slot — matching what an explicit
        // policy would have given.
        //
        // Bytes are allocated by `helper_policy_tokens_for_fn`
        // (`majit-macros/src/lib.rs`):
        //   1u8/2u8 — `dont_look_inside` Void/Int (`Plain`).
        //   3u8 — `elidable` Int.
        //   17u8/18u8 — `jit_loop_invariant` Void/Int.
        //   19u8 — `elidable_cannot_raise` Int.
        //   20u8 — `elidable_or_memerror` Int.
        //   21u8/22u8/23u8 — `elidable*` Ref.
        //   24u8 — `jit_loop_invariant` Ref.
        //   25u8 — `dont_look_inside` Ref (`Plain`).
        //   26u8 — Ref `MayForce` (rejected here).
        //   9u8/10u8 — `MayForce` (rejected by `cond_call_slot_for_policy`'s
        //   `jtransform.py` `_rewrite_op_cond_call` gate, but reach here at runtime — panic
        //   to match). Slots 13 and 14 stay reserved.
        // Unknown bytes (including `0u8` "unsupported") are rejected by
        // the call-site-specific inferred policy check before this slot is
        // used.  The fallback is kept only for defensive expansion.
        let slot_expr = if is_inferred {
            quote! {
                match __policy {
                    #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE => {
                        majit_metainterp::EffectInfoSlot::CanRaise
                    }
                    // `call.py getcalldescr` non-elidable EF_CANNOT_RAISE
                    // (`#[dont_look_inside_cannot_raise]` opt-in for void/int/ref).
                    #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                        majit_metainterp::EffectInfoSlot::CannotRaise
                    }
                    #INT_ELIDABLE | #REF_ELIDABLE => majit_metainterp::EffectInfoSlot::ElidableCanRaise,
                    #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #REF_LOOP_INVARIANT => {
                        majit_metainterp::EffectInfoSlot::LoopInvariant
                    }
                    #INT_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_CANNOT_RAISE => {
                        majit_metainterp::EffectInfoSlot::ElidableCannotRaise
                    }
                    #INT_ELIDABLE_OR_MEMERROR | #REF_ELIDABLE_OR_MEMERROR => {
                        majit_metainterp::EffectInfoSlot::ElidableOrMemerror
                    }
                    #VOID_MAY_FORCE | #INT_MAY_FORCE | #REF_MAY_FORCE => panic!(
                        "conditional_call! / conditional_call_elidable! / record_known_result! \
                         cannot dispatch MayForce callees \
                         (jtransform.py:1677 _rewrite_op_cond_call assert)",
                    ),
                    _ => #static_slot_token,
                }
            }
        } else {
            static_slot_token
        };
        if call_policy_is_wrapped(kind) {
            let policy_path =
                helper_policy_path(func).expect("wrapped helper policy requires a path expression");
            let inferred_policy_check = inferred_policy_check.unwrap_or_else(|| quote! {});
            quote! {
                let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                #inferred_policy_check
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
                    #slot_expr,
                    __save_err,
                );
            }
        } else {
            // An int-result target is registered through its widening shim;
            // see `word_result_addr_tokens`.  A void- or ref-result target has
            // no narrow return to widen and registers its own address.
            //
            // Only the shim is word-spelled, so only the shim is registered as
            // such: the helper's own address is whatever its Rust signature
            // says, and a `usize` or `&T` parameter there is narrower than the
            // word a compiled call passes.
            match word_result_addr {
                Some(shim_addr) => quote! {
                    let __fn_idx =
                        __builder.add_word_abi_fn_ptr_with_slot(#shim_addr, #slot_expr);
                },
                None => {
                    // A Ref target registered here is the raw function.
                    // `llmodel.py bh_call_r` reads one pointer word, so the
                    // return must be that wide. Void has no result word.
                    let ref_word = if call_policy_result_kind(kind) == Some(CallResultKind::Ref) {
                        assert_ref_result_one_word(func, func_arity)
                    } else {
                        quote! {}
                    };
                    quote! {
                        #ref_word
                        let __fn_idx =
                            __builder.add_fn_ptr_with_slot(#func as *const (), #slot_expr);
                    }
                }
            }
        }
    }
}
