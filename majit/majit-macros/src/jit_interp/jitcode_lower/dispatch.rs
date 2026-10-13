use super::*;
use crate::jit_interp::codegen_trace::call_is_insn_op_at_pc;

#[cfg(test)]
mod find_dispatch_loop_body_tests {
    use super::*;

    fn fn_block_from(src: &str) -> syn::Block {
        let item: syn::ItemFn = syn::parse_str(&format!("fn f() {{ {} }}", src)).unwrap();
        *item.block
    }

    fn first_match(block: &syn::Block) -> &syn::ExprMatch {
        fn find_in(stmt: &syn::Stmt) -> Option<&syn::ExprMatch> {
            match stmt {
                syn::Stmt::Expr(syn::Expr::Match(m), _) => Some(m),
                syn::Stmt::Expr(syn::Expr::While(w), _) => w.body.stmts.iter().find_map(find_in),
                syn::Stmt::Expr(syn::Expr::Loop(l), _) => l.body.stmts.iter().find_map(find_in),
                _ => None,
            }
        }
        block
            .stmts
            .iter()
            .find_map(find_in)
            .expect("no match in block")
    }

    #[test]
    fn null_mut_is_not_an_inferred_call() {
        let mut lowerer = Lowerer::new(None);
        lowerer.auto_calls = true;
        let body: syn::Expr = syn::parse_quote! {
            { p = core::ptr::null_mut(); }
        };
        assert!(
            !lowerer.arm_body_has_infer_call(&body),
            "Constant(nullptr) must not keep the arm off the dispatch JitCode"
        );
        let helper: syn::Expr = syn::parse_quote! {
            { helper(p) }
        };
        assert!(lowerer.arm_body_has_infer_call(&helper));
    }

    #[test]
    fn inline_pipeline_call_is_an_infer_call() {
        let policies = vec![(
            vec!["pipeline_double".to_string()],
            CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlinePipelineInt),
        )];
        let body: syn::Expr = syn::parse_quote! {
            { state.a = pipeline_double(state.a); }
        };
        assert!(expr_has_infer_or_pipeline_call(&body, &policies, false));
        let nop: syn::Expr = syn::parse_quote! {
            { state.a = state.a + 1; }
        };
        assert!(!expr_has_infer_or_pipeline_call(&nop, &policies, false));
    }

    #[test]
    fn finds_while_body() {
        let blk = fn_block_from("while x < 10 { match op { 0 => {}, _ => {} } }");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_some());
    }

    #[test]
    fn finds_loop_body() {
        let blk = fn_block_from("loop { match op { 0 => {}, _ => break } }");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_some());
    }

    #[test]
    fn returns_none_when_no_loop() {
        let blk = fn_block_from("match op { 0 => {}, _ => {} };");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_none());
    }
}

/// Walk a dispatch arm body and collect the parent-scope idents it
/// references, paired with the parent's [`Binding`] for each.
///
/// `pyopcode.py` keeps `oparg` / `next_instr` etc. as flowgraph
/// variables shared between the dispatch loop and the per-opcode
/// handler bodies; `jtransform.py:480 inline_call_<types>(jitcode,
/// args...)` then threads those variables as call args so the callee
/// jitcode sees them via its portal-input bindings.  Pyre's dispatch
/// arm lowerer emits `__builder.inline_call(__sub_idx)` with no args,
/// leaving the sub-frame without `pc` / `program` / `op` etc. — any
/// arm body that references these (`program.get_operand(pc - 1)`, say)
/// lowers to raw Rust that fails to compile inside
/// `__dispatch_jitcode_<fn>` where the names are out of scope.
///
/// This collector is the first slice (of three) closing that gap:
///   1. (here) identify the caller-locals the arm body references;
///   2. wire the sub-Lowerer to pre-bind those names as portal-input
///      bindings so `lower_value_expr` can resolve them;
///   3. emit `inline_call_<types>_v(__sub_idx, args...)` with the
///      caller's regs paired against the callee's portal-input slots.
///
/// **Field/method recognition**: `state.selected` only contributes
/// `state` (skipped because state isn't a parent binding); `expr.field`
/// only visits `expr`; `expr.method(args)` only visits the receiver +
/// args, never the method ident.  Multi-segment paths (`lj::stack_add`)
/// are skipped — only single-segment idents can match a parent binding.
///
/// **Local-binding suppression**: `let x = expr;` adds `x` to a local
/// suppression set so subsequent `x` references in the same arm are NOT
/// reported as caller-locals (the local shadows the caller scope).
/// Scope tracking is intentionally flat — over-suppression in nested
/// blocks is acceptable since the consumer just gets fewer
/// args, which is safe.  Under-suppression (let in inner block missed
/// by outer scope) is fine — sub-Lowerer ignores extra portal inputs.
/// Collect every identifier the pattern would bind, recursing into
/// `Pat::TupleStruct(OP_PUSH(value))` / `Pat::Tuple((a, b))` /
/// `Pat::Struct(Foo { x })` / `Pat::Or(A | B)` / `Pat::Reference(&x)`
/// /etc. so the dispatch arm pattern's bound names — distinct
/// flowgraph variables in PyPy `flowspace`/`SpaceOperation` parlance —
/// are shadowed in caller-local probes (`collect_arm_caller_locals`)
/// and runtime-constant-fallback gates (`expr_references_any_binding`'s
/// `visit_expr_match` arm-scope handling).
pub(super) fn collect_pat_bound_idents(pat: &Pat, out: &mut HashSet<String>) {
    match pat {
        Pat::Ident(pi) => {
            out.insert(pi.ident.to_string());
            if let Some((_, sub)) = &pi.subpat {
                collect_pat_bound_idents(sub, out);
            }
        }
        Pat::TupleStruct(ts) => {
            for p in &ts.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Tuple(tu) => {
            for p in &tu.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Struct(ps) => {
            for field in &ps.fields {
                collect_pat_bound_idents(&field.pat, out);
            }
        }
        Pat::Or(po) => {
            // PyPy `pyopcode.py:179` arm patterns carrying `Or`
            // (`A | B`) require both alternatives to bind the same
            // names; visiting any one suffices, but visiting all
            // is also safe — names dedupe in the HashSet.
            for p in &po.cases {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Reference(pr) => collect_pat_bound_idents(&pr.pat, out),
        Pat::Slice(ss) => {
            for p in &ss.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Type(pt) => collect_pat_bound_idents(&pt.pat, out),
        Pat::Paren(pp) => collect_pat_bound_idents(&pp.pat, out),
        Pat::Range(_)
        | Pat::Lit(_)
        | Pat::Path(_)
        | Pat::Wild(_)
        | Pat::Rest(_)
        | Pat::Const(_)
        | Pat::Macro(_)
        | Pat::Verbatim(_) => {}
        // syn::Pat is non_exhaustive — be conservative on future
        // additions (no bindings extracted).
        _ => {}
    }
}

pub(super) fn collect_arm_caller_locals(
    arm_body: &syn::Expr,
    arm_pat: &Pat,
    parent_bindings: &HashMap<String, Binding>,
) -> Vec<(String, Binding)> {
    use syn::visit::Visit;

    struct Collector<'a> {
        parent_bindings: &'a HashMap<String, Binding>,
        local_binds: HashSet<String>,
        visited: HashSet<String>,
        result: Vec<(String, Binding)>,
    }

    impl<'ast> Visit<'ast> for Collector<'_> {
        fn visit_expr_path(&mut self, p: &'ast ExprPath) {
            if p.qself.is_some() || p.path.segments.len() != 1 {
                return;
            }
            let seg = &p.path.segments[0];
            if !seg.arguments.is_none() {
                return;
            }
            let name = seg.ident.to_string();
            if self.local_binds.contains(&name) {
                return;
            }
            if !self.visited.insert(name.clone()) {
                return;
            }
            if let Some(binding) = self.parent_bindings.get(&name) {
                self.result.push((name, binding.clone()));
            }
        }

        fn visit_expr_field(&mut self, ef: &'ast syn::ExprField) {
            // Only the base expression is a user expression; the field
            // member ident is a struct-layout name, not a free var.
            self.visit_expr(&ef.base);
        }

        fn visit_expr_method_call(&mut self, mc: &'ast ExprMethodCall) {
            self.visit_expr(&mc.receiver);
            // Skip `mc.method` (the method ident) and `mc.turbofish`
            // (which carries syn::AngleBracketedGenericArguments — pure
            // type machinery, never references caller-scope idents).
            for arg in &mc.args {
                self.visit_expr(arg);
            }
        }

        fn visit_local(&mut self, local: &'ast Local) {
            // Visit the RHS first so any caller-local it references is
            // collected before the binding name shadows them.
            if let Some(init) = &local.init {
                self.visit_expr(&init.expr);
                // `let X = expr else { … };` else branch can use names
                // from the outer scope — visit it before adding `X`.
                if let Some((_, diverge)) = &init.diverge {
                    self.visit_expr(diverge);
                }
            }
            // Record the bound name(s) so subsequent `X` references in
            // the same flat scope are treated as locally-shadowed.
            collect_pat_bound_idents(&local.pat, &mut self.local_binds);
            // No auto-recursion fallback: we already visited init above.
        }
    }

    let mut collector = Collector {
        parent_bindings,
        local_binds: HashSet::new(),
        visited: HashSet::new(),
        result: Vec::new(),
    };
    // Pre-populate `local_binds` with names bound by the arm pattern
    // (`OP_PUSH(value)` binds `value`, `OP_TUPLE(a, b)` binds `a`/`b`,
    // etc.) so the visitor treats them as locally shadowed instead of
    // as free idents that could falsely match parent bindings.
    collect_pat_bound_idents(arm_pat, &mut collector.local_binds);
    collector.visit_expr(arm_body);
    collector.result
}

#[cfg(test)]
mod collect_arm_caller_locals_tests {
    use super::*;

    fn parent_bindings(entries: &[(&str, BindingKind, u16)]) -> HashMap<String, Binding> {
        entries
            .iter()
            .map(|(name, kind, reg)| {
                (
                    (*name).to_string(),
                    Binding {
                        reg: *reg,
                        kind: *kind,
                        depends_on_stack: false,
                        struct_type: None,
                    },
                )
            })
            .collect()
    }

    fn arm_body(src: &str) -> syn::Expr {
        // The collector takes an Expr. Wrap as a block expression so
        // multi-stmt arm bodies (let + tail expr) parse cleanly.
        syn::parse_str::<syn::Expr>(src).expect("arm body must parse as Expr")
    }

    /// Default arm pattern for tests that don't exercise pattern
    /// bindings — `_` wildcard binds nothing.
    fn wildcard_pat() -> syn::Pat {
        syn::parse_quote!(_)
    }

    fn names(result: &[(String, Binding)]) -> Vec<String> {
        result.iter().map(|(n, _)| n.clone()).collect()
    }

    #[test]
    fn collects_single_ident_match_in_method_call_args() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 0),
            ("program", BindingKind::Ref, 0),
        ]);
        let body = arm_body("{ program.get_operand(pc - 1) }");
        let pat = wildcard_pat();
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["pc".to_string(), "program".to_string()]);
    }

    #[test]
    fn skips_field_member_idents() {
        let bindings = parent_bindings(&[("state", BindingKind::Ref, 0)]);
        let body = arm_body("{ state.selected }");
        let pat = wildcard_pat();
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        // `selected` is a field name, not a parent binding — only
        // `state` should be picked up (and it IS in parent_bindings).
        assert_eq!(collected, vec!["state".to_string()]);
    }

    #[test]
    fn skips_method_ident_keeps_receiver_and_args() {
        let bindings =
            parent_bindings(&[("state", BindingKind::Ref, 0), ("v", BindingKind::Int, 0)]);
        let body = arm_body("{ state.push(v) }");
        let pat = wildcard_pat();
        // `push` is the method ident — must NOT appear in parent_bindings
        // probes; only `state` (receiver) and `v` (arg) match.
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["state".to_string(), "v".to_string()]);
    }

    #[test]
    fn skips_multi_segment_paths() {
        let bindings = parent_bindings(&[("stack_add", BindingKind::Int, 0)]);
        let body = arm_body("{ lj::stack_add(state.selected_ref) }");
        let pat = wildcard_pat();
        // `lj::stack_add` is a 2-segment path — must NOT match the
        // single-segment `stack_add` parent binding.  `state` is also
        // not in parent_bindings here, so result is empty.
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert!(collected.is_empty(), "got {:?}", collected);
    }

    #[test]
    fn local_binding_shadows_caller_scope() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 0),
            ("program", BindingKind::Ref, 0),
        ]);
        let body = arm_body("{ let value = program.get_operand(pc - 1); state.s.push(value); }");
        let pat = wildcard_pat();
        // RHS `program.get_operand(pc - 1)` is visited BEFORE `value`
        // joins local_binds, so `program` and `pc` are picked up.
        // Subsequent `value` reference does NOT appear in
        // parent_bindings, so it never enters the result; even if it
        // did, the local-bind suppression would skip it.
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["pc".to_string(), "program".to_string()]);
    }

    #[test]
    fn dedupes_repeated_idents() {
        let bindings = parent_bindings(&[("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ pc + pc + pc }");
        let pat = wildcard_pat();
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn skips_idents_not_in_parent_bindings() {
        let bindings = parent_bindings(&[("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ x + y + pc }");
        let pat = wildcard_pat();
        // `x` and `y` are not parent bindings — only `pc` is collected.
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn arm_pattern_bound_name_shadows_parent_binding() {
        // Round-3 line-by-line PyPy parity probe (`flowspace` /
        // `SpaceOperation` distinguishes pattern-bound from outer-scope
        // variables).  An arm pattern `OP_PUSH(value)` binds `value`
        // locally; even if `value` happens to be a parent binding, the
        // arm body's reference to `value` is the pattern's payload, NOT
        // the caller-frame value.  Walker must skip pattern-bound names.
        let bindings =
            parent_bindings(&[("value", BindingKind::Int, 5), ("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ state.s.push(value as i64); pc }");
        // `OP_PUSH(value)` — pattern binds `value`.
        let pat: syn::Pat = syn::parse_quote!(OP_PUSH(value));
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        // `value` is pattern-bound — must NOT be collected even though
        // parent_bindings contains it.  `pc` is a free var in the body
        // matching parent_bindings — IS collected.
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn arm_pattern_or_alternatives_share_bound_names() {
        // PyPy `pyopcode.py:179` `Or` patterns (`A | B`) require both
        // alternatives to bind the same name; pyre walks all to be
        // robust to syntactic variants.  Either side's binding suffices
        // to suppress the name in the body probe.
        let bindings = parent_bindings(&[("target", BindingKind::Int, 3)]);
        let body = arm_body("{ pc = target; }");
        let pat: syn::Pat = syn::parse_quote!(OP_JMP(target) | OP_BRANCH(target));
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert!(collected.is_empty(), "got {:?}", collected);
    }

    #[test]
    fn collected_binding_carries_kind_and_reg() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 7),
            ("program", BindingKind::Ref, 3),
        ]);
        let body = arm_body("{ program.get_op(pc) }");
        let pat = wildcard_pat();
        let result = collect_arm_caller_locals(&body, &pat, &bindings);
        let pc = result
            .iter()
            .find(|(n, _)| n == "pc")
            .expect("pc collected");
        assert_eq!(pc.1.reg, 7);
        assert!(matches!(pc.1.kind, BindingKind::Int));
        let program = result
            .iter()
            .find(|(n, _)| n == "program")
            .expect("program collected");
        assert_eq!(program.1.reg, 3);
        assert!(matches!(program.1.kind, BindingKind::Ref));
    }

    /// The portal env binding is the one `lower_ref_binding_getfield` consults.
    /// Without the env type, `program.field` never reaches
    /// `<T>::__MAJIT_IMMUTABLE_FIELDS` (`rclass.py _parse_field_list`).
    #[test]
    fn env_binding_carries_the_env_type() {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.state_type_name = "Machine".to_string();
        config.env_type_name = "Program".to_string();
        config.calls.push((
            vec!["insn_op".to_string()],
            CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::ElidableInt),
        ));
        let binding = super::portal_env_binding(&config.env_type_name);
        let struct_type = binding
            .struct_type
            .expect("env binding carries the env type");
        assert_eq!(quote::quote!(#struct_type).to_string(), "Program");
        assert!(matches!(binding.kind, BindingKind::Ref));

        let func: syn::ItemFn = syn::parse_quote! {
            fn mainloop(program: &Program) {
                loop {
                    let cap = program.cap;
                    jit_merge_point!(driver, program, pc; state);
                    let opcode = insn_op(program, pc);
                    match opcode {
                        0 => {}
                        _ => break,
                    }
                }
            }
        };
        let dispatch = crate::jit_interp::codegen_trace::find_dispatch_match(&func.block)
            .expect("opcode fetch must be the dispatch");
        let arms = crate::jit_interp::classify::classify_arms(&dispatch.arms);
        let generated = lower_dispatch_body(&config, &func.block, &arms, &func.sig.output)
            .expect("dispatch lowering must produce a body");
        let body = generated.body.to_string();
        assert!(
            body.contains("getfield_gc_i"),
            "program.cap must lower through lower_ref_binding_getfield: {body}"
        );
        assert!(
            body.contains("__MAJIT_IMMUTABLE_FIELDS"),
            "the env type's immutable-field list must be registered: {body}"
        );
    }
}
#[cfg(test)]
mod assign_caller_local_layout_tests {
    use super::*;

    fn caller_binding(kind: BindingKind, parent_reg: u16) -> Binding {
        Binding {
            reg: parent_reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    #[test]
    fn empty_input_yields_empty_layout_and_zero_advance() {
        let (layout, max_pre_bound) = assign_caller_local_layout(&[]);
        assert!(layout.is_empty());
        assert_eq!(max_pre_bound, 0);
    }

    #[test]
    fn per_bank_packed_callee_regs() {
        let caller_locals = vec![
            ("first_int".to_string(), caller_binding(BindingKind::Int, 5)),
            ("first_ref".to_string(), caller_binding(BindingKind::Ref, 9)),
            (
                "second_int".to_string(),
                caller_binding(BindingKind::Int, 6),
            ),
            (
                "second_ref".to_string(),
                caller_binding(BindingKind::Ref, 10),
            ),
            (
                "first_float".to_string(),
                caller_binding(BindingKind::Float, 2),
            ),
        ];
        let (layout, max_pre_bound) = assign_caller_local_layout(&caller_locals);
        // Per-bank packed: int → 0, 1; ref → 0, 1; float → 0.
        let by_name: std::collections::HashMap<String, (u16, BindingKind)> = layout
            .iter()
            .map(|l| (l.name.clone(), (l.callee_reg, l.kind)))
            .collect();
        assert_eq!(by_name["first_int"].0, 0u16);
        assert!(matches!(by_name["first_int"].1, BindingKind::Int));
        assert_eq!(by_name["second_int"].0, 1u16);
        assert!(matches!(by_name["second_int"].1, BindingKind::Int));
        assert_eq!(by_name["first_ref"].0, 0u16);
        assert!(matches!(by_name["first_ref"].1, BindingKind::Ref));
        assert_eq!(by_name["second_ref"].0, 1u16);
        assert!(matches!(by_name["second_ref"].1, BindingKind::Ref));
        assert_eq!(by_name["first_float"].0, 0u16);
        assert!(matches!(by_name["first_float"].1, BindingKind::Float));
        // Worst-case advance is the max per-bank slot count = 2 (int and ref).
        assert_eq!(max_pre_bound, 2);
    }

    #[test]
    fn parent_reg_is_preserved_in_layout() {
        let caller_locals = vec![
            ("pc".to_string(), caller_binding(BindingKind::Int, 7)),
            ("program".to_string(), caller_binding(BindingKind::Ref, 3)),
        ];
        let (layout, max_pre_bound) = assign_caller_local_layout(&caller_locals);
        let pc_layout = layout
            .iter()
            .find(|l| l.name == "pc")
            .expect("pc in layout");
        assert_eq!(pc_layout.parent_reg, 7);
        assert_eq!(pc_layout.callee_reg, 0);
        assert!(matches!(pc_layout.kind, BindingKind::Int));
        let program_layout = layout
            .iter()
            .find(|l| l.name == "program")
            .expect("program in layout");
        assert_eq!(program_layout.parent_reg, 3);
        // First Ref → ref_reg 0 (different bank from `pc` Int 0).
        assert_eq!(program_layout.callee_reg, 0);
        assert!(matches!(program_layout.kind, BindingKind::Ref));
        // One Int + one Ref pre-bound → next_reg advance is 1.
        assert_eq!(max_pre_bound, 1);
    }

    #[test]
    fn order_within_kind_matches_input() {
        let caller_locals = vec![
            ("a".to_string(), caller_binding(BindingKind::Int, 100)),
            ("b".to_string(), caller_binding(BindingKind::Int, 200)),
            ("c".to_string(), caller_binding(BindingKind::Int, 300)),
        ];
        let (layout, _) = assign_caller_local_layout(&caller_locals);
        // First-seen Int → 0, second → 1, third → 2.  The layout
        // preserves input order so the parent emit can pair
        // (parent_reg, callee_reg) deterministically.
        assert_eq!(layout[0].callee_reg, 0);
        assert_eq!(layout[0].parent_reg, 100);
        assert_eq!(layout[1].callee_reg, 1);
        assert_eq!(layout[1].parent_reg, 200);
        assert_eq!(layout[2].callee_reg, 2);
        assert_eq!(layout[2].parent_reg, 300);
    }
}

/// Walk `func_block` to find the dispatch while-loop, then lower every stmt
/// that appears before the dispatch match in source order.
///
/// interp_jit.py — stmts between jit_merge_point and the opcode
/// dispatch (e.g. `co_code = pycode.co_code`, `valuestackdepth = promote(...)`)
/// execute unconditionally before each dispatch. We lower them into the
/// dispatch JitCode body so the JIT sees them on every loop iteration.
///
/// Returns `Some(())` if at least one pre-dispatch stmt was found; `None` if
/// no while body or dispatch match could be located (caller continues anyway
/// since later tasks fill the dispatch chain).
pub(super) fn lower_pre_dispatch_stmts(
    lowerer: &mut Lowerer,
    func_block: &syn::Block,
) -> Option<()> {
    // Find the dispatch match expression anywhere in the function block.
    let dispatch_match = find_dispatch_match(func_block)?;

    // Find the dispatch loop body (while or loop) whose stmts contain the
    // dispatch match.
    let loop_body = find_dispatch_loop_body(func_block, dispatch_match)?;

    // A.3.6.1: pre-merge-point body-local `let` stmts are bound by
    // `bind_pre_merge_point_stmts` (called earlier in `lower_dispatch_body`).
    // This walker latches `seen_merge_point` upon reaching the macro stmt
    // and only processes stmts that appear AFTER it (avoiding a double-bind
    // collision with the pre-pass).
    let mut seen_merge_point = false;

    // Iterate stmts in the dispatch loop body before the stmt that contains
    // the match.
    for stmt in &loop_body.stmts {
        if stmt_contains_match(stmt, dispatch_match) {
            // Reached the dispatch site; no more pre-dispatch stmts.
            break;
        }
        // A.3.6.1: latch on the `jit_merge_point!()` macro stmt. Pre-merge
        // point stmts are handled by `bind_pre_merge_point_stmts`; the
        // post-merge-point body resumes for subsequent stmts.
        if is_jit_merge_point_macro(stmt) {
            seen_merge_point = true;
            continue;
        }
        if !seen_merge_point {
            continue;
        }
        // Try to lower opcode-fetch patterns before the
        // state-field filter so they are emitted as IR ops rather than
        // verbatim Rust (which would fail to compile inside
        // `__dispatch_jitcode_*` where `program`/`pc` are not in scope).
        if try_lower_opcode_fetch_stmt(lowerer, stmt) {
            continue;
        }
        // A.2.3a/b: inner `Expr::While` recognition + emission. RPython
        // `pyopcode.py:187-193` lays out the EXTENDED_ARG inner loop as
        // `while opcode == EXTENDED_ARG { ... }`. The proc macro is
        // token-only and cannot resolve constant integer values, so the
        // recognizer matches structural shape only: condition is
        // `<ident> == <ident>` (paren wrapping + reversed operands
        // accepted; one ident must already be in `lowerer.bindings` so
        // the OTHER is the const). When recognition succeeds, A.2.3b
        // emits the loop scaffold + body IR (per Pre-A.2.3 codex
        // BLOCKERs (c) merge arithmetic + (d) HAVE_ARGUMENT polarity).
        // When recognition or body emission fails, signal taint so
        // `lower_dispatch_body` returns `None` → dispatch body empty →
        // gate in `generate_state_fields_jit_state` misses
        // `BC_GETARRAYITEM_GC_I` and refuses install (fail-closed per
        // Pre-A.2.3 codex review item (a)).
        if let Some(while_expr) = stmt_as_inner_while(stmt) {
            if !is_recognized_extended_arg_while(while_expr) {
                lowerer.dispatch_tainted_reason = Some(
                    "A.2.3a fail-closed: inner Expr::While condition is not the \
                     recognized `<ident> == <ident>` shape (EXTENDED_ARG inner \
                     loop per pyopcode.py:187-193)",
                );
                return None;
            }
            if lower_extended_arg_inner_while(lowerer, while_expr).is_none() {
                lowerer.dispatch_tainted_reason = Some(
                    "A.2.3b fail-closed: inner Expr::While body could not be \
                     lowered to RPython EXTENDED_ARG IR (opcode2/arg2 fetch + \
                     HAVE_ARGUMENT range guard + pc += 2 + oparg merge + \
                     opcode reassign per pyopcode.py:188-193)",
                );
                return None;
            }
            continue;
        }
        // A.2.5.a: free-function call statement with resolvable helper
        // policy (e.g. `bytecode_only_trace_helper();` annotated with
        // `#[majit_macros::dont_look_inside]` under `auto_calls = true`).
        // RPython `pyopcode.py:174` `ec.bytecode_only_trace(self)` lowers
        // through `jtransform.py handle_residual_call rewrite_op` + `call.py:282-324  allow-line-citation
        // getcalldescr`'s analyzer trio at translation time. Pyre's
        // `resolve_call_policy` + `lower_config_call_stmt` is the
        // per-callsite equivalent before the analyzer
        // trio output to runtime helper-call sites. This recognizer is
        // a third path alongside opcode-fetch and state-modifying stmts
        // — it MUST NOT extend `stmt_modifies_jit_state`, which means
        // "touches lowered JIT state" (state-place reachability), a
        // different concept from "has analyzer-classified side effects"
        // (Pre-A.2.5 codex review BLOCKER 1).
        if try_lower_pre_dispatch_policy_call_stmt(lowerer, stmt) {
            continue;
        }
        // Only lower stmts that modify JIT state (e.g. state field writes,
        // promote calls on state fields) or contain a can_enter_jit! macro
        // (which emits BC_LOOP_HEADER). Skip remaining runtime-only stmts
        // that are neither opcode-fetch patterns nor state-modifying.
        if !lowerer.stmt_modifies_jit_state(stmt) && !stmt_contains_can_enter_jit(stmt) {
            continue;
        }
        let _ = lowerer.lower_stmt(stmt);
    }
    Some(())
}

/// Walk a statement AST to check if it contains a `can_enter_jit!` macro call
/// anywhere in its body (possibly nested inside if-blocks).
fn stmt_contains_can_enter_jit(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Macro(m) => {
            let path_str = m
                .mac
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            path_str == "can_enter_jit"
                || path_str.ends_with("::can_enter_jit")
                || path_str == "jit_loop_header"
                || path_str.ends_with("::jit_loop_header")
        }
        Stmt::Expr(expr, _) => expr_contains_can_enter_jit(expr),
        _ => false,
    }
}

fn expr_contains_can_enter_jit(expr: &Expr) -> bool {
    match expr {
        Expr::If(expr_if) => {
            block_contains_can_enter_jit(&expr_if.then_branch)
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, e)| expr_contains_can_enter_jit(e))
        }
        Expr::Block(expr_block) => block_contains_can_enter_jit(&expr_block.block),
        Expr::Match(expr_match) => expr_match
            .arms
            .iter()
            .any(|arm| expr_contains_can_enter_jit(&arm.body)),
        _ => false,
    }
}

fn block_contains_can_enter_jit(block: &syn::Block) -> bool {
    block.stmts.iter().any(stmt_contains_can_enter_jit)
}

/// A.2.5.a: lower a free-function call statement whose callee has a
/// resolvable helper policy (e.g. `#[majit_macros::dont_look_inside]`
/// under `auto_calls = true` or an explicit `calls = { ... }` entry).
/// Mirrors the dispatch-body equivalent of RPython `pyopcode.py`
/// `ec.bytecode_only_trace(self)`, whose effect class is at minimum
/// `DEFAULT_EFFECT_INFO` (EF_CAN_RAISE + saturated read/write descrs)
/// per Pre-A.2.5 codex review (`call.py getcalldescr` for the
/// upstream analyzer-trio classification; pyre's per-callsite hatch
/// before the analyzer outputs are plumbed).
///
/// Returns `true` if the stmt was consumed (lowered via
/// `lower_config_call_stmt`), `false` if the caller should continue
/// with the state-modifying gate or skip-silently fallback.
///
/// Recognises `Stmt::Expr(Expr::Call(_), _)` where the callee path
/// resolves to a `CallPolicySpec` via `resolve_call_policy`. The
/// existing `lower_config_call_stmt` (`jitcode_lower.rs`) handles
/// every `CallPolicyKind` (ResidualVoid / MayForceVoid / LoopInvariant
/// / Elidable / etc.); this recognizer is the gate that lets the same
/// path fire in the dispatch JitCode body, where `lower_stmt`'s
/// state-modifying filter would otherwise silently skip the call.
fn try_lower_pre_dispatch_policy_call_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(expr, _) = stmt else {
        return false;
    };
    let Expr::Call(call) = expr else {
        return false;
    };
    if lowerer.resolve_call_policy(&call.func).is_none() {
        return false;
    }
    lowerer.lower_config_call_stmt(expr).is_some()
}

/// Return the inner `ExprWhile` if `stmt` is `Stmt::Expr(Expr::While(_), _)`.
/// Used by `lower_pre_dispatch_stmts` to detect EXTENDED_ARG inner loops in
/// the dispatch body (RPython `pyopcode.py`).
fn stmt_as_inner_while(stmt: &Stmt) -> Option<&syn::ExprWhile> {
    let Stmt::Expr(Expr::While(while_expr), _) = stmt else {
        return None;
    };
    Some(while_expr)
}

/// Recognize the `while <ident> == <ident> { ... }` structural shape used
/// by RPython's EXTENDED_ARG inner loop (`pyopcode.py:187`
/// `while opcode == opcodedesc.EXTENDED_ARG.index`). The proc macro is
/// token-only and cannot resolve constant integer values, so this helper
/// validates only the AST shape:
///
/// - condition is `Expr::Binary { op: Eq, left, right }` (after stripping
///   any number of nested `Expr::Paren` wrappers)
/// - both `left` and `right` are bare single-segment `Expr::Path` idents
///   (no qualified paths, no leading `::`, no generics)
///
/// Either operand order is accepted (`opcode == EXTENDED_ARG` or
/// `EXTENDED_ARG == opcode`) because the macro cannot tell which side is
/// the constant.
fn is_recognized_extended_arg_while(while_expr: &syn::ExprWhile) -> bool {
    let cond = unwrap_expr_paren(&while_expr.cond);
    let Expr::Binary(bin) = cond else {
        return false;
    };
    if !matches!(bin.op, syn::BinOp::Eq(_)) {
        return false;
    }
    expr_single_ident(&bin.left).is_some() && expr_single_ident(&bin.right).is_some()
}

/// Strip any number of `Expr::Paren` wrappers from `expr`, returning the
/// innermost non-parenthesized expression.
fn unwrap_expr_paren(expr: &Expr) -> &Expr {
    let mut cur = expr;
    while let Expr::Paren(p) = cur {
        cur = &p.expr;
    }
    cur
}

/// Return the bound name from `let <pat> = ...` where `<pat>` is either
/// `Pat::Ident(name)` or `Pat::Type(Pat::Ident(name), <ty>)` (i.e.
/// `let X = ...` or `let X: T = ...`). Other pattern shapes (tuple,
/// struct, ref) are out of scope for the byte-fetch lowering.
fn pat_bound_ident_name(pat: &Pat) -> Option<String> {
    let inner = match pat {
        Pat::Type(pt) => pt.pat.as_ref(),
        other => other,
    };
    let Pat::Ident(pi) = inner else { return None };
    Some(pi.ident.to_string())
}

/// A.2.3b: lower a recognized EXTENDED_ARG inner while loop to dispatch
/// JitCode IR. Mirrors RPython `pyopcode.py`:
///
/// ```text
/// while opcode == opcodedesc.EXTENDED_ARG.index:
///     opcode = ord(co_code[next_instr])
///     arg    = ord(co_code[next_instr + 1])
///     if opcode < HAVE_ARGUMENT:
///         raise BytecodeCorruption
///     next_instr += 2
///     oparg = (oparg * 256) | arg
/// ```
///
/// Emit layout (single shared `opcode` register so the back-edge
/// re-tests against the freshly-fetched value):
///
/// ```text
///   load_const_i  EXTENDED_ARG_const_reg, EXTENDED_ARG  ;; hoisted
/// inner_loop_top:                                       ;; back-edge target
///   goto_if_not_int_eq  opcode_reg, EXTENDED_ARG_const_reg, after_loop
///   <body — opcode2/arg2 fetch with opcode2 aliased to opcode_reg per
///    RPython L188 reuses the same `opcode` variable + arg2 fresh; range
///    guard via goto_if_not_int_lt + abort + ok label; pc += 2 via
///    existing fetch helper; oparg merge via load_const_i(256) +
///    int_mul + int_or>
///   jump inner_loop_top
/// after_loop:
/// ```
///
/// `goto_if_not_int_eq` branches on FALSE
/// (`flatten.py:240-260`/`pyjitpl.py:510-522`), so the top-of-loop test
/// reaches `after_loop` precisely when `opcode != EXTENDED_ARG` —
/// matching RPython's `while opcode == EXTENDED_ARG: ...` semantic.
/// The unconditional `jump` at the bottom is the back-edge per Pre-A.2.3
/// codex review item (b): the back-edge target is the inner loop header
/// where the next iteration re-fetches opcode2/arg2/merge, NOT the outer
/// `loop_start_label`. No second `BC_JIT_MERGE_POINT` is emitted.
///
/// Returns `None` if any structural check fails (mismatched const ident
/// scope, body shape mismatch, missing required outer bindings) so the
/// caller can install the fail-closed gate.
fn lower_extended_arg_inner_while(
    lowerer: &mut Lowerer,
    while_expr: &syn::ExprWhile,
) -> Option<()> {
    // Identify which side of `<ident> == <ident>` is the bound local
    // (one of `lowerer.bindings` with Int kind) and which is the const.
    let (opcode_reg, extended_arg_const) = pick_local_and_const_idents(lowerer, &while_expr.cond)?;

    // Pre-pass: scan body for `<outer_local> = <inner_local>` reassigns
    // (e.g. `opcode = opcode2`) so we can alias the inner `let` of
    // `<inner_local>` to `<outer_local>`'s register. RPython L188 reuses
    // the same `opcode` variable; the Rust fixture uses `opcode2` plus
    // a trailing `opcode = opcode2` purely because `let` in a Rust block
    // shadows rather than rebinds.
    let mut inner_alias: HashMap<String, u16> = HashMap::new();
    for stmt in &while_expr.body.stmts {
        if let Some((lhs, rhs)) = match_simple_ident_assign(stmt)
            && let Some(b) = lowerer.bindings.get(&lhs)
            && matches!(b.kind, BindingKind::Int)
        {
            inner_alias.insert(rhs, b.reg);
        }
    }

    // Hoist `EXTENDED_ARG` constant load before the loop label so the
    // back-edge does not reload it every iteration. Using `as i64` lets
    // any module-level `const X: u8` (or `i32`/`u16`/etc.) widen to the
    // builder's `i64` argument cleanly.
    let extended_arg_const_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::LoadConstI,
            vec![],
            vec![Register::int(extended_arg_const_reg)],
        ),
        quote! {
            __builder.load_const_i_value(#extended_arg_const_reg as u16, #extended_arg_const as i64);
        },
    );

    let inner_loop_top = lowerer.alloc_label();
    let after_loop = lowerer.alloc_label();
    lowerer.emit_aux(quote! { let #inner_loop_top = __builder.new_label(); });
    lowerer.emit_aux(quote! { let #after_loop = __builder.new_label(); });
    lowerer.emit_label_def(&inner_loop_top);

    // jtransform.py optimize_goto_if_not fuses int_eq + goto_if_not into
    // goto_if_not_int_eq/iiL. `opcode_reg` is the canonical loop reg
    // updated each iteration by the inner BC_GETARRAYITEM_GC_I aliased
    // through `inner_alias`.
    //
    // flatten.py:258-260 emits -live- ahead of every goto_if_not. The
    // guard records through build_state_field_snapshot, which reads the
    // LIVE marker at guard_pc - SIZE_LIVE_OP; without a preceding -live-
    // the snapshot mis-decodes the prior op's bytes as a liveness offset.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote::quote! { let _ = __builder.live_placeholder(); },
    );
    lowerer.emit_op(
        OpMeta::conditional_guard_compare(
            Register::int(opcode_reg),
            Register::int(extended_arg_const_reg),
            after_loop.clone(),
        ),
        quote! {
            __builder.goto_if_not_int_eq(
                #opcode_reg as u16,
                #extended_arg_const_reg as u16,
                #after_loop,
            );
        },
    );

    lower_extended_arg_inner_while_body(lowerer, &while_expr.body, &inner_alias)?;

    lowerer.emit_jump(&inner_loop_top);
    lowerer.emit_label_def(&after_loop);
    Some(())
}

/// Walk the inner-while body's stmts and emit IR for each. Returns
/// `None` on any unrecognized stmt so the caller can fail-closed.
///
/// Recognized stmt shapes (RPython `pyopcode.py:188-193`):
///
/// 1. `let X = program[<idx>]` — opcode2/arg2 byte fetch. If `X` is in
///    `inner_alias`, the BC_GETARRAYITEM_GC_I writes into the aliased
///    register (RPython orthodoxy: `opcode = ord(co_code[next_instr])`
///    reuses the outer `opcode` slot). Otherwise allocates a fresh reg.
/// 2. `pc += N` — delegated to existing `try_lower_opcode_fetch_stmt`.
/// 3. `if X < CONST { panic!(...) }` — HAVE_ARGUMENT range guard.
///    Emits `load_const_i + goto_if_not_int_lt + abort + ok_label`.
///    `goto_if_not_int_lt(a, b, L)` branches when NOT(a < b), i.e.
///    when `a >= b` — so `a < b` fall-throughs into BC_ABORT, matching
///    RPython L190-191 `raise BytecodeCorruption` (Pre-A.2.3 codex
///    BLOCKER (d) polarity correction).
/// 4. `oparg = (<oparg> * 256) | <arg> as i64` — multi-byte oparg merge.
///    Emits `load_const_i 256 + int_mul + int_or` per Pre-A.2.3 codex
///    BLOCKER (c) (no `int_lshift_imm` in pyre; `jtransform.py:363-366`
///    leaves int_mul symmetric).
/// 5. `<outer> = <inner>` where `<inner>` is in `inner_alias` — no-op
///    (the alias was set up in pass 1; both names already point to the
///    outer register).
fn lower_extended_arg_inner_while_body(
    lowerer: &mut Lowerer,
    body: &syn::Block,
    inner_alias: &HashMap<String, u16>,
) -> Option<()> {
    for stmt in &body.stmts {
        if try_lower_inner_byte_fetch(lowerer, stmt, inner_alias) {
            continue;
        }
        if try_lower_opcode_fetch_stmt(lowerer, stmt) {
            continue;
        }
        if try_lower_have_argument_guard(lowerer, stmt) {
            continue;
        }
        if try_lower_oparg_merge_stmt(lowerer, stmt) {
            continue;
        }
        if try_lower_alias_assign_stmt(stmt, inner_alias) {
            continue;
        }
        return None;
    }
    Some(())
}

/// Match `Stmt::Local { pat: Pat::Ident(X), init: Some(program[<idx>]) }`.
/// If `X` is in `inner_alias`, emit `BC_GETARRAYITEM_GC_I_PURE` writing into
/// the aliased outer register and re-bind `X → outer_reg`. Otherwise
/// fall through to the existing `try_lower_opcode_fetch_stmt` (which
/// allocates a fresh register).
fn try_lower_inner_byte_fetch(
    lowerer: &mut Lowerer,
    stmt: &Stmt,
    inner_alias: &HashMap<String, u16>,
) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Some(lhs_name) = pat_bound_ident_name(&local.pat) else {
        return false;
    };
    let Some(&aliased_reg) = inner_alias.get(&lhs_name) else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    // Peel an outer `Expr::Cast` per `try_lower_opcode_fetch_stmt`'s
    // Pattern 1: `let X: i64 = program[idx] as i64` (a Rust widening
    // artifact) lowers identically to the bare form.
    let init_expr = match init.expr.as_ref() {
        Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    let Expr::Index(idx) = init_expr else {
        return false;
    };
    if expr_single_ident(&idx.expr).as_deref() != Some("program") {
        return false;
    }
    let Some(prog) = lowerer.bindings.get("program").cloned() else {
        return false;
    };
    let program_reg = prog.reg;
    let descr_tok = env_array_descr_expr(lowerer.config);
    let Some(index_reg) = lower_array_index_expr(lowerer, &idx.index) else {
        return false;
    };
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::Vable,
            vec![Register::ref_(program_reg), Register::int(index_reg)],
            vec![Register::int(aliased_reg)],
        ),
        quote! {
            let __descr_idx = #descr_tok;
            // Reads the same immutable env array as the outer fetch, so the
            // same always-pure spelling.
            __builder.getarrayitem_gc_i_pure(
                #aliased_reg as u16,
                #program_reg as u16,
                #index_reg as u16,
                __descr_idx,
            );
        },
    );
    lowerer.bindings.insert(
        lhs_name,
        Binding {
            reg: aliased_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        },
    );
    true
}

/// Match `if <ident> < <ident> { panic!(...); }` and emit the
/// HAVE_ARGUMENT range guard:
///
/// ```text
///   load_const_i  HAVE_ARGUMENT_const_reg, HAVE_ARGUMENT
///   goto_if_not_int_lt  opcode_reg, HAVE_ARGUMENT_const_reg, ok_label
///   abort
/// ok_label:
/// ```
///
/// The local ident in the comparison must already be in `lowerer.bindings`
/// as Int; the other ident is treated as the module-level const. Either
/// operand order (`opcode < HAVE_ARGUMENT` or `HAVE_ARGUMENT < opcode`)
/// is rejected — RPython L190 is unambiguously `opcode < HAVE_ARGUMENT`,
/// the comparison is asymmetric, and accepting the reversed form would
/// silently invert the guard.
fn try_lower_have_argument_guard(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::If(if_expr), _) = stmt else {
        return false;
    };
    if if_expr.else_branch.is_some() {
        return false;
    }
    let cond = unwrap_expr_paren(&if_expr.cond);
    let Expr::Binary(bin) = cond else {
        return false;
    };
    if !matches!(bin.op, syn::BinOp::Lt(_)) {
        return false;
    }
    let Some(lhs_name) = expr_single_ident(&bin.left) else {
        return false;
    };
    let Some(rhs_name) = expr_single_ident(&bin.right) else {
        return false;
    };
    // pyopcode.py `if opcode < HAVE_ARGUMENT` — LHS must be the
    // local (in bindings), RHS the const (out of bindings). Rejecting
    // the reversed form keeps the guard polarity unambiguous.
    let Some(local) = lowerer.bindings.get(&lhs_name).cloned() else {
        return false;
    };
    if !matches!(local.kind, BindingKind::Int) {
        return false;
    }
    if lowerer.bindings.contains_key(&rhs_name) {
        return false;
    }
    if !block_is_panic_only(&if_expr.then_branch) {
        return false;
    }
    let Expr::Path(rhs_path) = bin.right.as_ref() else {
        return false;
    };
    let Some(const_ident) = rhs_path.path.get_ident().cloned() else {
        return false;
    };

    let ok_label = lowerer.alloc_label();
    lowerer.emit_aux(quote! { let #ok_label = __builder.new_label(); });
    let local_reg = local.reg;
    // flatten.py:258-260 emits -live- ahead of every goto_if_not. The
    // guard records through build_state_field_snapshot, which reads the
    // LIVE marker at guard_pc - SIZE_LIVE_OP; without a preceding -live-
    // the snapshot mis-decodes the prior op's bytes as a liveness offset.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote! { let _ = __builder.live_placeholder(); },
    );
    lowerer.emit_op(
        OpMeta::conditional_guard(Register::int(local_reg), ok_label.clone()),
        quote! {
            __builder.goto_if_not_int_const(
                majit_ir::OpCode::IntLt,
                #local_reg as u16,
                #const_ident as i64,
                #ok_label,
            );
        },
    );
    // BC_ABORT is the canonical local bailout — `assembler.rs`
    // + `dispatch.rs` resume protocol. A codex review
    // BLOCKER (d): `guard_value` is the wrong shape (range vs equality);
    // BC_ABORT preserves RPython L190-191 `raise BytecodeCorruption`
    // semantics through the existing trace-abort path.
    lowerer.emit_op(
        OpMeta::terminal(Vec::new()),
        quote! {
            __builder.abort();
        },
    );
    lowerer.emit_label_def(&ok_label);
    true
}

/// Match `<oparg_ident> = (<oparg_ident> * <int_lit>) | <arg_ident> as <ty>`
/// and emit the multi-byte oparg merge per RPython L193:
///
/// ```text
///   load_const_i  c_reg, <int_lit>
///   int_mul       oparg_reg, oparg_reg, c_reg
///   int_or        oparg_reg, oparg_reg, arg_reg
/// ```
///
/// Pre-A.2.3 codex review BLOCKER (c): pyre lacks `BC_INT_LSHIFT_IMM`
/// (only the 2-arg `BC_INT_LSHIFT`), and `jtransform.py:363-366` leaves
/// `int_mul` as the symmetric primitive for `* 256`. Treating
/// `lshift(8)` as the source-port choice would be an optimizer-level
/// equivalence, not RPython parity.
fn try_lower_oparg_merge_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::Assign(assign), _) = stmt else {
        return false;
    };
    let Some(lhs_name) = expr_single_ident(&assign.left) else {
        return false;
    };
    let lhs_binding = lowerer.bindings.get(&lhs_name).cloned();
    let Some(lhs_binding) = lhs_binding else {
        return false;
    };
    if !matches!(lhs_binding.kind, BindingKind::Int) {
        return false;
    }

    let rhs = unwrap_expr_paren(assign.right.as_ref());
    let Expr::Binary(or_bin) = rhs else {
        return false;
    };
    if !matches!(or_bin.op, syn::BinOp::BitOr(_)) {
        return false;
    }
    // Left of `|` must be `(<lhs_name> * <int_lit>)` (paren-wrapped).
    let mul_expr = unwrap_expr_paren(or_bin.left.as_ref());
    let Expr::Binary(mul_bin) = mul_expr else {
        return false;
    };
    if !matches!(mul_bin.op, syn::BinOp::Mul(_)) {
        return false;
    }
    let Some(mul_lhs_name) = expr_single_ident(&mul_bin.left) else {
        return false;
    };
    if mul_lhs_name != lhs_name {
        return false;
    }
    let Some(mul_lit) = expr_int_literal_value(&mul_bin.right) else {
        return false;
    };
    if mul_lit <= 0 {
        return false;
    }
    // Right of `|` is `<arg_ident> as <ty>` (RPython has no cast — the
    // fixture's `as i64` is a Rust artifact for u8 → i64 widening).
    let or_rhs = unwrap_expr_paren(or_bin.right.as_ref());
    let arg_expr = match or_rhs {
        Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    let Some(arg_name) = expr_single_ident(arg_expr) else {
        return false;
    };
    let Some(arg_binding) = lowerer.bindings.get(&arg_name).cloned() else {
        return false;
    };
    if !matches!(arg_binding.kind, BindingKind::Int) {
        return false;
    }

    let oparg_reg = lhs_binding.reg;
    let arg_reg = arg_binding.reg;
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(oparg_reg)],
            vec![Register::int(oparg_reg)],
        ),
        quote! {
            __builder.record_binop_i_const(
                #oparg_reg as u16,
                majit_ir::OpCode::IntMul,
                #oparg_reg as u16,
                #mul_lit as i64,
            );
        },
    );
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(oparg_reg), Register::int(arg_reg)],
            vec![Register::int(oparg_reg)],
        ),
        quote! {
            __builder.record_binop_i(
                #oparg_reg as u16,
                majit_ir::OpCode::IntOr,
                #oparg_reg as u16,
                #arg_reg as u16,
            );
        },
    );
    true
}

/// Recognize `<outer> = <inner>` where `<inner>` is in `inner_alias`. The
/// alias was set up in `lower_extended_arg_inner_while`'s pre-pass, so
/// both names already resolve to the same outer register — the stmt is
/// a no-op for the IR (RPython L188's `opcode = ord(...)` is the
/// in-loop rebind; the Rust fixture's trailing `opcode = opcode2` is a
/// scoping artifact, not an RPython operation).
fn try_lower_alias_assign_stmt(stmt: &Stmt, inner_alias: &HashMap<String, u16>) -> bool {
    let Some((_lhs, rhs)) = match_simple_ident_assign(stmt) else {
        return false;
    };
    inner_alias.contains_key(&rhs)
}

/// Match `<lhs_ident> = <rhs_ident>;` where both sides are bare
/// single-segment idents. Returns `(lhs, rhs)` ident names.
fn match_simple_ident_assign(stmt: &Stmt) -> Option<(String, String)> {
    let Stmt::Expr(Expr::Assign(assign), _) = stmt else {
        return None;
    };
    let lhs = expr_single_ident(&assign.left)?;
    let rhs = expr_single_ident(&assign.right)?;
    Some((lhs, rhs))
}

/// Returns `true` if `block` consists of a single `panic!(...)` macro
/// stmt. Used to validate the corruption-bailout body of the
/// HAVE_ARGUMENT range guard so a stray non-panic body would fail
/// recognition (and the caller would mark the dispatch JitCode tainted).
fn block_is_panic_only(block: &syn::Block) -> bool {
    if block.stmts.len() != 1 {
        return false;
    }
    let stmt = &block.stmts[0];
    let mac = match stmt {
        Stmt::Macro(m) => &m.mac,
        Stmt::Expr(Expr::Macro(em), _) => &em.mac,
        _ => return false,
    };
    mac.path
        .segments
        .last()
        .map(|seg| seg.ident == "panic")
        .unwrap_or(false)
}

/// Disambiguate a recognized `<ident> == <ident>` while-condition into
/// `(local_reg, const_ident)`. The "local" side is the one already in
/// `lowerer.bindings` as `Int`; the "const" side is the other ident,
/// returned as a `syn::Ident` so the caller can interpolate it into the
/// `load_const_i` token tree. `None` if both sides or neither are bound
/// (genuinely ambiguous; fail closed per A.2.3a abandonment trigger).
fn pick_local_and_const_idents(lowerer: &Lowerer, cond: &Expr) -> Option<(u16, syn::Ident)> {
    let bin = match unwrap_expr_paren(cond) {
        Expr::Binary(b) => b,
        _ => return None,
    };
    let left_ident = match bin.left.as_ref() {
        Expr::Path(p) => p.path.get_ident().cloned()?,
        _ => return None,
    };
    let right_ident = match bin.right.as_ref() {
        Expr::Path(p) => p.path.get_ident().cloned()?,
        _ => return None,
    };
    let left_local = lowerer
        .bindings
        .get(&left_ident.to_string())
        .filter(|b| matches!(b.kind, BindingKind::Int));
    let right_local = lowerer
        .bindings
        .get(&right_ident.to_string())
        .filter(|b| matches!(b.kind, BindingKind::Int));
    match (left_local, right_local) {
        (Some(b), None) => Some((b.reg, right_ident)),
        (None, Some(b)) => Some((b.reg, left_ident)),
        _ => None,
    }
}

/// Try to lower one of the two opcode-fetch IR patterns:
///
/// 1. `let <name> = program[<index>];` where `<index>` is `pc` or `pc + N`
///    For `pc`, emits `BC_GETARRAYITEM_GC_I result_reg, program_reg(r0),
///    pc_reg(i0)`. For `pc + N` (RPython `pyopcode.py:180`
///    `co_code[next_instr + 1]`), emits an extra `load_const_i(tmp, N) +
///    int_add(offset, pc_reg, tmp)` pair to materialize the index, then
///    `BC_GETARRAYITEM_GC_I result_reg, program_reg, offset_reg`. Stores
///    `<name> → Binding { reg: result_reg, kind: Int }` in lowerer so
///    downstream arms can reference the fetched byte.
///
/// 2. `pc += N` (RPython `pyopcode.py:181` `next_instr += 2`)
///    Emits `load_const_i(tmp, N)` + `int_add(pc_reg, pc_reg, tmp)` to
///    model the pc increment without a literal const operand.
///
/// Identification uses name-based heuristics ("program" / "pc") rather
/// than type-level analysis. This matches the `#[jit_interp]` macro's
/// convention where the env parameter is named `program` and the loop
/// counter is named `pc`.
///
/// TODO: derive names from LowererConfig env/pc config fields when those
/// are added, instead of the hard-coded strings.
///
/// Returns `true` if the stmt was consumed (lowered or silently skipped),
/// `false` if the caller should continue with other lowering paths.
fn try_lower_opcode_fetch_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    // Pattern 1: `let <name> = program[<index>];`
    // AST: Stmt::Local { pat: Pat::Ident { ident: <name> },
    //                    init: Some(LocalInit { expr: Expr::Index {
    //                        expr: Expr::Path(program_path),
    //                        index: <pc | pc + N> } }) }
    if let Stmt::Local(local) = stmt
        && let Some(init) = &local.init
    {
        // Peel an outer `Expr::Cast` so `let X: i64 = program[idx] as i64`
        // matches alongside the bare `let X = program[idx]` form. The
        // cast is purely a Rust widening artifact (the byte fetch itself
        // writes into an i64-banked register either way; RPython's
        // `ord(co_code[next_instr + 1])` already returns a Python int).
        let init_expr = match init.expr.as_ref() {
            Expr::Cast(c) => c.expr.as_ref(),
            other => other,
        };
        // Recognise the index form `program[idx]` AND the method-call
        // PyPy `pyopcode.py:171 ord(co_code[next_instr])` is an
        // index form on the bytecode array; that's the ONLY shape
        // the codewriter recognises as the BC_GETARRAYITEM_GC_I
        // opcode-fetch.  Earlier pyre revisions also whitelisted
        // a method-call form `program.get_op(idx)` to accommodate
        // consumers wrapping the byte access in a method, but the
        // proc-macro cannot verify the method body actually equals
        // `code[idx]` — a wrapper that returns `code[idx] ^ key`
        // (or any non-trivial transformation) would silently lower
        // as a raw byte-array load with the wrong semantic.
        //
        // For strict line-by-line PyPy parity, only the index form
        // is recognised here.  Consumers using a method wrapper
        // must register the method as a call policy
        // (`#[jit_interp(calls = { Program::get_op =>
        // elidable_int })]`); `lower_value_expr` then emits a
        // `call_pure_int_canonical_via_target` op rather than a
        // hardcoded byte-array load.
        let opcode_fetch = match init_expr {
            Expr::Index(idx) => {
                let array_name = expr_single_ident(&idx.expr);
                if array_name.as_deref() == Some("program") {
                    Some(idx.index.as_ref())
                } else {
                    None
                }
            }
            _ => None,
        };
        // Method-call form: `let op = program.get_op(pc)` where
        // `get_op` is registered as an elidable call policy. Emit
        // `call_pure_int` and bind the result so `lower_dispatch_chain`
        // finds the opcode register.
        if opcode_fetch.is_none()
            && let Expr::Call(call) = init_expr
            && call_is_insn_op_at_pc(call)
        {
            if let Some(binding) = lowerer.lower_value_expr(init_expr) {
                if let Some(name) = pat_bound_ident_name(&local.pat) {
                    lowerer.bindings.insert(
                        name.clone(),
                        Binding {
                            reg: binding.reg,
                            kind: binding.kind,
                            depends_on_stack: false,
                            struct_type: None,
                        },
                    );
                    lowerer.opcode_var_name = Some(name);
                }
                return true;
            }
            // Recognized fetch, no call policy. `lower_dispatch_chain` would
            // then emit no arms and the JitCode would not run the source
            // dispatch. Same channel as an unlowerable dispatch pattern:
            // `compile_error!` in the generated body.
            lowerer.emit_aux(quote::quote! {
                compile_error!(
                    "opcode fetch `insn_op(program, pc)` has no call policy; register `insn_op` in `calls = { ... }`"
                );
            });
            return true;
        }
        if opcode_fetch.is_none()
            && let Expr::MethodCall(mc) = init_expr
        {
            let receiver_name = expr_single_ident(&mc.receiver);
            if receiver_name.as_deref() == Some("program")
                && let Some(binding) = lowerer.lower_value_expr(&syn::Expr::MethodCall(mc.clone()))
            {
                if let Some(name) = pat_bound_ident_name(&local.pat) {
                    lowerer.bindings.insert(
                        name.clone(),
                        Binding {
                            reg: binding.reg,
                            kind: binding.kind,
                            depends_on_stack: false,
                            struct_type: None,
                        },
                    );
                    lowerer.opcode_var_name = Some(name);
                }
                return true;
            }
        }
        if let Some(idx_expr) = opcode_fetch {
            // Binding names: "program" → r0, "pc" → i0 (installed by
            // lower_dispatch_body before this fn is called).
            let program_binding = lowerer.bindings.get("program").cloned();
            let Some(prog) = program_binding else {
                return false;
            };
            let program_reg = prog.reg;
            let descr_tok = env_array_descr_expr(lowerer.config);
            // Compute the index register: `pc` returns pc_reg directly,
            // `pc + N` emits load_const + int_add into a fresh reg.
            let Some(index_reg) = lower_array_index_expr(lowerer, idx_expr) else {
                return false;
            };
            // Allocate a fresh Int register for the fetch result.
            let result_reg = lowerer.alloc_reg();
            lowerer.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(program_reg), Register::int(index_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote::quote! {
                    let __descr_idx = #descr_tok;
                    // The bytecode array is fixed once the program is built,
                    // so the read is spelled with the always-pure opcode and
                    // the tracer folds it at a green index on the strength of
                    // that spelling rather than on who the caller is.
                    __builder.getarrayitem_gc_i_pure(
                        #result_reg as u16,
                        #program_reg as u16,
                        #index_reg as u16,
                        __descr_idx,
                    );
                },
            );
            // Record binding for `<name>` so downstream patterns
            // (dispatch chain, Task 1.5) can reference it. Peel
            // an outer `Pat::Type` so `let X: i64 = ...` (with
            // an explicit type annotation) is recognized
            // alongside the bare `let X = ...` form.
            if let Some(name) = pat_bound_ident_name(&local.pat) {
                lowerer.bindings.insert(
                    name.clone(),
                    Binding {
                        reg: result_reg,
                        kind: BindingKind::Int,
                        depends_on_stack: false,
                        struct_type: None,
                    },
                );
                // Slice ε.1: record the consumer's chosen
                // opcode-result name so `lower_dispatch_chain`
                // can find the binding regardless of whether the
                // consumer named it `opcode` (PyPy convention,
                // `pyopcode.py:171`) or `op` or anything else.
                lowerer.opcode_var_name = Some(name);
            }
            return true;
        }
    }

    // Pattern 2: `pc += N`
    // syn 2 AST: Stmt::Expr(Expr::Binary { op: BinOp::AddAssign,
    //                left: pc_path, right: Expr::Lit(LitInt(N)) })
    // Also handles `pc = pc + N` (Expr::Assign with binary Add).
    if let Some((lhs_name, increment)) = match_pc_increment_stmt(stmt)
        && lhs_name == "pc"
        && increment > 0
    {
        let pc_binding = lowerer.bindings.get("pc").cloned();
        let Some(pc) = pc_binding else {
            return false;
        };
        let pc_reg = pc.reg;
        // `int_add(pc_reg, Constant(N))`: the increment stays a constant
        // operand. `pyopcode.py dispatch_bytecode`'s `next_instr += 2` is
        // the canonical N=2 case.
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                vec![Register::int(pc_reg)],
                vec![Register::int(pc_reg)],
            ),
            quote::quote! {
                __builder.record_binop_i_const(
                    #pc_reg as u16,
                    majit_ir::OpCode::IntAdd,
                    #pc_reg as u16,
                    #increment as i64,
                );
            },
        );
        return true;
    }

    false
}

/// Compute the register holding the array-index value for the
/// `program[<idx_expr>]` opcode-fetch pattern. Supports two shapes:
///
/// - `pc` (single ident): returns `pc_reg` directly with no ops emitted.
/// - `pc + N` (binary Add with int literal RHS): emits a fresh
///   `load_const_i(const_reg, N)` + `int_add(offset_reg, pc_reg,
///   const_reg)` pair into the lowerer and returns `offset_reg`. RPython
///   `pyopcode.py:180` `co_code[next_instr + 1]` is the canonical N=1
///   case; `pc_reg` itself is preserved (matches RPython orthodoxy where
///   the index expression does NOT mutate next_instr — that is L181's
///   `next_instr += 2`).
///
/// Returns `None` for any other shape so the caller can abort lowering
/// and fall through to the state-modifies filter.
fn lower_array_index_expr(lowerer: &mut Lowerer, idx_expr: &Expr) -> Option<u16> {
    if let Some(name) = expr_single_ident(idx_expr) {
        if name == "pc" {
            return Some(lowerer.bindings.get("pc")?.reg);
        }
        return None;
    }
    let Expr::Binary(bin) = idx_expr else {
        return None;
    };
    if !matches!(bin.op, syn::BinOp::Add(_)) {
        return None;
    }
    let lhs_name = expr_single_ident(&bin.left)?;
    if lhs_name != "pc" {
        return None;
    }
    let rhs_val = expr_int_literal_value(&bin.right)?;
    if rhs_val <= 0 {
        return None;
    }
    let pc_reg = lowerer.bindings.get("pc")?.reg;
    let const_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
        quote::quote! {
            __builder.load_const_i_value(#const_reg as u16, #rhs_val as i64);
        },
    );
    let offset_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(pc_reg), Register::int(const_reg)],
            vec![Register::int(offset_reg)],
        ),
        quote::quote! {
            __builder.record_binop_i(
                #offset_reg as u16,
                majit_ir::OpCode::IntAdd,
                #pc_reg as u16,
                #const_reg as u16,
            );
        },
    );
    Some(offset_reg)
}

/// Extract the single ident string from `Expr::Path` if it has exactly one
/// segment (no leading `::`, no generics). Returns `None` otherwise.
fn expr_single_ident(expr: &Expr) -> Option<String> {
    let Expr::Path(ep) = expr else { return None };
    if ep.qself.is_some() || ep.path.leading_colon.is_some() {
        return None;
    }
    if ep.path.segments.len() != 1 {
        return None;
    }
    Some(ep.path.segments[0].ident.to_string())
}

/// Match a `pc += N` or `pc = pc + N` statement (where N is any int literal).
/// Returns `Some((lhs_name, increment))` if the pattern matches.
///
/// The caller filters by `lhs_name == "pc"` and `increment > 0`. RPython
/// `pyopcode.py:181` `next_instr += 2` is the canonical N=2 case; the
/// previous N=1 case (`pc += 1`) is preserved verbatim by the same
/// generalized AST shape.
fn match_pc_increment_stmt(stmt: &Stmt) -> Option<(String, i64)> {
    let expr = match stmt {
        Stmt::Expr(e, _) => e,
        _ => return None,
    };
    // `pc += N` — syn 2 parses compound assignments as Expr::Binary
    // with BinOp::AddAssign(syn::token::PlusEq).
    if let Expr::Binary(bin) = expr
        && matches!(bin.op, syn::BinOp::AddAssign(_))
    {
        let lhs = expr_single_ident(&bin.left)?;
        let increment = expr_int_literal_value(&bin.right)?;
        return Some((lhs, increment));
    }
    // `pc = pc + N`
    if let Expr::Assign(a) = expr {
        let lhs = expr_single_ident(&a.left)?;
        if let Expr::Binary(bin) = a.right.as_ref()
            && matches!(bin.op, syn::BinOp::Add(_))
        {
            let l_name = expr_single_ident(&bin.left)?;
            let increment = expr_int_literal_value(&bin.right)?;
            if l_name == lhs {
                return Some((lhs, increment));
            }
        }
    }
    None
}

/// Return the integer value of an `Expr::Lit(LitInt)` if it fits in i64.
fn expr_int_literal_value(expr: &Expr) -> Option<i64> {
    let Expr::Lit(el) = expr else { return None };
    let Lit::Int(li) = &el.lit else { return None };
    li.base10_parse::<i64>().ok()
}

/// Find the body block of the unique top-level loop in `func_block` whose
/// body contains `target_match`.
///
/// Recognises both `while cond { ... }` (e.g. `while pc < program.size`)
/// and `loop { ... }` (e.g. `rpython/jit/tl/tinyframe/tinyframe.py` and
/// other tinyframe-family interpreters whose dispatch loop is unconditional
/// with a `break`-driven exit).  Mirrors `codegen_trace.rs
/// expr_inner_match_block`'s recognition set.
pub(super) fn find_dispatch_loop_body<'b>(
    func_block: &'b syn::Block,
    target_match: &ExprMatch,
) -> Option<&'b syn::Block> {
    for stmt in &func_block.stmts {
        let expr = match stmt {
            Stmt::Expr(e, _) => e,
            _ => continue,
        };
        match expr {
            Expr::While(while_expr) if block_contains_match(&while_expr.body, target_match) => {
                return Some(&while_expr.body);
            }
            Expr::Loop(loop_expr) if block_contains_match(&loop_expr.body, target_match) => {
                return Some(&loop_expr.body);
            }
            _ => continue,
        }
    }
    None
}

/// Returns `true` if `stmt` is a `jit_merge_point!()` macro invocation.
pub(crate) fn is_jit_merge_point_macro(stmt: &Stmt) -> bool {
    stmt_macro_ends_with(stmt, "jit_merge_point")
}

/// Returns `true` if `stmt` is a `can_enter_jit!()` macro invocation.
pub(crate) fn is_can_enter_jit_macro(stmt: &Stmt) -> bool {
    stmt_macro_ends_with(stmt, "can_enter_jit")
}

fn stmt_macro_ends_with(stmt: &Stmt, name: &str) -> bool {
    let Stmt::Macro(mac_stmt) = stmt else {
        return false;
    };
    mac_stmt
        .mac
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == name)
}

/// Lower a matchless portal loop — `while cond { merge_point; body }` —
/// as one JitCode iteration. `warmspot.rewrite_can_enter_jit` puts the
/// interpreter tick at the header when the source has no `can_enter_jit`
/// of its own; that call is not a JitCode `loop_header`. The compiled
/// back-edge is the while-condition plus a `loop_header` + `goto`, the
/// same place OP_LOOP closes a three-opcode portal.
fn lower_matchless_portal_body(
    lowerer: &mut Lowerer,
    func_block: &syn::Block,
    loop_start_label: &syn::Ident,
) -> Option<()> {
    let portal = find_portal_loop(func_block)?;
    // `continue` in the body must re-check the while condition. The
    // dispatch label the caller installed is the merge-point label, so
    // a jump there would skip the condition emitted below and keep
    // iterating after it is false.
    let cond_label = lowerer.alloc_label();
    lowerer.emit_aux(quote::quote! { let #cond_label = __builder.new_label(); });
    let saved_loop = lowerer.dispatch_loop_label.replace(cond_label.clone());

    let mut seen_merge_point = false;
    for stmt in &portal.body.stmts {
        if is_jit_merge_point_macro(stmt) {
            seen_merge_point = true;
            continue;
        }
        if !seen_merge_point {
            continue;
        }
        // Header `can_enter_jit!` belongs to the interpreter threshold,
        // not the JitCode (`rewrite_can_enter_jits` runs after
        // `make_jitcodes`). A copy left after the merge point is the
        // same tick, not a back-edge `loop_header`.
        if is_can_enter_jit_macro(stmt) {
            continue;
        }
        lowerer.lower_stmt(stmt)?;
    }
    if !seen_merge_point {
        lowerer.dispatch_loop_label = saved_loop;
        return None;
    }
    if let Some(census_interp) = lowerer.config.map(|c| c.state_type_name.clone()) {
        // One implicit arm: the post-merge-point while body. The
        // opcode chain's census is the same install-time marker
        // `assert_no_degraded_dispatch_arms` reads.
        lowerer.emit_aux(quote::quote! {
            majit_metainterp::record_dispatch_arm_census(#census_interp, 1usize);
        });
    }
    // Restore the merge-point label so the back-edge `continue` after
    // a true condition lands on `jit_merge_point`, not the condition.
    lowerer.dispatch_loop_label = saved_loop;
    lowerer.emit_label_def(&cond_label);
    // Close the iteration the way marked.py's while does: test the
    // header condition after the body, then jump to the merge point.
    // `can_enter_jit!()` here is only the JitCode `loop_header` emit
    // (`handle_jit_marker__can_enter_jit`); the native tick stays at
    // the source header.
    if let Some(cond) = portal.while_cond {
        let back_edge: syn::Expr = syn::parse_quote! {
            if #cond {
                can_enter_jit!();
                continue;
            }
        };
        lowerer.lower_stmt(&Stmt::Expr(back_edge, None))?;
    } else {
        let header: syn::Stmt = syn::parse_quote! { can_enter_jit!(); };
        lowerer.lower_stmt(&header)?;
        lowerer.emit_jump(loop_start_label);
    }
    Some(())
}

/// Build the parent-side `__builder.inline_call_<types>_v(__sub_idx, ...)`
/// emit for a dispatch arm given its
/// [`CallerLocalLayout`] list (from
/// [`try_generate_jitcode_body_parts_with_caller_bindings`]).
///
/// Picks the family member by which banks are populated:
/// - any Float entry → `inline_call_irf_v` (Int+Ref+Float arg vectors);
/// - else any Int entry → `inline_call_ir_v` (Int+Ref arg vectors);
/// - else → `inline_call_r_v` (Ref-only arg vector; degenerates to the
///   no-arg form when layout is empty).
///
/// Arg pairs are `(parent_reg, callee_reg)` per `assembler.rs
/// inline_call_<types>_v` API.  Mirrors `inline_call_tokens` at
/// `:5098`'s family-by-bank pattern but always selects the void-result
/// variant — dispatch arms never produce an inline_call return value
/// (the arm body's return path is always the loop back-edge to
/// `jit_merge_point` or the default exit, never a value).
pub(super) fn dispatch_arm_inline_call_tokens(
    layout: &[CallerLocalLayout],
    identity_int_pairs: &[(u16, u16)],
) -> proc_macro2::TokenStream {
    use quote::quote;
    let has_float = layout.iter().any(|l| matches!(l.kind, BindingKind::Float));
    let pair_tokens = |kind: BindingKind| -> Vec<proc_macro2::TokenStream> {
        layout
            .iter()
            .filter(|l| l.kind == kind)
            .map(|l| {
                let parent = l.parent_reg;
                let callee = l.callee_reg;
                quote! { (#parent as u16, #callee as u16) }
            })
            .collect()
    };
    let args_i_pairs = extend_int_args_with_array_identity(
        layout
            .iter()
            .filter(|l| l.kind == BindingKind::Int)
            .map(|l| (l.parent_reg, l.callee_reg))
            .collect(),
        identity_int_pairs,
    );
    let has_int = !args_i_pairs.is_empty();
    let args_i = args_i_pairs.iter().map(|&(parent, callee)| {
        quote! { (#parent as u16, #callee as u16) }
    });
    let args_r = pair_tokens(BindingKind::Ref);
    let args_f = pair_tokens(BindingKind::Float);
    if has_float {
        quote! {
            __builder.inline_call_irf_v(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                &[#(#args_f),*],
                None,
            );
        }
    } else if has_int {
        quote! {
            __builder.inline_call_ir_v(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                None,
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_v(
                __sub_idx,
                &[#(#args_r),*],
                None,
            );
        }
    }
}

/// pc-returning variant of [`dispatch_arm_inline_call_tokens`]: emits the
/// `inline_call_<types>_i` form whose callee BC_INT_RETURN writes back into the
/// caller's `return_i_reg` (the dispatch loop's green pc register).  Used by
/// `split_dispatch` for pure forward-advancing arms so the heavy arm body lives
/// in its own sub-JitCode yet still advances the green pc — the `next_instr =
/// self.OPCODE(oparg, next_instr)` shape.
pub(super) fn dispatch_arm_inline_call_tokens_i(
    layout: &[CallerLocalLayout],
    return_i_reg: u16,
    identity_int_pairs: &[(u16, u16)],
) -> proc_macro2::TokenStream {
    use quote::quote;
    let has_float = layout.iter().any(|l| matches!(l.kind, BindingKind::Float));
    let pair_tokens = |kind: BindingKind| -> Vec<proc_macro2::TokenStream> {
        layout
            .iter()
            .filter(|l| l.kind == kind)
            .map(|l| {
                let parent = l.parent_reg;
                let callee = l.callee_reg;
                quote! { (#parent as u16, #callee as u16) }
            })
            .collect()
    };
    let args_i_pairs = extend_int_args_with_array_identity(
        layout
            .iter()
            .filter(|l| l.kind == BindingKind::Int)
            .map(|l| (l.parent_reg, l.callee_reg))
            .collect(),
        identity_int_pairs,
    );
    let has_int = !args_i_pairs.is_empty();
    let args_i = args_i_pairs.iter().map(|&(parent, callee)| {
        quote! { (#parent as u16, #callee as u16) }
    });
    let args_r = pair_tokens(BindingKind::Ref);
    let args_f = pair_tokens(BindingKind::Float);
    if has_float {
        quote! {
            __builder.inline_call_irf_i(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                &[#(#args_f),*],
                Some(#return_i_reg as u16),
            );
        }
    } else if has_int {
        quote! {
            __builder.inline_call_ir_i(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                Some(#return_i_reg as u16),
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_i(
                __sub_idx,
                &[#(#args_r),*],
                Some(#return_i_reg as u16),
            );
        }
    }
}

#[cfg(test)]
mod dispatch_arm_inline_call_tokens_tests {
    use super::*;

    fn entry(name: &str, parent_reg: u16, callee_reg: u16, kind: BindingKind) -> CallerLocalLayout {
        CallerLocalLayout {
            name: name.to_string(),
            parent_reg,
            callee_reg,
            kind,
        }
    }

    fn render(tokens: &proc_macro2::TokenStream) -> String {
        tokens.to_string()
    }

    #[test]
    fn empty_layout_emits_inline_call_r_v_no_args() {
        let out = dispatch_arm_inline_call_tokens(&[], &[]);
        let s = render(&out);
        assert!(s.contains("inline_call_r_v"), "got: {s}");
        // Both arg vec literal and the no-return None must appear.
        assert!(s.contains("& [] ,"), "got: {s}");
        assert!(s.contains("None"), "got: {s}");
    }

    #[test]
    fn ref_only_layout_uses_inline_call_r_v() {
        let layout = vec![entry("program", 3, 0, BindingKind::Ref)];
        let s = render(&dispatch_arm_inline_call_tokens(&layout, &[]));
        assert!(s.contains("inline_call_r_v"), "got: {s}");
        assert!(s.contains("3u16") || s.contains("3 as u16"), "got: {s}");
        assert!(!s.contains("inline_call_ir_v"), "got: {s}");
    }

    #[test]
    fn mixed_int_ref_uses_inline_call_ir_v() {
        let layout = vec![
            entry("pc", 7, 0, BindingKind::Int),
            entry("program", 3, 0, BindingKind::Ref),
        ];
        let s = render(&dispatch_arm_inline_call_tokens(&layout, &[]));
        assert!(s.contains("inline_call_ir_v"), "got: {s}");
        // Both parent regs land in their respective arg slices.
        assert!(s.contains("7"), "got: {s}");
        assert!(s.contains("3"), "got: {s}");
    }

    #[test]
    fn any_float_uses_inline_call_irf_v() {
        let layout = vec![
            entry("pc", 7, 0, BindingKind::Int),
            entry("flt", 9, 0, BindingKind::Float),
        ];
        let s = render(&dispatch_arm_inline_call_tokens(&layout, &[]));
        assert!(s.contains("inline_call_irf_v"), "got: {s}");
    }

    #[test]
    fn plain_array_identity_pairs_are_declared_int_args() {
        // `MIFrame.setup_call` / `bhimpl_inline_call_ir_v`: a value the
        // callee reads must be a declared argument. Empty caller-locals
        // still pass ptr/len at the identity slot numbers. Holes below
        // those slots are padded so the I-list stays positional.
        let s = render(&dispatch_arm_inline_call_tokens(&[], &[(2, 2), (3, 3)]));
        assert!(s.contains("inline_call_ir_v"), "got: {s}");
        assert!(s.contains("2u16 as u16"), "got: {s}");
        assert!(s.contains("3u16 as u16"), "got: {s}");
        assert!(s.contains("0u16 as u16"), "got: {s}");
        assert!(s.contains("1u16 as u16"), "got: {s}");
    }
}

/// Emit the dispatch chain for the opcode dispatch loop.
///
/// For each non-wildcard arm, emits a fused `goto_if_not_int_eq/iiL`
/// (BC_GOTO_IF_NOT_INT_EQ) that branches past the arm if the opcode does NOT
/// match. After all checks, emits an unconditional `jump` (BC_GOTO) to the
/// default label.
///
/// pyopcode.py+ if/elif chain over opcode constants.
/// jtransform.py optimize_goto_if_not fuses `int_eq + goto_if_not`
/// into `goto_if_not_int_eq/iiL`.
///
/// `default_label` is bound at the typed-return emission site in
/// `lower_dispatch_body`; `loop_start_label` is the back-edge target
/// (JIT_MERGE_POINT position) emitted after each matched arm's body.
/// `true` when the dispatch declares `pc` as a green (e.g.
/// `#[jit_interp(greens = [pc])]`).  PyPy `tl.py` `greens=['pc','code']`:
/// with pc green the opcode/operand reads at `program[pc]` and the dispatch
/// chain constant-fold, and the loop closes on the pc value.  Gates the
/// green-pc inline dispatch path (`try_inline_dispatch_arm`); red-pc
/// dispatches keep the sub-JitCode + BC_INLINE_CALL model.
pub(super) fn pc_is_green(config: &LowererConfig) -> bool {
    config.greens.iter().any(|green| {
        matches!(green, Expr::Path(p)
            if p.qself.is_none()
                && p.path.get_ident().map(|id| id == "pc").unwrap_or(false))
    })
}

/// Expression that mints the array descr for an `env` (`program[...]`) read,
/// scaling the index by the env element size. `&[u8]` keeps the historical
/// `add_gc_byte_array_descr()` (itemsize 1, unsigned — no `movsx` on bytes
/// ≥ 0x80); a wider element (`&[i64]`) reads the element at byte offset
/// `size_of::<elem>() * index`. The element size is unknowable to the
/// proc-macro (it holds only the alias name), so emit a
/// `size_of::<<Env as Index<usize>>::Output>()` const computed in the user
/// crate. All three `program[...]` lowering sites (operand reads, opcode
/// aliasing, inline-dispatch fetch) share this so an env's opcode AND its
/// operands read at the same stride. Falls back to the byte descr when the
/// env type name does not parse as a type.
pub(super) fn env_array_descr_expr(config: Option<&LowererConfig>) -> proc_macro2::TokenStream {
    let env_ty = config.and_then(|c| syn::parse_str::<syn::Type>(&c.env_type_name).ok());
    match env_ty {
        Some(env_ty) => quote::quote! {{
            let __env_item = ::core::mem::size_of::<
                <#env_ty as ::core::ops::Index<usize>>::Output,
            >();
            __builder.add_env_int_array_descr(__env_item, __env_item != 1)
        }},
        None => quote::quote! { __builder.add_gc_byte_array_descr() },
    }
}

/// Recognise a self-increment of `pc`: `pc += N` (BinOp::AddAssign) or
/// `pc = pc + N` (Assign of `pc + N`), returning the literal `N`.  Mirrors
/// `match_pc_increment_stmt` but operates on an `Expr` and is pc-specific —
/// used by `lower_pc_pinned_write` on the inline arm path.
fn pc_self_increment(expr: &Expr) -> Option<i64> {
    if let Expr::Binary(bin) = expr
        && matches!(bin.op, syn::BinOp::AddAssign(_))
        && expr_single_ident(&bin.left).as_deref() == Some("pc")
    {
        return expr_int_literal_value(&bin.right);
    }
    if let Expr::Assign(a) = expr
        && expr_single_ident(&a.left).as_deref() == Some("pc")
        && let Expr::Binary(bin) = a.right.as_ref()
        && matches!(bin.op, syn::BinOp::Add(_))
        && expr_single_ident(&bin.left).as_deref() == Some("pc")
    {
        return expr_int_literal_value(&bin.right);
    }
    None
}

/// The return terminator for a `return` whose operand lowered to `binding`
/// (`None` = no operand, or an operand that did not lower), as the
/// `(reads, emitter)` pair `OpMeta::terminal` + `emit_op` consume.
///
/// `blackhole.py:841-857`: a typed return reads the source register of its
/// declared kind, so the register must appear in the terminator's reads list —
/// that list is what keeps it alive through the backward liveness walk.
/// `blackhole.py:859-862`: `void_return` has no operand and therefore no reads.
///
/// Shared by both emission sites — the function-final return after the dispatch
/// loop and the in-arm `return` — so the two cannot drift in which builder call
/// a `BindingKind` maps to.
pub(super) fn typed_return_terminator(
    binding: Option<Binding>,
) -> (Vec<Register>, proc_macro2::TokenStream) {
    let Some(binding) = binding else {
        return (Vec::new(), quote::quote! { __builder.void_return(); });
    };
    let reg = binding.reg;
    match binding.kind {
        BindingKind::Int => (
            vec![Register::int(reg)],
            quote::quote! { __builder.int_return(#reg as u16); },
        ),
        BindingKind::Ref => (
            vec![Register::ref_(reg)],
            quote::quote! { __builder.ref_return(#reg as u16); },
        ),
        BindingKind::Float => (
            vec![Register::float(reg)],
            quote::quote! { __builder.float_return(#reg as u16); },
        ),
    }
}

/// `Some(N)` if `body` is a pure forward-advancing dispatch arm: straight-line
/// work followed by a single trailing `pc += N` (N > 0), with no back-edge
/// (`can_enter_jit!` / `jit_merge_point!`), no early `return` / `continue` /
/// `break`, no other `pc` write, and no function call.  Such an arm has no
/// merge point of its own and takes no mid-body resume snapshot, so under
/// `split_dispatch` it is lowered into a per-arm sub-JitCode that RETURNS the
/// advanced pc (`next_instr = self.OPCODE(oparg, next_instr)`), keeping the
/// dispatch JitCode's register/const footprint small.  Branch arms
/// (`pc = target; continue;`) and `return` arms own a merge point / back-edge
/// and are rejected so they stay force-inlined.
///
/// Arms that make a residual call are also rejected: a residual call records a
/// GuardNoException resume snapshot whose canonical "all-live" liveness lists
/// every int register `[0, num_regs_i)`, including the unused per-bank holes the
/// caller-local layout leaves (e.g. int reg 1 when `pc` is the sole int
/// caller-local but `program`/`state` occupy two ref regs); reading such a hole
/// register at resume hits an uninitialized slot.  Such arms keep their heavy
/// work out-of-line in the residual already, so force-inlining them costs the
/// dispatch little — they do not need the split.
fn arm_is_pure_pc_advance(body: &Expr) -> Option<i64> {
    use syn::visit::Visit;
    struct Forbid {
        hit: bool,
    }
    impl<'ast> Visit<'ast> for Forbid {
        // Load-bearing beyond "this arm is not a pure advance": a return arm
        // must NEVER reach the split path.  `dispatch_arm_inline_call_tokens_i`
        // wires the sub-JitCode's BC_INT_RETURN into the caller's
        // `return_i_reg`, which for `split_dispatch` IS the dispatch loop's
        // green pc register (`next_instr = self.OPCODE(oparg, next_instr)`).
        // A `return <value>` lowered into that sub-JitCode would therefore
        // write its return VALUE into the green pc — a miscompile that
        // compiles clean, not a missing return.  Return arms stay force-inlined
        // so `Lowerer::lower_return_stmt` emits the terminator in the dispatch
        // JitCode itself, where it ends the walk instead of feeding a register.
        fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
            self.hit = true;
        }
        fn visit_expr_continue(&mut self, _: &'ast syn::ExprContinue) {
            self.hit = true;
        }
        fn visit_expr_break(&mut self, _: &'ast syn::ExprBreak) {
            self.hit = true;
        }
        // Residual-call arms ARE eligible to split: their GuardNoException
        // resume snapshot's identity-slot references resolve through the
        // sized rd_virtuals (negative virtual numbering) and the reserved
        // sub-JitCode identity prefix, so they no longer read uninitialized
        // holes. Only control-flow / loop-header constructs below disqualify.
        fn visit_macro(&mut self, m: &'ast syn::Macro) {
            if let Some(id) = m.path.get_ident() {
                let n = id.to_string();
                if n == "can_enter_jit" || n == "jit_merge_point" {
                    self.hit = true;
                }
            }
            syn::visit::visit_macro(self, m);
        }
    }
    let mut forbid = Forbid { hit: false };
    forbid.visit_expr(body);
    if forbid.hit {
        return None;
    }

    let stmts = extract_stmts(body);
    let (last, work) = stmts.split_last()?;
    let Stmt::Expr(last_expr, _) = last else {
        return None;
    };
    let increment = pc_self_increment(last_expr)?;
    if increment <= 0 {
        return None;
    }
    // No earlier statement may write `pc` (the advance must be the sole, final
    // pc mutation — a mid-body pc write would not be a straight-line advance).
    for s in work {
        if let Stmt::Expr(e, _) = s {
            if pc_self_increment(e).is_some() {
                return None;
            }
            if let Expr::Assign(a) = e
                && expr_single_ident(&a.left).as_deref() == Some("pc")
            {
                return None;
            }
        }
    }
    Some(increment)
}

/// Tests for `arm_is_pure_pc_advance`, the predicate that gates `split_dispatch`.
///
/// These do not cover a path any crate in this repository takes. No tracked
/// crate sets `split_dispatch = true` — `rg '^\s*split_dispatch\s*=\s*true'
/// $(git ls-files)` returns nothing, tests included. The configuration is
/// shipped only by out-of-tree consumers that path-depend on this crate. These
/// tests are therefore the only in-repository coverage of that configuration.
///
/// The `^\s*` anchor is load-bearing; do not simplify it away. It restricts
/// the match to the attribute-argument form, which is the only spelling that
/// sets anything. Unanchored, the pattern matches the sentence above — this
/// note is a tracked file, so a census over an unanchored pattern would count
/// this prose as a setter. Anchoring rejects prose without rejecting the
/// attribute spelling it exists to find.
///
/// The gap that leaves is on the more dangerous half.
/// `dispatch_arm_inline_call_tokens_i` — the emitter these tests exist to keep
/// `return` arms away from — has no test at all: it appears in this file only
/// as its definition, one comment, and one call site. Misrouting a `return` arm
/// into it writes the arm's return VALUE into the green pc, which compiles
/// clean (see the note on `arm_is_pure_pc_advance`'s `visit_expr_return`).
///
/// Retirement: this note stops being true when
/// `rg '^\s*split_dispatch\s*=\s*true' $(git ls-files)` returns a hit — i.e.
/// when a tracked crate ships the configuration. Run it anchored, for the
/// reason above: unanchored it reports a hit that is this note, and retires a
/// note nothing has retired. At that point replace this note with a real test
/// on the emitter rather than deleting it. `switch_dispatch`, set on the
/// adjacent line by the same out-of-tree consumers, already has a tracked
/// setter in `majit-metainterp/tests/jit_interp_dispatch_ir_shape.rs`, so this is a
/// known-shaped piece of work rather than an aspiration.
#[cfg(test)]
mod arm_is_pure_pc_advance_tests {
    use super::*;

    fn body(src: &str) -> Expr {
        syn::parse_str::<Expr>(src).unwrap()
    }

    #[test]
    fn plain_forward_advance_is_pure() {
        assert_eq!(arm_is_pure_pc_advance(&body("{ pc += 3; }")), Some(3));
        assert_eq!(arm_is_pure_pc_advance(&body("{ pc = pc + 2; }")), Some(2));
        assert_eq!(
            arm_is_pure_pc_advance(&body(
                "{ let r = program[pc] as i8 as i64; storage_roll(&mut state, r); pc += 1; }"
            )),
            Some(1),
            "a residual helper does not own the caller's green-pc return channel",
        );
    }

    /// A `return` arm must NEVER be routed to the split sub-JitCode path: the
    /// paired `inline_call_<types>_i` wires the callee's BC_INT_RETURN into the
    /// caller's `return_i_reg`, which under `split_dispatch` is the dispatch
    /// loop's green pc register — so the arm would write its return VALUE into
    /// the pc.  Return arms stay force-inlined, where
    /// `Lowerer::lower_return_stmt` emits the terminator in the dispatch
    /// JitCode itself.
    #[test]
    fn return_arm_is_never_a_pure_pc_advance() {
        // Tail return, the shape every corpus interpreter writes.
        assert_eq!(
            arm_is_pure_pc_advance(&body("{ let r = program[pc + 1]; return state.regs[r]; }")),
            None,
        );
        // Bare return, and a return that still ends with a forward advance:
        // the advance must not make the arm look splittable.
        assert_eq!(arm_is_pure_pc_advance(&body("{ return; }")), None);
        assert_eq!(
            arm_is_pure_pc_advance(&body("{ if done { return state.a; } pc += 2; }")),
            None,
        );
    }

    #[test]
    fn branch_and_back_edge_arms_are_not_pure() {
        assert_eq!(
            arm_is_pure_pc_advance(&body("{ pc = tgt; continue; }")),
            None
        );
        assert_eq!(
            arm_is_pure_pc_advance(&body("{ can_enter_jit!(d, pc, s, p, || {}); pc += 1; }")),
            None,
        );
        // Backward / zero advances are not forward progress.
        assert_eq!(arm_is_pure_pc_advance(&body("{ pc += 0; }")), None);
    }
}

#[cfg(test)]
mod pc_pinned_write_tests {
    use super::*;

    fn binding(reg: u16, kind: BindingKind) -> Binding {
        Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    #[test]
    fn pc_assign_from_another_reg_is_int_copy_not_add_zero() {
        let mut lowerer = Lowerer::new(None);
        lowerer.pc_pinned = true;
        lowerer
            .bindings
            .insert("pc".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("tgt".to_string(), binding(3, BindingKind::Int));
        let expr: Expr = syn::parse_str("pc = tgt").expect("parse pc assign");

        assert_eq!(lowerer.lower_pc_pinned_write(&expr), Some(()));
        assert_eq!(lowerer.bindings["pc"].reg, 0);

        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(
            emitted.contains("move_i"),
            "pc = tgt must be int_copy, got {emitted}"
        );
        assert!(
            !emitted.contains("IntAdd"),
            "pc = tgt must not record IntAdd(x, 0), got {emitted}"
        );
    }

    /// `pc += 1` keeps the increment a constant operand of `int_add`
    /// (`assembler.py` `emit_const`), with no register load before it.
    #[test]
    fn pc_increment_is_an_add_with_a_constant_operand() {
        let mut lowerer = Lowerer::new(None);
        lowerer.pc_pinned = true;
        lowerer
            .bindings
            .insert("pc".to_string(), binding(0, BindingKind::Int));
        let expr: Expr = syn::parse_str("pc += 1").expect("parse pc increment");

        assert_eq!(lowerer.lower_pc_pinned_write(&expr), Some(()));
        assert_eq!(lowerer.op_metadata.len(), 1);
        assert_eq!(lowerer.op_metadata[0].reads, vec![Register::int(0)]);
        assert_eq!(lowerer.op_metadata[0].writes, vec![Register::int(0)]);
        let emitted = lowerer.statements[0].to_string();
        assert!(
            emitted.contains("record_binop_i_const") && emitted.contains("IntAdd"),
            "pc += 1 must be int_add with a constant, got {emitted}"
        );
        assert!(
            !emitted.contains("load_const_i_value"),
            "pc += 1 must not load the constant into a register, got {emitted}"
        );
    }
}

impl<'c> Lowerer<'c> {
    /// Green-pc inline dispatch pc-write pinning (see `Lowerer::pc_pinned`).
    /// Lowers `pc += N`, `pc = pc + N`, and the generic branch `pc = <expr>`
    /// so the result lands in pc's register (reg0), the slot the dispatch
    /// merge point reads.  `pc += N` emits `int_add(pc_reg, Constant(N))`
    /// through `record_binop_i_const` — the same advance shape as the dispatch-top
    /// opcode-fetch (`try_lower_opcode_fetch_stmt` Pattern 2).  `pc = target`
    /// lowers the RHS then copies it into pc_reg via `int_copy` / `move_i`
    /// (`blackhole.py bhimpl_int_copy`); that is a register rename, not an
    /// `IntAdd(rhs, 0)` resop.  Returns `None` when `!pc_pinned` or
    /// `expr` is not a pc-write, so the caller falls through to the normal
    /// statement lowering (and, off the inline path, to the existing
    /// SSA-rebind / drop behaviour — pinning is inert there).
    pub(super) fn lower_pc_pinned_write(&mut self, expr: &Expr) -> Option<()> {
        if !self.pc_pinned {
            return None;
        }
        let pc_reg = self.bindings.get("pc")?.reg;

        if let Some(increment) = pc_self_increment(expr) {
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    vec![Register::int(pc_reg)],
                    vec![Register::int(pc_reg)],
                ),
                quote::quote! {
                    __builder.record_binop_i_const(
                        #pc_reg as u16,
                        majit_ir::OpCode::IntAdd,
                        #pc_reg as u16,
                        #increment as i64,
                    );
                },
            );
            // pc binding stays at pc_reg (no rebind).
            return Some(());
        }

        if let Expr::Assign(assign) = expr
            && expr_single_ident(&assign.left).as_deref() == Some("pc")
        {
            let rhs = self.lower_value_expr(&assign.right)?;
            if !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            if rhs.reg != pc_reg {
                let rhs_reg = rhs.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Aux,
                        vec![Register::int(rhs_reg)],
                        vec![Register::int(pc_reg)],
                    ),
                    quote::quote! {
                        __builder.move_i(#pc_reg as u16, #rhs_reg as u16);
                    },
                );
            }
            // Re-pin `pc` to reg0, overriding any SSA rebind the RHS
            // lowering may have left.
            self.bindings.insert(
                "pc".to_string(),
                Binding {
                    reg: pc_reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                },
            );
            return Some(());
        }
        None
    }

    /// Whether an arm contains an auto-discovered residual helper. These calls
    /// stay in a sub-JitCode so their guard-resume liveness is scoped to the
    /// callee frame; a trailing green-pc advance is returned explicitly to the
    /// caller by the split path below.
    pub(super) fn arm_body_has_infer_call(&self, body: &Expr) -> bool {
        let auto_calls =
            self.auto_calls || matches!(self.inference_failure_mode, InferenceFailureMode::Panic);
        expr_has_infer_or_pipeline_call(body, &self.call_policies, auto_calls)
    }

    /// Lower a dispatch arm `body` INLINE into this dispatch JitCode (green-pc
    /// path).  Sets `pc_pinned` so pc-writes hit reg0, lowers each statement
    /// via `lower_stmt`, and on success drops the arm-local bindings so they
    /// do not leak into the next arm (matching the isolated sub-JitCode arm
    /// scope; emitted ops reference concrete registers, so this is safe).
    /// Returns [`InlineArmOutcome::Rejected`] and fully rolls back the partial
    /// emission when the body does not lower cleanly, so the caller falls back
    /// to the sub-JitCode path.
    ///
    /// The body's final statement is marked via `inline_arm_tail_stmt` so a
    /// `return` there lowers to a typed return terminator
    /// (`Lowerer::lower_return_stmt`); when one was emitted the outcome is
    /// [`InlineArmOutcome::InlinedTerminal`] and the caller must suppress the
    /// dispatch back-edge.
    pub(super) fn try_inline_dispatch_arm(&mut self, body: &Expr) -> InlineArmOutcome {
        let stmts = extract_stmts(body);
        let snap_stmts = self.statements.len();
        let snap_meta = self.op_metadata.len();
        let snap_reg = self.next_reg;
        // `next_label` belongs with `snap_stmts`: a lowerer that allocates a
        // label and then fails leaves the counter advanced, and this is the
        // one place that knows the whole attempt is being discarded.  Restoring
        // it anywhere the statements survive would re-issue a bound ident (see
        // `Lowerer::alloc_label`), so the two are snapshotted and put back
        // together.
        let snap_label = self.next_label;
        let snap_bindings = self.bindings.clone();
        let snap_opcode = self.opcode_var_name.clone();
        let snap_inline_prebuild = self.inline_liveness_prebuild.len();

        self.pc_pinned = true;
        let mut ok = true;
        for (idx, stmt) in stmts.iter().enumerate() {
            // One-shot marker for the tail statement; `lower_stmt` takes it on
            // entry, so only a `return` written directly as the body's last
            // statement can reach the terminator emission.
            self.inline_arm_tail_stmt = idx + 1 == stmts.len();
            if self.lower_stmt(stmt).is_none() {
                ok = false;
                break;
            }
        }
        // Clear unconditionally: `lower_stmt` consumes the flag on entry, but a
        // body whose tail statement never reached `lower_stmt` (the `break`
        // above) would otherwise leave it set for the next arm.
        self.inline_arm_tail_stmt = false;
        self.pc_pinned = false;

        if !ok {
            self.statements.truncate(snap_stmts);
            self.op_metadata.truncate(snap_meta);
            self.next_reg = snap_reg;
            self.next_label = snap_label;
            self.bindings = snap_bindings;
            self.opcode_var_name = snap_opcode;
            self.inline_liveness_prebuild.truncate(snap_inline_prebuild);
            return InlineArmOutcome::Rejected;
        }

        if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
            let kinds: Vec<String> = self.op_metadata[snap_meta..]
                .iter()
                .map(|m| format!("{:?}", m.kind))
                .collect();
            eprintln!(
                "[majit-macro] inline arm body ops ({}): {}",
                kinds.len(),
                kinds.join(",")
            );
        }
        // Inlined: restore the binding map / opcode name so arm-local `let`s
        // do not leak across arms, and reclaim the register space.  Each arm
        // sits behind its own opcode guard (the skip-label chain), so only one
        // arm body executes per dispatch iteration and arm-local temporaries
        // are dead at the back-edge (the merge point keeps only pc / program /
        // state slots live).  Reusing the same registers across the mutually
        // exclusive arm bodies keeps the dispatch JitCode's register count
        // bounded — without it ~20 inlined arms overflow the u8 const-ref slot
        // space (statements stay committed; the emitted ops already hold their
        // concrete register numbers).
        self.bindings = snap_bindings;
        self.opcode_var_name = snap_opcode;
        self.next_reg = snap_reg;

        // Did the body end in a return terminator?  `lower_stmt` takes the
        // one-shot tail marker, so only the body's final statement can emit
        // one — assert that, because a terminator anywhere but last would mean
        // the ops after it are unreachable.
        let arm_ops = &self.op_metadata[snap_meta..];
        match arm_ops
            .iter()
            .position(|m| matches!(m.control, ControlFlowClass::Terminal))
        {
            Some(pos) => {
                debug_assert_eq!(
                    pos,
                    arm_ops.len() - 1,
                    "a dispatch arm return terminator must be the arm's last op"
                );
                InlineArmOutcome::InlinedTerminal
            }
            None => InlineArmOutcome::Inlined,
        }
    }
}

/// Result of [`Lowerer::try_inline_dispatch_arm`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InlineArmOutcome {
    /// Not inlined: the body did not lower cleanly and the partial emission was
    /// rolled back.  The caller falls back to the sub-JitCode path.
    Rejected,
    /// Inlined; control falls out of the arm body and the caller emits the
    /// dispatch back-edge.
    Inlined,
    /// Inlined, and the body ends in a typed return terminator.  The caller
    /// must emit **neither** the loop back-edge nor the `default_label` jump:
    /// the terminator is the arm's own control transfer.
    ///
    /// This differs from `ArmPattern::Halt`, which diverts to `default_label`
    /// precisely *because* it has no operand of its own and must reach the
    /// function-final typed return to produce one.  A returning arm already
    /// carries its value, so routing it through `default_label` would re-lower
    /// the function's trailing expression over it.
    InlinedTerminal,
}

/// Where the dispatch chain's unmatched-opcode edge goes.
///
/// The edge used to be one thing: `default_label`, bound at the portal
/// function's trailing return (`lower_dispatch_body`). That is right for
/// `_ => break`, which is what almost every machine here writes — 55 of the
/// corpus's default arms, plus 30 more that end in a divergent macro. It is
/// wrong for the other two spellings, and wrongly in the worst way: the walk
/// reports a finished frame, `take_single_pass_finish` breaks the native loop,
/// and the portal returns a partial answer while `degraded_dispatch_arms()`
/// and the arm census both read healthy.
enum DefaultEdge {
    /// `_ => break` and the divergent macros: the loop is over, take the
    /// function's typed return.
    Return,
    /// `_ => {}`: fall out of the match and run the next iteration.
    NextIteration,
    /// A default arm with a body. It is an arm like any other, just with no
    /// opcode test in front of it, so lower it as one — a body that cannot
    /// lower then degrades to an abort stub, which is the fail-closed path
    /// every other arm already has.
    LowerAsArm,
}

fn classify_default_edge(body: &Expr) -> DefaultEdge {
    fn leaves_the_loop(body: &Expr) -> bool {
        match body {
            Expr::Break(_) | Expr::Return(_) => true,
            Expr::Macro(m) => is_divergent_macro(&m.mac.path),
            // Only when that is the whole body: an exit preceded by work
            // would leave the work out of the trace, which is the same loss
            // one level down.
            Expr::Block(b) => match b.block.stmts.as_slice() {
                [Stmt::Expr(expr, _)] => leaves_the_loop(expr),
                [Stmt::Macro(m)] => is_divergent_macro(&m.mac.path),
                _ => false,
            },
            _ => false,
        }
    }
    if leaves_the_loop(body) {
        return DefaultEdge::Return;
    }
    match body {
        Expr::Block(b) if b.block.stmts.is_empty() => DefaultEdge::NextIteration,
        _ => DefaultEdge::LowerAsArm,
    }
}

/// Macros that never return, so an arm ending in one cannot fall through.
fn is_divergent_macro(path: &syn::Path) -> bool {
    path.segments.last().is_some_and(|seg| {
        matches!(
            seg.ident.to_string().as_str(),
            "panic" | "unreachable" | "todo" | "unimplemented" | "compile_error"
        )
    })
}

/// True for the arm the dispatch chain falls through to: `_`, or a lower-case
/// binding.
fn is_default_dispatch_arm(arm: &crate::jit_interp::classify::ClassifiedArm) -> bool {
    matches!(arm.pat, Pat::Wild(_)) || is_lowercase_binding_pat(&arm.pat)
}

pub(super) fn lower_dispatch_chain(
    lowerer: &mut Lowerer,
    classified_arms: &[crate::jit_interp::classify::ClassifiedArm],
    config: &LowererConfig,
    loop_start_label: &syn::Ident,
) -> syn::Ident {
    // Allocate the default/exit label (BC_GOTO target when no arm matches).
    // Allocated before the opcode-reg guard so we always have a label to return.
    let default_label = lowerer.alloc_label();
    lowerer.emit_aux(quote::quote! { let #default_label = __builder.new_label(); });

    // Where the unmatched-opcode edge goes. A match over `u8` needs a default
    // arm to be exhaustive, so `None` here means this is not a dispatch shape;
    // leave that case at the return edge it already had.
    let default_kind = classified_arms
        .iter()
        .find(|arm| is_default_dispatch_arm(arm))
        .map_or(DefaultEdge::Return, |arm| {
            classify_default_edge(&arm.original_body)
        });
    let lower_default_as_arm = matches!(default_kind, DefaultEdge::LowerAsArm);
    // Bound at the default arm's own body below; the switch's default and the
    // guard chain's fall-through both target it.
    let default_body_label = lower_default_as_arm.then(|| {
        let label = lowerer.alloc_label();
        lowerer.emit_aux(quote::quote! { let #label = __builder.new_label(); });
        label
    });
    let default_edge = match &default_kind {
        DefaultEdge::Return => default_label.clone(),
        DefaultEdge::NextIteration => loop_start_label.clone(),
        DefaultEdge::LowerAsArm => default_body_label
            .clone()
            .expect("allocated on this arm of the same match"),
    };

    // Retrieve the opcode register installed by the opcode-fetch lowerer.
    // Slice ε.1: prefer the consumer's chosen opcode-result name (set by
    // `try_lower_opcode_fetch_stmt` when it recognised the
    // `let <name> = program[<idx>]` pattern); fall back to the literal
    // `"opcode"` (PyPy `pyopcode.py:171` canonical name) so existing
    // fixtures whose dispatch loops use that name continue to lower.
    // If neither is bound (e.g. skeleton without opcode-fetch), skip
    // chain emission.
    let opcode_lookup_name: String = lowerer
        .opcode_var_name
        .clone()
        .unwrap_or_else(|| "opcode".to_string());
    let opcode_reg = match lowerer.bindings.get(&opcode_lookup_name) {
        Some(b) if matches!(b.kind, BindingKind::Int) => b.reg,
        _ => return default_label,
    };

    let mut switch_arm_labels: Vec<Option<syn::Ident>> = vec![None; classified_arms.len()];
    if config.switch_dispatch {
        let mut switch_case_emitters = Vec::new();
        for (arm_idx, arm) in classified_arms.iter().enumerate() {
            if is_default_dispatch_arm(arm) {
                continue;
            }
            let arm_label = lowerer.alloc_label();
            lowerer.emit_aux(quote::quote! { let #arm_label = __builder.new_label(); });
            match extract_pat_switch_case_tokens(&arm.pat, &arm_label) {
                Some(mut emitters) => {
                    switch_case_emitters.append(&mut emitters);
                    switch_arm_labels[arm_idx] = Some(arm_label);
                }
                None => continue,
            }
        }
        lowerer.emit_aux(quote::quote! {
            let mut __switch_cases: Vec<(i64, u16)> = Vec::new();
            #(#switch_case_emitters)*
        });
        let live_targets = switch_arm_labels
            .iter()
            .filter_map(|label| label.clone())
            .collect::<Vec<_>>();
        // RPython flatten.py:270-308 emits one `-live-` before `switch`;
        // pyjitpl.py opimpl_switch records GuardValue on a hit and fallback guards
        // on a miss. The marker keeps the opcode and all case targets live.
        lowerer.emit_op(
            OpMeta::live_marker_with(vec![Register::int(opcode_reg)], live_targets),
            quote::quote! { let _ = __builder.live_placeholder(); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::Aux, vec![Register::int(opcode_reg)], vec![]),
            quote::quote! {
                __builder.switch(#opcode_reg as u16, &__switch_cases);
            },
        );
        lowerer.emit_jump(&default_edge);
    }

    // The denominator for `record_degraded_dispatch_arm` below, staged from the
    // same loop's own admission test so the two cannot drift: an arm is counted
    // here exactly when the loop emits a body for it.  Without it an empty
    // degraded registry reads as "nothing degraded" and as "no portal was
    // built" at once, and only the first is a pass.
    {
        let census_interp = config.state_type_name.clone();
        let census_arms = classified_arms
            .iter()
            .filter(|arm| !is_default_dispatch_arm(arm) || lower_default_as_arm)
            .count();
        lowerer.emit_aux(quote::quote! {
            majit_metainterp::record_dispatch_arm_census(#census_interp, #census_arms);
        });
    }

    for (arm_idx, arm) in classified_arms.iter().enumerate() {
        // `_` wildcard: skip here; handled by the default GOTO below.
        // All other patterns (including Pat::Ident like `OP_NOP`) are
        // treated as constant tests and emitted as goto_if_not_int_eq.
        let is_default_arm = is_default_dispatch_arm(arm);
        if is_default_arm && !lower_default_as_arm {
            continue;
        }

        let mut skip_label = None;
        if is_default_arm {
            // No opcode test in front of it: this is where every opcode no arm
            // matched arrives. Rust puts the default arm last, so the previous
            // arm's skip label falls straight into this body, and the switch
            // dispatch jumps here explicitly.
            lowerer.emit_label_def(
                default_body_label
                    .as_ref()
                    .expect("allocated whenever the default arm is lowered as one"),
            );
        } else if let Some(match_label) = switch_arm_labels[arm_idx].as_ref() {
            lowerer.emit_label_def(match_label);
        } else if config.switch_dispatch {
            continue;
        } else {
            // Extract token expressions for each value in the pattern.
            // extract_pat_value_tokens handles Pat::Lit, Pat::Path, Pat::Or.
            let value_tokens = match extract_pat_value_tokens(&arm.pat) {
                Some(v) => v,
                None => {
                    if pat_contains_range(&arm.pat) && !config.split_dispatch {
                        lowerer.emit_aux(quote::quote! {
                            compile_error!(
                                "jit_interp dispatch range patterns require `switch_dispatch = true`"
                            );
                        });
                    }
                    // Skipping emits NOTHING — no test, no body, and (unlike the
                    // three abort-stub sites below) no `record_degraded_arm`,
                    // because that call lives at stub construction and there is
                    // no stub here. So the arm used to vanish completely: the
                    // opcode fell through to `default_label`, which is bound at
                    // the typed-return terminator, meaning the trace *returns*
                    // rather than aborting. An interpreter all of whose arms
                    // took this path installed a dispatch whose entire optimized
                    // body was that return, while `Traces compiled` and
                    // `degraded_dispatch_arms()` both read healthy.
                    //
                    // Register it here so the next unsupported shape is named
                    // instead of disappearing. This is a diagnostic, not a
                    // repair: the arm is still absent from the dispatch, and the
                    // silent return above is still what happens at runtime.
                    let interp = &config.state_type_name;
                    let arm_name = {
                        let pat = &arm.pat;
                        quote::quote!(#pat).to_string()
                    };
                    lowerer.emit_aux(quote::quote! {
                        majit_metainterp::record_degraded_dispatch_arm(
                            #interp,
                            #arm_name,
                            "dispatch arm pattern is not a shape the macro can lower to a \
                             constant value; the arm was dropped from the dispatch entirely",
                        );
                    });
                    continue; // unsupported pattern shape — skip
                }
            };

            // Allocate a skip label: if opcode ≠ this arm's value, jump here.
            // Task 1.6 will emit the arm body between the check and this label.
            let arm_skip_label = lowerer.alloc_label();
            lowerer.emit_aux(quote::quote! { let #arm_skip_label = __builder.new_label(); });
            skip_label = Some(arm_skip_label.clone());

            let matched_label = if value_tokens.len() > 1 {
                let label = lowerer.alloc_label();
                lowerer.emit_aux(quote::quote! { let #label = __builder.new_label(); });
                Some(label)
            } else {
                None
            };

            for (value_idx, val_tok) in value_tokens.iter().enumerate() {
                // Load the pattern constant into a fresh int register.
                let const_reg = lowerer.alloc_reg();
                lowerer.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                    quote::quote! {
                        __builder.load_const_i_value(#const_reg as u16, #val_tok);
                    },
                );
                let is_last_value = value_idx + 1 == value_tokens.len();
                let miss_label = if is_last_value {
                    arm_skip_label.clone()
                } else {
                    let label = lowerer.alloc_label();
                    lowerer.emit_aux(quote::quote! { let #label = __builder.new_label(); });
                    label
                };
                // RPython flatten.py:258-260 emits `-live-` UNCONDITIONALLY ahead
                // of every goto_if_not / goto_if_not_<cmp>; optimize_goto_if_not
                // (jtransform.py) tags the fused compare `-live-before`. The
                // tracer records this guard through record_state_guard →
                // build_state_field_snapshot, which reads the LIVE marker at
                // `guard_pc - SIZE_LIVE_OP`. Without a preceding `-live-` the
                // snapshot mis-decodes the prior op's bytes as a liveness offset
                // and indexes out of bounds. One marker per chained alternative so
                // each goto_if_not_int_eq has its own preceding LIVE.
                lowerer.emit_op(
                    OpMeta::live_marker(),
                    quote::quote! { let _ = __builder.live_placeholder(); },
                );
                // Fused goto_if_not_int_eq: branch to the next alternative (or the
                // arm skip label) if opcode != const.
                lowerer.emit_op(
                    OpMeta::conditional_guard_compare(
                        Register::int(opcode_reg),
                        Register::int(const_reg),
                        miss_label.clone(),
                    ),
                    quote::quote! {
                        __builder.goto_if_not_int_eq(#opcode_reg as u16, #const_reg as u16, #miss_label);
                    },
                );
                if let Some(matched_label) = matched_label.as_ref() {
                    lowerer.emit_jump(matched_label);
                    if !is_last_value {
                        lowerer.emit_label_def(&miss_label);
                    }
                }
            }
            if let Some(matched_label) = matched_label.as_ref() {
                lowerer.emit_label_def(matched_label);
            }
        }

        // Green-pc gated inline (Option A, recursive-call): when `pc` is a declared
        // green, lower a Lowerable arm body DIRECTLY into this dispatch
        // JitCode so its pc-writes (operand `pc += N`, branch `pc = target`)
        // reach the dispatch loop's reg0.  A BC_INLINE_CALL into a sub-JitCode
        // copies args caller→callee only, so a sub-JitCode pc-write never
        // reaches reg0 (the merge-point pc) — the inline path is what makes
        // the green pc advance. `try_inline_dispatch_arm` is transactional, so
        // an inferred-policy call whose arguments cannot lower rolls back the
        // whole attempt and falls through to the diagnostic sub-JitCode abort
        // stub; a supported residual call keeps the caller's pc in this live
        // frame. Red-pc dispatches keep the sub-JitCode path unconditionally.
        //
        // `split_dispatch` opt-in: a pure forward-advancing green-pc arm is
        // routed to the per-arm sub-JitCode path (below) with a pc-returning
        // `inline_call_<types>_i` instead of being force-inlined here — this is
        // what keeps the dispatch JitCode's register/const footprint small.
        // `Some(N)` carries the arm's `pc += N` advance to the body generator.
        // This return channel is independent of whether the work contains an
        // inferred-policy residual call: if that work cannot lower, the
        // transactional sub-lowerer returns an abort stub; if it can, dropping
        // the green return would resume at the old opcode. RPython keeps `pc`
        // in the caller frame across either call shape.
        // Infer and `inline_pipeline_*` calls stay on the sub-JitCode path
        // even without the size opt-in. Infer resume guards belong to the
        // callee frame while the caller still owns green `pc`. Pipeline
        // callees are resolved at install (`trailing_return_info`); a
        // missing typed return emits `return None` via
        // `inference_failure_tokens`, and that `None` must stay inside the
        // per-arm IIFE so only that arm degrades. Force-inlining would
        // make the same `return None` abort the whole dispatch JitCode.
        let has_infer_call = lowerer.arm_body_has_infer_call(&arm.original_body);
        let pc_return_increment = if (config.split_dispatch || has_infer_call)
            && pc_is_green(config)
            && matches!(
                arm.pattern,
                crate::jit_interp::classify::ArmPattern::Lowerable
            ) {
            arm_is_pure_pc_advance(&arm.original_body)
        } else {
            None
        };
        let inline_outcome = if pc_return_increment.is_none()
            && matches!(
                arm.pattern,
                crate::jit_interp::classify::ArmPattern::Lowerable
            )
            && !has_infer_call
        {
            // Red `pc` still lives in the merge-point register file. A
            // sub-JitCode `pc += N` never writes that slot back
            // (`BC_INLINE_CALL` is caller→callee only), so the next
            // merge point rereads a stale pc. Inline with `pc_pinned`
            // is the same Variable-in-place write RPython's flatten
            // keeps for `next_instr`.
            lowerer.try_inline_dispatch_arm(&arm.original_body)
        } else {
            InlineArmOutcome::Rejected
        };
        let inlined = inline_outcome != InlineArmOutcome::Rejected;

        // Install-time name for an arm that degrades to an abort stub.
        //
        // `MAJIT_MACRO_DEBUG` below is a proc-macro-time census of EVERY arm:
        // it is invisible at install, absent from any rebuild that hits the
        // incremental cache, and its `inlined` field does not answer this
        // question — `inlined=false` only means "emitted as a sub-JitCode",
        // and the `Some(layout)` path below builds a real one.  So the abort
        // stubs get their own channel, keyed by the machine's `state = T`
        // name and the arm's source spelling, staged with the reason at each
        // emitting site (`majit_metainterp::record_degraded_dispatch_arm`).
        // It is deliberately NOT a `MC_DIAG` bump: slot 41 is the
        // `AbortReason::Generic` bucket `jitprof.rs` already documents as
        // unclassified, and the `Counters.ABORT_*` ids it indexes mirror
        // RPython's `Counters`, so there is no id to spend on this.
        let degraded_interp_name = config.state_type_name.clone();
        let degraded_arm_name = {
            let pat = &arm.pat;
            quote::quote!(#pat).to_string()
        };
        let record_degraded_arm = |reason: &str| {
            let interp = &degraded_interp_name;
            let arm_name = &degraded_arm_name;
            quote::quote! {
                majit_metainterp::record_degraded_dispatch_arm(#interp, #arm_name, #reason);
            }
        };
        // `jitcode.py JitCode.__init__ self.name = name` — the name every sub-JitCode built
        // for this arm carries.  The same spelling the degraded-arm record
        // uses, so a log line and a `degraded_dispatch_arms()` entry for one
        // arm can be matched by eye.
        //
        // Every arm gets it, not just the ones that degrade: `BC_ABORT` has
        // two emitter families and one opcode, and the abort log discriminates
        // them by frame shape (`depth>1 body=1`) precisely because the name was
        // empty.  A name only the abort stubs carried would still leave the
        // healthy arms indistinguishable from each other in the bytecode
        // encoder's ceiling audit, which prints one line per finished builder.
        let sub_jitcode_name = format!("{degraded_interp_name}::{degraded_arm_name}");

        if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
            let pat = &arm.pat;
            let pattern_name = match &arm.pattern {
                crate::jit_interp::classify::ArmPattern::Lowerable => "Lowerable".to_string(),
                crate::jit_interp::classify::ArmPattern::Nop => "Nop".to_string(),
                crate::jit_interp::classify::ArmPattern::Halt => "Halt".to_string(),
                crate::jit_interp::classify::ArmPattern::AbortPermanent => {
                    "AbortPermanent".to_string()
                }
                crate::jit_interp::classify::ArmPattern::Unsupported(reason) => {
                    format!("Unsupported({reason})")
                }
            };
            eprintln!(
                "[majit-macro] dispatch arm {} pattern={} inlined={} outcome={:?}",
                quote::quote!(#pat),
                pattern_name,
                inlined,
                inline_outcome,
            );
        }

        if !inlined {
            // PyPy `pyopcode.py` dispatch is one graph: a matched arm is
            // more ops in the same JitCode, not `inline_call` to a
            // per-arm sub-JitCode. Nop/Halt emit nothing (the jump below
            // is the control transfer). AbortPermanent / Unsupported
            // abort THIS frame. Only a Lowerable body that could not be
            // inlined still builds a sub-JitCode (split_dispatch /
            // infer-call).
            match &arm.pattern {
                crate::jit_interp::classify::ArmPattern::Nop
                | crate::jit_interp::classify::ArmPattern::Halt => {}
                crate::jit_interp::classify::ArmPattern::AbortPermanent => {
                    lowerer.emit_op(
                        OpMeta::terminal(Vec::new()),
                        quote::quote! { __builder.abort_permanent(); },
                    );
                }
                crate::jit_interp::classify::ArmPattern::Unsupported(reason) => {
                    let record_degraded = record_degraded_arm(reason);
                    lowerer.emit_aux(record_degraded);
                    lowerer.emit_op(
                        OpMeta::terminal(Vec::new()),
                        quote::quote! { __builder.abort(); },
                    );
                }
                crate::jit_interp::classify::ArmPattern::Lowerable => {
                    // jtransform.handle_regular_call — inline_call_* + trailing -live-.
                    // Build the arm sub-JitCode and register it; emit BC_INLINE_CALL.
                    // This executes when the arm MATCHED (guards fell through); the
                    // sub-JitCode encodes the opcode handler body.
                    //
                    // Dispatch-arm caller-local plumbing: walk the arm
                    // body to collect parent-scope idents (via `collect_arm_caller_locals`),
                    // pre-bind them on the sub-Lowerer at fresh per-bank callee regs
                    // (via `try_generate_jitcode_body_parts_with_caller_bindings`), and
                    // emit the typed `inline_call_<types>_v(__sub_idx, args_i, args_r,
                    // args_f)` so the callee jitcode receives them as portal-input
                    // bindings.  Mirrors `handle_regular_call` `inline_call_<types>(jitcode,
                    // args...)`.  When the arm body has no parent-scope refs the layout
                    // is empty and the emit reduces to the no-arg `inline_call_r_v`
                    // (equivalent to the previous `__builder.inline_call(__sub_idx)`).
                    let mut arm_inline_call_reads: Vec<Register> = Vec::new();
                    let (arm_body_tokens, arm_inline_call_emit): (
                        proc_macro2::TokenStream,
                        proc_macro2::TokenStream,
                    ) = {
                        let caller_locals = collect_arm_caller_locals(
                            &arm.original_body,
                            &arm.pat,
                            &lowerer.bindings,
                        );
                        let generated_parts = match pc_return_increment {
                            // `split_dispatch` pure arm: lower into a sub-JitCode
                            // that RETURNS pc + N; the third tuple slot carries the
                            // caller pc register the `_i` writeback targets.
                            Some(increment) => {
                                try_generate_jitcode_pc_return_body_with_caller_bindings(
                                    &arm.original_body,
                                    Some(config),
                                    &caller_locals,
                                    increment,
                                )
                                .map(|(generated, layout)| {
                                    let pc_reg = layout
                                        .iter()
                                        .find(|e| e.name == "pc")
                                        .map(|e| e.parent_reg)
                                        .unwrap_or(0);
                                    (generated, layout, Some(pc_reg))
                                })
                            }
                            None => try_generate_jitcode_body_parts_with_caller_bindings(
                                &arm.original_body,
                                Some(config),
                                &caller_locals,
                            )
                            .map(|(generated, layout)| (generated, layout, None)),
                        };
                        match generated_parts {
                            Ok((generated, layout, pc_return_reg)) => {
                                let body = generated.body;
                                let liveness_prebuild = generated.liveness_prebuild;
                                lowerer.inline_liveness_prebuild.push(liveness_prebuild);
                                // Carry the parent-side caller regs into the
                                // BC_INLINE_CALL OpMeta so the liveness walker
                                // accounts for them as live at the call site
                                // (assembler.py get_liveness_info reads).
                                for entry in &layout {
                                    arm_inline_call_reads
                                        .push(Register::new(entry.kind, entry.parent_reg));
                                }
                                for reg in config.identity_slot_registers() {
                                    if !arm_inline_call_reads.contains(&reg) {
                                        arm_inline_call_reads.push(reg);
                                    }
                                }
                                // `MIFrame.setup_call` / `bhimpl_inline_call_ir_v`
                                // give the callee only the declared argument
                                // boxes. A plain `[int]` ptr/len is appended
                                // as `(slot, slot)` here, not through
                                // `assign_caller_local_layout`.
                                let identity_int_pairs =
                                    plain_array_identity_int_pairs(Some(config));
                                let inline_call_emit = match pc_return_reg {
                                    Some(pc_reg) => dispatch_arm_inline_call_tokens_i(
                                        &layout,
                                        pc_reg,
                                        &identity_int_pairs,
                                    ),
                                    None => dispatch_arm_inline_call_tokens(
                                        &layout,
                                        &identity_int_pairs,
                                    ),
                                };
                                // Reserve the identity-slot prefix in the sub-JitCode
                                // register file so int[int_identity_base..int_end) /
                                // ref[ref_identity_base..ref_end) exist as real
                                // registers — the arm body's state-field ops address
                                // them and the resume path re-derives them at deopt.
                                // Identity slots are the portal's argument
                                // registers (`MIFrame.setup_call`) and must
                                // exist in every arm whose `load/store_state_field*`
                                // addresses them, including `switch_dispatch`.
                                let (__split_int_end, __split_ref_end) =
                                    config.split_identity_reg_ends();
                                let min_i_regs = layout
                                    .iter()
                                    .filter(|e| matches!(e.kind, BindingKind::Int))
                                    .map(|e| e.callee_reg + 1)
                                    .max()
                                    .unwrap_or(0)
                                    .max(__split_int_end);
                                let min_r_regs = layout
                                    .iter()
                                    .filter(|e| matches!(e.kind, BindingKind::Ref))
                                    .map(|e| e.callee_reg + 1)
                                    .max()
                                    .unwrap_or(0)
                                    .max(__split_ref_end);
                                let min_f_regs = layout
                                    .iter()
                                    .filter(|e| matches!(e.kind, BindingKind::Float))
                                    .map(|e| e.callee_reg + 1)
                                    .max()
                                    .unwrap_or(0);
                                let record_degraded = record_degraded_arm(
                                    "arm body lowering resolved an unsupported call policy at install",
                                );
                                (
                                    quote::quote! {
                                        // A runtime-resolved unsupported call policy
                                        // (`inference_failure_tokens` → `return None`,
                                        // e.g. a `#[dont_look_inside]` helper whose
                                        // signature has no marshalable C-ABI call
                                        // target) escapes arm-body lowering as the
                                        // IIFE's `None`.  Degrade THIS arm to an abort
                                        // sub-JitCode rather than failing the whole
                                        // `__dispatch_jitcode_*` build: jtransform.py /
                                        // `make_jitcodes()` builds the portal jitcode
                                        // even when an individual opcode lowers to a
                                        // residual the tracer can't follow — that opcode
                                        // aborts the trace when hit, it never disables
                                        // the JIT for every other opcode.
                                        let __arm_jc: Option<majit_metainterp::JitCode> =
                                            (|| -> Option<majit_metainterp::JitCode> {
                                                let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                                                __sub_builder.set_name(#sub_jitcode_name);
                                                __sub_builder.ensure_i_regs(#min_i_regs);
                                                __sub_builder.ensure_r_regs(#min_r_regs);
                                                __sub_builder.ensure_f_regs(#min_f_regs);
                                                let _live_offset_patch = __sub_builder.live_placeholder();
                                                {
                                                    let __builder = &mut __sub_builder;
                                                    #body
                                                }
                                                __sub_builder.finalize_liveness(__asm);
                                                Some(__sub_builder.finish())
                                            })();
                                        match __arm_jc {
                                            Some(__jc) => __jc,
                                            None => {
                                                // Abort stub with the arm's register
                                                // shape so the paired BC_INLINE_CALL's
                                                // arg copies stay in bounds before the
                                                // BC_ABORT.
                                                #record_degraded
                                                let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                                                __sub_builder.set_name(#sub_jitcode_name);
                                                __sub_builder.ensure_i_regs(#min_i_regs);
                                                __sub_builder.ensure_r_regs(#min_r_regs);
                                                __sub_builder.ensure_f_regs(#min_f_regs);
                                                __sub_builder.abort();
                                                __sub_builder.finish()
                                            }
                                        }
                                    },
                                    inline_call_emit,
                                )
                            }
                            // The lowering already knows WHICH statement stopped it;
                            // carry that reason to the install-time record instead
                            // of collapsing every cause into one string, the same
                            // way `ArmPattern::Unsupported(reason)` is carried below.
                            Err(reason) => {
                                let record_degraded = record_degraded_arm(&reason);
                                (
                                    quote::quote! {
                                        {
                                            #record_degraded
                                            let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                                            __sub_builder.set_name(#sub_jitcode_name);
                                            __sub_builder.abort();
                                            __sub_builder.finish()
                                        }
                                    },
                                    dispatch_arm_inline_call_tokens(&[], &[]),
                                )
                            }
                        }
                    };
                    // The sub-JitCode is built in a statement of its own: its
                    // body spells the callee's registers, which coloring this
                    // JitCode must not touch.
                    lowerer.emit_aux(quote::quote! {
                        let __sub_jitcode = { #arm_body_tokens };
                        let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                    });
                    lowerer.emit_op(
                        OpMeta::linear(OpKind::InlineCall, arm_inline_call_reads, vec![]),
                        arm_inline_call_emit,
                    );
                    // handle_regular_call — trailing -live- after inline_call_*.
                    lowerer.emit_post_inline_live_marker(
                        quote::quote! { let _ = __builder.live_placeholder(); },
                    );
                }
            }
        }
        // jtransform.py `handle_jit_marker__loop_header`:
        // RPython lowers `can_enter_jit()` at the user's source-code
        // back-edge (interp_jit.py `pypyjitdriver.can_enter_jit(...)`
        // inside `jump_absolute`'s BACKWARD branch only — `interp_jit.py
        // if jumpto >= next_instr: return jumpto` early-out skips the
        // forward path) into a `loop_header(jd.index)` op AT the same
        // source position.  Pyre's `can_enter_jit!()` recognition lives
        // in `Lowerer::lower_stmt` (`Stmt::Macro` arm) which emits the
        // `LoopHeader` IR + `__builder.loop_header(__jdindex)` at that
        // exact stmt position INSIDE the arm body sub-JitCode — so the
        // LH op only executes when the user's source-level conditional
        // (`if backward { can_enter_jit!(); ... }`) is taken at runtime.
        // No post-INLINE_CALL emission here: doing so would over-emit
        // on every arm execution including forward-progress arms (per
        // codex strict-parity audit — arm-level existence ≠ conditional
        // call-site).
        //
        // interp_jit.py — loop back-edge: after each matched arm,
        // jump back to jit_merge_point so the next iteration re-enters
        // the dispatch loop at the portal merge point.  The GOTO is
        // required for control-flow correctness regardless of whether
        // the arm body emitted any LH inside its sub-JitCode.
        // `break` exits the source dispatch `while`, so its empty arm must
        // flow to the function's typed return. Re-entering the loop here
        // would execute the loop header with the post-HALT pc instead.
        //
        // An arm that lowered its own `return` already emitted the terminator
        // that ends the walk, so it gets NEITHER edge.  Note this is not the
        // `Halt` treatment: `break` diverts to `default_label` because its arm
        // has no operand and must reach the function-final typed return to get
        // one, whereas a returning arm carries its own value and would have the
        // function's trailing expression lowered over it.
        if matches!(inline_outcome, InlineArmOutcome::InlinedTerminal)
            || matches!(
                arm.pattern,
                crate::jit_interp::classify::ArmPattern::AbortPermanent
                    | crate::jit_interp::classify::ArmPattern::Unsupported(_)
            )
        {
            // No jump: the return terminator or an in-frame abort is the
            // arm's control transfer.
        } else if matches!(arm.pattern, crate::jit_interp::classify::ArmPattern::Halt) {
            lowerer.emit_jump(&default_label);
        } else {
            lowerer.emit_jump(loop_start_label);
        }

        // Bind the skip label at the end of this arm's guard sequence.
        // Jumping here means "this arm did not match; proceed to next arm".
        if let Some(skip_label) = skip_label.as_ref() {
            lowerer.emit_label_def(skip_label);
        }
    }

    // Every arm has lowered, so the consulted-key set is final: report the
    // declared `int_fields` / `ref_fields` keys no access site asked about.
    //
    // Reported rather than rejected, and the reason is the degraded arm. An
    // arm that refused to lower never reached its field accesses, so a key
    // used only there is unconsulted through no fault of the declaration —
    // failing the build on it would turn one refusal into two. The consumer's
    // gate reads this alongside `degraded_dispatch_arms` and can tell them
    // apart; a compile error could not.
    {
        let interp = config.state_type_name.clone();
        let consulted = config.consulted_field_keys.borrow();
        let mut unconsulted: Vec<&String> = config
            .int_fields
            .keys()
            .chain(config.ref_fields.keys())
            .filter(|key| !consulted.contains(*key))
            .collect();
        unconsulted.sort();
        unconsulted.dedup();
        for key in unconsulted {
            lowerer.emit_aux(quote::quote! {
                majit_metainterp::record_unconsulted_field_declaration(#interp, #key);
            });
        }
    }

    // After all arm guards, the default/exit path: unconditional GOTO.
    // default_label is bound at the typed-return emission site in
    // lower_dispatch_body (Task 1.7).  `Halt` arms keep jumping there
    // whatever the default arm says: a `break` arm carries no operand and
    // reaches the function's typed return for one.
    if !config.switch_dispatch && !lower_default_as_arm {
        lowerer.emit_jump(&default_edge);
    }
    default_label
}

fn pat_contains_range(pat: &Pat) -> bool {
    match pat {
        Pat::Range(_) => true,
        Pat::Or(pat_or) => pat_or.cases.iter().any(pat_contains_range),
        _ => false,
    }
}

/// A.3.5 — emit a `-live-` + `<kind>_guard_value` pair for each declared green.
///
/// Mirrors `jtransform.py promote_greens`: for every green Variable
/// (constants are already promoted and skipped at the RPython level; pyre
/// has no compile-time green constants so every entry is promoted), emit a
/// `-live-` marker followed by `<kind>_guard_value(reg)`.  The guard forces
/// the runtime value to a constant before `BC_JIT_MERGE_POINT`, satisfying
/// `pyjitpl.py verify_green_args` which expects all greens to be constants at the merge
/// point.
///
/// Must be called after the portal-input bindings are installed (so that
/// `lowerer.bindings` maps green idents → `Binding`) and BEFORE the
/// `jit_merge_point` emit.
pub(super) fn emit_promote_greens(lowerer: &mut Lowerer, config: &LowererConfig) {
    for green in &config.greens {
        let ident = match green {
            syn::Expr::Path(p) => p.path.get_ident().unwrap_or_else(|| {
                panic!(
                    "A.3.5 (`jtransform.py` `promote_greens`): green expression must be a single-segment \
                     ident for promote_greens. Got: {:?}",
                    p.path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                )
            }),
            _ => panic!(
                "A.3.5 (`jtransform.py` `promote_greens`): green expression must be a single-segment \
                 ident for promote_greens. Got non-path expression: {}",
                quote::quote!(#green)
            ),
        };
        let ident_name = ident.to_string();
        let binding = lowerer.bindings.get(&ident_name).unwrap_or_else(|| {
            panic!(
                "A.3.5 (`jtransform.py` `promote_greens`): green '{}' declared in #[jit_interp(greens = ...)] \
                 but not bound at portal entry. Available bindings: {:?}",
                ident_name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        let reg = binding.reg;
        let kind = binding.kind;
        // The `-live-` before each guard_value is what lets the codewriter's
        // per-marker liveness analysis record the alive set at the guard; the
        // pair is shared with `handle_recursive_call`'s promote_greens.
        lowerer.emit_live_and_guard_value(kind, reg);
    }
}

/// A.3.2 — resolve green variable names to register-byte lists.
///
/// Mirrors `jtransform.py make_three_lists(op.args[2:2+num_green_args])`:
/// each green expression is expected to be a single-segment ident.  Dotted
/// paths (e.g. `state.pc`) are explicitly out of scope — task A.7.
///
/// Returns `(greens_i, greens_r, greens_f)` matching the
/// `jit_merge_point(..., greens_i, greens_r, greens_f, ...)` bucket order.
pub(super) fn resolve_greens(
    lowerer: &Lowerer,
    config: &LowererConfig,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut greens_i: Vec<u8> = Vec::new();
    let mut greens_r: Vec<u8> = Vec::new();
    let mut greens_f: Vec<u8> = Vec::new();

    for green in &config.greens {
        let ident = match green {
            syn::Expr::Path(p) => p.path.get_ident().unwrap_or_else(|| {
                panic!(
                    "A.3.2 (jtransform.py:1700): green expression must be a single-segment \
                     ident; dotted greens (state.pc) are scoped to follow-up task A.7. \
                     Got: {:?}",
                    p.path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                )
            }),
            _ => panic!(
                "A.3.2 (jtransform.py:1700): green expression must be a single-segment \
                 ident; dotted greens (state.pc) are scoped to follow-up task A.7. \
                 Got non-path expression: {}",
                quote::quote!(#green)
            ),
        };
        let ident_name = ident.to_string();
        let binding = lowerer.bindings.get(&ident_name).unwrap_or_else(|| {
            panic!(
                "A.3.2 (jtransform.py:1700): green '{}' declared in #[jit_interp(greens = ...)] \
                 but not bound at portal entry. Available bindings: {:?}",
                ident_name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        // `Binding.reg: u16` but `assembler.py:225` per-bank bitset addressing
        // is u8-bounded.  `Register::new()` asserts this on construction; the
        // `Binding`-shaped path does not, so we re-check at the boundary
        // before encoding the register byte into the jit_merge_point payload.
        let reg_byte = u8::try_from(binding.reg).unwrap_or_else(|_| {
            panic!(
                "A.3.2 (assembler.py:225): green register index {} for ident '{}' exceeds u8 \
                 encoding limit; jit_merge_point list-byte encoding is u8-bounded",
                binding.reg, ident_name,
            )
        });
        match binding.kind {
            BindingKind::Int => greens_i.push(reg_byte),
            BindingKind::Ref => greens_r.push(reg_byte),
            BindingKind::Float => greens_f.push(reg_byte),
        }
    }

    // Validate uniqueness within each bucket (jtransform.py:1701).
    // `make_three_lists` is invoked twice in jtransform — greens at :1693 and
    // reds at :1697 — and the `dict.fromkeys` assert runs over each return
    // value. `resolve_reds` mirrors this; backport the same check here.
    for (label, bucket) in [
        ("greens_i", &greens_i),
        ("greens_r", &greens_r),
        ("greens_f", &greens_f),
    ] {
        let mut seen = HashSet::new();
        for &b in bucket.iter() {
            if !seen.insert(b) {
                panic!(
                    "A.3.2 (jtransform.py:1701): duplicate register {} in {}",
                    b, label
                );
            }
        }
    }

    (greens_i, greens_r, greens_f)
}

/// A.3.3 — resolve red variable names to register-byte lists.
///
/// Mirrors `jtransform.py make_three_lists(op.args[2+num_green_args:])`:
/// pyre's portal inputs are `program` (Ref/r0), `pc` (Int/i0), and optionally
/// `vable_var` (Ref/r1).  The reds = portal-inputs minus declared greens.
///
/// TODO: `interp_jit.py reds = ['frame', 'ec']` uses
/// PyPy's frame+ec pair; pyre uses `[program, pc]` (minus greens) as its
/// minimal reds set.  Consumers that want the PyPy parity declaration can set
/// `greens = [pc, program]`, leaving reds = [], which is the intended A.6
/// follow-up mapping.
///
/// Returns `(reds_i, reds_r, reds_f)` matching the
/// `jit_merge_point(..., reds_i, reds_r, reds_f)` bucket order.
pub(super) fn resolve_reds(
    lowerer: &Lowerer,
    config: &LowererConfig,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut reds_i: Vec<u8> = Vec::new();
    let mut reds_r: Vec<u8> = Vec::new();
    let mut reds_f: Vec<u8> = Vec::new();

    // Slice (audit Issue #6) — when the consumer declares
    // `#[jit_interp(reds = [...])]` explicitly, use that list as the
    // canonical reds source matching RPython
    // `jtransform.py make_three_lists(op.args[2+num_green_args:])`
    // — the marker's tail args are the reds.  Pyre's marker is
    // stateless (no tail args), so the `reds` config slot replaces
    // them.  When `config.reds` is empty, fall back to the legacy
    // candidate list `[program, pc(+ optional vable)]` minus declared
    // greens (the pre-Issue-#6 pyre default).
    let explicit_red_names: Vec<String> = config
        .reds
        .iter()
        .filter_map(|expr| match expr {
            syn::Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
            _ => panic!(
                "Issue #6: red expression must be a single-segment ident; \
                 got non-path: {}",
                quote::quote!(#expr),
            ),
        })
        .collect();

    let owned_red_names: Vec<String>;
    let reds_names: Vec<&str> = if !explicit_red_names.is_empty() {
        // `promote_greens` only walks `jitdriver.greens`. A name that is
        // also listed in `reds` would be both guarded and passed as a
        // red — `handle_jit_marker__jit_merge_point` keeps the two lists
        // disjoint (`args[2:2+num_green]` vs the tail).
        let green_names: HashSet<String> = config
            .greens
            .iter()
            .filter_map(|expr| match expr {
                syn::Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
                _ => None,
            })
            .collect();
        owned_red_names = explicit_red_names
            .into_iter()
            .filter(|name| !green_names.contains(name))
            .collect();
        owned_red_names.iter().map(|s| s.as_str()).collect()
    } else {
        // Issue 2.2 (support.py:121 _kind2count = {'int':1,'ref':2,'float':3}):
        // `pc` (Int) precedes `program` (Ref) so the int→ref→float bucket
        // emit below preserves the canonical sort order.  Bucketing
        // sorts by kind regardless, but lining up the candidate list
        // with the bucket order keeps schema (`red_schema`) and payload
        // byte-for-byte aligned.
        let mut candidates: Vec<&str> = vec!["pc", "program"];
        if let Some(vable) = config.vable_var.as_ref() {
            candidates.push(vable.as_str());
        }
        let green_names: HashSet<String> = config
            .greens
            .iter()
            .filter_map(|expr| match expr {
                syn::Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
                _ => None,
            })
            .collect();
        candidates
            .into_iter()
            .filter(|name| !green_names.contains(*name))
            .collect()
    };

    for name in reds_names {
        let binding = lowerer.bindings.get(name).unwrap_or_else(|| {
            panic!(
                "A.3.3 (jtransform.py:1700): red '{}' is a portal-input name but has no \
                 binding at jit_merge_point emit. Available bindings: {:?}",
                name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        let reg_byte = u8::try_from(binding.reg).unwrap_or_else(|_| {
            panic!(
                "A.3.3 (assembler.py:225): red register index {} for ident '{}' exceeds u8 \
                 encoding limit; jit_merge_point list-byte encoding is u8-bounded",
                binding.reg, name,
            )
        });
        match binding.kind {
            BindingKind::Int => reds_i.push(reg_byte),
            BindingKind::Ref => reds_r.push(reg_byte),
            BindingKind::Float => reds_f.push(reg_byte),
        }
    }

    // `handle_jit_marker__jit_merge_point` / `make_three_lists`: every
    // declared red is a live Variable on the marker. State-field identity
    // slots (plain scalars and ptr then len per `[int]`) are those reds.
    for reg in config.identity_slot_registers() {
        match reg.kind {
            BindingKind::Int => reds_i.push(reg.index),
            BindingKind::Ref => reds_r.push(reg.index),
            BindingKind::Float => reds_f.push(reg.index),
        }
    }

    // Validate uniqueness within each bucket (jtransform.py:1701).
    for (label, bucket) in [
        ("reds_i", &reds_i),
        ("reds_r", &reds_r),
        ("reds_f", &reds_f),
    ] {
        let mut seen = HashSet::new();
        for &b in bucket.iter() {
            if !seen.insert(b) {
                panic!(
                    "A.3.3 (jtransform.py:1701): duplicate register {} in {}",
                    b, label
                );
            }
        }
    }

    (reds_i, reds_r, reds_f)
}

/// Slice (audit Issue #5) — extract `(name, green_type_token)` pairs
/// for the declared greens.  `BindingKind` maps to canonical
/// `majit_ir::GreenType::{Int, Ref, Float}`; an explicit `: str` /
/// `: unicode` tag in `config.green_type_tags` overrides the binding's
/// IR type with the upstream `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
/// distinction (`warmspot.py:663 _green_args_spec`).  Used at the
/// install path to populate `JitDriverStaticData::vars` via
/// `JitDriver::declare_schema_typed` so `green_args_spec` reports the
/// real lltype subtype instead of collapsing STR/UNICODE to `Ref`.
///
/// Output preserves declaration order.  RPython
/// `decode_hp_hint_args` (support.py) does not silently reorder
/// the JitDriver declaration; it computes `sort_vars(lst)` and asserts
/// `lst == lst2`, telling the user to reorder the greens/reds if needed.
/// Pyre mirrors that shape here: validate `int → ref → float` order, then
/// return the schema unchanged.  The bytecode payload encoder
/// (`resolve_greens`) still emits the RPython `make_three_lists` bucket
/// shape independently.
pub(super) fn green_schema(
    lowerer: &Lowerer,
    config: &LowererConfig,
) -> Vec<(String, TokenStream)> {
    use crate::jit_interp::green_type_tag::GreenTypeTag;
    let mut out: Vec<(u8, String, TokenStream)> = Vec::new();
    for (i, green) in config.greens.iter().enumerate() {
        // RPython `support.py decode_hp_hint_args` strictly
        // validates greens/reds count + ordering — it never silently
        // drops a malformed marker arg.  Pyre mirrors that strength
        // here: bare-ident is the only supported form (matching
        // `JitDriver(..., greens=['name', ...])` on the upstream
        // side); anything else is a structural mismatch the install
        // path could not surface as count/payload divergence
        // downstream.  Earlier `continue` quietly shrank the schema
        // and let the macro emit a payload that disagreed with
        // `JitDriverStaticData::vars`.
        let ident_name = match green {
            syn::Expr::Path(p) => match p.path.get_ident() {
                Some(i) => i.to_string(),
                None => panic!(
                    "#[jit_interp] greens[{i}]: only bare-ident greens are \
                     supported (matching `JitDriver(greens=['name'])` upstream); \
                     got a multi-segment path: {}",
                    quote::quote!(#green),
                ),
            },
            _ => panic!(
                "#[jit_interp] greens[{i}]: only bare-ident greens are \
                 supported (matching `JitDriver(greens=['name'])` upstream); \
                 got a non-path expression: {}",
                quote::quote!(#green),
            ),
        };
        let Some(binding) = lowerer.bindings.get(&ident_name) else {
            panic!(
                "#[jit_interp] greens[{i}]: unknown identifier `{ident_name}` — \
                 not a state field bound by `state_fields!` and not a JitDriver \
                 green declared in scope",
            );
        };
        let tag = config.green_type_tags.get(i).copied().flatten();
        // Tag wins over binding (warmspot.py:663 — _green_args_spec
        // reads the lltype directly from the JitDriver signature, not
        // from the codewriter's IR-collapsed view).
        let (kind_rank, gt_tok) = match tag {
            Some(GreenTypeTag::Int) => (1u8, quote::quote!(majit_ir::GreenType::Int)),
            Some(GreenTypeTag::Float) => (3u8, quote::quote!(majit_ir::GreenType::Float)),
            Some(GreenTypeTag::Ref) => (2u8, quote::quote!(majit_ir::GreenType::Ref)),
            Some(GreenTypeTag::Str) => (2u8, quote::quote!(majit_ir::GreenType::Str)),
            Some(GreenTypeTag::Unicode) => (2u8, quote::quote!(majit_ir::GreenType::Unicode)),
            None => match binding.kind {
                BindingKind::Int => (1u8, quote::quote!(majit_ir::GreenType::Int)),
                BindingKind::Ref => (2u8, quote::quote!(majit_ir::GreenType::Ref)),
                BindingKind::Float => (3u8, quote::quote!(majit_ir::GreenType::Float)),
            },
        };
        out.push((kind_rank, ident_name, gt_tok));
    }
    assert_kind_sorted("greens", &out);
    out.into_iter().map(|(_, n, t)| (n, t)).collect()
}

/// Companion to [`green_schema`] for the dispatch path's red layout.
/// Mirrors [`resolve_reds`] — when `config.reds` is non-empty, use that
/// explicit list (Issue #6 RPython parity); otherwise default to
/// `[pc, program(+ optional vable)]` minus declared greens.
///
/// Output preserves declaration order and asserts the same
/// `int → ref → float` invariant as RPython `decode_hp_hint_args`
/// (support.py:135-150).  The default red candidate list begins with
/// `pc` (Int) before `program` (Ref), so the implicit path is already
/// in RPython-accepted order.
pub(super) fn red_schema(lowerer: &Lowerer, config: &LowererConfig) -> Vec<(String, TokenStream)> {
    // RPython `support.py decode_hp_hint_args` parity: malformed
    // marker args panic instead of silently shrinking the schema (see
    // `green_schema` for the full rationale).
    let explicit: Vec<String> = config
        .reds
        .iter()
        .enumerate()
        .map(|(i, expr)| match expr {
            syn::Expr::Path(p) => match p.path.get_ident() {
                Some(id) => id.to_string(),
                None => panic!(
                    "#[jit_interp] reds[{i}]: only bare-ident reds are \
                     supported (matching `JitDriver(reds=['name'])` upstream); \
                     got a multi-segment path: {}",
                    quote::quote!(#expr),
                ),
            },
            _ => panic!(
                "#[jit_interp] reds[{i}]: only bare-ident reds are \
                 supported (matching `JitDriver(reds=['name'])` upstream); \
                 got a non-path expression: {}",
                quote::quote!(#expr),
            ),
        })
        .collect();
    let owned: Vec<String>;
    let explicit_was_provided = !explicit.is_empty();
    let names: Vec<&str> = if explicit_was_provided {
        owned = explicit;
        owned.iter().map(|s| s.as_str()).collect()
    } else {
        // Issue 2.2 (support.py:121 _kind2count): `pc` (Int=1) precedes
        // `program` (Ref=2).  Earlier shape `["program", "pc"]`
        // produced Ref-before-Int order; RPython would reject that in
        // `decode_hp_hint_args` instead of reordering it.
        let mut candidates: Vec<&str> = vec!["pc", "program"];
        if let Some(vable) = config.vable_var.as_ref() {
            candidates.push(vable.as_str());
        }
        // greens have already been validated at `green_schema` (which
        // runs before `red_schema` in the lowering pipeline), so any
        // non-bare-ident at this point is a programmer error worth
        // surfacing — but in `red_schema` the intent is just to filter
        // them out of the default candidate set.  Mirror the green
        // validation strictness instead of accepting silent drops:
        // assert each green entry is a bare ident.
        let green_names: HashSet<String> = config
            .greens
            .iter()
            .enumerate()
            .map(|(i, expr)| match expr {
                syn::Expr::Path(p) => match p.path.get_ident() {
                    Some(id) => id.to_string(),
                    None => panic!(
                        "#[jit_interp] greens[{i}] (re-validated in red_schema): \
                         only bare-ident greens are supported; got a \
                         multi-segment path: {}",
                        quote::quote!(#expr),
                    ),
                },
                _ => panic!(
                    "#[jit_interp] greens[{i}] (re-validated in red_schema): \
                     only bare-ident greens are supported; got a non-path \
                     expression: {}",
                    quote::quote!(#expr),
                ),
            })
            .collect();
        candidates
            .into_iter()
            .filter(|name| !green_names.contains(*name))
            .collect()
    };
    let mut out: Vec<(u8, String, TokenStream)> = Vec::new();
    for name in names {
        match lowerer.bindings.get(name) {
            Some(binding) => {
                let (rank, ty_tok) = match binding.kind {
                    BindingKind::Int => (1u8, quote::quote!(majit_ir::Type::Int)),
                    BindingKind::Ref => (2u8, quote::quote!(majit_ir::Type::Ref)),
                    BindingKind::Float => (3u8, quote::quote!(majit_ir::Type::Float)),
                };
                out.push((rank, name.to_string(), ty_tok));
            }
            None => {
                if explicit_was_provided {
                    // RPython `support.py decode_hp_hint_args`:
                    // every declared red name must appear in the
                    // function's local bindings; an unknown name is a
                    // declaration-vs-body mismatch that upstream
                    // surfaces as `KeyError` from
                    // `decode_hp_hint_args`'s `varlist[i]` lookup.
                    // Silent drop would mask the mismatch and let a
                    // misshaped `BC_JIT_MERGE_POINT` payload propagate
                    // into the dispatch JitCode.
                    panic!(
                        "#[jit_interp] reds: declared red `{name}` is not \
                         bound in the function body. Either remove it from \
                         the explicit `reds = [...]` list or introduce a \
                         matching `let {name}` binding (support.py:135-150 \
                         decode_hp_hint_args parity).",
                    );
                }
                // Implicit-default branch: the candidate list above is
                // speculative (`pc` / `program` / vable_var); only the
                // intersection with the actual bindings becomes the red
                // schema, matching pyre's per-#[jit_interp] minimal red
                // shape (some sites use `program` only, others add a
                // virtualizable).
            }
        }
    }
    assert_kind_sorted("reds", &out);
    // Identity slots are listed on the `jit_merge_point` marker
    // (`resolve_reds` / `handle_jit_marker__jit_merge_point`) and in
    // `-live-` (`identity_slot_registers`). They are not JitDriver reds:
    // `warmspot.py` `jd.portal_calldescr = cpu.calldescrof(portal_runner)`
    // describes the original function, whose arguments are the named
    // greens/reds. State-field identity lives in those reds the way
    // frame fields live in the portal frame, not as extra runner args.
    out.into_iter().map(|(_, n, t)| (n, t)).collect()
}

fn assert_kind_sorted(label: &str, vars: &[(u8, String, TokenStream)]) {
    for pair in vars.windows(2) {
        let (prev_rank, prev_name, _) = &pair[0];
        let (rank, name, _) = &pair[1];
        if prev_rank > rank {
            panic!(
                "support.py:135-150 decode_hp_hint_args parity: JitDriver {} \
                 must be declared in int -> ref -> float order. '{}' appears \
                 before '{}', but rank {} > {}. Reorder the variables instead \
                 of relying on pyre to sort them.",
                label, prev_name, name, prev_rank, rank
            );
        }
    }
}

/// Portal env binding (`program`).
///
/// `struct_type` is the `env = ...` type so `program.field` lowers through
/// `lower_ref_binding_getfield`. Other portal bindings stay untyped.
fn portal_env_binding(env_type_name: &str) -> Binding {
    Binding {
        reg: 0,
        kind: BindingKind::Ref,
        depends_on_stack: false,
        struct_type: syn::parse_str(env_type_name).ok(),
    }
}

/// Dispatch JitCode body lowerer.
///
/// Lowers a `#[jit_interp]` function's `while { jit_merge_point!(); ...
/// match opcode { ... } }` dispatch loop into a single dispatch JitCode
/// body. Mirrors RPython `pypy/module/pypyjit/interp_jit.py`
/// portal + `pypy/interpreter/pyopcode.py` dispatch_bytecode.
///
/// Output IR shape:
/// 1. `BC_LIVE` (canonical entry)
/// 2. `BC_JIT_MERGE_POINT(_C)` (interp_jit.py jit_merge_point hook)
/// 3. `BC_LOOP_HEADER`
/// 4. pre-dispatch ops, source-order (interp_jit.py)
/// 5. opcode/oparg fetch + pc advance (pyopcode.py)
/// 6. dispatch chain via existing `BC_GOTO_IF_NOT_*` ops
///    (jtransform.py optimize_goto_if_not conditional fusion)
/// 7. per-arm `BC_INLINE_CALL sub_jitcode_idx` (jtransform.py handle_regular_call)
/// 8. loop close `BC_GOTO 0` — Task 1.7
/// 9. default arm: typed return / dispatch-exit ABI — Task 1.7
pub(crate) fn lower_dispatch_body(
    config: &LowererConfig,
    func_block: &syn::Block,
    classified_arms: &[crate::jit_interp::classify::ClassifiedArm],
    portal_output: &syn::ReturnType,
) -> Option<GeneratedJitCodeBody> {
    let mut lowerer = Lowerer::new(Some(config));
    // RPython `assembler` emits exactly one `-live-` per source point;
    // the dispatch JitCode's leading-dummy `BC_LIVE` is already emitted
    // by `codegen_trace.rs`'s `__dispatch_jitcode_<fn>` wrapper as
    // `let _live_offset_patch = __builder.live_placeholder();`
    // (matching every per-arm sub-JitCode's single leading dummy).
    // Earlier pyre revisions emitted a SECOND entry placeholder here,
    // landing two consecutive `BC_LIVE` markers at the dispatch
    // JitCode's start — divergent from main's per-arm shape.  The
    // duplicate is retired; subsequent ops below are emitted in order.

    // Loop back-edge target (interp_jit.py:95-100): bound here so that the
    // back-edge GOTOs at the end of each matched arm land at the
    // jit_merge_point, re-entering the portal on the next iteration.
    let loop_start_label = lowerer.alloc_label();
    lowerer.emit_aux(quote::quote! { let #loop_start_label = __builder.new_label(); });
    lowerer.emit_label_def(&loop_start_label);
    lowerer.dispatch_loop_label = Some(loop_start_label.clone());

    // Register the portal-input bindings at proc-macro time before
    // resolve_greens (below) consults the binding map.  These are
    // pure proc-macro-time HashMap inserts — no runtime code is emitted
    // here; the corresponding __builder.ensure_*_regs calls that allocate
    // the register slots at runtime appear later (after loop_header).
    //
    // r0 = program (Ref). We install the binding but do NOT advance
    // next_reg (the Int-bank counter) because r0 lives in the Ref bank.
    // The struct type is the env type: `program.field` then lowers through
    // `lower_ref_binding_getfield`, which reads `<T>::__MAJIT_IMMUTABLE_FIELDS`
    // (`rclass.py _parse_field_list`).
    lowerer.bindings.insert(
        "program".to_owned(),
        portal_env_binding(&config.env_type_name),
    );
    // i0 = pc (Int). Advance next_reg past i0 so opcode_reg gets i1.
    lowerer.bindings.insert(
        "pc".to_owned(),
        Binding {
            reg: 0,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        },
    );
    // RPython `MIFrame.setup_call(original_boxes)` distributes every portal
    // argument by kind.  Function-parameter greens are therefore inputs too,
    // not values recovered later from a red state object.  Keep independent
    // per-bank cursors so their register numbers match the argument seeding in
    // `codegen_trace`.
    let mut next_input_i = 1u16;
    let mut next_input_r = 1u16;
    let mut next_input_f = 0u16;
    // Portal-parameter greens, then loop-carried greens. The per-bank
    // cursors match the argument pushes in `codegen_trace` (`program`,
    // `pc`, portal greens, carried greens, virtualizable).
    for (name, kind) in config
        .portal_green_params
        .iter()
        .chain(config.loop_carried_greens.iter())
    {
        let (reg, binding_kind) = match kind {
            ValueKind::Int => {
                let reg = next_input_i;
                next_input_i += 1;
                (reg, BindingKind::Int)
            }
            ValueKind::Ref => {
                let reg = next_input_r;
                next_input_r += 1;
                (reg, BindingKind::Ref)
            }
            ValueKind::Float => {
                let reg = next_input_f;
                next_input_f += 1;
                (reg, BindingKind::Float)
            }
        };
        lowerer.bindings.insert(
            name.clone(),
            Binding {
                reg,
                kind: binding_kind,
                depends_on_stack: false,
                struct_type: None,
            },
        );
    }
    // State-field scalars occupy reserved identity-slot prefixes:
    // int scalars at `int_regs[base..base+num_scalars)` where base =
    // `int_identity_base()` skips the dispatch JitCode's int inputs
    // (`pc` at i0, then portal and loop-carried int greens), and ref scalars at
    // `ref_regs[base..base+num_ref_scalars)` where base =
    // `ref_identity_base()` skips the dispatch JitCode's ref-bank arguments
    // (`program` at r0, vable identity at r1 when present), seeded at walk
    // start by `MIFrame.setup_call`.
    // `alloc_reg` draws BOTH int and ref working registers from the single
    // `next_reg` counter, so floor it above the larger prefix: a working
    // register that aliased an identity slot would clobber a red
    // `load_state_field*` / `store_state_field*` addresses. With no scalars
    // `int_identity_end` is just the base, which keeps pc's i0 reserved.
    // The vable identity is a Ref, not an extra int slot.
    let (int_identity_end, ref_identity_end) = config.split_identity_reg_ends();
    // Float scalars seed their own reserved prefix at
    // `float_regs[float_identity_base()..float_identity_end())`, restored by
    // the guard-failure resume seeder just like the int/ref banks. `alloc_reg`
    // also draws float working registers from `next_reg`, so a float-heavy
    // state (whose float_identity_end exceeds the int/ref ends) would otherwise
    // hand out a working register aliasing a float identity slot and corrupt a
    // sibling scalar (`{ a: float, b: float }`: a read of `a` could land on
    // `b`'s slot).
    lowerer.next_reg = lowerer
        .next_reg
        .max(next_input_i)
        .max(next_input_r)
        .max(next_input_f)
        .max(int_identity_end)
        .max(ref_identity_end)
        .max(config.float_identity_end());
    let pinned_floor = super::regalloc::RegisterCounts {
        ints: lowerer.next_reg,
        refs: lowerer.next_reg,
        floats: lowerer.next_reg,
    };

    // A.3.6.1 (`jtransform.py` `promote_greens`): bind body-local `let` stmts that
    // appear BEFORE `jit_merge_point!()` in the dispatch while-body, so
    // that consumer-declared `greens = [<body-local>]` (say
    // `greens = [ok]`) resolve via `lowerer.bindings` when
    // `emit_promote_greens` and `resolve_greens` consult it below.
    bind_pre_merge_point_stmts(&mut lowerer, func_block)?;
    if lowerer.dispatch_tainted_reason.is_some() {
        return None;
    }

    // Loop-local greens keep the header register. An inlined arm's
    // reassignment moves into it (`lower_local_reassign` when `join_merge`
    // names that register and `pc_pinned` is set). `emit_promote_greens`
    // then `guard_value`s the same register, so the guard snapshot carries
    // the value. Dispatch i0 is the portal pc (the r0/i0 entry convention)
    // and stays on the pinned-write path. Prefix locals that are not
    // greens are left out.
    for green in &config.greens {
        let syn::Expr::Path(path) = green else {
            continue;
        };
        let Some(ident) = path.path.get_ident() else {
            continue;
        };
        let name = ident.to_string();
        let Some(binding) = lowerer.bindings.get(&name) else {
            continue;
        };
        if binding.kind == BindingKind::Int && binding.reg == 0 {
            continue;
        }
        let reg = binding.reg;
        lowerer.join_merge.insert(name, reg);
    }

    // A.3.5 (`jtransform.py` `promote_greens`): emit a `-live-` + `<kind>_guard_value`
    // pair for each declared green BEFORE `jit_merge_point`.  Forces every
    // green to a constant at trace time; `pyjitpl.py verify_green_args` asserts all greens
    // are constants when the merge point is reached.
    emit_promote_greens(&mut lowerer, config);

    // jtransform.py returns `[op3, op1, op2]` from
    // `handle_jit_marker__jit_merge_point`:
    //
    //     op1 = SpaceOperation('jit_merge_point', args, None)
    //     op2 = SpaceOperation('-live-', [], None)
    //     # ^^^ we need a -live- for the case of do_recursive_call()
    //     op3 = SpaceOperation('-live-', [], None)
    //     # and one for inlined short preambles
    //     return ops + [op3, op1, op2]
    //
    // i.e. `promote_greens` results, then a `-live-` (op3, used by
    // inlined short preambles), then the merge-point op (op1), then
    // another `-live-` (op2, used by recursive-call resume).  Pyre
    // emits the trailing `-live-` (op2) below; this PRE-merge-point
    // `-live-` (op3) was previously missing — restore it for parity.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote::quote! { __builder.live_placeholder(); },
    );

    // interp_jit.py — pypyjitdriver.jit_merge_point(...) at the
    // portal entry. A.3.2 fills greens; A.3.3 fills reds;
    // jdindex is the `__jdindex: i64` runtime parameter of
    // `__dispatch_jitcode_*` (jtransform.py:1704 portal_jd.index).
    //
    // resolve_greens requires the portal-input bindings to be installed
    // (done just above) so that green ident → register-byte lookup works.
    let (greens_i, greens_r, greens_f) = resolve_greens(&lowerer, config);
    let (reds_i, reds_r, reds_f) = resolve_reds(&lowerer, config);
    let greens_i_lit: Vec<_> = greens_i.iter().map(|b| quote::quote!(#b)).collect();
    let greens_r_lit: Vec<_> = greens_r.iter().map(|b| quote::quote!(#b)).collect();
    let greens_f_lit: Vec<_> = greens_f.iter().map(|b| quote::quote!(#b)).collect();
    let reds_i_lit: Vec<_> = reds_i.iter().map(|b| quote::quote!(#b)).collect();
    let reds_r_lit: Vec<_> = reds_r.iter().map(|b| quote::quote!(#b)).collect();
    let reds_f_lit: Vec<_> = reds_f.iter().map(|b| quote::quote!(#b)).collect();
    let mut merge_reads: Vec<Register> = Vec::new();
    merge_reads.extend(greens_i.iter().map(|&b| Register::int(u16::from(b))));
    merge_reads.extend(greens_r.iter().map(|&b| Register::ref_(u16::from(b))));
    merge_reads.extend(greens_f.iter().map(|&b| Register::float(u16::from(b))));
    merge_reads.extend(reds_i.iter().map(|&b| Register::int(u16::from(b))));
    merge_reads.extend(reds_r.iter().map(|&b| Register::ref_(u16::from(b))));
    merge_reads.extend(reds_f.iter().map(|&b| Register::float(u16::from(b))));
    lowerer.emit_op(
        OpMeta::linear(OpKind::JitMergePoint, merge_reads, vec![]),
        quote::quote! {
            // __jdindex: jtransform.py:1704 portal_jd.index threaded as runtime param.
            // Identity reds include ptr then len per plain `[int]`
            // (`handle_jit_marker__jit_merge_point` / `make_three_lists`).
            __builder.jit_merge_point(
                __jdindex,
                &[#(#greens_i_lit),*],
                &[#(#greens_r_lit),*],
                &[#(#greens_f_lit),*],
                &[#(#reds_i_lit),*],
                &[#(#reds_r_lit),*],
                &[#(#reds_f_lit),*],
            );
        },
    );
    // jtransform.py:1707-1712 emits a trailing `-live-` after
    // `jit_merge_point`, used by recursive-call and short-preamble resume
    // paths.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote::quote! { __builder.live_placeholder(); },
    );

    // jtransform.py `handle_jit_marker__loop_header` emits the
    // `loop_header` op at the source-code `can_enter_jit()` call site
    // (interp_jit.py `pypyjitdriver.can_enter_jit(...)` inside
    // `jump_absolute`).  In the lowered bytecode this lands at each
    // back-edge — NOT immediately after `jit_merge_point` at the top of
    // the dispatch loop.  pyre's `lower_dispatch_chain` emits one
    // `loop_header(__jdindex)` op per matched arm, just before the
    // `goto loop_start_label` back-edge — see the per-arm emission
    // there.  Earlier pyre revisions emitted a single `loop_header` here
    // (right after `jit_merge_point`); that placement was structurally
    // backwards relative to RPython (which has loop_header at back-edges)
    // and is now retired.

    // Allocate input registers for the dispatch JitCode.
    //
    // The dispatch JitCode reads from two reds at its entry point:
    //   r0 — `program` (bytecode slice, Ref bank)
    //   i0 — `pc`      (program counter, Int bank)
    // Optional virtualizable input uses r1, because r0 is already the
    // bytecode object in the dispatch JitCode entry ABI.
    //
    // Mirrors interp_jit.py reds=['frame', 'ec'] for PyPy's portal;
    // pyre's per-#[jit_interp] dispatch uses (program, pc) as the minimal
    // reds.  The actual value seeding (binding i0/r0 to the outer Rust
    // `program`/`pc` at trace time) is the `__trace_*` rewrite.
    //
    // TODO: derive env_param_name / pc_var_name from LowererConfig or
    // macro config instead of hard-coding "program"/"pc".
    // Placeholder: actual register counts patched after dispatch chain
    // lowers all arms and we know the final next_reg.
    let ensure_regs_stmt_idx = lowerer.statements.len();
    lowerer.emit_aux(quote::quote! {
        __builder.ensure_r_regs(2u16);
        __builder.ensure_i_regs(1u16);
        __builder.ensure_f_regs(0u16);
    });

    // interp_jit.py:91-93: lower stmts that appear between jit_merge_point
    // and the dispatch match in source order. This covers promote calls
    // (jtransform.py:608-615: hint(x, promote=True) → -live- + guard_value),
    // opcode/oparg fetch (Task 1.4), and any other pre-dispatch stmts.
    //
    // Walk: find the dispatch match, then find the while body that contains it,
    // then iterate stmts before the match-containing stmt.
    //
    // Task 1.5: emit dispatch chain.
    // pyopcode.py+ if/elif chain over opcode value.
    // jtransform.py optimize_goto_if_not fuses int_eq + goto_if_not
    // into goto_if_not_int_eq/iiL (BC_GOTO_IF_NOT_INT_EQ).
    //
    // A matchless portal has no opcode chain: the post-merge-point
    // while-body is the iteration, closed by the while condition.
    let default_label = if classified_arms.is_empty() {
        let default_label = lowerer.alloc_label();
        lowerer.emit_aux(quote::quote! { let #default_label = __builder.new_label(); });
        lowerer.transactional(|inner| {
            lower_matchless_portal_body(inner, func_block, &loop_start_label)
        })?;
        default_label
    } else {
        // Pin pc-writes to i0 for the whole pre-dispatch walk. The branch-op
        // match that precedes the dispatch match (the branch opcodes) carries
        // `pc = program.get_label(pc - 1); ...; continue;` arms. Without the
        // pin, the assignment SSA-rebinds `pc` to a fresh register that dies
        // at the `continue` back-edge, so the next merge point reads the
        // STALE fall-through pc from i0 (pre-advanced `pc += 1`) — the
        // recorded green-pc channel diverges from concrete execution (an
        // unconditional JMP at the last program index records pc == len,
        // and `get_req_size(len)` is out of bounds). RPython has no rebind
        // hazard: `pc` is one Variable through jtransform, so every write
        // reaches the merge point by construction.
        lowerer.pc_pinned = true;
        let _lowered_pre_dispatch = lower_pre_dispatch_stmts(&mut lowerer, func_block);
        lowerer.pc_pinned = false;
        // A.2.3a fail-closed install gate: if pre-dispatch lowering detected
        // a structurally unrecognized inner construct (currently only the
        // `Expr::While` shape mismatch path), abort dispatch JitCode body
        // generation and return None. The caller (`codegen_trace.rs`'s
        // `generate_trace_fn`)
        // emits an empty body for the dispatch_jitcode_fn, so the runtime
        // gate in `generate_state_fields_jit_state` misses `BC_GETARRAYITEM_GC_I`
        // and refuses to register the singleton.
        if lowerer.dispatch_tainted_reason.is_some() {
            return None;
        }
        lower_dispatch_chain(&mut lowerer, classified_arms, config, &loop_start_label)
    };

    // Task 1.7: default arm typed return.
    // Bind default_label here so the dispatch chain's fall-through GOTO lands
    // at the typed-return emission (interp_jit.py return boundary).
    lowerer.emit_label_def(&default_label);

    // Locate the source dispatch loop and its post-loop epilogue.  A terminal
    // opcode (`break`) transfers control to this whole suffix, not merely to
    // its final expression.  Until the suffix is represented as JitCode, a
    // state-mutating statement there must fail closed: compiling just the
    // final read would skip the mutation on the machine-code path (TL's
    // `stackpos -= 1; stack[stackpos]` returned the pre-pop slot).
    let dispatch_loop_index = if let Some(dispatch_match) = find_dispatch_match(func_block) {
        func_block.stmts.iter().position(|stmt| {
            let Stmt::Expr(expr, _) = stmt else {
                return false;
            };
            match expr {
                Expr::While(w) => block_contains_match(&w.body, dispatch_match),
                Expr::Loop(l) => block_contains_match(&l.body, dispatch_match),
                _ => false,
            }
        })?
    } else {
        find_portal_loop(func_block)?.index
    };
    let post_loop = &func_block.stmts[dispatch_loop_index + 1..];
    // Positionally, not by searching backwards for the first expression-shaped
    // statement: a `while` / `if` / `match` / `loop` written in statement
    // position is also `Stmt::Expr(_, None)`, so a search would name one of
    // those as the tail, and `prefix_len` would then both re-lower it as
    // prefix and leave the real last statement unlowered.
    let return_expr = match post_loop.last() {
        Some(syn::Stmt::Expr(e, None)) => Some(e),
        _ => None,
    };
    // The terminal opcode's `break` reaches every statement after the source
    // loop before evaluating the tail expression.  Lower that epilogue in
    // source order.  `TraceAction::Finish` publishes its concrete result to
    // the single-pass merge hook, which returns it directly and therefore does
    // not execute this suffix a second time in native Rust.
    //
    // The epilogue fails open.  A statement the lowering refuses -- an I/O
    // loop, a green write -- rolls back to the loop exit and leaves the
    // binding unset, so the walk terminates in `void_return` and `Finish`
    // hands the whole suffix to native execution exactly once.  Propagating
    // the refusal instead would take the entire dispatch JitCode down with it,
    // and every trace of that interpreter becomes `AbortPermanent`.
    //
    // Prefix and tail roll back TOGETHER.  A tail the lowering refuses after a
    // committed prefix would leave the prefix in the JitCode and still
    // terminate in `void_return`, so the native suffix -- the prefix included
    // -- would run a second time once the `Finish` breaks the dispatch loop.
    // Lowering the tail alone when the prefix refuses would emit `int_return`
    // (`flatten.py GraphFlattener.make_return`) while dropping the prefix from
    // the JitCode; the Finish drain then returns without running the native
    // suffix, so prefix side effects vanish on the JIT path.  The native
    // suffix is the prefix's owner when it cannot be represented.  A
    // `void_return` Finish still has to write every live red back first
    // (`pyjitpl.py compile_done_with_this_frame` /
    // `virtualizable.py write_from_resume_data_partial`), or the suffix reads
    // the trace-start `state`.
    //
    // A portal whose declared return type has no finish projection lowers no
    // epilogue at all.  The two drains that make a lowered epilogue run once --
    // `FinishReturn::drain` after the back edge and
    // `FinishReturn::drain_single_pass` at the single-executor close -- are both
    // emitted only for a type `finish_return_for` recognises.  Without one they
    // are empty, so a published FINISH is consumed by nobody: the close breaks
    // the native loop and the source suffix runs a SECOND time, and the back
    // edge resumes at the merge point the compiled run already ran past.
    // Reading the same predicate the drains read is what keeps the two from
    // disagreeing about which types are projectable.
    let epilogue_has_finish_projection =
        crate::jit_interp::finish_return_for(portal_output).is_some();
    let prefix_len = post_loop
        .len()
        .saturating_sub(usize::from(return_expr.is_some()));
    let binding = epilogue_has_finish_projection
        .then(|| {
            lowerer.transactional(|inner| {
                for stmt in &post_loop[..prefix_len] {
                    inner.lower_stmt(stmt)?;
                }
                match return_expr {
                    Some(expr) => inner.lower_value_expr(expr).map(Some),
                    None => Some(None),
                }
            })
        })
        .flatten()
        .flatten();
    let (reads, emitter) = typed_return_terminator(binding);
    lowerer.emit_op(OpMeta::terminal(reads), emitter);

    // `codewriter.py CodeWriter.transform_graph_to_jitcode` runs
    // `regalloc.perform_register_allocation` on the portal graph like on
    // every other graph. Every register below the working-register floor is
    // an input or an identity slot, so it keeps its number; the working
    // registers are colored above it.
    let mut ensure_regs_stmt_idx = ensure_regs_stmt_idx;
    let (colored, _) = super::regalloc::compact_registers_pinned(
        &mut lowerer,
        pinned_floor,
        pinned_floor,
        None,
        Some(&mut ensure_regs_stmt_idx),
    );

    // Patch ensure_regs placeholder with actual register counts.
    // The ref bank must cover the ref-scalar identity slots at
    // `ref_regs[ref_identity_base..ref_identity_end)`, not just the
    // r0=program / r1=vable arguments. `MIFrame.setup` sizes `registers_r`
    // from `jitcode.num_regs_r()` (`pyjitpl.py`) and guard-failure
    // resume reads every ref register out of that bank, so a bank capped
    // at 2 would leave the ref scalars out of the frame and drop them from
    // the snapshot. `ref_identity_end` is 0 when there are no ref scalars,
    // so the no-ref-scalar case keeps the original count of 2.
    {
        let (portal_i_regs, portal_r_regs, portal_f_regs) = config.portal_input_kind_counts();
        let portal_r_regs = portal_r_regs + u16::from(config.vable_var.is_some());
        let final_i_regs = colored.ints.max(portal_i_regs);
        let final_r_regs = 2u16.max(portal_r_regs).max(ref_identity_end);
        let final_f_regs = colored
            .floats
            .max(portal_f_regs)
            .max(config.float_identity_end());
        let int_identity_end = config
            .int_identity_base()
            .saturating_add(config.plain_int_scalar_count())
            .saturating_add(config.plain_array_identity_slots());
        let final_i_regs = final_i_regs.max(int_identity_end);
        lowerer.statements[ensure_regs_stmt_idx] = quote::quote! {
            __builder.ensure_r_regs(#final_r_regs);
            __builder.ensure_i_regs(#final_i_regs);
            __builder.ensure_f_regs(#final_f_regs);
        };
    }

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    // Slice (audit Issue #5) — surface the dispatch JitCode's
    // (name, IR Type) green / red schemas to the install path so it
    // can populate `JitDriverStaticData::vars` via
    // `JitDriver::declare_schema`.  Computed BEFORE moving
    // `lowerer.statements` because the helpers borrow `&lowerer`.
    let green_schema_pairs = green_schema(&lowerer, config);
    let red_schema_pairs = red_schema(&lowerer, config);
    let statements = lowerer.statements;
    Some(GeneratedJitCodeBody {
        body: quote::quote! {
            #(#statements)*
        },
        liveness_prebuild,
        green_schema: green_schema_pairs,
        red_schema: red_schema_pairs,
    })
}

#[cfg(test)]
mod insn_op_fetch_tests {
    use super::*;

    fn portal(fetch: &str) -> syn::ItemFn {
        syn::parse_str(&format!(
            "fn mainloop(program: &Program) {{
                loop {{
                    jit_merge_point!(driver, program, pc; state);
                    {fetch}
                    match opcode {{
                        0 => {{}},
                        1 => {{}},
                        _ => break,
                    }}
                }}
            }}"
        ))
        .expect("portal fixture must parse")
    }

    fn config(with_policy: bool) -> LowererConfig {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.state_type_name = "Machine".to_string();
        config.env_type_name = "Program".to_string();
        if with_policy {
            config.calls.push((
                vec!["insn_op".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::ElidableInt),
            ));
        }
        config
    }

    fn lower_portal(fetch: &str, with_policy: bool) -> String {
        let func = portal(fetch);
        let dispatch = crate::jit_interp::codegen_trace::find_dispatch_match(&func.block)
            .expect("insn_op fetch must be the opcode dispatch");
        let arms = crate::jit_interp::classify::classify_arms(&dispatch.arms);
        let generated =
            lower_dispatch_body(&config(with_policy), &func.block, &arms, &func.sig.output)
                .expect("dispatch lowering must produce a body");
        generated.body.to_string()
    }

    #[test]
    fn cast_wrapped_insn_op_is_not_the_opcode_fetch() {
        let func = portal("let opcode = insn_op(program as _, pc as _);");
        assert!(
            crate::jit_interp::codegen_trace::find_dispatch_match(&func.block).is_none(),
            "a cast on either insn_op argument is not the opcode fetch"
        );
    }

    #[test]
    fn insn_op_without_a_call_policy_fails_the_portal() {
        let text = lower_portal("let opcode = insn_op(program, pc);", false);
        assert!(
            text.contains("compile_error"),
            "a recognized insn_op fetch with no call policy must fail loudly:\n{text}"
        );
        assert!(
            text.contains("insn_op") && text.contains("calls ="),
            "the diagnostic must name the calls registration:\n{text}"
        );
        assert!(
            !text.contains("goto_if_not_int_eq"),
            "no opcode binding must emit no dispatch arms:\n{text}"
        );
    }
}

#[cfg(test)]
mod suffix_loop_int_return_tests {
    use super::*;

    fn config() -> LowererConfig {
        let mut config = LowererConfig::inline_helper(&[], &[], &[], &[], &[], &[], &[], &[]);
        config.state_type_name = "Machine".to_string();
        config.env_type_name = "Program".to_string();
        config.state_scalars.insert("pos".to_string(), 0);
        config.state_scalars.insert("n".to_string(), 1);
        config.state_scalars.insert("a".to_string(), 2);
        config
    }

    /// An iterator `for` after the portal loop is unlowerable
    /// (`lower_for_loop` only unrolls a literal range). The prefix stays
    /// with the native suffix, so the walk must end in `void_return`
    /// (`pyjitpl.py finishframe` with `result_type == VOID` when the
    /// epilogue is not in the graph). Lowering the tail alone would drop
    /// the prefix from the JitCode.
    #[test]
    fn unlowerable_suffix_loop_emits_void_return() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn mainloop(program: &Program) -> i64 {
                while state.pos < state.n {
                    jit_merge_point!(driver, program, pc; state);
                    state.pos = state.pos + 1i64;
                }
                for _ in [0].iter() {}
                state.a
            }
        };
        let generated = lower_dispatch_body(&config(), &func.block, &[], &func.sig.output)
            .expect("dispatch lowering must produce a body");
        let body = generated.body.to_string();
        assert!(
            body.contains("void_return"),
            "an unlowerable suffix prefix must leave void_return so the \
             native suffix runs that prefix: {body}"
        );
        assert!(
            !body.contains("int_return"),
            "lowering only the tail would drop the prefix: {body}"
        );
    }

    /// A suffix that is only `return state.a` still follows
    /// `GraphFlattener.make_return` / `getkind`.
    #[test]
    fn lowerable_suffix_still_emits_int_return() {
        let func: syn::ItemFn = syn::parse_quote! {
            fn mainloop(program: &Program) -> i64 {
                while state.pos < state.n {
                    jit_merge_point!(driver, program, pc; state);
                    state.pos = state.pos + 1i64;
                }
                state.a
            }
        };
        let generated = lower_dispatch_body(&config(), &func.block, &[], &func.sig.output)
            .expect("dispatch lowering must produce a body");
        let body = generated.body.to_string();
        assert!(
            body.contains("int_return"),
            "a lowerable int tail must emit int_return \
             (flatten.py GraphFlattener.make_return / getkind): {body}"
        );
        assert!(
            !body.contains("void_return"),
            "a lowerable int tail must not emit void_return: {body}"
        );
    }
}
