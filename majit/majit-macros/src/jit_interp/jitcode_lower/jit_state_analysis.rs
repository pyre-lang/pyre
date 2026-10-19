use super::*;
use std::collections::HashSet;
use syn::visit::Visit;

/// Exhaustive walk for `state.<name>[...]` (Return, Tuple, Match guards,
/// Macro, …). `visit_expr_index` records the array, then keeps descending.
struct UsedStateArrayProbe<'a, 'c> {
    lowerer: &'a Lowerer<'c>,
    used: &'a mut HashSet<String>,
}

impl<'ast, 'a, 'c> Visit<'ast> for UsedStateArrayProbe<'a, 'c> {
    fn visit_expr_index(&mut self, idx: &'ast syn::ExprIndex) {
        self.lowerer.record_state_array_index(&idx.expr, self.used);
        syn::visit::visit_expr_index(self, idx);
    }
}

impl<'c> Lowerer<'c> {
    pub(super) fn stmt_modifies_jit_state(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_modifies_jit_state(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_modifies_jit_state(&init.expr)),
            _ => false,
        }
    }

    /// Check if an expression touches the storage pool or state fields.
    pub(super) fn expr_touches_storage(&self, expr: &Expr) -> bool {
        self.expr_has_jit_state_reference(expr) || self.expr_references_unknown_local(expr)
    }

    /// Statement-level [`Self::expr_touches_storage`]: whether the
    /// statement reads or writes jit state, storage, or user locals the
    /// trace function does not carry. Used to decide whether a statement
    /// that failed to lower can be dropped from the jitcode or must
    /// become a recording-time abort (`lower_stmt_fallback`).
    pub(super) fn stmt_touches_storage(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_touches_storage(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_touches_storage(&init.expr)),
            _ => false,
        }
    }

    /// Walks the expression looking for `Path` references to locals that
    /// are not visible in the generated trace function. The trace
    /// function's scope only carries `program`, `pc`, `__op`, plus the
    /// macro-managed `__builder` / `__ctx` / `__sym` handles. Any other
    /// bare identifier (e.g. user mainloop locals `op`, `stackok`,
    /// `is_queue`) cannot survive verbatim emission inside the trace
    /// function and must abort lowering instead.
    ///
    /// Type identifiers (uppercase first letter) and qualified paths are
    /// allowed — they resolve at module scope.
    fn expr_references_unknown_local(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(p) => {
                if let Some(ident) = p.path.get_ident() {
                    let s = ident.to_string();
                    // Whitelist trace-function scope.
                    if matches!(
                        s.as_str(),
                        "program" | "pc" | "__op" | "__sym" | "__ctx" | "__builder"
                    ) {
                        return false;
                    }
                    // Type / module / constant idents start uppercase or
                    // underscore-uppercase (e.g. `OP_MOV`, `VAL_PORT`).
                    let first = s.chars().next();
                    if first.is_some_and(|c| c.is_uppercase() || c == '_') {
                        return false;
                    }
                    // Bare lowercase identifier — assume it is a user
                    // local that the trace function will not have.
                    true
                } else {
                    // Qualified path (`a::b::c`) resolves at module scope.
                    false
                }
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_references_unknown_local(receiver)
                    || args.iter().any(|a| self.expr_references_unknown_local(a))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_references_unknown_local(func)
                    || args.iter().any(|a| self.expr_references_unknown_local(a))
            }
            Expr::Binary(ExprBinary { left, right, .. })
            | Expr::Assign(ExprAssign { left, right, .. }) => {
                self.expr_references_unknown_local(left)
                    || self.expr_references_unknown_local(right)
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. })
            | Expr::Try(syn::ExprTry { expr, .. }) => self.expr_references_unknown_local(expr),
            Expr::Field(syn::ExprField { base, .. }) => self.expr_references_unknown_local(base),
            Expr::Index(syn::ExprIndex { expr, index, .. }) => {
                self.expr_references_unknown_local(expr)
                    || self.expr_references_unknown_local(index)
            }
            Expr::Match(m) => {
                self.expr_references_unknown_local(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_references_unknown_local(&arm.body))
            }
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_references_unknown_local(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|s| self.stmt_references_unknown_local(s))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, e)| self.expr_references_unknown_local(e))
            }
            Expr::Block(b) => b
                .block
                .stmts
                .iter()
                .any(|s| self.stmt_references_unknown_local(s)),
            Expr::ForLoop(f) => {
                self.expr_references_unknown_local(&f.expr)
                    || f.body
                        .stmts
                        .iter()
                        .any(|s| self.stmt_references_unknown_local(s))
            }
            // See the `while`/`loop` note in `expr_modifies_jit_state`: without
            // these two arms a loop whose body names user locals the trace
            // function does not carry reported no reference at all (compilation-panic).
            Expr::While(w) => {
                self.expr_references_unknown_local(&w.cond)
                    || w.body
                        .stmts
                        .iter()
                        .any(|s| self.stmt_references_unknown_local(s))
            }
            Expr::Loop(l) => l
                .body
                .stmts
                .iter()
                .any(|s| self.stmt_references_unknown_local(s)),
            // A macro invocation's tokens are opaque to this walk. The
            // question it answers is whether the expression names anything
            // the generated scope will not carry, and it cannot answer that
            // about tokens it never reads — so reporting "no reference" is
            // an assumption, not a finding. `lower_local`'s runtime-constant
            // fallback reads this as permission to emit the initialiser
            // verbatim into the `__builder` block, where the portal's own
            // parameters and locals do not exist: `let bump = pick!(sel);`
            // ahead of the merge point put `match sel { .. }` in
            // `__dispatch_jitcode_*`, whose parameters are the assembler and
            // the driver index. `stmt_contains_call` below scores a macro the
            // same way, for the same reason.
            Expr::Macro(_) => true,
            // Literals, returns without expression, etc. are safe.
            _ => false,
        }
    }

    /// Statement form of [`Self::expr_references_unknown_local`]: probes
    /// both the expression of a `Stmt::Expr` and the initializer of a
    /// `Stmt::Local`.  A `let x = <unknown local>;` inside an `if` / `for`
    /// / block body references a name the trace function does not carry,
    /// so it must not be treated as reference-free.
    fn stmt_references_unknown_local(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_references_unknown_local(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_references_unknown_local(&init.expr)),
            _ => false,
        }
    }

    /// True when `stmt` contains any call, method call, or macro
    /// invocation.  A statement that reached `lower_stmt_fallback` was
    /// accepted by no lowering arm; if it still holds a call whose purity
    /// the macro cannot prove, dropping it as "inert" would delete a side
    /// effect the interpreter performs.  jtransform.py rejects operations
    /// it cannot transform (raising) rather than deleting them, so such a
    /// statement must abort lowering instead of being silently skipped.
    pub(super) fn stmt_contains_call(&self, stmt: &Stmt) -> bool {
        use syn::visit::Visit;
        struct CallProbe {
            hit: bool,
        }
        impl<'ast> Visit<'ast> for CallProbe {
            fn visit_expr_call(&mut self, c: &'ast ExprCall) {
                // `ptr::null` / `ptr::null_mut` is `Constant(nullptr)`, not a
                // call whose side effect would have to abort the arm.
                if !super::call_is_null_ptr(c) {
                    self.hit = true;
                }
                syn::visit::visit_expr_call(self, c);
            }
            fn visit_expr_method_call(&mut self, c: &'ast ExprMethodCall) {
                self.hit = true;
                syn::visit::visit_expr_method_call(self, c);
            }
            fn visit_macro(&mut self, _m: &'ast syn::Macro) {
                self.hit = true;
            }
        }
        let mut probe = CallProbe { hit: false };
        probe.visit_stmt(stmt);
        probe.hit
    }

    fn expr_modifies_jit_state(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Assign(ExprAssign { left, right, .. }) => {
                self.expr_is_jit_state_place(left)
                    || self.expr_modifies_jit_state(left)
                    || self.expr_modifies_jit_state(right)
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_modifies_jit_state(receiver)
                    || args.iter().any(|arg| self.expr_modifies_jit_state(arg))
            }
            Expr::Block(expr_block) => expr_block
                .block
                .stmts
                .iter()
                .any(|stmt| self.stmt_modifies_jit_state(stmt)),
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_modifies_jit_state(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_modifies_jit_state(stmt))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, expr)| self.expr_modifies_jit_state(expr))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_modifies_jit_state(func)
                    || args.iter().any(|arg| self.expr_modifies_jit_state(arg))
            }
            Expr::Binary(ExprBinary {
                left, right, op, ..
            }) => {
                if matches!(
                    op,
                    syn::BinOp::AddAssign(_)
                        | syn::BinOp::SubAssign(_)
                        | syn::BinOp::MulAssign(_)
                        | syn::BinOp::DivAssign(_)
                        | syn::BinOp::RemAssign(_)
                        | syn::BinOp::BitAndAssign(_)
                        | syn::BinOp::BitOrAssign(_)
                        | syn::BinOp::BitXorAssign(_)
                        | syn::BinOp::ShlAssign(_)
                        | syn::BinOp::ShrAssign(_)
                ) && self.expr_is_jit_state_place(left)
                {
                    return true;
                }
                self.expr_modifies_jit_state(left) || self.expr_modifies_jit_state(right)
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. }) => self.expr_modifies_jit_state(expr),
            Expr::Match(m) => {
                self.expr_modifies_jit_state(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_modifies_jit_state(&arm.body))
            }
            Expr::ForLoop(f) => {
                self.expr_modifies_jit_state(&f.expr)
                    || f.body.stmts.iter().any(|s| self.stmt_modifies_jit_state(s))
            }
            // `while` and `loop` descend for the same reason `for` does. They
            // answered `false` unconditionally until compilation-panic: a body writing jit
            // state was reported as writing none, and since the two sibling
            // probes were blind in the same way, `lower_stmt_fallback` scored
            // the whole loop inert and dropped it from a lowered arm.
            Expr::While(w) => {
                self.expr_modifies_jit_state(&w.cond)
                    || w.body.stmts.iter().any(|s| self.stmt_modifies_jit_state(s))
            }
            Expr::Loop(l) => l.body.stmts.iter().any(|s| self.stmt_modifies_jit_state(s)),
            Expr::Field(_)
            | Expr::Index(_)
            | Expr::Path(_)
            | Expr::Lit(_)
            | Expr::Try(_)
            | Expr::Break(_)
            | Expr::Continue(_)
            | Expr::Return(_)
            | Expr::Macro(_) => false,
            _ => false,
        }
    }

    fn expr_has_jit_state_reference(&self, expr: &Expr) -> bool {
        if self.expr_is_jit_state_place(expr) {
            return true;
        }
        // RPython parity: any reference to the state root (e.g.
        // `state.selected_dispatch_mut()`) touches the JIT-managed state.
        // The trace function does not have `state` in scope; without
        // this guard, lower_local's runtime-constant fallback would
        // emit the verbatim expression and fail to compile. Catching
        // the bare `state` path forces the macro to either lower the
        // expression to IR (Step 4b MethodCall lowering) or skip the
        // arm (treat as residual / not traced).
        if self.expr_is_state_root(expr) {
            return true;
        }
        match expr {
            Expr::Assign(ExprAssign { left, right, .. })
            | Expr::Binary(ExprBinary { left, right, .. }) => {
                self.expr_has_jit_state_reference(left) || self.expr_has_jit_state_reference(right)
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_has_jit_state_reference(receiver)
                    || args
                        .iter()
                        .any(|arg| self.expr_has_jit_state_reference(arg))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_has_jit_state_reference(func)
                    || args
                        .iter()
                        .any(|arg| self.expr_has_jit_state_reference(arg))
            }
            Expr::Block(expr_block) => expr_block
                .block
                .stmts
                .iter()
                .any(|stmt| self.stmt_touches_jit_state(stmt)),
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_has_jit_state_reference(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_touches_jit_state(stmt))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, expr)| self.expr_has_jit_state_reference(expr))
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. })
            | Expr::Try(syn::ExprTry { expr, .. }) => self.expr_has_jit_state_reference(expr),
            Expr::Index(syn::ExprIndex { expr, index, .. }) => {
                self.expr_has_jit_state_reference(expr) || self.expr_has_jit_state_reference(index)
            }
            Expr::Field(syn::ExprField { base, .. }) => self.expr_has_jit_state_reference(base),
            Expr::Match(m) => {
                self.expr_has_jit_state_reference(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_has_jit_state_reference(&arm.body))
            }
            Expr::ForLoop(f) => {
                self.expr_has_jit_state_reference(&f.expr)
                    || f.body
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_touches_jit_state(stmt))
            }
            // See the `while`/`loop` note in `expr_modifies_jit_state`: the
            // third sibling probe was blind the same way, so nothing in the
            // inert conjunction could see a loop body at all (compilation-panic).
            Expr::While(w) => {
                self.expr_has_jit_state_reference(&w.cond)
                    || w.body
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_touches_jit_state(stmt))
            }
            Expr::Loop(l) => l
                .body
                .stmts
                .iter()
                .any(|stmt| self.stmt_touches_jit_state(stmt)),
            _ => false,
        }
    }

    fn stmt_touches_jit_state(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_has_jit_state_reference(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_has_jit_state_reference(&init.expr)),
            _ => false,
        }
    }

    fn expr_is_jit_state_place(&self, expr: &Expr) -> bool {
        let config = match self.config {
            Some(c) => c,
            None => return false,
        };
        match expr {
            Expr::Field(field) => {
                if !self.expr_is_state_root(&field.base) {
                    return false;
                }
                let member = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(idx) => idx.index.to_string(),
                };
                config.state_scalars.contains_key(&member)
                    || config.state_arrays.contains_key(&member)
                    || config.state_virt_arrays.contains_key(&member)
                    // Float and ref scalars also lower field writes to
                    // `store_state_field_{float,ref}` (lower_vable.rs). Omitting
                    // them here makes an arm whose only state effect is
                    // `state.x = <float>` / `state.r = <ref>` register as
                    // runtime-only, so `lower_dispatch_body` drops the store and
                    // compiled execution leaves the field stale.
                    || config.state_float_scalars.contains_key(&member)
                    || config.state_ref_scalars.contains_key(&member)
            }
            Expr::Index(syn::ExprIndex { expr, .. }) => self.expr_is_jit_state_place(expr),
            _ => false,
        }
    }

    fn expr_is_state_root(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(path) => path.path.is_ident("state"),
            Expr::Paren(ExprParen { expr, .. }) | Expr::Reference(ExprReference { expr, .. }) => {
                self.expr_is_state_root(expr)
            }
            _ => false,
        }
    }

    /// Plain `[int]`/`[float]` names whose `state.<name>[...]` the body
    /// reads or writes, in `state_array_names_in_order`. Split-dispatch /
    /// pc-return arms declare only these as packed `{ptr, len}` arguments
    /// (`Transformer.handle_regular_call` / `MIFrame.setup_call`).
    pub(super) fn body_used_state_array_names(&self, stmts: &[Stmt]) -> Vec<String> {
        let Some(config) = self.config else {
            return Vec::new();
        };
        let mut used = HashSet::new();
        let mut probe = UsedStateArrayProbe {
            lowerer: self,
            used: &mut used,
        };
        for stmt in stmts {
            syn::visit::Visit::visit_stmt(&mut probe, stmt);
        }
        config
            .state_array_names_in_order()
            .into_iter()
            .filter(|name| used.contains(*name))
            .map(str::to_string)
            .collect()
    }

    fn record_state_array_index(&self, indexed: &Expr, used: &mut HashSet<String>) {
        let Some(config) = self.config else {
            return;
        };
        let field = match indexed {
            Expr::Field(field) => field,
            Expr::Paren(ExprParen { expr, .. }) | Expr::Reference(ExprReference { expr, .. }) => {
                self.record_state_array_index(expr, used);
                return;
            }
            _ => return,
        };
        if !self.expr_is_state_root(&field.base) {
            return;
        }
        let member = match &field.member {
            syn::Member::Named(ident) => ident.to_string(),
            syn::Member::Unnamed(idx) => idx.index.to_string(),
        };
        if config.state_arrays.contains_key(&member) {
            used.insert(member);
        }
    }
}
