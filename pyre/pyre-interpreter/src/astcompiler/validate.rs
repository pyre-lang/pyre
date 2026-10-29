//! A visitor to validate an AST object.
//!
//! `compile()` given a tree of `_ast` objects converts it and then walks the
//! result here before handing it to the code generator, so a tree the parser
//! could never have produced is rejected with a message rather than compiled.
//!
//! The walk is over the compiler AST, which cannot hold every shape the
//! objects can: a list element that is missing, a `Dict` whose keys and values
//! have different lengths, defaults that outnumber their parameters. Those are
//! rejected by the conversion in `module::_ast::convert`, which is the last
//! place they are still visible; everything else is checked here.

use rustpython_compiler::ast;

use crate::PyError;
use rustpython_wtf8::Wtf8Buf;

type ValidateResult = Result<(), PyError>;

/// Seen as a ValueError.
fn validation_error(message: impl Into<Wtf8Buf>) -> PyError {
    PyError::value_error(message)
}

/// Seen as a TypeError.
fn validation_type_error(message: impl Into<Wtf8Buf>) -> PyError {
    PyError::type_error(message)
}

pub fn validate_ast(node: &ast::Mod) -> ValidateResult {
    let validator = AstValidator;
    match node {
        ast::Mod::Module(node) => validator.validate_stmts(&node.body),
        ast::Mod::Expression(node) => validator.validate_expr(&node.body, ast::ExprContext::Load),
    }
}

/// CPython 3.14 `validate_arguments` visits vararg/kwarg annotations in the
/// slots where PyPy `AstValidator.visit_arguments` marks its two missing
/// checks, and visits every annotation before comparing either defaults list.
/// Object conversion collapses those parallel lists into Ruff's
/// `ParameterWithDefault`, so it calls this while that order is representable.
pub(crate) fn validate_parameter_annotations(
    posonlyargs: &[ast::Parameter],
    args: &[ast::Parameter],
    vararg: Option<&ast::Parameter>,
    kwonlyargs: &[ast::Parameter],
    kwarg: Option<&ast::Parameter>,
) -> ValidateResult {
    let validator = AstValidator;
    for parameter in posonlyargs {
        validator.visit_parameter(parameter)?;
    }
    for parameter in args {
        validator.visit_parameter(parameter)?;
    }
    if let Some(parameter) = vararg {
        validator.visit_parameter(parameter)?;
    }
    for parameter in kwonlyargs {
        validator.visit_parameter(parameter)?;
    }
    if let Some(parameter) = kwarg {
        validator.visit_parameter(parameter)?;
    }
    Ok(())
}

fn expr_context_name(ctx: ast::ExprContext) -> &'static str {
    match ctx {
        ast::ExprContext::Load => "Load",
        ast::ExprContext::Store => "Store",
        ast::ExprContext::Del => "Del",
        ast::ExprContext::Invalid => "??",
    }
}

fn check_context(expected_ctx: ast::ExprContext, actual_ctx: ast::ExprContext) -> ValidateResult {
    if expected_ctx != actual_ctx {
        return Err(validation_error(format!(
            "expression must have {} context but has {} instead",
            expr_context_name(expected_ctx),
            expr_context_name(actual_ctx)
        )));
    }
    Ok(())
}

/// The context an expression carries, for the ones that carry one.  The rest
/// cannot be assigned to at all.
fn context_of(expr: &ast::Expr) -> Option<ast::ExprContext> {
    match expr {
        ast::Expr::Name(node) => Some(node.ctx),
        ast::Expr::List(node) => Some(node.ctx),
        ast::Expr::Tuple(node) => Some(node.ctx),
        ast::Expr::Starred(node) => Some(node.ctx),
        ast::Expr::Subscript(node) => Some(node.ctx),
        ast::Expr::Attribute(node) => Some(node.ctx),
        _ => None,
    }
}

/// Recursive function to validate a Constant value.
fn validate_constant(value: &ast::ConstantValue) -> ValidateResult {
    match value {
        ast::ConstantValue::None
        | ast::ConstantValue::Ellipsis
        | ast::ConstantValue::Boolean(_)
        | ast::ConstantValue::Integer(_)
        | ast::ConstantValue::Float(_)
        | ast::ConstantValue::Complex { .. }
        | ast::ConstantValue::Str(_)
        | ast::ConstantValue::Bytes(_) => Ok(()),
        ast::ConstantValue::Tuple(items) | ast::ConstantValue::Frozenset(items) => {
            items.iter().try_for_each(validate_constant)
        }
    }
}

/// The operand a literal pattern is allowed to negate.
fn is_number(expr: &ast::Expr) -> bool {
    matches!(expr, ast::Expr::Constant(constant) if matches!(
        constant.value,
        ast::ConstantValue::Integer(_)
            | ast::ConstantValue::Float(_)
            | ast::ConstantValue::Complex { .. }
    ))
}

/// The real half of the complex literal a pattern may spell out, which is
/// written either on its own or negated.
fn is_signed_real(expr: &ast::Expr) -> bool {
    let expr = match expr {
        ast::Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::USub => &unary.operand,
        _ => expr,
    };
    matches!(expr, ast::Expr::Constant(constant) if matches!(
        constant.value,
        ast::ConstantValue::Integer(_) | ast::ConstantValue::Float(_)
    ))
}

/// The imaginary half of that literal, which carries its own sign.
fn is_imaginary(expr: &ast::Expr) -> bool {
    matches!(
        expr,
        ast::Expr::Constant(constant) if matches!(constant.value, ast::ConstantValue::Complex { .. })
    )
}

struct AstValidator;

impl AstValidator {
    fn validate_stmts(&self, stmts: &[ast::Stmt]) -> ValidateResult {
        stmts.iter().try_for_each(|stmt| self.visit_stmt(stmt))
    }

    fn validate_expr(&self, expr: &ast::Expr, ctx: ast::ExprContext) -> ValidateResult {
        match context_of(expr) {
            Some(actual) => check_context(ctx, actual)?,
            None if ctx != ast::ExprContext::Load => {
                return Err(validation_error(format!(
                    "expression which can't be assigned to in {} context",
                    expr_context_name(ctx)
                )));
            }
            None => {}
        }
        self.walkabout_with_ctx(expr, ctx)
    }

    /// `List`, `Tuple` and `Starred` pass their context down to what they
    /// hold; every other expression is entered in Load context.
    fn walkabout_with_ctx(&self, expr: &ast::Expr, ctx: ast::ExprContext) -> ValidateResult {
        match expr {
            ast::Expr::List(node) => self.validate_exprs(&node.elts, ctx),
            ast::Expr::Tuple(node) => self.validate_exprs(&node.elts, ctx),
            ast::Expr::Starred(node) => self.validate_expr(&node.value, ctx),
            _ => self.visit_expr(expr),
        }
    }

    /// Upstream takes a `null_ok` for the two lists that may hold a missing
    /// element -- `Dict` keys and `kw_defaults`.  Neither survives into the
    /// compiler AST as a list with holes, so there is nothing here to allow.
    fn validate_exprs(&self, exprs: &[ast::Expr], ctx: ast::ExprContext) -> ValidateResult {
        exprs
            .iter()
            .try_for_each(|expr| self.validate_expr(expr, ctx))
    }

    fn validate_body(&self, body: &[ast::Stmt], owner: &str) -> ValidateResult {
        self.validate_nonempty_seq(body.len(), "body", owner)?;
        self.validate_stmts(body)
    }

    fn validate_nonempty_seq(&self, len: usize, what: &str, owner: &str) -> ValidateResult {
        if len == 0 {
            return Err(validation_error(format!("empty {what} on {owner}")));
        }
        Ok(())
    }

    /// `validate_import_names`: a non-empty alias list whose names cannot be
    /// the `None`/`True`/`False` constants.
    fn validate_import_names(
        &self,
        names: &[ast::Alias],
        what: &str,
        owner: &str,
    ) -> ValidateResult {
        self.validate_nonempty_seq(names.len(), what, owner)?;
        for alias in names {
            self.validate_name(&alias.name)?;
            if let Some(asname) = &alias.asname {
                self.validate_name(asname)?;
            }
        }
        Ok(())
    }

    fn validate_name(&self, name: &str) -> ValidateResult {
        if matches!(name, "None" | "True" | "False") {
            return Err(validation_error(format!(
                "identifier field can't represent '{name}' constant"
            )));
        }
        Ok(())
    }

    // Statements

    fn visit_stmt(&self, stmt: &ast::Stmt) -> ValidateResult {
        match stmt {
            ast::Stmt::FunctionDef(node) => self.visit_function_def(node),
            ast::Stmt::ClassDef(node) => self.visit_class_def(node),
            ast::Stmt::Return(node) => match &node.value {
                Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                None => Ok(()),
            },
            ast::Stmt::Delete(node) => {
                self.validate_nonempty_seq(node.targets.len(), "targets", "Delete")?;
                self.validate_exprs(&node.targets, ast::ExprContext::Del)
            }
            ast::Stmt::Assign(node) => {
                self.validate_nonempty_seq(node.targets.len(), "targets", "Assign")?;
                self.validate_exprs(&node.targets, ast::ExprContext::Store)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::AugAssign(node) => {
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::AnnAssign(node) => {
                if node.simple && !matches!(&*node.target, ast::Expr::Name(_)) {
                    return Err(validation_type_error(
                        "AnnAssign with simple non-Name target",
                    ));
                }
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.annotation, ast::ExprContext::Load)?;
                match &node.value {
                    Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                    None => Ok(()),
                }
            }
            ast::Stmt::TypeAlias(node) => {
                if !matches!(&*node.name, ast::Expr::Name(_)) {
                    return Err(validation_type_error("TypeAlias with non-Name name"));
                }
                self.validate_expr(&node.name, ast::ExprContext::Store)?;
                self.validate_type_params(node.type_params.as_deref())?;
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Stmt::For(node) => {
                let owner = if node.is_async { "AsyncFor" } else { "For" };
                self.validate_expr(&node.target, ast::ExprContext::Store)?;
                self.validate_expr(&node.iter, ast::ExprContext::Load)?;
                self.validate_body(&node.body, owner)?;
                self.validate_stmts(&node.orelse)
            }
            ast::Stmt::While(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_body(&node.body, "While")?;
                self.validate_stmts(&node.orelse)
            }
            ast::Stmt::If(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_body(&node.body, "If")?;
                // An `elif` is the `orelse` holding one more `If`, so its own
                // body carries the same requirement; a plain `else` is only a
                // statement list.
                for clause in &node.elif_else_clauses {
                    match &clause.test {
                        Some(test) => {
                            self.validate_expr(test, ast::ExprContext::Load)?;
                            self.validate_body(&clause.body, "If")?;
                        }
                        None => self.validate_stmts(&clause.body)?,
                    }
                }
                Ok(())
            }
            ast::Stmt::With(node) => {
                let owner = if node.is_async { "AsyncWith" } else { "With" };
                self.validate_nonempty_seq(node.items.len(), "items", owner)?;
                for item in &node.items {
                    self.validate_expr(&item.context_expr, ast::ExprContext::Load)?;
                    if let Some(vars) = item.optional_vars.as_deref() {
                        self.validate_expr(vars, ast::ExprContext::Store)?;
                    }
                }
                self.validate_body(&node.body, owner)
            }
            ast::Stmt::Raise(node) => match (&node.exc, &node.cause) {
                (Some(exc), cause) => {
                    self.validate_expr(exc, ast::ExprContext::Load)?;
                    match cause {
                        Some(cause) => self.validate_expr(cause, ast::ExprContext::Load),
                        None => Ok(()),
                    }
                }
                (None, Some(_)) => Err(validation_error("Raise with cause but no exception")),
                (None, None) => Ok(()),
            },
            ast::Stmt::Try(node) => self.visit_try(node),
            ast::Stmt::Assert(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                match &node.msg {
                    Some(msg) => self.validate_expr(msg, ast::ExprContext::Load),
                    None => Ok(()),
                }
            }
            ast::Stmt::Import(node) => self.validate_import_names(&node.names, "names", "Import"),
            ast::Stmt::ImportFrom(node) => {
                self.validate_import_names(&node.names, "names", "ImportFrom")
            }
            ast::Stmt::Global(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "Global")
            }
            ast::Stmt::Nonlocal(node) => {
                self.validate_nonempty_seq(node.names.len(), "names", "Nonlocal")
            }
            ast::Stmt::Expr(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Stmt::Match(node) => self.visit_match(node),
            ast::Stmt::Pass(_) | ast::Stmt::Break(_) | ast::Stmt::Continue(_) => Ok(()),
            ast::Stmt::IpyEscapeCommand(_) => Ok(()),
        }
    }

    fn visit_function_def(&self, node: &ast::StmtFunctionDef) -> ValidateResult {
        let owner = if node.is_async {
            "AsyncFunctionDef"
        } else {
            "FunctionDef"
        };
        self.validate_body(&node.body, owner)?;
        self.validate_name(&node.name)?;
        self.validate_type_params(node.type_params.as_deref())?;
        self.visit_parameters(&node.parameters)?;
        self.validate_decorators(&node.decorator_list)?;
        match &node.returns {
            Some(returns) => self.validate_expr(returns, ast::ExprContext::Load),
            None => Ok(()),
        }
    }

    fn visit_class_def(&self, node: &ast::StmtClassDef) -> ValidateResult {
        self.validate_body(&node.body, "ClassDef")?;
        self.validate_name(&node.name)?;
        self.validate_type_params(node.type_params.as_deref())?;
        if let Some(arguments) = node.arguments.as_deref() {
            self.validate_exprs(&arguments.args, ast::ExprContext::Load)?;
            for keyword in &arguments.keywords {
                self.validate_expr(&keyword.value, ast::ExprContext::Load)?;
            }
        }
        self.validate_decorators(&node.decorator_list)
    }

    /// A type parameter carries a name, and everything else it holds is an
    /// expression read in Load context. The checked-in tree predates PEP 695
    /// and has no counterpart, so this answers to 3.14.
    fn validate_type_params(&self, type_params: Option<&ast::TypeParams>) -> ValidateResult {
        let Some(type_params) = type_params else {
            return Ok(());
        };
        for type_param in &type_params.type_params {
            let (name, bound, default) = match type_param {
                ast::TypeParam::TypeVar(node) => {
                    (&node.name, node.bound.as_deref(), node.default.as_deref())
                }
                ast::TypeParam::TypeVarTuple(node) => (&node.name, None, node.default.as_deref()),
                ast::TypeParam::ParamSpec(node) => (&node.name, None, node.default.as_deref()),
            };
            self.validate_name(name)?;
            for expr in [bound, default].into_iter().flatten() {
                self.validate_expr(expr, ast::ExprContext::Load)?;
            }
        }
        Ok(())
    }

    fn validate_decorators(&self, decorators: &[ast::Decorator]) -> ValidateResult {
        decorators.iter().try_for_each(|decorator| {
            self.validate_expr(&decorator.expression, ast::ExprContext::Load)
        })
    }

    fn visit_parameters(&self, node: &ast::Parameters) -> ValidateResult {
        for parameter in node
            .posonlyargs
            .iter()
            .chain(&node.args)
            .chain(&node.kwonlyargs)
        {
            self.visit_parameter(&parameter.parameter)?;
            if let Some(default) = parameter.default.as_deref() {
                self.validate_expr(default, ast::ExprContext::Load)?;
            }
        }
        for parameter in [node.vararg.as_deref(), node.kwarg.as_deref()]
            .into_iter()
            .flatten()
        {
            self.visit_parameter(parameter)?;
        }
        Ok(())
    }

    fn visit_parameter(&self, node: &ast::Parameter) -> ValidateResult {
        match &node.annotation {
            Some(annotation) => self.validate_expr(annotation, ast::ExprContext::Load),
            None => Ok(()),
        }
    }

    fn visit_try(&self, node: &ast::StmtTry) -> ValidateResult {
        let owner = if node.is_star { "TryStar" } else { "Try" };
        self.validate_body(&node.body, owner)?;
        if node.handlers.is_empty() && node.finalbody.is_empty() {
            return Err(validation_error(format!(
                "{owner} has neither except handlers nor finalbody"
            )));
        }
        if node.handlers.is_empty() && !node.orelse.is_empty() {
            return Err(validation_error(format!(
                "{owner} has orelse but no except handlers"
            )));
        }
        for handler in &node.handlers {
            let ast::ExceptHandler::ExceptHandler(handler) = handler;
            if let Some(type_) = handler.type_.as_deref() {
                self.validate_expr(type_, ast::ExprContext::Load)?;
            }
            if !node.is_star
                && let Some(name) = &handler.name
            {
                self.validate_name(name)?;
            }
            self.validate_body(&handler.body, "ExceptHandler")?;
        }
        self.validate_stmts(&node.orelse)?;
        self.validate_stmts(&node.finalbody)
    }

    // pattern matching

    /// A second star in the same sequence is left to the code generator, which
    /// reports it where 3.14 does, as a SyntaxError; upstream raises here and
    /// so answers with a ValueError.
    fn validate_patterns(&self, patterns: &[ast::Pattern], star_ok: bool) -> ValidateResult {
        for pattern in patterns {
            if let ast::Pattern::MatchStar(node) = pattern
                && star_ok
            {
                if let Some(name) = &node.name {
                    self.validate_capture(name)?;
                }
                continue;
            }
            self.visit_pattern(pattern)?;
        }
        Ok(())
    }

    fn validate_capture(&self, name: &str) -> ValidateResult {
        if name == "_" {
            return Err(validation_error("can't capture name '_' in patterns"));
        }
        self.validate_name(name)
    }

    fn visit_match(&self, node: &ast::StmtMatch) -> ValidateResult {
        self.validate_nonempty_seq(node.cases.len(), "cases", "Match")?;
        self.validate_expr(&node.subject, ast::ExprContext::Load)?;
        for case in &node.cases {
            if let Some(guard) = case.guard.as_deref() {
                self.validate_expr(guard, ast::ExprContext::Load)?;
            }
            self.validate_stmts(&case.body)?;
            self.visit_pattern(&case.pattern)?;
        }
        Ok(())
    }

    fn visit_pattern(&self, pattern: &ast::Pattern) -> ValidateResult {
        match pattern {
            ast::Pattern::MatchValue(node) => {
                self.validate_expr(&node.value, ast::ExprContext::Load)?;
                match node.value.as_ref() {
                    ast::Expr::Constant(constant) => {
                        // Ellipsis and the immutable containers are out; True,
                        // False and None belong to MatchSingleton.
                        if matches!(
                            constant.value,
                            ast::ConstantValue::Integer(_)
                                | ast::ConstantValue::Float(_)
                                | ast::ConstantValue::Complex { .. }
                                | ast::ConstantValue::Str(_)
                                | ast::ConstantValue::Bytes(_)
                        ) {
                            Ok(())
                        } else {
                            Err(validation_error(
                                "unexpected constant inside of a literal pattern",
                            ))
                        }
                    }
                    // An attribute lookup is always permitted.
                    ast::Expr::Attribute(_) => Ok(()),
                    // A negated number and a complex literal reach here as the
                    // operations that build them, and only in those shapes.
                    ast::Expr::UnaryOp(unary)
                        if unary.op == ast::UnaryOp::USub && is_number(&unary.operand) =>
                    {
                        Ok(())
                    }
                    ast::Expr::BinOp(binop)
                        if matches!(binop.op, ast::Operator::Add | ast::Operator::Sub)
                            && is_signed_real(&binop.left)
                            && is_imaginary(&binop.right) =>
                    {
                        Ok(())
                    }
                    // An f-string is left for the code generator to report.
                    ast::Expr::FString(_) => Ok(()),
                    // Upstream stops at the constant check and lets everything
                    // else through; 3.14 rejects it here, before the code
                    // generator reports the same thing as a SyntaxError.
                    _ => Err(validation_error(
                        "patterns may only match literals and attribute lookups",
                    )),
                }
            }
            ast::Pattern::MatchSingleton(_) => Ok(()),
            ast::Pattern::MatchSequence(node) => self.validate_patterns(&node.patterns, true),
            ast::Pattern::MatchMapping(node) => {
                // Upstream compares the two only when both are non-empty, so a
                // node carrying keys and no patterns reaches the code
                // generator, which zips them; 3.14 compares unconditionally.
                // `{**rest}` leaves both empty and still passes.
                if node.keys.len() != node.patterns.len() {
                    return Err(validation_error(
                        "MatchMapping doesn't have the same number of keys as patterns",
                    ));
                }
                for key in &node.keys {
                    if !matches!(key, ast::Expr::Constant(_) | ast::Expr::Attribute(_)) {
                        // Upstream words this as "can only have Constant or
                        // Attribute in the keys of a MatchMapping".
                        return Err(validation_error(
                            "patterns may only match literals and attribute lookups",
                        ));
                    }
                }
                self.validate_exprs(&node.keys, ast::ExprContext::Load)?;
                if let Some(rest) = &node.rest {
                    self.validate_capture(rest)?;
                }
                // Upstream walks `keys` here, which are expressions, so a
                // pattern nested in the mapping goes unchecked; 3.14 walks the
                // patterns and this follows it.
                self.validate_patterns(&node.patterns, false)
            }
            ast::Pattern::MatchClass(node) => {
                self.validate_expr(&node.cls, ast::ExprContext::Load)?;
                let mut cls = node.cls.as_ref();
                loop {
                    match cls {
                        ast::Expr::Name(_) => break,
                        ast::Expr::Attribute(attribute) => cls = &attribute.value,
                        _ => {
                            return Err(validation_error(
                                "MatchClass cls field can only contain Name or Attribute nodes.",
                            ));
                        }
                    }
                }
                for keyword in &node.arguments.keywords {
                    self.validate_name(&keyword.attr)?;
                }
                self.validate_patterns(&node.arguments.patterns, false)?;
                for keyword in &node.arguments.keywords {
                    self.visit_pattern(&keyword.pattern)?;
                }
                Ok(())
            }
            ast::Pattern::MatchStar(_) => Err(validation_error("can't use MatchStar here")),
            ast::Pattern::MatchAs(node) => {
                if let Some(name) = &node.name {
                    self.validate_capture(name)?;
                }
                match (&node.pattern, &node.name) {
                    (Some(_), None) => Err(validation_error(
                        "MatchAs must specify a target name if a pattern is given",
                    )),
                    (Some(pattern), Some(_)) => self.visit_pattern(pattern),
                    (None, _) => Ok(()),
                }
            }
            ast::Pattern::MatchOr(node) => {
                if node.patterns.len() < 2 {
                    return Err(validation_error("MatchOr requires at least 2 patterns"));
                }
                self.validate_patterns(&node.patterns, false)
            }
        }
    }

    // Expressions

    fn visit_expr(&self, expr: &ast::Expr) -> ValidateResult {
        match expr {
            ast::Expr::Name(node) => self.validate_name(&node.id),
            ast::Expr::Constant(node) => validate_constant(&node.value),
            ast::Expr::BoolOp(node) => {
                if node.values.len() < 2 {
                    return Err(validation_error("BoolOp with less than 2 values"));
                }
                self.validate_exprs(&node.values, ast::ExprContext::Load)
            }
            ast::Expr::UnaryOp(node) => self.validate_expr(&node.operand, ast::ExprContext::Load),
            ast::Expr::BinOp(node) => {
                self.validate_expr(&node.left, ast::ExprContext::Load)?;
                self.validate_expr(&node.right, ast::ExprContext::Load)
            }
            ast::Expr::Lambda(node) => {
                if let Some(parameters) = node.parameters.as_deref() {
                    self.visit_parameters(parameters)?;
                }
                self.validate_expr(&node.body, ast::ExprContext::Load)
            }
            ast::Expr::If(node) => {
                self.validate_expr(&node.test, ast::ExprContext::Load)?;
                self.validate_expr(&node.body, ast::ExprContext::Load)?;
                self.validate_expr(&node.orelse, ast::ExprContext::Load)
            }
            ast::Expr::Dict(node) => {
                for item in &node.items {
                    if let Some(key) = &item.key {
                        self.validate_expr(key, ast::ExprContext::Load)?;
                    }
                    self.validate_expr(&item.value, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::Set(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            ast::Expr::ListComp(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::SetComp(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::Generator(node) => {
                self.validate_comprehension(&node.generators)?;
                self.validate_expr(&node.elt, ast::ExprContext::Load)
            }
            ast::Expr::DictComp(node) => {
                self.validate_comprehension(&node.generators)?;
                // Unpacking `{**d for ...}` stores the operand in `value`
                // with `key` empty. Public `DictComp` is `expr key, expr?
                // value`; the native node accepts either shape.
                if let Some(key) = node.key.as_deref() {
                    self.validate_expr(key, ast::ExprContext::Load)?;
                }
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Expr::Await(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Yield(node) => match &node.value {
                Some(value) => self.validate_expr(value, ast::ExprContext::Load),
                None => Ok(()),
            },
            ast::Expr::YieldFrom(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Compare(node) => {
                if node.operands.len() < 2 {
                    return Err(validation_error("Compare with no comparators"));
                }
                if node.ops.len() + 1 != node.operands.len() {
                    return Err(validation_error(
                        "Compare has a different number of comparators and operands",
                    ));
                }
                // validate.py visit_Compare: comparators first, then left.
                self.validate_exprs(node.comparators(), ast::ExprContext::Load)?;
                self.validate_expr(node.first_operand(), ast::ExprContext::Load)
            }
            ast::Expr::Call(node) => {
                self.validate_expr(&node.func, ast::ExprContext::Load)?;
                self.validate_exprs(&node.arguments.args, ast::ExprContext::Load)?;
                for keyword in &node.arguments.keywords {
                    self.validate_expr(&keyword.value, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::Attribute(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::Subscript(node) => {
                self.validate_expr(&node.value, ast::ExprContext::Load)?;
                self.validate_expr(&node.slice, ast::ExprContext::Load)
            }
            ast::Expr::Slice(node) => {
                for part in [
                    node.lower.as_deref(),
                    node.upper.as_deref(),
                    node.step.as_deref(),
                ]
                .into_iter()
                .flatten()
                {
                    self.validate_expr(part, ast::ExprContext::Load)?;
                }
                Ok(())
            }
            ast::Expr::FString(node) => self.visit_fstring(node),
            ast::Expr::TString(node) => self.visit_tstring(node),
            ast::Expr::Named(node) => {
                if !matches!(node.target.as_ref(), ast::Expr::Name(_)) {
                    return Err(validation_type_error("NamedExpr target must be a Name"));
                }
                self.validate_expr(&node.value, ast::ExprContext::Load)
            }
            ast::Expr::Starred(node) => self.validate_expr(&node.value, ast::ExprContext::Load),
            ast::Expr::List(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            ast::Expr::Tuple(node) => self.validate_exprs(&node.elts, ast::ExprContext::Load),
            // A literal the parser produced, or a node with nothing to check.
            ast::Expr::StringLiteral(_)
            | ast::Expr::BytesLiteral(_)
            | ast::Expr::NumberLiteral(_)
            | ast::Expr::BooleanLiteral(_)
            | ast::Expr::NoneLiteral(_)
            | ast::Expr::EllipsisLiteral(_)
            | ast::Expr::IpyEscapeCommand(_) => Ok(()),
        }
    }

    /// RustPython `_ast::validate::validate_interpolated_elements`: both
    /// string forms validate the expression and whichever representation of
    /// the format spec their object-to-native boundary supplied.
    fn validate_interpolated_elements<'a>(
        &self,
        elements: impl IntoIterator<Item = &'a ast::InterpolatedStringElement>,
    ) -> ValidateResult {
        for element in elements {
            if let ast::InterpolatedStringElement::Interpolation(interpolation) = element {
                self.validate_expr(&interpolation.expression, ast::ExprContext::Load)?;
                if let Some(spec) = interpolation.runtime_formatted_value_format_spec.as_deref() {
                    self.validate_expr(spec, ast::ExprContext::Load)?;
                } else if let Some(spec) =
                    interpolation.runtime_interpolation_format_spec.as_deref()
                {
                    self.validate_expr(spec, ast::ExprContext::Load)?;
                } else if let Some(spec) = interpolation.format_spec.as_deref() {
                    self.validate_interpolated_elements(spec.elements.iter())?;
                }
            }
        }
        Ok(())
    }

    fn validate_runtime_expr_slots(
        &self,
        values: Option<&Vec<Option<ast::Expr>>>,
    ) -> ValidateResult {
        if let Some(values) = values {
            for value in values.iter().flatten() {
                self.validate_expr(value, ast::ExprContext::Load)?;
            }
        }
        Ok(())
    }

    /// The values of a `JoinedStr` reach the compiler AST as the parts to
    /// join, and a `FormattedValue` spec as the expression to format with.
    fn visit_fstring(&self, node: &ast::ExprFString) -> ValidateResult {
        self.validate_runtime_expr_slots(node.runtime_values.as_ref())?;
        if let Some(values) = node.runtime_joined_str.as_deref() {
            return self.validate_exprs(values, ast::ExprContext::Load);
        }
        self.validate_interpolated_elements(node.value.elements())
    }

    /// RustPython `_ast::validate::validate_expr` `Expr::TString` arm.
    fn visit_tstring(&self, node: &ast::ExprTString) -> ValidateResult {
        self.validate_runtime_expr_slots(node.runtime_values.as_ref())?;
        if let Some(values) = node.runtime_template_str.as_deref() {
            return self.validate_exprs(values, ast::ExprContext::Load);
        }
        self.validate_interpolated_elements(node.value.elements())
    }

    fn validate_comprehension(&self, generators: &[ast::Comprehension]) -> ValidateResult {
        if generators.is_empty() {
            return Err(validation_error("comprehension with no generators"));
        }
        for comprehension in generators {
            self.validate_expr(&comprehension.target, ast::ExprContext::Store)?;
            self.validate_expr(&comprehension.iter, ast::ExprContext::Load)?;
            self.validate_exprs(&comprehension.ifs, ast::ExprContext::Load)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the walk and 3.14 disagree the shared roundtrip fixture cannot
    /// carry the case: every backend's output there is compared against PyPy's.
    fn validate(body: Vec<ast::Stmt>) -> ValidateResult {
        validate_ast(&ast::Mod::Module(ast::ModModule {
            node_index: Default::default(),
            range: Default::default(),
            body: body.into(),
            runtime_body: None,
        }))
    }

    fn name(id: &str, ctx: ast::ExprContext) -> ast::Expr {
        ast::Expr::Name(ast::ExprName {
            node_index: Default::default(),
            range: Default::default(),
            id: id.into(),
            ctx,
        })
    }

    fn constant(value: i32) -> ast::Expr {
        ast::Expr::Constant(ast::ExprConstant {
            node_index: Default::default(),
            range: Default::default(),
            value: ast::ConstantValue::Integer(value.to_string().into_boxed_str()),
            kind: None,
            invalid_type: None,
        })
    }

    fn pass() -> ast::Stmt {
        ast::Stmt::Pass(ast::StmtPass {
            node_index: Default::default(),
            range: Default::default(),
        })
    }

    fn matched(pattern: ast::Pattern) -> Vec<ast::Stmt> {
        vec![ast::Stmt::Match(ast::StmtMatch {
            node_index: Default::default(),
            range: Default::default(),
            subject: Box::new(constant(1)),
            cases: vec![ast::MatchCase {
                node_index: Default::default(),
                range: Default::default(),
                pattern,
                guard: None,
                body: vec![pass()].into(),
                runtime_body: None,
            }],
        })]
    }

    fn mapping(keys: Vec<ast::Expr>, patterns: Vec<ast::Pattern>) -> ast::Pattern {
        ast::Pattern::MatchMapping(ast::PatternMatchMapping {
            node_index: Default::default(),
            range: Default::default(),
            keys: keys.into(),
            patterns: patterns.into(),
            rest: None,
            runtime_keys: None,
            runtime_patterns: None,
        })
    }

    fn literal(value: ast::ConstantValue) -> ast::Expr {
        ast::Expr::Constant(ast::ExprConstant {
            node_index: Default::default(),
            range: Default::default(),
            value,
            kind: None,
            invalid_type: None,
        })
    }

    fn unary(op: ast::UnaryOp, operand: ast::Expr) -> ast::Expr {
        ast::Expr::UnaryOp(ast::ExprUnaryOp {
            node_index: Default::default(),
            range: Default::default(),
            op,
            operand: Box::new(operand),
        })
    }

    fn binop(left: ast::Expr, op: ast::Operator, right: ast::Expr) -> ast::Expr {
        ast::Expr::BinOp(ast::ExprBinOp {
            node_index: Default::default(),
            range: Default::default(),
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    fn value(expr: ast::Expr) -> ast::Pattern {
        ast::Pattern::MatchValue(ast::PatternMatchValue {
            node_index: Default::default(),
            range: Default::default(),
            value: Box::new(expr),
        })
    }

    fn message(result: ValidateResult) -> String {
        result.unwrap_err().message_text()
    }

    #[test]
    fn match_value_takes_only_a_literal_or_an_attribute() {
        let value = ast::Pattern::MatchValue(ast::PatternMatchValue {
            node_index: Default::default(),
            range: Default::default(),
            value: Box::new(name("foo", ast::ExprContext::Load)),
        });
        assert_eq!(
            message(validate(matched(value))),
            "patterns may only match literals and attribute lookups"
        );

        let value = ast::Pattern::MatchValue(ast::PatternMatchValue {
            node_index: Default::default(),
            range: Default::default(),
            value: Box::new(ast::Expr::Attribute(ast::ExprAttribute {
                node_index: Default::default(),
                range: Default::default(),
                value: Box::new(name("o", ast::ExprContext::Load)),
                attr: ast::Identifier::new("a", Default::default()),
                ctx: ast::ExprContext::Load,
            })),
        });
        assert!(validate(matched(value)).is_ok());
    }

    #[test]
    fn match_value_takes_a_negated_number_and_a_complex_literal() {
        let float = || literal(ast::ConstantValue::Float(1.5));
        let imaginary = || {
            literal(ast::ConstantValue::Complex {
                real: 0.0,
                imag: 2.0,
            })
        };
        let text = || literal(ast::ConstantValue::Str("a".into()));
        let accepted = [
            unary(ast::UnaryOp::USub, constant(1)),
            unary(ast::UnaryOp::USub, float()),
            unary(ast::UnaryOp::USub, imaginary()),
            binop(constant(1), ast::Operator::Add, imaginary()),
            binop(float(), ast::Operator::Sub, imaginary()),
            binop(
                unary(ast::UnaryOp::USub, constant(1)),
                ast::Operator::Add,
                imaginary(),
            ),
        ];
        for expr in accepted {
            assert!(validate(matched(value(expr))).is_ok());
        }

        let rejected = [
            // Only a negation, and only over a number.
            unary(ast::UnaryOp::USub, text()),
            unary(ast::UnaryOp::UAdd, constant(1)),
            unary(ast::UnaryOp::Invert, constant(1)),
            unary(ast::UnaryOp::USub, unary(ast::UnaryOp::USub, constant(1))),
            // A real half and then an imaginary one, added or subtracted.
            binop(imaginary(), ast::Operator::Add, constant(1)),
            binop(constant(1), ast::Operator::Add, constant(2)),
            binop(text(), ast::Operator::Add, imaginary()),
            binop(constant(1), ast::Operator::Mult, imaginary()),
            binop(
                constant(1),
                ast::Operator::Add,
                unary(ast::UnaryOp::USub, imaginary()),
            ),
        ];
        for expr in rejected {
            assert_eq!(
                message(validate(matched(value(expr)))),
                "patterns may only match literals and attribute lookups"
            );
        }
    }

    #[test]
    fn match_mapping_counts_keys_against_patterns() {
        assert_eq!(
            message(validate(matched(mapping(vec![constant(1)], Vec::new())))),
            "MatchMapping doesn't have the same number of keys as patterns"
        );
        // `{**rest}` leaves both empty, which is not a mismatch.
        assert!(validate(matched(mapping(Vec::new(), Vec::new()))).is_ok());
    }

    #[test]
    fn match_mapping_reads_its_keys_in_load_context() {
        let key = ast::Expr::Attribute(ast::ExprAttribute {
            node_index: Default::default(),
            range: Default::default(),
            value: Box::new(name("o", ast::ExprContext::Store)),
            attr: ast::Identifier::new("a", Default::default()),
            ctx: ast::ExprContext::Load,
        });
        let pattern = ast::Pattern::MatchAs(ast::PatternMatchAs {
            node_index: Default::default(),
            range: Default::default(),
            pattern: None,
            name: Some(ast::Identifier::new("v", Default::default())),
        });
        assert_eq!(
            message(validate(matched(mapping(vec![key], vec![pattern])))),
            "expression must have Load context but has Store instead"
        );
    }

    #[test]
    fn match_mapping_walks_the_patterns_not_the_keys() {
        // Upstream walks `keys`, so a star nested here goes unreported.
        let star = ast::Pattern::MatchStar(ast::PatternMatchStar {
            node_index: Default::default(),
            range: Default::default(),
            name: Some(ast::Identifier::new("a", Default::default())),
        });
        assert_eq!(
            message(validate(matched(mapping(vec![constant(1)], vec![star])))),
            "can't use MatchStar here"
        );
    }

    #[test]
    fn type_params_are_read_in_load_context() {
        let type_params = |type_param| {
            Some(Box::new(ast::TypeParams {
                node_index: Default::default(),
                range: Default::default(),
                type_params: vec![type_param],
                runtime_type_params: None,
            }))
        };
        let function = |type_param| {
            vec![ast::Stmt::FunctionDef(ast::StmtFunctionDef {
                node_index: Default::default(),
                range: Default::default(),
                is_async: false,
                decorator_list: thin_vec::ThinVec::new(),
                name: ast::Identifier::new("f", Default::default()),
                type_params: type_params(type_param),
                parameters: Box::new(ast::Parameters::default()),
                returns: None,
                body: vec![pass()].into(),
                runtime_decorator_list: None,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            })]
        };

        let bound = ast::TypeParam::TypeVar(ast::TypeParamTypeVar {
            node_index: Default::default(),
            range: Default::default(),
            name: ast::Identifier::new("T", Default::default()),
            bound: Some(Box::new(name("x", ast::ExprContext::Store))),
            default: None,
        });
        assert_eq!(
            message(validate(function(bound))),
            "expression must have Load context but has Store instead"
        );

        let named = ast::TypeParam::ParamSpec(ast::TypeParamParamSpec {
            node_index: Default::default(),
            range: Default::default(),
            name: ast::Identifier::new("None", Default::default()),
            default: None,
        });
        assert_eq!(
            message(validate(function(named))),
            "identifier field can't represent 'None' constant"
        );
    }

    #[test]
    fn named_expr_target_must_be_a_name() {
        let named = |target| {
            vec![ast::Stmt::Expr(ast::StmtExpr {
                node_index: Default::default(),
                range: Default::default(),
                value: Box::new(ast::Expr::Named(ast::ExprNamed {
                    node_index: Default::default(),
                    range: Default::default(),
                    target: Box::new(target),
                    value: Box::new(constant(1)),
                })),
            })]
        };

        // The class is the whole case: a runtime without this check reaches
        // the code generator and reports the target as an unassignable
        // expression, a ValueError.
        let error = validate(named(constant(1))).unwrap_err();
        assert_eq!(error.kind, crate::error::PyErrorKind::TypeError);
        assert_eq!(error.message_text(), "NamedExpr target must be a Name");

        // Only the shape is checked here; the context the target carries is
        // not, so a `Store` name passes the walk.
        assert!(validate(named(name("x", ast::ExprContext::Store))).is_ok());
    }

    fn alias(name: &str, asname: Option<&str>) -> ast::Alias {
        ast::Alias {
            node_index: Default::default(),
            range: Default::default(),
            name: ast::Identifier::new(name, Default::default()),
            asname: asname.map(|name| ast::Identifier::new(name, Default::default())),
        }
    }

    #[test]
    fn identifier_fields_reject_none_true_false() {
        let forbidden = |name| format!("identifier field can't represent '{name}' constant");
        for constant in ["None", "True", "False"] {
            let function = vec![ast::Stmt::FunctionDef(ast::StmtFunctionDef {
                node_index: Default::default(),
                range: Default::default(),
                is_async: false,
                decorator_list: thin_vec::ThinVec::new(),
                name: ast::Identifier::new(constant, Default::default()),
                type_params: None,
                parameters: Box::new(ast::Parameters::default()),
                returns: None,
                body: vec![pass()].into(),
                runtime_decorator_list: None,
                runtime_type_comment: None,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            })];
            assert_eq!(message(validate(function)), forbidden(constant));

            let class = vec![ast::Stmt::ClassDef(ast::StmtClassDef {
                node_index: Default::default(),
                range: Default::default(),
                decorator_list: thin_vec::ThinVec::new(),
                name: ast::Identifier::new(constant, Default::default()),
                type_params: None,
                arguments: None,
                body: vec![pass()].into(),
                runtime_decorator_list: None,
                runtime_body: None,
            })];
            assert_eq!(message(validate(class)), forbidden(constant));

            let import = vec![ast::Stmt::Import(ast::StmtImport {
                node_index: Default::default(),
                range: Default::default(),
                names: vec![alias(constant, None)].into(),
                is_lazy: false,
            })];
            assert_eq!(message(validate(import)), forbidden(constant));

            let import_as = vec![ast::Stmt::Import(ast::StmtImport {
                node_index: Default::default(),
                range: Default::default(),
                names: vec![alias("a", Some(constant))].into(),
                is_lazy: false,
            })];
            assert_eq!(message(validate(import_as)), forbidden(constant));

            let try_except = vec![ast::Stmt::Try(ast::StmtTry {
                node_index: Default::default(),
                range: Default::default(),
                body: vec![pass()].into(),
                handlers: vec![ast::ExceptHandler::ExceptHandler(
                    ast::ExceptHandlerExceptHandler {
                        range: Default::default(),
                        node_index: Default::default(),
                        type_: Some(Box::new(name("A", ast::ExprContext::Load))),
                        name: Some(ast::Identifier::new(constant, Default::default())),
                        body: vec![pass()].into(),
                        runtime_body: None,
                    },
                )]
                .into(),
                orelse: Vec::new().into(),
                finalbody: Vec::new().into(),
                is_star: false,
                runtime_body: None,
                runtime_handlers: None,
                runtime_orelse: None,
                runtime_finalbody: None,
            })];
            assert_eq!(message(validate(try_except)), forbidden(constant));
        }
    }
}
