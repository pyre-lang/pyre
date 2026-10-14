//! RustPython/Ruff AST → interpreter-level `_ast` objects.
//!
//! PyPy's `ast.Node.to_object(space)` performs this same boundary conversion:
//! parser nodes stay native to the compiler, while the public `ast` module sees
//! ordinary heap objects carrying ASDL fields and source locations.

use pyre_object::{PY_NULL, PyObjectRef};
use rustpython_compiler::codegen::{
    interpolated_string_literal_value, interpolation_debug_text, string_literal_part_value,
    string_literal_value,
};
use rustpython_compiler::core::{SourceFile, SourceFileBuilder};
use rustpython_compiler::{ast, parser};
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use thin_vec::ThinVec;

type AstResult<T> = Result<T, crate::PyError>;

/// RustPython `_ast::should_report_unsupported_syntax_error`: Ruff records
/// every version-shaped difference, including compatibility syntax CPython
/// deliberately still accepts for older feature versions.  Only these are
/// parser errors at the public `feature_version` boundary.
fn should_report_unsupported_syntax_error(error: &parser::UnsupportedSyntaxError) -> bool {
    use parser::UnsupportedSyntaxErrorKind as Kind;
    matches!(
        error.kind,
        Kind::Match
            | Kind::Walrus
            | Kind::ExceptStar
            | Kind::PositionalOnlyParameter
            | Kind::TypeParameterList
            | Kind::TypeAliasStatement
            | Kind::TypeParamDefault
            | Kind::TemplateStrings
            | Kind::UnparenthesizedExceptionTypes
            | Kind::LazyImportStatement
            | Kind::ParenthesizedKeywordArgumentName
    )
}

/// PyPy's `PythonParser` applies the requested `CompileInfo.feature_version`
/// while reducing these grammar productions. Ruff's supported-version table
/// starts after some of those historical boundaries, so its unsupported-
/// syntax side channel cannot represent them. Recover the 3.5/3.6 gates
/// from the native tree and numeric tokens before publishing an `_ast` tree.
/// [3.14-spec] PyPy `BaseParser.check_version` owns this tree-site structure
/// but says "Python (3, N) and above"; CPython 3.14 `CHECK_VERSION` exposes
/// "Python 3.N and greater", so the app-visible strings below follow 3.14.
fn legacy_feature_version_error(
    module: &ast::Mod,
    tokens: &ast::token::Tokens,
    source: &str,
    feature_version: i64,
) -> Option<(&'static str, ruff_text_size::TextRange)> {
    use ast::visitor::Visitor;

    if feature_version < 0 || feature_version >= 6 {
        return None;
    }

    #[derive(Default)]
    struct Finder {
        minor: i64,
        error: Option<(&'static str, ruff_text_size::TextRange)>,
    }

    impl Finder {
        fn record(&mut self, message: &'static str, range: ruff_text_size::TextRange) {
            if self
                .error
                .is_none_or(|(_, current)| range.start() < current.start())
            {
                self.error = Some((message, range));
            }
        }
    }

    impl<'a> Visitor<'a> for Finder {
        fn visit_stmt(&mut self, statement: &'a ast::Stmt) {
            if self.minor < 5 {
                match statement {
                    ast::Stmt::FunctionDef(node) if node.is_async => self.record(
                        "Async functions are only supported in Python 3.5 and greater",
                        ruff_text_size::TextRange::empty(node.range.end()),
                    ),
                    ast::Stmt::For(node) if node.is_async => self.record(
                        "Async for loops are only supported in Python 3.5 and greater",
                        ruff_text_size::TextRange::empty(node.range.end()),
                    ),
                    ast::Stmt::With(node) if node.is_async => self.record(
                        "Async with statements are only supported in Python 3.5 and greater",
                        ruff_text_size::TextRange::empty(node.range.end()),
                    ),
                    _ => {}
                }
            }
            ast::visitor::walk_stmt(self, statement);
        }

        fn visit_expr(&mut self, expression: &'a ast::Expr) {
            if self.minor < 5 {
                match expression {
                    ast::Expr::Await(node) => self.record(
                        "Await expressions are only supported in Python 3.5 and greater",
                        ruff_text_size::TextRange::empty(node.range.end()),
                    ),
                    ast::Expr::BinOp(node) if node.op == ast::Operator::MatMult => self.record(
                        "The '@' operator is only supported in Python 3.5 and greater",
                        ruff_text_size::TextRange::empty(node.range.end()),
                    ),
                    _ => {}
                }
            }
            if self.minor < 6 {
                let async_comprehension = match expression {
                    ast::Expr::ListComp(node) => Some(&node.generators),
                    ast::Expr::SetComp(node) => Some(&node.generators),
                    ast::Expr::DictComp(node) => Some(&node.generators),
                    ast::Expr::Generator(node) => Some(&node.generators),
                    _ => None,
                };
                if let Some(generators) = async_comprehension
                    && let Some(generator) = generators.iter().find(|generator| generator.is_async)
                {
                    self.record(
                        "Async comprehensions are only supported in Python 3.6 and greater",
                        ruff_text_size::TextRange::empty(generator.range.end()),
                    );
                }
            }
            ast::visitor::walk_expr(self, expression);
        }
    }

    let mut finder = Finder {
        minor: feature_version,
        error: None,
    };
    match module {
        ast::Mod::Module(module) => finder.visit_body(&module.body),
        ast::Mod::Expression(expression) => finder.visit_expr(&expression.body),
    }

    if feature_version < 6 {
        for token in tokens.iter() {
            let (kind, range) = token.as_tuple();
            if feature_version < 5 && kind == ast::token::TokenKind::AtEqual {
                finder.record(
                    "The '@' operator is only supported in Python 3.5 and greater",
                    range,
                );
            }
            if matches!(
                kind,
                ast::token::TokenKind::Int
                    | ast::token::TokenKind::Float
                    | ast::token::TokenKind::Complex
            ) && source[range.start().to_usize()..range.end().to_usize()].contains('_')
            {
                finder.record(
                    "Underscores in numeric literals are only supported in Python 3.6 and greater",
                    range,
                );
            }
        }
    }
    finder.error
}

/// CPython `_PyPegen_new_identifier` refuses a lexer name whose NFKC form is
/// one of the three constant keywords.  Ruff already performs that same NFKC
/// conversion when it builds the AST, but (unlike CPython) leaves the result
/// as an identifier.  Inspecting only `Name` tokens keeps this a parser check:
/// a literal `None`/`True`/`False` token is, of course, valid.
fn validate_parser_identifiers(source: &str, tokens: &ast::token::Tokens) -> AstResult<()> {
    for token in tokens.iter() {
        if token.kind() != ast::token::TokenKind::Name {
            continue;
        }
        let (_, range) = token.as_tuple();
        let spelling = &source[range.start().to_usize()..range.end().to_usize()];
        if spelling.is_ascii() {
            continue;
        }
        let normalized = rustpython_unicode::normalize(
            rustpython_unicode::NormalizeForm::Nfkc,
            Wtf8::new(spelling),
        );
        let normalized = normalized.to_string_lossy();
        if matches!(normalized.as_ref(), "None" | "True" | "False") {
            return Err(crate::PyError::value_error(format!(
                "identifier field can't represent '{normalized}' constant"
            )));
        }
    }
    Ok(())
}

/// Ruff currently records `TypeParamDefault` for `T = int`, but not for the
/// sibling `*Ts = int` and `**P = int` spellings.  CPython 3.14 applies the
/// same 3.13 grammar boundary to all three, so fill only that parser metadata
/// gap after Ruff has produced the native tree.
fn has_variadic_type_param_default(module: &ast::Mod) -> bool {
    use ast::visitor::Visitor;

    #[derive(Default)]
    struct Finder {
        found: bool,
    }

    impl<'a> Visitor<'a> for Finder {
        fn visit_type_param(&mut self, type_param: &'a ast::TypeParam) {
            self.found |= match type_param {
                ast::TypeParam::TypeVar(_) => false,
                ast::TypeParam::TypeVarTuple(node) => node.default.is_some(),
                ast::TypeParam::ParamSpec(node) => node.default.is_some(),
            };
            if !self.found {
                ast::visitor::walk_type_param(self, type_param);
            }
        }
    }

    let mut finder = Finder::default();
    if let ast::Mod::Module(module) = module {
        finder.visit_body(&module.body);
    }
    finder.found
}

/// Convert an interpreter-level `_ast` tree back into Ruff's compiler AST and
/// compile it.  This is the reverse of `Converter`, corresponding to PyPy's
/// generated `ast_from_object` boundary.
pub fn compile_object(
    mut object: PyObjectRef,
    filename: &str,
    warning_filename: &Wtf8,
    mode: crate::compile::Mode,
    opts: crate::compile::CompileOpts,
) -> AstResult<crate::compile::CodeObject> {
    let ast_module = pyre_object::with_roots!(object => crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("_ast"),
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    ))?;
    let mut converter = ObjectConverter {
        ast_module,
        depth: 0,
        carried_ignores: Vec::new(),
        line_len: 0,
    };
    pyre_object::with_roots!(object => converter.validate_module_mode(object, mode))?;
    // The compiler reads a node's position out of the source text the range
    // indexes, so a tree that came from objects needs a text to index.  Stand
    // one up whose lines are wide enough for every column the tree names; the
    // characters are never read, only counted.
    let text = pyre_object::with_roots!(object => converter.synthetic_source(object))?;
    let module = converter.module(object)?;
    // compiling.py:73 — the tree is walked before it reaches the compiler.
    crate::astcompiler::validate::validate_ast(&module)?;
    let source_file =
        rustpython_compiler::core::SourceFileBuilder::new(filename, text.clone()).finish();
    crate::syntax_warnings::compile_ast_with_codegen_warnings(
        module,
        source_file,
        &text,
        warning_filename,
        mode,
        opts,
    )
    .map_err(|error| match error {
        crate::syntax_warnings::SourceCompileError::Compile(error) => {
            crate::PyError::syntax_error(error.to_string())
        }
        crate::syntax_warnings::SourceCompileError::Warning(error) => error,
    })
}

/// CPython 3.14 `_PyCompile_AstPreprocess`: apply the same AST preprocessing
/// pass used immediately before bytecode generation.  `syntax_check_only`
/// is true for plain `PyCF_ONLY_AST` and false for `PyCF_OPTIMIZED_AST`.
fn preprocess_module(
    module: &mut ast::Mod,
    mode: crate::compile::Mode,
    opts: crate::compile::CompileOpts,
    syntax_check_only: bool,
) {
    let future_annotations = opts
        .future_features
        .contains(crate::compile::CodeFlags::FUTURE_ANNOTATIONS)
        || rustpython_compiler::codegen::preprocess::has_future_annotations(module);
    if matches!(mode, crate::compile::Mode::Single)
        && let ast::Mod::Module(module) = module
    {
        rustpython_compiler::codegen::preprocess::preprocess_statements(
            &mut module.body,
            opts.optimize,
            future_annotations,
            syntax_check_only,
        );
    } else {
        rustpython_compiler::codegen::preprocess::preprocess_mod(
            module,
            opts.optimize,
            future_annotations,
            syntax_check_only,
        );
    }
}

/// CPython 3.14 `compile(ast_obj, ..., flags=PyCF_*_AST)`: convert and
/// validate the public tree, preprocess it, then convert the native tree back
/// to a fresh public `_ast` object.
pub fn preprocess_object_to_object(
    mut object: PyObjectRef,
    source: &str,
    mode: crate::compile::Mode,
    opts: crate::compile::CompileOpts,
    syntax_check_only: bool,
) -> crate::PyResult {
    let mut ast_module = pyre_object::with_roots!(object => crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("_ast"),
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    ))?;
    let mut converter = ObjectConverter {
        ast_module,
        depth: 0,
        carried_ignores: Vec::new(),
        line_len: 0,
    };
    // Both directions have to read positions out of the same text, and a tree
    // handed in as objects arrives without one.  The synthetic source stands
    // in for it so the round trip gives every node back the line and column it
    // came with.
    let synthetic =
        pyre_object::with_roots!(ast_module, object => converter.synthetic_source(object))?;
    let mut module = pyre_object::with_roots!(ast_module => converter.module(object))?;
    crate::astcompiler::validate::validate_ast(&module)?;
    preprocess_module(&mut module, mode, opts, syntax_check_only);
    let source = if source.is_empty() {
        &synthetic
    } else {
        source
    };
    module_to_object(
        module,
        source,
        mode,
        ast_module,
        &converter.carried_ignores,
        false,
    )
}

struct ObjectConverter {
    ast_module: PyObjectRef,
    depth: usize,
    /// The `TypeIgnore`s the incoming `Module` carried. The compiler AST has
    /// no field for them, so they ride here and are republished unchanged.
    carried_ignores: Vec<super::type_comments::TypeComment>,
    /// Characters on each line of the synthetic source, excluding the newline.
    /// Zero until `synthetic_source` has measured the tree.
    line_len: usize,
}

impl ObjectConverter {
    /// CPython `_PyAST_CheckMode`: an AST root supplied to `compile()` must
    /// match the requested parser start rule.  PyPy's generated AST converter
    /// owns the same boundary, while Ruff's code generator accepts a `Module`
    /// with `Mode::Eval` and would otherwise silently produce a code object.
    fn validate_module_mode(
        &self,
        mut object: PyObjectRef,
        mode: crate::compile::Mode,
    ) -> AstResult<()> {
        let expected = match mode {
            crate::compile::Mode::Exec | crate::compile::Mode::BlockExpr => "Module",
            crate::compile::Mode::Eval => "Expression",
            crate::compile::Mode::Single => "Interactive",
        };
        if pyre_object::with_roots!(object => self.is_node(object, expected))? {
            return Ok(());
        }
        for actual in ["Module", "Expression", "Interactive", "FunctionType"] {
            if pyre_object::with_roots!(object => self.is_node(object, actual))? {
                return Err(crate::PyError::type_error(format!(
                    "expected {expected} node, got {actual}"
                )));
            }
        }
        Ok(())
    }

    /// Blank text with a line for every line the tree names and columns for
    /// the widest of them, so that `location` can turn a node's `lineno` and
    /// `col_offset` into an offset the compiler can map back.
    fn synthetic_source(&mut self, object: PyObjectRef) -> AstResult<String> {
        let mut extent = (0usize, 0usize);
        self.scan_extent(object, &mut extent)?;
        let (max_line, max_col) = extent;
        if max_line == 0 {
            return Ok(String::new());
        }
        self.line_len = max_col.saturating_add(1);
        let mut source = String::new();
        let width = self.line_len.saturating_add(1);
        source
            .try_reserve(width.saturating_mul(max_line))
            .map_err(|_| crate::PyError::memory_error("source location is too large"))?;
        for _ in 0..max_line {
            source.extend(core::iter::repeat_n(' ', self.line_len));
            source.push('\n');
        }
        Ok(source)
    }

    /// Widen `extent` to the last line and rightmost column any node in the
    /// tree claims.  The walk is over `_fields` rather than the typed
    /// conversion below, because it runs before anything has been validated
    /// and must not reject a tree the converter would go on to accept.
    fn scan_extent(&mut self, object: PyObjectRef, extent: &mut (usize, usize)) -> AstResult<()> {
        if unsafe { pyre_object::is_list(object) } {
            let items = unsafe {
                pyre_object::w_list_items_copy_as_vec_mode(
                    object,
                    majit_metainterp::jit::we_are_jitted(),
                )
            };
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&items);
            for index in 0..items.len() {
                let item = pyre_object::gc_roots::shadow_stack_get(base + index);
                self.recurse(|this| this.scan_extent(item, extent))?;
            }
            return Ok(());
        }
        if unsafe { pyre_object::is_tuple(object) } {
            let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(object) };
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::pin_roots(&items);
            for index in 0..items.len() {
                let item = pyre_object::gc_roots::shadow_stack_get(base + index);
                self.recurse(|this| this.scan_extent(item, extent))?;
            }
            return Ok(());
        }
        let _node_roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(object);
        if !self.is_node(pyre_object::gc_roots::shadow_stack_get(obj_slot), "AST")? {
            return Ok(());
        }
        for (line_field, column_field) in
            [("lineno", "col_offset"), ("end_lineno", "end_col_offset")]
        {
            if let Some(value) = self.optional_field(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                line_field,
            )? && let Ok(line) = self.obj_to_int(value)
                && line > 0
            {
                extent.0 = extent.0.max(line as usize);
            }
            if let Some(value) = self.optional_field(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                column_field,
            )? && let Ok(column) = self.obj_to_int(value)
                && column > 0
            {
                extent.1 = extent.1.max(column as usize);
            }
        }
        let Some(fields) =
            self.optional_field(pyre_object::gc_roots::shadow_stack_get(obj_slot), "_fields")?
        else {
            return Ok(());
        };
        if !unsafe { pyre_object::is_tuple(fields) } {
            return Ok(());
        }
        let names = unsafe { pyre_object::w_tuple_items_copy_as_vec(fields) };
        let name_base = pyre_object::gc_roots::pin_roots(&names);
        for index in 0..names.len() {
            let name = pyre_object::gc_roots::shadow_stack_get(name_base + index);
            if !unsafe { pyre_object::is_str(name) } {
                continue;
            }
            // A lone surrogate names no attribute this scan can reach.
            let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(name) }) else {
                continue;
            };
            let name = name.to_string();
            if let Some(value) =
                self.optional_field(pyre_object::gc_roots::shadow_stack_get(obj_slot), &name)?
            {
                self.recurse(|this| this.scan_extent(value, extent))?;
            }
        }
        Ok(())
    }

    /// Offset of a one-based line and a zero-based column in the synthetic
    /// source, clamped to the line it names.
    fn offset_of(&self, line: i64, column: i64) -> u32 {
        let line = usize::try_from(line).unwrap_or(1).max(1);
        let column = usize::try_from(column).unwrap_or(0).min(self.line_len);
        let width = self.line_len.saturating_add(1);
        let offset = (line - 1).saturating_mul(width).saturating_add(column);
        u32::try_from(offset).unwrap_or(u32::MAX)
    }

    fn recurse<T>(&mut self, f: impl FnOnce(&mut Self) -> AstResult<T>) -> AstResult<T> {
        // PyPy's generated ast_from_object calls space.getexecutioncontext()
        // recursion guards around nested ASDL nodes.  Keep the state on this
        // conversion, never in TLS or a global side table.
        if self.depth >= 200 {
            return Err(crate::PyError::recursion_error(
                "maximum recursion depth exceeded while traversing AST node",
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn is_node(&self, mut object: PyObjectRef, name: &str) -> AstResult<bool> {
        let ty = pyre_object::with_roots!(object => crate::baseobjspace::getattr_str(self.ast_module, name))?;
        Ok(unsafe { crate::baseobjspace::isinstance_w(object, ty) })
    }

    fn field(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<PyObjectRef> {
        crate::baseobjspace::getattr_str(object, field).map_err(|_| {
            crate::PyError::type_error(format!("required field {field:?} missing from {node}"))
        })
    }

    /// PyPy `asdl_py.py:get_field_extractor` checks a present required field
    /// for `None` after fetching it and before converting its ASDL value.
    fn required_field(
        &self,
        mut object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<PyObjectRef> {
        let value = pyre_object::with_roots!(object => self.field(object, field, node))?;
        if unsafe { pyre_object::is_none(value) } {
            return Err(crate::PyError::value_error(format!(
                "field '{field}' is required for {}",
                class_name(object)
            )));
        }
        Ok(value)
    }

    fn optional_field(&self, object: PyObjectRef, field: &str) -> AstResult<Option<PyObjectRef>> {
        match crate::baseobjspace::getattr_str(object, field) {
            Ok(value) if unsafe { pyre_object::is_none(value) } => Ok(None),
            Ok(value) => Ok(Some(value)),
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn list(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<Vec<PyObjectRef>> {
        let value = self.field(object, field, node)?;
        if !unsafe { pyre_object::is_list(value) } {
            return Err(crate::PyError::type_error(format!(
                "{node} field {field:?} must be a list, not a {}",
                class_name(value)
            )));
        }
        Ok(unsafe {
            pyre_object::w_list_items_copy_as_vec_mode(
                value,
                majit_metainterp::jit::we_are_jitted(),
            )
        })
    }

    fn with_list<T>(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
        mut each: impl FnMut(&mut Self, PyObjectRef) -> AstResult<T>,
    ) -> AstResult<Vec<T>> {
        let items = self.list(object, field, node)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&items);
        let mut out = Vec::with_capacity(items.len());
        for index in 0..items.len() {
            let value = pyre_object::gc_roots::shadow_stack_get(base + index);
            out.push(each(self, value)?);
        }
        Ok(out)
    }

    fn string(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<String> {
        let value = self.field(object, field, node)?;
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST identifier must be of type str",
            ));
        }
        Ok(utf8_only(value)?.to_string())
    }

    /// How `%R` names the value an error rejected.
    fn repr(&self, object: PyObjectRef) -> AstResult<rustpython_wtf8::Wtf8Buf> {
        unsafe { crate::display::py_repr_wtf8(object) }
    }

    /// `obj_to_int` (ast.py) — an integer field takes an `int`, or an
    /// instance of a subclass of one.  Nothing else is asked for `__index__`.
    fn obj_to_int(&self, value: PyObjectRef) -> AstResult<i64> {
        if !unsafe { crate::baseobjspace::isinstance_int_w(value) } {
            return Err(crate::PyError::value_error(crate::display::wtf8_format!(
                "invalid integer value: ",
                self.repr(value)?
            )));
        }
        crate::builtins::space_index_w(value)
    }

    /// The source range an object carries as attributes, placed in the
    /// synthetic source.  `lineno` and `col_offset` are required; the two end
    /// fields are optional and fall back to the start, which is what a node
    /// built by hand without them describes.
    fn location(
        &self,
        mut object: PyObjectRef,
        node: &str,
    ) -> AstResult<ruff_text_size::TextRange> {
        let line = pyre_object::with_roots!(object => self.int_field(object, "lineno", node))?;
        let column =
            pyre_object::with_roots!(object => self.int_field(object, "col_offset", node))?;
        let end_line = match pyre_object::with_roots!(object => self.optional_field(object, "end_lineno"))?
        {
            Some(value) => pyre_object::with_roots!(object => self.obj_to_int(value))?,
            None => line,
        };
        let end_column = match self.optional_field(object, "end_col_offset")? {
            Some(value) => self.obj_to_int(value)?,
            None => column,
        };

        // [3.14-spec] CPython `VALIDATE_POSITIONS` rejects these ranges before
        // code generation.  PyPy `AstValidator` has no corresponding check,
        // but the invalid range and ValueError are observable at compile().
        if line > end_line {
            return Err(crate::PyError::value_error(format!(
                "AST node line range ({line}, {end_line}) is not valid"
            )));
        }
        if (line < 0 && end_line != line) || (column < 0 && column != end_column) {
            return Err(crate::PyError::value_error(format!(
                "AST node column range ({column}, {end_column}) for line range ({line}, {end_line}) is not valid"
            )));
        }
        if line == end_line && column > end_column {
            return Err(crate::PyError::value_error(format!(
                "line {line}, column {column}-{end_column} is not a valid range"
            )));
        }

        let start = self.offset_of(line, column);
        // Zero and negative line numbers are accepted when the paired end
        // position satisfies the checks above.  The synthetic source maps all
        // of them to its first line, so preserve that valid input even when
        // the mapping makes its byte offsets coincide or reverse.
        let end = self.offset_of(end_line, end_column).max(start);
        Ok(ruff_text_size::TextRange::new(start.into(), end.into()))
    }

    /// `Module.type_ignores` (ast.py) holds the `# type: ignore` comments a
    /// `type_comments=True` parse collected.  The compiler AST has nowhere to
    /// keep them, so they are carried beside it and handed back unchanged; a
    /// list holding anything but a `TypeIgnore` is reported here.
    fn type_ignores(
        &self,
        object: PyObjectRef,
    ) -> AstResult<Vec<super::type_comments::TypeComment>> {
        let value = match crate::baseobjspace::getattr_str(object, "type_ignores") {
            Ok(value) => value,
            // An unset field stands for the empty list a parse without type
            // comments produces.
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => {
                return Ok(Vec::new());
            }
            Err(error) => return Err(error),
        };
        if !unsafe { pyre_object::is_list(value) } {
            return Err(crate::PyError::type_error(format!(
                "Module field \"type_ignores\" must be a list, not a {}",
                class_name(value)
            )));
        }
        let mut out = Vec::new();
        for mut item in unsafe {
            pyre_object::w_list_items_copy_as_vec_mode(
                value,
                majit_metainterp::jit::we_are_jitted(),
            )
        } {
            if unsafe { pyre_object::is_none(item) } {
                continue;
            }
            if pyre_object::with_roots!(item => self.is_node(item, "TypeIgnore"))? {
                out.push(super::type_comments::TypeComment::new(
                    pyre_object::with_roots!(item => self.int_field(item, "lineno", "TypeIgnore"))?
                        as u32,
                    self.string(item, "tag", "TypeIgnore")?,
                ));
                continue;
            }
            return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of type_ignore, but got ",
                self.repr(item)?
            )));
        }
        Ok(out)
    }

    fn module(&mut self, mut object: PyObjectRef) -> AstResult<ast::Mod> {
        let node = if pyre_object::with_roots!(object => self.is_node(object, "Module"))? {
            self.carried_ignores = pyre_object::with_roots!(object => self.type_ignores(object))?;
            "Module"
        } else if pyre_object::with_roots!(object => self.is_node(object, "Interactive"))? {
            "Interactive"
        } else if pyre_object::with_roots!(object => self.is_node(object, "Expression"))? {
            let body = self.required_field(object, "body", "Expression")?;
            return Ok(ast::Mod::Expression(ast::ModExpression {
                node_index: Default::default(),
                range: Default::default(),
                body: Box::new(self.recurse(|this| this.expr(body))?),
            }));
        } else {
            return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of mod, but got ",
                self.repr(object)?
            )));
        };
        Ok(ast::Mod::Module(ast::ModModule {
            node_index: Default::default(),
            range: Default::default(),
            body: self.body(object, "body", node)?,
            runtime_body: None,
        }))
    }

    fn stmt(&mut self, mut object: PyObjectRef) -> AstResult<ast::Stmt> {
        let range = pyre_object::with_roots!(object => self.location(object, "stmt"))?;
        if pyre_object::with_roots!(object => self.is_node(object, "FunctionDef"))?
            || pyre_object::with_roots!(object => self.is_node(object, "AsyncFunctionDef"))?
        {
            let is_async =
                pyre_object::with_roots!(object => self.is_node(object, "AsyncFunctionDef"))?;
            let node = if is_async {
                "AsyncFunctionDef"
            } else {
                "FunctionDef"
            };
            let name = pyre_object::with_roots!(object => self.identifier(object, "name", node))?;
            let args =
                pyre_object::with_roots!(object => self.required_field(object, "args", node))?;
            let parameters = Box::new(self.recurse(|this| this.parameters(args))?);
            let body = pyre_object::with_roots!(object => self.body(object, "body", node))?;
            let decorator_list = pyre_object::with_roots!(object => self.decorators(object, node))?;
            let returns = pyre_object::with_roots!(object => self.opt_expr(object, "returns"))?;
            let type_params = pyre_object::with_roots!(object => self.type_params(object, node))?;
            Ok(ast::Stmt::FunctionDef(ast::StmtFunctionDef {
                node_index: Default::default(),
                range,
                is_async,
                decorator_list,
                name,
                type_params,
                parameters,
                returns,
                body,
                runtime_decorator_list: None,
                runtime_type_comment: self.opt_type_comment(object)?,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Pass"))? {
            Ok(ast::Stmt::Pass(ast::StmtPass {
                node_index: Default::default(),
                range,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Expr"))? {
            let value = self.required_field(object, "value", "Expr")?;
            Ok(ast::Stmt::Expr(ast::StmtExpr {
                node_index: Default::default(),
                range,
                value: Box::new(self.recurse(|this| this.expr(value))?),
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Return"))? {
            let value = self.optional_field(object, "value")?;
            Ok(ast::Stmt::Return(ast::StmtReturn {
                node_index: Default::default(),
                range,
                value: value
                    .map(|value| self.recurse(|this| this.expr(value)).map(Box::new))
                    .transpose()?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Assign"))? {
            let targets = pyre_object::with_roots!(object => self
                .list(object, "targets", "Assign"))?
            .into_iter()
            .map(|value| {
                self.require_node(value, "expression")?;
                self.recurse(|this| this.expr(value))
            })
            .collect::<Result<Vec<_>, _>>()?;
            let value =
                pyre_object::with_roots!(object => self.required_field(object, "value", "Assign"))?;
            Ok(ast::Stmt::Assign(ast::StmtAssign {
                node_index: Default::default(),
                range,
                targets,
                value: Box::new(self.recurse(|this| this.expr(value))?),
                runtime_targets: None,
                runtime_type_comment: self.opt_type_comment(object)?,
                runtime_type_comment_bytes: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "ClassDef"))? {
            let name =
                pyre_object::with_roots!(object => self.identifier(object, "name", "ClassDef"))?;
            let bases =
                pyre_object::with_roots!(object => self.exprs(object, "bases", "ClassDef"))?;
            let keywords = pyre_object::with_roots!(object => self
                .list(object, "keywords", "ClassDef"))?
            .into_iter()
            .map(|keyword| self.recurse(|this| this.keyword(keyword)))
            .collect::<Result<Vec<_>, _>>()?;
            let body = pyre_object::with_roots!(object => self.body(object, "body", "ClassDef"))?;
            let decorator_list =
                pyre_object::with_roots!(object => self.decorators(object, "ClassDef"))?;
            let type_params = self.type_params(object, "ClassDef")?;
            // An absent argument list and an empty one are different trees, and
            // only the former elides the parentheses.
            let arguments = if bases.is_empty() && keywords.is_empty() {
                None
            } else {
                Some(Box::new(ast::Arguments {
                    node_index: Default::default(),
                    range,
                    args: bases.into_boxed_slice(),
                    keywords: keywords.into(),
                    runtime_args: None,
                    runtime_bases: None,
                }))
            };
            Ok(ast::Stmt::ClassDef(ast::StmtClassDef {
                node_index: Default::default(),
                range,
                decorator_list,
                name,
                type_params,
                arguments,
                body,
                runtime_decorator_list: None,
                runtime_body: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Delete"))? {
            Ok(ast::Stmt::Delete(ast::StmtDelete {
                node_index: Default::default(),
                range,
                targets: self.exprs(object, "targets", "Delete")?,
                runtime_targets: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "TypeAlias"))? {
            let name =
                pyre_object::with_roots!(object => self.req_expr(object, "name", "TypeAlias"))?;
            let type_params =
                pyre_object::with_roots!(object => self.type_params(object, "TypeAlias"))?;
            let value = self.req_expr(object, "value", "TypeAlias")?;
            Ok(ast::Stmt::TypeAlias(ast::StmtTypeAlias {
                node_index: Default::default(),
                range,
                name,
                type_params,
                value,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "AugAssign"))? {
            let target =
                pyre_object::with_roots!(object => self.req_expr(object, "target", "AugAssign"))?;
            let mut op =
                pyre_object::with_roots!(object => self.required_field(object, "op", "AugAssign"))?;
            let value =
                pyre_object::with_roots!(op => self.req_expr(object, "value", "AugAssign"))?;
            Ok(ast::Stmt::AugAssign(ast::StmtAugAssign {
                node_index: Default::default(),
                range,
                target,
                op: self.operator(op)?,
                value,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "AnnAssign"))? {
            let target =
                pyre_object::with_roots!(object => self.req_expr(object, "target", "AnnAssign"))?;
            let annotation = pyre_object::with_roots!(object => self.req_expr(object, "annotation", "AnnAssign"))?;
            let value = pyre_object::with_roots!(object => self.opt_expr(object, "value"))?;
            let simple = self.int_field(object, "simple", "AnnAssign")?;
            Ok(ast::Stmt::AnnAssign(ast::StmtAnnAssign {
                node_index: Default::default(),
                range,
                target,
                annotation,
                value,
                simple: simple != 0,
                runtime_simple: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "For"))?
            || pyre_object::with_roots!(object => self.is_node(object, "AsyncFor"))?
        {
            let is_async = pyre_object::with_roots!(object => self.is_node(object, "AsyncFor"))?;
            let node = if is_async { "AsyncFor" } else { "For" };
            let target = pyre_object::with_roots!(object => self.req_expr(object, "target", node))?;
            let iter = pyre_object::with_roots!(object => self.req_expr(object, "iter", node))?;
            let body = pyre_object::with_roots!(object => self.body(object, "body", node))?;
            let orelse = pyre_object::with_roots!(object => self.body(object, "orelse", node))?;
            Ok(ast::Stmt::For(ast::StmtFor {
                node_index: Default::default(),
                range,
                is_async,
                target,
                iter,
                body,
                orelse,
                runtime_type_comment: self.opt_type_comment(object)?,
                runtime_type_comment_bytes: None,
                runtime_body: None,
                runtime_orelse: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "While"))? {
            let test = pyre_object::with_roots!(object => self.req_expr(object, "test", "While"))?;
            let body = pyre_object::with_roots!(object => self.body(object, "body", "While"))?;
            let orelse = self.body(object, "orelse", "While")?;
            Ok(ast::Stmt::While(ast::StmtWhile {
                node_index: Default::default(),
                range,
                test,
                body,
                orelse,
                runtime_body: None,
                runtime_orelse: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "If"))? {
            let test = pyre_object::with_roots!(object => self.req_expr(object, "test", "If"))?;
            let body = pyre_object::with_roots!(object => self.body(object, "body", "If"))?;
            // `If` carries its alternatives as a nested `orelse`, while the
            // compiler AST keeps them in one flat clause list.  An `elif` and an
            // `else` holding a single `if` are indistinguishable here, exactly as
            // they are to the parser, so both flatten the same way.
            let mut elif_else_clauses = Vec::new();
            let mut orelse = self.list(object, "orelse", "If")?;
            while !orelse.is_empty() {
                if orelse.len() == 1 && self.is_node(orelse[0], "If")? {
                    let mut nested = orelse[0];
                    let clause_test =
                        pyre_object::with_roots!(nested => self.req_expr(nested, "test", "If"))?;
                    let clause_body =
                        pyre_object::with_roots!(nested => self.body(nested, "body", "If"))?;
                    elif_else_clauses.push(ast::ElifElseClause {
                        range: pyre_object::with_roots!(nested => self.location(nested, "If"))?,
                        node_index: Default::default(),
                        test: Some(*clause_test),
                        body: clause_body,
                        runtime_body: None,
                        runtime_orelse: None,
                    });
                    orelse = self.list(nested, "orelse", "If")?;
                } else {
                    let values = std::mem::take(&mut orelse);
                    let clause_body = values
                        .into_iter()
                        .map(|value| {
                            self.require_node(value, "statement")?;
                            self.recurse(|this| this.stmt(value))
                        })
                        .collect::<Result<ThinVec<_>, _>>()?;
                    elif_else_clauses.push(ast::ElifElseClause {
                        range,
                        node_index: Default::default(),
                        test: None,
                        body: clause_body,
                        runtime_body: None,
                        runtime_orelse: None,
                    });
                }
            }
            Ok(ast::Stmt::If(ast::StmtIf {
                node_index: Default::default(),
                range,
                test,
                body,
                elif_else_clauses,
                runtime_body: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "With"))?
            || pyre_object::with_roots!(object => self.is_node(object, "AsyncWith"))?
        {
            let is_async = pyre_object::with_roots!(object => self.is_node(object, "AsyncWith"))?;
            let node = if is_async { "AsyncWith" } else { "With" };
            let items = pyre_object::with_roots!(object => self
                .list(object, "items", node))?
            .into_iter()
            .map(|item| self.recurse(|this| this.with_item(item)))
            .collect::<Result<Vec<_>, _>>()?;
            let body = pyre_object::with_roots!(object => self.body(object, "body", node))?;
            Ok(ast::Stmt::With(ast::StmtWith {
                node_index: Default::default(),
                range,
                is_async,
                items,
                body,
                runtime_type_comment: self.opt_type_comment(object)?,
                runtime_type_comment_bytes: None,
                runtime_body: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Raise"))? {
            Ok(ast::Stmt::Raise(ast::StmtRaise {
                node_index: Default::default(),
                range,
                exc: pyre_object::with_roots!(object => self.opt_expr(object, "exc"))?,
                cause: self.opt_expr(object, "cause")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Try"))?
            || pyre_object::with_roots!(object => self.is_node(object, "TryStar"))?
        {
            let is_star = pyre_object::with_roots!(object => self.is_node(object, "TryStar"))?;
            let node = if is_star { "TryStar" } else { "Try" };
            let body = pyre_object::with_roots!(object => self.body(object, "body", node))?;
            let handlers = pyre_object::with_roots!(object => self
                .list(object, "handlers", node))?
            .into_iter()
            .map(|handler| self.recurse(|this| this.handler(handler)))
            .collect::<Result<Vec<_>, _>>()?;
            let orelse = pyre_object::with_roots!(object => self.body(object, "orelse", node))?;
            let finalbody = self.body(object, "finalbody", node)?;
            Ok(ast::Stmt::Try(ast::StmtTry {
                node_index: Default::default(),
                range,
                body,
                handlers,
                orelse,
                finalbody,
                is_star,
                runtime_body: None,
                runtime_handlers: None,
                runtime_orelse: None,
                runtime_finalbody: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Assert"))? {
            let test = pyre_object::with_roots!(object => self.req_expr(object, "test", "Assert"))?;
            let msg = self.opt_expr(object, "msg")?;
            Ok(ast::Stmt::Assert(ast::StmtAssert {
                node_index: Default::default(),
                range,
                test,
                msg,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Import"))? {
            Ok(ast::Stmt::Import(ast::StmtImport {
                node_index: Default::default(),
                range,
                names: self.aliases(object, "Import")?,
                is_lazy: false,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "ImportFrom"))? {
            let module = pyre_object::with_roots!(object => self.opt_identifier(object, "module"))?;
            let names = pyre_object::with_roots!(object => self.aliases(object, "ImportFrom"))?;
            // `level` is optional on a hand-built node and defaults to absolute.
            let level = match self.optional_field(object, "level")? {
                Some(value) => self.obj_to_int(value)?,
                None => 0,
            };
            if level < 0 {
                return Err(crate::PyError::value_error("Negative ImportFrom level"));
            }
            // The field is read as a Python index, so a value past the range a
            // level is stored in has to stop here rather than wrap and turn a
            // relative import into an absolute one.
            let level = u32::try_from(level).map_err(|_| {
                crate::PyError::overflow_error("Python int too large to convert to C int")
            })?;
            Ok(ast::Stmt::ImportFrom(ast::StmtImportFrom {
                node_index: Default::default(),
                range,
                module,
                names,
                level,
                is_lazy: false,
                runtime_level: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Global"))? {
            Ok(ast::Stmt::Global(ast::StmtGlobal {
                node_index: Default::default(),
                range,
                names: self.identifiers(object, "names", "Global")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Nonlocal"))? {
            Ok(ast::Stmt::Nonlocal(ast::StmtNonlocal {
                node_index: Default::default(),
                range,
                names: self.identifiers(object, "names", "Nonlocal")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Break"))? {
            Ok(ast::Stmt::Break(ast::StmtBreak {
                node_index: Default::default(),
                range,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Continue"))? {
            Ok(ast::Stmt::Continue(ast::StmtContinue {
                node_index: Default::default(),
                range,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Match"))? {
            let subject =
                pyre_object::with_roots!(object => self.req_expr(object, "subject", "Match"))?;
            let cases = self
                .list(object, "cases", "Match")?
                .into_iter()
                .map(|case| self.recurse(|this| this.match_case(case)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Stmt::Match(ast::StmtMatch {
                node_index: Default::default(),
                range,
                subject,
                cases,
            }))
        } else {
            Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of stmt, but got ",
                self.repr(object)?
            )))
        }
    }

    fn parameters(&mut self, mut object: PyObjectRef) -> AstResult<ast::Parameters> {
        let posonlyargs =
            pyre_object::with_roots!(object => self.parameter_list(object, "posonlyargs"))?;
        let args = pyre_object::with_roots!(object => self.parameter_list(object, "args"))?;
        let kwonlyargs =
            pyre_object::with_roots!(object => self.parameter_list(object, "kwonlyargs"))?;
        let vararg = pyre_object::with_roots!(object => self.opt_parameter(object, "vararg"))?;
        let kwarg = pyre_object::with_roots!(object => self.opt_parameter(object, "kwarg"))?;
        crate::astcompiler::validate::validate_parameter_annotations(
            &posonlyargs,
            &args,
            vararg.as_deref(),
            &kwonlyargs,
            kwarg.as_deref(),
        )?;
        // `defaults` covers the tail of posonlyargs ++ args, while `kw_defaults`
        // runs alongside kwonlyargs with a hole for every parameter that has
        // none.  The compiler AST carries each default on its own parameter, so
        // both lists are distributed here.
        let defaults =
            pyre_object::with_roots!(object => self.exprs(object, "defaults", "arguments"))?;
        let kw_defaults = self.list(object, "kw_defaults", "arguments")?;
        if kw_defaults.len() != kwonlyargs.len() {
            return Err(crate::PyError::value_error(
                "length of kwonlyargs is not the same as kw_defaults on arguments",
            ));
        }
        let mut positional: Vec<ast::Parameter> = posonlyargs;
        let posonly_count = positional.len();
        positional.extend(args);
        if defaults.len() > positional.len() {
            return Err(crate::PyError::value_error(
                "more positional defaults than args on arguments",
            ));
        }
        let first_default = positional.len() - defaults.len();
        let mut positional: Vec<ast::ParameterWithDefault> = positional
            .into_iter()
            .map(|parameter| ast::ParameterWithDefault {
                range: Default::default(),
                node_index: Default::default(),
                parameter,
                default: None,
            })
            .collect();
        for (offset, default) in defaults.into_iter().enumerate() {
            positional[first_default + offset].default = Some(Box::new(default));
        }
        let args = positional.split_off(posonly_count);
        let posonlyargs = positional;
        let mut kwonly: Vec<ast::ParameterWithDefault> = kwonlyargs
            .into_iter()
            .map(|parameter| ast::ParameterWithDefault {
                range: Default::default(),
                node_index: Default::default(),
                parameter,
                default: None,
            })
            .collect();
        let _kw_roots = pyre_object::gc_roots::push_roots();
        let kw_base = pyre_object::gc_roots::pin_roots(&kw_defaults);
        for index in 0..kw_defaults.len() {
            let default = pyre_object::gc_roots::shadow_stack_get(kw_base + index);
            if unsafe { pyre_object::is_none(default) } {
                continue;
            }
            kwonly[index].default = Some(Box::new(self.recurse(|this| this.expr(default))?));
        }
        Ok(ast::Parameters {
            range: Default::default(),
            node_index: Default::default(),
            posonlyargs: posonlyargs.into(),
            args: args.into(),
            vararg,
            kwonlyargs: kwonly.into(),
            kwarg,
            runtime_defaults: None,
        })
    }

    fn parameter_list(
        &mut self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Vec<ast::Parameter>> {
        self.with_list(object, field, "arguments", |this, value| {
            this.recurse(|this| this.parameter(value))
        })
    }

    fn opt_parameter(
        &mut self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Option<Box<ast::Parameter>>> {
        self.optional_field(object, field)?
            .map(|value| self.recurse(|this| this.parameter(value)).map(Box::new))
            .transpose()
    }

    fn parameter(&mut self, mut object: PyObjectRef) -> AstResult<ast::Parameter> {
        let range = pyre_object::with_roots!(object => self.location(object, "arg"))?;
        Ok(ast::Parameter {
            range,
            node_index: Default::default(),
            name: pyre_object::with_roots!(object => self.identifier(object, "arg", "arg"))?,
            annotation: pyre_object::with_roots!(object => self.opt_expr(object, "annotation"))?,
            runtime_type_comment: self.opt_type_comment(object)?,
            runtime_type_comment_bytes: None,
        })
    }

    fn with_item(&mut self, mut object: PyObjectRef) -> AstResult<ast::WithItem> {
        let context_expr =
            pyre_object::with_roots!(object => self.req_expr(object, "context_expr", "withitem"))?;
        Ok(ast::WithItem {
            range: Default::default(),
            node_index: Default::default(),
            context_expr: *context_expr,
            optional_vars: self.opt_expr(object, "optional_vars")?,
        })
    }

    fn handler(&mut self, mut object: PyObjectRef) -> AstResult<ast::ExceptHandler> {
        let range = pyre_object::with_roots!(object => self.location(object, "excepthandler"))?;
        if !pyre_object::with_roots!(object => self.is_node(object, "ExceptHandler"))? {
            return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of excepthandler, but got ",
                self.repr(object)?
            )));
        }
        Ok(ast::ExceptHandler::ExceptHandler(
            ast::ExceptHandlerExceptHandler {
                range,
                node_index: Default::default(),
                type_: pyre_object::with_roots!(object => self.opt_expr(object, "type"))?,
                name: pyre_object::with_roots!(object => self.opt_identifier(object, "name"))?,
                body: self.body(object, "body", "ExceptHandler")?,
                runtime_body: None,
            },
        ))
    }

    fn comprehension(&mut self, mut object: PyObjectRef) -> AstResult<ast::Comprehension> {
        let target =
            pyre_object::with_roots!(object => self.req_expr(object, "target", "comprehension"))?;
        let iter =
            pyre_object::with_roots!(object => self.req_expr(object, "iter", "comprehension"))?;
        let ifs = pyre_object::with_roots!(object => self.exprs(object, "ifs", "comprehension"))?;
        let is_async = self.int_field(object, "is_async", "comprehension")?;
        Ok(ast::Comprehension {
            range: Default::default(),
            node_index: Default::default(),
            target: *target,
            iter: *iter,
            ifs,
            is_async: is_async != 0,
            runtime_ifs: None,
            runtime_is_async: None,
        })
    }

    fn comprehensions(
        &mut self,
        object: PyObjectRef,
        node: &str,
    ) -> AstResult<Vec<ast::Comprehension>> {
        self.with_list(object, "generators", node, |this, value| {
            this.recurse(|this| this.comprehension(value))
        })
    }

    fn aliases(&mut self, object: PyObjectRef, node: &str) -> AstResult<Vec<ast::Alias>> {
        self.with_list(object, "names", node, |this, mut value| {
            let range = pyre_object::with_roots!(value => this.location(value, "alias"))?;
            Ok(ast::Alias {
                range,
                node_index: Default::default(),
                name: pyre_object::with_roots!(value => this.identifier(value, "name", "alias"))?,
                asname: this.opt_identifier(value, "asname")?,
            })
        })
    }

    fn decorators(
        &mut self,
        object: PyObjectRef,
        node: &str,
    ) -> AstResult<ThinVec<ast::Decorator>> {
        Ok(self
            .with_list(object, "decorator_list", node, |this, value| {
                this.require_node(value, "expression")?;
                Ok(ast::Decorator {
                    range: Default::default(),
                    node_index: Default::default(),
                    expression: this.recurse(|this| this.expr(value))?,
                })
            })?
            .into())
    }

    /// `type_params` postdates the original positional constructors, so a
    /// manually built legacy node without the field means an empty list.
    fn type_params(
        &mut self,
        object: PyObjectRef,
        node: &str,
    ) -> AstResult<Option<Box<ast::TypeParams>>> {
        let Some(field) = self.optional_field(object, "type_params")? else {
            return Ok(None);
        };
        if !unsafe { pyre_object::is_list(field) } {
            return Err(crate::PyError::type_error(format!(
                "{node} field \"type_params\" must be a list, not a {}",
                class_name(field)
            )));
        }
        let values = unsafe {
            pyre_object::w_list_items_copy_as_vec_mode(
                field,
                majit_metainterp::jit::we_are_jitted(),
            )
        };
        if values.is_empty() {
            return Ok(None);
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&values);
        let mut type_params = Vec::with_capacity(values.len());
        for index in 0..values.len() {
            let value = pyre_object::gc_roots::shadow_stack_get(base + index);
            type_params.push(self.recurse(|this| this.type_param(value))?);
        }
        Ok(Some(Box::new(ast::TypeParams {
            range: Default::default(),
            node_index: Default::default(),
            type_params,
            runtime_type_params: None,
        })))
    }

    fn type_param(&mut self, mut object: PyObjectRef) -> AstResult<ast::TypeParam> {
        let range = pyre_object::with_roots!(object => self.location(object, "type_param"))?;
        if pyre_object::with_roots!(object => self.is_node(object, "TypeVar"))? {
            let name =
                pyre_object::with_roots!(object => self.identifier(object, "name", "TypeVar"))?;
            Ok(ast::TypeParam::TypeVar(ast::TypeParamTypeVar {
                node_index: Default::default(),
                range,
                name,
                bound: pyre_object::with_roots!(object => self.opt_expr(object, "bound"))?,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "TypeVarTuple"))? {
            let name = pyre_object::with_roots!(object => self.identifier(object, "name", "TypeVarTuple"))?;
            Ok(ast::TypeParam::TypeVarTuple(ast::TypeParamTypeVarTuple {
                node_index: Default::default(),
                range,
                name,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "ParamSpec"))? {
            let name =
                pyre_object::with_roots!(object => self.identifier(object, "name", "ParamSpec"))?;
            Ok(ast::TypeParam::ParamSpec(ast::TypeParamParamSpec {
                node_index: Default::default(),
                range,
                name,
                default: self.opt_expr(object, "default_value")?,
            }))
        } else {
            Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of type_param, but got ",
                self.repr(object)?
            )))
        }
    }

    fn match_case(&mut self, mut object: PyObjectRef) -> AstResult<ast::MatchCase> {
        let pattern = pyre_object::with_roots!(object => self.required_field(object, "pattern", "match_case"))?;
        let pattern = self.recurse(|this| this.pattern(pattern))?;
        Ok(ast::MatchCase {
            range: Default::default(),
            node_index: Default::default(),
            pattern,
            guard: pyre_object::with_roots!(object => self.opt_expr(object, "guard"))?,
            body: self.body(object, "body", "match_case")?,
            runtime_body: None,
        })
    }

    fn patterns(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Vec<ast::Pattern>> {
        self.with_list(object, field, node, |this, value| {
            this.recurse(|this| this.pattern(value))
        })
    }

    fn pattern(&mut self, mut object: PyObjectRef) -> AstResult<ast::Pattern> {
        let range = pyre_object::with_roots!(object => self.location(object, "pattern"))?;
        if pyre_object::with_roots!(object => self.is_node(object, "MatchValue"))? {
            Ok(ast::Pattern::MatchValue(ast::PatternMatchValue {
                node_index: Default::default(),
                range,
                value: self.req_expr(object, "value", "MatchValue")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchSingleton"))? {
            let value = self.field(object, "value", "MatchSingleton")?;
            let value = unsafe {
                if pyre_object::is_none(value) {
                    ast::Singleton::None
                } else if pyre_object::is_bool(value) {
                    if pyre_object::w_bool_get_value(value) {
                        ast::Singleton::True
                    } else {
                        ast::Singleton::False
                    }
                } else {
                    return Err(crate::PyError::value_error(
                        "MatchSingleton can only contain True, False and None",
                    ));
                }
            };
            Ok(ast::Pattern::MatchSingleton(ast::PatternMatchSingleton {
                node_index: Default::default(),
                range,
                value,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchSequence"))? {
            Ok(ast::Pattern::MatchSequence(ast::PatternMatchSequence {
                node_index: Default::default(),
                range,
                patterns: self.patterns(object, "patterns", "MatchSequence")?.into(),
                runtime_patterns: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchMapping"))? {
            let keys =
                pyre_object::with_roots!(object => self.exprs(object, "keys", "MatchMapping"))?;
            let patterns = pyre_object::with_roots!(object => self.patterns(object, "patterns", "MatchMapping"))?;
            Ok(ast::Pattern::MatchMapping(ast::PatternMatchMapping {
                node_index: Default::default(),
                range,
                keys: keys.into(),
                patterns: patterns.into(),
                rest: self.opt_identifier(object, "rest")?,
                runtime_keys: None,
                runtime_patterns: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchClass"))? {
            let cls =
                pyre_object::with_roots!(object => self.req_expr(object, "cls", "MatchClass"))?;
            let patterns = pyre_object::with_roots!(object => self.patterns(object, "patterns", "MatchClass"))?;
            let attrs = pyre_object::with_roots!(object => self.identifiers(object, "kwd_attrs", "MatchClass"))?;
            let kwd_patterns = self.patterns(object, "kwd_patterns", "MatchClass")?;
            if attrs.len() != kwd_patterns.len() {
                return Err(crate::PyError::value_error(
                    "MatchClass doesn't have the same number of keyword attributes as patterns",
                ));
            }
            let keywords = attrs
                .into_iter()
                .zip(kwd_patterns)
                .map(|(attr, pattern)| ast::PatternKeyword {
                    range,
                    node_index: Default::default(),
                    attr,
                    pattern,
                })
                .collect();
            Ok(ast::Pattern::MatchClass(ast::PatternMatchClass {
                node_index: Default::default(),
                range,
                cls,
                arguments: ast::PatternArguments {
                    range,
                    node_index: Default::default(),
                    patterns: patterns.into(),
                    keywords,
                },
                runtime_patterns: None,
                runtime_kwd_attrs: None,
                runtime_kwd_patterns: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchStar"))? {
            Ok(ast::Pattern::MatchStar(ast::PatternMatchStar {
                node_index: Default::default(),
                range,
                name: self.opt_identifier(object, "name")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchAs"))? {
            let pattern = pyre_object::with_roots!(object => self
                .optional_field(object, "pattern"))?
            .map(|value| self.recurse(|this| this.pattern(value)).map(Box::new))
            .transpose()?;
            Ok(ast::Pattern::MatchAs(ast::PatternMatchAs {
                node_index: Default::default(),
                range,
                pattern,
                name: self.opt_identifier(object, "name")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "MatchOr"))? {
            Ok(ast::Pattern::MatchOr(ast::PatternMatchOr {
                node_index: Default::default(),
                range,
                patterns: self.patterns(object, "patterns", "MatchOr")?,
                runtime_patterns: None,
            }))
        } else {
            Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of pattern, but got ",
                self.repr(object)?
            )))
        }
    }

    /// `_validate_stmts` / `_validate_exprs` (validate.py, :151) reject a
    /// missing element of a statement or expression list.  The compiler AST
    /// has nowhere to hold one, so unlike the rest of the validation this runs
    /// during the conversion, where it is still visible.  Lists of the other
    /// ASDL nodes carry no such check.
    fn require_node(&self, value: PyObjectRef, kind: &str) -> AstResult<()> {
        if unsafe { pyre_object::is_none(value) } {
            return Err(crate::PyError::value_error(format!(
                "None disallowed in {kind} list"
            )));
        }
        Ok(())
    }

    fn body(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<ThinVec<ast::Stmt>> {
        Ok(self
            .with_list(object, field, node, |this, value| {
                this.require_node(value, "statement")?;
                this.recurse(|this| this.stmt(value))
            })?
            .into())
    }

    fn exprs(&mut self, object: PyObjectRef, field: &str, node: &str) -> AstResult<Vec<ast::Expr>> {
        self.with_list(object, field, node, |this, value| {
            this.require_node(value, "expression")?;
            this.recurse(|this| this.expr(value))
        })
    }

    fn req_expr(
        &mut self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Box<ast::Expr>> {
        let value = self.required_field(object, field, node)?;
        Ok(Box::new(self.recurse(|this| this.expr(value))?))
    }

    fn opt_expr(&mut self, object: PyObjectRef, field: &str) -> AstResult<Option<Box<ast::Expr>>> {
        self.optional_field(object, field)?
            .map(|value| self.recurse(|this| this.expr(value)).map(Box::new))
            .transpose()
    }

    fn identifier(
        &self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<ast::Identifier> {
        let value = self.required_field(object, field, node)?;
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST identifier must be of type str",
            ));
        }
        Ok(ast::Identifier::new(
            utf8_only(value)?.to_string(),
            Default::default(),
        ))
    }

    fn opt_identifier(
        &self,
        object: PyObjectRef,
        field: &str,
    ) -> AstResult<Option<ast::Identifier>> {
        let Some(value) = self.optional_field(object, field)? else {
            return Ok(None);
        };
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST identifier must be of type str",
            ));
        }
        Ok(Some(ast::Identifier::new(
            utf8_only(value)?.to_string(),
            Default::default(),
        )))
    }

    /// A node's `type_comment`, which every ASDL kind that has one spells the
    /// same way: an optional plain string.
    fn opt_type_comment(&self, object: PyObjectRef) -> AstResult<Option<Box<str>>> {
        let Some(value) = self.optional_field(object, "type_comment")? else {
            return Ok(None);
        };
        if !unsafe { pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(
                "AST type_comment must be of type str",
            ));
        }
        Ok(Some(utf8_only(value)?.to_string().into()))
    }

    fn identifiers(
        &self,
        object: PyObjectRef,
        field: &str,
        node: &str,
    ) -> AstResult<Vec<ast::Identifier>> {
        self.list(object, field, node)?
            .into_iter()
            .map(|value| {
                if !unsafe { pyre_object::is_str(value) } {
                    return Err(crate::PyError::type_error(
                        "AST identifier must be of type str",
                    ));
                }
                Ok(ast::Identifier::new(
                    utf8_only(value)?.to_string(),
                    Default::default(),
                ))
            })
            .collect()
    }

    fn int_field(&self, object: PyObjectRef, field: &str, node: &str) -> AstResult<i64> {
        let value = self.field(object, field, node)?;
        self.obj_to_int(value)
    }

    fn expr(&mut self, mut object: PyObjectRef) -> AstResult<ast::Expr> {
        let range = pyre_object::with_roots!(object => self.location(object, "expr"))?;
        if pyre_object::with_roots!(object => self.is_node(object, "UnaryOp"))? {
            let mut operand = pyre_object::with_roots!(object => self.required_field(object, "operand", "UnaryOp"))?;
            let op =
                pyre_object::with_roots!(operand => self.required_field(object, "op", "UnaryOp"))?;
            Ok(ast::Expr::UnaryOp(ast::ExprUnaryOp {
                node_index: Default::default(),
                range,
                op: pyre_object::with_roots!(operand => self.unaryop(op))?,
                operand: Box::new(self.recurse(|this| this.expr(operand))?),
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "BinOp"))? {
            let mut left =
                pyre_object::with_roots!(object => self.required_field(object, "left", "BinOp"))?;
            let mut right = pyre_object::with_roots!(left, object => self.required_field(object, "right", "BinOp"))?;
            let op = pyre_object::with_roots!(left, right => self.required_field(object, "op", "BinOp"))?;
            Ok(ast::Expr::BinOp(ast::ExprBinOp {
                node_index: Default::default(),
                range,
                left: Box::new(self.recurse(|this| this.expr(left))?),
                op: pyre_object::with_roots!(right => self.operator(op))?,
                right: Box::new(self.recurse(|this| this.expr(right))?),
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Call"))? {
            let mut func =
                pyre_object::with_roots!(object => self.required_field(object, "func", "Call"))?;
            let args = pyre_object::with_roots!(func, object => self
                .list(object, "args", "Call"))?
            .into_iter()
            .map(|arg| {
                self.require_node(arg, "expression")?;
                self.recurse(|this| this.expr(arg))
            })
            .collect::<Result<Vec<_>, _>>()?;
            let keywords = pyre_object::with_roots!(func => self
                .list(object, "keywords", "Call"))?
            .into_iter()
            .map(|keyword| self.recurse(|this| this.keyword(keyword)))
            .collect::<Result<Vec<_>, _>>()?;
            Ok(ast::Expr::Call(ast::ExprCall {
                node_index: Default::default(),
                range_start: range.start(),
                func: Box::new(self.recurse(|this| this.expr(func))?),
                arguments: ast::Arguments {
                    node_index: Default::default(),
                    range,
                    args: args.into_boxed_slice(),
                    keywords: keywords.into(),
                    runtime_args: None,
                    runtime_bases: None,
                },
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Attribute"))? {
            let mut value = pyre_object::with_roots!(object => self.required_field(object, "value", "Attribute"))?;
            let mut ctx = pyre_object::with_roots!(object, value => self.required_field(object, "ctx", "Attribute"))?;
            Ok(ast::Expr::Attribute(ast::ExprAttribute {
                node_index: Default::default(),
                range,
                value: Box::new(self.recurse(|this| this.expr(value))?),
                attr: ast::Identifier::new(
                    pyre_object::with_roots!(ctx => self.string(object, "attr", "Attribute"))?,
                    Default::default(),
                ),
                ctx: self.context(ctx)?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "List"))?
            || pyre_object::with_roots!(object => self.is_node(object, "Tuple"))?
        {
            let is_tuple = pyre_object::with_roots!(object => self.is_node(object, "Tuple"))?;
            let elements = pyre_object::with_roots!(object => self
                .list(object, "elts", if is_tuple { "Tuple" } else { "List" }))?
            .into_iter()
            .map(|element| {
                self.require_node(element, "expression")?;
                self.recurse(|this| this.expr(element))
            })
            .collect::<Result<Vec<_>, _>>()?;
            let ctx = self.context(self.required_field(object, "ctx", "sequence")?)?;
            if is_tuple {
                Ok(ast::Expr::Tuple(ast::ExprTuple {
                    node_index: Default::default(),
                    range,
                    elts: elements,
                    ctx,
                    parenthesized: true,
                    runtime_elts: None,
                }))
            } else {
                Ok(ast::Expr::List(ast::ExprList {
                    node_index: Default::default(),
                    range,
                    elts: elements,
                    ctx,
                    runtime_elts: None,
                }))
            }
        } else if pyre_object::with_roots!(object => self.is_node(object, "Name"))? {
            let ctx =
                pyre_object::with_roots!(object => self.required_field(object, "ctx", "Name"))?;
            let ctx = pyre_object::with_roots!(object => self.context(ctx))?;
            Ok(ast::Expr::Name(ast::ExprName {
                node_index: Default::default(),
                range,
                id: ast::name::Name::new(self.string(object, "id", "Name")?),
                ctx,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Constant"))? {
            let value =
                pyre_object::with_roots!(object => self.field(object, "value", "Constant"))?;
            Ok(ast::Expr::Constant(ast::ExprConstant {
                node_index: Default::default(),
                range,
                value: pyre_object::with_roots!(object => self.constant_value(value))?,
                kind: self.constant_kind(object)?,
                invalid_type: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "BoolOp"))? {
            let op =
                pyre_object::with_roots!(object => self.required_field(object, "op", "BoolOp"))?;
            Ok(ast::Expr::BoolOp(ast::ExprBoolOp {
                node_index: Default::default(),
                range,
                op: pyre_object::with_roots!(object => self.boolop(op))?,
                values: self.exprs(object, "values", "BoolOp")?,
                runtime_values: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "NamedExpr"))? {
            let target =
                pyre_object::with_roots!(object => self.req_expr(object, "target", "NamedExpr"))?;
            let value = self.req_expr(object, "value", "NamedExpr")?;
            Ok(ast::Expr::Named(ast::ExprNamed {
                node_index: Default::default(),
                range,
                target,
                value,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Lambda"))? {
            let args =
                pyre_object::with_roots!(object => self.required_field(object, "args", "Lambda"))?;
            let parameters = self.recurse(|this| this.parameters(args))?;
            let body = self.req_expr(object, "body", "Lambda")?;
            Ok(ast::Expr::Lambda(ast::ExprLambda {
                node_index: Default::default(),
                range,
                parameters: Some(Box::new(parameters)),
                body,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "IfExp"))? {
            let test = pyre_object::with_roots!(object => self.req_expr(object, "test", "IfExp"))?;
            let body = pyre_object::with_roots!(object => self.req_expr(object, "body", "IfExp"))?;
            let orelse = self.req_expr(object, "orelse", "IfExp")?;
            Ok(ast::Expr::If(ast::ExprIf {
                node_index: Default::default(),
                range,
                test,
                body,
                orelse,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Dict"))? {
            let keys = pyre_object::with_roots!(object => self.list(object, "keys", "Dict"))?;
            let values = self.list(object, "values", "Dict")?;
            if keys.len() != values.len() {
                return Err(crate::PyError::value_error(
                    "Dict doesn't have the same number of keys as values",
                ));
            }
            let _roots = pyre_object::gc_roots::push_roots();
            // Two slices: publish both before the first normalize, which is
            // itself a GC safepoint (`pin_roots` multi-slice contract).
            let key_base = pyre_object::gc_roots::publish_roots(&keys);
            let value_base = pyre_object::gc_roots::publish_roots(&values);
            pyre_object::gc_roots::normalize_roots(key_base, keys.len() + values.len());
            // A `None` key is the `**mapping` spread, which has no key node.
            let mut items = Vec::with_capacity(keys.len());
            for index in 0..keys.len() {
                let key = pyre_object::gc_roots::shadow_stack_get(key_base + index);
                let value = pyre_object::gc_roots::shadow_stack_get(value_base + index);
                let key = if unsafe { pyre_object::is_none(key) } {
                    None
                } else {
                    Some(self.recurse(|this| this.expr(key))?)
                };
                self.require_node(value, "expression")?;
                items.push(ast::DictItem {
                    key,
                    value: self.recurse(|this| this.expr(value))?,
                });
            }
            Ok(ast::Expr::Dict(ast::ExprDict {
                node_index: Default::default(),
                range,
                items,
                runtime_values: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Set"))? {
            Ok(ast::Expr::Set(ast::ExprSet {
                node_index: Default::default(),
                range,
                elts: self.exprs(object, "elts", "Set")?,
                runtime_elts: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "ListComp"))? {
            let elt = pyre_object::with_roots!(object => self.req_expr(object, "elt", "ListComp"))?;
            let generators = self.comprehensions(object, "ListComp")?;
            Ok(ast::Expr::ListComp(ast::ExprListComp {
                node_index: Default::default(),
                range,
                elt,
                generators,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "SetComp"))? {
            let elt = pyre_object::with_roots!(object => self.req_expr(object, "elt", "SetComp"))?;
            let generators = self.comprehensions(object, "SetComp")?;
            Ok(ast::Expr::SetComp(ast::ExprSetComp {
                node_index: Default::default(),
                range,
                elt,
                generators,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "DictComp"))? {
            let key = pyre_object::with_roots!(object => self.req_expr(object, "key", "DictComp"))?;
            let value =
                pyre_object::with_roots!(object => self.req_expr(object, "value", "DictComp"))?;
            let generators = self.comprehensions(object, "DictComp")?;
            Ok(ast::Expr::DictComp(ast::ExprDictComp {
                node_index: Default::default(),
                range,
                key: Some(key),
                value,
                generators,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "GeneratorExp"))? {
            let elt =
                pyre_object::with_roots!(object => self.req_expr(object, "elt", "GeneratorExp"))?;
            let generators = self.comprehensions(object, "GeneratorExp")?;
            Ok(ast::Expr::Generator(ast::ExprGenerator {
                node_index: Default::default(),
                range,
                elt,
                generators,
                parenthesized: true,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Await"))? {
            Ok(ast::Expr::Await(ast::ExprAwait {
                node_index: Default::default(),
                range,
                value: self.req_expr(object, "value", "Await")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Yield"))? {
            Ok(ast::Expr::Yield(ast::ExprYield {
                node_index: Default::default(),
                range,
                value: self.opt_expr(object, "value")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "YieldFrom"))? {
            Ok(ast::Expr::YieldFrom(ast::ExprYieldFrom {
                node_index: Default::default(),
                range,
                value: self.req_expr(object, "value", "YieldFrom")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Compare"))? {
            let left =
                pyre_object::with_roots!(object => self.req_expr(object, "left", "Compare"))?;
            let ops = pyre_object::with_roots!(object => self
                .list(object, "ops", "Compare"))?
            .into_iter()
            .map(|op| self.cmpop(op))
            .collect::<Result<Vec<_>, _>>()?;
            let comparators = self.exprs(object, "comparators", "Compare")?;
            Ok(ast::Expr::Compare(ast::ExprCompare {
                node_index: Default::default(),
                range,
                left,
                ops: ops.into_boxed_slice(),
                comparators: comparators.into_boxed_slice(),
                runtime_comparators: None,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Subscript"))? {
            let value =
                pyre_object::with_roots!(object => self.req_expr(object, "value", "Subscript"))?;
            let slice =
                pyre_object::with_roots!(object => self.req_expr(object, "slice", "Subscript"))?;
            let ctx = self.context(self.required_field(object, "ctx", "Subscript")?)?;
            Ok(ast::Expr::Subscript(ast::ExprSubscript {
                node_index: Default::default(),
                range,
                value,
                slice,
                ctx,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Starred"))? {
            let value =
                pyre_object::with_roots!(object => self.req_expr(object, "value", "Starred"))?;
            let ctx = self.context(self.required_field(object, "ctx", "Starred")?)?;
            Ok(ast::Expr::Starred(ast::ExprStarred {
                node_index: Default::default(),
                range,
                value,
                ctx,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Slice"))? {
            Ok(ast::Expr::Slice(ast::ExprSlice {
                node_index: Default::default(),
                range,
                lower: pyre_object::with_roots!(object => self.opt_expr(object, "lower"))?,
                upper: pyre_object::with_roots!(object => self.opt_expr(object, "upper"))?,
                step: self.opt_expr(object, "step")?,
            }))
        } else if pyre_object::with_roots!(object => self.is_node(object, "JoinedStr"))? {
            let values = self.exprs(object, "values", "JoinedStr")?;
            Ok(fstring(range, Vec::new(), Some(values), false))
        } else if pyre_object::with_roots!(object => self.is_node(object, "FormattedValue"))? {
            let element = self.interpolation(object)?;
            Ok(fstring(range, vec![element], None, true))
        } else if pyre_object::with_roots!(object => self.is_node(object, "TemplateStr"))? {
            let range = pyre_object::with_roots!(object => self.location(object, "TemplateStr"))?;
            let values = self.exprs(object, "values", "TemplateStr")?;
            Ok(tstring(range, Vec::new(), Some(values)))
        } else if pyre_object::with_roots!(object => self.is_node(object, "Interpolation"))? {
            let range = pyre_object::with_roots!(object => self.location(object, "Interpolation"))?;
            let element = self.tstring_interpolation(object)?;
            Ok(tstring(range, vec![element], None))
        } else {
            Err(crate::PyError::type_error(crate::display::wtf8_format!(
                "expected some sort of expr, but got ",
                self.repr(object)?
            )))
        }
    }

    fn interpolation(
        &mut self,
        mut object: PyObjectRef,
    ) -> AstResult<ast::InterpolatedStringElement> {
        let expression =
            pyre_object::with_roots!(object => self.req_expr(object, "value", "FormattedValue"))?;
        let conversion =
            pyre_object::with_roots!(object => self.conversion(object, "FormattedValue"))?;
        let format_spec = pyre_object::with_roots!(object => self.opt_expr(object, "format_spec"))?;
        Ok(ast::InterpolatedStringElement::Interpolation(
            ast::InterpolatedElement {
                node_index: Default::default(),
                range: self.location(object, "FormattedValue")?,
                expression,
                debug_text: None,
                conversion,
                // A spec is an expression that gets compiled like any other
                // (`visit_FormattedValue`, codegen.py). The compiler AST
                // instead keeps the elements the parser found between the
                // braces, which an object does not carry, so the spec rides
                // the field below and the parsed shape stays empty.
                format_spec: None,
                runtime_str: None,
                runtime_interpolation_format_spec: None,
                runtime_formatted_value_format_spec: format_spec,
            },
        ))
    }

    /// RustPython `_ast::string::tstring_interpolation_from_object_with_range`:
    /// the public `Interpolation` is represented by a one-element native
    /// t-string.  Its `str` and object-form format spec stay in the runtime
    /// fields consumed by `compile_runtime_interpolation`.
    fn tstring_interpolation(
        &mut self,
        mut object: PyObjectRef,
    ) -> AstResult<ast::InterpolatedStringElement> {
        let range = pyre_object::with_roots!(object => self.location(object, "Interpolation"))?;
        let expression =
            pyre_object::with_roots!(object => self.req_expr(object, "value", "Interpolation"))?;
        let str_value =
            pyre_object::with_roots!(object => self.field(object, "str", "Interpolation"))?;
        let runtime_str = Some(pyre_object::with_roots!(object => self.constant_value(str_value))?);
        let conversion =
            pyre_object::with_roots!(object => self.conversion(object, "Interpolation"))?;
        let format_spec = self.opt_expr(object, "format_spec")?;
        Ok(ast::InterpolatedStringElement::Interpolation(
            ast::InterpolatedElement {
                node_index: Default::default(),
                range,
                expression,
                debug_text: None,
                conversion,
                format_spec: None,
                runtime_str,
                runtime_interpolation_format_spec: format_spec,
                runtime_formatted_value_format_spec: None,
            },
        ))
    }

    /// `visit_FormattedValue` (codegen.py) matches `s`, `r` and `a` and
    /// leaves anything else at no conversion at all; 3.14 stops instead, so a
    /// character it does not know is an error here.
    fn conversion(&self, object: PyObjectRef, node: &str) -> AstResult<ast::ConversionFlag> {
        match self.int_field(object, "conversion", node)? {
            -1 => Ok(ast::ConversionFlag::None),
            value if value == i64::from(b's') => Ok(ast::ConversionFlag::Str),
            value if value == i64::from(b'r') => Ok(ast::ConversionFlag::Repr),
            value if value == i64::from(b'a') => Ok(ast::ConversionFlag::Ascii),
            value => Err(crate::PyError::system_error(format!(
                "Unrecognized conversion character {value}"
            ))),
        }
    }

    fn boolop(&self, mut object: PyObjectRef) -> AstResult<ast::BoolOp> {
        for (name, op) in [("And", ast::BoolOp::And), ("Or", ast::BoolOp::Or)] {
            if pyre_object::with_roots!(object => self.is_node(object, name))? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected some sort of boolop, but got ",
            self.repr(object)?
        )))
    }

    fn cmpop(&self, mut object: PyObjectRef) -> AstResult<ast::CmpOp> {
        for (name, op) in [
            ("Eq", ast::CmpOp::Eq),
            ("NotEq", ast::CmpOp::NotEq),
            ("Lt", ast::CmpOp::Lt),
            ("LtE", ast::CmpOp::LtE),
            ("Gt", ast::CmpOp::Gt),
            ("GtE", ast::CmpOp::GtE),
            ("Is", ast::CmpOp::Is),
            ("IsNot", ast::CmpOp::IsNot),
            ("In", ast::CmpOp::In),
            ("NotIn", ast::CmpOp::NotIn),
        ] {
            if pyre_object::with_roots!(object => self.is_node(object, name))? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected some sort of cmpop, but got ",
            self.repr(object)?
        )))
    }

    fn keyword(&mut self, mut object: PyObjectRef) -> AstResult<ast::Keyword> {
        let range = pyre_object::with_roots!(object => self.location(object, "keyword"))?;
        let arg = pyre_object::with_roots!(object => self
            .optional_field(object, "arg"))?
        .map(|value| {
            if !unsafe { pyre_object::is_str(value) } {
                return Err(crate::PyError::type_error(
                    "AST identifier must be of type str",
                ));
            }
            Ok(ast::Identifier::new(
                utf8_only(value)?.to_string(),
                Default::default(),
            ))
        })
        .transpose()?;
        let value = self.required_field(object, "value", "keyword")?;
        Ok(ast::Keyword {
            node_index: Default::default(),
            range,
            arg,
            value: self.recurse(|this| this.expr(value))?,
        })
    }

    fn constant_value(&self, object: PyObjectRef) -> AstResult<ast::ConstantValue> {
        unsafe {
            if pyre_object::is_none(object) {
                Ok(ast::ConstantValue::None)
            } else if pyre_object::is_bool(object) {
                Ok(ast::ConstantValue::Boolean(pyre_object::w_bool_get_value(
                    object,
                )))
            } else if pyre_object::is_int(object) {
                Ok(ast::ConstantValue::Integer(
                    pyre_object::w_int_get_value(object)
                        .to_string()
                        .into_boxed_str(),
                ))
            } else if pyre_object::is_long(object) {
                Ok(ast::ConstantValue::Integer(
                    pyre_object::w_long_get_value(object)
                        .to_string()
                        .into_boxed_str(),
                ))
            } else if pyre_object::is_float(object) {
                Ok(ast::ConstantValue::Float(pyre_object::w_float_get_value(
                    object,
                )))
            } else if pyre_object::is_str(object) {
                Ok(ast::ConstantValue::Str(
                    utf8_only(object)?.to_string().into_boxed_str(),
                ))
            } else if pyre_object::is_bytes(object) {
                Ok(ast::ConstantValue::Bytes(
                    pyre_object::w_bytes_data(object)
                        .to_vec()
                        .into_boxed_slice(),
                ))
            } else if pyre_object::is_complex(object) {
                Ok(ast::ConstantValue::Complex {
                    real: pyre_object::w_complex_get_real(object),
                    imag: pyre_object::w_complex_get_imag(object),
                })
            } else if pyre_object::is_ellipsis(object) {
                Ok(ast::ConstantValue::Ellipsis)
            } else if pyre_object::is_tuple(object) {
                // A container constant never comes out of the parser; it reaches
                // here from a tree an optimizer folded, and it nests. The node
                // depth guard does not apply: it counts AST nodes, and a
                // constant nested past it still compiles where 3.14 compiles.
                let items = pyre_object::w_tuple_items_copy_as_vec(object);
                let _roots = pyre_object::gc_roots::push_roots();
                let base = pyre_object::gc_roots::pin_roots(&items);
                let mut values = Vec::with_capacity(items.len());
                for index in 0..items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(base + index);
                    values.push(self.constant_value(item)?);
                }
                Ok(ast::ConstantValue::Tuple(values))
            } else if pyre_object::is_frozenset(object) {
                let items = pyre_object::w_set_items(object);
                let _roots = pyre_object::gc_roots::push_roots();
                let base = pyre_object::gc_roots::pin_roots(&items);
                let mut values = Vec::with_capacity(items.len());
                for index in 0..items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(base + index);
                    values.push(self.constant_value(item)?);
                }
                Ok(ast::ConstantValue::Frozenset(values))
            } else {
                Err(crate::PyError::type_error(format!(
                    "got an invalid type in Constant: {}",
                    crate::error::type_name_of(object)
                )))
            }
        }
    }

    /// `Constant.kind` — the `u` a string literal may have been written with.
    /// `check_string` (ast.py) takes bytes here as well, and the prefix is
    /// only ever read back as text, so bytes leave the field unset.
    fn constant_kind(&self, object: PyObjectRef) -> AstResult<Option<Box<str>>> {
        let Some(value) = self.optional_field(object, "kind")? else {
            return Ok(None);
        };
        if unsafe { crate::baseobjspace::isinstance_str_w(value) } {
            return Ok(Some(utf8_only(value)?.to_string().into_boxed_str()));
        }
        if unsafe { crate::baseobjspace::isinstance_bytes_w(value) } {
            return Ok(None);
        }
        Err(crate::PyError::type_error("AST string must be of type str"))
    }

    fn context(&self, mut object: PyObjectRef) -> AstResult<ast::ExprContext> {
        // The three the ASDL declares; `Invalid` is a compiler-AST state with
        // no `_ast` class behind it, so no object can carry one.
        for (name, ctx) in [
            ("Load", ast::ExprContext::Load),
            ("Store", ast::ExprContext::Store),
            ("Del", ast::ExprContext::Del),
        ] {
            if pyre_object::with_roots!(object => self.is_node(object, name))? {
                return Ok(ctx);
            }
        }
        Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected some sort of expr_context, but got ",
            self.repr(object)?
        )))
    }

    fn unaryop(&self, mut object: PyObjectRef) -> AstResult<ast::UnaryOp> {
        for (name, op) in [
            ("Invert", ast::UnaryOp::Invert),
            ("Not", ast::UnaryOp::Not),
            ("UAdd", ast::UnaryOp::UAdd),
            ("USub", ast::UnaryOp::USub),
        ] {
            if pyre_object::with_roots!(object => self.is_node(object, name))? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected some sort of unaryop, but got ",
            self.repr(object)?
        )))
    }

    fn operator(&self, mut object: PyObjectRef) -> AstResult<ast::Operator> {
        for (name, op) in [
            ("Add", ast::Operator::Add),
            ("Sub", ast::Operator::Sub),
            ("Mult", ast::Operator::Mult),
            ("MatMult", ast::Operator::MatMult),
            ("Div", ast::Operator::Div),
            ("Mod", ast::Operator::Mod),
            ("Pow", ast::Operator::Pow),
            ("LShift", ast::Operator::LShift),
            ("RShift", ast::Operator::RShift),
            ("BitOr", ast::Operator::BitOr),
            ("BitXor", ast::Operator::BitXor),
            ("BitAnd", ast::Operator::BitAnd),
            ("FloorDiv", ast::Operator::FloorDiv),
        ] {
            if pyre_object::with_roots!(object => self.is_node(object, name))? {
                return Ok(op);
            }
        }
        Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expected some sort of operator, but got ",
            self.repr(object)?
        )))
    }
}

/// An f-string expression built from `_ast` objects.  `visit_JoinedStr`
/// (codegen.py:2357) compiles the values one after another and joins them,
/// which is the shape an object carries; the compiler AST instead holds the
/// literal and interpolated parts the parser split, so the values are handed
/// over as the parts to join and that part list stays empty.  A
/// `FormattedValue` on its own is a single interpolated part, which does fit
/// the part list.
fn fstring(
    range: ruff_text_size::TextRange,
    elements: Vec<ast::InterpolatedStringElement>,
    runtime_joined_str: Option<Vec<ast::Expr>>,
    standalone_formatted_value: bool,
) -> ast::Expr {
    ast::Expr::FString(ast::ExprFString {
        node_index: Default::default(),
        range,
        value: ast::FStringValue::single(ast::FString {
            node_index: Default::default(),
            range,
            elements: elements.into(),
            flags: ast::FStringFlags::empty(),
        }),
        runtime_joined_str,
        // RustPython `_ast::string::formatted_value_to_expr` uses the same
        // compiler-side carrier for a public FormattedValue and JoinedStr.
        // An empty slot vector marks the former without changing codegen.
        runtime_values: standalone_formatted_value.then(Vec::new),
    })
}

/// RustPython `_ast::string::template_str_to_expr`: object-form values ride
/// beside an otherwise empty native t-string, while a standalone public
/// `Interpolation` is represented by its single native element.
fn tstring(
    range: ruff_text_size::TextRange,
    elements: Vec<ast::InterpolatedStringElement>,
    runtime_template_str: Option<Vec<ast::Expr>>,
) -> ast::Expr {
    ast::Expr::TString(ast::ExprTString {
        node_index: Default::default(),
        range,
        value: ast::TStringValue::single(ast::TString {
            node_index: Default::default(),
            range,
            elements: elements.into(),
            flags: ast::TStringFlags::empty(),
        }),
        runtime_template_str,
        runtime_values: None,
    })
}

pub fn parse_to_object(source: &str, mode: crate::compile::Mode) -> crate::PyResult {
    parse_to_object_with_opts(
        source,
        mode,
        crate::compile::CompileOpts::default(),
        true,
        false,
        -1,
    )
}

/// CPython 3.14 `Py_CompileStringObject` AST-returning branch: parse, run
/// `_PyCompile_AstPreprocess`, and expose the resulting native tree.
pub fn parse_to_object_with_opts(
    source: &str,
    mode: crate::compile::Mode,
    opts: crate::compile::CompileOpts,
    syntax_check_only: bool,
    type_comments: bool,
    feature_version: i64,
) -> crate::PyResult {
    // The tokenizer sees a source whose line terminators are all `\n`
    // (`pytokenizer.py universal_newline`), so the same rewrite runs here; the nodes and
    // the text `module_to_object` slices segments out of then agree.
    let source = &*crate::compile::universal_newline(source);
    // A comment leaves no node behind, so a `type_comments=True` parse reads
    // the token list the parser hands back beside the tree.
    let parse_mode = match mode {
        crate::compile::Mode::Eval => parser::Mode::Expression,
        crate::compile::Mode::Exec
        | crate::compile::Mode::Single
        | crate::compile::Mode::BlockExpr => parser::Mode::Module,
    };
    // PyPy `CompileInfo.feature_version` carries the requested grammar version
    // into `PythonParser.parse_source`.  CPython 3.14 treats every negative
    // value and every future minor as the current grammar, so never expose
    // Ruff's preview 3.15 grammar through this 3.14 boundary.
    let minor = if feature_version < 0 {
        14
    } else {
        feature_version.min(14) as u8
    };
    let options = parser::ParseOptions::from(parse_mode)
        .with_target_version(ast::PythonVersion { major: 3, minor });
    let source_file = SourceFileBuilder::new("<unknown>", source).finish();
    let parsed = parser::parse(source, options).map_err(|error| {
        crate::builtins::compile_err_to_syntax_error(
            crate::compile::CompileError::from_ruff_parse_error(error, &source_file, mode),
            source,
            mode,
        )
    })?;
    // [3.14-spec] CPython `_PyPegen_new_identifier` performs this check as it
    // interns each identifier.  PyPy `new_identifier` has the same NFKC
    // spelling, while its parser-owned tree deliberately bypasses
    // `AstValidator`; keep the check at that parser boundary too.
    validate_parser_identifiers(source, parsed.tokens())?;
    if let Some(error) = parsed
        .unsupported_syntax_errors()
        .iter()
        .find(|error| should_report_unsupported_syntax_error(error))
    {
        return Err(crate::PyError::syntax_error(error.to_string()));
    }
    if let Some((message, range)) =
        legacy_feature_version_error(parsed.syntax(), parsed.tokens(), source, feature_version)
    {
        return Err(crate::builtins::syntax_error_from_source_range(
            message, source, range,
        ));
    }
    // PyPy `BaseParser.parse_number` lets the object-space integer conversion
    // enforce `sys.int_max_str_digits`, then turns that ValueError into the
    // parser's SyntaxError with the hexadecimal advice appended.  The pinned
    // RustPython helper performs the same token-level check for Ruff.
    if let Some(error) = rustpython_compiler::long_decimal_integer_literal_error(
        &source_file,
        parsed.tokens(),
        crate::module::sys::state::int_max_str_digits().max(0) as usize,
    ) {
        return Err(crate::builtins::compile_err_to_syntax_error(
            error, source, mode,
        ));
    }
    let mut collected = super::type_comments::TypeComments::default();
    if type_comments {
        // An expression has none of the five positions a `TYPE_COMMENT` is
        // accepted in, so the comments are collected here only to be refused.
        collected = super::type_comments::collect(parsed.tokens(), source);
    }
    let mut module = parsed.into_syntax();
    if minor < 13 && has_variadic_type_param_default(&module) {
        return Err(crate::PyError::syntax_error(
            "Type parameter defaults are only supported in Python 3.13 and greater",
        ));
    }
    if type_comments {
        collected.attach(&mut module);
        // What attachment left over is a token no rule accepted, which the
        // parser would have failed on.
        let file_input = matches!(
            mode,
            crate::compile::Mode::Exec | crate::compile::Mode::BlockExpr
        );
        if let Some(error) = collected.misplaced(source, file_input) {
            return Err(error);
        }
    }
    preprocess_module(&mut module, mode, opts, syntax_check_only);

    let ast_module = crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("_ast"),
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    )?;
    module_to_object(module, source, mode, ast_module, &collected.ignores, true)
}

/// PyPy `PythonParser.func_type` / `type_expressions`:
/// `(' type_expressions? ')' '->' expression`, where leading `*`/`**`
/// markers classify the last one or two argument expressions but are not
/// represented in `FunctionType.argtypes`.
///
/// Ruff has no func-type start rule.  Validate the inner list through its
/// ordinary call-argument grammar, then parse a byte-for-byte-length-preserving
/// expression: top-level `->` becomes `, ` and accepted `*`/`**` markers become
/// spaces.  Every real expression consequently keeps its original TextRange,
/// including nested and multi-line nodes.
pub fn parse_func_type_to_object(source: &str, feature_version: i64) -> crate::PyResult {
    use ast::token::TokenKind;

    let source = &*crate::compile::universal_newline(source);
    let minor = if feature_version < 0 {
        14
    } else {
        feature_version.min(14) as u8
    };
    let options = parser::ParseOptions::from(parser::Mode::Expression)
        .with_target_version(ast::PythonVersion { major: 3, minor });
    let source_file = SourceFileBuilder::new("<unknown>", source).finish();
    let unchecked = parser::parse_unchecked(source, options.clone());

    let mut significant = Vec::new();
    for token in unchecked.tokens().iter() {
        let (kind, range) = token.as_tuple();
        if matches!(
            kind,
            TokenKind::Comment
                | TokenKind::Newline
                | TokenKind::NonLogicalNewline
                | TokenKind::EndOfFile
        ) {
            continue;
        }
        significant.push((kind, range));
    }

    let invalid = |range: ruff_text_size::TextRange| {
        let error = parser::ParseError {
            error: parser::ParseErrorType::OtherError("invalid syntax".to_owned()),
            location: range,
        };
        crate::builtins::compile_err_to_syntax_error(
            crate::compile::CompileError::from_ruff_parse_error(
                error,
                &source_file,
                crate::compile::Mode::Eval,
            ),
            source,
            crate::compile::Mode::Eval,
        )
    };
    let eof = ruff_text_size::TextSize::new(source.len() as u32);
    let eof_range = ruff_text_size::TextRange::empty(eof);
    if significant.is_empty() || significant[0].0 != TokenKind::Lpar {
        return Err(invalid(
            significant.first().map_or(eof_range, |entry| entry.1),
        ));
    }

    let mut depth = 0i64;
    let mut close_index = None;
    for index in 0..significant.len() {
        match significant[index].0 {
            TokenKind::Lpar | TokenKind::Lsqb | TokenKind::Lbrace => depth += 1,
            TokenKind::Rpar | TokenKind::Rsqb | TokenKind::Rbrace => {
                depth -= 1;
                if depth == 0 {
                    close_index = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close_index) = close_index else {
        return Err(invalid(eof_range));
    };
    if significant[close_index].0 != TokenKind::Rpar {
        return Err(invalid(significant[close_index].1));
    }
    let arrow_index = close_index + 1;
    if arrow_index >= significant.len() || significant[arrow_index].0 != TokenKind::Rarrow {
        let range = if arrow_index < significant.len() {
            significant[arrow_index].1
        } else {
            eof_range
        };
        return Err(invalid(range));
    }
    if arrow_index + 1 >= significant.len() {
        return Err(invalid(eof_range));
    }
    if close_index > 1 && significant[close_index - 1].0 == TokenKind::Comma {
        return Err(invalid(significant[close_index - 1].1));
    }
    // `PythonParser.type_expressions` consumes ordinary `expression` nodes,
    // not the call grammar's implicit generator argument. Any `for` at the
    // outer func-type-parentheses depth therefore needs another delimiter
    // pair around it; comprehensions inside `()`, `[]`, or `{}` remain valid.
    let mut inner_depth = 1i64;
    for &(kind, range) in &significant[1..close_index] {
        match kind {
            TokenKind::Lpar | TokenKind::Lsqb | TokenKind::Lbrace => inner_depth += 1,
            TokenKind::Rpar | TokenKind::Rsqb | TokenKind::Rbrace => inner_depth -= 1,
            TokenKind::For if inner_depth == 1 => return Err(invalid(range)),
            _ => {}
        }
    }
    let open_range = significant[0].1;
    let close_range = significant[close_index].1;
    let inner_start = open_range.end().to_usize();
    let inner_end = close_range.start().to_usize();
    let inner = &source[inner_start..inner_end];
    const WRAPPER: &str = "__pyre_func_type__(";
    let wrapped = format!("{WRAPPER}{inner})");
    // Ruff's pre-3.5 token stream classifies `await` as a Name and rejects a
    // valid await expression before it can publish the node that PyPy
    // `PythonParser.await_primary` hands to `BaseParser.check_version`. The
    // wrapper validates only the timeless `type_expressions` list shape, so
    // use the current grammar for that one old-version corner; the transformed
    // source below still goes through the requested grammar after its PyPy
    // tree-site version checks have run.
    let wrapper_options = if (0..5).contains(&feature_version) {
        parser::ParseOptions::from(parser::Mode::Expression).with_target_version(
            ast::PythonVersion {
                major: 3,
                minor: 14,
            },
        )
    } else {
        options.clone()
    };
    let wrapped_parsed = parser::parse(&wrapped, wrapper_options).map_err(|error| {
        let map = |offset: ruff_text_size::TextSize| {
            let offset = offset.to_usize().saturating_sub(WRAPPER.len());
            ruff_text_size::TextSize::new((inner_start + offset.min(inner.len())) as u32)
        };
        let mapped = parser::ParseError {
            error: error.error,
            location: ruff_text_size::TextRange::new(
                map(error.location.start()),
                map(error.location.end()),
            ),
        };
        crate::builtins::compile_err_to_syntax_error(
            crate::compile::CompileError::from_ruff_parse_error(
                mapped,
                &source_file,
                crate::compile::Mode::Eval,
            ),
            source,
            crate::compile::Mode::Eval,
        )
    })?;
    let ast::Mod::Expression(wrapped_expression) = wrapped_parsed.into_syntax() else {
        return Err(invalid(eof_range));
    };
    let ast::Expr::Call(call) = *wrapped_expression.body else {
        return Err(invalid(eof_range));
    };

    let mut marker_ranges = Vec::new();
    let positional_len = call.arguments.args.len();
    let mut seen_star = false;
    for index in 0..positional_len {
        if let ast::Expr::Starred(starred) = &call.arguments.args[index] {
            if seen_star || index + 1 != positional_len {
                return Err(invalid(significant[close_index - 1].1));
            }
            seen_star = true;
            let start = starred
                .range
                .start()
                .to_usize()
                .saturating_sub(WRAPPER.len());
            marker_ranges.push((inner_start + start, 1usize));
        }
    }
    let mut seen_double_star = false;
    for keyword in &call.arguments.keywords {
        if keyword.arg.is_some() || seen_double_star {
            let start = keyword
                .range
                .start()
                .to_usize()
                .saturating_sub(WRAPPER.len());
            let start = ruff_text_size::TextSize::new((inner_start + start) as u32);
            return Err(invalid(ruff_text_size::TextRange::empty(start)));
        }
        seen_double_star = true;
        let start = keyword
            .range
            .start()
            .to_usize()
            .saturating_sub(WRAPPER.len());
        marker_ranges.push((inner_start + start, 2usize));
    }
    let arg_count = positional_len + call.arguments.keywords.len();

    let mut transformed = source.as_bytes().to_vec();
    for &(start, len) in &marker_ranges {
        for offset in 0..len {
            transformed[start + offset] = b' ';
        }
    }
    let arrow = significant[arrow_index].1;
    transformed[arrow.start().to_usize()] = b',';
    transformed[arrow.start().to_usize() + 1] = b' ';
    let transformed = String::from_utf8(transformed).expect("ASCII-only func_type rewrite");
    if (0..5).contains(&feature_version) {
        let current_options = parser::ParseOptions::from(parser::Mode::Expression)
            .with_target_version(ast::PythonVersion {
                major: 3,
                minor: 14,
            });
        if let Ok(current) = parser::parse(&transformed, current_options)
            && let Some((message, range)) = legacy_feature_version_error(
                current.syntax(),
                current.tokens(),
                source,
                feature_version,
            )
        {
            return Err(crate::builtins::syntax_error_from_source_range(
                message, source, range,
            ));
        }
    }
    let parsed = parser::parse(&transformed, options).map_err(|error| {
        crate::builtins::compile_err_to_syntax_error(
            crate::compile::CompileError::from_ruff_parse_error(
                error,
                &source_file,
                crate::compile::Mode::Eval,
            ),
            source,
            crate::compile::Mode::Eval,
        )
    })?;
    validate_parser_identifiers(source, parsed.tokens())?;
    if let Some(error) = parsed
        .unsupported_syntax_errors()
        .iter()
        .find(|error| should_report_unsupported_syntax_error(error))
    {
        return Err(crate::PyError::syntax_error(error.to_string()));
    }
    if let Some((message, range)) =
        legacy_feature_version_error(parsed.syntax(), parsed.tokens(), source, feature_version)
    {
        return Err(crate::builtins::syntax_error_from_source_range(
            message, source, range,
        ));
    }
    if let Some(error) = rustpython_compiler::long_decimal_integer_literal_error(
        &source_file,
        parsed.tokens(),
        crate::module::sys::state::int_max_str_digits().max(0) as usize,
    ) {
        return Err(crate::builtins::compile_err_to_syntax_error(
            error,
            source,
            crate::compile::Mode::Eval,
        ));
    }
    let ast::Mod::Expression(expression) = parsed.into_syntax() else {
        return Err(invalid(eof_range));
    };
    let ast::Expr::Tuple(mut top) = *expression.body else {
        return Err(invalid(eof_range));
    };
    if top.elts.len() != 2 {
        return Err(invalid(eof_range));
    }
    let returns = top.elts.pop().expect("two-element func_type tuple");
    let left = top.elts.pop().expect("two-element func_type tuple");
    let argtypes = if arg_count == 0 {
        let ast::Expr::Tuple(tuple) = left else {
            return Err(invalid(open_range));
        };
        if !tuple.elts.is_empty() {
            return Err(invalid(open_range));
        }
        Vec::new()
    } else if arg_count == 1 {
        vec![left]
    } else {
        let ast::Expr::Tuple(tuple) = left else {
            return Err(invalid(open_range));
        };
        if tuple.elts.len() != arg_count {
            return Err(invalid(open_range));
        }
        tuple.elts
    };
    let ast_module_object = crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("_ast"),
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let ast_module = Rooted(roots.base());
    let _ = roots.pin_root(ast_module_object);
    let converter = Converter {
        source,
        source_file,
        ast_module,
        parser_locations: true,
    };
    let mut args = Vec::with_capacity(argtypes.len());
    for arg in &argtypes {
        args.push(converter.expr(arg)?);
    }
    let args = converter.list(args);
    let returns = converter.expr(&returns)?;
    Ok(converter
        .node(
            "FunctionType",
            None,
            &[("argtypes", args), ("returns", returns)],
        )?
        .get())
}

fn module_to_object(
    module: ast::Mod,
    source: &str,
    mode: crate::compile::Mode,
    module_object: PyObjectRef,
    ignores: &[super::type_comments::TypeComment],
    parser_locations: bool,
) -> crate::PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let ast_module = Rooted(pyre_object::gc_roots::shadow_stack_len());
    let _ = pyre_object::gc_roots::pin_root(module_object);
    let converter = Converter {
        source,
        source_file: SourceFileBuilder::new("<unknown>", source).finish(),
        ast_module,
        parser_locations,
    };
    let root = match module {
        ast::Mod::Expression(module) => converter.node(
            "Expression",
            None,
            &[("body", converter.expr(&module.body)?)],
        ),
        ast::Mod::Module(module) => {
            let root_name = if matches!(mode, crate::compile::Mode::Single) {
                "Interactive"
            } else {
                "Module"
            };
            let body = converter.stmt_list(&module.body)?;
            if root_name == "Module" {
                let type_ignores = ignores
                    .iter()
                    .map(|ignore| {
                        converter.node(
                            "TypeIgnore",
                            None,
                            &[
                                (
                                    "lineno",
                                    converter.pin(pyre_object::w_int_new(ignore.lineno as i64)),
                                ),
                                ("tag", converter.string(&ignore.text)),
                            ],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let type_ignores = converter.list(type_ignores);
                converter.node(
                    root_name,
                    None,
                    &[("body", body), ("type_ignores", type_ignores)],
                )
            } else {
                converter.node(root_name, None, &[("body", body)])
            }
        }
    }?;
    Ok(root.get())
}

/// A value published in [`module_to_object`]'s root scope, held as its shadow
/// slot rather than as a pointer.
///
/// Every value the tree is built from is produced by a call that allocates, and
/// a node's fields are produced one after another before the node exists to
/// hold any of them: a `PyObjectRef` copy of the first field addresses the
/// pre-move object by the time the last one is built.  Only the slot survives
/// that, so nothing here passes a bare `PyObjectRef` around — a value is read
/// out of its slot at the point it is used and nowhere earlier.
#[derive(Clone, Copy)]
struct Rooted(usize);

impl Rooted {
    fn get(self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.0)
    }
}

type RootedResult = Result<Rooted, crate::PyError>;

struct Converter<'a> {
    source: &'a str,
    /// A literal's parsed value cannot hold a lone surrogate -- the escape
    /// decoder answers U+FFFD for one -- so `string_literal_value` and its
    /// siblings re-read the text the node came from.  They need the source as
    /// a `SourceFile` to slice it by range.
    source_file: SourceFile,
    ast_module: Rooted,
    /// Ruff parser nodes include a few tokens that public PyPy/CPython AST
    /// locations exclude.  ObjectConverter inputs already carry public
    /// locations and must never receive those parser-only adaptations again.
    parser_locations: bool,
}

impl Converter<'_> {
    /// Publish `value` as a root of `module_to_object`'s scope.  The pin is
    /// what makes the collector forward it; the slot is how a later use finds
    /// where it was forwarded to.
    fn pin(&self, value: PyObjectRef) -> Rooted {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        Rooted(slot)
    }

    fn list(&self, values: Vec<Rooted>) -> Rooted {
        // Read the members back only here: `w_list_new` pins what it is handed,
        // so the vector it receives has to be current at the call, and building
        // it any earlier would hand over addresses the members have left.
        self.pin(pyre_object::w_list_new(
            values.into_iter().map(Rooted::get).collect(),
        ))
    }

    fn string(&self, value: &str) -> Rooted {
        self.pin(pyre_object::w_str_new_managed(value))
    }

    fn wtf8(&self, value: Wtf8Buf) -> Rooted {
        self.pin(pyre_object::w_str_from_wtf8_managed(value))
    }

    fn none(&self) -> Rooted {
        self.pin(pyre_object::w_none())
    }

    fn optional(&self, value: Option<Rooted>) -> Rooted {
        value.unwrap_or_else(|| self.none())
    }

    fn node(
        &self,
        name: &str,
        range: Option<(u32, u32)>,
        fields: &[(&str, Rooted)],
    ) -> RootedResult {
        let node_type = crate::baseobjspace::getattr_str(self.ast_module.get(), name)?;
        let node = self.pin(pyre_object::w_instance_new(node_type));
        // Every `setattr_str` below runs Python, so the node and the remaining
        // field values move under the loop; each is read at the store that
        // consumes it.
        for &(field, value) in fields {
            crate::baseobjspace::setattr_str(node.get(), field, value.get())?;
        }
        if let Some((start, end)) = range {
            let (lineno, col_offset) = self.location(start as usize);
            let (mut end_lineno, mut end_col_offset) = self.location(end as usize);
            // [3.14-spec] CPython `remove_docstring` replaces a sole optimized
            // docstring with a four-column synthetic Pass on its start line.
            // RustPython `remove_docstring_from_body` has to encode that in a
            // byte TextRange, which crosses a newline for a short multiline
            // literal; PyPy has no optimized-AST parse path to preserve here.
            if name == "Pass" && end == start.saturating_add(4) && end_lineno != lineno {
                end_lineno = lineno;
                end_col_offset = col_offset.saturating_add(4);
            }
            for (field, value) in [
                ("lineno", lineno),
                ("col_offset", col_offset),
                ("end_lineno", end_lineno),
                ("end_col_offset", end_col_offset),
            ] {
                // Box the position before reading the node back: `w_int_new`
                // allocates, so a receiver read ahead of it is the pre-move one.
                let w_value = pyre_object::w_int_new(value as i64);
                crate::baseobjspace::setattr_str(node.get(), field, w_value)?;
            }
        }
        Ok(node)
    }

    fn location(&self, offset: usize) -> (usize, usize) {
        let bytes = self.source.as_bytes();
        let offset = offset.min(bytes.len());
        let prefix = &bytes[..offset];
        let line_start = prefix
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map_or(0, |i| i + 1);
        (
            prefix.iter().filter(|byte| **byte == b'\n').count() + 1,
            offset - line_start,
        )
    }

    /// PyPy `BaseParser.set_decorators` attaches decorators while preserving
    /// the raw FunctionDef/ClassDef `target.location()`.  Ruff instead starts
    /// the statement range at the first `@`, so recover the first code token
    /// after the final decorator for the public node's location.
    fn definition_range(
        &self,
        statement: ruff_text_size::TextRange,
        decorators: &[ast::Decorator],
    ) -> (u32, u32) {
        let (start, end) = range(statement);
        if !self.parser_locations {
            return (start, end);
        }
        let Some(last) = decorators.last() else {
            return (start, end);
        };
        let after_decorator = last.range.end().to_u32();
        let bytes = self.source.as_bytes();
        let mut offset = after_decorator as usize;
        let limit = (end as usize).min(bytes.len());
        while offset < limit {
            while offset < limit && matches!(bytes[offset], b' ' | b'\t' | b'\x0c' | b'\n') {
                offset += 1;
            }
            if offset < limit && bytes[offset] == b'#' {
                while offset < limit && bytes[offset] != b'\n' {
                    offset += 1;
                }
                continue;
            }
            if offset < limit {
                return (offset as u32, end);
            }
        }
        (start, end)
    }

    fn stmt_list(&self, stmts: &[ast::Stmt]) -> RootedResult {
        stmts
            .iter()
            .map(|stmt| self.stmt(stmt))
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn expr_list(&self, exprs: &[ast::Expr]) -> RootedResult {
        exprs
            .iter()
            .map(|expr| self.expr(expr))
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn name_list<T: AsRef<str>>(&self, names: &[T]) -> Rooted {
        self.list(
            names
                .iter()
                .map(|name| self.string(name.as_ref()))
                .collect(),
        )
    }

    fn stmt(&self, stmt: &ast::Stmt) -> RootedResult {
        use ast::Stmt;
        match stmt {
            Stmt::FunctionDef(node) => {
                let name = if node.is_async {
                    "AsyncFunctionDef"
                } else {
                    "FunctionDef"
                };
                let decorators = node
                    .decorator_list
                    .iter()
                    .map(|d| self.expr(&d.expression))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    name,
                    Some(self.definition_range(node.range, &node.decorator_list)),
                    &[
                        ("name", self.string(node.name.as_str())),
                        ("args", self.parameters(&node.parameters)?),
                        ("body", self.stmt_list(&node.body)?),
                        ("decorator_list", self.list(decorators)),
                        (
                            "returns",
                            self.optional(
                                node.returns.as_deref().map(|v| self.expr(v)).transpose()?,
                            ),
                        ),
                        (
                            "type_comment",
                            self.optional(
                                node.runtime_type_comment.as_deref().map(|v| self.string(v)),
                            ),
                        ),
                        (
                            "type_params",
                            self.type_params(node.type_params.as_deref())?,
                        ),
                    ],
                )
            }
            Stmt::ClassDef(node) => {
                let decorators = node
                    .decorator_list
                    .iter()
                    .map(|d| self.expr(&d.expression))
                    .collect::<Result<Vec<_>, _>>()?;
                let (bases, keywords) = if let Some(arguments) = node.arguments.as_deref() {
                    (
                        self.expr_list(&arguments.args)?,
                        self.keyword_list(&arguments.keywords)?,
                    )
                } else {
                    (self.list(Vec::new()), self.list(Vec::new()))
                };
                self.node(
                    "ClassDef",
                    Some(self.definition_range(node.range, &node.decorator_list)),
                    &[
                        ("name", self.string(node.name.as_str())),
                        ("bases", bases),
                        ("keywords", keywords),
                        ("body", self.stmt_list(&node.body)?),
                        ("decorator_list", self.list(decorators)),
                        (
                            "type_params",
                            self.type_params(node.type_params.as_deref())?,
                        ),
                    ],
                )
            }
            Stmt::Return(node) => self.node(
                "Return",
                Some(range(node.range)),
                &[(
                    "value",
                    self.optional(node.value.as_deref().map(|v| self.expr(v)).transpose()?),
                )],
            ),
            Stmt::Delete(node) => self.node(
                "Delete",
                Some(range(node.range)),
                &[("targets", self.expr_list(&node.targets)?)],
            ),
            Stmt::TypeAlias(node) => self.node(
                "TypeAlias",
                Some(range(node.range)),
                &[
                    ("name", self.expr(&node.name)?),
                    (
                        "type_params",
                        self.type_params(node.type_params.as_deref())?,
                    ),
                    ("value", self.expr(&node.value)?),
                ],
            ),
            Stmt::Assign(node) => self.node(
                "Assign",
                Some(range(node.range)),
                &[
                    ("targets", self.expr_list(&node.targets)?),
                    ("value", self.expr(&node.value)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::AugAssign(node) => self.node(
                "AugAssign",
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("op", self.operator(node.op)?),
                    ("value", self.expr(&node.value)?),
                ],
            ),
            Stmt::AnnAssign(node) => self.node(
                "AnnAssign",
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("annotation", self.expr(&node.annotation)?),
                    (
                        "value",
                        self.optional(node.value.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "simple",
                        self.pin(pyre_object::w_int_new(
                            node.runtime_simple.unwrap_or(node.simple as i32) as i64,
                        )),
                    ),
                ],
            ),
            Stmt::For(node) => self.node(
                if node.is_async { "AsyncFor" } else { "For" },
                Some(range(node.range)),
                &[
                    ("target", self.expr(&node.target)?),
                    ("iter", self.expr(&node.iter)?),
                    ("body", self.stmt_list(&node.body)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::While(node) => self.node(
                "While",
                Some(range(node.range)),
                &[
                    ("test", self.expr(&node.test)?),
                    ("body", self.stmt_list(&node.body)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                ],
            ),
            Stmt::If(node) => {
                let mut orelse = Vec::new();
                for clause in node.elif_else_clauses.iter().rev() {
                    let body = self.stmt_list(&clause.body)?;
                    if let Some(test) = clause.test.as_ref() {
                        // PyPy PEG `elif_stmt` is recursive, so its LOCATIONS
                        // end at the tail of the whole elif/else chain.
                        let (start, clause_end) = range(clause.range);
                        let end = if self.parser_locations {
                            node.range.end().to_u32()
                        } else {
                            clause_end
                        };
                        orelse = vec![self.node(
                            "If",
                            Some((start, end)),
                            &[
                                ("test", self.expr(test)?),
                                ("body", body),
                                ("orelse", self.list(orelse)),
                            ],
                        )?];
                    } else {
                        // The members come out of the list as bare pointers, so
                        // each is published before the next clause allocates.
                        orelse = unsafe {
                            pyre_object::w_list_items_copy_as_vec_mode(
                                body.get(),
                                majit_metainterp::jit::we_are_jitted(),
                            )
                        }
                        .into_iter()
                        .map(|item| self.pin(item))
                        .collect();
                    }
                }
                self.node(
                    "If",
                    Some(range(node.range)),
                    &[
                        ("test", self.expr(&node.test)?),
                        ("body", self.stmt_list(&node.body)?),
                        ("orelse", self.list(orelse)),
                    ],
                )
            }
            Stmt::With(node) => self.node(
                if node.is_async { "AsyncWith" } else { "With" },
                Some(range(node.range)),
                &[
                    ("items", self.with_items(&node.items)?),
                    ("body", self.stmt_list(&node.body)?),
                    (
                        "type_comment",
                        self.optional(node.runtime_type_comment.as_deref().map(|v| self.string(v))),
                    ),
                ],
            ),
            Stmt::Raise(node) => self.node(
                "Raise",
                Some(range(node.range)),
                &[
                    (
                        "exc",
                        self.optional(node.exc.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "cause",
                        self.optional(node.cause.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Stmt::Try(node) => self.node(
                if node.is_star { "TryStar" } else { "Try" },
                Some(range(node.range)),
                &[
                    ("body", self.stmt_list(&node.body)?),
                    ("handlers", self.handlers(&node.handlers)?),
                    ("orelse", self.stmt_list(&node.orelse)?),
                    ("finalbody", self.stmt_list(&node.finalbody)?),
                ],
            ),
            Stmt::Assert(node) => self.node(
                "Assert",
                Some(range(node.range)),
                &[
                    ("test", self.expr(&node.test)?),
                    (
                        "msg",
                        self.optional(node.msg.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Stmt::Import(node) => self.node(
                "Import",
                Some(range(node.range)),
                &[("names", self.aliases(&node.names)?)],
            ),
            Stmt::ImportFrom(node) => self.node(
                "ImportFrom",
                Some(range(node.range)),
                &[
                    (
                        "module",
                        self.optional(node.module.as_ref().map(|v| self.string(v.as_str()))),
                    ),
                    ("names", self.aliases(&node.names)?),
                    (
                        "level",
                        self.pin(pyre_object::w_int_new(
                            node.runtime_level.unwrap_or(node.level as i32) as i64,
                        )),
                    ),
                ],
            ),
            Stmt::Global(node) => self.node(
                "Global",
                Some(range(node.range)),
                &[("names", self.name_list(&node.names))],
            ),
            Stmt::Nonlocal(node) => self.node(
                "Nonlocal",
                Some(range(node.range)),
                &[("names", self.name_list(&node.names))],
            ),
            Stmt::Expr(node) => self.node(
                "Expr",
                Some(range(node.range)),
                &[("value", self.expr(&node.value)?)],
            ),
            Stmt::Pass(node) => self.node("Pass", Some(range(node.range)), &[]),
            Stmt::Break(node) => self.node("Break", Some(range(node.range)), &[]),
            Stmt::Continue(node) => self.node("Continue", Some(range(node.range)), &[]),
            Stmt::Match(node) => {
                let cases = node
                    .cases
                    .iter()
                    .map(|case| self.match_case(case))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Match",
                    Some(range(node.range)),
                    &[
                        ("subject", self.expr(&node.subject)?),
                        ("cases", self.list(cases)),
                    ],
                )
            }
            Stmt::IpyEscapeCommand(_) => Err(crate::PyError::not_implemented(
                "AST conversion for IPython escape commands is not implemented",
            )),
        }
    }

    fn expr(&self, expr: &ast::Expr) -> RootedResult {
        use ast::Expr;
        use ruff_text_size::Ranged;
        match expr {
            Expr::BoolOp(n) => self.node(
                "BoolOp",
                Some(range(n.range)),
                &[
                    ("op", self.boolop(n.op)?),
                    ("values", self.expr_list(&n.values)?),
                ],
            ),
            Expr::Named(n) => self.node(
                "NamedExpr",
                Some(range(n.range)),
                &[
                    ("target", self.expr(&n.target)?),
                    ("value", self.expr(&n.value)?),
                ],
            ),
            Expr::BinOp(n) => self.node(
                "BinOp",
                Some(range(n.range)),
                &[
                    ("left", self.expr(&n.left)?),
                    ("op", self.operator(n.op)?),
                    ("right", self.expr(&n.right)?),
                ],
            ),
            Expr::UnaryOp(n) => self.node(
                "UnaryOp",
                Some(range(n.range)),
                &[
                    ("op", self.unaryop(n.op)?),
                    ("operand", self.expr(&n.operand)?),
                ],
            ),
            Expr::Lambda(n) => self.node(
                "Lambda",
                Some(range(n.range)),
                &[
                    ("args", self.parameters_opt(n.parameters.as_deref())?),
                    ("body", self.expr(&n.body)?),
                ],
            ),
            Expr::If(n) => self.node(
                "IfExp",
                Some(range(n.range)),
                &[
                    ("test", self.expr(&n.test)?),
                    ("body", self.expr(&n.body)?),
                    ("orelse", self.expr(&n.orelse)?),
                ],
            ),
            Expr::Dict(n) => {
                let keys = n
                    .items
                    .iter()
                    .map(|item| {
                        item.key
                            .as_ref()
                            .map(|key| self.expr(key))
                            .transpose()
                            .map(|v| self.optional(v))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let values = n
                    .items
                    .iter()
                    .map(|item| self.expr(&item.value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Dict",
                    Some(range(n.range)),
                    &[("keys", self.list(keys)), ("values", self.list(values))],
                )
            }
            Expr::Set(n) => self.node(
                "Set",
                Some(range(n.range)),
                &[("elts", self.expr_list(&n.elts)?)],
            ),
            Expr::ListComp(n) => self.node(
                "ListComp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::SetComp(n) => self.node(
                "SetComp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::DictComp(n) => {
                let key = n.key.as_deref().ok_or_else(|| {
                    crate::PyError::value_error("field 'key' is required for DictComp")
                })?;
                self.node(
                    "DictComp",
                    Some(range(n.range)),
                    &[
                        ("key", self.expr(key)?),
                        ("value", self.expr(&n.value)?),
                        ("generators", self.comprehensions(&n.generators)?),
                    ],
                )
            }
            Expr::Generator(n) => self.node(
                "GeneratorExp",
                Some(range(n.range)),
                &[
                    ("elt", self.expr(&n.elt)?),
                    ("generators", self.comprehensions(&n.generators)?),
                ],
            ),
            Expr::Await(n) => self.node(
                "Await",
                Some(range(n.range)),
                &[("value", self.expr(&n.value)?)],
            ),
            Expr::Yield(n) => self.node(
                "Yield",
                Some(range(n.range)),
                &[(
                    "value",
                    self.optional(n.value.as_deref().map(|v| self.expr(v)).transpose()?),
                )],
            ),
            Expr::YieldFrom(n) => self.node(
                "YieldFrom",
                Some(range(n.range)),
                &[("value", self.expr(&n.value)?)],
            ),
            Expr::Compare(n) => {
                let ops = n
                    .ops
                    .iter()
                    .map(|op| self.cmpop(*op))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "Compare",
                    Some(range(n.range)),
                    &[
                        ("left", self.expr(&n.left)?),
                        ("ops", self.list(ops)),
                        ("comparators", self.expr_list(&n.comparators)?),
                    ],
                )
            }
            Expr::Call(n) => self.node(
                "Call",
                Some(range(n.range())),
                &[
                    ("func", self.expr(&n.func)?),
                    ("args", self.expr_list(&n.arguments.args)?),
                    ("keywords", self.keyword_list(&n.arguments.keywords)?),
                ],
            ),
            Expr::StringLiteral(n) => self.constant(
                range(n.range),
                self.wtf8(string_literal_value(&self.source_file, &n.value)),
                if n.value.is_unicode() {
                    self.string("u")
                } else {
                    self.none()
                },
            ),
            Expr::BytesLiteral(n) => self.constant(
                range(n.range),
                self.pin(pyre_object::w_bytes_from_bytes(
                    &n.value.bytes().collect::<Vec<_>>(),
                )),
                self.none(),
            ),
            Expr::NumberLiteral(n) => {
                self.constant(range(n.range), self.number(&n.value)?, self.none())
            }
            Expr::BooleanLiteral(n) => self.constant(
                range(n.range),
                self.pin(pyre_object::w_bool_from(n.value)),
                self.none(),
            ),
            Expr::NoneLiteral(n) => self.constant(range(n.range), self.none(), self.none()),
            Expr::EllipsisLiteral(n) => self.constant(
                range(n.range),
                self.pin(pyre_object::w_ellipsis()),
                self.none(),
            ),
            Expr::Constant(n) => self.constant(
                range(n.range),
                self.constant_value(&n.value)?,
                self.optional(n.kind.as_deref().map(|v| self.string(v))),
            ),
            Expr::Attribute(n) => self.node(
                "Attribute",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("attr", self.string(n.attr.as_str())),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Subscript(n) => self.node(
                "Subscript",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("slice", self.expr(&n.slice)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Starred(n) => self.node(
                "Starred",
                Some(range(n.range)),
                &[
                    ("value", self.expr(&n.value)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Name(n) => self.node(
                "Name",
                Some(range(n.range)),
                &[
                    ("id", self.string(n.id.as_str())),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::List(n) => self.node(
                "List",
                Some(range(n.range)),
                &[
                    ("elts", self.expr_list(&n.elts)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Tuple(n) => self.node(
                "Tuple",
                Some(range(n.range)),
                &[
                    ("elts", self.expr_list(&n.elts)?),
                    ("ctx", self.context(n.ctx)?),
                ],
            ),
            Expr::Slice(n) => self.node(
                "Slice",
                Some(range(n.range)),
                &[
                    (
                        "lower",
                        self.optional(n.lower.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "upper",
                        self.optional(n.upper.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                    (
                        "step",
                        self.optional(n.step.as_deref().map(|v| self.expr(v)).transpose()?),
                    ),
                ],
            ),
            Expr::FString(n) => self.fstring(n),
            Expr::TString(n) => self.tstring(n),
            Expr::IpyEscapeCommand(_) => Err(crate::PyError::not_implemented(
                "AST conversion for IPython escape commands is not implemented",
            )),
        }
    }

    fn fstring(&self, node: &ast::ExprFString) -> RootedResult {
        if let Some(values) = node.runtime_joined_str.as_deref() {
            return self.node(
                "JoinedStr",
                Some(range(node.range)),
                &[("values", self.expr_list(values)?)],
            );
        }
        let standalone_formatted_value = node.runtime_values.as_ref().is_some_and(Vec::is_empty);
        if let Some(values) = node.runtime_values.as_deref()
            && !standalone_formatted_value
        {
            let values = values
                .iter()
                .map(|value| {
                    value
                        .as_ref()
                        .map(|value| self.expr(value))
                        .transpose()
                        .map(|value| self.optional(value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.node(
                "JoinedStr",
                Some(range(node.range)),
                &[("values", self.list(values))],
            );
        }

        let mut parts = Vec::new();
        for part in node.value.iter() {
            match part {
                ast::FStringPart::Literal(literal) => push_literal(
                    &mut parts,
                    range(literal.range),
                    &string_literal_part_value(&self.source_file, literal),
                ),
                ast::FStringPart::FString(fstring) => {
                    self.interpolated_elements(&fstring.elements, fstring.flags.into(), &mut parts)?
                }
            }
        }
        if standalone_formatted_value && let [JoinedPart::Value(value)] = parts.as_slice() {
            return Ok(*value);
        }
        let values = self.joined_values(parts)?;
        self.node(
            "JoinedStr",
            Some(range(node.range)),
            &[("values", self.list(values))],
        )
    }

    /// RustPython `_ast::string::tstring_to_object`: publish the compiler's
    /// t-string parts as the 3.14 `TemplateStr`/`Interpolation` public nodes.
    fn tstring(&self, node: &ast::ExprTString) -> RootedResult {
        if let Some(values) = node.runtime_template_str.as_deref() {
            return self.node(
                "TemplateStr",
                Some(range(node.range)),
                &[("values", self.expr_list(values)?)],
            );
        }
        if let Some(values) = node.runtime_values.as_deref() {
            let values = values
                .iter()
                .map(|value| {
                    value
                        .as_ref()
                        .map(|value| self.expr(value))
                        .transpose()
                        .map(|value| self.optional(value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.node(
                "TemplateStr",
                Some(range(node.range)),
                &[("values", self.list(values))],
            );
        }

        // RustPython `_ast::string::standalone_tstring_interpolation_to_object`:
        // an object-form `Interpolation` uses a one-part native t-string as
        // its carrier and must come back as that node, not as a nested
        // `TemplateStr`.
        if let [tstring] = node.value.as_slice() {
            let mut elements = tstring.elements.iter();
            if let Some(ast::InterpolatedStringElement::Interpolation(interpolation)) =
                elements.next()
                && elements.next().is_none()
                && interpolation.runtime_str.is_some()
            {
                let mut parts = Vec::new();
                self.template_elements(&tstring.elements, tstring.flags.into(), &mut parts)?;
                if let [JoinedPart::Value(value)] = parts.as_slice() {
                    return Ok(*value);
                }
            }
        }

        let mut parts = Vec::new();
        for tstring in node.value.as_slice() {
            self.template_elements(&tstring.elements, tstring.flags.into(), &mut parts)?;
        }
        let values = self.joined_values(parts)?;
        self.node(
            "TemplateStr",
            Some(range(node.range)),
            &[("values", self.list(values))],
        )
    }

    fn template_elements(
        &self,
        elements: &[ast::InterpolatedStringElement],
        flags: ast::AnyStringFlags,
        parts: &mut Vec<JoinedPart>,
    ) -> Result<(), crate::PyError> {
        use ruff_text_size::Ranged;

        for element in elements {
            match element {
                ast::InterpolatedStringElement::Literal(literal) => push_literal(
                    parts,
                    range(literal.range),
                    &interpolated_string_literal_value(&self.source_file, literal, flags),
                ),
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    let mut conversion = interpolation.conversion;
                    let expression_str =
                        if let Some(runtime_str) = interpolation.runtime_str.as_ref() {
                            self.constant_value(runtime_str)?
                        } else if let Some(debug_text) = interpolation.debug_text.as_ref() {
                            let (text, text_range) = interpolation_debug_text(
                                &self.source_file,
                                debug_text,
                                interpolation.expression.range(),
                            );
                            push_literal(parts, range(text_range), Wtf8::new(text.as_str()));
                            conversion =
                                debug_conversion(conversion, interpolation.format_spec.is_some());
                            let expression_range = extend_expr_range_with_wrapping_parens(
                                self.source,
                                interpolation.range,
                                interpolation.expression.range(),
                            )
                            .unwrap_or_else(|| interpolation.expression.range());
                            let (start, end) = range(expression_range);
                            let expression = self.source[start as usize..end as usize].to_owned();
                            self.string(&strip_interpolation_expr(
                                &[
                                    debug_text.leading(),
                                    expression.as_str(),
                                    debug_text.trailing(),
                                ]
                                .concat(),
                            ))
                        } else {
                            self.string(&self.tstring_interpolation_expr_str(interpolation))
                        };
                    let format_spec = if let Some(spec) =
                        interpolation.runtime_interpolation_format_spec.as_deref()
                    {
                        Some(self.expr(spec)?)
                    } else {
                        self.format_spec(&interpolation.format_spec, flags)?
                    };
                    parts.push(JoinedPart::Value(self.node(
                        "Interpolation",
                        Some(range(interpolation.range)),
                        &[
                            ("value", self.expr(&interpolation.expression)?),
                            ("str", expression_str),
                            (
                                "conversion",
                                self.pin(pyre_object::w_int_new(conversion as i8 as i64)),
                            ),
                            ("format_spec", self.optional(format_spec)),
                        ],
                    )?));
                }
            }
        }
        Ok(())
    }

    fn format_spec(
        &self,
        format_spec: &Option<Box<ast::InterpolatedStringFormatSpec>>,
        flags: ast::AnyStringFlags,
    ) -> Result<Option<Rooted>, crate::PyError> {
        format_spec
            .as_deref()
            .map(|spec| {
                let mut spec_parts = Vec::new();
                self.interpolated_elements(&spec.elements, flags, &mut spec_parts)?;
                let values = self.joined_values(spec_parts)?;
                // RustPython `_ast::string::ruff_format_spec_to_joined_str`:
                // the public `JoinedStr` includes the opening colon while its
                // constant children begin after it.
                let (mut start, end) = range(spec.range);
                if start > 0 && self.source.as_bytes().get(start as usize - 1) == Some(&b':') {
                    start -= 1;
                }
                self.node(
                    "JoinedStr",
                    Some((start, end)),
                    &[("values", self.list(values))],
                )
            })
            .transpose()
    }

    fn tstring_interpolation_expr_str(&self, interpolation: &ast::InterpolatedElement) -> String {
        use ruff_text_size::Ranged;

        let interpolation_range = interpolation.range;
        let expression_range = extend_expr_range_with_wrapping_parens(
            self.source,
            interpolation_range,
            interpolation.expression.range(),
        )
        .unwrap_or_else(|| interpolation.expression.range());
        let after_open_brace = interpolation_range.start() + ruff_text_size::TextSize::from(1);
        let start = if after_open_brace > expression_range.end() {
            expression_range.start()
        } else {
            after_open_brace
        };
        let start = start.to_u32() as usize;
        let end = expression_range.end().to_u32() as usize;
        strip_interpolation_expr(&self.source[start..end])
    }

    fn joined_values(&self, parts: Vec<JoinedPart>) -> Result<Vec<Rooted>, crate::PyError> {
        parts
            .into_iter()
            .map(|part| match part {
                JoinedPart::Literal { start, end, value } => {
                    self.constant((start, end), self.wtf8(value), self.none())
                }
                JoinedPart::Value(value) => Ok(value),
            })
            .collect()
    }

    fn interpolated_elements(
        &self,
        elements: &[ast::InterpolatedStringElement],
        flags: ast::AnyStringFlags,
        parts: &mut Vec<JoinedPart>,
    ) -> Result<(), crate::PyError> {
        for element in elements {
            match element {
                ast::InterpolatedStringElement::Literal(literal) => push_literal(
                    parts,
                    range(literal.range),
                    &interpolated_string_literal_value(&self.source_file, literal, flags),
                ),
                ast::InterpolatedStringElement::Interpolation(interpolation) => {
                    let mut conversion = interpolation.conversion;
                    if let Some(debug_text) = interpolation.debug_text.as_ref() {
                        self.push_debug_text(parts, interpolation, debug_text);
                        // fstring.py:315 — a debugging expression defaults to
                        // `!r` when neither a conversion nor a format spec
                        // says how to render the value.
                        if matches!(
                            (conversion, &interpolation.format_spec),
                            (ast::ConversionFlag::None, None)
                        ) {
                            conversion = ast::ConversionFlag::Repr;
                        }
                    }
                    let format_spec = if let Some(spec) =
                        interpolation.runtime_formatted_value_format_spec.as_deref()
                    {
                        Some(self.expr(spec)?)
                    } else {
                        self.format_spec(&interpolation.format_spec, flags)?
                    };
                    parts.push(JoinedPart::Value(self.node(
                        "FormattedValue",
                        Some(range(interpolation.range)),
                        &[
                            ("value", self.expr(&interpolation.expression)?),
                            (
                                "conversion",
                                self.pin(pyre_object::w_int_new(conversion as i8 as i64)),
                            ),
                            ("format_spec", self.optional(format_spec)),
                        ],
                    )?));
                }
            }
        }
        Ok(())
    }

    /// The literal an `=` conversion echoes.  `fstring_find_expr`
    /// (fstring.py) takes it as one slice of the source, from the start of
    /// the expression through the `=` and the whitespace after it; the parser
    /// here hands over that frame as the text on either side of the
    /// expression, so the slice is put back together from the two.
    fn push_debug_text(
        &self,
        parts: &mut Vec<JoinedPart>,
        interpolation: &ast::InterpolatedElement,
        debug_text: &ast::DebugText,
    ) {
        use ruff_text_size::Ranged;
        let (start, end) = range(interpolation.expression.range());
        let leading = strip_debug_comments(debug_text.leading());
        let trailing = strip_debug_comments(debug_text.trailing());
        let expression = &self.source[start as usize..end as usize];
        push_literal(
            parts,
            (
                start.saturating_sub(leading.len() as u32),
                end + trailing.len() as u32,
            ),
            Wtf8::new(
                [leading.as_str(), expression, trailing.as_str()]
                    .concat()
                    .as_str(),
            ),
        );
    }

    fn match_case(&self, case: &ast::MatchCase) -> RootedResult {
        self.node(
            "match_case",
            None,
            &[
                ("pattern", self.pattern(&case.pattern)?),
                (
                    "guard",
                    self.optional(case.guard.as_deref().map(|v| self.expr(v)).transpose()?),
                ),
                ("body", self.stmt_list(&case.body)?),
            ],
        )
    }

    fn pattern(&self, pattern: &ast::Pattern) -> RootedResult {
        match pattern {
            ast::Pattern::MatchValue(node) => self.node(
                "MatchValue",
                Some(range(node.range)),
                &[("value", self.expr(&node.value)?)],
            ),
            ast::Pattern::MatchSingleton(node) => self.node(
                "MatchSingleton",
                Some(range(node.range)),
                &[(
                    "value",
                    match node.value {
                        ast::Singleton::None => self.none(),
                        ast::Singleton::True => self.pin(pyre_object::w_bool_from(true)),
                        ast::Singleton::False => self.pin(pyre_object::w_bool_from(false)),
                    },
                )],
            ),
            ast::Pattern::MatchSequence(node) => self.node(
                "MatchSequence",
                Some(range(node.range)),
                &[("patterns", self.pattern_list(&node.patterns)?)],
            ),
            ast::Pattern::MatchMapping(node) => self.node(
                "MatchMapping",
                Some(range(node.range)),
                &[
                    ("keys", self.expr_list(&node.keys)?),
                    ("patterns", self.pattern_list(&node.patterns)?),
                    (
                        "rest",
                        self.optional(node.rest.as_ref().map(|name| self.string(name.as_str()))),
                    ),
                ],
            ),
            ast::Pattern::MatchClass(node) => {
                let kwd_attrs = node
                    .arguments
                    .keywords
                    .iter()
                    .map(|keyword| self.string(keyword.attr.as_str()))
                    .collect();
                let kwd_patterns = node
                    .arguments
                    .keywords
                    .iter()
                    .map(|keyword| self.pattern(&keyword.pattern))
                    .collect::<Result<Vec<_>, _>>()?;
                self.node(
                    "MatchClass",
                    Some(range(node.range)),
                    &[
                        ("cls", self.expr(&node.cls)?),
                        ("patterns", self.pattern_list(&node.arguments.patterns)?),
                        ("kwd_attrs", self.list(kwd_attrs)),
                        ("kwd_patterns", self.list(kwd_patterns)),
                    ],
                )
            }
            ast::Pattern::MatchStar(node) => self.node(
                "MatchStar",
                Some(range(node.range)),
                &[(
                    "name",
                    self.optional(node.name.as_ref().map(|name| self.string(name.as_str()))),
                )],
            ),
            ast::Pattern::MatchAs(node) => self.node(
                "MatchAs",
                Some(range(node.range)),
                &[
                    (
                        "pattern",
                        self.optional(
                            node.pattern
                                .as_deref()
                                .map(|pattern| self.pattern(pattern))
                                .transpose()?,
                        ),
                    ),
                    (
                        "name",
                        self.optional(node.name.as_ref().map(|name| self.string(name.as_str()))),
                    ),
                ],
            ),
            ast::Pattern::MatchOr(node) => self.node(
                "MatchOr",
                Some(range(node.range)),
                &[("patterns", self.pattern_list(&node.patterns)?)],
            ),
        }
    }

    fn pattern_list(&self, patterns: &[ast::Pattern]) -> RootedResult {
        patterns
            .iter()
            .map(|pattern| self.pattern(pattern))
            .collect::<Result<Vec<_>, _>>()
            .map(|patterns| self.list(patterns))
    }

    fn constant(&self, range: (u32, u32), value: Rooted, kind: Rooted) -> RootedResult {
        self.node("Constant", Some(range), &[("value", value), ("kind", kind)])
    }

    fn number(&self, value: &ast::Number) -> RootedResult {
        Ok(match value {
            ast::Number::Int(value) => {
                // Ruff's Int stores an overflowing non-decimal literal by
                // its original token spelling.  PyPy astbuilder.py parse_number
                // routes that spelling through `_string_to_int_or_long`
                // with the token's radix instead of decimal int().
                let spelling = value.to_string();
                let source = self.string(&spelling);
                self.pin(crate::builtins::parse_int_from_str(
                    source.get(),
                    &spelling,
                    0,
                )?)
            }
            ast::Number::Float(value) => self.pin(pyre_object::w_float_new(*value)),
            ast::Number::Complex { real, imag } => {
                self.pin(pyre_object::w_complex_new(*real, *imag))
            }
        })
    }

    fn constant_value(&self, value: &ast::ConstantValue) -> RootedResult {
        Ok(match value {
            ast::ConstantValue::None => self.none(),
            ast::ConstantValue::Boolean(value) => self.pin(pyre_object::w_bool_from(*value)),
            ast::ConstantValue::Str(value) => self.string(value),
            ast::ConstantValue::Bytes(value) => self.pin(pyre_object::w_bytes_from_bytes(value)),
            ast::ConstantValue::Integer(value) => {
                // PyPy astbuilder.py `parse_number` sends integer
                // tokens to `_string_to_int_or_long` with the literal's
                // radix. RustPython's ConstantValue retains the original
                // spelling, so let the same internal parser infer that radix
                // rather than feeding a hexadecimal token to decimal int().
                let source = self.string(value);
                self.pin(crate::builtins::parse_int_from_str(source.get(), value, 0)?)
            }
            ast::ConstantValue::Float(value) => self.pin(pyre_object::w_float_new(*value)),
            ast::ConstantValue::Complex { real, imag } => {
                self.pin(pyre_object::w_complex_new(*real, *imag))
            }
            ast::ConstantValue::Ellipsis => self.pin(pyre_object::w_ellipsis()),
            ast::ConstantValue::Tuple(values) => {
                let values = values
                    .iter()
                    .map(|v| self.constant_value(v))
                    .collect::<Result<Vec<_>, _>>()?;
                // A tuple header is untraced until it is pinned, so its members
                // are read only once there is nothing left to allocate before
                // the store, and the pin's returned word is what is kept.
                self.pin(pyre_object::w_tuple_new(
                    values.into_iter().map(Rooted::get).collect(),
                ))
            }
            ast::ConstantValue::Frozenset(values) => {
                let values = values
                    .iter()
                    .map(|value| self.constant_value(value))
                    .collect::<Result<Vec<_>, _>>()?;
                // PyPy `PythonCodeGenerator._optimize_comparator` builds a
                // `W_FrozensetObject`, and `Constant.to_object` passes that
                // wrapped value through unchanged. Rebuilding RustPython's
                // structural spelling therefore uses the same ordinary
                // frozenset storage, just like tuple constants above.
                // Root every recursively-built member across allocation of the
                // container, since no untraced Rust vector can own GC objects.
                self.pin(pyre_object::w_frozenset_from_items(
                    &values.into_iter().map(Rooted::get).collect::<Vec<_>>(),
                ))
            }
        })
    }

    fn singleton(&self, name: &str) -> RootedResult {
        let typ = crate::baseobjspace::getattr_str(self.ast_module.get(), name)?;
        Ok(self.pin(pyre_object::w_instance_new(typ)))
    }

    fn context(&self, value: ast::ExprContext) -> RootedResult {
        self.singleton(match value {
            ast::ExprContext::Load => "Load",
            ast::ExprContext::Store => "Store",
            ast::ExprContext::Del => "Del",
            ast::ExprContext::Invalid => "Load",
        })
    }
    fn boolop(&self, value: ast::BoolOp) -> RootedResult {
        self.singleton(match value {
            ast::BoolOp::And => "And",
            ast::BoolOp::Or => "Or",
        })
    }
    fn operator(&self, value: ast::Operator) -> RootedResult {
        self.singleton(match value {
            ast::Operator::Add => "Add",
            ast::Operator::Sub => "Sub",
            ast::Operator::Mult => "Mult",
            ast::Operator::MatMult => "MatMult",
            ast::Operator::Div => "Div",
            ast::Operator::Mod => "Mod",
            ast::Operator::Pow => "Pow",
            ast::Operator::LShift => "LShift",
            ast::Operator::RShift => "RShift",
            ast::Operator::BitOr => "BitOr",
            ast::Operator::BitXor => "BitXor",
            ast::Operator::BitAnd => "BitAnd",
            ast::Operator::FloorDiv => "FloorDiv",
        })
    }
    fn unaryop(&self, value: ast::UnaryOp) -> RootedResult {
        self.singleton(match value {
            ast::UnaryOp::Invert => "Invert",
            ast::UnaryOp::Not => "Not",
            ast::UnaryOp::UAdd => "UAdd",
            ast::UnaryOp::USub => "USub",
        })
    }
    fn cmpop(&self, value: ast::CmpOp) -> RootedResult {
        self.singleton(match value {
            ast::CmpOp::Eq => "Eq",
            ast::CmpOp::NotEq => "NotEq",
            ast::CmpOp::Lt => "Lt",
            ast::CmpOp::LtE => "LtE",
            ast::CmpOp::Gt => "Gt",
            ast::CmpOp::GtE => "GtE",
            ast::CmpOp::Is => "Is",
            ast::CmpOp::IsNot => "IsNot",
            ast::CmpOp::In => "In",
            ast::CmpOp::NotIn => "NotIn",
        })
    }

    fn parameters_opt(&self, parameters: Option<&ast::Parameters>) -> RootedResult {
        match parameters {
            Some(p) => self.parameters(p),
            None => self.parameters(&ast::Parameters::default()),
        }
    }

    fn parameters(&self, p: &ast::Parameters) -> RootedResult {
        let posonlyargs = p
            .posonlyargs
            .iter()
            .map(|p| self.parameter(&p.parameter, 0))
            .collect::<Result<Vec<_>, _>>()?;
        let args = p
            .args
            .iter()
            .map(|p| self.parameter(&p.parameter, 0))
            .collect::<Result<Vec<_>, _>>()?;
        let kwonlyargs = p
            .kwonlyargs
            .iter()
            .map(|p| self.parameter(&p.parameter, 0))
            .collect::<Result<Vec<_>, _>>()?;
        let mut defaults = Vec::new();
        defaults.extend(
            p.posonlyargs
                .iter()
                .chain(&p.args)
                .filter_map(|p| p.default.as_deref())
                .map(|v| self.expr(v))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let kw_defaults = p
            .kwonlyargs
            .iter()
            .map(|p| {
                p.default
                    .as_deref()
                    .map(|v| self.expr(v))
                    .transpose()
                    .map(|v| self.optional(v))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.node(
            "arguments",
            None,
            &[
                ("posonlyargs", self.list(posonlyargs)),
                ("args", self.list(args)),
                (
                    "vararg",
                    self.optional(
                        p.vararg
                            .as_deref()
                            .map(|v| self.parameter(v, 1))
                            .transpose()?,
                    ),
                ),
                ("kwonlyargs", self.list(kwonlyargs)),
                ("kw_defaults", self.list(kw_defaults)),
                (
                    "kwarg",
                    self.optional(
                        p.kwarg
                            .as_deref()
                            .map(|v| self.parameter(v, 2))
                            .transpose()?,
                    ),
                ),
                ("defaults", self.list(defaults)),
            ],
        )
    }

    fn parameter(&self, p: &ast::Parameter, star_width: u32) -> RootedResult {
        let (start, end) = range(p.range);
        // PyPy PEG `star_etc`/`kwds` build `ast.arg` from `param_no_default`,
        // after consuming `*` or `**`; Ruff includes those tokens in range.
        let star_width = if self.parser_locations { star_width } else { 0 };
        let start = start.saturating_add(star_width).min(end);
        self.node(
            "arg",
            Some((start, end)),
            &[
                ("arg", self.string(p.name.as_str())),
                (
                    "annotation",
                    self.optional(p.annotation.as_deref().map(|v| self.expr(v)).transpose()?),
                ),
                (
                    "type_comment",
                    self.optional(p.runtime_type_comment.as_deref().map(|v| self.string(v))),
                ),
            ],
        )
    }

    fn keyword_list(&self, keywords: &[ast::Keyword]) -> RootedResult {
        keywords
            .iter()
            .map(|k| {
                self.node(
                    "keyword",
                    Some(range(k.range)),
                    &[
                        (
                            "arg",
                            self.optional(k.arg.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                        ("value", self.expr(&k.value)?),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn aliases(&self, aliases: &[ast::Alias]) -> RootedResult {
        aliases
            .iter()
            .map(|a| {
                self.node(
                    "alias",
                    Some(range(a.range)),
                    &[
                        ("name", self.string(a.name.as_str())),
                        (
                            "asname",
                            self.optional(a.asname.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn with_items(&self, items: &[ast::WithItem]) -> RootedResult {
        items
            .iter()
            .map(|item| {
                self.node(
                    "withitem",
                    None,
                    &[
                        ("context_expr", self.expr(&item.context_expr)?),
                        (
                            "optional_vars",
                            self.optional(
                                item.optional_vars
                                    .as_deref()
                                    .map(|v| self.expr(v))
                                    .transpose()?,
                            ),
                        ),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn comprehensions(&self, comprehensions: &[ast::Comprehension]) -> RootedResult {
        comprehensions
            .iter()
            .map(|c| {
                self.node(
                    "comprehension",
                    None,
                    &[
                        ("target", self.expr(&c.target)?),
                        ("iter", self.expr(&c.iter)?),
                        ("ifs", self.expr_list(&c.ifs)?),
                        (
                            "is_async",
                            self.pin(pyre_object::w_int_new(c.is_async as i64)),
                        ),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn handlers(&self, handlers: &[ast::ExceptHandler]) -> RootedResult {
        handlers
            .iter()
            .map(|handler| match handler {
                ast::ExceptHandler::ExceptHandler(h) => self.node(
                    "ExceptHandler",
                    Some(range(h.range)),
                    &[
                        (
                            "type",
                            self.optional(h.type_.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                        (
                            "name",
                            self.optional(h.name.as_ref().map(|v| self.string(v.as_str()))),
                        ),
                        ("body", self.stmt_list(&h.body)?),
                    ],
                ),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }

    fn type_params(&self, params: Option<&ast::TypeParams>) -> RootedResult {
        let Some(params) = params else {
            return Ok(self.list(Vec::new()));
        };
        params
            .type_params
            .iter()
            .map(|param| match param {
                ast::TypeParam::TypeVar(p) => self.node(
                    "TypeVar",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "bound",
                            self.optional(p.bound.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
                ast::TypeParam::TypeVarTuple(p) => self.node(
                    "TypeVarTuple",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
                ast::TypeParam::ParamSpec(p) => self.node(
                    "ParamSpec",
                    Some(range(p.range)),
                    &[
                        ("name", self.string(p.name.as_str())),
                        (
                            "default_value",
                            self.optional(p.default.as_deref().map(|v| self.expr(v)).transpose()?),
                        ),
                    ],
                ),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|items| self.list(items))
    }
}

/// `%T`-style class name of an AST object.  The `_ast` node types are heap
/// types, so the name sits on the class the instance points at rather than on
/// the layout its `ob_type` names.
fn class_name(object: PyObjectRef) -> &'static str {
    match crate::typedef::r#type(object) {
        Some(w_type) => unsafe { pyre_object::w_type_get_name(w_type.as_ptr()) },
        None => unsafe { pyre_object::type_name_of(object) },
    }
}

/// A value of a `JoinedStr` under construction.  `add_constant_string`
/// (fstring.py) folds a piece into the `Constant` before it, keeping that
/// node's start and taking the new end, and an empty piece is dropped
/// (`f_string_to_ast_node`, fstring.py); what the parser kept apart -- an
/// implicit concatenation, the text an `=` conversion echoes -- therefore
/// reaches the tree as one node, not one per piece.
enum JoinedPart {
    Literal {
        start: u32,
        end: u32,
        value: Wtf8Buf,
    },
    Value(Rooted),
}

fn push_literal(parts: &mut Vec<JoinedPart>, (start, end): (u32, u32), value: &Wtf8) {
    if value.is_empty() {
        return;
    }
    if let Some(JoinedPart::Literal {
        end: last_end,
        value: last,
        ..
    }) = parts.last_mut()
    {
        *last_end = end;
        last.push_wtf8(value);
        return;
    }
    parts.push(JoinedPart::Literal {
        start,
        end,
        value: value.to_owned(),
    });
}

/// The `&str` a compiler AST field is, or the refusal `PyUnicode_AsUTF8`
/// answers with for the same value.
///
/// `ast::ConstantValue::Str` and every identifier field are `str`, so a lone
/// surrogate has nowhere to go on the way back into the compiler.  It gets
/// there through an ordinary `ast.parse` round trip, since the tree the parse
/// answers does carry one; [`w_str_get_value_opt`] returns `None` and this
/// path raises `UnicodeEncodeError`.
///
/// [`w_str_get_value_opt`]: pyre_object::w_str_get_value_opt
fn utf8_only(value: PyObjectRef) -> AstResult<&'static str> {
    if let Some(text) = unsafe { pyre_object::w_str_get_value_opt(value) } {
        return Ok(text);
    }
    let position = unsafe { pyre_object::w_str_get_wtf8(value) }
        .code_points()
        .position(|point| point.to_char().is_none())
        .expect("not valid UTF-8, so one of the code points is a surrogate");
    Err(crate::typedef::unicode_encode_error(
        "utf-8",
        value,
        position as i64,
        (position + 1) as i64,
        "surrogates not allowed",
    ))
}

/// A comment inside the braces runs to the end of the line and is no part of
/// the text an `=` conversion echoes.
fn strip_debug_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_comment = false;
    for ch in text.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
                result.push(ch);
            }
        } else if ch == '#' {
            in_comment = true;
        } else {
            result.push(ch);
        }
    }
    result
}

/// RustPython `_ast::string::debug_conversion`, following CPython
/// `_get_interpolation_conversion` for `{expr=}`.
fn debug_conversion(conversion: ast::ConversionFlag, has_format_spec: bool) -> ast::ConversionFlag {
    if matches!(conversion, ast::ConversionFlag::None) && !has_format_spec {
        ast::ConversionFlag::Repr
    } else {
        conversion
    }
}

/// RustPython `_ast::string::extend_expr_range_with_wrapping_parens` keeps
/// parentheses which belong to the spelling stored in `Interpolation.str`.
fn extend_expr_range_with_wrapping_parens(
    source: &str,
    interpolation_range: ruff_text_size::TextRange,
    expression_range: ruff_text_size::TextRange,
) -> Option<ruff_text_size::TextRange> {
    let (interpolation_start, interpolation_end) = range(interpolation_range);
    let (expression_start, expression_end) = range(expression_range);
    let left_slice = &source[interpolation_start as usize..expression_start as usize];
    let (left_index, left_char) = left_slice
        .char_indices()
        .rev()
        .find(|(_, ch)| !ch.is_whitespace())?;
    if left_char != '(' {
        return None;
    }

    let right_slice = &source[expression_end as usize..interpolation_end as usize];
    let (right_index, right_char) = right_slice
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())?;
    if right_char != ')' {
        return None;
    }

    Some(ruff_text_size::TextRange::new(
        (interpolation_start + left_index as u32).into(),
        (expression_end + right_index as u32 + 1).into(),
    ))
}

/// CPython `_strip_interpolation_expr`: remove the debug marker and trailing
/// whitespace from the expression spelling exposed as `Interpolation.str`.
fn strip_interpolation_expr(expression: &str) -> String {
    let mut end = expression.len();
    for (index, ch) in expression.char_indices().rev() {
        if ch.is_whitespace() || ch == '=' {
            end = index;
        } else {
            break;
        }
    }
    expression[..end].to_owned()
}

fn range(range: impl RangeParts) -> (u32, u32) {
    range.parts()
}

trait RangeParts {
    fn parts(self) -> (u32, u32);
}

impl RangeParts for ruff_text_size::TextRange {
    fn parts(self) -> (u32, u32) {
        (self.start().to_u32(), self.end().to_u32())
    }
}
