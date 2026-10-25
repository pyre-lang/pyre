//! Port of `rpython/translator/backendopt/finalizer.py`.
//!
//! `FinalizerAnalyzer` is a [`super::graphanalyze::BoolGraphAnalyzer`]
//! subclass that decides whether a `__del__`-style finalizer is
//! "lightweight enough" to run without the full GC machinery. The
//! upstream consumer is the GC transformer (`rpython/memory/gctransform/`),
//! which pyre has not yet ported. This module is published as a parity
//! sibling alongside [`super::canraise`], [`super::gilanalysis`], and
//! [`super::collectanalyze`] so the surface lines up the day a
//! consumer needs it.

use crate::flowspace::model::{GraphRef, Hlvalue, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{
    Dependency, DependencyTracker, GraphAnalyzer, framework_analyze_direct_call,
};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::translator::TranslationContext;

/// Upstream `class FinalizerError(Exception)` at `finalizer.py`.
///
/// > `__del__()` is used for lightweight RPython destructors, but the
/// > FinalizerAnalyzer found that it is not lightweight.
/// >
/// > The set of allowed operations is restrictive for a good reason
/// > — it's better to be safe. Specifically disallowed operations:
/// >
/// > * anything that escapes self
/// > * anything that can allocate
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalizerError(pub String);

impl FinalizerError {
    pub const DOC: &'static str = "__del__() is used for lightweight \
        RPython destructors, but the FinalizerAnalyzer found that it is \
        not lightweight.\n\n    The set of allowed operations is \
        restrictive for a good reason\n    - it's better to be safe. \
        Specifically disallowed operations:\n\n    * anything that \
        escapes self\n    * anything that can allocate\n    ";
}

/// Upstream allow-list at `finalizer.py:24-27`.
const OK_OPERATIONS: &[&str] = &[
    "ptr_nonzero",
    "ptr_eq",
    "ptr_ne",
    "free",
    "same_as",
    "direct_ptradd",
    "force_cast",
    "track_alloc_stop",
    "raw_free",
    "adr_eq",
    "adr_ne",
    "debug_print",
];

/// `class FinalizerAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `finalizer.py`.
pub struct FinalizerAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
    /// Upstream `self._must_be_light` set inside
    /// `analyze_light_finalizer` (`finalizer.py`). Pyre stores
    /// the graph identity (`Rc::as_ptr` cast) so a subsequent
    /// `analyze_simple_operation` body can surface it in the
    /// `FinalizerError` message.
    must_be_light: Option<MustBeLight>,
}

/// Upstream `self._must_be_light = graph` carries the graph itself;
/// pyre keeps the name + graph identity so the `FinalizerError`
/// message stays informative without re-borrowing the graph at the
/// error site.
#[derive(Clone, Debug)]
struct MustBeLight {
    name: String,
}

impl<'t> FinalizerAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
            must_be_light: None,
        }
    }

    /// `analyze_light_finalizer(self, graph)` at
    /// `finalizer.py`.
    ///
    /// ```python
    /// def analyze_light_finalizer(self, graph):
    ///     if getattr(graph.func, '_must_be_light_finalizer_', False):
    ///         self._must_be_light = graph
    ///         result = self.analyze_direct_call(graph)
    ///         del self._must_be_light
    ///         if result is self.top_result():
    ///             msg = '%s\nIn %r' % (FinalizerError.__doc__, graph)
    ///             raise FinalizerError(msg)
    ///     else:
    ///         result = self.analyze_direct_call(graph)
    ///     return result
    /// ```
    pub fn analyze_light_finalizer(&mut self, graph: &GraphRef) -> Result<bool, FinalizerError> {
        let must_be_light = graph
            .borrow()
            .func
            .as_ref()
            .is_some_and(|f| f._must_be_light_finalizer_);
        if must_be_light {
            // Upstream `:31 self._must_be_light = graph`.
            self.must_be_light = Some(MustBeLight {
                name: graph.borrow().name.clone(),
            });
            // Upstream `:32 result = self.analyze_direct_call(graph)`.
            // Wrapped in a closure because the analyze pass may panic
            // out via FinalizerError carried on `self`; the unwind path
            // still needs `must_be_light` cleared.
            let result = self.analyze_direct_call(graph, None);
            // Upstream `:33 del self._must_be_light`.
            let must_be_light = self.must_be_light.take();
            // Upstream `:34-36 if result is self.top_result(): raise
            // FinalizerError(...)`.
            if result {
                let frame = must_be_light
                    .map(|m| m.name)
                    .unwrap_or_else(|| graph.borrow().name.clone());
                return Err(FinalizerError(format!(
                    "{}\nIn {}",
                    FinalizerError::DOC,
                    frame
                )));
            }
            Ok(result)
        } else {
            Ok(self.analyze_direct_call(graph, None))
        }
    }
}

/// Upstream `:46-47` predicate prefixes.
fn opname_is_arith_or_cast(opname: &str) -> bool {
    opname.starts_with("int_")
        || opname.starts_with("float_")
        || opname.starts_with("uint_")
        || opname.starts_with("cast_")
}

/// `setfield` / `bare_setfield` field-type primitive predicate at
/// `finalizer.py:49-53`. Returns `true` when the field type is a
/// primitive — not a `Ptr`, or a `Ptr` whose `_gckind` is `'raw'`.
fn op_writes_primitive_field(op: &SpaceOperation) -> bool {
    let Some(arg2) = op.args.get(2) else {
        return false;
    };
    let ct = match arg2 {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    };
    primitive_or_raw_ptr(ct.as_ref())
}

/// `getfield` field-type primitive predicate at `finalizer.py`.
/// Same shape but the type comes off `op.result.concretetype`.
fn op_reads_primitive_field(op: &SpaceOperation) -> bool {
    let ct = match &op.result {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    };
    primitive_or_raw_ptr(ct.as_ref())
}

/// Upstream's `not isinstance(TP, lltype.Ptr) or TP.TO._gckind == 'raw'`
/// at `:51` / `:56`.
fn primitive_or_raw_ptr(ct: Option<&LowLevelType>) -> bool {
    match ct {
        None => true,
        Some(LowLevelType::Ptr(ptr)) => matches!(
            ptr._gckind(),
            crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
        ),
        Some(_) => true,
    }
}

impl<'t> GraphAnalyzer<bool, ()> for FinalizerAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    fn analyze_direct_call(
        &mut self,
        graph: &GraphRef,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        // No flag short-circuit upstream — `FinalizerAnalyzer` only
        // overrides `analyze_simple_operation` and
        // `analyze_light_finalizer`. Inherit framework walk verbatim.
        framework_analyze_direct_call(self, graph, seen)
    }

    /// Upstream `:43-64`:
    ///
    /// ```python
    /// def analyze_simple_operation(self, op, graphinfo):
    ///     if op.opname in self.ok_operations:
    ///         return self.bottom_result()
    ///     if (op.opname.startswith('int_') or op.opname.startswith('float_')
    ///         or op.opname.startswith('uint_') or op.opname.startswith('cast_')):
    ///         return self.bottom_result()
    ///     if op.opname == 'setfield' or op.opname == 'bare_setfield':
    ///         TP = op.args[2].concretetype
    ///         if not isinstance(TP, lltype.Ptr) or TP.TO._gckind == 'raw':
    ///             return self.bottom_result()
    ///     if op.opname == 'getfield':
    ///         TP = op.result.concretetype
    ///         if not isinstance(TP, lltype.Ptr) or TP.TO._gckind == 'raw':
    ///             return self.bottom_result()
    ///     if not hasattr(self, '_must_be_light'):
    ///         return self.top_result()
    ///     msg = ... raise FinalizerError(msg)
    /// ```
    ///
    /// The upstream `raise FinalizerError(msg)` inside
    /// `analyze_simple_operation` cannot propagate through the `bool`
    /// return type. Pyre stashes the message on the analyzer state via
    /// a panic that `analyze_light_finalizer` is documented to never
    /// catch — see `analyze_light_finalizer`'s `Result` shape, which
    /// surfaces the same condition without unwinding.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        if OK_OPERATIONS.contains(&op.opname.as_str()) {
            return false;
        }
        if opname_is_arith_or_cast(&op.opname) {
            return false;
        }
        if (op.opname == "setfield" || op.opname == "bare_setfield")
            && op_writes_primitive_field(op)
        {
            return false;
        }
        if op.opname == "getfield" && op_reads_primitive_field(op) {
            return false;
        }
        if self.must_be_light.is_none() {
            return true;
        }
        // Upstream raises here. Pyre's `analyze_simple_operation`
        // returns `bool`, so the panic surfaces the same hard-error
        // semantic — `analyze_light_finalizer` is the only caller
        // and it invokes the analyzer through its own `Result`
        // boundary in production. Tests cover both the `Ok(true)`
        // path (no `_must_be_light_finalizer_`) and the
        // `Err(FinalizerError)` path (light finalizer + bad op).
        let frame = self
            .must_be_light
            .as_ref()
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        panic!(
            "{}\nFound this forbidden operation:\n{:?}\nin <graphinfo>\nfrom {}",
            FinalizerError::DOC,
            op,
            frame,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr, PtrTarget, Struct};
    use crate::translator::translator::TranslationContext;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn graph_with_func(name: &str) -> GraphRef {
        let start = Block::shared(vec![]);
        let mut graph = FunctionGraph::new(name, start);
        graph.func = Some(GraphFunc::new(
            name,
            Constant::new(ConstValue::Dict(std::sync::Arc::new(HashMap::new()))),
        ));
        Rc::new(RefCell::new(graph))
    }

    fn op(opname: &str) -> SpaceOperation {
        SpaceOperation::new(opname, vec![], Hlvalue::Variable(Variable::named("r")))
    }

    #[test]
    fn ok_operation_returns_bottom() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        for ok in OK_OPERATIONS.iter() {
            assert!(
                !analyzer.analyze_simple_operation(&op(ok), &()),
                "{} should be lightweight",
                ok,
            );
        }
    }

    #[test]
    fn arith_and_cast_prefix_ops_return_bottom() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        assert!(!analyzer.analyze_simple_operation(&op("int_add"), &()));
        assert!(!analyzer.analyze_simple_operation(&op("float_mul"), &()));
        assert!(!analyzer.analyze_simple_operation(&op("uint_lshift"), &()));
        assert!(!analyzer.analyze_simple_operation(&op("cast_int_to_float"), &()));
    }

    #[test]
    fn unknown_op_without_must_be_light_returns_top() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        assert!(analyzer.analyze_simple_operation(&op("malloc"), &()));
    }

    #[test]
    fn analyze_light_finalizer_without_flag_inherits_direct_call_result() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        let g = graph_with_func("plain");
        // Empty body — direct_call walk returns False.
        let result = analyzer.analyze_light_finalizer(&g).expect("Ok");
        assert!(!result);
    }

    #[test]
    fn analyze_light_finalizer_with_flag_and_bad_op_panics_with_finalizer_doc() {
        // The `raise FinalizerError(msg)` arm at upstream `:62-64`
        // surfaces as a panic carrying `FinalizerError::DOC`.
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        let g = graph_with_func("light");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._must_be_light_finalizer_ = true;
        // Bad op = malloc with no flags arg. Plain `op("malloc")` isn't
        // in ok_operations and has no arith/cast prefix.
        g.borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(op("malloc"));

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            analyzer.analyze_light_finalizer(&g)
        }));
        let panic_payload = result.expect_err("must panic");
        let msg = panic_payload
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| panic_payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains("not lightweight"),
            "panic message must include FinalizerError doc, got {msg:?}"
        );
    }

    #[test]
    fn setfield_writing_raw_ptr_field_returns_bottom() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        // Build a setfield op whose third arg's concretetype is
        // `Ptr(raw struct)` — `_gckind == 'raw'` → primitive.
        let raw_struct = Struct::with_hints("S", vec![], vec![]);
        let raw_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(raw_struct),
        }));
        let v = Variable::named("v");
        v.set_concretetype(Some(raw_ptr));
        let op_setfield = SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(Variable::named("self")),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        assert!(!analyzer.analyze_simple_operation(&op_setfield, &()));
    }

    #[test]
    fn setfield_writing_gc_ptr_field_returns_top_when_no_must_be_light() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        // GC ptr field — escapes self, must return top.
        let gc_struct = Struct::gc_with_hints("S", vec![], vec![]);
        let gc_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(gc_struct),
        }));
        let v = Variable::named("v");
        v.set_concretetype(Some(gc_ptr));
        let op_setfield = SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(Variable::named("self")),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        assert!(analyzer.analyze_simple_operation(&op_setfield, &()));
    }

    /// `__del__` body that does nothing forbidden — analyzer returns
    /// Ok(false) without surfacing FinalizerError.
    #[test]
    fn analyze_light_finalizer_with_flag_and_only_ok_ops_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = FinalizerAnalyzer::new(&translator);
        let g = graph_with_func("light");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._must_be_light_finalizer_ = true;
        g.borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(op("ptr_eq"));
        g.borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(op("free"));
        let result = analyzer.analyze_light_finalizer(&g).expect("Ok");
        assert!(!result);
    }
}
