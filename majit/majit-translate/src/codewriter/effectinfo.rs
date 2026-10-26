//! Port of the `BoolGraphAnalyzer` subclasses in
//! `rpython/jit/codewriter/effectinfo.py VirtualizableAnalyzer`.
//!
//! The `EffectInfo` data class itself lives in `majit-ir`
//! (`majit/majit-ir/src/effectinfo.rs`) per the IR-extraction
//! adaptation — that crate cannot see `crate::flowspace` and so cannot
//! host the analyzer subclasses, which walk flowspace graphs through
//! [`super::super::translator::backendopt::graphanalyze::GraphAnalyzer`]
//! (the same framework `canraise::RaiseAnalyzer`,
//! `collectanalyze::CollectAnalyzer`, and `gilanalysis::GilAnalyzer`
//! consume). The analyzers therefore live beside the codewriter that
//! consumes them, matching their upstream module
//! (`jit/codewriter/effectinfo.py`).
//!
//! [`VirtualizableAnalyzer`] and [`QuasiImmutAnalyzer`] are pure
//! `analyze_simple_operation` opname checks; both inherit the default
//! `analyze_external_call` (`bottom_result()` = `False`,
//! `graphanalyze.py`). [`RandomEffectsAnalyzer`]
//! (`effectinfo.py`) overrides `analyze_external_call` to read
//! `funcobj.random_effects_on_gcobjs` off the flowspace funcobj attrs —
//! the same attribute mirror `collectanalyze::CollectAnalyzer` reads —
//! failing loud on a missing attr (upstream's direct attribute access)
//! and otherwise delegating to the base `_callbacks` join
//! (`bottom_result()` = `False` with no callbacks); its
//! `analyze_simple_operation` is always `False` (random effects arise
//! only from external calls).

pub use super::call::effectinfo_from_writeanalyze;
use crate::flowspace::model::SpaceOperation;
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{
    Dependency, DependencyTracker, GraphAnalyzer, framework_analyze_external_call,
};
use crate::translator::rtyper::lltypesystem::lltype::_func;
use crate::translator::translator::TranslationContext;
pub use majit_ir::effectinfo::compute_bitstrings;
pub use majit_ir::{
    CallInfoCollection, EffectInfo, UnsupportedFieldExc, consider_array, consider_struct,
    frozenset_or_none,
};

/// `class VirtualizableAnalyzer(BoolGraphAnalyzer)` at
/// `effectinfo.py`.
pub struct VirtualizableAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> VirtualizableAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }
}

impl<'t> GraphAnalyzer<bool, ()> for VirtualizableAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:402-404`:
    /// `return op.opname in ('jit_force_virtualizable', 'jit_force_virtual')`.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        matches!(
            op.opname.as_str(),
            "jit_force_virtualizable" | "jit_force_virtual"
        )
    }
}

/// `class QuasiImmutAnalyzer(BoolGraphAnalyzer)` at
/// `effectinfo.py`.
pub struct QuasiImmutAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> QuasiImmutAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }
}

impl<'t> GraphAnalyzer<bool, ()> for QuasiImmutAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:407-408`:
    /// `return op.opname == 'jit_force_quasi_immutable'`.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        op.opname == "jit_force_quasi_immutable"
    }
}

/// `class RandomEffectsAnalyzer(BoolGraphAnalyzer)` at
/// `effectinfo.py`.
pub struct RandomEffectsAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> RandomEffectsAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }
}

impl<'t> GraphAnalyzer<bool, ()> for RandomEffectsAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:411-415`:
    ///
    /// ```python
    /// def analyze_external_call(self, funcobj, seen=None):
    ///     if funcobj.random_effects_on_gcobjs:
    ///         return True
    ///     return super(RandomEffectsAnalyzer, self).analyze_external_call(
    ///         funcobj, seen)
    /// ```
    ///
    /// The `analyze` dispatcher passes the unwrapped `funcobj` (`_func`).
    /// `random_effects_on_gcobjs` is read off `_func.attrs` — the same
    /// attribute mirror `collectanalyze::CollectAnalyzer` reads.
    fn analyze_external_call(
        &mut self,
        funcobj: &_func,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        // `if funcobj.random_effects_on_gcobjs: return True` (:412-413) —
        // Python truthiness of the flag read off the `_func.attrs` mirror.
        // A missing attr is upstream's direct-attribute-access
        // AttributeError and fails loud.
        let value = match funcobj.attrs.get("random_effects_on_gcobjs") {
            Some(value) => value,
            None => panic!("effectinfo.py:412 funcobj.random_effects_on_gcobjs missing"),
        };
        if value.truthy().unwrap_or_else(|| {
            panic!("effectinfo.py:412 random_effects_on_gcobjs has unknown truthiness: {value:?}")
        }) {
            return true;
        }
        // `return super(RandomEffectsAnalyzer, self).analyze_external_call(
        //  funcobj, seen)` (:414-415) — the base walk: `bottom_result()`
        // (`False`) unless a `_callbacks` graph proves it.
        framework_analyze_external_call(self, funcobj, seen)
    }

    /// Upstream `:417-418`: `return False` — random effects arise only
    /// from external calls, never from a plain operation.
    fn analyze_simple_operation(&mut self, _op: &SpaceOperation, _graphinfo: &()) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphKey, GraphRef, Hlvalue, Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn graph_with_op(name: &str, opname: &str) -> GraphRef {
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new(name, start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            vec![],
            Hlvalue::Variable(Variable::named("v")),
        ));
        Rc::new(RefCell::new(graph))
    }

    fn empty_graph(name: &str) -> GraphRef {
        let start = Block::shared(vec![]);
        Rc::new(RefCell::new(FunctionGraph::new(name, start)))
    }

    fn direct_call_to(graph: &GraphRef) -> SpaceOperation {
        let graph_key = Some(GraphKey::of(graph).as_usize());
        let ptr = lltype::functionptr(
            lltype::FuncType {
                args: Vec::new(),
                result: lltype::LowLevelType::Void,
            },
            "callee",
            graph_key,
            None::<String>,
        );
        SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::LLPtr(
                Box::new(ptr),
            )))],
            Hlvalue::Variable(Variable::named("result")),
        )
    }

    #[test]
    fn virtualizable_analyzer_detects_force_virtualizable_op() {
        let graph = graph_with_op("entry", "jit_force_virtualizable");
        let translator = TranslationContext::new();
        let mut analyzer = VirtualizableAnalyzer::new(&translator);
        assert!(analyzer.analyze_direct_call(&graph, None));
    }

    #[test]
    fn virtualizable_analyzer_detects_force_virtual_op() {
        let graph = graph_with_op("entry", "jit_force_virtual");
        let translator = TranslationContext::new();
        let mut analyzer = VirtualizableAnalyzer::new(&translator);
        assert!(analyzer.analyze_direct_call(&graph, None));
    }

    #[test]
    fn virtualizable_analyzer_ignores_unrelated_op() {
        let graph = graph_with_op("entry", "int_add");
        let translator = TranslationContext::new();
        let mut analyzer = VirtualizableAnalyzer::new(&translator);
        assert!(!analyzer.analyze_direct_call(&graph, None));
    }

    #[test]
    fn virtualizable_analyzer_follows_transitive_call() {
        let entry = empty_graph("entry");
        let callee = graph_with_op("callee", "jit_force_virtualizable");
        entry
            .borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(direct_call_to(&callee));

        let translator = TranslationContext::new();
        translator
            .graphs
            .borrow_mut()
            .extend([entry.clone(), callee]);

        let mut analyzer = VirtualizableAnalyzer::new(&translator);
        assert!(analyzer.analyze_direct_call(&entry, None));
    }

    #[test]
    fn quasiimmut_analyzer_detects_force_quasi_immutable_op() {
        let graph = graph_with_op("entry", "jit_force_quasi_immutable");
        let translator = TranslationContext::new();
        let mut analyzer = QuasiImmutAnalyzer::new(&translator);
        assert!(analyzer.analyze_direct_call(&graph, None));
    }

    #[test]
    fn quasiimmut_analyzer_ignores_force_virtualizable_op() {
        let graph = graph_with_op("entry", "jit_force_virtualizable");
        let translator = TranslationContext::new();
        let mut analyzer = QuasiImmutAnalyzer::new(&translator);
        assert!(!analyzer.analyze_direct_call(&graph, None));
    }

    /// An external `_func` whose `attrs` carry exactly `attrs`. The
    /// `analyze` dispatcher hands the unwrapped funcobj to
    /// `analyze_external_call`. Mirrors `collectanalyze.rs`'s fixture.
    fn external_func_with_attrs(attrs: HashMap<String, ConstValue>) -> lltype::_func {
        let functype = lltype::FuncType {
            args: vec![],
            result: lltype::LowLevelType::Void,
        };
        lltype::_func::new(functype, "ext".to_string(), None, None::<String>, attrs)
    }

    #[test]
    fn random_effects_external_call_with_attr_returns_true() {
        // effectinfo.py — `funcobj.random_effects_on_gcobjs` → True.
        let translator = TranslationContext::new();
        let mut analyzer = RandomEffectsAnalyzer::new(&translator);
        let mut attrs = HashMap::new();
        attrs.insert(
            "random_effects_on_gcobjs".to_string(),
            ConstValue::Bool(true),
        );
        let funcobj = external_func_with_attrs(attrs);
        assert!(analyzer.analyze_external_call(&funcobj, None));
    }

    #[test]
    fn random_effects_external_call_with_false_attr_returns_false() {
        // effectinfo.py:414-415 — `random_effects_on_gcobjs = False` reaches
        // the super-call; with no `_callbacks` the base `GraphAnalyzer`
        // returns `bottom_result()` == False (graphanalyze.py).
        let translator = TranslationContext::new();
        let mut analyzer = RandomEffectsAnalyzer::new(&translator);
        let mut attrs = HashMap::new();
        attrs.insert(
            "random_effects_on_gcobjs".to_string(),
            ConstValue::Bool(false),
        );
        let funcobj = external_func_with_attrs(attrs);
        assert!(!analyzer.analyze_external_call(&funcobj, None));
    }

    #[test]
    #[should_panic(expected = "random_effects_on_gcobjs missing")]
    fn random_effects_external_call_missing_attr_fails_loud() {
        // Upstream reads `funcobj.random_effects_on_gcobjs` directly, so a
        // missing attr is not the same as a false attr.
        let translator = TranslationContext::new();
        let mut analyzer = RandomEffectsAnalyzer::new(&translator);
        let funcobj = external_func_with_attrs(HashMap::new());
        let _ = analyzer.analyze_external_call(&funcobj, None);
    }

    #[test]
    fn random_effects_simple_operation_always_false() {
        // effectinfo.py analyze_simple_operation — random effects arise only from external
        // calls, never from a plain operation.
        let translator = TranslationContext::new();
        let mut analyzer = RandomEffectsAnalyzer::new(&translator);
        for opname in [
            "int_add",
            "jit_force_virtualizable",
            "malloc",
            "gc_load_indexed",
        ] {
            let op = SpaceOperation::new(opname, vec![], Hlvalue::Variable(Variable::named("v")));
            assert!(!analyzer.analyze_simple_operation(&op, &()));
        }
    }
}
