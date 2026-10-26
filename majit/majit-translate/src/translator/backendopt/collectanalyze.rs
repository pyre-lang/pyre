//! Port of `rpython/translator/backendopt/collectanalyze.py`.
//!
//! `CollectAnalyzer` is the [`super::graphanalyze::BoolGraphAnalyzer`]
//! subclass that decides whether a call site can collect (i.e. trigger
//! a GC). Upstream consumers live in `rpython/memory/gctransform/`,
//! which pyre has not yet ported. This module is published as a parity
//! sibling alongside [`super::canraise`] and [`super::gilanalysis`] so
//! the surface lines up the day a consumer needs it.
//!
//! Per upstream:
//!
//! * `analyze_direct_call(self, graph, seen=None)` short-circuits to
//!   `bottom_result()` when the callee carries
//!   `_gctransformer_hint_cannot_collect_`, and to `top_result()`
//!   when it carries `_gctransformer_hint_close_stack_`. Otherwise
//!   falls through to the framework walk
//!   (`graphanalyze.BoolGraphAnalyzer.analyze_direct_call`).
//! * `analyze_external_call(self, funcobj, seen=None)` returns
//!   `top_result()` when `funcobj.random_effects_on_gcobjs` is set;
//!   otherwise delegates to the inherited
//!   `BoolGraphAnalyzer.analyze_external_call` — which is the base
//!   `GraphAnalyzer.analyze_external_call` (`bottom_result()` plus the
//!   `_callbacks` graph walk), NOT a conservative `top_result()` default.
//! * `analyze_simple_operation(self, op, graphinfo)` returns `True`
//!   for `malloc` / `malloc_varsize` with `flavor='gc'`, and
//!   otherwise reads `LL_OPERATIONS[opname].canmallocgc`.

use crate::flowspace::model::{ConstValue, FunctionGraph, GraphRef, Hlvalue, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{
    Dependency, DependencyTracker, GraphAnalyzer, framework_analyze_direct_call,
    framework_analyze_external_call,
};
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::rtyper::lltypesystem::lltype::_func;
use crate::translator::translator::TranslationContext;

/// `class CollectAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `collectanalyze.py`.
pub struct CollectAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> CollectAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }
}

/// Helper for the `func._gctransformer_hint_*` short-circuits at
/// upstream `:11-18`. Returns `Some(verdict)` when one of the two
/// hints is set and the framework should bypass the call-graph walk;
/// `None` when neither hint is present.
fn graph_hint_verdict(graph: &FunctionGraph) -> Option<bool> {
    let func = graph.func.as_ref()?;
    if func._gctransformer_hint_cannot_collect_ {
        return Some(false);
    }
    if func._gctransformer_hint_close_stack_ {
        return Some(true);
    }
    None
}

impl<'t> GraphAnalyzer<bool, ()> for CollectAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:9-20`:
    ///
    /// ```python
    /// def analyze_direct_call(self, graph, seen=None):
    ///     try:
    ///         func = graph.func
    ///     except AttributeError:
    ///         pass
    ///     else:
    ///         if getattr(func, '_gctransformer_hint_cannot_collect_', False):
    ///             return False
    ///         if getattr(func, '_gctransformer_hint_close_stack_', False):
    ///             return True
    ///     return graphanalyze.BoolGraphAnalyzer.analyze_direct_call(self,
    ///                                                               graph, seen)
    /// ```
    fn analyze_direct_call(
        &mut self,
        graph: &GraphRef,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        if let Some(verdict) = graph_hint_verdict(&graph.borrow()) {
            return verdict;
        }
        framework_analyze_direct_call(self, graph, seen)
    }

    /// Upstream `:21-25`:
    ///
    /// ```python
    /// def analyze_external_call(self, funcobj, seen=None):
    ///     if funcobj.random_effects_on_gcobjs:
    ///         return True
    ///     return graphanalyze.BoolGraphAnalyzer.analyze_external_call(
    ///         self, funcobj, seen)
    /// ```
    ///
    /// The `analyze` dispatcher passes the unwrapped `funcobj` (`_func`).
    /// `random_effects_on_gcobjs` is read off `_func.attrs` — the same
    /// attribute mirror that carries `canraise`.
    fn analyze_external_call(
        &mut self,
        funcobj: &_func,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        // `if funcobj.random_effects_on_gcobjs: return True` — Python
        // truthiness of the flag read off the `_func.attrs` mirror. A
        // missing attr is upstream's direct-attribute-access AttributeError
        // and fails loud; a present value (the `rffi.py:156`-normalised
        // bool on the regular path, or a raw `functionptr(...)` operand) is
        // evaluated through the same truthiness as the upstream `if`.
        let value = match funcobj.attrs.get("random_effects_on_gcobjs") {
            Some(value) => value,
            None => panic!("collectanalyze.py:22 funcobj.random_effects_on_gcobjs missing"),
        };
        if value.truthy().unwrap_or_else(|| {
            panic!(
                "collectanalyze.py:22 random_effects_on_gcobjs has unknown truthiness: {value:?}"
            )
        }) {
            return true;
        }
        // `return graphanalyze.BoolGraphAnalyzer.analyze_external_call(
        //      self, funcobj, seen)` — the base walk: `bottom_result()`
        // (`False`) unless a `_callbacks` graph proves collection.
        framework_analyze_external_call(self, funcobj, seen)
    }

    /// Upstream `:27-33`:
    ///
    /// ```python
    /// def analyze_simple_operation(self, op, graphinfo):
    ///     if op.opname in ('malloc', 'malloc_varsize'):
    ///         flags = op.args[1].value
    ///         return flags['flavor'] == 'gc'
    ///     else:
    ///         return (op.opname in LL_OPERATIONS and
    ///                 LL_OPERATIONS[op.opname].canmallocgc)
    /// ```
    ///
    /// Pyre stores `flags` as `ConstValue::Dict<ConstValue,
    /// ConstValue>`; the `flavor` key resolves to `ConstValue::UniStr`
    /// or `ConstValue::ByteStr` depending on the source. Both spellings
    /// are accepted to match upstream's Python dict semantics.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        if op.opname == "malloc" || op.opname == "malloc_varsize" {
            // Upstream `:29-30 flags = op.args[1].value;
            // return flags['flavor'] == 'gc'`. The reads are direct: a
            // missing `args[1]`, a non-constant operand, a non-dict value,
            // or an absent `'flavor'` key is malformed and fails loud
            // (upstream `IndexError` / `AttributeError` / `KeyError`). The
            // `== 'gc'` comparison itself is fail-soft: any other flavor
            // value is simply not a GC allocation.
            let Some(Hlvalue::Constant(c)) = op.args.get(1) else {
                panic!("collectanalyze.py:29 malloc op args[1] is not a constant");
            };
            let ConstValue::Dict(flags) = &c.value else {
                panic!(
                    "collectanalyze.py:29 malloc flags is not a dict ({:?})",
                    c.value
                );
            };
            let flavor = flags
                .get(&ConstValue::UniStr("flavor".to_string()))
                .or_else(|| flags.get(&ConstValue::ByteStr(b"flavor".to_vec())))
                .unwrap_or_else(|| {
                    panic!("collectanalyze.py:30 malloc flags missing 'flavor' key")
                });
            return matches!(flavor, ConstValue::UniStr(s) if s == "gc")
                || matches!(flavor, ConstValue::ByteStr(s) if s == b"gc");
        }
        // Upstream `:32-33`: opname in LL_OPERATIONS and canmallocgc.
        match ll_operations().get(op.opname.as_str()) {
            Some(desc) => desc.canmallocgc,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use crate::translator::translator::TranslationContext;
    use indexmap::IndexMap;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn graph_with_func(name: &str) -> GraphRef {
        let start = Block::shared(vec![]);
        let mut graph = FunctionGraph::new(name, start);
        graph.func = Some(GraphFunc::new(
            name,
            Constant::new(ConstValue::Dict(std::sync::Arc::new(IndexMap::new()))),
        ));
        Rc::new(RefCell::new(graph))
    }

    /// `flags={'flavor': 'gc'}` malloc op fixture.
    fn malloc_op(flavor: &str) -> SpaceOperation {
        let mut flags = IndexMap::new();
        flags.insert(
            ConstValue::UniStr("flavor".to_string()),
            ConstValue::UniStr(flavor.to_string()),
        );
        SpaceOperation::new(
            "malloc",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Constant(Constant::new(ConstValue::Dict(std::sync::Arc::new(flags)))),
            ],
            Hlvalue::Variable(Variable::named("result")),
        )
    }

    fn external_func_with_attrs(attrs: HashMap<String, ConstValue>) -> lltype::_func {
        let functype = lltype::FuncType {
            args: vec![],
            result: lltype::LowLevelType::Void,
        };
        lltype::_func::new(functype, "ext".to_string(), None, None, attrs)
    }

    #[test]
    fn cannot_collect_hint_short_circuits_to_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let g = graph_with_func("g");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._gctransformer_hint_cannot_collect_ = true;
        // Even with a body that would otherwise collect, the hint
        // forces False.
        g.borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(malloc_op("gc"));
        assert!(!analyzer.analyze_direct_call(&g, None));
    }

    #[test]
    fn close_stack_hint_short_circuits_to_true() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let g = graph_with_func("g");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._gctransformer_hint_close_stack_ = true;
        // Empty body — the hint alone forces True.
        assert!(analyzer.analyze_direct_call(&g, None));
    }

    #[test]
    fn malloc_with_gc_flavor_returns_true() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        assert!(analyzer.analyze_simple_operation(&malloc_op("gc"), &()));
    }

    #[test]
    fn malloc_with_raw_flavor_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        assert!(!analyzer.analyze_simple_operation(&malloc_op("raw"), &()));
    }

    #[test]
    fn unknown_opname_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "totally_made_up_op",
            vec![],
            Hlvalue::Variable(Variable::named("r")),
        );
        assert!(!analyzer.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn external_call_with_random_effects_on_gcobjs_returns_true() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);

        // Build a direct_call op with attrs = {"random_effects_on_gcobjs": True}.
        let mut attrs = HashMap::new();
        attrs.insert(
            "random_effects_on_gcobjs".to_string(),
            ConstValue::Bool(true),
        );
        let funcobj = external_func_with_attrs(attrs);
        assert!(analyzer.analyze_external_call(&funcobj, None));
    }

    #[test]
    fn external_call_with_random_effects_false_and_no_callbacks_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        // `random_effects_on_gcobjs = False` reaches the upstream
        // super-call. With no `_callbacks`, base `GraphAnalyzer`
        // returns `bottom_result() == False` (graphanalyze.py).
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
    fn external_call_missing_random_effects_fails_loud() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        // Upstream reads `funcobj.random_effects_on_gcobjs` directly,
        // so a missing attr is not the same as a false attr.
        let funcobj = external_func_with_attrs(HashMap::new());
        let _ = analyzer.analyze_external_call(&funcobj, None);
    }
}
