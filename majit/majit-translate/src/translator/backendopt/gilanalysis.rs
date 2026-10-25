//! Port of `rpython/translator/backendopt/gilanalysis.py`.
//!
//! Not an optimisation. Upstream walks user graphs to verify any
//! function tagged `@no_release_gil` does not transitively call into
//! code that releases the GIL, raising a hard error when the invariant
//! breaks (`gilanalysis.py:50-51`).
//!
//! majit/pyre deliberately has no process-wide GIL. The module keeps
//! the upstream filename, public API, and attribute spelling for
//! line-by-line parity, but the local invariant is not literally "can
//! release the GIL". A graph marked `_no_release_gil_` is treated as a
//! freethreaded "no thread-safepoint" region: no transitive callee may
//! close the stack, break a transaction, block in an unknown external
//! function, or otherwise cross a boundary where another thread/GC
//! state could be observed.
//!
//! The transitive call-graph walk is supplied by
//! [`super::graphanalyze::BoolGraphAnalyzer`] — an alias for
//! `GraphAnalyzer<bool, _>`. [`GilAnalyzer`] is the corresponding
//! subclass at `gilanalysis.py`, overriding
//! `analyze_simple_operation` / `analyze_external_call` to always
//! return `False` and `analyze_direct_call` to short-circuit on the
//! per-`func` `_gctransformer_hint_close_stack_` /
//! `_transaction_break_` flags.

use crate::flowspace::model::{FunctionGraph, GraphRef, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{
    Dependency, DependencyTracker, GraphAnalyzer, framework_analyze_direct_call,
};
use crate::translator::rtyper::lltypesystem::lltype::_func;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// `class GilAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `gilanalysis.py`.
///
/// The name is intentionally historical. Upstream's class asks
/// "can this path release the GIL?". In pyre's freethreaded runtime
/// there is no GIL, so this implementation asks "can this path cross
/// a thread-safepoint boundary?" while keeping the RPython method and
/// attribute surface intact.
///
/// Per upstream:
///
/// * `analyze_external_call(self, op, seen=None)` always returns
///   `False` (`:23-24`). pyre intentionally adapts this one hook:
///   an unresolved external call in a no-thread-safepoint region is
///   a hazard unless later effect information proves it cannot block
///   or schedule.
/// * `analyze_simple_operation(self, op, graphinfo)` always returns
///   `False` (`:26-27`).
/// * `analyze_direct_call(self, graph, seen=None)` first inspects
///   `graph.func._gctransformer_hint_close_stack_` and
///   `graph.func._transaction_break_` (`:9-18`) — either flag set
///   forces the GIL-release verdict — and otherwise falls through to
///   the inherited `BoolGraphAnalyzer.analyze_direct_call` walk.
///
pub struct GilAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> GilAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }

    /// Upstream `:11-18`: `getattr(func, '_gctransformer_hint_close_stack_',
    /// False)` / `getattr(func, '_transaction_break_', False)`.
    ///
    /// In freethreaded pyre these are not GIL-release markers; they
    /// are safepoint-like hazards that cannot appear below a graph
    /// carrying `_no_release_gil_`.
    fn flagged(graph: &FunctionGraph) -> bool {
        graph
            .func
            .as_ref()
            .is_some_and(|func| func._gctransformer_hint_close_stack_ || func._transaction_break_)
    }
}

impl<'t> GraphAnalyzer<bool, ()> for GilAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:26-27` — every simple op is GIL-safe.
    fn analyze_simple_operation(&mut self, _op: &SpaceOperation, _graphinfo: &()) -> bool {
        false
    }

    /// Upstream `:23-24` — every external call is GIL-safe.
    ///
    /// pyre freethreading adaptation: unknown external calls are
    /// treated as thread-safepoint hazards. This is deliberately
    /// more conservative than upstream's GIL-only rule; once
    /// low-level effect information exposes a "cannot block / cannot
    /// schedule / cannot collect" bit, this hook can admit those
    /// calls explicitly.
    fn analyze_external_call(
        &mut self,
        _funcobj: &_func,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        true
    }

    /// Upstream `:9-21`. Flag-driven short-circuit, otherwise
    /// delegate to the framework's recursive walk via the free
    /// `framework_analyze_direct_call` super-call analogue.
    fn analyze_direct_call(
        &mut self,
        graph: &GraphRef,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        if Self::flagged(&graph.borrow()) {
            return true;
        }
        // Upstream `:20-21 return
        // graphanalyze.BoolGraphAnalyzer.analyze_direct_call(self,
        // graph, seen)`.
        framework_analyze_direct_call(self, graph, seen)
    }
}

/// RPython `gilanalysis.analyze(graphs, translator)` at
/// `gilanalysis.py`.
///
/// Iterates each graph. When `func._no_release_gil_` is set,
/// constructs a [`GilAnalyzer`] and runs `analyze_direct_call`
/// against the graph. A `True` result raises (upstream
/// `:50-51 raise Exception("'no_release_gil' function can release
/// the GIL: %s\n%s" % ...)`); the Rust port surfaces this as a
/// `TaskError`.
///
pub fn analyze(graphs: &[GraphRef], translator: &TranslationContext) -> Result<(), TaskError> {
    let mut analyzer = GilAnalyzer::new(translator);
    for graph in graphs {
        let g = graph.borrow();
        if no_release_gil(&g) {
            // Upstream `:34 if gilanalyzer.analyze_direct_call(graph)`.
            drop(g);
            if analyzer.analyze_direct_call(graph, None) {
                let name = graph.borrow().name.clone();
                return Err(TaskError {
                    message: format!(
                        "gilanalysis.py:50 'no_release_gil' function {name} \
                         crosses a freethreaded thread-safepoint boundary"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// RPython `getattr(func, '_no_release_gil_', False)` at
/// `gilanalysis.py:33`.
///
/// Despite the historical name, pyre reads the attribute as a
/// no-thread-safepoint contract.
fn no_release_gil(graph: &FunctionGraph) -> bool {
    graph
        .func
        .as_ref()
        .is_some_and(|func| func._no_release_gil_)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphFunc, GraphKey, Hlvalue, SpaceOperation,
        Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use crate::translator::translator::TranslationContext;
    use indexmap::IndexMap;
    use std::cell::RefCell;
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

    fn direct_call_to(graph: Option<&GraphRef>) -> SpaceOperation {
        let graph_key = graph.map(|graph| GraphKey::of(graph).as_usize());
        let ptr = lltype::functionptr(
            lltype::FuncType {
                args: Vec::new(),
                result: lltype::LowLevelType::Void,
            },
            "callee",
            graph_key,
            None,
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
    fn analyze_passes_when_no_graph_carries_no_release_gil() {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v)]);
        let graph = FunctionGraph::new("entry", start);
        let graph = Rc::new(RefCell::new(graph));

        let translator = TranslationContext::new();
        assert!(analyze(&[graph], &translator).is_ok());
    }

    #[test]
    fn no_release_gil_graph_rejects_close_stack_callee() {
        let entry = graph_with_func("entry");
        let callee = graph_with_func("callee");
        entry
            .borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(direct_call_to(Some(&callee)));
        entry.borrow_mut().func.as_mut().unwrap()._no_release_gil_ = true;
        callee
            .borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._gctransformer_hint_close_stack_ = true;

        let translator = TranslationContext::new();
        translator
            .graphs
            .borrow_mut()
            .extend([entry.clone(), callee]);

        let err = analyze(&[entry], &translator).unwrap_err();
        assert!(err.message.contains("thread-safepoint boundary"));
    }

    #[test]
    fn no_release_gil_graph_allows_plain_callee() {
        let entry = graph_with_func("entry");
        let callee = graph_with_func("callee");
        entry
            .borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(direct_call_to(Some(&callee)));
        entry.borrow_mut().func.as_mut().unwrap()._no_release_gil_ = true;

        let translator = TranslationContext::new();
        translator
            .graphs
            .borrow_mut()
            .extend([entry.clone(), callee]);

        assert!(analyze(&[entry], &translator).is_ok());
    }

    #[test]
    fn no_release_gil_graph_rejects_external_direct_call() {
        let entry = graph_with_func("entry");
        entry
            .borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(direct_call_to(None));
        entry.borrow_mut().func.as_mut().unwrap()._no_release_gil_ = true;

        let translator = TranslationContext::new();
        translator.graphs.borrow_mut().push(entry.clone());

        assert!(analyze(&[entry], &translator).is_err());
    }
}
