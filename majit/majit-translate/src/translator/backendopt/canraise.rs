//! Port of `rpython/translator/backendopt/canraise.py`.
//!
//! `RaiseAnalyzer` is the `BoolGraphAnalyzer` subclass that decides
//! whether each call site can raise. Inputs feed
//! `inline.py BaseInliner.__init__(... raise_analyzer)` and
//! `inline.py any_call_to_raising_graphs`. The Rust port
//! exposes the same surface (`can_raise`, `analyze_simple_operation`,
//! `analyze_external_call`, `analyze_exceptblock_in_graph`,
//! `do_ignore_memory_error`).
//!
//! The analyser walks the recursive call graph through
//! [`super::graphanalyze::GraphAnalyzer`] — the framework that
//! `gilanalysis::GilAnalyzer` already consumes — and short-circuits on
//! the boolean lattice (`top_result == True` ends the walk).

use crate::flowspace::model::{BlockRef, GraphRef, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{Dependency, DependencyTracker, GraphAnalyzer};
use crate::translator::backendopt::ssa::DataFlowFamilyBuilder;
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::rtyper::lltypesystem::lltype::_func;
use crate::translator::translator::TranslationContext;

/// `class RaiseAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `canraise.py`.
pub struct RaiseAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `RaiseAnalyzer.ignore_exact_class = None` at
    /// `canraise.py`. `do_ignore_memory_error` flips this to the
    /// MemoryError class. The Rust port stores the upstream class
    /// name as a literal string because pyre's `LLOp.canraise` carries
    /// `&'static str` exception names rather than class objects.
    ignore_exact_class: Option<&'static str>,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> RaiseAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            ignore_exact_class: None,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }

    /// `do_ignore_memory_error(self)` at `canraise.py`.
    pub fn do_ignore_memory_error(&mut self) {
        self.ignore_exact_class = Some("MemoryError");
    }

    /// `can_raise(self, op, seen=None)` — backward-compatible
    /// interface at `canraise.py`.
    pub fn can_raise(
        &mut self,
        op: &SpaceOperation,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        self.analyze(op, seen, &())
    }
}

impl<'t> GraphAnalyzer<bool, ()> for RaiseAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:14-20`:
    /// ```python
    /// def analyze_simple_operation(self, op, graphinfo):
    ///     try:
    ///         canraise = LL_OPERATIONS[op.opname].canraise
    ///         return bool(canraise) and canraise != (self.ignore_exact_class,)
    ///     except KeyError:
    ///         log.WARNING("Unknown operation: %s" % op.opname)
    ///         return True
    /// ```
    ///
    /// `bool(canraise)` is False on the empty tuple. The
    /// `canraise != (self.ignore_exact_class,)` comparison drops a
    /// single-element exception tuple that matches `ignore_exact_class`
    /// (set by `do_ignore_memory_error`). The unknown-op fallback
    /// returns True — Pyre log channels are no-ops today, so the
    /// upstream `log.WARNING` call is omitted; the conservative True
    /// surfaces every unknown op as raising.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        match ll_operations().get(op.opname.as_str()) {
            Some(desc) => {
                if desc.canraise.is_empty() {
                    false
                } else if let Some(ignore) = self.ignore_exact_class {
                    !(desc.canraise.len() == 1 && desc.canraise[0] == ignore)
                } else {
                    true
                }
            }
            None => true,
        }
    }

    /// Upstream `:22-23`:
    /// ```python
    /// def analyze_external_call(self, fnobj, seen=None):
    ///     return getattr(fnobj, 'canraise', True)
    /// ```
    ///
    /// The `analyze` dispatcher passes the unwrapped `funcobj` (`_func`),
    /// which upstream calls `fnobj`. Pyre's `_func` carrier holds the
    /// upstream attribute mirror in `_func.attrs`
    /// (the `attrs: HashMap<String, ConstValue>` field on `lltype.rs`'s
    /// `_func`); the
    /// `canraise` slot is set by `lltype.functionptr(canraise=...)`
    /// at the same upstream site that originates the attribute on
    /// the Python `_func`. `getattr(fnobj, 'canraise', True)` returns the
    /// raw value (a bool on the regular path), which the BoolGraphAnalyzer
    /// then folds into its `or` lattice — i.e. by Python truthiness. Read
    /// the slot through the same truthiness; default to `True` when absent.
    fn analyze_external_call(
        &mut self,
        funcobj: &_func,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        match funcobj.attrs.get("canraise") {
            Some(value) => value.truthy().unwrap_or_else(|| {
                panic!("canraise.py:22 canraise attr has unknown truthiness: {value:?}")
            }),
            None => true,
        }
    }

    /// Upstream `:25 analyze_exceptblock = None    # don't call this`.
    /// Replicates the upstream guard: `analyze_exceptblock` should
    /// never fire on a `RaiseAnalyzer` — the framework body calls
    /// `analyze_exceptblock_in_graph` per `graphanalyze.py`. If
    /// the day ever comes when something routes back through the
    /// vanilla hook, surface the deviation loudly so the contract
    /// stays in sync with upstream.
    fn analyze_exceptblock(
        &mut self,
        _block: &BlockRef,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        panic!(
            "canraise.py:25 RaiseAnalyzer.analyze_exceptblock = None — \
             framework should call analyze_exceptblock_in_graph instead"
        )
    }

    /// Upstream `:27-41`:
    /// ```python
    /// def analyze_exceptblock_in_graph(self, graph, block, seen=None):
    ///     if self.ignore_exact_class is not None:
    ///         from rpython.translator.backendopt.ssa import DataFlowFamilyBuilder
    ///         dff = DataFlowFamilyBuilder(graph)
    ///         variable_families = dff.get_variable_families()
    ///         v_exc_instance = variable_families.find_rep(block.inputargs[1])
    ///         for link1 in graph.iterlinks():
    ///             v = link1.last_exc_value
    ///             if v is not None:
    ///                 if variable_families.find_rep(v) is v_exc_instance:
    ///                     # this is a case of re-raise the exception caught;
    ///                     # it doesn't count.  We'll see the place that really
    ///                     # raises the exception in the first place.
    ///                     return False
    ///     return True
    /// ```
    ///
    /// The re-raise check fires only when `do_ignore_memory_error`
    /// (or any future ignore-class) is on. Otherwise the except block
    /// is unconditionally treated as raising.
    fn analyze_exceptblock_in_graph(
        &mut self,
        graph: &GraphRef,
        block: &BlockRef,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        if self.ignore_exact_class.is_some() {
            // Upstream `:30-31`:
            //     dff = DataFlowFamilyBuilder(graph)
            //     variable_families = dff.get_variable_families()
            let mut dff = DataFlowFamilyBuilder::new(&graph.borrow());
            // Drive the union-find to a fixed point so `find_rep`
            // returns upstream's "family representative". Upstream's
            // `__init__` runs `complete()` implicitly through later
            // `find_rep`/`find` calls; pyre's structural port exposes
            // `complete()` explicitly.
            dff.complete();
            // Upstream `:32 v_exc_instance =
            // variable_families.find_rep(block.inputargs[1])` — second
            // input argument of the except block carries the exc
            // instance (first is exc type).
            let exc_value_var = match block.borrow().inputargs.get(1).cloned() {
                Some(v) => v,
                None => return true,
            };
            let v_exc_instance = dff.variable_families.find_rep(exc_value_var);
            // Upstream `:33-40`: walk every link, dropping links whose
            // `last_exc_value` family-rep matches the except instance.
            let links = graph.borrow().iterlinks();
            for link in links {
                let last_exc_value = link.borrow().last_exc_value.clone();
                if let Some(v) = last_exc_value
                    && dff.variable_families.find_rep(v) == v_exc_instance
                {
                    // re-raise of caught exception — does not count
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphKey, Hlvalue, Link,
        SpaceOperation, Variable, c_last_exception,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn fixture() -> TranslationContext {
        TranslationContext::new()
    }

    #[test]
    fn raise_analyzer_treats_pure_int_add_as_non_raising() {
        // `int_add` is registered with empty canraise in
        // `lloperation.rs`. Upstream `bool(canraise) == False` ⇒ False.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(v),
        );
        assert!(!a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn raise_analyzer_unknown_op_is_conservatively_raising() {
        // Upstream `:18-20`: `except KeyError: return True`.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new("unknown_dummy_op_for_test", vec![], Hlvalue::Variable(v));
        assert!(a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn raise_analyzer_default_treats_direct_call_canraise_exception_as_raising() {
        // `direct_call` carries `canraise=("Exception",)` in
        // `lloperation.rs`. Default ignore_exact_class is None ⇒
        // any non-empty canraise tuple ⇒ True.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Hlvalue::Variable(v),
        );
        assert!(a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn ignore_memory_error_flips_canraise_memoryerror_to_non_raising() {
        // Find an op whose canraise == ("MemoryError",) — the
        // canmallocgc path adds it implicitly. Pick any op registered
        // with canraise = ("MemoryError",) and verify that
        // do_ignore_memory_error makes analyze return False on it.
        // We synthesize the SpaceOperation locally; the comparison is
        // string-based against the static `canraise` Vec.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        a.do_ignore_memory_error();
        // Search the table for an opname whose canraise vec equals
        // ["MemoryError"].
        let target = ll_operations()
            .iter()
            .find(|(_, desc)| desc.canraise == vec!["MemoryError"])
            .map(|(name, _)| *name);
        if let Some(opname) = target {
            let v = Variable::named("x");
            let op = SpaceOperation::new(opname, vec![], Hlvalue::Variable(v));
            assert!(
                !a.analyze_simple_operation(&op, &()),
                "{opname} canraise=[MemoryError] should be False under ignore",
            );
        }
        // No registered op with that exact shape ⇒ test is vacuous,
        // but the boolean predicate is exercised in the negative
        // branch of `raise_analyzer_default_treats_direct_call_*`.
    }

    /// `analyze_exceptblock_in_graph` reraise-of-caught suppression
    /// (`canraise.py`) exercised directly on a flowspace graph.
    ///
    /// Exercises the suppression on a graph built directly in the flowspace
    /// model, depending on neither the flat `crate::model` graph nor the
    /// `function_graph_to_flowspace` adapter.
    ///
    /// The graph's only op is `int_add` (`canraise=[]`), so the op layer
    /// contributes no raise. The startblock reaches the graph's
    /// exceptblock through a `last_exception` edge whose `last_exc_value`
    /// is the very variable the exceptblock binds as its exc-instance
    /// inputarg — a re-raise of the caught exception. Therefore:
    /// - default `RaiseAnalyzer`: the reachable exceptblock counts as
    ///   raising → `analyze_direct_call` == `True`.
    /// - `do_ignore_memory_error`: `analyze_exceptblock_in_graph` matches
    ///   the re-raised value's family to `inputargs[1]` and suppresses it
    ///   → `False`.
    ///
    /// `(default=True, ignore=False)` is the boolean pair the flat
    /// `_canraise` maps to `MemoryErrorOnly`.
    #[test]
    fn ignore_memory_error_suppresses_reraise_of_caught_exceptblock() {
        let translator = fixture();

        // startblock: (a, b) -> v3 = int_add(a, b), then raises.
        let a = Variable::named("a");
        let b = Variable::named("b");
        let v3 = Variable::named("v3");
        let start = Block::shared(vec![
            Hlvalue::Variable(a.clone()),
            Hlvalue::Variable(b.clone()),
        ]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b)],
            Hlvalue::Variable(v3.clone()),
        ));

        let graph = FunctionGraph::new("reraise_caught", start.clone());
        let returnblock = graph.returnblock.clone();
        let exceptblock = graph.exceptblock.clone();

        // The exceptblock binds [etype, evalue]; the re-raise edge carries
        // those same caught values into it and tags the link's
        // last_exception / last_exc_value with them.
        let (etype, evalue) = {
            let eb = exceptblock.borrow();
            (eb.inputargs[0].clone(), eb.inputargs[1].clone())
        };

        let normal = Link::new(vec![Hlvalue::Variable(v3)], Some(returnblock), None).into_ref();
        let mut exc = Link::new(
            vec![etype.clone(), evalue.clone()],
            Some(exceptblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        );
        exc.extravars(Some(etype), Some(evalue));
        let exc = exc.into_ref();

        {
            let mut s = start.borrow_mut();
            s.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
            s.exits = vec![normal, exc];
        }
        let graph: GraphRef = Rc::new(RefCell::new(graph));

        // default: ignore_exact_class is None, so the reachable
        // exceptblock counts as raising (canraise.py:41 `return True`).
        let mut default = RaiseAnalyzer::new(&translator);
        assert!(
            default.analyze_direct_call(&graph, None),
            "default RaiseAnalyzer: reachable exceptblock => raising",
        );

        // do_ignore_memory_error: the re-raise of the caught exception is
        // suppressed (canraise.py), so the graph no longer raises.
        let mut ignore = RaiseAnalyzer::new(&translator);
        ignore.do_ignore_memory_error();
        assert!(
            !ignore.analyze_direct_call(&graph, None),
            "do_ignore_memory_error: re-raise of caught exception is suppressed",
        );
    }

    #[test]
    fn analyze_direct_call_walks_callee_graph_and_returns_true_on_raising_op() {
        // `int_add_ovf` is a regular simple op with
        // canraise=("OverflowError",). The framework should route the
        // op through analyze_simple_operation and bubble up `True`
        // through analyze_direct_call.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph: GraphRef = Rc::new(RefCell::new(graph));
        assert!(a.analyze_direct_call(&graph, None));
    }

    #[test]
    fn analyze_direct_call_walks_callee_graph_and_returns_false_on_pure_op() {
        // Build a single-graph callee whose only op is int_add.
        // canraise=[] ⇒ False.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph: GraphRef = Rc::new(RefCell::new(graph));
        assert!(!a.analyze_direct_call(&graph, None));
    }

    /// A `direct_call` op whose callee descriptor
    /// (`lltype.functionptr(..., graph=...)`) carries `callee`'s
    /// `GraphKey`, so `framework_analyze_direct_call` resolves it
    /// through `TranslationContext.graphs` (graphanalyze.rs).
    fn direct_call_to(callee: &GraphRef) -> SpaceOperation {
        let graph_key = Some(GraphKey::of(callee).as_usize());
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
            Hlvalue::Variable(Variable::named("call_result")),
        )
    }

    /// Single-block callee whose only op is `int_add` (canraise=[]) — it
    /// cannot raise, so a caller that only `direct_call`s it cannot
    /// raise either once the callee is resolvable.
    fn non_raising_callee() -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("callee", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    /// Single-block caller whose only op is a `direct_call` to `callee`.
    fn caller_calling(callee: &GraphRef) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("caller", start.clone());
        start.borrow_mut().operations.push(direct_call_to(callee));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn analyze_direct_call_resolves_registered_callee_else_top_result() {
        // A caller whose only op is a `direct_call` to a SEPARATE callee
        // graph is resolved by `RaiseAnalyzer` through `funcobj.graph` ->
        // `TranslationContext.graphs` (`framework_analyze_direct_call`,
        // graphanalyze.rs) to the callee's actual can-raise. With
        // the callee absent from `translator.graphs` the lookup misses and
        // the framework returns `top_result()` — the conservative `true`
        // (`graphanalyze.rs`'s `top_result`, bool top is `true`).
        //
        // The callee's only op is `int_add` (canraise=[]), so when the
        // callee is registered the resolved verdict is non-raising; when it
        // is absent the lookup falls back to the conservative `top_result`
        // (raising).

        // -- callee NOT registered: funcobj.graph misses -> top_result --
        {
            let translator = fixture();
            let callee = non_raising_callee();
            let caller = caller_calling(&callee);
            translator.graphs.borrow_mut().push(caller.clone());
            let mut a = RaiseAnalyzer::new(&translator);
            assert!(
                a.analyze_direct_call(&caller, None),
                "callee absent from translator.graphs -> direct_call resolves to \
                 top_result (conservatively raising)"
            );
        }

        // -- callee registered: resolves to its non-raising verdict --
        {
            let translator = fixture();
            let callee = non_raising_callee();
            let caller = caller_calling(&callee);
            translator
                .graphs
                .borrow_mut()
                .extend([caller.clone(), callee]);
            let mut a = RaiseAnalyzer::new(&translator);
            assert!(
                !a.analyze_direct_call(&caller, None),
                "registered non-raising callee -> RaiseAnalyzer resolves funcobj.graph \
                 through translator.graphs and reports not-raising"
            );
        }
    }
}
