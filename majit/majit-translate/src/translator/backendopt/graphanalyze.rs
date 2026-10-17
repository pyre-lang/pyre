//! Port of `rpython/translator/backendopt/graphanalyze.py`.
//!
//! Generic dataflow framework that walks every reachable graph from
//! an entry, accumulating a result via subclass-supplied
//! `result_builder` / `add_to_result` / `join_two_results`. The
//! `BoolGraphAnalyzer` specialisation collapses the framework to the
//! boolean lattice (False ⊑ True, top-result short-circuits the walk)
//! — its concrete subclasses include
//! [`super::gilanalysis::GilAnalyzer`].
//!
//! Cycles are tolerated through [`DependencyTracker`], which uses
//! [`crate::tool::algo::unionfind::UnionFind`] to merge the analysis
//! result of every graph in a strongly-connected component (matching
//! upstream `graphanalyze.py`).

use crate::flowspace::model::{
    BlockRef, ConstValue, GraphKey, GraphRef, Hlvalue, LinkRef, SpaceOperation,
};
use crate::tool::algo::unionfind::{UnionFind, UnionFindInfo};
use crate::translator::rtyper::lltypesystem::lltype;
use crate::translator::translator::TranslationContext;

/// `Dependency` (`graphanalyze.py`). Per-graph cell holding
/// the accumulated analyser result. `merge_with_result` and `absorb`
/// match upstream verbatim.
#[derive(Clone)]
pub struct Dependency<R: AnalyzerResult> {
    result: R,
}

impl<R: AnalyzerResult> Dependency<R> {
    pub fn new(bottom: R) -> Self {
        Self { result: bottom }
    }

    /// `merge_with_result(self, result)` (`graphanalyze.py`).
    pub fn merge_with_result(&mut self, result: R) {
        self.result = R::join_two_results(self.result.clone(), result);
    }

    /// `absorb(self, other)` (`graphanalyze.py`). Upstream
    /// is `self.merge_with_result(other._result)`.
    pub fn absorb(&mut self, other: &Self) {
        self.merge_with_result(other.result.clone());
    }

    pub fn result(&self) -> R {
        self.result.clone()
    }
}

impl<R: AnalyzerResult> UnionFindInfo for Dependency<R> {
    /// `Dependency.absorb(other)` (`graphanalyze.py`). The
    /// pyre `UnionFindInfo` trait passes `other` by value so the
    /// absorbed partition's info is consumed; pyre's
    /// `merge_with_result` only needs `other.result`, so consume it.
    fn absorb(&mut self, other: Self) {
        self.merge_with_result(other.result);
    }
}

/// Analyser-result lattice. Concrete subclasses implement
/// `bottom_result` / `top_result` / `is_top_result` /
/// `result_builder` / `add_to_result` / `finalize_builder` /
/// `join_two_results` per upstream `graphanalyze.py`.
pub trait AnalyzerResult: Clone + 'static {
    /// `bottom_result()` (`graphanalyze.py`).
    fn bottom_result() -> Self;

    /// `top_result()` (`graphanalyze.py`).
    fn top_result() -> Self;

    /// `is_top_result(result)` (`graphanalyze.py`).
    fn is_top_result(result: &Self) -> bool;

    /// `result_builder()` (`graphanalyze.py`).
    fn result_builder() -> Self;

    /// `add_to_result(result, other)` (`graphanalyze.py`).
    fn add_to_result(result: Self, other: Self) -> Self;

    /// `finalize_builder(result)` (`graphanalyze.py`).
    fn finalize_builder(result: Self) -> Self;

    /// `join_two_results(result1, result2)`
    /// (`graphanalyze.py`).
    fn join_two_results(result1: Self, result2: Self) -> Self;
}

/// Boolean lattice for [`BoolGraphAnalyzer`] subclasses
/// (`graphanalyze.py`). `False` is the bottom; `True` is the
/// top and short-circuits every walk.
impl AnalyzerResult for bool {
    fn bottom_result() -> Self {
        false
    }

    fn top_result() -> Self {
        true
    }

    fn is_top_result(result: &Self) -> bool {
        *result
    }

    fn result_builder() -> Self {
        false
    }

    fn add_to_result(result: Self, other: Self) -> Self {
        Self::join_two_results(result, other)
    }

    fn finalize_builder(result: Self) -> Self {
        result
    }

    fn join_two_results(result1: Self, result2: Self) -> Self {
        result1 || result2
    }
}

/// Per-graph metadata cached by [`GraphAnalyzer::compute_graph_info`]
/// (`graphanalyze.py`). The default is `None` for every graph;
/// subclasses override to populate the slot. Pyre's `bool` lattice
/// passes through `()` because no concrete subclass uses graph info
/// today.
pub trait GraphInfo: Clone + Default + 'static {}

impl GraphInfo for () {}

/// `GraphAnalyzer` (`graphanalyze.py`). The Rust port uses a
/// trait so subclasses can be plain structs; the recursive walk
/// (`analyze_direct_call`, `analyze_indirect_call`, `analyze`,
/// `analyze_all`) lives in default methods so subclasses only need
/// to override the per-op hooks.
///
/// The trait is parameterised over the result lattice (`R`) and the
/// graph-info type (`I`). Subclasses pin both via the impl.
pub trait GraphAnalyzer<R: AnalyzerResult, I: GraphInfo>: Sized {
    // upstream-overridable hooks

    /// `compute_graph_info(graph)` (`graphanalyze.py`). Default
    /// is `None` (`I::default()`). Subclasses override to compute
    /// per-graph metadata once per `analyze_direct_call`.
    fn compute_graph_info(&mut self, _graph: &GraphRef) -> I {
        I::default()
    }

    /// `analyze_simple_operation(op, graphinfo)`
    /// (`graphanalyze.py`). No default — every concrete
    /// analyzer overrides this with the per-op verdict.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, graphinfo: &I) -> R;

    /// `analyze_external_call(funcobj, seen)`
    /// (`graphanalyze.py`):
    ///
    /// ```python
    /// def analyze_external_call(self, funcobj, seen=None):
    ///     result = self.bottom_result()
    ///     if hasattr(funcobj, '_callbacks'):
    ///         bk = self.translator.annotator.bookkeeper
    ///         for function in funcobj._callbacks.callbacks:
    ///             desc = bk.getdesc(function)
    ///             for graph in desc.getgraphs():
    ///                 result = self.join_two_results(
    ///                     result, self.analyze_direct_call(graph, seen))
    ///     return result
    /// ```
    ///
    /// Like upstream, this receives the unwrapped `funcobj` (`_func`):
    /// `analyze`'s `direct_call` arm performs the `op.args[0].value._obj`
    /// unwrap once (`graphanalyze.py`) and hands the result here, so
    /// the method surface matches `analyze_external_call(self, funcobj,
    /// seen)`.
    fn analyze_external_call(
        &mut self,
        funcobj: &lltype::_func,
        seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        framework_analyze_external_call(self, funcobj, seen)
    }

    /// `analyze_exceptblock(block, seen)`
    /// (`graphanalyze.py`). Default is `bottom_result()`.
    /// Subclasses (e.g. `canraise.RaiseAnalyzer.analyze_exceptblock`
    /// at `canraise.py = None`) keep the upstream `block` parameter
    /// so the surface stays compatible with the caller in
    /// [`framework_analyze_direct_call`] and with the
    /// `analyze_exceptblock_in_graph` default below.
    fn analyze_exceptblock(
        &mut self,
        _block: &BlockRef,
        _seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        R::bottom_result()
    }

    /// `analyze_exceptblock_in_graph(graph, block, seen)`
    /// (`graphanalyze.py`). Upstream default routes through
    /// [`Self::analyze_exceptblock`] — subclasses override this hook
    /// directly when they need the enclosing graph (canraise's
    /// `analyze_exceptblock_in_graph` at `canraise.py` walks
    /// `graph.iterlinks()` to suppress re-raise of caught
    /// exceptions). The driver in
    /// [`framework_analyze_direct_call`] calls the `_in_graph`
    /// variant per upstream `:155`.
    fn analyze_exceptblock_in_graph(
        &mut self,
        _graph: &GraphRef,
        block: &BlockRef,
        seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        self.analyze_exceptblock(block, seen)
    }

    /// `analyze_startblock(block, seen)`
    /// (`graphanalyze.py`). Default is `bottom_result()`.
    fn analyze_startblock(
        &mut self,
        _block: &BlockRef,
        _seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        R::bottom_result()
    }

    /// `analyze_link(exit, seen)` per the call site at
    /// `graphanalyze.py self.analyze_link(exit, seen)`. Upstream's
    /// declared signature (`:71 def analyze_link(self, graph, link)`)
    /// is at odds with the call site; the body just returns
    /// `bottom_result()` and ignores arguments, so no concrete
    /// subclass exercises the discrepancy. Pyre tracks the call-site
    /// arity (one link, the seen tracker).
    fn analyze_link(&mut self, _exit: &LinkRef, _seen: Option<&mut DependencyTracker<R>>) -> R {
        R::bottom_result()
    }

    /// Hand the framework the surrounding [`TranslationContext`] so
    /// `analyze_direct_call`'s `indirect_call` arm can resolve
    /// `Constant(graphs, Void)` keys back to graphs. Mirrors
    /// upstream's reliance on the implicit `self.translator` field
    /// (`graphanalyze.py`).
    fn translator(&self) -> &TranslationContext;

    /// Reach into the analyzer's internal `_analyzed_calls`
    /// `UnionFind`. Pyre cannot use upstream's instance-attribute
    /// pattern through a trait; the accessor lets the default
    /// `analyze_direct_call` implementation reuse the cache without
    /// the trait owning a concrete field.
    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<R>>;

    // dispatch

    /// `analyze(op, seen, graphinfo)` (`graphanalyze.py`).
    fn analyze(
        &mut self,
        op: &SpaceOperation,
        seen: Option<&mut DependencyTracker<R>>,
        graphinfo: &I,
    ) -> R {
        match op.opname.as_str() {
            "direct_call" => {
                // Upstream `:95-96 funcobj = op.args[0].value._obj`. The
                // read is direct: a `direct_call` always carries a constant
                // function pointer as `args[0]`. A missing arg, a
                // non-constant operand, or a non-pointer constant is
                // malformed and fails loud, matching upstream's `IndexError`
                // / `AttributeError` on `op.args[0].value._obj` (only the
                // `DelayedPointer` and null-pointer cases below are caught).
                let Some(arg0) = op.args.first() else {
                    panic!("graphanalyze.py:96 direct_call has no args[0] callee");
                };
                let Hlvalue::Constant(c) = arg0 else {
                    panic!("graphanalyze.py:96 direct_call args[0] is not a constant callee");
                };
                let ConstValue::LLPtr(f) = &c.value else {
                    panic!(
                        "graphanalyze.py:96 direct_call args[0] is not a function pointer ({:?})",
                        c.value
                    );
                };
                if !f.nonzero() {
                    // Upstream `funcobj is None`: a null call would
                    // crash and may be on a dead path, so return
                    // bottom rather than a hard hazard.
                    return R::bottom_result();
                }
                let funcobj = match f._obj() {
                    Ok(lltype::_ptr_obj::Func(funcobj)) => funcobj,
                    Ok(_) => return R::top_result(),
                    Err(lltype::DelayedPointer) => return R::top_result(),
                };
                // Upstream `:104-108`:
                //     if getattr(funcobj, 'external', None) is not None:
                //         x = self.analyze_external_call(funcobj, seen)
                //         ...
                //         return x
                // pyre's `_func` mirrors upstream `**attrs`-set members
                // through `_func.attrs`. `getattr(funcobj, 'external',
                // None) is not None` is False both when the attribute is
                // absent AND when its value is the literal `None`; the
                // Rust analogue must therefore reject
                // `Some(ConstValue::None)` as well as the missing-key
                // case. The `external='C'` kwarg from
                // `lltype.functionptr` (`rffi.py`) lands in
                // `attrs["external"]` as a non-`None` ConstValue.
                if !matches!(funcobj.attrs.get("external"), None | Some(ConstValue::None)) {
                    return self.analyze_external_call(&funcobj, seen);
                }
                // Upstream `:109-112`:
                //     try:
                //         graph = funcobj.graph
                //     except AttributeError:
                //         return self.top_result()
                // Upstream's `funcobj.graph` raises AttributeError when
                // the slot is absent. Pyre stores `None` in the same
                // slot for graphless internal funcobjs; collapse both
                // arms here — graph absent and not-external upstream
                // surfaces as `top_result()`, which mirrors pyre's
                // current behaviour for an opaque funcobj.
                let Some(graph_key) = funcobj.graph else {
                    return R::top_result();
                };
                let graph = {
                    let trans_graphs = self.translator().graphs.borrow();
                    trans_graphs
                        .iter()
                        .find(|graph| GraphKey::of(graph).as_usize() == graph_key)
                        .cloned()
                };
                let Some(graph) = graph else {
                    return R::top_result();
                };
                self.analyze_direct_call(&graph, seen)
            }
            "indirect_call" => {
                // Upstream `:117-126`: trailing `Constant(graphs,
                // Void)` arg carries the candidate graph list.
                // Resolve through `TranslationContext.graphs` and
                // iterate. None / unknown surfaces as
                // `top_result()` (matching upstream `:120-122`).
                // `PtrRepr.rtype_simple_call` (`rptr.py`) appends
                // `inputconst(Void, None)` when the callee is not a
                // Constant; `guess_call_kind` (`call.py`) then
                // classifies `graphs is None` as residual.
                let last = op.args.last();
                let Some(Hlvalue::Constant(c)) = last else {
                    return R::top_result();
                };
                if matches!(c.value, ConstValue::None) {
                    return R::top_result();
                }
                let ConstValue::Graphs(keys) = &c.value else {
                    return R::top_result();
                };
                let graphs: Vec<GraphRef> = {
                    let trans_graphs = self.translator().graphs.borrow();
                    let mut graphs = Vec::with_capacity(keys.len());
                    for key in keys {
                        let Some(graph) = trans_graphs
                            .iter()
                            .find(|g| GraphKey::of(g).as_usize() == *key)
                            .cloned()
                        else {
                            return R::top_result();
                        };
                        graphs.push(graph);
                    }
                    graphs
                };
                self.analyze_indirect_call(&graphs, seen)
            }
            _ => self.analyze_simple_operation(op, graphinfo),
        }
    }

    /// `analyze_direct_call(graph, seen)`
    /// (`graphanalyze.py`). Subclasses that need to wrap
    /// the framework body (e.g. flag-based short-circuit before
    /// the recursive walk, mirroring upstream's `super().analyze_direct_call`
    /// idiom) call [`framework_analyze_direct_call`] directly.
    fn analyze_direct_call(
        &mut self,
        graph: &GraphRef,
        seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        framework_analyze_direct_call(self, graph, seen)
    }

    /// `analyze_indirect_call(graphs, seen)`
    /// (`graphanalyze.py`). Same super-call escape hatch as
    /// [`Self::analyze_direct_call`].
    fn analyze_indirect_call(
        &mut self,
        graphs: &[GraphRef],
        seen: Option<&mut DependencyTracker<R>>,
    ) -> R {
        framework_analyze_indirect_call(self, graphs, seen)
    }

    /// `analyze_all(graphs=None)` (`graphanalyze.py`).
    /// Walks every graph and feeds each op into [`Self::analyze`].
    /// Upstream takes the per-graph result and discards it; the same
    /// shape applies here.
    fn analyze_all(&mut self, graphs: Option<&[GraphRef]>) {
        let owned;
        let graphs: &[GraphRef] = if let Some(gs) = graphs {
            gs
        } else {
            owned = self.translator().graphs.borrow().clone();
            // Re-borrow as slice — iterate the snapshot so the
            // RefCell lock is dropped before per-op handlers run.
            unsafe { std::slice::from_raw_parts(owned.as_ptr(), owned.len()) }
        };
        for graph in graphs {
            let blocks = graph.borrow().iterblocks();
            for block in blocks {
                let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
                for op in &ops {
                    let _ = self.analyze(op, None, &I::default());
                }
            }
        }
    }
}

/// Framework body of `analyze_direct_call`
/// (`graphanalyze.py`), exposed as a free function so
/// subclass overrides can call it like upstream's
/// `BoolGraphAnalyzer.analyze_direct_call(self, ...)` super-call
/// (`gilanalysis.py`). Rust traits cannot have a `super` keyword;
/// the free function plays the same role.
pub fn framework_analyze_direct_call<A, R, I>(
    analyzer: &mut A,
    graph: &GraphRef,
    seen: Option<&mut DependencyTracker<R>>,
) -> R
where
    A: GraphAnalyzer<R, I>,
    R: AnalyzerResult,
    I: GraphInfo,
{
    // Upstream `:140-141`: `if seen is None: seen =
    // DependencyTracker(self)`.
    let mut owned_tracker = DependencyTracker::new();
    let seen = seen.unwrap_or(&mut owned_tracker);
    // Upstream `:142-143`: enter; if already seen, return cached.
    let key = GraphKey::of(graph).as_usize();
    if !seen.enter(key, analyzer.analyzed_calls()) {
        return seen.get_cached_result(key, analyzer.analyzed_calls());
    }

    let mut result = R::result_builder();
    let graphinfo = analyzer.compute_graph_info(graph);
    let (blocks, startblock, exceptblock) = {
        let g = graph.borrow();
        (g.iterblocks(), g.startblock.clone(), g.exceptblock.clone())
    };
    for block in blocks {
        // Upstream `:147 if block is graph.startblock` /
        // `:152 elif block is graph.exceptblock` — Python `is`
        // identity. `Rc::ptr_eq` matches the upstream identity check.
        if std::rc::Rc::ptr_eq(&block, &startblock) {
            result = R::add_to_result(result, analyzer.analyze_startblock(&block, Some(seen)));
        } else if std::rc::Rc::ptr_eq(&block, &exceptblock) {
            // Upstream `:155 self.analyze_exceptblock_in_graph(graph,
            // block, seen)`. The graph-aware variant is what concrete
            // analyzers (canraise.RaiseAnalyzer) override.
            result = R::add_to_result(
                result,
                analyzer.analyze_exceptblock_in_graph(graph, &block, Some(seen)),
            );
        }
        if !R::is_top_result(&result) {
            let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
            for op in &ops {
                result = R::add_to_result(result, analyzer.analyze(op, Some(seen), &graphinfo));
                if R::is_top_result(&result) {
                    break;
                }
            }
        }
        if !R::is_top_result(&result) {
            // Upstream `:166-170 for exit in block.exits:
            // self.analyze_link(exit, seen)`. Snapshot the exits so
            // the inner `borrow()` is dropped before analyzers reach
            // back into the block.
            let exits: Vec<LinkRef> = block.borrow().exits.to_vec();
            for exit in &exits {
                result = R::add_to_result(result, analyzer.analyze_link(exit, Some(seen)));
                if R::is_top_result(&result) {
                    break;
                }
            }
        }
        if R::is_top_result(&result) {
            break;
        }
    }
    result = R::finalize_builder(result);
    seen.leave_with(key, result.clone(), analyzer.analyzed_calls());
    result
}

/// Framework body of `analyze_indirect_call`
/// (`graphanalyze.py`).
pub fn framework_analyze_indirect_call<A, R, I>(
    analyzer: &mut A,
    graphs: &[GraphRef],
    seen: Option<&mut DependencyTracker<R>>,
) -> R
where
    A: GraphAnalyzer<R, I>,
    R: AnalyzerResult,
    I: GraphInfo,
{
    let mut result = R::result_builder();
    let mut owned_tracker = DependencyTracker::new();
    let seen = seen.unwrap_or(&mut owned_tracker);
    for graph in graphs {
        result = R::add_to_result(result, analyzer.analyze_direct_call(graph, Some(seen)));
        if R::is_top_result(&result) {
            break;
        }
    }
    R::finalize_builder(result)
}

/// Framework body of `analyze_external_call` (`graphanalyze.py`).
///
/// ```python
/// def analyze_external_call(self, funcobj, seen=None):
///     result = self.bottom_result()
///     if hasattr(funcobj, '_callbacks'):
///         bk = self.translator.annotator.bookkeeper
///         for function in funcobj._callbacks.callbacks:
///             desc = bk.getdesc(function)
///             for graph in desc.getgraphs():
///                 result = self.join_two_results(
///                     result, self.analyze_direct_call(graph, seen))
///     return result
/// ```
///
/// Upstream (`graphanalyze.py`) resolves the callback graphs through the
/// annotator bookkeeper: `for function in funcobj._callbacks.callbacks:
/// bk.getdesc(function).getgraphs()`. Pyre normalises `_callbacks` at the
/// `_func.attrs` mirror (the same mirror that carries `external` / `canraise`
/// / `random_effects_on_gcobjs`) into the already-resolved `ConstValue::Graphs`
/// graph-key list, resolved through `TranslationContext.graphs` exactly like
/// the `indirect_call` arm. A subclass override
/// (e.g. [`super::collectanalyze`]) delegates here for the no-special-case
/// path, matching upstream's `BoolGraphAnalyzer.analyze_external_call` super-call.
pub fn framework_analyze_external_call<A, R, I>(
    analyzer: &mut A,
    funcobj: &lltype::_func,
    seen: Option<&mut DependencyTracker<R>>,
) -> R
where
    A: GraphAnalyzer<R, I>,
    R: AnalyzerResult,
    I: GraphInfo,
{
    // `result = self.bottom_result()`.
    let result = R::bottom_result();
    // `if hasattr(funcobj, '_callbacks'):` — absent attr keeps the
    // `bottom_result()` (`hasattr == False` path).
    let Some(callbacks) = funcobj.attrs.get("_callbacks") else {
        return result;
    };
    // The attr is present, so pyre's normalisation invariant requires a
    // graph-key list; any other shape is a malformed `_callbacks` and
    // fails loud rather than silently degrading to bottom/top. Upstream's
    // direct `funcobj._callbacks.callbacks` access is likewise fail-loud
    // on a malformed attr.
    let ConstValue::Graphs(keys) = callbacks else {
        panic!(
            "_callbacks funcobj attr is not a ConstValue::Graphs key list ({callbacks:?}) \
             — pyre normalises callbacks to graph keys (graphanalyze.py:60-69)"
        );
    };
    // Resolve all callback graphs up front so the `translator()` borrow is
    // dropped before the recursive `analyze_direct_call` (which re-borrows
    // `self`). Upstream obtains every callback graph from
    // `bk.getdesc(function).getgraphs()`, so resolution never drops a
    // callback; an unresolved key here is a registration bug, surfaced as
    // `top_result()` exactly like the `indirect_call` unknown-graph path.
    let graphs: Vec<GraphRef> = {
        let trans_graphs = analyzer.translator().graphs.borrow();
        let mut graphs = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(graph) = trans_graphs
                .iter()
                .find(|g| GraphKey::of(g).as_usize() == *key)
                .cloned()
            else {
                return R::top_result();
            };
            graphs.push(graph);
        }
        graphs
    };
    // `seen` is threaded into each callee walk; default an owned tracker
    // when the caller passed `None`, as the framework does.
    let mut owned_tracker = DependencyTracker::new();
    let seen = seen.unwrap_or(&mut owned_tracker);
    // `for ... for graph in desc.getgraphs(): result =
    //  self.join_two_results(result, self.analyze_direct_call(...))`.
    let mut result = result;
    for graph in &graphs {
        result = R::join_two_results(result, analyzer.analyze_direct_call(graph, Some(seen)));
    }
    result
}

/// `DependencyTracker` (`graphanalyze.py`). Tracks the
/// active call-stack so cycles in the analysed call graph can be
/// detected and merged via the shared
/// [`crate::tool::algo::unionfind::UnionFind`] cache. The key is the
/// caller's graph identity (`GraphKey::as_usize()` for a flowspace
/// graph; a codewriter analyzer may key on `CallPath`). The result
/// lattice value lives inside [`Dependency`].
pub struct DependencyTracker<R: AnalyzerResult, K: Eq + std::hash::Hash + Clone = usize> {
    current_stack: Vec<K>,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: AnalyzerResult, K: Eq + std::hash::Hash + Clone> Default for DependencyTracker<R, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: AnalyzerResult, K: Eq + std::hash::Hash + Clone> DependencyTracker<R, K> {
    pub fn new() -> Self {
        Self {
            current_stack: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// `enter(graph)` (`graphanalyze.py`). Returns `true`
    /// when the graph is new (caller must analyse it); `false` when
    /// the graph is already on the stack (caller reads the cached
    /// result and merges the strongly-connected component).
    pub fn enter(&mut self, key: K, analyzed: &mut UnionFind<K, Dependency<R>>) -> bool {
        if !analyzed.contains(&key) {
            self.current_stack.push(key.clone());
            // Upstream `:236 self.graph_results.find(graph)` —
            // initialises the union-find slot through the factory
            // passed at `UnionFind::new` time.
            let _ = analyzed.find(key);
            true
        } else {
            // Upstream `:239-247`: cycle detection — find the
            // shared rep and union every stack frame above it.
            let graph_rep = analyzed.find_rep(key.clone());
            for j in 0..self.current_stack.len() {
                let other_rep = analyzed.find_rep(self.current_stack[j].clone());
                if graph_rep == other_rep {
                    for i in j..self.current_stack.len() {
                        analyzed.union(self.current_stack[i].clone(), key.clone());
                    }
                    break;
                }
            }
            false
        }
    }

    /// `leave_with(result)` (`graphanalyze.py`).
    pub fn leave_with(&mut self, _key: K, result: R, analyzed: &mut UnionFind<K, Dependency<R>>) {
        let popped = self.current_stack.pop().expect("leave_with: stack empty");
        if let Some(dep) = analyzed.get_mut(&popped) {
            dep.merge_with_result(result);
        }
    }

    /// `get_cached_result(graph)` (`graphanalyze.py`).
    pub fn get_cached_result(&self, key: K, analyzed: &mut UnionFind<K, Dependency<R>>) -> R {
        analyzed
            .get(&key)
            .map_or_else(R::bottom_result, |d| d.result())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Link, Variable,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Trivial subclass — every op is bottom (false).
    struct TrivialAnalyzer<'t> {
        translator: &'t TranslationContext,
        cache: UnionFind<usize, Dependency<bool>>,
    }

    impl<'t> TrivialAnalyzer<'t> {
        fn new(translator: &'t TranslationContext) -> Self {
            Self {
                translator,
                cache: UnionFind::new(|_| Dependency::new(false)),
            }
        }
    }

    impl<'t> GraphAnalyzer<bool, ()> for TrivialAnalyzer<'t> {
        fn analyze_simple_operation(&mut self, _op: &SpaceOperation, _graphinfo: &()) -> bool {
            false
        }

        fn translator(&self) -> &TranslationContext {
            self.translator
        }

        fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
            &mut self.cache
        }
    }

    fn empty_graph(name: &str) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new(name.to_string(), start.clone());
        let return_target = graph.returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn trivial_analyzer_returns_bottom_for_op_free_graph() {
        let translator = TranslationContext::new();
        let graph = empty_graph("entry");
        translator.graphs.borrow_mut().push(graph.clone());
        let mut analyzer = TrivialAnalyzer::new(&translator);
        assert!(!analyzer.analyze_direct_call(&graph, None));
    }

    #[test]
    fn indirect_call_missing_graph_key_returns_top() {
        let translator = TranslationContext::new();
        let mut analyzer = TrivialAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "indirect_call",
            vec![
                Hlvalue::Variable(Variable::named("fnptr")),
                Hlvalue::Constant(Constant::new(ConstValue::Graphs(vec![usize::MAX]))),
            ],
            Hlvalue::Variable(Variable::named("result")),
        );

        assert!(analyzer.analyze(&op, None, &()));
    }

    #[test]
    fn indirect_call_void_none_graphs_returns_top() {
        // `PtrRepr.rtype_simple_call` (`rptr.py`) appends
        // `inputconst(Void, None)` when the callee is not a Constant.
        let translator = TranslationContext::new();
        let mut analyzer = TrivialAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "indirect_call",
            vec![
                Hlvalue::Variable(Variable::named("fnptr")),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
            ],
            Hlvalue::Variable(Variable::named("result")),
        );

        assert!(analyzer.analyze(&op, None, &()));
    }
}
