//! Register allocation: port of PyPy's two-file split.
//!
//! Mirrors:
//!
//!   * `rpython/jit/codewriter/regalloc.py perform_register_allocation
//!     perform_register_allocation(graph, kind)` — thin 2-arg wrapper.
//!     Pyre's analog is `perform_register_allocation(graph, kind)`
//!     below.
//!   * `rpython/tool/algo/regalloc.py
//!     perform_register_allocation(graph, consider_var, ListOfKind)`:
//!     ```python
//!     regalloc = RegAllocator(graph, consider_var, ListOfKind)
//!     regalloc.make_dependencies()    # interference graph
//!     regalloc.coalesce_variables()   # union-find on jump edges
//!     regalloc.find_node_coloring()   # chordal coloring
//!     ```
//!     Pyre's analog is the `RegAllocator` struct below plus its three
//!     private methods of the same name.
//!   * `rpython/jit/codewriter/flatten.py enforce_input_args` —
//!     after coloring, `swapcolors` rotates inputarg colors into
//!     `0..n-1`. Pyre's analog is the `enforce_input_args` free
//!     function below (called by `GraphFlattener::enforce_input_args`
//!     at `flatten.rs` to mirror `flatten.py flattener.enforce_input_args()`,
//!     and directly by `codewriter.rs` before it builds the canonical
//!     `codewriter.py:53 flatten_graph(graph, regallocs, cpu)` stream).
//!   * `rpython/jit/codewriter/codewriter.py:62-67` —
//!     `num_regs[kind] = max(coloring)+1` per kind, packed into the
//!     `JitCode`. Pyre's analog is `RegAllocator::find_num_colors`
//!     plus the `AllocationResult.num_regs` field.
//!
//! The chordal coloring algorithm itself is shared with
//! `majit-translate`'s flow-graph regalloc through
//! `majit_jitcode::tool::algo::color::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py`).

use rustc_hash::{FxHashMap, FxHashSet};

/// `VariableId` is a `u32` newtype, and `make_dependencies` probes these tables
/// once per operand of every operation in every block.  Upstream keys the same
/// dictionaries on `Variable` objects hashed by identity, so its probe is a
/// pointer read; `std`'s default `RandomState` runs SipHash-1-3 over the four
/// bytes instead, which is the dominant cost of allocating registers for a
/// jitcode.  These ids never leave the codewriter, so nothing needs the
/// collision resistance.  No iteration here is order-sensitive: `die_at` is
/// sorted by die time before use, and `coloring` is only point-read and
/// value-rewritten.
type HashMap<K, V> = FxHashMap<K, V>;
type HashSet<T> = FxHashSet<T>;

/// The coloring a [`GraphAllocationResult`] carries, nameable by the
/// codewriter passes that build one directly or take one by reference.
pub type Coloring = FxHashMap<super::flow::VariableId, u16>;

use majit_jitcode::tool::algo::color::DependencyGraph;
use majit_jitcode::tool::algo::unionfind::UnionFind;

use super::flatten::{DescrOperand, Insn, Kind, Operand, Register, SSARepr, TLabel};
use super::flow::{ExitSwitch, ExitSwitchElement, FlowValue, FunctionGraph as FlowGraph, Variable};

#[derive(Debug, Clone)]
pub struct GraphAllocationResult {
    pub coloring: Coloring,
    pub num_colors: u16,
}

impl GraphAllocationResult {
    /// `rpython/tool/algo/regalloc.py RegAllocator.getcolor` —
    /// return the post-coloring color for a Variable.  Panics when
    /// `v` is not colored (matches PyPy `_coloring[...]` KeyError).
    /// Pyre's `enforce_input_args` short-circuits via direct
    /// `coloring.get` to skip the inputargs-never-referenced case (the
    /// pyre walker can produce that shape; PyPy's `make_dependencies`
    /// always adds inputargs as nodes, so the case can't arise there).
    pub fn getcolor(&self, v: super::flow::VariableId) -> u16 {
        *self.coloring.get(&v).unwrap_or_else(|| {
            panic!("GraphAllocationResult::getcolor: missing color for {v:?}");
        })
    }

    /// `rpython/tool/algo/regalloc.py RegAllocator.swapcolors`
    /// — swap every occurrence of `col1` and `col2` across the coloring
    /// dict.  Called by `enforce_input_args` (`flatten.py`) when
    /// an inputarg's coloring lands on a higher color than its
    /// positional `realcol`.
    pub fn swapcolors(&mut self, col1: u16, col2: u16) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }
}

/// Field names follow `rpython/tool/algo/regalloc.py __init__` —
/// `self.graph = graph` (py:22), `self.consider_var = consider_var`
/// (py:23 — pyre uses a `kind: Kind` filter because `Kind` is a closed
/// enum), `_depgraph` (`make_dependencies`, py:77), `_unionfind`
/// (`coalesce_variables`, py:80), `_coloring` (`find_node_coloring`,
/// py:115).  `self.ListOfKind` is omitted because pyre has exactly one
/// such type (`FlowListOfKind`).  The union-find is the real
/// `tool/algo/unionfind.py UnionFind` port (`()` info, matching
/// upstream `info_factory=None`).
struct RegAllocator<'a> {
    graph: &'a FlowGraph,
    kind: Kind,
    _depgraph: DependencyGraph<super::flow::VariableId>,
    _unionfind: UnionFind<super::flow::VariableId, ()>,
    _coloring: HashMap<super::flow::VariableId, u16>,
}

impl<'a> RegAllocator<'a> {
    fn new(graph: &'a FlowGraph, kind: Kind) -> Self {
        Self {
            graph,
            kind,
            _depgraph: DependencyGraph::new(),
            _unionfind: UnionFind::new(|_| ()),
            _coloring: HashMap::default(),
        }
    }

    fn make_dependencies(&mut self) {
        self.make_dependencies_among(None);
    }

    /// `_unionfind.find_rep(id)` when a pre-merge put `id` in the
    /// union-find, `id` itself otherwise.  `find_rep` would answer the same
    /// for an absent `id`, but it would also enter it as a new singleton,
    /// which nothing here reads back.
    fn merged_rep(&mut self, id: super::flow::VariableId) -> super::flow::VariableId {
        if self._unionfind.contains(&id) {
            self._unionfind.find_rep(id)
        } else {
            id
        }
    }

    /// `make_dependencies`, recording only the edges whose two endpoints are
    /// both in `tracked` (every edge when `None`).  An untracked variable
    /// gets no node either.
    fn make_dependencies_among(&mut self, tracked: Option<&HashSet<super::flow::VariableId>>) {
        let kind = self.kind;
        let is_tracked = |v: &super::flow::VariableId| tracked.is_none_or(|set| set.contains(v));
        // Per-block tables, emptied at the end of each block rather than
        // reallocated for the next one.
        let mut die_at: HashMap<super::flow::VariableId, usize> = HashMap::default();
        let mut die_list: Vec<(usize, super::flow::VariableId)> = Vec::new();
        let mut livevar_reps: Vec<super::flow::VariableId> = Vec::new();
        let mut livevars: HashSet<super::flow::VariableId> = HashSet::default();
        for block in self.graph.iterblocks() {
            let block_borrow = block.borrow();
            for arg in &block_borrow.inputargs {
                if let Some(v) = arg.as_variable() {
                    if v.kind == Some(kind) {
                        // ADAPTATION: project each Variable ID through
                        // `_unionfind.find_rep` so pre-merged pairs
                        // (from `perform_register_allocation_with_pairs`'s
                        // `extra_coalesce_pairs`) share a single
                        // live-set identity.  Walker scratch variables
                        // pinned to a local-i inputarg slot otherwise
                        // get distinct entries here and an interference
                        // edge gets recorded between them — preventing
                        // the later `try_coalesce` from merging them
                        // (regalloc.py:106 `has_edge` early return).
                        // When `_unionfind` has no pre-merges, find_rep
                        // returns the input ID unchanged so this matches
                        // upstream `regalloc.py:26-77` exactly.
                        let rep = self.merged_rep(v.id);
                        die_at.insert(rep, 0);
                    }
                }
            }
            for (i, op) in block_borrow.operations.iter().enumerate() {
                for arg in &op.args {
                    for v in arg.variables() {
                        if v.kind == Some(kind) {
                            let rep = self.merged_rep(v.id);
                            die_at.insert(rep, i);
                        }
                    }
                }
                if let Some(v) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if v.kind == Some(kind) {
                        let rep = self.merged_rep(v.id);
                        die_at.insert(rep, i + 1);
                    }
                }
            }
            match &block_borrow.exitswitch {
                Some(ExitSwitch::Value(value)) => {
                    if let Some(v) = value.as_variable() {
                        let rep = self.merged_rep(v.id);
                        die_at.remove(&rep);
                    }
                }
                Some(ExitSwitch::Tuple(values)) => {
                    for value in values {
                        if let ExitSwitchElement::Value(value) = value {
                            if let Some(v) = value.as_variable() {
                                let rep = self.merged_rep(v.id);
                                die_at.remove(&rep);
                            }
                        }
                    }
                }
                None => {}
            }
            for link in &block_borrow.exits {
                for arg in &link.borrow().args {
                    if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                        let rep = self.merged_rep(v.id);
                        die_at.remove(&rep);
                    }
                }
            }
            die_list.clear();
            die_list.extend(die_at.drain().map(|(var, time)| (time, var)));
            die_list.sort_by_key(|(time, _)| *time);
            die_list.push((usize::MAX, super::flow::VariableId(u32::MAX)));

            livevar_reps.clear();
            livevar_reps.extend(
                block_borrow
                    .inputargs
                    .iter()
                    .filter_map(FlowValue::as_variable)
                    .filter(|v| v.kind == Some(kind))
                    .map(|v| self.merged_rep(v.id)),
            );
            for (i, &v) in livevar_reps.iter().enumerate() {
                if !is_tracked(&v) {
                    continue;
                }
                self._depgraph.add_node(v);
                for j in 0..i {
                    // Pre-merged inputargs can collapse to the same
                    // representative.  RPython's DependencyGraph asserts
                    // against self-edges, so skip the edge here instead
                    // of weakening the shared color.py port.
                    if livevar_reps[j] != v && is_tracked(&livevar_reps[j]) {
                        self._depgraph.add_edge(livevar_reps[j], v);
                    }
                }
            }
            // upstream: `livevars = set(livevars)`.  An untracked variable
            // takes part in no recorded edge, so it is left out.
            livevars.clear();
            livevars.extend(livevar_reps.iter().copied().filter(|v| is_tracked(v)));
            let mut die_index = 0;
            for (i, op) in block_borrow.operations.iter().enumerate() {
                while die_list[die_index].0 == i {
                    livevars.remove(&die_list[die_index].1);
                    die_index += 1;
                }
                if let Some(result) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if result.kind == Some(kind) {
                        let rep = self.merged_rep(result.id);
                        // upstream (`regalloc.py:73`): add an edge from
                        // every live var to `result`.  `result` is added
                        // to `livevars` only *after* the loop upstream.
                        // Pyre's pin pre-merge can make an already-live
                        // inputarg and this result share a representative,
                        // so keep the RPython add_edge invariant locally.
                        if is_tracked(&rep) {
                            self._depgraph.add_node(rep);
                            for &v in &livevars {
                                if v != rep {
                                    self._depgraph.add_edge(v, rep);
                                }
                            }
                            livevars.insert(rep);
                        }
                    }
                }
            }
        }
    }

    fn coalesce_variables(&mut self) {
        let kind = self.kind;
        let mut pendingblocks = self.graph.iterblocks();
        while let Some(block) = pendingblocks.pop() {
            // Match `rpython/tool/algo/regalloc.py:82-86`: walk from the
            // end of the graph first because resume/blackhole execution
            // typically restarts in the middle rather than at the entry.
            let block_borrow = block.borrow();
            for link in &block_borrow.exits {
                let link_borrow = link.borrow();
                if let Some(v) = link_borrow.last_exception {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(v.id);
                    }
                }
                if let Some(v) = link_borrow.last_exc_value {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(v.id);
                    }
                }
                let Some(target) = link_borrow.target.clone() else {
                    continue;
                };
                let target_borrow = target.borrow();
                for (arg, target_input) in
                    link_borrow.args.iter().zip(target_borrow.inputargs.iter())
                {
                    let Some(src) = arg.as_ref().and_then(FlowValue::as_variable) else {
                        continue;
                    };
                    let Some(dst) = target_input.as_variable() else {
                        continue;
                    };
                    self.try_coalesce(src, dst);
                }
            }
        }
    }

    /// `rpython/tool/algo/regalloc.py _try_coalesce` — kind
    /// check + identity short-circuit + interference check + union.
    /// Both endpoints are normally in `_depgraph` already because
    /// `make_dependencies` registered every op result, inputarg, and
    /// link-arg-derived Variable.  Pyre's external pin pre-merge can make
    /// `find_rep()` return a different surviving id, so seed the reps before
    /// calling `DependencyGraph.coalesce`.
    fn try_coalesce(&mut self, v: Variable, w: Variable) {
        if v.kind != Some(self.kind) || w.kind != Some(self.kind) {
            return;
        }
        if v.id == w.id {
            return;
        }
        let v0 = self._unionfind.find_rep(v.id);
        let w0 = self._unionfind.find_rep(w.id);
        self._depgraph.add_node(v0);
        self._depgraph.add_node(w0);
        if v0 == w0 {
            return;
        }
        if self._depgraph.has_edge(&v0, &w0) {
            return;
        }
        let (_, rep) = self._unionfind.union(v0, w0);
        debug_assert_eq!(self._unionfind.find_rep(v0), rep);
        debug_assert_eq!(self._unionfind.find_rep(w0), rep);
        if rep == v0 {
            self._depgraph.coalesce(w0, v0);
        } else {
            debug_assert_eq!(rep, w0);
            self._depgraph.coalesce(v0, w0);
        }
    }

    /// Variable-id-keyed pin coalesce for `walker_pin_pairs` —
    /// pyre-only ADAPTATION called from
    /// `perform_register_allocation_with_pairs` to honour the walker's
    /// scratch↔inputarg slot pinning.  Walker scratch variables that
    /// don't appear as op operands/results in the canonical graph
    /// aren't registered via `make_dependencies`, so they're absent
    /// from `_depgraph.all_nodes`.  `DependencyGraph::coalesce` only
    /// modifies `neighbours` (not `all_nodes`), so
    /// `find_node_coloring`'s `getnodes` filter would skip a coalesced
    /// surviving node that was never explicitly added — yielding
    /// `None` for `getcolor` and dropping the chain's inputarg from
    /// the final coloring map.  This defensive `add_node` can be
    /// dropped once coalesced survivors are always registered via
    /// `make_dependencies`.
    fn try_coalesce_pin_ids(
        &mut self,
        v_id: super::flow::VariableId,
        w_id: super::flow::VariableId,
    ) {
        if v_id == w_id {
            return;
        }
        let v0 = self._unionfind.find_rep(v_id);
        let w0 = self._unionfind.find_rep(w_id);
        self._depgraph.add_node(v0);
        self._depgraph.add_node(w0);
        if v0 == w0 {
            return;
        }
        if self._depgraph.has_edge(&v0, &w0) {
            return;
        }
        let (_, rep) = self._unionfind.union(v0, w0);
        debug_assert_eq!(self._unionfind.find_rep(v0), rep);
        debug_assert_eq!(self._unionfind.find_rep(w0), rep);
        if rep == v0 {
            self._depgraph.coalesce(w0, v0);
        } else {
            debug_assert_eq!(rep, w0);
            self._depgraph.coalesce(v0, w0);
        }
    }

    fn find_node_coloring(&mut self) {
        self._coloring = self
            ._depgraph
            .find_node_coloring()
            .into_iter()
            .map(|(value, color)| (value, color as u16))
            .collect();
    }

    fn getcolor(&mut self, v: Variable) -> Option<u16> {
        let rep = self.merged_rep(v.id);
        self._coloring.get(&rep).copied()
    }

    fn find_num_colors(&self) -> u16 {
        self._coloring.values().copied().max().map_or(0, |m| m + 1)
    }
}

/// `rpython/jit/codewriter/regalloc.py perform_register_allocation(graph, kind)`
/// — thin wrapper over `rpython/tool/algo/regalloc.py
/// perform_register_allocation(graph, consider_var, ListOfKind=())`.
///
/// Pyre bakes `consider_var` into the single `kind` filter because
/// `Kind` is a closed enum (Int/Ref/Float) whereas upstream's
/// `consider_var` is an open predicate over lltype concreteness.
/// `ListOfKind` is not a parameter because pyre has exactly one such
/// class (`FlowListOfKind`).
///
/// Invoked from production via `perform_register_allocation_all_kinds`
/// at `codewriter.rs:transform_graph_to_jitcode`, where its result
/// feeds the canonical `flatten_graph` splice regalloc, which colors
/// the production stream.  `coalesce_variables` runs here over
/// `link.args ↔ target.inputargs` pairs.
pub(super) fn perform_register_allocation(graph: &FlowGraph, kind: Kind) -> GraphAllocationResult {
    perform_register_allocation_with_pairs(graph, kind, &[])
}

/// ADAPTATION variant: runs the same `RegAllocator` pipeline as
/// [`perform_register_allocation`] but applies `extra_coalesce_pairs`
/// between the upstream-orthodox `coalesce_variables` and
/// `find_node_coloring` steps.
///
/// Each pair `(scratch_id, inputarg_id)` requests that
/// `scratch_id`'s post-coloring color equal `inputarg_id`'s color.
/// The mechanism is `try_coalesce`: the two variables are unioned in
/// the regalloc union-find (if no interference edge blocks it), so
/// the subsequent chordal coloring assigns them the same color.
/// `enforce_input_args` then rotates the unified cluster onto the
/// inputarg's `0..nlocals-1` slot.
///
/// Upstream RPython has no analog because PyPy's flowgraph never
/// produces "scratch local-i" Variables disjoint from the
/// `startblock.inputargs[i]` variable — the same Variable flows
/// through every read/write of local i.  Pyre's walker
/// (`codewriter.rs::transform_graph_to_jitcode`) emits fresh
/// scratch Variables for each `LOAD_FAST` / `STORE_FAST` and pins
/// them to slot=i via `walker_slot_for_variable`; this helper lets
/// the canonical graph regalloc honor that same pin so the bytes it
/// emits match the walker's inline emission slot-for-slot.
pub fn perform_register_allocation_with_pairs(
    graph: &FlowGraph,
    kind: Kind,
    extra_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> GraphAllocationResult {
    // `rpython/tool/algo/regalloc.py:11-15`:
    //     regalloc = RegAllocator(graph, consider_var, ListOfKind)
    //     regalloc.make_dependencies()
    //     regalloc.coalesce_variables()
    //     regalloc.find_node_coloring()
    let mut allocator = RegAllocator::new(graph, kind);
    // ADAPTATION: pre-merge external pairs into `_unionfind` BEFORE
    // `make_dependencies` so the live-set tracking (which projects
    // every Variable ID through `_unionfind.find_rep`) treats each
    // pinned scratch↔inputarg pair as a single node.  Without the
    // pre-merge, walker scratch and the corresponding canonical
    // inputarg get separate live entries and `make_dependencies`
    // records an interference edge between them; the post-coalesce
    // `try_coalesce_ids` then early-returns at the `has_edge` check
    // (regalloc.py:106) and the pin has no effect on coloring.
    // `find_rep` auto-creates a singleton partition for IDs not yet
    // tracked, so unknown scratch IDs are handled safely.
    for &(v_id, w_id) in extra_coalesce_pairs {
        let v0 = allocator._unionfind.find_rep(v_id);
        let w0 = allocator._unionfind.find_rep(w_id);
        if v0 != w0 {
            allocator._unionfind.union(v0, w0);
        }
    }
    allocator.make_dependencies();
    allocator.coalesce_variables();
    // External pins — re-apply via `try_coalesce_pin_ids` after
    // `make_dependencies` so the surviving rep is explicitly added to
    // `_depgraph.all_nodes` even when neither endpoint appeared as an
    // op result/arg in the canonical graph.  With the union-find
    // pre-merge above these calls are no-ops on the union-find side
    // (`find_rep` already returns a common rep), but `add_node` still
    // matters for `find_node_coloring`'s `getnodes` filter.
    for &(v_id, w_id) in extra_coalesce_pairs {
        allocator.try_coalesce_pin_ids(v_id, w_id);
    }
    allocator.find_node_coloring();

    let mut coloring = HashMap::default();
    // A variable met again already has its color, so only its first
    // occurrence asks the allocator.
    let mut record = |v: Variable| {
        if v.kind != Some(kind) {
            return;
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = coloring.entry(v.id) {
            if let Some(color) = allocator.getcolor(v) {
                entry.insert(color);
            }
        }
    };
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        // `Block.getvariables()` walked in place.
        let block_variables = block_borrow
            .inputargs
            .iter()
            .filter_map(FlowValue::as_variable)
            .chain(block_borrow.operations.iter().flat_map(|op| {
                op.args
                    .iter()
                    .flat_map(|arg| arg.variables())
                    .chain(op.result.as_ref().and_then(FlowValue::as_variable))
            }));
        for variable in block_variables {
            record(variable);
        }
        for link in &block_borrow.exits {
            let link_borrow = link.borrow();
            let link_variables = [link_borrow.last_exception, link_borrow.last_exc_value]
                .into_iter()
                .flatten()
                .chain(
                    link_borrow
                        .args
                        .iter()
                        .filter_map(|arg| arg.as_ref().and_then(FlowValue::as_variable)),
                );
            for v in link_variables {
                record(v);
            }
        }
    }

    GraphAllocationResult {
        coloring,
        num_colors: allocator.find_num_colors(),
    }
}

/// Apply PyPy's `_try_coalesce` interference check
/// (`rpython/tool/algo/regalloc.py`, the `v0 not in
/// dg.neighbours[w0]` guard at py:105) to a candidate list of coalesce
/// pairs and return only the pairs PyPy would accept.
///
/// `perform_register_allocation_with_pairs` pre-merges its
/// `extra_coalesce_pairs` into the union-find BEFORE `make_dependencies`,
/// which bypasses the interference check — silently coalescing pairs PyPy
/// would reject when the two endpoints are simultaneously live (e.g. a
/// short-circuit `(i and C)` PHI result that coalesces with the loop var
/// `i` on the `and`-false edge, yet both are live across the following
/// `or` guard: `i` for the loop's `i = i + 1`, the PHI result as the
/// guard's kept operand-stack temp).  That merge collapses the kept slot's
/// color onto `i`, so a const-folded kept value (`(i and 2.5) == 2.5`) is
/// never recoverable at the guard snapshot (#124 float tail).
///
/// PyPy runs the check on the PRE-renaming graph (the CFG coalesce sweep
/// precedes `flatten.py insert_renamings`); this filter does the same
/// by building the dependency graph on `graph` (the same pre-renaming graph
/// `collect_cfg_coalesce_pairs` reads) with an EMPTY union-find, so
/// `make_dependencies` records true interference.  It then replays the
/// `_try_coalesce` chain incrementally — projecting each endpoint through
/// the cumulative union-find and coalescing the dependency graph on each
/// accepted pair — so transitive interference is honoured exactly as
/// upstream `coalesce_variables` does.  Pairs whose endpoints already share
/// a rep are kept (the pre-merge is a no-op for them); pairs whose reps
/// interfere are dropped.
///
/// This is the parity-correct replacement for the unconditional pre-merge:
/// non-interfering pins (walker scratch ↔ inputarg) survive, so the
/// canonical coloring still matches the walker's emit, while the
/// interfering `(i, PHI)` pair is rejected.
///
/// The per-PC `-live-` graph ops carry every frame-live Ref Variable as a
/// force-alive arg (`liveness.py:8-12`), so `make_dependencies` here
/// already models CPython frame-slot liveness: a pair whose endpoints'
/// frame lifetimes overlap interferes structurally and is rejected by the
/// same `has_edge` guard — no external interference seeding.
pub fn filter_coalesce_pairs_by_interference(
    graph: &FlowGraph,
    kind: Kind,
    pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> Vec<(super::flow::VariableId, super::flow::VariableId)> {
    let mut allocator = RegAllocator::new(graph, kind);
    // Every `has_edge` below asks about two representatives of merged
    // endpoint classes, and a merged node's neighbours are the union of its
    // members' (`DependencyGraph.coalesce`).  So only edges between two pair
    // endpoints can change an answer; the rest of the graph is not built.
    let endpoints: HashSet<super::flow::VariableId> =
        pairs.iter().flat_map(|&(v, w)| [v, w]).collect();
    allocator.make_dependencies_among(Some(&endpoints));
    let mut kept = Vec::with_capacity(pairs.len());
    for &(v_id, w_id) in pairs {
        if v_id == w_id {
            continue;
        }
        let v0 = allocator._unionfind.find_rep(v_id);
        let w0 = allocator._unionfind.find_rep(w_id);
        if v0 == w0 {
            // Already merged by an earlier accepted pair — pre-merging it
            // again is a no-op, so keep it (matches `_try_coalesce`'s
            // `v0 is w0` early return, which leaves the pair coalesced).
            kept.push((v_id, w_id));
            continue;
        }
        // `DependencyGraph.coalesce` requires both endpoints to be present
        // in the graph, matching `try_coalesce_pin_ids`.  Some external pin
        // pairs name slot representatives that have not appeared in ordinary
        // SSA liveness yet, so seed them before replaying `_try_coalesce`.
        allocator._depgraph.add_node(v0);
        allocator._depgraph.add_node(w0);
        if allocator._depgraph.has_edge(&v0, &w0) {
            // `regalloc.py:105` rejects an interfering pair.
            continue;
        }
        let (_, rep) = allocator._unionfind.union(v0, w0);
        if rep == v0 {
            allocator._depgraph.coalesce(w0, v0);
        } else {
            allocator._depgraph.coalesce(v0, w0);
        }
        kept.push((v_id, w_id));
    }
    kept
}

/// Run `perform_register_allocation` once per `Kind` and collect
/// the per-kind `GraphAllocationResult`s, mirroring
/// `rpython/jit/codewriter/codewriter.py:44-46`:
///
/// ```python
/// regallocs = {}
/// for kind in KINDS:
///     regallocs[kind] = perform_register_allocation(graph, kind)
/// ```
///
/// The resulting `[GraphAllocationResult; 3]` is indexed by
/// `Kind::index()` (`Int=0, Ref=1, Float=2`).  Upstream uses a Python
/// dict; pyre uses `[T; 3]`: the
/// RPython `KINDS` list has 3 statically-known entries so the dict
/// degenerates to a position-indexed array in any RPython-orthodox
/// port.  This is the input shape that the canonical
/// `flatten_graph(graph, regallocs, ...)` driver consumes.
pub fn perform_register_allocation_all_kinds(graph: &FlowGraph) -> [GraphAllocationResult; 3] {
    perform_register_allocation_all_kinds_with_pairs(graph, &[])
}

/// ADAPTATION variant: invokes the per-kind
/// `perform_register_allocation_with_pairs` for `Kind::Ref` with
/// `ref_coalesce_pairs`.  Int and Float kinds use the empty-pair
/// path because walker's `walker_slot_for_variable` only tracks Ref
/// slots (every `FrameState.mergeable()` position is Ref-kind:
/// locals, stack, last_exc pair).
pub fn perform_register_allocation_all_kinds_with_pairs(
    graph: &FlowGraph,
    ref_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> [GraphAllocationResult; 3] {
    [
        perform_register_allocation(graph, Kind::Int),
        perform_register_allocation_with_pairs(graph, Kind::Ref, ref_coalesce_pairs),
        perform_register_allocation(graph, Kind::Float),
    ]
}

/// Mirrors `rpython/jit/codewriter/flatten.py enforce_input_args`
///

/// Walks the startblock's inputargs in source order; for each inputarg
/// of kind `K` whose current color in `regallocs[K]` does not equal
/// the next "real" color for that kind (`0, 1, 2, ...` per appearance),
/// invokes `swapcolors(realcol, curcol)` over the entire
/// `coloring` map (`rpython/tool/algo/regalloc.py`).
///
/// Upstream `flatten_graph` runs this immediately after
/// `regallocs[kind] = perform_register_allocation(graph, kind)` and
/// before `generate_ssa_form` walks links, so every downstream
/// observer sees the post-swap coloring.  Pyre's canonical
/// `flatten_graph` entry (`flatten.rs::flatten_graph`) and the
/// walker-side post-walk path both call this free function rather
/// than a `GraphFlattener` method: pyre's `get_register` closure
/// captures `&regallocs` immutably, so the `&mut regallocs`
/// swap must run BEFORE the closure is constructed.
pub fn enforce_input_args(graph: &FlowGraph, regallocs: &mut [GraphAllocationResult; 3]) {
    let inputargs = graph.startblock.borrow().inputargs.clone();
    // RPython `numkinds = {}` (flatten.py); pyre stores the per-kind
    // counter in a `[u16; 3]` array indexed by `Kind::index()`.
    let mut numkinds: [u16; 3] = [0; 3];
    for arg in &inputargs {
        let Some(v) = arg.as_variable() else { continue };
        let Some(kind) = v.kind else { continue };
        let kind_idx = kind.index();
        let realcol = numkinds[kind_idx];
        numkinds[kind_idx] = realcol + 1;
        let alloc = &mut regallocs[kind_idx];
        // Inputarg never appeared in any instruction — coloring
        // skipped it. Swap is unnecessary because no register refers
        // to its color.
        let Some(&curcol) = alloc.coloring.get(&v.id) else {
            continue;
        };
        if curcol == realcol {
            continue;
        }
        assert!(
            curcol > realcol,
            "enforce_input_args: inputarg color {} must be >= realcol {} \
             (regalloc.py invariant)",
            curcol,
            realcol,
        );
        // `flatten.py self.regallocs[kind].swapcolors(realcol, curcol)`.
        alloc.swapcolors(realcol, curcol);
    }
}

/// Per-kind coloring facts the assembler needs, packed for the
/// `codewriter.rs` production path.
///
/// `rename` carries a per-kind pre→post coloring map: `[Vec<u16>; 3]`
/// indexed by `Kind::index()`, each inner `Vec<u16>` indexed by the
/// pre-coloring slot and yielding the post-coloring color.  Entries
/// past the vector's length implicitly map to identity.  Mirrors
/// RPython's `(kind, pre) → post` dict at `codewriter.py:62-67`
/// projected onto pyre's u16 slot space.  The production path builds
/// this map empty: the canonical `flatten_graph` splice coloring is
/// the sole authority over the stream's register indices, so every
/// lookup resolves to identity.
///
/// `num_regs` carries the per-kind `max(color)+1` value the assembler
/// stores in `JitCode.num_regs_*` (codewriter.py:62-67).
pub(super) struct AllocationResult {
    pub rename: [Vec<u16>; 3],
    /// Per-kind `max(coloring)+1` indexed by `Kind::index()`.
    /// Mirrors RPython
    /// `codewriter.py:62-67 num_regs[kind]` — pyre's `KINDS` array
    /// of 3 statically-known kinds collapses the dict to `[u16; 3]`.
    pub num_regs: [u16; 3],
}

/// Lookup helper for the kind-indexed rename vec: returns the post
/// coloring for `pre`, falling back to identity when no rename was
/// recorded.
#[inline]
pub(super) fn rename_lookup(rename: &[Vec<u16>; 3], kind: Kind, pre: u16) -> u16 {
    rename[kind.index()]
        .get(pre as usize)
        .copied()
        .filter(|&p| p != u16::MAX)
        .unwrap_or(pre)
}

#[cfg(test)]
mod tests {
    use super::super::flatten::Kind;
    use super::super::flow::{
        Block, Constant, FlowListOfKind, FunctionGraph, Link, SpaceOperation, Variable, VariableId,
        push_op,
    };
    use super::*;

    fn flow_var(id: u32, kind: Kind) -> Variable {
        Variable::new(VariableId(id), kind)
    }

    #[test]
    fn all_kinds_driver_produces_regalloc_for_every_kind() {
        // Build a graph with one variable per kind as startblock inputs,
        // and a returnblock that takes a single Int so the link has
        // matching arity.
        let v0 = flow_var(0, Kind::Int);
        let vr = flow_var(1, Kind::Ref);
        let vf = flow_var(2, Kind::Float);
        let start = Block::shared(vec![v0.into(), vr.into(), vf.into()]);
        let graph = FunctionGraph::new("all_kinds", start.clone(), Some(v0));
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let regallocs = perform_register_allocation_all_kinds(&graph);
        for &kind in &Kind::ALL {
            let result = &regallocs[kind.index()];
            // Each kind has at least one variable (Int: v0 twice via
            // return link; Ref: vr in startblock inputargs; Float: vf).
            // Colorings never exceed num_colors.
            assert!(
                result.num_colors >= 1,
                "kind {kind:?} expected at least one color, got {}",
                result.num_colors
            );
            for (_id, color) in &result.coloring {
                assert!(
                    *color < result.num_colors,
                    "kind {kind:?} color {color} exceeds num_colors {}",
                    result.num_colors
                );
            }
        }
    }

    #[test]
    fn graph_regalloc_reuses_color_for_non_overlapping_values() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("graph_regalloc", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new("same_as", vec![v0.into()], Some(v1.into()), 0),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn perform_register_allocation_with_pairs_shares_color_for_pinned_scratch() {
        // Two INTERFERING Ref variables: v0 (inputarg) is kept live
        // past v1's definition by carrying both on the outgoing link,
        // so the unpinned chordal coloring must assign them different
        // colors.  `extra_coalesce_pairs` pre-merges them in the
        // union-find before `make_dependencies` so they collapse into
        // a single node and share a color — bypassing the interference
        // edge that `_try_coalesce` (regalloc.py) would otherwise
        // honour.
        let build_graph = || {
            let v0 = flow_var(0, Kind::Ref);
            let v1 = flow_var(1, Kind::Ref);
            let start = Block::shared(vec![v0.into()]);
            let mut graph = FunctionGraph::new("pin_share_color", start.clone(), None);
            push_op(
                &start,
                SpaceOperation::new("ref_copy", vec![v0.into()], Some(v1.into()), 0),
            );
            let v2 = flow_var(2, Kind::Ref);
            let v3 = flow_var(3, Kind::Ref);
            let next = graph.new_block(vec![v2.into(), v3.into()]);
            // Both v0 and v1 carried forward so the live-set at the
            // outgoing link contains both, forcing an interference
            // edge under the unpinned allocator.
            start.closeblock(vec![
                Link::new(vec![v0.into(), v1.into()], Some(next.clone()), None).into_ref(),
            ]);
            // returnblock arity is always 1 (a fresh untyped variable
            // when `return_var = None` was passed to FunctionGraph::new).
            next.closeblock(vec![
                Link::new(vec![v2.into()], Some(graph.returnblock.clone()), None).into_ref(),
            ]);
            (graph, v0, v1)
        };

        // Baseline: without pins, the interference forces distinct colors.
        let (graph_unpinned, v0_u, v1_u) = build_graph();
        let unpinned = perform_register_allocation_with_pairs(&graph_unpinned, Kind::Ref, &[]);
        let unpinned_v0 = unpinned.coloring.get(&v0_u.id).copied();
        let unpinned_v1 = unpinned.coloring.get(&v1_u.id).copied();
        assert!(unpinned_v0.is_some() && unpinned_v1.is_some());
        assert_ne!(
            unpinned_v0, unpinned_v1,
            "without pins, interfering v0 and v1 must get distinct colors"
        );

        // Pinned: pre-merge unifies them into one node before
        // make_dependencies so the interference edge never gets recorded.
        let (graph_pinned, v0_p, v1_p) = build_graph();
        let pin_pairs = vec![(v1_p.id, v0_p.id)];
        let pinned = perform_register_allocation_with_pairs(&graph_pinned, Kind::Ref, &pin_pairs);
        let pinned_v0 = pinned.coloring.get(&v0_p.id).copied();
        let pinned_v1 = pinned.coloring.get(&v1_p.id).copied();
        assert!(pinned_v0.is_some() && pinned_v1.is_some());
        assert_eq!(
            pinned_v0, pinned_v1,
            "pin must unify v0 and v1 even across an interference edge"
        );
    }

    #[test]
    fn filter_coalesce_pairs_by_interference_live_marker_rejects_ssa_disjoint_pair() {
        // v0 (inputarg) is copied into v1 and dies at the copy, so v0 and v1
        // have DISJOINT SSA live ranges: `make_dependencies` records no edge
        // between them and the coalesce pair (v0, v1) is accepted.  A `-live-`
        // graph op carrying both as force-alive args (`liveness.py:8-12`)
        // models a CPython-slot co-live separation the plain SSA graph cannot
        // see (two locals co-live at a guard across `LOAD_FAST` re-reads);
        // `make_dependencies` then records the edge and the `has_edge` guard
        // rejects the pair.
        let build = |with_live_marker: bool| {
            let v0 = flow_var(0, Kind::Ref);
            let v1 = flow_var(1, Kind::Ref);
            let start = Block::shared(vec![v0.into()]);
            let graph = FunctionGraph::new("xslot_colive", start.clone(), None);
            push_op(
                &start,
                SpaceOperation::new("ref_copy", vec![v0.into()], Some(v1.into()), 0),
            );
            if with_live_marker {
                push_op(
                    &start,
                    SpaceOperation::new(
                        super::super::flatten::OPNAME_LIVE,
                        vec![v0.into(), v1.into()],
                        None,
                        0,
                    ),
                );
            }
            start.closeblock(vec![
                Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
            ]);
            (graph, v0.id, v1.id)
        };

        // SSA-liveness only: the disjoint pair is accepted.
        let (g_ssa, a, b) = build(false);
        let kept_ssa = filter_coalesce_pairs_by_interference(&g_ssa, Kind::Ref, &[(a, b)]);
        assert_eq!(
            kept_ssa,
            vec![(a, b)],
            "SSA-disjoint pair must be accepted without a forcing -live- marker"
        );

        // With a force-alive `-live-` marker, the has_edge guard rejects it.
        let (g_colive, a, b) = build(true);
        let kept_colive = filter_coalesce_pairs_by_interference(&g_colive, Kind::Ref, &[(a, b)]);
        assert!(
            kept_colive.is_empty(),
            "a forcing -live- marker must reject the otherwise-accepted pair"
        );
    }

    #[test]
    fn perform_register_allocation_with_pairs_handles_unknown_scratch_id() {
        // Walker may produce scratch Variable IDs that never appear in
        // the canonical graph (walker-only emit sites).  Pin pairs
        // containing such IDs must not panic and must not strip the
        // inputarg's color entry.
        let v0 = flow_var(0, Kind::Ref);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("pin_unknown_scratch", start.clone(), None);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        // v_99 is not in the graph; pin (99, 0) should be benign.
        let pin_pairs = vec![(VariableId(99), v0.id)];
        let result = perform_register_allocation_with_pairs(&graph, Kind::Ref, &pin_pairs);
        assert!(
            result.coloring.get(&v0.id).is_some(),
            "inputarg v0 must retain a color even when pinned scratch ID 99 is absent from the graph"
        );
    }

    #[test]
    fn graph_regalloc_coalesces_goto_link_args_with_target_inputargs() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_goto", start.clone(), None);
        let next = graph.new_block(vec![v1.into()]);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_seeds_exception_extravars_as_colorable_nodes() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("graph_exc", start.clone(), None);
        let target = graph.new_block(Vec::new());
        let exc_type = flow_var(10, Kind::Int);
        let mut link = Link::new(Vec::new(), Some(target), None);
        link.extravars(Some(exc_type), None);
        start.closeblock(vec![link.into_ref()]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&exc_type.id), Some(&0));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_marks_listofkind_args_as_uses() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("graph_listofkind", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new(
                "same_as",
                vec![Constant::signed(1).into()],
                Some(v1.into()),
                0,
            ),
        );
        push_op(
            &start,
            SpaceOperation::new(
                "consume",
                vec![FlowListOfKind::new(Kind::Int, vec![v0.into()]).into()],
                None,
                0,
            ),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_ne!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 2);
    }

    /// `flatten.py enforce_input_args` parity at the graph
    /// allocator level: after the swap, every kind's startblock
    /// inputargs occupy colors `0, 1, 2, …` in source order.
    #[test]
    fn enforce_input_args_graph_side_normalises_inputarg_colors() {
        // 2 Ref inputargs + 1 Int inputarg, all live across an op
        // that defines fresh Variables of each kind so the chordal
        // coloring has to place every node on its own color.
        let a = flow_var(0, Kind::Ref);
        let b = flow_var(1, Kind::Ref);
        let i = flow_var(2, Kind::Int);
        let r0 = flow_var(10, Kind::Ref);
        let i0 = flow_var(11, Kind::Int);
        let start = Block::shared(vec![a.into(), b.into(), i.into()]);
        let mut graph = FunctionGraph::new("enforce_sim", start.clone(), None);
        push_op(
            &start,
            SpaceOperation::new("consume_ref", vec![a.into(), b.into()], Some(r0.into()), 0),
        );
        push_op(
            &start,
            SpaceOperation::new("consume_int", vec![i.into()], Some(i0.into()), 0),
        );
        let next = graph.new_block(vec![r0.into(), i0.into()]);
        start.closeblock(vec![
            Link::new(vec![r0.into(), i0.into()], Some(next), None).into_ref(),
        ]);

        let mut regallocs = perform_register_allocation_all_kinds(&graph);
        enforce_input_args(&graph, &mut regallocs);

        let ref_colors = &regallocs[Kind::Ref.index()].coloring;
        let int_colors = &regallocs[Kind::Int.index()].coloring;
        assert_eq!(
            ref_colors.get(&a.id).copied(),
            Some(0),
            "first Ref inputarg must occupy color 0 post-enforce_input_args"
        );
        assert_eq!(
            ref_colors.get(&b.id).copied(),
            Some(1),
            "second Ref inputarg must occupy color 1 post-enforce_input_args"
        );
        assert_eq!(
            int_colors.get(&i.id).copied(),
            Some(0),
            "first Int inputarg must occupy color 0 post-enforce_input_args"
        );
    }
}
