//! Register allocation on the control flow graph.
//!
//! RPython equivalent: `rpython/tool/algo/regalloc.py`.
//!
//! Operates on `FunctionGraph` (Block structure), NOT on flattened ops.
//! RPython runs regalloc BEFORE flatten: codewriter.py:45-47.
//!
//! 1. Build interference graph per-block (die_at analysis)
//! 2. Coalesce variables connected by Goto link args
//! 3. Greedy graph coloring via lexicographic BFS

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

/// The per-variable tables — coloring, interference, liveness — key on
/// flow-graph `Variable`s, which upstream's dictionaries hash by object
/// identity, so a probe there is a pointer read.  `std`'s default
/// `RandomState` runs SipHash-1-3 per probe instead, and the assembler reads
/// the coloring once per variable occurrence in the flattened graph.  Ordering
/// is not observable: every read is a point lookup, a `len`, or a value-only
/// rewrite.  The per-kind map keeps `RandomState`; it holds one entry per
/// `RegKind` and is not probed in a loop.
pub type VarMap<V> = FxHashMap<crate::flowspace::model::Variable, V>;

/// The coloring an [`AllocationResult`] carries, nameable by callers that
/// build one directly.
pub type Coloring = VarMap<usize>;
type VarSet = FxHashSet<crate::flowspace::model::Variable>;

use crate::flatten::RegKind;
use crate::model::{Block, ConcreteType, FunctionGraph, OpKind};
pub use crate::tool::algo::color::DependencyGraph;

// ── UnionFind (RPython tool/algo/unionfind.py) ────────────────────

#[derive(Debug, Clone)]
struct UnionFind<N: Eq + std::hash::Hash + Clone> {
    parent: FxHashMap<N, N>,
    weight: FxHashMap<N, usize>,
}

impl<N: Eq + std::hash::Hash + Clone> UnionFind<N> {
    fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            weight: FxHashMap::default(),
        }
    }

    fn find_rep(&mut self, v: N) -> N {
        if !self.parent.contains_key(&v) {
            self.parent.insert(v.clone(), v.clone());
            self.weight.insert(v.clone(), 1);
            return v;
        }
        let mut root = v.clone();
        while self.parent[&root] != root {
            root = self.parent[&root].clone();
        }
        let mut current = v;
        while current != root {
            let next = self.parent[&current].clone();
            self.parent.insert(current, root.clone());
            current = next;
        }
        root
    }

    fn union(&mut self, v1: N, v2: N) -> N {
        let rep1 = self.find_rep(v1);
        let rep2 = self.find_rep(v2);
        if rep1 == rep2 {
            return rep1;
        }
        let w1 = self.weight.get(&rep1).copied().unwrap_or(1);
        let w2 = self.weight.get(&rep2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (rep1, rep2) } else { (rep2, rep1) };
        self.parent.insert(loser.clone(), winner.clone());
        self.weight.remove(&loser);
        *self.weight.entry(winner.clone()).or_insert(0) = w1 + w2;
        winner
    }
}

// ── RegAllocator (RPython tool/algo/regalloc.py) ──────────────────

/// Private register-allocation work state on FunctionGraph.
///
/// RPython: `regalloc.py::RegAllocator`.
/// Runs BEFORE flatten, on Block/SpaceOperation structure.
#[derive(Debug)]
struct RegAllocatorState {
    depgraph: DependencyGraph<crate::flowspace::model::Variable>,
    unionfind: UnionFind<crate::flowspace::model::Variable>,
    coloring: VarMap<usize>,
}

impl RegAllocatorState {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: UnionFind::new(),
            coloring: VarMap::default(),
        }
    }

    /// RPython: `RegAllocator.make_dependencies()` — regalloc.py.
    /// Per-block die_at analysis.
    fn make_dependencies(
        &mut self,
        graph: &FunctionGraph,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        // `for block in self.graph.iterblocks()` (regalloc.py:27): walk
        // reachable blocks in `iterblocks()` DFS order — not raw `graph.blocks`
        // storage order — so depgraph node insertion (and the coloring derived
        // from it) matches, consistent with `coalesce_variables`.
        for bid in graph.iterblocks_order() {
            self.process_block(graph.block(bid), consider);
        }
    }

    /// Process one block: compute die_at, build interference edges.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    fn process_block(
        &mut self,
        block: &Block,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        // die_at: last usage index of each variable in this block.
        // Keyed on the backing Variable so the coalesce / coloring
        // passes downstream operate on the upstream-orthodox identity
        // (`tool/algo/regalloc.py:31 coloring: dict[Variable, int]`).
        // Pyre models a block's entry values either as `Block.inputargs` or
        // as `OpKind::Input` operations at block entry; both are alive from
        // entry and must be colored like upstream `block.inputargs`.  Collect
        // both so an `Input`-op result used only as an operand still gets a
        // node (otherwise regalloc never colors it and the codewriter
        // liveness pass panics on the uncolored operand).
        let block_input_vars: Vec<crate::flowspace::model::Variable> = block
            .input_variables()
            .cloned()
            .chain(block.operations.iter().filter_map(|op| match &op.kind {
                OpKind::Input { .. } => op.result.clone(),
                _ => None,
            }))
            .collect();
        let mut die_at: VarMap<usize> = VarMap::default();
        for var in &block_input_vars {
            die_at.insert(var.clone(), 0);
        }
        for (i, op) in block.operations.iter().enumerate() {
            if matches!(op.kind, OpKind::Input { .. }) {
                // Pyre's `OpKind::Input` mirrors `Block.inputargs`; upstream
                // RPython has no operation for these values, so they must not
                // extend liveness or look like a second definition.  Their
                // result is already seeded as a block input above.
                continue;
            }
            for var in crate::inline::op_variable_refs(&op.kind) {
                die_at.insert(var, i);
            }
            if let Some(result_var) = op.result.clone() {
                die_at.insert(result_var, i + 1);
            }
        }
        // Variables used in exit links stay alive until block end.
        // RPython `liveness.py` `compute_liveness`:
        // iterate `block.exits` for `link.args` + `block.exitswitch` for the
        // branch condition.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    die_at.remove(var);
                }
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(cond)) => {
                die_at.remove(cond);
            }
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    die_at.remove(arg);
                }
            }
            Some(crate::model::ExitSwitch::LastException) | None => {}
        }
        let mut die_list: Vec<(usize, crate::flowspace::model::Variable)> =
            die_at.into_iter().map(|(v, t)| (t, v)).collect();
        die_list.sort_by_key(|(t, _)| *t);

        // inputargs all interfere with each other
        let livevars: Vec<crate::flowspace::model::Variable> = block_input_vars
            .iter()
            .filter(|var| consider(var))
            .cloned()
            .collect();
        for (i, v) in livevars.iter().enumerate() {
            self.depgraph.add_node(v.clone());
            for prior in livevars.iter().take(i) {
                // Pyre can carry duplicate block input Variables after
                // parity-preserving flow rewrites.  RPython's shared
                // DependencyGraph asserts against self-edges, so skip them
                // locally instead of weakening the color.py port.
                if *prior != *v {
                    self.depgraph.add_edge(prior.clone(), v.clone());
                }
            }
        }
        let mut alive: VarSet = livevars.into_iter().collect();

        // Scan ops, kill at die_at, add interference edges
        let mut die_index = 0;
        for (i, op) in block.operations.iter().enumerate() {
            while die_index < die_list.len() && die_list[die_index].0 == i {
                alive.remove(&die_list[die_index].1);
                die_index += 1;
            }
            if matches!(op.kind, OpKind::Input { .. }) {
                continue;
            }
            if let Some(result_var) = op.result.clone()
                && consider(&result_var)
            {
                self.depgraph.add_node(result_var.clone());
                for v in &alive {
                    // The result can already be represented in `alive`
                    // when upstream-shaped exception/control-flow
                    // rewrites reuse a Variable.  Keep the shared
                    // color.py invariant by not asking it for a
                    // self-edge.
                    if *v != result_var {
                        self.depgraph.add_edge(v.clone(), result_var.clone());
                    }
                }
                alive.insert(result_var);
            }
        }
    }

    /// RPython: `RegAllocator.coalesce_variables()` — regalloc.py.
    /// Coalesce link.args[i] with target.inputargs[i] for every exit
    /// link.  Upstream materialises `list(self.graph.iterblocks())` and
    /// `pop()`s from the END (regalloc.py), coalescing from the
    /// tail of the graph because the tail typically runs more often
    /// during blackholing; this visit order is load-bearing for which
    /// equally-valid coloring — and hence which interior register
    /// numbering — wins.  We reproduce it by walking
    /// [`FunctionGraph::iterblocks_order`] in reverse, NOT `graph.blocks`
    /// storage order (which is not guaranteed to equal `iterblocks()`
    /// order).  Upstream also pre-seeds the depgraph with nodes for
    /// `link.last_exception` / `link.last_exc_value` so any downstream
    /// `getcolor(v)` against those extravars finds a colored node.
    fn coalesce_variables(
        &mut self,
        graph: &FunctionGraph,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        // SSI copies of a merge-point red must still share its colour
        // (`reserve_portal_red_identity` pins that colour). A call
        // result (`scope_from_frame`) or the GETFIELD/vable load that
        // remains when that helper is inlined must not join that
        // class: the back edge would otherwise put the Scope in the
        // reserved vm register, and a mid-opcode guard snapshots it
        // there.
        let portal_reds: VarSet = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .into_iter()
            .flat_map(|kind| portal_merge_point_reds(graph, kind))
            .collect();
        let call_results = collect_protected_results(graph);
        let non_copy_defs = collect_non_copy_results(graph);
        let all_vars: Vec<crate::flowspace::model::Variable> = graph.iter_variables();
        let order = graph.iterblocks_order();
        for &bid in order.iter().rev() {
            let block = graph.block(bid);
            for link in &block.exits {
                // RPython `regalloc.py:92-95`: add `link.last_exception` and
                // `link.last_exc_value` to the dep graph so subsequent
                // `_try_coalesce` calls find a node for them in
                // `DependencyGraph.coalesce`'s `neighbours[vnew]`
                // lookup.  Without this, coalescing a link arg into an
                // exception-target inputarg leaves the rep outside
                // `neighbours`, and `find_node_coloring` silently skips
                // it (`color.py getnodes()` filters
                // `_all_nodes` by `neighbours.contains_key`).
                if let Some(arg) = &link.last_exception
                    && let Some(var) = arg.as_variable()
                {
                    self.depgraph.add_node(var.clone());
                }
                if let Some(arg) = &link.last_exc_value
                    && let Some(var) = arg.as_variable()
                {
                    self.depgraph.add_node(var.clone());
                }
                let target_block = graph.block(link.target);
                let target_input_vars: Vec<crate::flowspace::model::Variable> =
                    target_block.input_variables().cloned().collect();
                for (arg, target_var) in link.args.iter().zip(target_input_vars.iter()) {
                    if let Some(arg_var) = arg.as_variable() {
                        if consider(arg_var) {
                            self.depgraph.add_node(arg_var.clone());
                        }
                        self.try_coalesce(
                            arg_var,
                            target_var,
                            consider,
                            &portal_reds,
                            &call_results,
                            &non_copy_defs,
                            &all_vars,
                        );
                    }
                }
            }
        }
    }

    /// `regalloc.py:_try_coalesce` direct port — operands are
    /// `Variable` instances (matching upstream's `for v, w in
    /// zip(link.args, target.inputargs)`), and the `consider`
    /// predicate reads off the same Variable handle.
    fn try_coalesce(
        &mut self,
        v: &crate::flowspace::model::Variable,
        w: &crate::flowspace::model::Variable,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
        portal_reds: &VarSet,
        call_results: &VarSet,
        non_copy_defs: &VarSet,
        all_vars: &[crate::flowspace::model::Variable],
    ) {
        if !consider(v) || !consider(w) {
            return;
        }
        let v0 = self.unionfind.find_rep(v.clone());
        let w0 = self.unionfind.find_rep(w.clone());
        if v0 == w0 {
            return;
        }
        let v_has_red = class_hits_set(&mut self.unionfind, &v0, portal_reds, all_vars);
        let w_has_red = class_hits_set(&mut self.unionfind, &w0, portal_reds, all_vars);
        let v_has_call = class_hits_set(&mut self.unionfind, &v0, call_results, all_vars);
        let w_has_call = class_hits_set(&mut self.unionfind, &w0, call_results, all_vars);
        if (v_has_red && w_has_call) || (w_has_red && v_has_call) {
            return;
        }
        // A later FieldRead / residual that is not in `call_results`
        // (inlined `scope_from_frame`, a reborrow, a differently
        // named load) must still stay off the reserved red: otherwise
        // the class holds both `self` and `scope` and a mid-opcode
        // guard snapshots Scope in the vm register.
        let v_has_noncopy = class_hits_set(&mut self.unionfind, &v0, non_copy_defs, all_vars);
        let w_has_noncopy = class_hits_set(&mut self.unionfind, &w0, non_copy_defs, all_vars);
        if (v_has_red && w_has_noncopy) || (w_has_red && v_has_noncopy) {
            return;
        }
        if (v_has_red || w_has_red) && std::env::var_os("MAJIT_REGALLOC_DEBUG").is_some() {
            let other = if v_has_red {
                w.name_prefix()
            } else {
                v.name_prefix()
            };
            if other != "self" && other != "frame" && other != "v" {
                eprintln!(
                    "[regalloc] coalesce red with {other} (v={} w={})",
                    v.name_prefix(),
                    w.name_prefix(),
                );
            }
        }
        if self
            .depgraph
            .neighbours
            .get(&w0)
            .is_some_and(|ns| ns.contains(&v0))
        {
            return;
        }
        let rep = self.unionfind.union(v0.clone(), w0.clone());
        if rep == v0 {
            self.depgraph.coalesce(w0, v0);
        } else {
            self.depgraph.coalesce(v0, w0);
        }
    }

    /// Keep portal merge-point reds off every other same-kind colour.
    ///
    /// Generated `#[jit_interp]` states put identity in
    /// `[ref_identity_base, ref_end)` and raise the body's `next_reg`
    /// past that range so a temp cannot reuse the slot. The LLBC
    /// portal has no such floor: `frame` / `vm` are ordinary SSA
    /// values, and once their live range ends the colourer hands the
    /// register to a later ref. A guard mid-opcode then snapshots
    /// that later object under the merge-point's red-R index, and
    /// `rebind_bridge_reds` that trusts the index writes a Vm over a
    /// frame or scope.
    ///
    /// After coalescing, the red and every SSI-threaded copy share a
    /// rep. Interfering that rep with every other considered rep is
    /// the same reservation: one colour, never reused. Graphs with
    /// no `JitMergePoint` are unchanged.
    fn reserve_portal_red_identity(
        &mut self,
        graph: &FunctionGraph,
        kind: RegKind,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        // Identity reds are Ref (frame / vm). Int and float reds are
        // loop-carried values with ordinary lifetimes; reserving those
        // too can push a register-heavy portal past the assembler's
        // 256-register cap.
        if kind != RegKind::Ref {
            return;
        }
        let reds = portal_merge_point_reds(graph, kind);
        if reds.is_empty() {
            return;
        }
        let red_reps: VarSet = reds
            .into_iter()
            .map(|var| self.unionfind.find_rep(var))
            .collect();
        let other_reps: VarSet = graph
            .iter_variables()
            .into_iter()
            .filter(|var| consider(var))
            .map(|var| self.unionfind.find_rep(var))
            .filter(|rep| !red_reps.contains(rep))
            .collect();
        for red in &red_reps {
            self.depgraph.add_node(red.clone());
            for other in &other_reps {
                self.depgraph.add_node(other.clone());
                if red != other {
                    self.depgraph.add_edge(red.clone(), other.clone());
                }
            }
        }
    }

    fn find_node_coloring(&mut self) {
        self.coloring = self.depgraph.find_node_coloring();
    }

    fn getcolor(&mut self, var: &crate::flowspace::model::Variable) -> Option<usize> {
        let rep = self.unionfind.find_rep(var.clone());
        self.coloring.get(&rep).copied()
    }
}

// ── Public API ────────────────────────────────────────────────────

/// Result of register allocation for one kind.
///
/// RPython: `regalloc.py::RegAllocator`, returned by
/// `perform_register_allocation`.
///
/// `coloring` is keyed on the backing
/// [`crate::flowspace::model::Variable`] —
/// matching upstream RPython's `coloring: dict[Variable, int]`
/// (`tool/algo/regalloc.py`).  Consumers hold `&Variable` directly
/// (`flatten.rs:GraphFlattener::getcolor(&Variable)`,
/// `liveness::variable_to_register(&Variable, regallocs)`) and call
/// [`Self::color_for_variable`] / [`Self::contains_variable`].
#[derive(Debug, Clone)]
pub struct RegAllocator {
    pub coloring: VarMap<usize>,
    pub num_regs: usize,
}

/// Register allocation result for the original flowspace graph.
///
/// RPython: `rpython.tool.algo.regalloc.RegAllocator`.  The existing
/// [`RegAllocator`] is the codewriter graph adapter; `shadowcolor.py` runs one
/// phase earlier and therefore keeps the flowspace graph's variable identities
/// directly, exactly as upstream's `perform_register_allocation(graph,
/// consider_var)` does.
#[derive(Debug, Clone)]
pub struct FlowRegAllocator {
    coloring: VarMap<usize>,
    /// RPython `RegAllocator.numcolors`, populated by `find_num_colors()`.
    pub numcolors: usize,
}

impl FlowRegAllocator {
    /// RPython `RegAllocator.getcolor`.
    pub fn getcolor(&self, var: &crate::flowspace::model::Variable) -> usize {
        self.coloring[var]
    }

    /// RPython `RegAllocator.checkcolor`.
    pub fn checkcolor(&self, var: &crate::flowspace::model::Variable, color: usize) -> bool {
        self.coloring.get(var).copied() == Some(color)
    }
}

/// `regalloc.py::perform_register_allocation(graph, consider_var)` for an
/// RPython flowspace graph.
///
/// Rust cannot overload the codewriter adapter's identically named function,
/// so the owner-qualified name records the only API-shape adaptation.  The
/// dependency, reverse coalescing, and coloring passes below follow
/// `RegAllocator.make_dependencies`, `coalesce_variables`, and
/// `find_node_coloring` in their upstream order.
#[expect(
    clippy::mutable_key_type,
    reason = "Variable hashes by immutable identity, matching RPython identity-keyed dicts"
)]
pub fn perform_flowspace_register_allocation(
    graph: &crate::flowspace::model::FunctionGraph,
    consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
) -> FlowRegAllocator {
    use crate::flowspace::model::Hlvalue;

    let mut depgraph = DependencyGraph::new();

    for block_ref in graph.iterblocks() {
        let block = block_ref.borrow();
        let mut die_at: VarMap<usize> = VarMap::default();
        for value in &block.inputargs {
            if let Hlvalue::Variable(var) = value {
                die_at.insert(var.clone(), 0);
            }
        }
        for (index, op) in block.operations.iter().enumerate() {
            for value in &op.args {
                if let Hlvalue::Variable(var) = value {
                    die_at.insert(var.clone(), index);
                }
            }
            if let Hlvalue::Variable(result) = &op.result {
                die_at.insert(result.clone(), index + 1);
            }
        }
        if let Some(Hlvalue::Variable(var)) = &block.exitswitch {
            die_at.remove(var);
        }
        for link_ref in &block.exits {
            for value in link_ref.borrow().args.iter().flatten() {
                if let Hlvalue::Variable(var) = value {
                    die_at.remove(var);
                }
            }
        }
        let mut die_at: Vec<(usize, crate::flowspace::model::Variable)> = die_at
            .into_iter()
            .map(|(var, index)| (index, var))
            .collect();
        die_at.sort_by_key(|(index, _)| *index);

        let inputvars: Vec<_> = block
            .inputargs
            .iter()
            .filter_map(|value| match value {
                Hlvalue::Variable(var) if consider(var) => Some(var.clone()),
                _ => None,
            })
            .collect();
        for (index, var) in inputvars.iter().enumerate() {
            depgraph.add_node(var.clone());
            for other in &inputvars[..index] {
                depgraph.add_edge(other.clone(), var.clone());
            }
        }
        let mut livevars: VarSet = inputvars.into_iter().collect();
        let mut die_index = 0;
        for (index, op) in block.operations.iter().enumerate() {
            while die_at
                .get(die_index)
                .is_some_and(|(last_use, _)| *last_use == index)
            {
                livevars.remove(&die_at[die_index].1);
                die_index += 1;
            }
            if let Hlvalue::Variable(result) = &op.result
                && consider(result)
            {
                depgraph.add_node(result.clone());
                for var in &livevars {
                    if var != result && consider(var) {
                        depgraph.add_edge(var.clone(), result.clone());
                    }
                }
                livevars.insert(result.clone());
            }
        }
    }

    let mut unionfind = UnionFind::new();
    let mut blocks = graph.iterblocks();
    while let Some(block_ref) = blocks.pop() {
        for link_ref in &block_ref.borrow().exits {
            let link = link_ref.borrow();
            if let Some(Hlvalue::Variable(var)) = &link.last_exception {
                depgraph.add_node(var.clone());
            }
            if let Some(Hlvalue::Variable(var)) = &link.last_exc_value {
                depgraph.add_node(var.clone());
            }
            let Some(target) = &link.target else {
                continue;
            };
            for (source, target) in link.args.iter().zip(&target.borrow().inputargs) {
                let (Some(Hlvalue::Variable(source)), Hlvalue::Variable(target)) =
                    (source.as_ref(), target)
                else {
                    continue;
                };
                if !consider(source) || !consider(target) {
                    continue;
                }
                let source_rep = unionfind.find_rep(source.clone());
                let target_rep = unionfind.find_rep(target.clone());
                if source_rep == target_rep || depgraph.has_edge(&target_rep, &source_rep) {
                    continue;
                }
                let representative = unionfind.union(source_rep.clone(), target_rep.clone());
                if representative == source_rep {
                    depgraph.coalesce(target_rep, source_rep);
                } else {
                    depgraph.coalesce(source_rep, target_rep);
                }
            }
        }
    }

    let representative_coloring = depgraph.find_node_coloring();
    let mut coloring = VarMap::default();
    for block_ref in graph.iterblocks() {
        for var in block_ref.borrow().getvariables() {
            if consider(&var) {
                let representative = unionfind.find_rep(var.clone());
                if let Some(color) = representative_coloring.get(&representative) {
                    coloring.insert(var, *color);
                }
            }
        }
    }
    let numcolors = coloring
        .values()
        .copied()
        .max()
        .map_or(0, |color| color + 1);
    FlowRegAllocator {
        coloring,
        numcolors,
    }
}

impl RegAllocator {
    /// Look up the register color assigned to `var` — matches upstream
    /// `coloring: dict[Variable, int]` (`tool/algo/regalloc.py:31`).
    /// Returns `None` when the Variable has no coloring (Void /
    /// Unknown / different kind class).
    pub fn color_for_variable(&self, var: &crate::flowspace::model::Variable) -> Option<usize> {
        self.coloring.get(var).copied()
    }

    /// `true` iff `var` has a coloring in this kind class.
    pub fn contains_variable(&self, var: &crate::flowspace::model::Variable) -> bool {
        self.coloring.contains_key(var)
    }

    /// `tool/algo/regalloc.py swapcolors(col1, col2)` — swap
    /// every Variable holding `col1` with `col2` and vice versa.
    /// Used by `flatten.py enforce_input_args` to renumber
    /// the startblock inputargs into the dense `0..N` prefix of
    /// each kind's color range.
    pub fn swapcolors(&mut self, col1: usize, col2: usize) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }
}

// `perform_register_allocation` reads kinds directly from
// `FunctionGraph::concretetype_of(&v)`, matching upstream
// `regalloc.py::perform_register_allocation(graph, kind)` where
// every Variable's kind comes from `getkind(v.concretetype)`.
// See [`perform_register_allocation`] below.

/// Stamp the canonical `exceptblock.inputargs` kinds onto the graph
/// when they are still `Unknown`.
///
/// Upstream `rpython/rtyper/rclass.py` assigns `(etype, evalue)`
/// concretetypes `Ptr(OBJECT_VTABLE)` / `Ptr(OBJECT)` so
/// `flatten.py:143 raise %r`, `flatten.py:220-231 last_exception/>i`
/// + `goto_if_exception_mismatch/i` see canonical kinds.  Pyre's
///   codewriter creates the canonical exceptblock eagerly in
///   `FunctionGraph::new` with `Unknown` placeholders; this helper
///   stamps the canonical Signed / GcRef kinds whenever the rtyper
///   hand-off (`apply_to_graph` / `apply_from_flowspace_variables`)
///   did not. This function writes directly to each backing
///   `Variable.concretetype` cell through
///   `FunctionGraph::set_concretetype_of_inline`, preserving the graph as the
///   sole kind owner.
pub(crate) fn augment_canonical_exceptblock_on_graph(graph: &mut FunctionGraph) {
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        if matches!(
            FunctionGraph::concretetype_of(&except_args[0]),
            ConcreteType::Unknown
        ) {
            FunctionGraph::set_concretetype_of_inline(&except_args[0], ConcreteType::Signed);
        }
        if matches!(
            FunctionGraph::concretetype_of(&except_args[1]),
            ConcreteType::Unknown
        ) {
            FunctionGraph::set_concretetype_of_inline(&except_args[1], ConcreteType::GcRef);
        }
    }
}

/// Perform register allocation for all three kinds — `&FunctionGraph`-only.
///
/// RPython parity: every `Variable.concretetype` is the source of
/// kind; pyre reads each per-value kind via
/// `FunctionGraph::concretetype_of(&v)`, projecting the
/// [`ConcreteType`] enum onto the JIT codewriter's
/// [`RegKind`] partitioning axis.  Canonical exceptblock inputargs
/// are stamped on the graph up-front via
/// [`augment_canonical_exceptblock_on_graph`].
pub(crate) fn perform_all_register_allocations(
    graph: &FunctionGraph,
) -> HashMap<RegKind, RegAllocator> {
    // Fail loud if the canonical exceptblock inputargs are still
    // `Unknown` — `variable_regkind` silently drops `Unknown` so a
    // missed call to [`augment_canonical_exceptblock_on_graph`] would
    // leave `last_exception` / `last_exc_value` un-coloured (no
    // register class), and any later flatten/assembler pass would
    // emit ops that reference uncolored values without any diagnostic.
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        assert!(
            !matches!(
                FunctionGraph::concretetype_of(&except_args[0]),
                ConcreteType::Unknown
            ) && !matches!(
                FunctionGraph::concretetype_of(&except_args[1]),
                ConcreteType::Unknown
            ),
            "perform_all_register_allocations: canonical exceptblock inputargs are still \
             Unknown — caller must run augment_canonical_exceptblock_on_graph() before \
             register allocation (graph: {})",
            graph.name,
        );
    }
    let mut result = HashMap::new();
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        result.insert(kind, perform_register_allocation(graph, kind));
    }
    result
}

/// `regalloc.py::perform_register_allocation(graph, kind)` direct
/// port.  Runs on FunctionGraph (Block structure), BEFORE flatten.
/// Reads kind from `FunctionGraph::concretetype_of(&v)` exactly like
/// upstream reads `getkind(v.concretetype)`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn perform_register_allocation(graph: &FunctionGraph, kind: RegKind) -> RegAllocator {
    let consider =
        |var: &crate::flowspace::model::Variable| -> bool { variable_regkind(var) == Some(kind) };
    let mut allocator = RegAllocatorState::new();
    allocator.make_dependencies(graph, &consider);
    allocator.coalesce_variables(graph, &consider);
    allocator.reserve_portal_red_identity(graph, kind, &consider);
    allocator.find_node_coloring();

    let mut coloring: VarMap<usize> = VarMap::default();
    let mut max_reg = 0usize;
    // Walk every Variable minted on the graph and pick those whose
    // concretetype lands in `kind`.  `getcolor` projects through the
    // unionfind rep to recover the chordal coloring entry — matches
    // upstream `regalloc.py:118 self.coloring[self.unionfind.find_rep(v)]`.
    for var in graph.iter_variables() {
        if variable_regkind(&var) == Some(kind)
            && let Some(color) = allocator.getcolor(&var)
        {
            coloring.insert(var.clone(), color);
            if color + 1 > max_reg {
                max_reg = color + 1;
            }
        }
    }
    if kind == RegKind::Ref && std::env::var_os("MAJIT_REGALLOC_DEBUG").is_some() {
        let reds = portal_merge_point_reds(graph, kind);
        if !reds.is_empty() || graph.name.contains("run_frame") {
            eprintln!(
                "[regalloc] graph {} merge_reds={} colored={}",
                graph.name,
                reds.len(),
                coloring.len(),
            );
        }
        log_portal_ref_colors(graph, &coloring);
    }
    RegAllocator {
        coloring,
        num_regs: max_reg,
    }
}

/// [`crate::flowspace::model::Variable`] → [`RegKind`] projection,
/// reading the Variable's inline `concretetype` cell directly.  Mirrors
/// upstream RPython's `getkind(v.concretetype)` (`history.py`).
fn variable_regkind(var: &crate::flowspace::model::Variable) -> Option<RegKind> {
    let ct = match var.concretetype.borrow().as_ref() {
        Some(lltype) => crate::model::getkind(lltype),
        None => ConcreteType::Unknown,
    };
    concretetype_to_regkind(&ct)
}

/// `getkind`'s [`ConcreteType`] → [`RegKind`] projection:
/// Signed → Int, GcRef → Ref, Float → Float.  Void / Unknown have
/// no register class (the same way RPython's regalloc skips Void
/// Variables, `flatten.py:325`).
/// Merge-point reds of `kind` on this graph, if it is a portal.
fn portal_merge_point_reds(
    graph: &FunctionGraph,
    kind: RegKind,
) -> Vec<crate::flowspace::model::Variable> {
    let mut reds = Vec::new();
    for bid in graph.iterblocks_order() {
        for op in &graph.block(bid).operations {
            let OpKind::JitMergePoint {
                reds_i,
                reds_r,
                reds_f,
                ..
            } = &op.kind
            else {
                continue;
            };
            let list = match kind {
                RegKind::Int => reds_i.as_slice(),
                RegKind::Ref => reds_r.as_slice(),
                RegKind::Float => reds_f.as_slice(),
            };
            for var in list {
                if variable_regkind(var) == Some(kind) {
                    reds.push(var.clone());
                }
            }
        }
    }
    reds
}

fn is_defining_call(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call { .. }
            | OpKind::CallResidual { .. }
            | OpKind::CallElidable { .. }
            | OpKind::CallMayForce { .. }
            | OpKind::IndirectCall { .. }
            | OpKind::InlineCall { .. }
    )
}

fn log_portal_ref_colors(graph: &FunctionGraph, coloring: &VarMap<usize>) {
    let reds = portal_merge_point_reds(graph, RegKind::Ref);
    if reds.is_empty() {
        return;
    }
    let red_colors: Vec<Option<usize>> =
        reds.iter().map(|var| coloring.get(var).copied()).collect();
    let mut call_hits = 0usize;
    let mut scope_calls = 0usize;
    let mut scope_fields = 0usize;
    for bid in graph.iterblocks_order() {
        for op in &graph.block(bid).operations {
            if is_defining_call(&op.kind) {
                call_hits += 1;
                if call_kind_mentions_scope(&op.kind) {
                    scope_calls += 1;
                }
            }
            if is_scope_load(&op.kind) {
                scope_fields += 1;
            }
        }
    }
    let protected = collect_protected_results(graph);
    let shared = protected
        .iter()
        .filter(|var| {
            coloring
                .get(*var)
                .is_some_and(|color| red_colors.contains(&Some(*color)))
        })
        .count();
    let mut red_occupancy: Vec<(usize, usize)> = Vec::new();
    for color in red_colors.iter().flatten() {
        let n = coloring.values().filter(|c| *c == color).count();
        red_occupancy.push((*color, n));
    }
    let mut kind_hits: Vec<(String, usize)> = Vec::new();
    let mut samples: Vec<String> = Vec::new();
    for bid in graph.iterblocks_order() {
        for op in &graph.block(bid).operations {
            let Some(result) = &op.result else {
                continue;
            };
            let Some(color) = coloring.get(result) else {
                continue;
            };
            if !red_colors.contains(&Some(*color)) {
                continue;
            }
            let label = match &op.kind {
                OpKind::Call { target, .. }
                | OpKind::CallResidual {
                    funcptr: crate::model::CallFuncPtr::Target(target),
                    ..
                }
                | OpKind::CallElidable {
                    funcptr: crate::model::CallFuncPtr::Target(target),
                    ..
                }
                | OpKind::CallMayForce {
                    funcptr: crate::model::CallFuncPtr::Target(target),
                    ..
                } => format!("call:{target}"),
                OpKind::FieldRead { field, .. } | OpKind::InteriorFieldRead { field, .. } => {
                    format!("field:{}", field.name)
                }
                OpKind::VableFieldRead { .. } => "vable".into(),
                OpKind::Input { name, .. } => format!("input:{name}"),
                other => format!("{other:?}")
                    .split_once(' ')
                    .map(|(head, _)| head.to_string())
                    .unwrap_or_else(|| format!("{other:?}")),
            };
            if let Some((_, n)) = kind_hits.iter_mut().find(|(k, _)| *k == label) {
                *n += 1;
            } else {
                kind_hits.push((label.clone(), 1));
            }
            if samples.len() < 24 {
                samples.push(format!("r{color}:{label}"));
            }
        }
    }
    kind_hits.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut name_hits: Vec<(String, usize)> = Vec::new();
    for (var, color) in coloring {
        if !red_colors.contains(&Some(*color)) {
            continue;
        }
        let prefix = var.name_prefix();
        if let Some((_, n)) = name_hits.iter_mut().find(|(k, _)| *k == prefix) {
            *n += 1;
        } else {
            name_hits.push((prefix, 1));
        }
    }
    name_hits.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if name_hits.len() > 16 {
        name_hits.truncate(16);
    }
    eprintln!(
        "[regalloc] portal {} ref reds={} colors={red_colors:?} \
         occupancy={red_occupancy:?} calls={call_hits} scope_calls={scope_calls} \
         scope_fields={scope_fields} protected_sharing_red={shared} colored={} \
         red_kinds={kind_hits:?} samples={samples:?} red_names={name_hits:?}",
        graph.name,
        reds.len(),
        coloring.len(),
    );
}

fn call_kind_mentions_scope(kind: &OpKind) -> bool {
    let target = match kind {
        OpKind::Call { target, .. }
        | OpKind::CallElidable {
            funcptr: crate::model::CallFuncPtr::Target(target),
            ..
        }
        | OpKind::CallResidual {
            funcptr: crate::model::CallFuncPtr::Target(target),
            ..
        }
        | OpKind::CallMayForce {
            funcptr: crate::model::CallFuncPtr::Target(target),
            ..
        } => target,
        _ => return false,
    };
    target.to_string().contains("scope_from_frame")
}

fn is_scope_load(kind: &OpKind) -> bool {
    match kind {
        OpKind::FieldRead { field, .. } | OpKind::InteriorFieldRead { field, .. } => {
            field.name == "scope"
        }
        OpKind::VableFieldRead { .. } => true,
        _ => false,
    }
}

fn is_protected_result_op(kind: &OpKind) -> bool {
    // Any producer that is not an SSI/`same_as` copy can be a later
    // object (Scope, engine, a residual). Seed the frontier with it
    // so a reverse-order coalesce cannot union its inputarg copy
    // with a portal red before the producer is attached.
    !is_copy_kind(kind)
}

fn is_copy_kind(kind: &OpKind) -> bool {
    match kind {
        OpKind::Input { .. } => true,
        OpKind::UnaryOp { op, .. } if op == "same_as" => true,
        _ => false,
    }
}

fn collect_non_copy_results(graph: &FunctionGraph) -> VarSet {
    let mut results = VarSet::default();
    for bid in graph.iterblocks_order() {
        for op in &graph.block(bid).operations {
            if let Some(result) = &op.result
                && !is_copy_kind(&op.kind)
            {
                results.insert(result.clone());
            }
        }
    }
    results
}

fn collect_protected_results(graph: &FunctionGraph) -> VarSet {
    let mut results = VarSet::default();
    let mut frontier = VarSet::default();
    for bid in graph.iterblocks_order() {
        for op in &graph.block(bid).operations {
            if let Some(result) = &op.result {
                if is_defining_call(&op.kind) {
                    results.insert(result.clone());
                }
                if is_protected_result_op(&op.kind) {
                    results.insert(result.clone());
                    frontier.insert(result.clone());
                }
            }
        }
    }
    // Copies of a protected load are what the body actually uses:
    // `same_as`, and the SSI inputarg a predecessor passes the load
    // into. Leaving those unmarked lets a later back edge union the
    // copy with the vm red while the GETFIELD result stays a
    // different colour — occupancy then shows ~1300 vars in the red
    // class and a mid-opcode guard snapshots Scope under the vm
    // index.
    let mut grew = true;
    while grew {
        grew = false;
        for bid in graph.iterblocks_order() {
            let block = graph.block(bid);
            for op in &block.operations {
                let OpKind::UnaryOp {
                    op: name, operand, ..
                } = &op.kind
                else {
                    continue;
                };
                if name != "same_as" || !frontier.contains(operand) {
                    continue;
                }
                if let Some(result) = &op.result
                    && frontier.insert(result.clone())
                {
                    results.insert(result.clone());
                    grew = true;
                }
            }
            for link in &block.exits {
                let target_inputs: Vec<crate::flowspace::model::Variable> = graph
                    .block(link.target)
                    .input_variables()
                    .cloned()
                    .collect();
                for (arg, target) in link.args.iter().zip(target_inputs.iter()) {
                    let Some(arg_var) = arg.as_variable() else {
                        continue;
                    };
                    if frontier.contains(arg_var) && frontier.insert(target.clone()) {
                        results.insert(target.clone());
                        grew = true;
                    }
                }
            }
        }
    }
    results
}

fn class_hits_set(
    unionfind: &mut UnionFind<crate::flowspace::model::Variable>,
    root: &crate::flowspace::model::Variable,
    marked: &VarSet,
    all: &[crate::flowspace::model::Variable],
) -> bool {
    all.iter()
        .any(|var| marked.contains(var) && &unionfind.find_rep(var.clone()) == root)
}

fn concretetype_to_regkind(ty: &ConcreteType) -> Option<RegKind> {
    match ty {
        ConcreteType::Signed => Some(RegKind::Int),
        ConcreteType::GcRef => Some(RegKind::Ref),
        ConcreteType::Float => Some(RegKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallTarget, ExitCase, ExitSwitch, FunctionGraph, Link, OpKind, ValueType};

    fn push_int_input(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        name: &str,
    ) -> crate::flowspace::model::Variable {
        let var = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: name.into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, var.clone());
        var
    }

    #[test]
    fn non_overlapping_lifetimes_share_register() {
        // v0 = Input; v1 = BinOp(v0, v0); Return v1
        // v0 dies when v1 is defined → no interference → can share register.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let v1_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v0_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        // v0 and v1 don't overlap → can share
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn overlapping_lifetimes_need_different_registers() {
        // v0 = Input; v1 = Input; v2 = BinOp(v0, v1); Return v2
        // v0 and v1 are both alive when v2 is defined → v0 and v1 interfere
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let v1_var = push_int_input(&mut graph, entry, "b");
        let v2_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v1_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v2_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v2_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_ne!(
            result.color_for_variable(&v0_var),
            result.color_for_variable(&v1_var),
            "v0 and v1 are simultaneously alive → different registers"
        );
        // v2 can share with v0 or v1 (they die before v2's definition)
        assert!(result.num_regs >= 2);
    }

    #[test]
    fn goto_link_coalescing() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let (block1, block1_args) = graph.create_block_with_arg_vars(1);
        let v1_var = block1_args[0].clone();
        graph.set_goto(entry, block1, vec![v0_var.clone()]);
        graph.set_return(block1, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_eq!(
            result.color_for_variable(&v0_var),
            result.color_for_variable(&v1_var),
        );
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn fused_exitswitch_args_stay_live_until_branch() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let seed_var = push_int_input(&mut graph, entry, "seed");
        let x_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: seed_var.clone(),
                    rhs: seed_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let y_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: seed_var.clone(),
                    rhs: seed_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let false_block = graph.create_block();
        let true_block = graph.create_block();
        graph.set_return(false_block, None);
        graph.set_return(true_block, None);
        let false_link =
            Link::from_variables(&graph, vec![], false_block, Some(ExitCase::Bool(false)));
        let true_link =
            Link::from_variables(&graph, vec![], true_block, Some(ExitCase::Bool(true)));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Fused {
                opname: "int_lt".into(),
                args: vec![x_var.clone(), y_var.clone()],
            }),
            vec![false_link, true_link],
        );

        FunctionGraph::set_concretetype_of_inline(&seed_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&x_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&y_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_ne!(
            result.color_for_variable(&x_var),
            result.color_for_variable(&y_var),
            "fused exitswitch operands are both read by GotoIfNotOp"
        );
    }

    #[test]
    fn coloring_unbounded() {
        let mut dg = DependencyGraph::<u16>::new();
        for i in 0..100u16 {
            dg.add_node(i);
        }
        for i in 0..99u16 {
            dg.add_edge(i, i + 1);
        }
        let coloring = dg.find_node_coloring();
        assert_eq!(coloring.len(), 100);
        let max_color = coloring.values().max().copied().unwrap_or(0);
        assert!(
            max_color <= 1,
            "chain needs at most 2 colors, got {}",
            max_color + 1
        );
    }

    fn push_ref_input(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        name: &str,
    ) -> crate::flowspace::model::Variable {
        let var = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: name.into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, var.clone());
        var
    }

    #[test]
    fn portal_merge_point_red_does_not_share_a_register_with_a_later_ref() {
        // frame dies at the merge point; temp is born after. Without the
        // reservation they share a colour (non_overlapping_lifetimes).
        // The portal red must keep its colour so a mid-opcode guard still
        // names the Vm/frame, not the later object.
        let mut graph = FunctionGraph::new("portal");
        let entry = graph.startblock;
        let frame = push_ref_input(&mut graph, entry, "frame");
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![frame.clone()],
                reds_f: vec![],
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        let temp = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        graph.set_return(entry, Some(temp.clone()));

        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&temp, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_ne!(
            result.color_for_variable(&frame),
            result.color_for_variable(&temp),
            "a portal red must not share its colour with a later ref",
        );
        assert!(result.num_regs >= 2);
    }

    #[test]
    fn a_graph_without_a_merge_point_still_shares_non_overlapping_refs() {
        let mut graph = FunctionGraph::new("helper");
        let entry = graph.startblock;
        let early = push_ref_input(&mut graph, entry, "early");
        let late = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        graph.set_return(entry, Some(late.clone()));

        FunctionGraph::set_concretetype_of_inline(&early, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&late, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_eq!(
            result.color_for_variable(&early),
            result.color_for_variable(&late),
            "reservation is inert off a portal",
        );
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn portal_red_does_not_coalesce_with_a_call_result() {
        // Back edge would otherwise union the call result with the
        // merge-point red and put the Scope in the reserved vm colour.
        let mut graph = FunctionGraph::new("portal");
        let entry = graph.startblock;
        let frame = push_ref_input(&mut graph, entry, "frame");
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![frame.clone()],
                reds_f: vec![],
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        let temp = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["scope_from_frame"]),
                    args: crate::model::call_args(vec![frame.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        graph.set_goto(entry, entry, vec![temp.clone()]);

        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&temp, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_ne!(
            result.color_for_variable(&frame),
            result.color_for_variable(&temp),
            "a call result must not share the merge-point red's colour",
        );
    }

    #[test]
    fn portal_red_does_not_coalesce_with_a_scope_field_read() {
        // The live portal still inlines `scope_from_frame` to a GETFIELD
        // of `frame.scope`. That result must stay off the reserved vm
        // colour the same way a residual call result does.
        let mut graph = FunctionGraph::new("portal");
        let entry = graph.startblock;
        let frame = push_ref_input(&mut graph, entry, "frame");
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![frame.clone()],
                reds_f: vec![],
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        let temp = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: frame.clone(),
                    field: crate::model::FieldDescriptor::new("scope", Some("GrainFrame".into())),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_goto(entry, entry, vec![temp.clone()]);

        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&temp, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_ne!(
            result.color_for_variable(&frame),
            result.color_for_variable(&temp),
            "a scope field read must not share the merge-point red's colour",
        );
    }

    #[test]
    fn portal_red_does_not_coalesce_with_a_threaded_scope_copy() {
        // The GETFIELD result is passed into a successor inputarg; that
        // copy is what a later back edge would union with the red.
        let mut graph = FunctionGraph::new("portal");
        let entry = graph.startblock;
        let mid = graph.create_block();
        let frame = push_ref_input(&mut graph, entry, "frame");
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![frame.clone()],
                reds_f: vec![],
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        let loaded = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: frame.clone(),
                    field: crate::model::FieldDescriptor::new("scope", Some("GrainFrame".into())),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let mid_in = graph
            .push_op_var(
                mid,
                OpKind::Input {
                    name: "scope_copy".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(mid, mid_in.clone());
        graph.set_goto(entry, mid, vec![loaded.clone()]);
        graph.set_goto(mid, entry, vec![mid_in.clone()]);

        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&loaded, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&mid_in, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_ne!(
            result.color_for_variable(&frame),
            result.color_for_variable(&mid_in),
            "a threaded scope copy must not share the merge-point red's colour",
        );
    }

    #[test]
    fn portal_red_does_not_coalesce_with_a_threaded_non_scope_load() {
        // Inlined helpers leave FieldReads whose name is not `scope`
        // (`engine`, a reborrow). Those were not in `call_results`, so a
        // back edge could union them with the vm red.
        let mut graph = FunctionGraph::new("portal");
        let entry = graph.startblock;
        let mid = graph.create_block();
        let frame = push_ref_input(&mut graph, entry, "frame");
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![],
                greens_r: vec![],
                greens_f: vec![],
                reds_i: vec![],
                reds_r: vec![frame.clone()],
                reds_f: vec![],
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        let loaded = graph
            .push_op_var(
                entry,
                OpKind::FieldRead {
                    base: frame.clone(),
                    field: crate::model::FieldDescriptor::new("engine", Some("Vm".into())),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let mid_in = graph
            .push_op_var(
                mid,
                OpKind::Input {
                    name: "engine_copy".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(mid, mid_in.clone());
        graph.set_goto(entry, mid, vec![loaded.clone()]);
        graph.set_goto(mid, entry, vec![mid_in.clone()]);

        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&loaded, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&mid_in, ConcreteType::GcRef);
        let result = perform_register_allocation(&graph, RegKind::Ref);
        assert_ne!(
            result.color_for_variable(&frame),
            result.color_for_variable(&mid_in),
            "a threaded non-scope load must not share the merge-point red's colour",
        );
    }
}
