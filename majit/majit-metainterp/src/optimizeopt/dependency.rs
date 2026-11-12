//! Dependency graph for vectorization.
//!
//! Mirrors RPython's `dependency.py`: builds a DAG of data dependencies
//! between operations in a loop body. Used by the vector optimizer to
//! identify independent operations that can be packed into SIMD instructions.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicI32, Ordering};

use indexmap::IndexSet;

use crate::optimizeopt::schedule::Pack;
use majit_ir::operand::Operand;
use majit_ir::{Op, OpCode, OpRef};

// ── dependency.py: LOAD/MODIFY_COMPLEX_OBJ tables ─────────

/// dependency.py: LOAD_COMPLEX_OBJ — returns (complex_obj_arg_idx, index_arg_idx).
/// index_arg_idx == -1 means no index argument (field access, not array).
fn load_complex_obj_args(opcode: OpCode) -> (usize, i32) {
    match opcode {
        // Array loads: (array, index)
        OpCode::GetarrayitemGcI
        | OpCode::GetarrayitemGcF
        | OpCode::GetarrayitemGcR
        | OpCode::GetarrayitemRawI
        | OpCode::GetarrayitemRawF
        | OpCode::RawLoadI
        | OpCode::RawLoadF
        | OpCode::VecLoadI
        | OpCode::VecLoadF => (0, 1),
        // Interior field: (obj, index)
        OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcF | OpCode::GetinteriorfieldGcR => {
            (0, 1)
        }
        // Field loads: (obj, no index)
        OpCode::GetfieldGcI
        | OpCode::GetfieldGcR
        | OpCode::GetfieldGcF
        | OpCode::GetfieldRawI
        | OpCode::GetfieldRawR
        | OpCode::GetfieldRawF => (0, -1),
        _ => (0, -1),
    }
}

/// dependency.py: MODIFY_COMPLEX_OBJ — returns (complex_obj_arg_idx, cell_arg_idx).
/// cell_arg_idx == -1 means no cell argument (field store, not array).
fn modify_complex_obj_args(opcode: OpCode) -> Option<(usize, i32)> {
    match opcode {
        // Array stores: (array, index)
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw | OpCode::RawStore | OpCode::VecStore => {
            Some((0, 1))
        }
        // Interior field stores: (obj, no cell)
        OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => Some((0, -1)),
        // Field stores: (obj, no cell)
        OpCode::SetfieldGc | OpCode::SetfieldRaw => Some((0, -1)),
        // Other
        OpCode::ZeroArray => Some((0, -1)),
        OpCode::Strsetitem | OpCode::Unicodesetitem => Some((0, -1)),
        _ => None,
    }
}

/// dependency.py: side_effect_arguments — determine which args are
/// destroyed (modified) by the operation. Returns Vec<(arg, argcell, destroyed)>.
/// `arg_type_of` resolves an OpRef to its result type for the float check.
fn side_effect_arguments(
    op: &Op,
    arg_type_of: &dyn Fn(OpRef) -> majit_ir::Type,
) -> Vec<(OpRef, Option<OpRef>, bool)> {
    let mut result = Vec::new();
    if op.opcode.is_complex_modify() {
        // dependency.py:218-230: known complex modification patterns
        if let Some((obj_idx, cell_idx)) = modify_complex_obj_args(op.opcode)
            && obj_idx < op.num_args()
        {
            if cell_idx >= 0 && (cell_idx as usize) < op.num_args() {
                result.push((
                    op.arg(obj_idx).to_opref(),
                    Some(op.arg(cell_idx as usize).to_opref()),
                    true,
                ));
                for j in (cell_idx as usize + 1)..op.num_args() {
                    result.push((op.arg(j).to_opref(), None, false));
                }
            } else {
                result.push((op.arg(obj_idx).to_opref(), None, true));
                for j in (obj_idx + 1)..op.num_args() {
                    result.push((op.arg(j).to_opref(), None, false));
                }
            }
        }
    } else {
        // dependency.py:232-240: generic side effect
        for arg in op.getarglist().iter() {
            // dependency.py:237: arg.is_constant() or arg.type == 'f' → not destroyed
            if arg.is_constant() || arg_type_of(arg.to_opref()) == majit_ir::Type::Float {
                result.push((arg.to_opref(), None, false));
            } else {
                result.push((arg.to_opref(), None, true));
            }
        }
    }
    result
}

/// dependency.py `Path`.
///
/// RPython stores `Node` objects directly. The Rust dependency graph uses
/// stable node indices in `DependencyGraph.nodes` — including imaginary
/// nodes (`op=None`) — so `Path` stores those indices uniformly and accepts
/// the node slice when it needs to inspect operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Path {
    path: Vec<usize>,
}

impl Path {
    pub fn new(path: Vec<usize>) -> Self {
        Self { path }
    }

    /// dependency.py `second`.
    pub fn second(&self) -> Option<usize> {
        if self.path.len() <= 1 {
            return None;
        }
        self.node_at(1)
    }

    /// dependency.py `last_but_one`.
    pub fn last_but_one(&self) -> Option<usize> {
        if self.path.len() < 2 {
            return None;
        }
        self.node_at(self.path.len() - 2)
    }

    /// dependency.py `last`.
    pub fn last(&self) -> Option<usize> {
        if self.path.is_empty() {
            return None;
        }
        self.node_at(self.path.len() - 1)
    }

    /// dependency.py `first`.
    pub fn first(&self) -> Option<usize> {
        self.node_at(0)
    }

    /// `self.path[index]` — the node index at `index` (real or imaginary).
    fn node_at(&self, index: usize) -> Option<usize> {
        self.path.get(index).copied()
    }

    /// dependency.py `is_always_pure`.
    pub fn is_always_pure(&self, nodes: &[Node], exclude_first: bool, exclude_last: bool) -> bool {
        let mut i = usize::from(exclude_first);
        let mut count = self.path.len();
        if exclude_last {
            count = count.saturating_sub(1);
        }
        while i < count {
            let Some(node) = nodes.get(self.path[i]) else {
                return false;
            };
            // dependency.py:84-86: skip imaginary segments.
            if node.is_imaginary() {
                i += 1;
                continue;
            }
            let op = node.op();
            if op.opcode.is_guard() {
                let exits_early = op.with_fail_descr(|fd| fd.exits_early()).unwrap_or(false);
                if !exits_early {
                    return false;
                }
            } else if !op.opcode.is_always_pure() {
                return false;
            }
            i += 1;
        }
        true
    }

    /// dependency.py `set_schedule_priority` — sets the priority on
    /// every segment, imaginary nodes included.
    pub fn set_schedule_priority(&self, nodes: &mut [Node], priority: i32) {
        for &index in &self.path {
            if let Some(node) = nodes.get_mut(index) {
                node.setpriority(priority);
            }
        }
    }

    /// dependency.py `walk`.
    pub fn walk_node(&mut self, node: usize) {
        self.path.push(node);
    }

    /// dependency.py `cut_off_at`.
    pub fn cut_off_at(&mut self, index: usize) {
        self.path.truncate(index);
    }

    /// dependency.py `check_acyclic`.
    pub fn check_acyclic(&self) -> bool {
        for (index, item) in self.path.iter().enumerate() {
            if self.path[..index].iter().any(|previous| previous == item) {
                return false;
            }
        }
        true
    }

    /// dependency.py `clone`.
    pub fn clone_path(&self) -> Self {
        self.clone()
    }

    /// dependency.py `as_str`.
    pub fn as_str(&self, nodes: &[Node]) -> String {
        self.path
            .iter()
            .map(|&index| match nodes.get(index) {
                Some(node) if node.is_imaginary() => node
                    .dotlabel
                    .clone()
                    .unwrap_or_else(|| format!("imaginary({index})")),
                _ => format!("Node({index})"),
            })
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

static IMAGINARY_NODE_INDEX: AtomicI32 = AtomicI32::new(987_654_321);

/// dependency.py Node: A node in the dependency graph.
/// Each node wraps one operation and maintains forward/backward dependency edges.
#[derive(Clone, Debug)]
pub struct Node {
    /// Index in the ops list (dependency.py:134: opidx).
    pub idx: usize,
    /// The operation (dependency.py:133: op). `None` for an imaginary node —
    /// `ImaginaryNode.__init__` passes `op=None` (dependency.py).
    pub op: Option<Op>,
    /// dependency.py:400 `dotlabel` — debug label carried only by imaginary
    /// nodes; `None` for real nodes.
    pub dotlabel: Option<String>,
    /// dependency.py:135: adjacent_list — forward dependency edges (this → target).
    pub adjacent_list: Vec<Dependency>,
    /// dependency.py:136: adjacent_list_back — backward dependency edges (source → this).
    pub adjacent_list_back: Vec<Dependency>,
    /// dependency.py: memory_ref — MemoryRef for array access ops.
    pub memory_ref: Option<MemoryRef>,
    /// dependency.py:138: pack — which Pack this node belongs to.
    pub pack: Option<usize>,
    /// dependency.py:139: pack_position
    pub pack_position: i32,
    /// dependency.py:140: emitted — whether this node has been scheduled.
    pub emitted: bool,
    /// dependency.py:141: schedule_position
    pub schedule_position: i32,
    /// dependency.py:142: priority — scheduling priority.
    pub priority: i32,
    /// Compat: indices of operations this one depends on.
    pub deps: Vec<usize>,
    /// Compat: indices of operations that depend on this one.
    pub users: Vec<usize>,
}

impl Node {
    pub fn new(op: Op, opidx: usize) -> Self {
        Node {
            idx: opidx,
            op: Some(op),
            dotlabel: None,
            adjacent_list: Vec::new(),
            adjacent_list_back: Vec::new(),
            memory_ref: None,
            pack: None,
            pack_position: -1,
            emitted: false,
            schedule_position: -1,
            priority: 0,
            deps: Vec::new(),
            users: Vec::new(),
        }
    }

    /// dependency.py `ImaginaryNode(label)` — a synthetic dependency
    /// vertex with `op=None`. Untranslated it carries a `dotlabel` and a fake
    /// index drawn from a big monotonic counter.
    pub fn new_imaginary(label: impl Into<String>) -> Self {
        let index = IMAGINARY_NODE_INDEX.fetch_add(1, Ordering::Relaxed) as usize;
        let mut node = Node::new_placeholder(index);
        node.dotlabel = Some(label.into());
        node
    }

    fn new_placeholder(idx: usize) -> Self {
        Node {
            idx,
            op: None,
            dotlabel: None,
            adjacent_list: Vec::new(),
            adjacent_list_back: Vec::new(),
            memory_ref: None,
            pack: None,
            pack_position: -1,
            emitted: false,
            schedule_position: -1,
            priority: 0,
            deps: Vec::new(),
            users: Vec::new(),
        }
    }

    /// dependency.py `getoperation` — `self.op`, `None` when imaginary.
    pub fn getoperation(&self) -> Option<&Op> {
        self.op.as_ref()
    }

    /// The operation of a real node. Panics on an imaginary node, mirroring
    /// upstream where `node.op.<attr>` on an `ImaginaryNode` (op=None) raises.
    pub fn op(&self) -> &Op {
        self.op
            .as_ref()
            .expect("Node::op called on an imaginary node (op=None)")
    }

    /// dependency.py / 405-406 `is_imaginary` — true iff `op is None`.
    pub fn is_imaginary(&self) -> bool {
        self.op.is_none()
    }

    /// dependency.py: setpriority
    pub fn setpriority(&mut self, value: i32) {
        self.priority = value;
    }

    /// dependency.py: provides_count
    pub fn provides_count(&self) -> usize {
        self.adjacent_list.len()
    }

    /// dependency.py: depends_count
    pub fn depends_count(&self) -> usize {
        self.adjacent_list_back.len()
    }

    /// dependency.py `provides` — forward dependency edges
    /// (this node → target). Each edge's `target_node()` is the successor.
    pub fn provides(&self) -> &[Dependency] {
        &self.adjacent_list
    }

    /// dependency.py `depends` — backward dependency edges. These are
    /// the reversed back-edges built by `add_edge`, so each edge's
    /// `target_node()` is the predecessor.
    pub fn depends(&self) -> &[Dependency] {
        &self.adjacent_list_back
    }

    /// dependency.py: is_after
    pub fn is_after(&self, other_idx: usize) -> bool {
        self.idx > other_idx
    }

    /// dependency.py: is_before
    pub fn is_before(&self, other_idx: usize) -> bool {
        self.idx < other_idx
    }

    /// dependency.py: is_pure
    pub fn is_pure(&self) -> bool {
        self.op().opcode.is_always_pure()
    }

    /// dependency.py: exits_early
    pub fn exits_early(&self) -> bool {
        if self.op().opcode.is_guard() {
            // dependency.py: descr = self.op.getdescr(); return descr.exits_early()
            self.op()
                .with_fail_descr(|fd| fd.exits_early())
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// dependency.py: loads_from_complex_object
    pub fn loads_from_complex_object(&self) -> bool {
        self.op().opcode.is_complex_load()
    }

    /// dependency.py: modifies_complex_object
    pub fn modifies_complex_object(&self) -> bool {
        self.op().opcode.is_complex_modify()
    }
}

/// dependency.py: DependencyGraph — dependency graph for a loop body.
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    pub nodes: Vec<Node>,
    /// dependency.py: memory_refs — node index → MemoryRef
    pub memory_refs: indexmap::IndexMap<usize, MemoryRef>,
    /// dependency.py: index_vars — OpRef → IndexVar
    pub index_vars: indexmap::IndexMap<OpRef, IndexVar>,
    /// dependency.py:571: guards — guard node indices
    pub guards: Vec<usize>,
    /// dependency.py:565: invariant_vars — loop-invariant variables
    pub invariant_vars: indexmap::IndexMap<OpRef, ()>,
}

impl DependencyGraph {
    /// dependency.py __init__: Build a dependency graph from loop operations.
    /// Uses DefTracker and IntegralForwardModification for precise analysis.
    pub fn build(ops: &[Op], constant_of: &dyn Fn(&Operand) -> Option<i64>) -> Self {
        let nodes: Vec<Node> = ops
            .iter()
            .enumerate()
            .map(|(idx, op)| Node::new(op.clone(), idx))
            .collect();

        let mut graph = DependencyGraph {
            nodes,
            memory_refs: indexmap::IndexMap::new(),
            index_vars: indexmap::IndexMap::new(),
            guards: Vec::new(),
            invariant_vars: indexmap::IndexMap::new(),
        };

        graph.build_dependencies(ops, constant_of);
        graph
    }

    /// dependency.py — append an imaginary node (`op=None`) and return its
    /// index so it can be walked into a `Path` alongside real node indices.
    pub fn add_imaginary_node(&mut self, label: impl Into<String>) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node::new_imaginary(label));
        index
    }

    /// dependency.py `Node.iterate_paths(to, backwards, path_max_len,
    /// blacklist)`. Enumerates every path from `from_idx` toward `to`
    /// (`None` = all maximal paths). Upstream is a generator on `Node`; the
    /// Rust graph is index-addressed, so this lives on `DependencyGraph` (it
    /// resolves each edge's target through `self.nodes`) and collects the
    /// yielded paths into a `Vec`. `backwards` walks `depends()` rather than
    /// `provides()`; `blacklist` records visited nodes so a path property
    /// need only be checked once per already-visited subtree.
    pub fn iterate_paths(
        &self,
        from_idx: usize,
        to: Option<usize>,
        backwards: bool,
        path_max_len: i64,
        blacklist: bool,
    ) -> Vec<Path> {
        let mut paths = Vec::new();
        if Some(from_idx) == to {
            return paths;
        }
        let mut blacklist_visit: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut path = Path::new(vec![from_idx]);
        // (edge index into the node's iter-direction list, node index, pathlen)
        let mut worklist: Vec<(usize, usize, i64)> = vec![(0, from_idx, 1)];
        while let Some((mut index, node_idx, mut pathlen)) = worklist.pop() {
            let iterdir = if backwards {
                self.nodes[node_idx].depends()
            } else {
                self.nodes[node_idx].provides()
            };
            let iterdir_len = iterdir.len();
            if index >= iterdir_len {
                // dependency.py: a leaf reached on its first visit is a
                // maximal path when no explicit destination was requested.
                if to.is_none() && index == 0 {
                    paths.push(path.clone_path());
                }
                if blacklist {
                    blacklist_visit.insert(node_idx);
                }
                continue;
            }
            let next_node = iterdir[index].target_node();
            index += 1;
            // dependency.py:330-334: keep exploring this node's remaining edges,
            // else mark it fully visited.
            if index < iterdir_len {
                worklist.push((index, node_idx, pathlen));
            } else {
                blacklist_visit.insert(node_idx);
            }
            path.cut_off_at(pathlen as usize);
            path.walk_node(next_node);
            // dependency.py:336-339: a blacklisted successor closes the path.
            if blacklist && blacklist_visit.contains(&next_node) {
                paths.push(path.clone_path());
                continue;
            }
            pathlen += 1;
            if Some(next_node) == to || (path_max_len > 0 && pathlen >= path_max_len) {
                paths.push(path.clone_path());
            } else {
                worklist.push((0, next_node, pathlen));
            }
        }
        paths
    }

    /// dependency.py `Node.edge_to(to, arg, failarg, label)` — add a
    /// dependency edge `from_idx → to_idx` (and its reversed back-edge). The
    /// Rust graph is index-addressed, so this is a graph-level method taking
    /// the two node indices; `add_edge` is the shared implementation and also
    /// keeps the `deps`/`users` side-vectors the scheduler reads consistent.
    pub fn edge_to(&mut self, from_idx: usize, to_idx: usize, arg: Option<OpRef>, failarg: bool) {
        Self::add_edge(&mut self.nodes, from_idx, to_idx, arg, failarg);
    }

    /// dependency.py `Node.remove_edge_to(node)` — delete the forward
    /// edge `from_idx → to_idx` from `from_idx.adjacent_list` and the matching
    /// reversed back-edge (whose `to` is `from_idx`) from
    /// `to_idx.adjacent_list_back`. The `deps`/`users` side-vectors, which
    /// `add_edge` maintains and the scheduler consumes, are pruned in step so
    /// the mutated graph reschedules faithfully.
    pub fn remove_edge_to(&mut self, from_idx: usize, to_idx: usize) {
        if let Some(pos) = self.nodes[from_idx]
            .adjacent_list
            .iter()
            .position(|dep| dep.to_idx == to_idx)
        {
            self.nodes[from_idx].adjacent_list.remove(pos);
        }
        if let Some(pos) = self.nodes[to_idx]
            .adjacent_list_back
            .iter()
            .position(|dep| dep.to_idx == from_idx)
        {
            self.nodes[to_idx].adjacent_list_back.remove(pos);
        }
        // Compat: mirror the deletion into deps/users (add_edge maintains them).
        if let Some(pos) = self.nodes[to_idx].deps.iter().position(|&d| d == from_idx) {
            self.nodes[to_idx].deps.remove(pos);
        }
        if let Some(pos) = self.nodes[from_idx].users.iter().position(|&u| u == to_idx) {
            self.nodes[from_idx].users.remove(pos);
        }
    }

    /// dependency.py: build_dependencies — construct def-use chains
    /// with DefTracker and IntegralForwardModification.
    fn build_dependencies(&mut self, ops: &[Op], constant_of: &dyn Fn(&Operand) -> Option<i64>) {
        let mut tracker = DefTracker::new(self);
        let mut intformod = IntegralForwardModification::new(constant_of);

        for i in 0..self.nodes.len() {
            if self.nodes[i].is_imaginary() {
                continue;
            }
            let op = self.nodes[i].op().clone();

            // dependency.py:613-616: set priority for pure/guard ops
            if op.opcode.is_always_pure() {
                self.nodes[i].setpriority(1);
            }
            if op.opcode.is_guard() {
                self.nodes[i].setpriority(2);
            }

            // dependency.py:620: inspect for index variables and memory refs
            intformod.inspect_operation(&op, i);
            if let Some(mref) = intformod.memory_refs.get(&i) {
                self.nodes[i].memory_ref = Some(mref.clone());
                self.memory_refs.insert(i, mref.clone());
            }

            // dependency.py: define result variable
            if op.opcode.result_type() != majit_ir::Type::Void {
                tracker.define(op.pos().get(), i);
            }

            // dependency.py:626-644: build edges based on op type
            if op.opcode.is_always_pure() || op.opcode.is_final() {
                // dependency.py:628-629: pure/final — depend on all args
                let args: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                for arg in &args {
                    Self::depends_on_arg_static(&tracker, *arg, i, &mut self.nodes);
                }
            } else if op.opcode.is_guard() {
                // dependency.py: guard dependencies
                if !self.nodes[i].exits_early() {
                    // dependency.py:635-640: guard ordering + non-pure deps
                    if !self.guards.is_empty() {
                        let last_guard = *self.guards.last().unwrap();
                        Self::add_edge(&mut self.nodes, last_guard, i, None, true);
                    }
                    for &np_idx in &tracker.non_pure.clone() {
                        Self::add_edge(&mut self.nodes, np_idx, i, None, true);
                    }
                    tracker.non_pure.clear();
                }
                self.guards.push(i);
                // dependency.py: build_guard_dependencies
                self.build_guard_dependencies(i, &mut tracker, ops);
            } else {
                // dependency.py:644: non-pure (memory side effects)
                self.build_non_pure_dependencies(i, &mut tracker, ops);
            }
        }

        // Copy index_vars from intformod
        self.index_vars = intformod.index_vars;
    }

    /// dependency.py: build_guard_dependencies
    fn build_guard_dependencies(
        &mut self,
        guard_idx: usize,
        tracker: &mut DefTracker,
        _ops: &[Op],
    ) {
        let op = self.nodes[guard_idx].op().clone();
        // dependency.py:710-712: ignore invalidated & future condition & early exit guards
        if matches!(
            op.opcode,
            OpCode::GuardFutureCondition | OpCode::GuardAlwaysFails | OpCode::GuardNotInvalidated
        ) {
            return;
        }
        // dependency.py: true dependencies on args
        for arg in op.getarglist().iter() {
            Self::depends_on_arg_static(tracker, arg.to_opref(), guard_idx, &mut self.nodes);
        }
        // dependency.py: guard_argument_protection
        self.guard_argument_protection(guard_idx, tracker);
        // dependency.py: descr.exits_early() check
        if self.nodes[guard_idx].exits_early() {
            return;
        }
        // dependency.py: fail_args dependencies — iterate ALL redefinitions
        if let Some(fail_args) = op.guard_fail_args() {
            let fa = fail_args.to_vec();
            for arg in &fa {
                if arg.is_none() {
                    continue;
                }
                if !tracker.is_defined(arg.to_opref()) {
                    continue;
                }
                // dependency.py: for at in tracker.redefinitions(arg)
                let redefs = tracker.redefinitions(arg.to_opref());
                for at_idx in redefs {
                    if self.nodes[at_idx].is_before(guard_idx) {
                        Self::add_edge(
                            &mut self.nodes,
                            at_idx,
                            guard_idx,
                            Some(arg.to_opref()),
                            true,
                        );
                    }
                }
            }
        }
    }

    /// dependency.py: guard_argument_protection
    fn guard_argument_protection(&mut self, guard_idx: usize, tracker: &mut DefTracker) {
        let op = self.nodes[guard_idx].op().clone();
        // dependency.py:657-664: redefine non-constant, non-int, non-float args (pointers)
        for arg in op.getarglist().iter() {
            if arg.is_constant() || arg.is_none() {
                continue;
            }
            // dependency.py:658: arg.type not in ('i','f')
            // Look up the defining op's result type to determine arg type.
            let arg_type = tracker
                .definition(arg.to_opref())
                .map(|def_idx| self.nodes[def_idx].op().opcode.result_type())
                .unwrap_or(majit_ir::Type::Ref); // unknown → assume ref (conservative)
            if arg_type != majit_ir::Type::Int && arg_type != majit_ir::Type::Float {
                tracker.define(arg.to_opref(), guard_idx);
            }
        }
        // dependency.py:665-698: special guard priorities
        match op.opcode {
            OpCode::GuardNotForced2 => {
                self.nodes[guard_idx].setpriority(-10);
            }
            OpCode::GuardOverflow | OpCode::GuardNoOverflow => {
                self.nodes[guard_idx].setpriority(100);
                // Find preceding overflow operation
                let mut j = guard_idx;
                while j > 0 {
                    j -= 1;
                    if self.nodes[j].op().opcode.is_ovf() {
                        Self::add_edge(&mut self.nodes, j, guard_idx, None, false);
                        break;
                    }
                }
            }
            OpCode::GuardNoException | OpCode::GuardException | OpCode::GuardNotForced => {
                self.nodes[guard_idx].setpriority(100);
                // Find preceding can-raise operation
                let mut j = guard_idx;
                while j > 0 {
                    j -= 1;
                    if self.nodes[j].op().opcode.can_raise() || self.nodes[j].op().opcode.is_guard()
                    {
                        Self::add_edge(&mut self.nodes, j, guard_idx, None, false);
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    /// dependency.py: build_non_pure_dependencies
    fn build_non_pure_dependencies(
        &mut self,
        node_idx: usize,
        tracker: &mut DefTracker,
        _ops: &[Op],
    ) {
        let op = self.nodes[node_idx].op().clone();

        if self.nodes[node_idx].loads_from_complex_object() {
            // dependency.py: LOAD_COMPLEX_OBJ dispatch
            // (opnum, complex_obj_arg_idx, index_arg_idx)
            let (cobj_idx, index_idx) = load_complex_obj_args(op.opcode);
            if cobj_idx < op.num_args() {
                let cobj = op.arg(cobj_idx).to_opref();
                if index_idx >= 0 && (index_idx as usize) < op.num_args() {
                    // dependency.py: argcell-aware depends_on
                    let index_var = op.arg(index_idx as usize).to_opref();
                    Self::depends_on_arg_static(tracker, cobj, node_idx, &mut self.nodes);
                    Self::depends_on_arg_static(tracker, index_var, node_idx, &mut self.nodes);
                } else {
                    // dependency.py:750: no index arg
                    Self::depends_on_arg_static(tracker, cobj, node_idx, &mut self.nodes);
                }
            }
        } else {
            // dependency.py: side_effect_arguments processing
            let nodes_ref = &self.nodes;
            let arg_type_of = |opref: OpRef| -> majit_ir::Type {
                // Look up the defining op's result type
                nodes_ref
                    .iter()
                    .filter_map(|n| n.getoperation())
                    .find(|op| op.pos().get() == opref)
                    .map(|op| op.opcode.result_type())
                    .unwrap_or(majit_ir::Type::Int)
            };
            let side_effects = side_effect_arguments(&op, &arg_type_of);
            for (arg, argcell, destroyed) in &side_effects {
                if let Some(cell) = argcell {
                    // dependency.py:754-757: exact cell tracking
                    Self::depends_on_arg_static(tracker, *arg, node_idx, &mut self.nodes);
                    Self::depends_on_arg_static(tracker, *cell, node_idx, &mut self.nodes);
                } else if *destroyed {
                    // dependency.py: WAR/WAW dependencies
                    if let Some(def_idx) = tracker.definition(*arg) {
                        // dependency.py:767-769: war edges from def's users
                        let provides: Vec<usize> = self.nodes[def_idx]
                            .adjacent_list
                            .iter()
                            .map(|d| d.to_idx)
                            .collect();
                        for to in provides {
                            if to != node_idx {
                                Self::add_edge(&mut self.nodes, to, node_idx, *argcell, false);
                            }
                        }
                        // dependency.py: def_node.edge_to(node)
                        Self::add_edge(&mut self.nodes, def_idx, node_idx, *argcell, false);
                    }
                } else {
                    // dependency.py:774-775: normal use
                    Self::depends_on_arg_static(tracker, *arg, node_idx, &mut self.nodes);
                }
                if *destroyed {
                    // dependency.py:776-777: redefine
                    tracker.define(*arg, node_idx);
                }
            }

            // dependency.py: non-pure must follow last guard
            if !self.guards.is_empty() {
                let last_guard = *self.guards.last().unwrap();
                Self::add_edge(&mut self.nodes, last_guard, node_idx, None, false);
            }
            // dependency.py:784: track as non-pure
            tracker.add_non_pure(node_idx);
        }
    }

    /// Helper: add a dependency edge between two nodes (dependency.py Node.edge_to).
    fn add_edge(
        nodes: &mut [Node],
        from_idx: usize,
        to_idx: usize,
        arg: Option<OpRef>,
        failarg: bool,
    ) {
        if from_idx == to_idx {
            return;
        }
        // Check if edge already exists
        let existing = nodes[from_idx]
            .adjacent_list
            .iter()
            .position(|d| d.to_idx == to_idx);
        if let Some(pos) = existing {
            // dependency.py:186-194: update existing edge
            if let Some(a) = arg
                && !nodes[from_idx].adjacent_list[pos].because_of(a)
            {
                nodes[from_idx].adjacent_list[pos].args.push((from_idx, a));
            }
            // dependency.py:190-191: a normal dependency overwriting a failarg
            // clears the flag. dependency.py:457-458 also propagates this to the
            // linked back-edge via `dep.backward`; pyre keeps paired edges by
            // index, so update both halves explicitly.
            if !(nodes[from_idx].adjacent_list[pos].failarg && failarg) {
                Self::set_edge_failarg(nodes, from_idx, to_idx, false);
            }
        } else {
            // dependency.py:176-180: create new edge + backward edge
            let dep = Dependency::new(from_idx, to_idx, arg, failarg);
            nodes[from_idx].adjacent_list.push(dep);
            let dep_back = Dependency::new(to_idx, from_idx, arg, failarg);
            nodes[to_idx].adjacent_list_back.push(dep_back);
            // Compat: update deps/users
            if !nodes[to_idx].deps.contains(&from_idx) {
                nodes[to_idx].deps.push(from_idx);
                nodes[from_idx].users.push(to_idx);
            }
        }
    }

    fn set_edge_failarg(nodes: &mut [Node], from_idx: usize, to_idx: usize, failarg: bool) {
        if let Some(dep) = nodes[from_idx]
            .adjacent_list
            .iter_mut()
            .find(|dep| dep.to_idx == to_idx)
        {
            dep.failarg = failarg;
        }
        if let Some(dep) = nodes[to_idx]
            .adjacent_list_back
            .iter_mut()
            .find(|dep| dep.to_idx == from_idx)
        {
            dep.failarg = failarg;
        }
    }

    /// Helper: depends_on_arg using DefTracker (works with &mut nodes borrow).
    fn depends_on_arg_static(tracker: &DefTracker, arg: OpRef, to_idx: usize, nodes: &mut [Node]) {
        if let Some(at_idx) = tracker.definition(arg)
            && at_idx != to_idx
        {
            // Inline add_edge logic to avoid double borrow issues
            let existing = nodes[at_idx]
                .adjacent_list
                .iter()
                .position(|d| d.to_idx == to_idx);
            if let Some(pos) = existing {
                if !nodes[at_idx].adjacent_list[pos].because_of(arg) {
                    nodes[at_idx].adjacent_list[pos].args.push((at_idx, arg));
                }
                Self::set_edge_failarg(nodes, at_idx, to_idx, false);
            } else {
                let dep = Dependency::new(at_idx, to_idx, Some(arg), false);
                nodes[at_idx].adjacent_list.push(dep);
                let dep_back = Dependency::new(to_idx, at_idx, Some(arg), false);
                nodes[to_idx].adjacent_list_back.push(dep_back);
                if !nodes[to_idx].deps.contains(&at_idx) {
                    nodes[to_idx].deps.push(at_idx);
                    nodes[at_idx].users.push(to_idx);
                }
            }
        }
    }

    /// Find groups of independent, isomorphic operations that can be packed.
    ///
    /// Two ops are "isomorphic" if they have the same opcode and their
    /// args come from independent sources (no data dependency between them).
    pub fn find_packable_groups(&self) -> Vec<Pack> {
        let mut groups: Vec<Pack> = Vec::new();
        let mut used: IndexSet<usize> = IndexSet::new();

        // Group by opcode
        let mut by_opcode: indexmap::IndexMap<OpCode, Vec<usize>> = indexmap::IndexMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let Some(op) = node.getoperation() else {
                continue;
            };
            if op.opcode.to_vector().is_some() && !op.opcode.is_guard() {
                by_opcode.entry(op.opcode).or_default().push(i);
            }
        }

        // For each opcode, find independent pairs/groups
        for (opcode, indices) in by_opcode.iter() {
            let vec_opcode = match opcode.to_vector() {
                Some(v) => v,
                None => continue,
            };

            let mut group_indices = Vec::new();

            for &i in indices {
                if used.contains(&i) {
                    continue;
                }

                // Check independence from already-grouped ops
                let mut independent = true;
                for &already in &group_indices {
                    if self.has_dependency(i, already) {
                        independent = false;
                        break;
                    }
                }

                // vector.py: isomorphic check — ops must have the same
                // number of args and compatible types (same opcode already
                // guaranteed by the grouping).
                if independent && !group_indices.is_empty() {
                    let first = self.nodes[group_indices[0]].op();
                    let candidate = self.nodes[i].op();
                    if first.num_args() != candidate.num_args() {
                        independent = false;
                    }
                }

                if independent {
                    group_indices.push(i);
                }
            }

            // Need at least 2 ops to form a pack
            if group_indices.len() >= 2 {
                for &idx in &group_indices {
                    used.insert(idx);
                }
                groups.push(Pack {
                    scalar_opcode: *opcode,
                    vector_opcode: vec_opcode,
                    members: group_indices,
                    is_accumulating: false,
                    position: -1,
                    operator: None,
                });
            }
        }

        groups
    }

    /// Check if there's a direct or transitive dependency between two nodes.
    pub fn has_dependency(&self, a: usize, b: usize) -> bool {
        // Direct dependency check (sufficient for basic vectorization)
        self.nodes[a].deps.contains(&b) || self.nodes[b].deps.contains(&a)
    }
}

// ── Instruction Scheduling ──────────────────────────────────────────────

/// Reorder operations to maximize instruction-level parallelism.
///
/// Uses a topological sort with priority scheduling: among all operations
/// whose dependencies are satisfied, choose the one with the highest
/// "height" (longest path to a leaf in the dependency graph).
///
/// This mirrors RPython's `schedule.py`, which reorders the loop body to
/// improve ILP before packing decisions are made.
pub(crate) fn schedule_operations(graph: &DependencyGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute heights in reverse topological order.
    // Height = 1 + max height among users (successors in the DAG).
    let mut heights = vec![0usize; n];
    for i in (0..n).rev() {
        let max_user_height = graph.nodes[i]
            .users
            .iter()
            .map(|&u| heights[u])
            .max()
            .unwrap_or(0);
        heights[i] = 1 + max_user_height;
    }

    // Compute in-degrees from deps. `deps`/`users` are keyed by node position,
    // so `in_degree` is indexed by position too — an imaginary node's `idx`
    // field is a synthetic sentinel (dependency.py ImaginaryNode), not its position.
    let mut in_degree = vec![0usize; n];
    for (i, node) in graph.nodes.iter().enumerate() {
        in_degree[i] = node.deps.len();
    }

    // Seed the priority queue with all zero-in-degree nodes.
    // BinaryHeap is a max-heap: (height, index) — higher height = higher priority.
    let mut ready: BinaryHeap<(usize, usize)> = BinaryHeap::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            ready.push((heights[i], i));
        }
    }

    let mut schedule = Vec::with_capacity(n);
    while let Some((_, idx)) = ready.pop() {
        // Imaginary nodes (op=None) impose ordering constraints — e.g. the
        // "early exit" node from analyse_index_calculations — but map to no
        // operation, so they are traversed for in-degree yet never emitted.
        if !graph.nodes[idx].is_imaginary() {
            schedule.push(idx);
        }
        for &user in &graph.nodes[idx].users {
            in_degree[user] -= 1;
            if in_degree[user] == 0 {
                ready.push((heights[user], user));
            }
        }
    }

    schedule
}

// ── dependency.py: IndexVar ──────────────────────────

/// dependency.py IndexVar: Linear combination of an index variable.
/// Represents `var * (coefficient_mul / coefficient_div) + constant`.
#[derive(Clone, Debug)]
pub struct IndexVar {
    /// The base SSA variable.
    pub var: OpRef,
    /// The BOUND operand for `var`, captured from the real op arg at build time
    /// so `get_operations` can carry `var` as `Operand::Op`/`InputArg` instead
    /// of a position-only box. RPython's IndexVar holds the box object
    /// (`dependency.py self.var`); pyre's flat-OpRef `var` loses the
    /// producer, so the operand is captured alongside. `None` when no bound
    /// operand was available at construction (e.g. a synthetic-result var) —
    /// then `get_operations` binds a synthetic producer via `bound_from_opref`.
    /// Its `to_opref()` always equals `var`.
    pub var_box: Option<majit_ir::operand::Operand>,
    /// If `var` is a ConstInt OpRef, the resolved integer value.
    /// dependency.py:1117-1118: isinstance(svar, ConstInt) comparison.
    pub var_const: Option<i64>,
    /// Multiplicative coefficient (numerator).
    pub coefficient_mul: i64,
    /// Divisive coefficient (denominator).
    pub coefficient_div: i64,
    /// Additive constant.
    pub constant: i64,
}

impl IndexVar {
    pub fn new(var: OpRef) -> Self {
        IndexVar {
            var,
            var_box: None,
            var_const: None,
            coefficient_mul: 1,
            coefficient_div: 1,
            constant: 0,
        }
    }

    /// Like [`IndexVar::new`] but capturing the BOUND operand for `var`
    /// (`var_box.to_opref() == var`), used so `get_operations` carries a bound
    /// operand rather than a position-only one.
    pub fn new_boxed(var_box: majit_ir::operand::Operand) -> Self {
        let var = var_box.to_opref();
        IndexVar {
            var,
            var_box: Some(var_box),
            var_const: None,
            coefficient_mul: 1,
            coefficient_div: 1,
            constant: 0,
        }
    }

    /// Create an IndexVar for a constant variable.
    pub fn new_const(var: OpRef, value: i64) -> Self {
        IndexVar {
            var,
            var_box: None,
            var_const: Some(value),
            coefficient_mul: 1,
            coefficient_div: 1,
            constant: 0,
        }
    }

    /// dependency.py same_variable
    pub fn same_variable(&self, other: &IndexVar) -> bool {
        self.var == other.var
    }

    /// dependency.py same_mulfactor
    pub fn same_mulfactor(&self, other: &IndexVar) -> bool {
        if self.coefficient_mul == other.coefficient_mul
            && self.coefficient_div == other.coefficient_div
        {
            return true;
        }
        let selfmod = self.coefficient_mul % self.coefficient_div;
        let othermod = other.coefficient_mul % other.coefficient_div;
        if selfmod == 0 && othermod == 0 {
            let selfdiv = self.coefficient_mul / self.coefficient_div;
            let otherdiv = other.coefficient_mul / other.coefficient_div;
            return selfdiv == otherdiv;
        }
        false
    }

    /// dependency.py constant_diff
    pub fn constant_diff(&self, other: &IndexVar) -> i64 {
        self.constant - other.constant
    }

    /// dependency.py is_identity
    pub fn is_identity(&self) -> bool {
        self.coefficient_mul == 1 && self.coefficient_div == 1 && self.constant == 0
    }

    /// dependency.py: compare(other)
    ///
    /// Returns `(valid, ordering)` where `ordering` is the signed constant
    /// difference between self and other when the linear coefficients match.
    /// Returns `(false, 0)` if the two IndexVars are not comparable.
    pub fn compare(&self, other: &IndexVar) -> (bool, i64) {
        if !self.same_mulfactor(other) {
            return (false, 0);
        }
        let c = self.constant - other.constant;
        // dependency.py:1117-1118: both ConstInt → always comparable.
        // RPython returns (True, svar.getint() - ovar.getint()) without c.
        if let (Some(sv), Some(ov)) = (self.var_const, other.var_const) {
            return (true, sv - ov);
        }
        if self.var == other.var {
            return (true, c);
        }
        (false, 0)
    }

    /// dependency.py: getvariable()
    pub fn getvariable(&self) -> OpRef {
        self.var
    }

    /// dependency.py clone
    pub fn clone_var(&self) -> Self {
        IndexVar {
            var: self.var,
            var_box: self.var_box.clone(),
            var_const: self.var_const,
            coefficient_mul: self.coefficient_mul,
            coefficient_div: self.coefficient_div,
            constant: self.constant,
        }
    }

    /// dependency.py: get_operations()
    ///
    /// Materialize the linear combination as IR operations:
    ///   var * coefficient_mul / coefficient_div + constant
    ///
    /// `next_const`: callback to allocate a constant OpRef for a given i64 value.
    /// In RPython this is `ConstInt(value)` — an inline constant box.
    /// In majit, constants need explicit OpRef allocation.
    ///
    /// Box-carrying note: the `var` operand of the FIRST emitted op is
    /// `self.var` — the IndexVar's base SSA variable (the loop's index inputarg
    /// or another loop-body producer). RPython carries `self.var` as the live
    /// index-var box object; pyre's flat-OpRef `var` lost it, so the bound
    /// operand is captured at build time into `self.var_box` (see
    /// `get_or_create`) and
    /// re-installed here, carrying `Operand::Op`/`InputArg`. The first-var arm
    /// binds a synthetic producer via `bound_from_opref` only when no operand
    /// was captured (a synthetic-result var). The CHAINED `var = op.pos().get()`
    /// references (when `coefficient_mul != 1`, i.e. an `IntAdd` / `IntSub`
    /// consuming the prior `IntMul`) point at the just-created local `Op`
    /// value — this fn returns `Vec<Op>`, not `Vec<OpRc>`, so there is no
    /// producer `Rc` to bind to; the chained `var` binds a synthetic producer
    /// carrying the same `pos` (`bound_from_opref`, `to_opref()`-identical).
    /// `to_opref()` is preserved in every case; the constant arg `c` always
    /// sheds to `Operand::Const`.
    pub fn get_operations(&self, mut next_const: impl FnMut(i64) -> OpRef) -> Vec<majit_ir::Op> {
        use majit_ir::{Op, OpCode};
        // First-var box: the captured bound box when its `to_opref()` still
        // matches `self.var` (it always does — `var`/`var_box` move together),
        // else a position-only box for a box-less (synthetic) var.
        let first_var = match &self.var_box {
            Some(b) if b.to_opref() == self.var => b.clone(),
            _ => majit_ir::operand::Operand::bound_from_opref(self.var),
        };
        let mut var = self.var;
        let mut first = true;
        let mut tolist = Vec::new();
        // Carry `var` bound for the FIRST emitted op (from `first_var`); the
        // chained references thereafter point at local `Op` values (no `Rc`),
        // so they bind a synthetic producer carrying the same `pos`.
        let var_box = |var: OpRef, first: &mut bool| -> majit_ir::operand::Operand {
            if *first {
                *first = false;
                first_var.clone()
            } else {
                majit_ir::operand::Operand::bound_from_opref(var)
            }
        };
        if self.coefficient_mul != 1 {
            // dependency.py:1069: args = [var, ConstInt(self.coefficient_mul)]
            let c = next_const(self.coefficient_mul);
            let op = Op::new(
                OpCode::IntMul,
                &[
                    var_box(var, &mut first),
                    majit_ir::operand::Operand::from_opref(c),
                ],
            );
            var = op.pos().get();
            tolist.push(op);
        }
        // dependency.py:1072-1074: coefficient_div != 1 → assert 0
        assert!(
            self.coefficient_div == 1,
            "IndexVar.get_operations: coefficient_div != 1 not supported"
        );
        if self.constant > 0 {
            // dependency.py:1076: args = [var, ConstInt(self.constant)]
            let c = next_const(self.constant);
            let op = Op::new(
                OpCode::IntAdd,
                &[
                    var_box(var, &mut first),
                    majit_ir::operand::Operand::from_opref(c),
                ],
            );
            var = op.pos().get();
            tolist.push(op);
        }
        if self.constant < 0 {
            // dependency.py:1080-1081: var = ResOperation(INT_SUB, [var, ConstInt(-self.constant)])
            let c = next_const(-self.constant);
            let op = Op::new(
                OpCode::IntSub,
                &[
                    var_box(var, &mut first),
                    majit_ir::operand::Operand::from_opref(c),
                ],
            );
            #[allow(unused_assignments)]
            {
                var = op.pos().get();
            }
            tolist.push(op);
        }
        tolist
    }

    /// dependency.py: emit_operations(opt, result_box)
    ///
    /// Emit the linear operations into the output list.
    /// Returns the result OpRef (last emitted op, or var if identity).
    ///
    /// `next_const`: callback to allocate a constant OpRef.
    pub fn emit_operations(
        &self,
        new_ops: &mut Vec<majit_ir::Op>,
        next_const: impl FnMut(i64) -> OpRef,
    ) -> OpRef {
        if self.is_identity() {
            return self.var;
        }
        let ops = self.get_operations(next_const);
        let mut last = self.var;
        for op in ops {
            last = op.pos().get();
            new_ops.push(op);
        }
        last
    }
}

// ── dependency.py: MemoryRef ────────────────────────

/// dependency.py MemoryRef: A memory reference to an array object.
/// Tracks the array pointer, descriptor, and index variable (linear
/// combination) for adjacent-memory analysis.
#[derive(Clone, Debug)]
pub struct MemoryRef {
    /// The array pointer (op.getarg(0))
    pub array: OpRef,
    /// The array descriptor
    pub descr: majit_ir::DescrRef,
    /// The index as a linear combination
    pub index_var: IndexVar,
    /// Whether this is a raw (byte-level) access
    pub raw_access: bool,
}

impl MemoryRef {
    pub fn new(array: OpRef, descr: majit_ir::DescrRef, index_var: IndexVar) -> Self {
        MemoryRef {
            array,
            descr,
            index_var,
            raw_access: false,
        }
    }

    /// dependency.py is_adjacent_to: symmetric adjacency check
    pub fn is_adjacent_to(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return false;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return false;
        }
        let stride = self.stride();
        self.index_var
            .constant_diff(&other.index_var)
            .abs()
            .saturating_sub(stride)
            == 0
    }

    /// dependency.py is_adjacent_after: asymmetric adjacency (self is after other)
    pub fn is_adjacent_after(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return false;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return false;
        }
        let stride = self.stride();
        other.index_var.constant_diff(&self.index_var) == stride
    }

    /// dependency.py: alias check
    pub fn alias(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return true;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return true;
        }
        self.index_var.constant_diff(&other.index_var).abs() < self.stride()
    }

    /// dependency.py: same_array — array identity + descriptor equality.
    /// RPython uses `self.descr == other.descr` (value equality).
    /// In majit, Descr is a trait object; we compare by index() for value equality,
    /// falling back to Arc::ptr_eq for descriptors without assigned indices.
    pub fn same_array(&self, other: &MemoryRef) -> bool {
        if self.array != other.array {
            return false;
        }
        let si = self.descr.index();
        let oi = other.descr.index();
        if si != u32::MAX && oi != u32::MAX {
            si == oi
        } else {
            std::sync::Arc::ptr_eq(&self.descr, &other.descr)
        }
    }

    /// dependency.py: stride in elements (1) or bytes (for raw)
    pub fn stride(&self) -> i64 {
        if !self.raw_access {
            1
        } else {
            self.descr
                .as_array_descr()
                .map(|ad| ad.item_size() as i64)
                .unwrap_or(std::mem::size_of::<usize>() as i64)
        }
    }
}

// ── dependency.py: Dependency (rich edge) ─────────────

/// dependency.py Dependency: A dependency edge in the graph.
/// Carries which args caused the dependency and whether it's a failarg dep.
#[derive(Clone, Debug)]
pub struct Dependency {
    /// Index of the source node.
    pub at_idx: usize,
    /// Index of the target node.
    pub to_idx: usize,
    /// (source_node_idx, arg OpRef) pairs that caused this dependency.
    pub args: Vec<(usize, OpRef)>,
    /// Whether this is a failarg dependency.
    pub failarg: bool,
}

impl Dependency {
    /// dependency.py `Dependency.__init__(at, to, arg, failarg=False)`.
    pub fn new(at_idx: usize, to_idx: usize, arg: Option<OpRef>, failarg: bool) -> Self {
        let mut d = Dependency {
            at_idx,
            to_idx,
            args: Vec::new(),
            failarg,
        };
        if let Some(a) = arg {
            d.args.push((at_idx, a));
        }
        d
    }

    /// dependency.py: because_of
    pub fn because_of(&self, var: OpRef) -> bool {
        self.args.iter().any(|(_, a)| *a == var)
    }

    /// dependency.py `target_node` — the `to` endpoint index. For a
    /// forward (`provides`) edge this is the successor; for a reversed
    /// `depends` back-edge it is the predecessor.
    pub fn target_node(&self) -> usize {
        self.to_idx
    }

    /// dependency.py `origin_node` — the `at` endpoint index.
    pub fn origin_node(&self) -> usize {
        self.at_idx
    }

    /// dependency.py `is_failarg`.
    pub fn is_failarg(&self) -> bool {
        self.failarg
    }
}

// ── dependency.py: DefTracker ─────────────────────────

/// dependency.py DefTracker: Tracks definitions of OpRefs during
/// dependency graph construction. Maps each OpRef to the node(s)
/// that define it, enabling def-use chain queries.
pub struct DefTracker {
    /// OpRef → list of (defining node index, optional memory ref cell)
    pub defs: indexmap::IndexMap<OpRef, Vec<(usize, Option<usize>)>>,
    /// Nodes with side effects (non-pure).
    pub non_pure: Vec<usize>,
}

impl DefTracker {
    pub fn new(_graph: &DependencyGraph) -> Self {
        DefTracker {
            defs: indexmap::IndexMap::new(),
            non_pure: Vec::new(),
        }
    }

    /// dependency.py add_non_pure
    pub fn add_non_pure(&mut self, node_idx: usize) {
        self.non_pure.push(node_idx);
    }

    /// dependency.py: define — register that node_idx defines arg.
    pub fn define(&mut self, arg: OpRef, node_idx: usize) {
        // dependency.py:483-484: skip constants.
        if arg.is_constant() {
            return;
        }
        self.defs.entry(arg).or_default().push((node_idx, None));
    }

    /// dependency.py: redefinitions — yield all nodes defining arg.
    pub fn redefinitions(&self, arg: OpRef) -> Vec<usize> {
        self.defs
            .get(&arg)
            .map(|chain| chain.iter().map(|(idx, _)| *idx).collect())
            .unwrap_or_default()
    }

    /// dependency.py is_defined
    pub fn is_defined(&self, arg: OpRef) -> bool {
        self.defs.contains_key(&arg)
    }

    /// dependency.py: definition — find the defining node for arg.
    pub fn definition(&self, arg: OpRef) -> Option<usize> {
        if arg.is_constant() {
            return None;
        }
        let chain = self.defs.get(&arg)?;
        if chain.is_empty() {
            return None;
        }
        Some(chain.last()?.0)
    }

    /// dependency.py: depends_on_arg — add edge from definition to `to_idx`.
    pub fn depends_on_arg(&self, arg: OpRef, to_idx: usize, graph: &mut [Vec<usize>]) {
        if let Some(at_idx) = self.definition(arg)
            && at_idx != to_idx
            && !graph[at_idx].contains(&to_idx)
        {
            graph[at_idx].push(to_idx);
        }
    }
}

// ── dependency.py: IntegralForwardModification ────────

/// dependency.py IntegralForwardModification: Calculates integral modifications on integer
/// boxes. Propagates INT_ADD/INT_SUB/INT_MUL through IndexVar linear
/// combinations, and recognizes array access patterns for MemoryRef.
pub struct IntegralForwardModification<'a> {
    /// OpRef → IndexVar mapping
    pub index_vars: indexmap::IndexMap<OpRef, IndexVar>,
    /// Node index → MemoryRef mapping
    pub memory_refs: indexmap::IndexMap<usize, MemoryRef>,
    /// Callback to resolve constant Operand → i64 value.
    /// dependency.py IntegralForwardModification.is_const_integral + box.getint()
    constant_of: &'a dyn Fn(&Operand) -> Option<i64>,
}

impl<'a> IntegralForwardModification<'a> {
    pub fn new(constant_of: &'a dyn Fn(&Operand) -> Option<i64>) -> Self {
        IntegralForwardModification {
            index_vars: indexmap::IndexMap::new(),
            memory_refs: indexmap::IndexMap::new(),
            constant_of,
        }
    }

    fn set_index_var(&mut self, key: OpRef, idx: IndexVar) {
        self.index_vars.insert(key, idx);
    }

    fn set_memory_ref(&mut self, node_idx: usize, mref: MemoryRef) {
        self.memory_refs.insert(node_idx, mref);
    }

    fn is_const(arg: &Operand) -> bool {
        arg.is_constant()
    }

    fn const_val(&self, arg: &Operand) -> Option<i64> {
        (self.constant_of)(arg)
    }

    /// `arg_box` is the BOUND operand for `arg` (`arg_box.to_opref() == arg`),
    /// used to seed `IndexVar::var_box` so a freshly-created (un-tracked) index
    /// var can carry `var` as a bound operand in `get_operations`. A tracked var
    /// already in `index_vars` keeps its own (possibly box-carrying) entry.
    fn get_or_create(&mut self, arg: OpRef, arg_box: &Operand) -> IndexVar {
        self.index_vars.get(&arg).cloned().unwrap_or_else(|| {
            if Self::is_const(arg_box) {
                let val = self.const_val(arg_box).unwrap_or(0);
                IndexVar::new_const(arg, val)
            } else {
                IndexVar::new_boxed(arg_box.clone())
            }
        })
    }

    /// dependency.py operation_INT_ADD / operation_INT_SUB.
    fn inspect_additive(&mut self, op: &Op, is_sub: bool) {
        let result = op.pos().get();
        let b0 = op.arg(0);
        let b1 = op.arg(1);
        let a0 = b0.to_opref();
        let a1 = b1.to_opref();
        if Self::is_const(&b0) && Self::is_const(&b1) {
            let mut idx = IndexVar::new(result);
            let v0 = self.const_val(&b0).unwrap_or(0);
            let v1 = self.const_val(&b1).unwrap_or(0);
            idx.constant = if is_sub { v0 - v1 } else { v0 + v1 };
            self.set_index_var(result, idx);
        } else if Self::is_const(&b0) {
            let mut idx = self.get_or_create(a1, &b1);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(&b0) {
                if is_sub {
                    idx.constant -= v;
                } else {
                    idx.constant += v;
                }
            }
            self.set_index_var(result, idx);
        } else if Self::is_const(&b1) {
            let mut idx = self.get_or_create(a0, &b0);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(&b1) {
                if is_sub {
                    idx.constant -= v;
                } else {
                    idx.constant += v;
                }
            }
            self.set_index_var(result, idx);
        }
        // No var/var branch: `additive_func_source` only handles const/const,
        // const/var, and var/const (`dependency.py operation_INT_ADD`); a
        // non-const ± non-const result is intentionally left untracked in
        // `index_vars`.
    }

    /// dependency.py operation_INT_MUL.
    fn inspect_multiplicative(&mut self, op: &Op) {
        let result = op.pos().get();
        let b0 = op.arg(0);
        let b1 = op.arg(1);
        let a0 = b0.to_opref();
        let a1 = b1.to_opref();
        if Self::is_const(&b0) && Self::is_const(&b1) {
            let mut idx = IndexVar::new(result);
            let v0 = self.const_val(&b0).unwrap_or(0);
            let v1 = self.const_val(&b1).unwrap_or(0);
            idx.constant = v0 * v1;
            self.set_index_var(result, idx);
        } else if Self::is_const(&b0) {
            let mut idx = self.get_or_create(a1, &b1);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(&b0) {
                idx.coefficient_mul *= v;
                idx.constant *= v;
            }
            self.set_index_var(result, idx);
        } else if Self::is_const(&b1) {
            let mut idx = self.get_or_create(a0, &b0);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(&b1) {
                idx.coefficient_mul *= v;
                idx.constant *= v;
            }
            self.set_index_var(result, idx);
        }
    }

    /// dependency.py:950-975: inspect array access ops.
    /// Only creates MemoryRef for primitive array accesses (dependency.py).
    fn inspect_array_access(&mut self, op: &Op, node_idx: usize, raw_access: bool) {
        if op.num_args() < 2 {
            return;
        }
        let array = op.arg(0).to_opref();
        let index_box = op.arg(1);
        let index = index_box.to_opref();
        let idx_var = self.get_or_create(index, &index_box);
        if let Some(descr) = op.getdescr() {
            // dependency.py:954: descr.is_array_of_primitives()
            let is_prim = descr
                .as_array_descr()
                .map(|ad| ad.is_array_of_primitives())
                .unwrap_or(false);
            if !is_prim {
                return;
            }
            let mref = MemoryRef {
                array,
                descr,
                index_var: idx_var,
                raw_access,
            };
            self.set_memory_ref(node_idx, mref);
        }
    }

    /// dependency.py: inspect_operation dispatcher (integral_dispatch_opt)
    pub fn inspect_operation(&mut self, op: &Op, node_idx: usize) {
        match op.opcode {
            OpCode::IntAdd => self.inspect_additive(op, false),
            OpCode::IntSub => self.inspect_additive(op, true),
            OpCode::IntMul => self.inspect_multiplicative(op),
            // Array access ops
            OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore => {
                self.inspect_array_access(op, node_idx, true);
            }
            OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawF
            | OpCode::SetarrayitemRaw
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcF
            | OpCode::SetarrayitemGc => {
                self.inspect_array_access(op, node_idx, false);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::value::Const;

    fn int_operand(index: u32) -> Operand {
        Operand::const_(Const::Int(index.into()))
    }

    #[test]
    fn path_accessors_and_mutators_follow_dependency_py_shape() {
        let mut path = Path::new(vec![0, 1, 2]);

        assert_eq!(path.first(), Some(0));
        assert_eq!(path.second(), Some(1));
        assert_eq!(path.last_but_one(), Some(1));
        assert_eq!(path.last(), Some(2));
        assert!(path.check_acyclic());

        path.walk_node(3);
        assert_eq!(path.last(), Some(3));
        path.cut_off_at(2);
        assert_eq!(path.last(), Some(1));
        assert_eq!(path.clone_path(), path);
        assert_eq!(path.as_str(&[]), "Node(0) -> Node(1)");
    }

    #[test]
    fn path_purity_skips_imaginary_nodes_and_updates_priority() {
        let pure_op = Op::new(OpCode::IntAdd, &[int_operand(0), int_operand(1)]);
        let impure_op = Op::new(OpCode::SetfieldGc, &[int_operand(2), int_operand(3)]);
        let mut nodes = vec![Node::new(pure_op, 0), Node::new(impure_op, 1)];
        let imaginary_idx = nodes.len();
        nodes.push(Node::new_imaginary("synthetic"));

        let path = Path::new(vec![0, imaginary_idx]);

        assert!(nodes[imaginary_idx].is_imaginary());
        assert_eq!(nodes[imaginary_idx].dotlabel.as_deref(), Some("synthetic"));
        assert_eq!(path.as_str(&nodes), "Node(0) -> synthetic");
        assert!(path.is_always_pure(&nodes, false, false));
        path.set_schedule_priority(&mut nodes, 5);
        assert_eq!(nodes[imaginary_idx].priority, 5);

        let impure_path = Path::new(vec![0, 1]);
        assert!(!impure_path.is_always_pure(&nodes, false, false));
        assert!(impure_path.is_always_pure(&nodes, false, true));

        impure_path.set_schedule_priority(&mut nodes, 7);
        assert_eq!(nodes[0].priority, 7);
        assert_eq!(nodes[1].priority, 7);
        assert!(!nodes[0].is_imaginary());
    }

    fn imaginary_graph(n: usize) -> DependencyGraph {
        DependencyGraph {
            nodes: (0..n).map(|_| Node::new_imaginary("t")).collect(),
            memory_refs: indexmap::IndexMap::new(),
            index_vars: indexmap::IndexMap::new(),
            guards: Vec::new(),
            invariant_vars: indexmap::IndexMap::new(),
        }
    }

    /// Mirror `add_edge` (dependency.py:176-180): a forward edge plus its
    /// reversed back-edge, so `depends()` sees the predecessor.
    fn add_test_edge(g: &mut DependencyGraph, from: usize, to: usize) {
        g.nodes[from]
            .adjacent_list
            .push(Dependency::new(from, to, None, false));
        g.nodes[to]
            .adjacent_list_back
            .push(Dependency::new(to, from, None, false));
    }

    #[test]
    fn iterate_paths_enumerates_forward_backward_and_respects_max_len() {
        // Diamond: 0 -> 1 -> 3 and 0 -> 2 -> 3.
        let mut g = imaginary_graph(4);
        add_test_edge(&mut g, 0, 1);
        add_test_edge(&mut g, 0, 2);
        add_test_edge(&mut g, 1, 3);
        add_test_edge(&mut g, 2, 3);

        let fwd = g.iterate_paths(0, Some(3), false, -1, false);
        assert_eq!(fwd.len(), 2);
        assert!(fwd.contains(&Path::new(vec![0, 1, 3])));
        assert!(fwd.contains(&Path::new(vec![0, 2, 3])));

        // to=None yields the maximal root->leaf paths.
        let maximal = g.iterate_paths(0, None, false, -1, false);
        assert_eq!(maximal.len(), 2);
        assert!(maximal.contains(&Path::new(vec![0, 1, 3])));
        assert!(maximal.contains(&Path::new(vec![0, 2, 3])));

        // backwards walks depends(): predecessor chain 3 -> {1,2} -> 0.
        let back = g.iterate_paths(3, Some(0), true, -1, false);
        assert_eq!(back.len(), 2);
        assert!(back.contains(&Path::new(vec![3, 1, 0])));
        assert!(back.contains(&Path::new(vec![3, 2, 0])));

        // path_max_len caps enumeration to length-2 prefixes.
        let capped = g.iterate_paths(0, Some(3), false, 2, false);
        assert_eq!(capped.len(), 2);
        assert!(capped.contains(&Path::new(vec![0, 1])));
        assert!(capped.contains(&Path::new(vec![0, 2])));

        // self == to yields nothing (dependency.py:317-318).
        assert!(g.iterate_paths(1, Some(1), false, -1, false).is_empty());

        // Edge accessors expose endpoints / failarg flag.
        assert_eq!(g.nodes[0].provides().len(), 2);
        assert_eq!(g.nodes[3].depends().len(), 2);
        assert_eq!(g.nodes[0].provides()[0].target_node(), 1);
        assert_eq!(g.nodes[0].provides()[0].origin_node(), 0);
        assert!(!g.nodes[0].provides()[0].is_failarg());
    }

    fn real_graph(n: usize) -> DependencyGraph {
        let op = || Op::new(OpCode::IntAdd, &[int_operand(0), int_operand(1)]);
        DependencyGraph {
            nodes: (0..n).map(|i| Node::new(op(), i)).collect(),
            memory_refs: indexmap::IndexMap::new(),
            index_vars: indexmap::IndexMap::new(),
            guards: Vec::new(),
            invariant_vars: indexmap::IndexMap::new(),
        }
    }

    #[test]
    fn existing_edge_failarg_downgrade_updates_forward_and_back_edges() {
        let mut g = real_graph(2);

        g.edge_to(0, 1, None, true);
        assert!(g.nodes[0].provides()[0].is_failarg());
        assert!(g.nodes[1].depends()[0].is_failarg());

        g.edge_to(0, 1, None, false);
        assert!(!g.nodes[0].provides()[0].is_failarg());
        assert!(!g.nodes[1].depends()[0].is_failarg());
    }

    #[test]
    fn depends_on_arg_downgrades_existing_failarg_edge_pair() {
        let mut g = real_graph(2);
        let arg = OpRef::int_op(0);
        let mut tracker = DefTracker::new(&g);
        tracker.define(arg, 0);

        DependencyGraph::add_edge(&mut g.nodes, 0, 1, None, true);
        assert!(g.nodes[0].provides()[0].is_failarg());
        assert!(g.nodes[1].depends()[0].is_failarg());

        DependencyGraph::depends_on_arg_static(&tracker, arg, 1, &mut g.nodes);
        assert!(!g.nodes[0].provides()[0].is_failarg());
        assert!(!g.nodes[1].depends()[0].is_failarg());
        assert!(g.nodes[0].provides()[0].because_of(arg));
    }

    #[test]
    fn edge_to_and_remove_edge_to_keep_adjacency_and_compat_vectors_in_sync() {
        // Three real ops; 0 -> 2 and 1 -> 2.
        let mut g = real_graph(3);
        g.edge_to(0, 2, None, false);
        g.edge_to(1, 2, None, false);

        // edge_to builds forward + reversed back-edge + deps/users.
        assert_eq!(g.nodes[0].provides()[0].target_node(), 2);
        assert_eq!(g.nodes[2].depends().len(), 2);
        assert!(g.nodes[2].deps.contains(&0) && g.nodes[2].deps.contains(&1));
        assert!(g.nodes[0].users.contains(&2) && g.nodes[1].users.contains(&2));

        // Insert an "early exit" imaginary node between guard 0 and its successor 2.
        let earlyexit = g.add_imaginary_node("early exit");
        g.edge_to(0, earlyexit, None, false);
        g.edge_to(earlyexit, 2, None, true);
        g.remove_edge_to(0, 2);

        // remove_edge_to prunes forward, back, and both compat vectors.
        assert!(g.nodes[0].provides().iter().all(|d| d.target_node() != 2));
        assert!(g.nodes[2].depends().iter().all(|d| d.target_node() != 0));
        assert!(!g.nodes[2].deps.contains(&0));
        assert!(!g.nodes[0].users.contains(&2));
        // The rerouted edges landed, carrying the failarg flag on earlyexit -> 2.
        assert!(g.nodes[2].deps.contains(&earlyexit));
        assert!(g.nodes[earlyexit].deps.contains(&0));
        assert!(g.nodes[earlyexit].provides()[0].is_failarg());

        // schedule_operations drops the imaginary node yet honors its ordering:
        // 0 -> earlyexit -> 2 forces 2 last, and 0/1 (in-degree 0) come first.
        let schedule = schedule_operations(&g);
        assert_eq!(schedule.len(), 3);
        assert!(!schedule.contains(&earlyexit));
        assert_eq!(schedule.last(), Some(&2));
        assert!(schedule.contains(&0) && schedule.contains(&1));
    }
}
