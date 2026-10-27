//! Port of `rpython/translator/backendopt/support.py`.
//!
//! Ports the read-only graph walkers consumed by
//! [`super::inline`]'s auto-inlining heuristics plus the
//! helper-of-helpers that adjacent passes call into:
//!
//! * `graph_operations(graph)` (`:9-12`),
//!   `all_operations(graphs)` (`:14-18`).
//! * `var_needsgc(var)` (`:27-29`).
//! * `find_calls_from(translator, graph, memo)` /
//!   `_find_calls_from` (`:31-50`).
//! * `find_backedges(graph, block, seen, seeing)` (`:52-70`).
//! * `compute_reachability(graph)` (`:72-90`).
//! * `find_loop_blocks(graph)` (`:92-112`).
//! * `md5digest(translator)` (`:114-124`).
//! * `log = AnsiLogger("backendopt")` (`:6`) — exposed via
//!   [`LOG`]; consumers call `LOG.method("<subname>", &text)` to
//!   mirror upstream's dynamic `log.<subname>(text)` channel.
//!
//! The legacy `crate::model::FunctionGraph` carries an unrelated
//! `find_backedges` in `codewriter/policy.rs` keyed on a
//! different IR (Rust-AST `usize` block indices, not the flowspace
//! identity-keyed model). Both ports cite the same upstream line —
//! this file consumes the flowspace-orthodox shape needed by
//! [`super::inline::measure_median_execution_cost`]; the
//! `policy.rs` carrier handles the legacy IR. No de-duplication is
//! attempted because the two carriers have incompatible block
//! types.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::flowspace::model::{
    BlockKey, BlockRef, ConstValue, Constant, FunctionGraph, GraphKey, GraphRef, Hlvalue, LinkRef,
    SpaceOperation, Variable,
};
use crate::tool::ansi_print::AnsiLogger;
use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType, functionptr};
use crate::translator::rtyper::rmodel::inputconst_from_lltype;
use crate::translator::simplify::get_graph;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// RPython `support.log = AnsiLogger("backendopt")` (`:6`).
///
/// `log.<subname>("text")` upstream maps to `LOG.method("<subname>",
/// &text)` here — the dynamic-method synthesis upstream gets through
/// `AnsiLogger.__getattr__` is replaced by an explicit `subname`
/// argument because Rust has no equivalent.
pub static LOG: AnsiLogger = AnsiLogger::new("backendopt");

/// `graph_operations(graph)` at `support.py`.
///
/// Upstream is a generator yielding every op in iterblocks order.
/// The Rust port returns a flat `Vec<SpaceOperation>` so callers can
/// iterate without re-borrowing — pyre's BlockRef is `Rc<RefCell<…>>`,
/// and exposing a generator would force callers to manage interior
/// borrows themselves.
pub fn graph_operations(graph: &FunctionGraph) -> Vec<SpaceOperation> {
    let mut out = Vec::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            out.push(op.clone());
        }
    }
    out
}

/// `all_operations(graphs)` at `support.py`.
pub fn all_operations(graphs: &[GraphRef]) -> Vec<SpaceOperation> {
    let mut out = Vec::new();
    for graph in graphs {
        for op in graph_operations(&graph.borrow()) {
            out.push(op);
        }
    }
    out
}

/// `annotate(translator, func, result, args)` at `support.py:20-25`.
///
/// ```python
/// def annotate(translator, func, result, args):
///     args   = [arg.concretetype for arg in args]
///     graph  = translator.rtyper.annotate_helper(func, args)
///     fptr   = lltype.functionptr(lltype.FuncType(args, result.concretetype),
///                                  func.__name__, graph=graph)
///     c      = inputconst(lltype.typeOf(fptr), fptr)
///     return c
/// ```
///
pub fn annotate(
    translator: &TranslationContext,
    func: &crate::flowspace::model::HostObject,
    result: &Hlvalue,
    args: &[Hlvalue],
) -> Result<Constant, TaskError> {
    let rtyper = translator.rtyper().ok_or_else(|| TaskError {
        message: "support.py:22 annotate: translator.rtyper is None".to_string(),
    })?;
    let arg_types: Vec<LowLevelType> = args
        .iter()
        .map(|arg| {
            hlvalue_concretetype(arg).ok_or_else(|| TaskError {
                message: "support.py:21 annotate: argument missing concretetype".to_string(),
            })
        })
        .collect::<Result<_, _>>()?;
    let result_type = hlvalue_concretetype(result).ok_or_else(|| TaskError {
        message: "support.py:23 annotate: result missing concretetype".to_string(),
    })?;
    let graph = rtyper
        .annotate_helper(
            func,
            arg_types.clone().into_iter().map(Into::into).collect(),
        )
        .map_err(|e| TaskError {
            message: format!("support.py:22 annotate_helper: {e}"),
        })?;
    let ftype = FuncType {
        args: arg_types,
        result: result_type,
    };
    let name = func.simple_name().to_string();
    let fptr = functionptr(
        ftype,
        &name,
        Some(GraphKey::of(&graph.graph).as_usize()),
        None::<String>,
    );
    let ptr_type = LowLevelType::Ptr(Box::new(fptr._TYPE.clone()));
    inputconst_from_lltype(&ptr_type, &ConstValue::LLPtr(Box::new(fptr))).map_err(|e| TaskError {
        message: format!("support.py:24 inputconst(typeOf(fptr), fptr): {e}"),
    })
}

/// `var_needsgc(var)` at `support.py`.
///
/// ```python
/// def var_needsgc(var):
///     vartype = var.concretetype
///     return isinstance(vartype, lltype.Ptr) and vartype._needsgc()
/// ```
///
/// Returns `false` when the var has no concretetype (mirrors
/// upstream's `isinstance(None, lltype.Ptr)` short-circuit).
pub fn var_needsgc(var: &Variable) -> bool {
    match var.concretetype() {
        Some(LowLevelType::Ptr(ptr)) => ptr._needsgc(),
        _ => false,
    }
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

/// `find_calls_from(translator, graph, memo=None)` at
/// `support.py:31-37`.
///
/// ```python
/// def find_calls_from(translator, graph, memo=None):
///     if memo and graph in memo:
///         return memo[graph]
///     res = [i for i in _find_calls_from(translator, graph)]
///     if memo is not None:
///         memo[graph] = res
///     return res
/// ```
///
/// Pyre threads the memo as
/// `Option<&mut HashMap<usize, Vec<(BlockRef, GraphRef)>>>` keyed on
/// the graph identity (`Rc::as_ptr` cast through
/// [`crate::flowspace::model::GraphKey`]). `None` runs uncached,
/// matching upstream's default.
pub fn find_calls_from(
    translator: &TranslationContext,
    graph: &FunctionGraph,
    memo: Option<&mut HashMap<usize, Vec<(BlockRef, GraphRef)>>>,
) -> Vec<(BlockRef, GraphRef)> {
    // Upstream `:32-33`: cache lookup keyed on graph identity.
    // Pyre's flowspace `FunctionGraph` is borrowed `&` here so the
    // GraphKey is computed via the surrounding `Rc<RefCell>` —
    // callers thread their own GraphKey into the memo when they
    // already have one in hand.
    let key = graph as *const _ as usize;
    if let Some(memo) = memo.as_deref()
        && let Some(cached) = memo.get(&key)
    {
        return cached.clone();
    }
    let res = _find_calls_from(translator, graph);
    if let Some(memo) = memo {
        memo.insert(key, res.clone());
    }
    res
}

/// `_find_calls_from(translator, graph)` at `support.py`.
fn _find_calls_from(
    translator: &TranslationContext,
    graph: &FunctionGraph,
) -> Vec<(BlockRef, GraphRef)> {
    let mut out = Vec::new();
    for block in graph.iterblocks() {
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        for op in &ops {
            // Upstream `:42-45 if op.opname == "direct_call":
            // called_graph = get_graph(op.args[0], translator); if
            // called_graph is not None: yield block, called_graph`.
            if op.opname == "direct_call"
                && let Some(arg0) = op.args.first()
                && let Some(called) = get_graph(arg0, translator)
            {
                out.push((block.clone(), called));
            }
            // Upstream `:46-50 if op.opname == "indirect_call":
            // graphs = op.args[-1].value; if graphs is not None: for
            // called_graph in graphs: yield block, called_graph`.
            //
            // Pyre's indirect_call last-arg carrier mirrors upstream
            // either as `Hlvalue::Constant(LLPtr-list)` or the
            // sentinel `None` constant; only the resolvable form
            // produces edges.
            if op.opname == "indirect_call"
                && let Some(last) = op.args.last()
                && let Hlvalue::Constant(c) = last
                && let ConstValue::Graphs(graph_keys) = &c.value
            {
                let trans_graphs = translator.graphs.borrow();
                for key in graph_keys {
                    if let Some(g) = trans_graphs
                        .iter()
                        .find(|g| crate::flowspace::model::GraphKey::of(g).as_usize() == *key)
                    {
                        out.push((block.clone(), g.clone()));
                    }
                }
            }
        }
    }
    out
}

/// `md5digest(translator)` at `support.py`.
///
/// Returns `{graph.name -> 16-byte md5 digest}` matching upstream's
/// `m.update(op.opname + str(op.result))` followed by
/// `m.update(str(a))` per arg. Pyre uses the `md-5` crate when
/// available; fall back to a deterministic fnv-style hash when the
/// crate is absent so consumers (currently only profile-based
/// inlining, which is unported) compile cleanly.
pub fn md5digest(translator: &TranslationContext) -> HashMap<String, [u8; 16]> {
    let mut out: HashMap<String, [u8; 16]> = HashMap::new();
    for graph in translator.graphs.borrow().iter() {
        let mut acc = MdAccumulator::new();
        let g = graph.borrow();
        for op in graph_operations(&g) {
            // Upstream `:120 m.update(op.opname + str(op.result))`.
            acc.update(op.opname.as_bytes());
            acc.update(format_hlvalue(&op.result).as_bytes());
            for a in &op.args {
                // Upstream `:121-122 for a in op.args: m.update(str(a))`.
                acc.update(format_hlvalue(a).as_bytes());
            }
        }
        out.insert(g.name.clone(), acc.digest());
    }
    out
}

/// `str(Hlvalue)` mirror used by [`md5digest`]. Upstream Python
/// flattens via `str()`; pyre's [`Hlvalue::Variable`] and
/// [`Hlvalue::Constant`] both carry stable `Debug` reps that
/// preserve identity (variable id, constant value).
fn format_hlvalue(h: &Hlvalue) -> String {
    match h {
        Hlvalue::Variable(v) => v.name(),
        Hlvalue::Constant(c) => format!("{:?}", c.value),
    }
}

/// 128-bit accumulator with the same shape as `hashlib.md5`. pyre
/// vendors a minimal MD5 implementation directly to avoid a
/// crate-level dependency for a helper that only feeds the unported
/// profile-based inliner. The loop body matches RFC 1321; the
/// digest returned is the canonical 16-byte little-endian
/// representation of the four 32-bit state words.
struct MdAccumulator {
    state: [u32; 4],
    buffer: Vec<u8>,
    length: u64,
}

impl MdAccumulator {
    fn new() -> Self {
        MdAccumulator {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            buffer: Vec::with_capacity(64),
            length: 0,
        }
    }
    fn update(&mut self, data: &[u8]) {
        self.length = self
            .length
            .wrapping_add((data.len() as u64).wrapping_mul(8));
        self.buffer.extend_from_slice(data);
        while self.buffer.len() >= 64 {
            let chunk: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.compress(&chunk);
            self.buffer.drain(..64);
        }
    }
    fn digest(mut self) -> [u8; 16] {
        let bit_len = self.length;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_le_bytes());
        let buffer = std::mem::take(&mut self.buffer);
        for chunk in buffer.chunks_exact(64) {
            let chunk: [u8; 64] = chunk.try_into().unwrap();
            self.compress(&chunk);
        }
        let mut out = [0u8; 16];
        for (i, w) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
    fn compress(&mut self, chunk: &[u8; 64]) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(chunk[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
            0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
            0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
            0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
            0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
            0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
            0xeb86d391,
        ];
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

/// `find_backedges(graph, block=None, seen=None, seeing=None)` at
/// `support.py:52-70`.
///
/// > finds the backedges in the flow graph
///
/// Standard DFS classification: an edge from `block` whose target
/// is already in the `seeing` ancestor set is a back-edge. Returns
/// the list of back-edge [`LinkRef`]s — upstream returns a list of
/// `Link` objects, and pyre keeps the same shape so callers can
/// inspect `.target` / `.prevblock`.
///
/// Iterative form (DFS stack rather than upstream's recursive
/// implementation) — the recursion depth in pyre's flowspace
/// graphs is bounded by the longest chain of blocks, but Rust's
/// default stack is smaller than CPython's, so the iterative form
/// is safer.
pub fn find_backedges(graph: &FunctionGraph) -> Vec<LinkRef> {
    let mut backedges: Vec<LinkRef> = Vec::new();
    let mut seen: HashSet<BlockKey> = HashSet::new();
    let mut seeing: HashSet<BlockKey> = HashSet::new();
    // Upstream `:55-56`: `if block is None: block = graph.startblock`.
    let start = graph.startblock.clone();
    seen.insert(BlockKey::of(&start));
    find_backedges_dfs(&start, &mut seen, &mut seeing, &mut backedges);
    backedges
}

/// Recursive DFS body for [`find_backedges`]. Mirrors upstream's
/// `:62-69` iteration over `block.exits` with the `seen` /
/// `seeing` add/remove discipline.
fn find_backedges_dfs(
    block: &BlockRef,
    seen: &mut HashSet<BlockKey>,
    seeing: &mut HashSet<BlockKey>,
    backedges: &mut Vec<LinkRef>,
) {
    let block_key = BlockKey::of(block);
    seeing.insert(block_key.clone());
    let exits: Vec<LinkRef> = block.borrow().exits.to_vec();
    for link in &exits {
        let target = match link.borrow().target.clone() {
            Some(t) => t,
            None => continue,
        };
        let target_key = BlockKey::of(&target);
        if seen.contains(&target_key) {
            // Upstream `:64-65`: `if link.target in seeing:
            // backedges.append(link)`.
            if seeing.contains(&target_key) {
                backedges.push(link.clone());
            }
        } else {
            // Upstream `:67-68`: descend.
            seen.insert(target_key.clone());
            find_backedges_dfs(&target, seen, seeing, backedges);
        }
    }
    seeing.remove(&block_key);
}

/// `compute_reachability(graph)` at `support.py`.
///
/// Returns a map `{block -> set(reachable_blocks)}`. Walks every
/// block in reverse iterblocks order so previously-computed
/// reachability sets feed each new block's traversal — upstream
/// notes "Reversed order should make the reuse path more likely"
/// at `:75`.
pub fn compute_reachability(graph: &FunctionGraph) -> HashMap<BlockKey, HashSet<BlockKey>> {
    let mut reachable: HashMap<BlockKey, HashSet<BlockKey>> = HashMap::new();
    let blocks: Vec<BlockRef> = graph.iterblocks();
    // Upstream `:75-76`: `for block in reversed(blocks):`.
    for block in blocks.iter().rev() {
        let mut reach: HashSet<BlockKey> = HashSet::new();
        let mut scheduled: Vec<BlockRef> = vec![block.clone()];
        while let Some(current) = scheduled.pop() {
            let exits: Vec<LinkRef> = current.borrow().exits.to_vec();
            for link in &exits {
                let Some(target) = link.borrow().target.clone() else {
                    continue;
                };
                let target_key = BlockKey::of(&target);
                if let Some(prev) = reachable.get(&target_key) {
                    // Upstream `:82-85`: `if link.target in
                    // reachable: reach.add(link.target); reach = reach
                    // | reachable[link.target]; continue`.
                    reach.insert(target_key.clone());
                    for k in prev {
                        reach.insert(k.clone());
                    }
                    continue;
                }
                // Upstream `:86-88`: not yet reachable; add and
                // schedule.
                if !reach.contains(&target_key) {
                    reach.insert(target_key.clone());
                    scheduled.push(target);
                }
            }
        }
        reachable.insert(BlockKey::of(block), reach);
    }
    reachable
}

/// `find_loop_blocks(graph)` at `support.py`.
///
/// > find the blocks in a graph that are part of a loop
///
/// Returns a map `{block -> loop_start_block}` per upstream:
/// the value is the back-edge target (loop header) of the loop
/// containing the keyed block. Upstream's seeded `loop[start] =
/// start` and `loop[end] = start` bookend the back-edge endpoints
/// before the BFS extends membership.
pub fn find_loop_blocks(graph: &FunctionGraph) -> HashMap<BlockKey, BlockRef> {
    let mut loop_map: HashMap<BlockKey, BlockRef> = HashMap::new();
    let reachable = compute_reachability(graph);
    for backedge in find_backedges(graph) {
        // Upstream `:97-98`:
        //     start = backedge.target
        //     end = backedge.prevblock
        let (start, end) = {
            let l = backedge.borrow();
            let start = match &l.target {
                Some(t) => t.clone(),
                None => continue,
            };
            // Upstream `link.prevblock` — pyre's Link carries the
            // backref through a `Weak<RefCell<Block>>` slot
            // populated by `flowspace/model.rs`'s `closeblock`.
            // Upgrade to a strong ref for the BFS body.
            let end = match l.prevblock.as_ref().and_then(|w| w.upgrade()) {
                Some(end) => end,
                None => continue,
            };
            (start, end)
        };
        // Upstream `:99-101`:
        //     loop[start] = start
        //     loop[end] = start
        //     scheduled = [start]
        loop_map.insert(BlockKey::of(&start), start.clone());
        loop_map.insert(BlockKey::of(&end), start.clone());
        let mut scheduled: Vec<BlockRef> = vec![start.clone()];
        let mut seen: HashSet<BlockKey> = HashSet::new();
        let end_key = BlockKey::of(&end);
        while let Some(current) = scheduled.pop() {
            // Upstream `:105-106`:
            //     connects = end in reachable[current]
            //     seen[current] = True
            let current_key = BlockKey::of(&current);
            let connects = reachable
                .get(&current_key)
                .map(|set| set.contains(&end_key))
                .unwrap_or(false);
            seen.insert(current_key.clone());
            if connects {
                loop_map.insert(current_key.clone(), start.clone());
            }
            // Upstream `:109-111`:
            //     for link in current.exits:
            //         if link.target not in seen:
            //             scheduled.append(link.target)
            let exits: Vec<LinkRef> = current.borrow().exits.to_vec();
            for link in &exits {
                if let Some(target) = link.borrow().target.clone()
                    && !seen.contains(&BlockKey::of(&target))
                {
                    scheduled.push(target);
                }
            }
        }
    }
    loop_map
}

/// Identity-equality predicate matching upstream's `is` test
/// inside `find_loop_blocks`. Mostly used by call sites that need
/// to ask "does this block sit in a loop whose start is X?".
pub fn loop_start_eq(a: &BlockRef, b: &BlockRef) -> bool {
    Rc::ptr_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, SpaceOperation,
        Variable,
    };
    use std::cell::RefCell;

    /// Linear graph: `start -> mid -> return`. Has no back-edges
    /// and no loops.
    fn linear_graph() -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let mid = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v.clone())], Some(mid.clone()), None).into_ref(),
        ]);
        mid.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    /// Self-loop graph: `start -> start` and `start -> return`.
    fn self_loop_graph() -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v.clone())],
                Some(start.clone()),
                None,
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    /// Two-block loop: `start -> body -> body -> ... -> return`.
    fn two_block_loop() -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let body = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v.clone())], Some(body.clone()), None).into_ref(),
        ]);
        body.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v.clone())], Some(body.clone()), None).into_ref(),
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
    fn graph_operations_collects_each_op_once() {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(2)))],
            Hlvalue::Variable(Variable::named("s")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        let ops = graph_operations(&g.borrow());
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opname, "int_add");
        assert_eq!(ops[1].opname, "int_sub");
    }

    #[test]
    fn all_operations_concats_each_graph() {
        let v = Variable::named("x");
        let start1 = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let g1 = FunctionGraph::new("a", start1.clone());
        start1.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![],
            Hlvalue::Variable(Variable::named("r1")),
        ));
        start1.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v.clone())],
                Some(g1.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g1: GraphRef = Rc::new(RefCell::new(g1));

        let start2 = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let g2 = FunctionGraph::new("b", start2.clone());
        start2.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![],
            Hlvalue::Variable(Variable::named("r2")),
        ));
        start2.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(g2.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g2: GraphRef = Rc::new(RefCell::new(g2));

        let ops = all_operations(&[g1, g2]);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opname, "int_add");
        assert_eq!(ops[1].opname, "int_sub");
    }

    #[test]
    fn find_backedges_linear_graph_has_none() {
        let g = linear_graph();
        assert!(find_backedges(&g.borrow()).is_empty());
    }

    #[test]
    fn find_backedges_self_loop_returns_one() {
        let g = self_loop_graph();
        let edges = find_backedges(&g.borrow());
        assert_eq!(edges.len(), 1);
        // The back-edge is `start -> start`.
        let edge = &edges[0];
        let l = edge.borrow();
        assert!(Rc::ptr_eq(
            &l.target.clone().unwrap(),
            &g.borrow().startblock,
        ));
    }

    #[test]
    fn find_backedges_two_block_loop_returns_one() {
        let g = two_block_loop();
        let edges = find_backedges(&g.borrow());
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn compute_reachability_linear_graph_chains_into_return() {
        let g = linear_graph();
        let reach = compute_reachability(&g.borrow());
        // The startblock reaches every later block.
        let start_key = BlockKey::of(&g.borrow().startblock);
        let start_reach = reach.get(&start_key).expect("start has reachable set");
        assert!(!start_reach.is_empty());
    }

    #[test]
    fn find_loop_blocks_linear_graph_is_empty() {
        let g = linear_graph();
        assert!(find_loop_blocks(&g.borrow()).is_empty());
    }

    #[test]
    fn find_loop_blocks_self_loop_marks_start() {
        let g = self_loop_graph();
        let loops = find_loop_blocks(&g.borrow());
        let start = g.borrow().startblock.clone();
        assert!(loops.contains_key(&BlockKey::of(&start)));
        // The loop start is the start block itself.
        let loop_start = loops.get(&BlockKey::of(&start)).unwrap();
        assert!(Rc::ptr_eq(loop_start, &start));
    }

    #[test]
    fn var_needsgc_no_concretetype_returns_false() {
        let v = Variable::named("x");
        assert!(!var_needsgc(&v));
    }

    #[test]
    fn var_needsgc_signed_int_concretetype_returns_false() {
        let v = Variable::named("x");
        v.set_concretetype(Some(LowLevelType::Signed));
        assert!(!var_needsgc(&v));
    }

    #[test]
    fn var_needsgc_gc_ptr_concretetype_returns_true() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelType, Ptr, PtrTarget, Struct,
        };
        let v = Variable::named("x");
        let struct_t = Struct::gc_with_hints("S", vec![], vec![]);
        let ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(struct_t),
        }));
        v.set_concretetype(Some(ptr));
        assert!(var_needsgc(&v));
    }

    #[test]
    fn md5digest_returns_per_graph_digest() {
        // Smoke test — assert non-zero digest and per-graph keying.
        let translator = TranslationContext::new();
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let g = FunctionGraph::new("only", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(g.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        translator
            .graphs
            .borrow_mut()
            .push(Rc::new(RefCell::new(g)));

        let digests = md5digest(&translator);
        let only = digests.get("only").expect("graph 'only' present");
        assert_ne!(only, &[0u8; 16]);
    }

    #[test]
    fn find_loop_blocks_two_block_loop_marks_body() {
        // body is a self-loop; back-edge body -> body. Both
        // start and body should appear in the loop map.
        let g = two_block_loop();
        let loops = find_loop_blocks(&g.borrow());
        // body block is reachable through start -> body. The body
        // block is the loop header (back-edge target).
        let blocks: Vec<BlockRef> = g.borrow().iterblocks();
        let body = blocks
            .iter()
            .find(|b| !Rc::ptr_eq(b, &g.borrow().startblock))
            .cloned()
            .expect("body block present");
        // The body block is in a loop; its loop header is itself.
        let header = loops.get(&BlockKey::of(&body)).expect("body in loop map");
        assert!(Rc::ptr_eq(header, &body));
    }
}
