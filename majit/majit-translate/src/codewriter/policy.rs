//! JIT inlining policy.
//!
//! Translated from `rpython/jit/codewriter/policy.py`.
//!
//! `JitPolicy` decides which graphs the codewriter should "look inside"
//! and inline-trace.  RPython models this as a base class with a virtual
//! `look_inside_function`; subclasses (e.g. `StopAtXPolicy`) override that
//! one method.  In Rust we use a trait + state struct so subclasses share
//! the bookkeeping fields.
//!
//! ## Subclasses
//!
//! `pypy/module/pypyjit/policy.py` `PyPyJitPolicy` overrides
//! `look_inside_function` to reject whole interpreter modules by
//! `func.__module__`.  Its pyre port lives beside the interpreter
//! (`pyre-interpreter/src/module/pypyjit/policy.rs`) and reaches the
//! translator through `analyze_multiple_pipeline_from_llbc_with_modules`'s
//! `policy` argument, the way `targetpypystandalone.py jitpolicy` hands it to
//! `apply_jit`.  It reads the module off `graph.func.module`
//! ([`crate::model::FuncEffects::module`]), which the front end stamps from
//! each function's Charon name. `_elidable_function_`, `_jit_look_inside_`
//! and `_jit_unroll_safe_` are the same object's fields.
//!
//! A callee with no registered graph never reaches this policy:
//! `call.py guess_call_kind` answers `'residual'` for a funcobj without a
//! graph, and `find_all_graphs` only enqueues callees that have one.  In pyre
//! that covers every body outside the lowered crate set (Rust stdlib,
//! unextracted crates).  The contract is pinned by
//! `tests/test_phase_d_find_all_graphs_parity.rs::
//! find_all_graphs_leaves_unregistered_targets_as_residual`.

use std::collections::HashSet;

use crate::flowspace::model::ConstValue;
use crate::model::{Block, BlockId, FunctionGraph, LinkArg, OpKind, ValueType};

/// policy.py: shared mutable state and the default classifier.
///
/// `JitPolicy.__init__` initializes:
///   - `self.unsafe_loopy_graphs = set()`
///   - `self.supports_floats = False`
///   - `self.supports_longlong = False`
///   - `self.supports_singlefloats = False`
///   - `self.jithookiface = jithookiface`
#[derive(Debug, Clone, Default)]
pub struct JitPolicyState {
    pub unsafe_loopy_graphs: HashSet<String>,
    pub supports_floats: bool,
    pub supports_longlong: bool,
    pub supports_singlefloats: bool,
    /// policy.py:16: optional `jithookiface`.  Pyre does not yet expose
    /// JIT hooks, so this stays as a marker placeholder.
    pub jithookiface: Option<()>,
}

impl JitPolicyState {
    /// policy.py:11-16: constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// policy.py:18-19
    pub fn set_supports_floats(&mut self, flag: bool) {
        self.supports_floats = flag;
    }

    /// policy.py:21-22
    pub fn set_supports_longlong(&mut self, flag: bool) {
        self.supports_longlong = flag;
    }

    /// policy.py:24-25
    pub fn set_supports_singlefloats(&mut self, flag: bool) {
        self.supports_singlefloats = flag;
    }

    /// policy.py `dump_unsafe_loops`.
    ///
    /// ```python
    /// def dump_unsafe_loops(self):
    ///     f = udir.join("unsafe-loops.txt").open('w')
    ///     strs = [str(graph) for graph in self.unsafe_loopy_graphs]
    ///     strs.sort()
    ///     for graph in strs:
    ///         print(graph, file=f)
    ///     f.close()
    /// ```
    ///
    /// RPython's `udir` is the translator's per-run temp directory.
    /// [`Self::dump_unsafe_loops_to_udir`] writes `udir/unsafe-loops.txt`.
    /// Tests pass an explicit path.
    pub fn dump_unsafe_loops(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut strs: Vec<&String> = self.unsafe_loopy_graphs.iter().collect();
        strs.sort();
        let mut f = std::fs::File::create(path)?;
        for graph in strs {
            writeln!(f, "{}", graph)?;
        }
        Ok(())
    }

    /// `udir.join("unsafe-loops.txt")`.
    pub fn dump_unsafe_loops_to_udir(&self) -> std::io::Result<()> {
        self.dump_unsafe_loops(&crate::tool::udir::udir().join("unsafe-loops.txt"))
    }
}

/// `JitPolicy` interface.
///
/// policy.py:10 `class JitPolicy`. `look_inside_function` is the only
/// upstream method designed for subclassing; everything else is a default
/// implementation calling it.
pub trait JitPolicy {
    fn state(&self) -> &JitPolicyState;
    fn state_mut(&mut self) -> &mut JitPolicyState;

    /// policy.py — return `True` for every function by default.
    /// `StopAtXPolicy` overrides this.
    ///
    /// Upstream passes `graph.func`. The function object's attributes this
    /// policy reads live on [`crate::model::FuncEffects`]: `elidable`,
    /// `jit_look_inside`, `unroll_safe`, and `module`.
    fn look_inside_function(&self, _func: &FunctionGraph) -> bool {
        true
    }

    /// policy.py `_reject_function(func)`.
    ///
    /// `_elidable_function_` is always opaque. So is a function whose
    /// module starts with `rpython.rtyper.module.` — the helpers under
    /// `majit_translate::translator::rtyper::module::`. `ll_math` lives
    /// under `lltypesystem.module` and is not in that prefix.
    /// `func.__module__ or '?'` is `'?'` when the attribute is missing
    /// or empty, and `'?'` does not match the prefix.
    fn _reject_function(&self, func: &FunctionGraph) -> bool {
        if func.func.elidable {
            return true;
        }
        let module = func
            .func
            .module
            .as_deref()
            .filter(|module| !module.is_empty())
            .unwrap_or("?");
        module.starts_with("majit_translate::translator::rtyper::module::")
    }

    /// policy.py `look_inside_graph(graph)`.
    ///
    /// `func._jit_look_inside_` overrides everything; otherwise we
    /// combine `look_inside_function` and `_reject_function`.  Loops
    /// disqualify a graph unless it is `_jit_unroll_safe_`.  A
    /// reject due to loops is recorded in `unsafe_loopy_graphs`.
    ///
    /// A codewriter [`FunctionGraph`] always carries `func`. The
    /// `AttributeError` arm (see the function, skip `unroll_safe`) is
    /// the flow-graph case with no func object, which this type does
    /// not represent.
    fn look_inside_graph(&mut self, graph: &FunctionGraph) -> bool {
        let mut contains_loop = !find_backedges(graph).is_empty();
        let see_function = if let Some(flag) = graph.func.jit_look_inside {
            // policy.py:56-57: `_jit_look_inside_` override.
            flag
        } else {
            self.look_inside_function(graph) && !self._reject_function(graph)
        };
        contains_loop = contains_loop && !graph.func.unroll_safe;

        let res = see_function
            && !contains_unsupported_variable_type(
                graph,
                self.state().supports_floats,
                self.state().supports_longlong,
                self.state().supports_singlefloats,
            );
        if res && contains_loop {
            self.state_mut()
                .unsafe_loopy_graphs
                .insert(graph.name.clone());
        }
        let res = res && !contains_loop;
        // policy.py:71-83 `access_directly` virtualizable safety gate.
        //
        // RPython raises `ValueError("access_directly on a function which
        // we don't see ...")` when three conditions meet:
        //   - `see_function` is True (annotator determined the function is
        //     part of the JIT-visible graph set),
        //   - `res` is False (loops or unsupported types mean
        //     `look_inside_graph` decided not to trace into it),
        //   - `graph.access_directly` is True (annotator set this because
        //     an ARGUMENT carried the `access_directly` flag, see
        //     `default_specialize` in `rpython/annotator/specialize.py`).
        //
        // Turning the call into a residual call while the function
        // accesses a virtualizable would silently desynchronise the
        // virtualizable from the JIT's view; upstream therefore aborts
        // translation loudly. Pyre carries the same flag where upstream
        // does, on the graph: `FunctionGraph::access_directly`, beside
        // `hints`.
        //
        // `find_all_graphs` stamps the flag on a callee that receives the
        // result of `hint_access_directly` before this gate, which is the
        // read `default_specialize` performs on the argument annotation.
        // The prepass annotator still runs after the BFS and writes
        // `PyGraph.access_directly`; `cutover::check_access_directly_sanity`
        // asserts that no graph outside the JIT graph set carries the flag.
        if see_function && !res && graph.access_directly {
            panic!(
                "access_directly on a function which we don't see: {}",
                graph.name
            );
        }
        // A `false` here is the policy's refusal to let this callee become
        // a JitCode, and it reaches the caller as a bare `bool` — four
        // structurally different clauses collapsed into one answer.
        // `unsafe_loopy_graphs` already records ONE of them (and only when
        // `res` was still true at that point), so it cannot stand in for
        // the rest.  Re-derive which clause refused; guarded on the census
        // so the extra predicate calls never run on the decision path they
        // measure.
        if !res && crate::decline::enabled() {
            let reason = if !see_function {
                if graph.func.jit_look_inside == Some(false) {
                    "dont_look_inside-hint"
                } else if graph.func.elidable {
                    "elidable-hint"
                } else if self._reject_function(graph) {
                    "rtyper-module"
                } else {
                    "look_inside_function-said-no"
                }
            } else if contains_loop {
                "loop-without-unroll_safe"
            } else {
                "unsupported-variable-type"
            };
            crate::decline::record(
                crate::decline::gate::LOOK_INSIDE_GRAPH,
                reason,
                format_args!(
                    "{}",
                    graph
                        .source_identity
                        .as_deref()
                        .unwrap_or(graph.name.as_str())
                ),
            );
        }
        res
    }
}

/// Default policy: equivalent to instantiating `JitPolicy()` in RPython.
#[derive(Debug, Clone, Default)]
pub struct DefaultJitPolicy {
    pub state: JitPolicyState,
}

impl DefaultJitPolicy {
    pub fn new() -> Self {
        Self {
            state: JitPolicyState::new(),
        }
    }
}

impl JitPolicy for DefaultJitPolicy {
    fn state(&self) -> &JitPolicyState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut JitPolicyState {
        &mut self.state
    }
}

/// policy.py `class StopAtXPolicy(JitPolicy)`.
///
/// Excludes a fixed list of function names from inlining.  Used by
/// translator tests that need to JIT-compile one half of a graph and
/// keep the other half opaque.
#[derive(Debug, Clone, Default)]
pub struct StopAtXPolicy {
    pub state: JitPolicyState,
    /// policy.py: `self.funcs = funcs` — list of opaque names.
    pub funcs: Vec<String>,
}

impl StopAtXPolicy {
    pub fn new(funcs: Vec<String>) -> Self {
        Self {
            state: JitPolicyState::new(),
            funcs,
        }
    }
}

impl JitPolicy for StopAtXPolicy {
    fn state(&self) -> &JitPolicyState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut JitPolicyState {
        &mut self.state
    }
    /// policy.py: `return func not in self.funcs`.
    fn look_inside_function(&self, func: &FunctionGraph) -> bool {
        !self.funcs.iter().any(|f| f == &func.name)
    }
}

/// `policy.py contains_unsupported_variable_type(graph, ...)`.
///
/// ```python
/// def contains_unsupported_variable_type(graph, supports_floats,
///                                               supports_longlong,
///                                               supports_singlefloats):
///     getkind = history.getkind
///     try:
///         for block in graph.iterblocks():
///             for v in block.inputargs:
///                 getkind(v.concretetype, ...)
///             for op in block.operations:
///                 for v in op.args:
///                     getkind(v.concretetype, ...)
///                 v = op.result
///                 getkind(v.concretetype, ...)
///     except NotImplementedError as e:
///         log.WARNING('%s, ignoring graph' % (e,))
///         log.WARNING('  %s' % (graph,))
///         return True
///     return False
/// ```
///
/// Upstream reaches every value's type through `v.concretetype` and calls
/// [`crate::model::try_getkind`] on `block.inputargs`, `op.args` and
/// `op.result`. At `find_all_graphs` that cell is usually still unset.
/// The width the codewriter later reads lives on the [`ValueType`] an
/// [`OpKind`] declares, so the same `getkind` rules are applied there
/// ([`value_type_has_kind`], [`collect_declared_value_types`]) over the
/// startblock-reachable closure `graph.iterblocks()` yields. A set
/// `concretetype` is asked as well, including link constants.
///
/// A `true` result is the `NotImplementedError` that residualizes the
/// graph. `look_inside_graph` records it as `"unsupported-variable-type"`.
pub fn contains_unsupported_variable_type(
    graph: &FunctionGraph,
    supports_floats: bool,
    supports_longlong: bool,
    supports_singlefloats: bool,
) -> bool {
    // `iterblocks()` parity (`rpython/flowspace/model.py`): the
    // startblock-reachable closure over `Block.exits`, id-keyed because
    // block ids need not be index-aligned with `blocks` storage order.
    let by_id: std::collections::HashMap<BlockId, &Block> =
        graph.blocks.iter().map(|b| (b.id, b)).collect();
    let mut block_seen: HashSet<BlockId> = HashSet::new();
    let mut stack = vec![graph.startblock];
    let mut declared: Vec<&ValueType> = Vec::new();
    while let Some(bid) = stack.pop() {
        if !block_seen.insert(bid) {
            continue;
        }
        let Some(block) = by_id.get(&bid) else {
            continue;
        };
        if block.inputargs.iter().any(|var| {
            variable_concretetype_refused(
                var,
                supports_floats,
                supports_longlong,
                supports_singlefloats,
            )
        }) {
            return true;
        }
        for op in &block.operations {
            declared.clear();
            collect_declared_value_types(&op.kind, &mut declared);
            if declared.iter().any(|ty| {
                !value_type_has_kind(
                    ty,
                    supports_floats,
                    supports_longlong,
                    supports_singlefloats,
                )
            }) {
                return true;
            }
            if crate::inline::op_variable_refs(&op.kind).iter().any(|var| {
                variable_concretetype_refused(
                    var,
                    supports_floats,
                    supports_longlong,
                    supports_singlefloats,
                )
            }) {
                return true;
            }
            if op.result.as_ref().is_some_and(|var| {
                variable_concretetype_refused(
                    var,
                    supports_floats,
                    supports_longlong,
                    supports_singlefloats,
                )
            }) {
                return true;
            }
        }
        if block.exits.iter().flat_map(|link| &link.args).any(|arg| {
            link_arg_refused(
                arg,
                supports_floats,
                supports_longlong,
                supports_singlefloats,
            )
        }) {
            return true;
        }
        stack.extend(block.exits.iter().rev().map(|e| e.target));
    }
    false
}

fn variable_concretetype_refused(
    var: &crate::flowspace::model::Variable,
    supports_floats: bool,
    supports_longlong: bool,
    supports_singlefloats: bool,
) -> bool {
    var.concretetype().is_some_and(|ty| {
        crate::model::try_getkind(
            &ty,
            supports_floats,
            supports_longlong,
            supports_singlefloats,
        )
        .is_err()
    })
}

fn link_arg_refused(
    arg: &LinkArg,
    supports_floats: bool,
    supports_longlong: bool,
    supports_singlefloats: bool,
) -> bool {
    match arg {
        LinkArg::Value(var) => variable_concretetype_refused(
            var,
            supports_floats,
            supports_longlong,
            supports_singlefloats,
        ),
        LinkArg::Const(constant) => {
            if matches!(
                constant.value,
                ConstValue::Int128(_) | ConstValue::UInt128(_)
            ) {
                return true;
            }
            if matches!(constant.value, ConstValue::Float(_)) && !supports_floats {
                return true;
            }
            constant.concretetype.as_ref().is_some_and(|ty| {
                crate::model::try_getkind(
                    ty,
                    supports_floats,
                    supports_longlong,
                    supports_singlefloats,
                )
                .is_err()
            })
        }
    }
}

/// Whether `history.py getkind(TYPE, ...)` has a register kind for a
/// [`ValueType`], or raises `NotImplementedError` on it.
///
/// `Float` needs `supports_floats`. `SingleFloat` needs
/// `supports_singlefloats` and then banks as an int. `Int128` /
/// `UInt128` are 16 bytes; `supports_longlong` only rescues a primitive
/// whose width is exactly 8, and no [`ValueType`] is that case (`Int`
/// is word-sized). On a 64-bit host an 8-byte `SignedLongLong`
/// concretetype is `Signed` even when `supports_longlong` is false —
/// that check lives in [`crate::model::try_getkind`], not here.
fn value_type_has_kind(
    ty: &ValueType,
    supports_floats: bool,
    supports_longlong: bool,
    supports_singlefloats: bool,
) -> bool {
    let _ = supports_longlong;
    match ty {
        ValueType::Int128 | ValueType::UInt128 => false,
        ValueType::SingleFloat => supports_singlefloats,
        ValueType::Float => supports_floats,
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Bool
        | ValueType::State
        | ValueType::Ref(_)
        | ValueType::Str
        | ValueType::StringBuilder
        | ValueType::Unknown
        | ValueType::Void => true,
    }
}

/// Append the [`ValueType`]s one operation declares to `out`.
///
/// Two surfaces declare one.  Most variants carry it in a `ty` /
/// `item_ty` / `result_ty` field, which is what the `value_type_to_kind`
/// copies read.  The `ConstInt128` / `ConstUInt128` variants carry no
/// such field — the width is in the variant name and the payload is a
/// Rust literal — but they are the op form of upstream's
/// `Constant(value, SignedLongLongLong)` operand, which `policy.py:96-98`
/// reaches through `op.args` and refuses like any other value.  They
/// report the type they materialise, so a graph holding a 128-bit
/// literal is refused here instead of panicking later in
/// `assembler.rs`'s opname formation.
///
/// The match is exhaustive and deliberately carries no wildcard arm: an
/// `OpKind` variant added with a `ValueType` field has to be classified
/// here, and until it is the crate does not compile.  A wildcard would
/// let a new carrier of a 128-bit type pass the policy gate and reach
/// `value_type_to_kind`, which panics rather than declining.
pub fn collect_declared_value_types<'a>(kind: &'a OpKind, out: &mut Vec<&'a ValueType>) {
    // The 128-bit constant variants have no `ValueType` field to borrow
    // from, and `ValueType` owns a `String` so it cannot be promoted to a
    // `&'static` from a `const`.  Name the two shapes once instead.
    static INT128: ValueType = ValueType::Int128;
    static UINT128: ValueType = ValueType::UInt128;
    static SINGLEFLOAT: ValueType = ValueType::SingleFloat;
    static FLOAT: ValueType = ValueType::Float;

    match kind {
        // The type of the value the op reads or writes.
        OpKind::Input { ty, .. }
        | OpKind::ConstSymbolic { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::FieldWrite { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::VableFieldWrite { ty, .. }
        | OpKind::LoadStatic { ty, .. } => out.push(ty),

        // The element type of the array or interior field addressed.
        OpKind::NewArray { item_ty, .. }
        | OpKind::NewArrayClear { item_ty, .. }
        | OpKind::NewListClear { item_ty, .. }
        | OpKind::ArrayRead { item_ty, .. }
        | OpKind::ArrayWrite { item_ty, .. }
        | OpKind::RawLoad { item_ty, .. }
        | OpKind::RawStore { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::InteriorFieldWrite { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. }
        | OpKind::VableArrayWrite { item_ty, .. }
        | OpKind::VableArrayLen { item_ty, .. } => out.push(item_ty),

        // The declared type of the op's result.
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. }
        | OpKind::IsInstance { result_ty, .. } => out.push(result_ty),

        // `policy.py:96-98`'s `for v in op.args: getkind(v.concretetype)`
        // over a `Constant` of the 16-byte primitive.
        OpKind::ConstInt128(_) => out.push(&INT128),
        OpKind::ConstUInt128(_) => out.push(&UINT128),
        // Likewise `Constant(value, SingleFloat)`.  This is the only way
        // the walk sees an `f32` literal: the variant declares no
        // `ValueType` field, and the literal's own type channel is not
        // one this walk reads.
        OpKind::ConstSingleFloat(_) => out.push(&SINGLEFLOAT),
        // `Constant(value, Float)`.
        OpKind::ConstFloat(_) => out.push(&FLOAT),

        // No value type declared.  The remaining constant variants carry
        // a Rust literal whose kind is fixed by the variant name, and the
        // call variants downstream of `jtransform` carry a `result_kind`
        // char that `value_type_to_kind` already produced.
        OpKind::ConstInt(_)
        | OpKind::ConstFnAddr { .. }
        | OpKind::ConstUInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstInternedStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. }
        | OpKind::RawMalloc { .. }
        | OpKind::RawFree { .. }
        | OpKind::ArrayLen { .. }
        | OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::GuardClass { .. }
        | OpKind::VtableMethodPtr { .. }
        | OpKind::VableForce { .. }
        | OpKind::Hint { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::CurrentTraceLength
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. }
        | OpKind::RecordKnownResult { .. }
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::Live
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::NewTuple { .. }
        | OpKind::NewList { .. }
        | OpKind::GetSlice { .. }
        | OpKind::LoweredBlackholeOp { .. } => {}
    }
}

/// `rpython.translator.backendopt.support.find_backedges(graph)`.
///
/// Standard DFS classification: edges from a block back to an ancestor
/// in the current DFS stack are back edges.  Returns the list of back
/// edges as `(from_block, to_block)` pairs.
///
/// Walks the startblock-reachable, non-`dead` closure — the same
/// `iterblocks()` set `contains_unsupported_variable_type` uses.
/// Charon keeps unreachable BBs (orphan `on_unwind` chains) and `dead`
/// stubs in `graph.blocks`; RPython's flow graph never contains them.
fn find_backedges(graph: &FunctionGraph) -> Vec<(usize, usize)> {
    let by_id: std::collections::HashMap<BlockId, &Block> = graph
        .blocks
        .iter()
        .filter(|b| !b.dead)
        .map(|b| (b.id, b))
        .collect();
    let mut backedges = Vec::new();
    let mut seen: HashSet<BlockId> = HashSet::new();
    let mut seeing: HashSet<BlockId> = HashSet::new();
    if !by_id.contains_key(&graph.startblock) {
        return backedges;
    }
    seen.insert(graph.startblock);
    find_backedges_dfs(
        &by_id,
        graph.startblock,
        &mut seen,
        &mut seeing,
        &mut backedges,
    );
    backedges
}

fn find_backedges_dfs(
    by_id: &std::collections::HashMap<BlockId, &Block>,
    block_id: BlockId,
    seen: &mut HashSet<BlockId>,
    seeing: &mut HashSet<BlockId>,
    backedges: &mut Vec<(usize, usize)>,
) {
    seeing.insert(block_id);
    let Some(block) = by_id.get(&block_id) else {
        seeing.remove(&block_id);
        return;
    };
    // `iterblocks` derives the successor set from `Block.exits` only;
    // final blocks (`exits == ()`) have no outgoing targets. Skip a
    // `dead` / missing target — it is a Charon CFG artefact, not a
    // source-level back-edge.
    for target in block.exits.iter().map(|link| link.target) {
        if !by_id.contains_key(&target) {
            continue;
        }
        if seen.contains(&target) {
            if seeing.contains(&target) {
                backedges.push((block_id.0, target.0));
            }
        } else {
            seen.insert(target);
            find_backedges_dfs(by_id, target, seen, seeing, backedges);
        }
    }
    seeing.remove(&block_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FunctionGraph;

    fn make_func(name: &str, hints: Vec<&str>) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        for hint in hints {
            graph.push_hint(hint);
        }
        graph
    }

    #[test]
    fn default_look_inside_function_returns_true() {
        let policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec![]);
        assert!(policy.look_inside_function(&f));
    }

    #[test]
    fn elidable_hint_rejects_function() {
        let policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec!["elidable"]);
        assert!(policy._reject_function(&f));
    }

    #[test]
    fn jit_look_inside_overrides_default() {
        let mut policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec!["jit_look_inside=false"]);
        assert!(!policy.look_inside_graph(&f));
    }

    #[test]
    fn stop_at_x_policy_excludes_named_funcs() {
        let policy = StopAtXPolicy::new(vec!["stop_me".into()]);
        let stop = make_func("stop_me", vec![]);
        let other = make_func("other", vec![]);
        assert!(!policy.look_inside_function(&stop));
        assert!(policy.look_inside_function(&other));
    }

    /// `policy.py:71-83`: a graph the codewriter refuses to look inside must
    /// not be `access_directly`. Pins the carrier as well as the gate: the
    /// flag reaches here on the `FunctionGraph`.
    #[test]
    #[should_panic(expected = "access_directly on a function which we don't see")]
    fn access_directly_on_a_loopy_graph_aborts() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        g.access_directly = true;
        policy.look_inside_graph(&g);
    }

    /// The same graph without the flag is an ordinary decline, not an abort.
    #[test]
    fn a_loopy_graph_without_the_flag_only_declines() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        assert!(!policy.look_inside_graph(&g));
    }

    /// `policy.py`: a graph holding a value `history.getkind`
    /// refuses is refused here, not carried to the codewriter — where
    /// `value_type_to_kind` panics rather than declining.
    #[test]
    fn a_128_bit_result_type_is_unsupported() {
        let mut g = FunctionGraph::new("wide");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int128,
                class_root: None,
            },
            true,
        );
        assert!(contains_unsupported_variable_type(&g, true, true, true));
    }

    /// The 128-bit constants declare their width through the variant
    /// name rather than a `ValueType` field, and are refused all the
    /// same — `policy.py:96-98` reaches upstream's equivalent
    /// `Constant(value, SignedLongLongLong)` through `op.args`.
    #[test]
    fn a_128_bit_constant_is_unsupported() {
        let mut g = FunctionGraph::new("wide_const");
        let entry = g.startblock;
        g.push_op_var(entry, OpKind::ConstUInt128(1), true);
        assert!(contains_unsupported_variable_type(&g, true, true, true));
    }

    #[test]
    fn a_128_bit_link_constant_is_unsupported() {
        let mut g = FunctionGraph::new("wide_link_const");
        let entry = g.startblock;
        let (target, _) = g.create_block_with_arg_vars(1);
        g.block_mut(entry).exits = vec![
            crate::model::Link::new_mixed(
                vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                    ConstValue::UInt128(1),
                ))],
                target,
                None,
            )
            .with_prevblock(entry),
        ];
        assert!(contains_unsupported_variable_type(&g, true, true, true));
    }

    /// `history.py:58-61`: a singlefloat is refused unless the CPU
    /// supports one, and pyre's effective answer is the base backend's
    /// `False` (`backend/model.py:20`) because `warmspot.py:250` has no
    /// port. The flag is honoured rather than hardcoded, so both answers
    /// are asserted here — the `true` leg is what upstream's x86 gets
    /// (`backend/x86/runner.py`), and reaching it in pyre would need
    /// the singlefloat casts first.
    #[test]
    fn a_singlefloat_is_unsupported_unless_the_cpu_supports_one() {
        let mut g = FunctionGraph::new("narrow_float");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::SingleFloat,
                class_root: None,
            },
            true,
        );
        assert!(contains_unsupported_variable_type(&g, true, true, false));
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
    }

    /// The `f32` literal is the carrier with no `ValueType` field
    /// anywhere on its path — it declares none, and nothing else in a
    /// literal-only graph declares one either. `policy.py:96-98` reaches
    /// upstream's `Constant(value, SingleFloat)` through `op.args`, and
    /// this is how the walk reaches its op form.
    #[test]
    fn a_singlefloat_constant_is_unsupported() {
        let mut g = FunctionGraph::new("narrow_float_const");
        let entry = g.startblock;
        g.push_op_var(entry, OpKind::ConstSingleFloat(1.0f32.to_bits()), true);
        assert!(contains_unsupported_variable_type(&g, true, true, false));
    }

    /// Word-sized values keep their kinds. `Float` needs `supports_floats`.
    #[test]
    fn ordinary_value_types_are_supported() {
        let mut g = FunctionGraph::new("narrow");
        let entry = g.startblock;
        for ty in [
            ValueType::Int,
            ValueType::Unsigned,
            ValueType::Bool,
            ValueType::Float,
            ValueType::Void,
            ValueType::Ref(None),
        ] {
            g.push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty,
                    class_root: None,
                },
                true,
            );
        }
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
        assert!(contains_unsupported_variable_type(&g, false, false, false));

        let mut ints = FunctionGraph::new("ints");
        ints.push_op_var(
            ints.startblock,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int,
                class_root: None,
            },
            true,
        );
        assert!(!contains_unsupported_variable_type(
            &ints, false, false, false
        ));
    }

    /// The gate that consumes it: a graph the codewriter cannot give a
    /// register kind is declined rather than reaching
    /// `value_type_to_kind`.
    #[test]
    fn look_inside_graph_declines_a_128_bit_graph() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("wide");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::UInt128,
                class_root: None,
            },
            true,
        );
        assert!(!policy.look_inside_graph(&g));
    }

    #[test]
    fn unroll_safe_disables_loop_rejection() {
        let mut policy = DefaultJitPolicy::new();
        // Build a graph with a self-loop on block 0.
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        // Without `unroll_safe`, the loop disqualifies the graph.
        assert!(!policy.look_inside_graph(&g));
        assert!(policy.state().unsafe_loopy_graphs.contains("loopy"));

        // With `unroll_safe`, the loop is ignored.
        let mut unroll_safe = g;
        unroll_safe.push_hint("unroll_safe");
        assert!(policy.look_inside_graph(&unroll_safe));
    }

    #[test]
    fn dont_look_inside_hint_overrides_default_to_false() {
        // test_policy.py `test_dont_look_inside`.
        let mut policy = DefaultJitPolicy::new();
        let f = make_func("h", vec!["dont_look_inside"]);
        assert!(!policy.look_inside_graph(&f));
    }

    #[test]
    fn jit_look_inside_hint_overrides_subclass_to_true() {
        // test_policy.py `test_look_inside`.
        struct NoPolicy(JitPolicyState);
        impl JitPolicy for NoPolicy {
            fn state(&self) -> &JitPolicyState {
                &self.0
            }
            fn state_mut(&mut self) -> &mut JitPolicyState {
                &mut self.0
            }
            fn look_inside_function(&self, _: &FunctionGraph) -> bool {
                false
            }
        }
        let mut policy = NoPolicy(JitPolicyState::new());
        let h1 = make_func("h1", vec![]);
        let h2 = make_func("h2", vec!["jit_look_inside"]);
        assert!(!policy.look_inside_graph(&h1));
        assert!(policy.look_inside_graph(&h2));
    }

    #[test]
    fn dump_unsafe_loops_writes_sorted_names() {
        let mut state = JitPolicyState::new();
        state.unsafe_loopy_graphs.insert("zeta".into());
        state.unsafe_loopy_graphs.insert("alpha".into());
        state.unsafe_loopy_graphs.insert("mu".into());
        let tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        let path = tmp.path();
        state.dump_unsafe_loops(path).expect("write");
        let body = std::fs::read_to_string(path).expect("read");
        assert_eq!(body, "alpha\nmu\nzeta\n");
    }

    #[test]
    fn find_backedges_detects_self_loop() {
        let mut g = FunctionGraph::new("loop");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        let edges = find_backedges(&g);
        assert_eq!(edges, vec![(entry.0, entry.0)]);
    }

    /// Harvested `unroll_safe` lands on `FunctionGraph.hints`.
    #[test]
    fn unroll_safe_on_graph_hints_opts_in_a_loop() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        g.push_hint("unroll_safe");
        assert!(policy.look_inside_graph(&g));
    }

    /// `iterblocks()` never yields a block the startblock cannot reach.
    /// A self-loop on an orphan Charon BB is not a source-level loop.
    #[test]
    fn find_backedges_ignores_unreachable_self_loop() {
        let mut g = FunctionGraph::new("linear");
        let entry = g.startblock;
        let sink = g.create_block();
        g.set_goto(entry, sink, Vec::new());
        let orphan = g.create_block();
        g.set_goto(orphan, orphan, Vec::new());
        assert!(find_backedges(&g).is_empty());
        let mut policy = DefaultJitPolicy::new();
        assert!(policy.look_inside_graph(&g));
    }

    /// A `dead` stub (orphan `on_unwind` cleanup) may still have a
    /// residual self-edge. `iterblocks` never sees it.
    #[test]
    fn find_backedges_ignores_dead_block_self_loop() {
        let mut g = FunctionGraph::new("dead_loop");
        let entry = g.startblock;
        let dead = g.create_block();
        g.set_goto(entry, dead, Vec::new());
        g.set_goto(dead, dead, Vec::new());
        g.block_mut(dead).dead = true;
        assert!(find_backedges(&g).is_empty());
        let mut policy = DefaultJitPolicy::new();
        assert!(policy.look_inside_graph(&g));
    }

    /// `rpython.rtyper.module.` is
    /// `majit_translate::translator::rtyper::module::`. `ll_math` is under
    /// `lltypesystem.module` and stays visible. An empty module is `'?'`.
    #[test]
    fn rtyper_module_prefix_rejects_and_ll_math_does_not() {
        let policy = DefaultJitPolicy::new();
        let mut helper = FunctionGraph::new("ll_os");
        helper.func.module = Some("majit_translate::translator::rtyper::module::ll_os".to_string());
        assert!(policy._reject_function(&helper));

        let mut ll_math = FunctionGraph::new("ll_math_sqrt");
        ll_math.func.module =
            Some("majit_translate::translator::rtyper::lltypesystem::module::ll_math".to_string());
        assert!(!policy._reject_function(&ll_math));

        let mut empty = FunctionGraph::new("no_mod");
        empty.func.module = Some(String::new());
        assert!(!policy._reject_function(&empty));
        assert!(!policy._reject_function(&FunctionGraph::new("absent")));
    }

    /// The first `_jit_look_inside_` token wins, matching the old scan.
    #[test]
    fn the_first_look_inside_hint_wins() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("h");
        g.push_hint("jit_look_inside");
        g.push_hint("dont_look_inside");
        assert_eq!(g.func.jit_look_inside, Some(true));
        assert!(policy.look_inside_graph(&g));
    }

    #[test]
    fn a_float_constant_needs_supports_floats() {
        let mut g = FunctionGraph::new("float_const");
        g.push_op_var(g.startblock, OpKind::ConstFloat(1.0f64.to_bits()), true);
        assert!(contains_unsupported_variable_type(&g, false, true, true));
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
    }

    #[test]
    fn a_float_link_constant_needs_supports_floats() {
        let mut g = FunctionGraph::new("float_link");
        let entry = g.startblock;
        let (target, _) = g.create_block_with_arg_vars(1);
        g.block_mut(entry).exits = vec![
            crate::model::Link::new_mixed(
                vec![LinkArg::Const(crate::flowspace::model::Constant::new(
                    ConstValue::float(1.0),
                ))],
                target,
                None,
            )
            .with_prevblock(entry),
        ];
        assert!(contains_unsupported_variable_type(&g, false, true, true));
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
    }

    /// `getkind` on a 128-bit concretetype raises. The policy residualizes
    /// the graph instead of panicking.
    #[test]
    fn a_128_bit_concretetype_is_unsupported() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut g = FunctionGraph::new("wide_ct");
        let var = crate::flowspace::model::Variable::named("v");
        var.set_concretetype(Some(LowLevelType::SignedLongLongLong));
        g.block_mut(g.startblock).inputargs.push(var);
        assert!(contains_unsupported_variable_type(&g, true, true, true));
        let mut policy = DefaultJitPolicy::new();
        assert!(!policy.look_inside_graph(&g));
    }

    /// On 64-bit, `sizeof(SignedLongLong) == sizeof(Signed)`, so
    /// `supports_longlong == false` does not refuse it.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn signed_long_long_concretetype_is_signed_on_64_bit() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut g = FunctionGraph::new("ll");
        let var = crate::flowspace::model::Variable::named("v");
        var.set_concretetype(Some(LowLevelType::SignedLongLong));
        g.block_mut(g.startblock).inputargs.push(var);
        assert!(!contains_unsupported_variable_type(&g, true, false, true));
    }

    /// On 32-bit the 8-byte longlong takes the float slot only when
    /// `supports_longlong` is set.
    #[cfg(target_pointer_width = "32")]
    #[test]
    fn signed_long_long_concretetype_needs_longlong_support_on_32_bit() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut g = FunctionGraph::new("ll");
        let var = crate::flowspace::model::Variable::named("v");
        var.set_concretetype(Some(LowLevelType::SignedLongLong));
        g.block_mut(g.startblock).inputargs.push(var);
        assert!(contains_unsupported_variable_type(&g, true, false, true));
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
    }
}
