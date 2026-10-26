//! Port of `rpython/translator/backendopt/writeanalyze.py`.
//!
//! `WriteAnalyzer` is the [`super::graphanalyze::GraphAnalyzer`]
//! subclass that computes, for each graph, the set of memory writes a
//! call can perform — modelled as a set of effect tuples. Its
//! `ReadWriteAnalyzer` subclass additionally tracks the reads.
//! Upstream's consumer is the JIT codewriter's `EffectInfo` construction
//! (`jit/codewriter/call.py:38` builds a `ReadWriteAnalyzer`; `:320`
//! `getcalldescr` queries it). Pyre's `getcalldescr` still uses the flat
//! `analyze_readwrite` scanner in `codewriter/call.rs`; wiring this
//! flowspace analyzer in its place is unported. Until then this module
//! is published as a parity sibling
//! alongside [`super::canraise`], [`super::collectanalyze`] and
//! [`super::gilanalysis`].
//!
//! Two divergences from a strict tuple port, both forced by Rust's
//! type system and documented here:
//!
//! * Upstream stores effects as heterogeneous Python tuples in a
//!   `frozenset`. Rust models them as the [`Effect`] enum and the set
//!   as [`WriteEffects`] (`Top` sentinel + `HashSet<Effect>`). Both
//!   `LowLevelType` and `ConstValue` implement `Eq + Hash`, so the
//!   `HashSet` reproduces `frozenset`'s order-insensitive equality and
//!   automatic dedup exactly (`writeanalyze.py`).
//! * Upstream's `graphinfo` is either `None` or a `FreshMallocs`. The
//!   `GraphAnalyzer` trait threads a concrete `I: GraphInfo`, so the
//!   port uses `I = Option<FreshMallocs>`: `None` is upstream's `None`
//!   (the default when `analyze` runs without a graph walk),
//!   `Some(fm)` is `compute_graph_info`'s `FreshMallocs(graph)`.

// `TYPE` field/binding names mirror upstream's effect-tuple element
// names (`writeanalyze.py analyze_simple_operation` `op.args[0].concretetype` slot, llmemory
// `ofs.TYPE`); keep the upstream spelling rather than snake_case.
#![allow(non_snake_case)]

use crate::flowspace::model::{
    ConstValue, FunctionGraph, GraphRef, Hlvalue, SpaceOperation, Variable,
};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{Dependency, GraphAnalyzer, GraphInfo};
use crate::translator::rtyper::lltypesystem::llmemory::AddressOffset;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr};
use crate::translator::translator::TranslationContext;
use std::collections::HashSet;

/// One element of the write/read effect set. Upstream stores Python
/// tuples whose first element is a kind string (`"struct"`, `"array"`,
/// `"interiorfield"`, and the `"read"`-prefixed variants); the Rust
/// port lifts that string into the enum discriminant.
///
/// The `TYPE` slot carries `op.args[0].concretetype` (or `lltype.Ptr(T)`
/// for the `gc_{load,store}_indexed` offset path). Upstream reads
/// `.concretetype` unconditionally and builds `lltype.Ptr(T)`
/// unconditionally — neither yields `None` on a valid rtyped graph — so
/// the slot is a plain `LowLevelType`; the producers fail loud rather
/// than synthesising a `None`-typed effect.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    /// `("struct", op.args[0].concretetype, op.args[1].value)`.
    Struct {
        TYPE: LowLevelType,
        fieldname: ConstValue,
    },
    /// `("array", TYPE)`.
    Array { TYPE: LowLevelType },
    /// `("interiorfield", TYPE, fieldname)`.
    InteriorField {
        TYPE: LowLevelType,
        fieldname: ConstValue,
    },
    /// `("readstruct", op.args[0].concretetype, op.args[1].value)`.
    ReadStruct {
        TYPE: LowLevelType,
        fieldname: ConstValue,
    },
    /// `("readarray", op.args[0].concretetype)`.
    ReadArray { TYPE: LowLevelType },
    /// `("readinteriorfield", op.args[0].concretetype, fieldname)`.
    ReadInteriorField {
        TYPE: LowLevelType,
        fieldname: ConstValue,
    },
}

/// The result lattice. Upstream uses `top_set = object()` as a sentinel
/// distinct from any `frozenset`, and `empty_set = frozenset()` /
/// `set()` for the bottom/builder. `Set` carries a `HashSet<Effect>` —
/// order-insensitive equality and automatic dedup, matching `frozenset`;
/// `Top` is the `top_set` sentinel.
#[derive(Clone, Debug, PartialEq)]
pub enum WriteEffects {
    /// `top_set` (`writeanalyze.py`) — "can write anything".
    Top,
    /// `frozenset(...)` / `set()` of [`Effect`]s.
    Set(HashSet<Effect>),
}

impl crate::translator::backendopt::graphanalyze::AnalyzerResult for WriteEffects {
    /// `bottom_result` (`writeanalyze.py`): `empty_set`.
    fn bottom_result() -> Self {
        WriteEffects::Set(HashSet::new())
    }

    /// `top_result` (`writeanalyze.py`): `top_set`.
    fn top_result() -> Self {
        WriteEffects::Top
    }

    /// `is_top_result` (`writeanalyze.py`): `result is top_set`.
    fn is_top_result(result: &Self) -> bool {
        matches!(result, WriteEffects::Top)
    }

    /// `result_builder` (`writeanalyze.py`): `set()`.
    fn result_builder() -> Self {
        WriteEffects::Set(HashSet::new())
    }

    /// `add_to_result` (`writeanalyze.py`):
    ///
    /// ```python
    /// def add_to_result(self, result, other):
    ///     if other is top_set:
    ///         return top_set
    ///     result.update(other)
    ///     return result
    /// ```
    fn add_to_result(result: Self, other: Self) -> Self {
        if matches!(other, WriteEffects::Top) {
            return WriteEffects::Top;
        }
        match result {
            WriteEffects::Top => WriteEffects::Top,
            WriteEffects::Set(mut acc) => {
                if let WriteEffects::Set(o) = other {
                    acc.extend(o);
                }
                WriteEffects::Set(acc)
            }
        }
    }

    /// `finalize_builder` (`writeanalyze.py`):
    ///
    /// ```python
    /// def finalize_builder(self, result):
    ///     if result is top_set:
    ///         return result
    ///     return frozenset(result)
    /// ```
    ///
    /// The `HashSet` builder is already deduped, so freezing is the
    /// identity on the `Set` variant.
    fn finalize_builder(result: Self) -> Self {
        result
    }

    /// `join_two_results` (`writeanalyze.py`):
    ///
    /// ```python
    /// def join_two_results(self, result1, result2):
    ///     if result1 is top_set or result2 is top_set:
    ///         return top_set
    ///     return result1.union(result2)
    /// ```
    fn join_two_results(result1: Self, result2: Self) -> Self {
        if matches!(result1, WriteEffects::Top) || matches!(result2, WriteEffects::Top) {
            return WriteEffects::Top;
        }
        let WriteEffects::Set(mut acc) = result1 else {
            unreachable!()
        };
        if let WriteEffects::Set(o) = result2 {
            acc.extend(o);
        }
        WriteEffects::Set(acc)
    }
}

/// `class FreshMallocs(object)` at `writeanalyze.py`. Tracks
/// which variables in a graph hold a pointer to freshly-malloc'd memory
/// that has not escaped, so writes through them can be ignored.
#[derive(Clone, Default)]
pub struct FreshMallocs {
    /// `self.nonfresh` — variables proven *not* to be a fresh malloc.
    nonfresh: HashSet<Variable>,
    /// `self.allvariables` — every variable defined in the graph.
    allvariables: HashSet<Variable>,
}

/// `Option<FreshMallocs>` is the graph-info lattice: `None` mirrors
/// upstream's `graphinfo is None`, `Some(_)` the computed
/// `FreshMallocs(graph)`.
impl GraphInfo for Option<FreshMallocs> {}

impl FreshMallocs {
    /// `FreshMallocs.__init__(self, graph)` (`writeanalyze.py`).
    pub fn new(graph: &FunctionGraph) -> Self {
        let mut fm = FreshMallocs {
            // `self.nonfresh = set(graph.getargs())`.
            nonfresh: graph
                .getargs()
                .into_iter()
                .filter_map(as_variable)
                .collect(),
            allvariables: HashSet::new(),
        };
        // `pendingblocks = list(graph.iterblocks())`.
        let mut pendingblocks = graph.iterblocks();
        // `for block in pendingblocks: self.allvariables.update(block.inputargs)`.
        for block in &pendingblocks {
            for v in block
                .borrow()
                .inputargs
                .iter()
                .cloned()
                .filter_map(as_variable)
            {
                fm.allvariables.insert(v);
            }
        }
        // `pendingblocks.reverse()` — the `while ... pop()` then drains
        // it in original `iterblocks` order.
        pendingblocks.reverse();
        while let Some(block) = pendingblocks.pop() {
            for op in &block.borrow().operations {
                // `self.allvariables.add(op.result)`.
                if let Some(r) = as_variable(op.result.clone()) {
                    fm.allvariables.insert(r);
                }
                // `if op.opname in ('malloc', 'malloc_varsize', 'new'): continue`.
                if op.opname == "malloc" || op.opname == "malloc_varsize" || op.opname == "new" {
                    continue;
                }
                // `elif op.opname in ('cast_pointer', 'same_as'):
                //      if self.is_fresh_malloc(op.args[0]): continue`.
                if (op.opname == "cast_pointer" || op.opname == "same_as")
                    && let Some(arg0) = op.args.first()
                    && fm.is_fresh_malloc(arg0)
                {
                    continue;
                }
                // `self.nonfresh.add(op.result)`.
                if let Some(r) = as_variable(op.result.clone()) {
                    fm.nonfresh.insert(r);
                }
            }
            // `for link in block.exits:` — snapshot so the block borrow
            // is dropped before `link.target.inputargs` is read.
            let exits: Vec<_> = block.borrow().exits.to_vec();
            for link in exits {
                let link = link.borrow();
                // `self.nonfresh.update(link.getextravars())` /
                // `self.allvariables.update(link.getextravars())`.
                for v in link.getextravars() {
                    fm.nonfresh.insert(v.clone());
                    fm.allvariables.insert(v);
                }
                let prevlen = fm.nonfresh.len();
                // `for v1, v2 in zip(link.args, link.target.inputargs):`.
                let Some(target) = link.target.as_ref() else {
                    continue;
                };
                let target_inputargs = target.borrow().inputargs.clone();
                for (v1, v2) in link.args.iter().zip(target_inputargs.iter()) {
                    // `if not self.is_fresh_malloc(v1): self.nonfresh.add(v2)`.
                    // `v1` is `Option<Hlvalue>` (transient undefined-local
                    // sentinel); a `None` slot is conservatively not-fresh.
                    let v1_fresh = v1.as_ref().is_some_and(|hv| fm.is_fresh_malloc(hv));
                    if !v1_fresh && let Some(v2v) = as_variable(v2.clone()) {
                        fm.nonfresh.insert(v2v);
                    }
                }
                // `if len(self.nonfresh) > prevlen: pendingblocks.append(link.target)`.
                if fm.nonfresh.len() > prevlen {
                    pendingblocks.push(target.clone());
                }
            }
        }
        fm
    }

    /// `is_fresh_malloc(self, v)` (`writeanalyze.py`):
    ///
    /// ```python
    /// def is_fresh_malloc(self, v):
    ///     if not isinstance(v, Variable):
    ///         return False
    ///     assert v in self.allvariables
    ///     return v not in self.nonfresh
    /// ```
    pub fn is_fresh_malloc(&self, v: &Hlvalue) -> bool {
        let Hlvalue::Variable(var) = v else {
            return false;
        };
        assert!(
            self.allvariables.contains(var),
            "is_fresh_malloc: variable not in allvariables"
        );
        !self.nonfresh.contains(var)
    }
}

/// `Hlvalue::Variable` projection used wherever upstream relies on a
/// list/set of `Variable`s ignoring `Constant`s.
fn as_variable(hv: Hlvalue) -> Option<Variable> {
    match hv {
        Hlvalue::Variable(v) => Some(v),
        Hlvalue::Constant(_) => None,
    }
}

/// `op.args[i].concretetype` for an `Hlvalue` arg
/// (`finalizer.py` `analyze_simple_operation`). Upstream reaches `.concretetype`
/// directly; a missing arg slot or an absent `concretetype` is a
/// malformed/un-rtyped graph, so the port fails loud (IndexError /
/// AttributeError parity) rather than synthesising a `None`-typed effect.
fn arg_concretetype(op: &SpaceOperation, i: usize) -> LowLevelType {
    let ct = match op.args.get(i) {
        Some(Hlvalue::Variable(v)) => v.concretetype(),
        Some(Hlvalue::Constant(c)) => c.concretetype.clone(),
        None => panic!("op.args[{i}] missing"),
    };
    ct.unwrap_or_else(|| panic!("op.args[{i}].concretetype is None (un-rtyped graph)"))
}

/// The `AddressOffset` carried by an offset constant's value. Upstream's
/// `_get_effect_for_offset` `assert False`s on anything that is not a
/// known offset subclass; a non-offset value here is that same fail-loud
/// case (`writeanalyze.py`).
fn as_address_offset(v: &ConstValue) -> &AddressOffset {
    match v {
        ConstValue::AddressOffset(ofs) => ofs,
        _ => panic!("implement me"),
    }
}

/// The `WriteAnalyzer` instance-method surface (`writeanalyze.py`).
/// Upstream these are methods on the `WriteAnalyzer` class, inherited by
/// `ReadWriteAnalyzer`; the port exposes them as trait default methods so
/// both analyzers share one body, mirroring the Python inheritance. They
/// take `&self` even though the bodies are pure, keeping the method
/// surface 1:1 with upstream rather than scattering free functions.
trait WriteAnalyzerMethods {
    /// `_getinteriorname(self, op)` (`writeanalyze.py`):
    ///
    /// ```python
    /// def _getinteriorname(self, op):
    ///     if (isinstance(op.args[1], Constant) and
    ///         isinstance(op.args[1].value, str)):
    ///         return op.args[1].value
    ///     return op.args[2].value
    /// ```
    fn getinteriorname(&self, op: &SpaceOperation) -> ConstValue {
        if let Some(Hlvalue::Constant(c)) = op.args.get(1) {
            // `isinstance(op.args[1].value, str)` — Python 2 `str` is
            // bytes-only, so `ConstValue::as_pystr` matches `ByteStr`
            // and rejects `UniStr` (Python 2 `unicode`), per the
            // `flowspace/model.rs` `as_pystr` rule.
            if c.value.as_pystr().is_some() {
                return c.value.clone();
            }
        }
        // `return op.args[2].value` — `.value` exists only on a Constant
        // upstream, so a non-Constant slot fails loud (AttributeError parity).
        arg_value(op, 2)
    }

    /// `_array_result(self, TYPE)` (`writeanalyze.py`):
    /// `frozenset([("array", TYPE)])`.
    fn array_result(&self, TYPE: LowLevelType) -> WriteEffects {
        WriteEffects::Set(HashSet::from([Effect::Array { TYPE }]))
    }

    /// `_interiorfield_result(self, TYPE, fieldname)` (`writeanalyze.py`):
    /// `frozenset([("interiorfield", TYPE, fieldname)])`.
    fn interiorfield_result(&self, TYPE: LowLevelType, fieldname: ConstValue) -> WriteEffects {
        WriteEffects::Set(HashSet::from([Effect::InteriorField { TYPE, fieldname }]))
    }

    /// `_gc_store_indexed_result(self, op)` (`writeanalyze.py`):
    ///
    /// ```python
    /// def _gc_store_indexed_result(self, op):
    ///     base_ofs = op.args[4].value
    ///     effect = self._get_effect_for_offset(base_ofs)
    ///     return frozenset([effect])
    /// ```
    fn gc_store_indexed_result(&self, op: &SpaceOperation) -> WriteEffects {
        let base_ofs = arg_value(op, 4);
        let effect = self.get_effect_for_offset(as_address_offset(&base_ofs), false);
        WriteEffects::Set(HashSet::from([effect]))
    }

    /// `_get_effect_for_offset(self, ofs, prefix='')`
    /// (`writeanalyze.py`). `read` selects the `'read'` prefix.
    fn get_effect_for_offset(&self, ofs: &AddressOffset, read: bool) -> Effect {
        match ofs {
            // `if isinstance(ofs, llmemory.CompositeOffset):`.
            AddressOffset::CompositeOffset(sub_offsets) => {
                // `effect = self._get_effect_for_offset(sub_offsets[0], prefix)`.
                let effect = self.get_effect_for_offset(&sub_offsets.offsets[0], read);
                // `for sub_ofs in sub_offsets[1:]:` — only ArrayItemsOffset
                // is tolerated (mid-array reads === beginning reads).
                for sub_ofs in &sub_offsets.offsets[1..] {
                    match sub_ofs {
                        AddressOffset::ArrayItemsOffset(_) => {}
                        _ => panic!("implement me"),
                    }
                }
                effect
            }
            // `elif isinstance(ofs, llmemory.FieldOffset):`.
            AddressOffset::FieldOffset(offset) => {
                // `return (prefix + 'interiorfield', lltype.Ptr(T), ofs.fldname)`.
                // `lltype.Ptr(T)` raises `TypeError` on a non-container `T`;
                // the port fails loud rather than lowering the failure to None.
                let ptr_type = ptr_to(&offset.TYPE);
                let fieldname = ConstValue::byte_str(offset.fldname.as_bytes());
                if read {
                    Effect::ReadInteriorField {
                        TYPE: ptr_type,
                        fieldname,
                    }
                } else {
                    Effect::InteriorField {
                        TYPE: ptr_type,
                        fieldname,
                    }
                }
            }
            // `elif isinstance(ofs, llmemory.ArrayItemsOffset):`.
            AddressOffset::ArrayItemsOffset(offset) => {
                // `return (prefix + 'array', lltype.Ptr(ofs.TYPE))`.
                let ptr_type = ptr_to(&offset.0);
                if read {
                    Effect::ReadArray { TYPE: ptr_type }
                } else {
                    Effect::Array { TYPE: ptr_type }
                }
            }
            // `else: assert False, 'implement me'`.
            _ => panic!("implement me"),
        }
    }

    /// `WriteAnalyzer.analyze_simple_operation(self, op, graphinfo)`
    /// (`writeanalyze.py`). Named distinctly from the trait method
    /// so `ReadWriteAnalyzer` can issue the upstream
    /// `WriteAnalyzer.analyze_simple_operation(self, op, graphinfo)`
    /// super-call (`writeanalyze.py`).
    fn write_analyze_simple_operation(
        &self,
        op: &SpaceOperation,
        graphinfo: &Option<FreshMallocs>,
    ) -> WriteEffects {
        // `graphinfo is None or not graphinfo.is_fresh_malloc(op.args[0])`.
        let not_fresh = |op: &SpaceOperation| match graphinfo {
            None => true,
            Some(fm) => match op.args.first() {
                Some(arg0) => !fm.is_fresh_malloc(arg0),
                None => true,
            },
        };
        match op.opname.as_str() {
            // `if op.opname == "setfield":`.
            "setfield" => {
                if not_fresh(op) {
                    // `frozenset([("struct", op.args[0].concretetype, op.args[1].value)])`.
                    return WriteEffects::Set(HashSet::from([Effect::Struct {
                        TYPE: arg_concretetype(op, 0),
                        fieldname: arg_value(op, 1),
                    }]));
                }
            }
            // `elif op.opname == "setarrayitem":`.
            "setarrayitem" => {
                if not_fresh(op) {
                    return self.array_result(arg_concretetype(op, 0));
                }
            }
            // `elif op.opname == "setinteriorfield":`.
            "setinteriorfield" => {
                if not_fresh(op) {
                    let name = self.getinteriorname(op);
                    return self.interiorfield_result(arg_concretetype(op, 0), name);
                }
            }
            // `elif op.opname == "gc_store_indexed":`.
            "gc_store_indexed" => {
                if not_fresh(op) {
                    return self.gc_store_indexed_result(op);
                }
            }
            // `elif op.opname == 'gc_add_memory_pressure': return top_set`.
            "gc_add_memory_pressure" => {
                return WriteEffects::Top;
            }
            _ => {}
        }
        // `return empty_set`.
        WriteEffects::Set(HashSet::new())
    }
}

/// `lltype.Ptr(T)` for an effect-offset container type. Raises upstream
/// (`TypeError: cannot make a pointer to ...`) on a non-container `T`;
/// the port matches that fail-loud contract.
fn ptr_to(TYPE: &LowLevelType) -> LowLevelType {
    match Ptr::from_container_type(TYPE.clone()) {
        Ok(ptr) => LowLevelType::from(ptr),
        Err(e) => panic!("lltype.Ptr({TYPE:?}): {e}"),
    }
}

/// `op.args[i].value` for a `Constant` arg. Upstream reaches `.value`
/// directly, which `AttributeError`s on a `Variable`; the port mirrors
/// that fail-loud contract rather than silently substituting a default.
fn arg_value(op: &SpaceOperation, i: usize) -> ConstValue {
    match op.args.get(i) {
        Some(Hlvalue::Constant(c)) => c.value.clone(),
        Some(Hlvalue::Variable(_)) => {
            panic!("op.args[{i}].value read on a Variable (expected Constant)")
        }
        None => panic!("op.args[{i}] missing"),
    }
}

/// `class WriteAnalyzer(graphanalyze.GraphAnalyzer)` at
/// `writeanalyze.py`.
pub struct WriteAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py`).
    analyzed_calls: UnionFind<usize, Dependency<WriteEffects>>,
}

impl<'t> WriteAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(WriteEffects::Set(HashSet::new()))),
        }
    }
}

/// `WriteAnalyzer`'s own helper-method surface (default bodies).
impl WriteAnalyzerMethods for WriteAnalyzer<'_> {}

impl<'t> GraphAnalyzer<WriteEffects, Option<FreshMallocs>> for WriteAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<WriteEffects>> {
        &mut self.analyzed_calls
    }

    /// `compute_graph_info(self, graph)` (`writeanalyze.py`):
    /// `return FreshMallocs(graph)`.
    fn compute_graph_info(&mut self, graph: &GraphRef) -> Option<FreshMallocs> {
        Some(FreshMallocs::new(&graph.borrow()))
    }

    fn analyze_simple_operation(
        &mut self,
        op: &SpaceOperation,
        graphinfo: &Option<FreshMallocs>,
    ) -> WriteEffects {
        self.write_analyze_simple_operation(op, graphinfo)
    }
}

/// `class ReadWriteAnalyzer(WriteAnalyzer)` at `writeanalyze.py`.
pub struct ReadWriteAnalyzer<'t> {
    translator: &'t TranslationContext,
    analyzed_calls: UnionFind<usize, Dependency<WriteEffects>>,
}

impl<'t> ReadWriteAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(WriteEffects::Set(HashSet::new()))),
        }
    }

    /// `_gc_load_indexed_result(self, op)` (`writeanalyze.py`):
    ///
    /// ```python
    /// def _gc_load_indexed_result(self, op):
    ///     base_offset = op.args[3].value
    ///     effect = self._get_effect_for_offset(base_offset, prefix='read')
    ///     return frozenset([effect])
    /// ```
    ///
    /// `ReadWriteAnalyzer`-only method; the offset dispatch it calls
    /// (`get_effect_for_offset`) is inherited via [`WriteAnalyzerMethods`].
    fn gc_load_indexed_result(&self, op: &SpaceOperation) -> WriteEffects {
        let base_offset = arg_value(op, 3);
        let effect = self.get_effect_for_offset(as_address_offset(&base_offset), true);
        WriteEffects::Set(HashSet::from([effect]))
    }
}

/// Inherits `WriteAnalyzer`'s helper methods (`ReadWriteAnalyzer(WriteAnalyzer)`).
impl WriteAnalyzerMethods for ReadWriteAnalyzer<'_> {}

impl<'t> GraphAnalyzer<WriteEffects, Option<FreshMallocs>> for ReadWriteAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<WriteEffects>> {
        &mut self.analyzed_calls
    }

    /// Inherited from `WriteAnalyzer.compute_graph_info`.
    fn compute_graph_info(&mut self, graph: &GraphRef) -> Option<FreshMallocs> {
        Some(FreshMallocs::new(&graph.borrow()))
    }

    /// `analyze_simple_operation(self, op, graphinfo)`
    /// (`writeanalyze.py`).
    fn analyze_simple_operation(
        &mut self,
        op: &SpaceOperation,
        graphinfo: &Option<FreshMallocs>,
    ) -> WriteEffects {
        match op.opname.as_str() {
            // `if op.opname == "getfield":`
            // `frozenset([("readstruct", op.args[0].concretetype, op.args[1].value)])`.
            "getfield" => WriteEffects::Set(HashSet::from([Effect::ReadStruct {
                TYPE: arg_concretetype(op, 0),
                fieldname: arg_value(op, 1),
            }])),
            // `elif op.opname == "getarrayitem":`
            // `frozenset([("readarray", op.args[0].concretetype)])`.
            // The foldable `getarrayitem_pure` (rlist.py:721-724) reads the
            // same array, so it records the identical `("readarray", T)`.
            "getarrayitem" | "getarrayitem_pure" => {
                WriteEffects::Set(HashSet::from([Effect::ReadArray {
                    TYPE: arg_concretetype(op, 0),
                }]))
            }
            // `elif op.opname == "getinteriorfield":`.
            "getinteriorfield" => {
                let name = self.getinteriorname(op);
                WriteEffects::Set(HashSet::from([Effect::ReadInteriorField {
                    TYPE: arg_concretetype(op, 0),
                    fieldname: name,
                }]))
            }
            // `elif op.opname == "gc_load_indexed":`.
            "gc_load_indexed" => self.gc_load_indexed_result(op),
            // `return WriteAnalyzer.analyze_simple_operation(self, op, graphinfo)`.
            _ => self.write_analyze_simple_operation(op, graphinfo),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, Constant, FunctionGraph, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::backendopt::graphanalyze::AnalyzerResult;
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::translator::TranslationContext;

    fn var_with_type(name: &str, ty: LowLevelType) -> Variable {
        let v = Variable::named(name);
        *v.concretetype.borrow_mut() = Some(ty);
        v
    }

    fn setfield_op(s: Variable, field: &str) -> SpaceOperation {
        SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(field))),
                Hlvalue::Variable(Variable::named("v")),
            ],
            Hlvalue::Variable(Variable::named("r")),
        )
    }

    #[test]
    fn setfield_yields_struct_effect_when_no_graphinfo() {
        let translator = TranslationContext::new();
        let mut a = WriteAnalyzer::new(&translator);
        let s = var_with_type("s", LowLevelType::Signed);
        let op = setfield_op(s, "x");
        let r = a.analyze_simple_operation(&op, &None);
        assert_eq!(
            r,
            WriteEffects::Set(HashSet::from([Effect::Struct {
                TYPE: LowLevelType::Signed,
                fieldname: ConstValue::byte_str("x"),
            }]))
        );
    }

    #[test]
    fn setarrayitem_yields_array_effect() {
        let translator = TranslationContext::new();
        let mut a = WriteAnalyzer::new(&translator);
        let s = var_with_type("s", LowLevelType::Signed);
        let op = SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Variable(Variable::named("i")),
                Hlvalue::Variable(Variable::named("v")),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        let r = a.analyze_simple_operation(&op, &None);
        assert_eq!(
            r,
            WriteEffects::Set(HashSet::from([Effect::Array {
                TYPE: LowLevelType::Signed,
            }]))
        );
    }

    #[test]
    fn gc_add_memory_pressure_yields_top() {
        let translator = TranslationContext::new();
        let mut a = WriteAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "gc_add_memory_pressure",
            vec![],
            Hlvalue::Variable(Variable::named("r")),
        );
        let r = a.analyze_simple_operation(&op, &None);
        assert!(WriteEffects::is_top_result(&r));
    }

    /// `gc_load_indexed` routes through `_gc_load_indexed_result` →
    /// `_get_effect_for_offset(.., prefix='read')`. An `ArrayItemsOffset`
    /// yields a `readarray` effect (`writeanalyze.py`).
    #[test]
    fn gc_load_indexed_arrayitems_offset_yields_readarray() {
        use crate::translator::rtyper::lltypesystem::llmemory::ArrayItemsOffset;
        use crate::translator::rtyper::lltypesystem::lltype::{Array, Ptr};
        let translator = TranslationContext::new();
        let mut a = ReadWriteAnalyzer::new(&translator);
        // `ArrayItemsOffset.TYPE` is the array *container*, so
        // `lltype.Ptr(ofs.TYPE)` is representable; a non-container here
        // would fail loud (`writeanalyze.py:110`).
        let array_ty = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Signed)));
        let ofs = AddressOffset::ArrayItemsOffset(ArrayItemsOffset(array_ty.clone()));
        let op = SpaceOperation::new(
            "gc_load_indexed",
            vec![
                Hlvalue::Variable(Variable::named("base")),
                Hlvalue::Variable(Variable::named("idx")),
                Hlvalue::Variable(Variable::named("scale")),
                Hlvalue::Constant(Constant::new(ConstValue::AddressOffset(ofs))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        let r = a.analyze_simple_operation(&op, &None);
        let expected_ptr = LowLevelType::from(Ptr::from_container_type(array_ty).unwrap());
        assert_eq!(
            r,
            WriteEffects::Set(HashSet::from([Effect::ReadArray { TYPE: expected_ptr }]))
        );
    }

    /// A `gc_store_indexed` whose offset arg is not an `AddressOffset`
    /// fails loud — upstream `_get_effect_for_offset` `assert False`s
    /// (`writeanalyze.py`) rather than returning an empty set.
    #[test]
    #[should_panic(expected = "implement me")]
    fn gc_store_indexed_non_offset_fails_loud() {
        let translator = TranslationContext::new();
        let mut a = WriteAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "gc_store_indexed",
            vec![
                Hlvalue::Variable(Variable::named("base")),
                Hlvalue::Variable(Variable::named("idx")),
                Hlvalue::Variable(Variable::named("scale")),
                Hlvalue::Variable(Variable::named("v")),
                // op.args[4] is an int constant, not an AddressOffset.
                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        let _ = a.analyze_simple_operation(&op, &None);
    }

    #[test]
    fn unknown_op_yields_empty() {
        let translator = TranslationContext::new();
        let mut a = WriteAnalyzer::new(&translator);
        let op = SpaceOperation::new("int_add", vec![], Hlvalue::Variable(Variable::named("r")));
        let r = a.analyze_simple_operation(&op, &None);
        assert_eq!(r, WriteEffects::Set(HashSet::new()));
    }

    #[test]
    fn readwrite_getfield_yields_readstruct_effect() {
        let translator = TranslationContext::new();
        let mut a = ReadWriteAnalyzer::new(&translator);
        let s = var_with_type("s", LowLevelType::Signed);
        let op = SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(s),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("x"))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        );
        let r = a.analyze_simple_operation(&op, &None);
        assert_eq!(
            r,
            WriteEffects::Set(HashSet::from([Effect::ReadStruct {
                TYPE: LowLevelType::Signed,
                fieldname: ConstValue::byte_str("x"),
            }]))
        );
    }

    /// A `setfield` on a freshly-malloc'd, non-escaped struct is
    /// suppressed (`writeanalyze.py` — `not is_fresh_malloc`).
    #[test]
    fn setfield_on_fresh_malloc_is_suppressed() {
        // Build a one-block graph:  s = malloc(...); setfield(s, "x", v).
        let s = var_with_type("s", LowLevelType::Signed);
        let malloc = SpaceOperation::new(
            "malloc",
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Hlvalue::Variable(s.clone()),
        );
        let set = setfield_op(s.clone(), "x");
        let start = Block::shared(vec![]);
        start.borrow_mut().operations.push(malloc);
        start.borrow_mut().operations.push(set.clone());
        let graph = FunctionGraph::new("g", start);
        let fm = FreshMallocs::new(&graph);
        // `s` is a fresh malloc → write through it is ignored.
        assert!(fm.is_fresh_malloc(&Hlvalue::Variable(s)));
        let translator = TranslationContext::new();
        let a = WriteAnalyzer::new(&translator);
        let r = a.write_analyze_simple_operation(&set, &Some(fm));
        assert_eq!(r, WriteEffects::Set(HashSet::new()));
    }

    /// An external call whose funcobj carries `_callbacks` surfaces the
    /// callback graphs' write effects through the inherited
    /// `GraphAnalyzer::analyze_external_call` (`graphanalyze.py`,
    /// upstream `test_llexternal_with_callback`). WriteAnalyzer does not
    /// override the base method; this proves the inheritance is live.
    #[test]
    fn external_call_with_callbacks_joins_callback_write_effects() {
        use crate::flowspace::model::{GraphFunc, GraphKey};
        use crate::translator::rtyper::lltypesystem::lltype::{
            _func, _ptr, _ptr_obj, FuncType, Ptr, PtrTarget,
        };
        use indexmap::IndexMap;
        use std::collections::HashMap;

        let translator = TranslationContext::new();

        // Callback graph: a single setfield(s, "cb", v) → struct write.
        // `s` is the startblock inputarg so it is a defined (non-fresh)
        // variable — FreshMallocs requires every analyzed var to be known.
        let s = var_with_type("s", LowLevelType::Signed);
        let cb_start = Block::shared(vec![Hlvalue::Variable(s.clone())]);
        cb_start.borrow_mut().operations.push(setfield_op(s, "cb"));
        let mut cb_graph = FunctionGraph::new("cb", cb_start);
        cb_graph.func = Some(GraphFunc::new(
            "cb",
            Constant::new(ConstValue::Dict(std::sync::Arc::new(IndexMap::new()))),
        ));
        let cb_graph = std::rc::Rc::new(std::cell::RefCell::new(cb_graph));
        let cb_key = GraphKey::of(&cb_graph).as_usize();
        translator.graphs.borrow_mut().push(cb_graph);

        // External funcobj: external='C' + _callbacks=Graphs([cb_key]).
        let mut attrs: HashMap<String, ConstValue> = HashMap::new();
        attrs.insert("external".to_string(), ConstValue::byte_str("C"));
        attrs.insert("_callbacks".to_string(), ConstValue::Graphs(vec![cb_key]));
        let functype = FuncType {
            args: vec![],
            result: LowLevelType::Void,
        };
        let ptr = _ptr::new(
            Ptr {
                TO: PtrTarget::Func(functype.clone()),
            },
            Ok(Some(_ptr_obj::Func(_func::new(
                functype,
                "ext".to_string(),
                None,
                None,
                attrs,
            )))),
        );
        let op = SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::LLPtr(
                Box::new(ptr),
            )))],
            Hlvalue::Variable(Variable::named("r")),
        );

        let mut a = WriteAnalyzer::new(&translator);
        let r = a.analyze(&op, None, &None);
        assert_eq!(
            r,
            WriteEffects::Set(HashSet::from([Effect::Struct {
                TYPE: LowLevelType::Signed,
                fieldname: ConstValue::byte_str("cb"),
            }]))
        );
    }
}
