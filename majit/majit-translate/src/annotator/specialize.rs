//! Port of `rpython/annotator/specialize.py` specialization helpers.
//!
//! This module mirrors the standalone functions and tables that
//! `FunctionDesc.specialize` dispatches into. The `memo` family
//! (`MemoTable`, `memo`, `all_values`, `cartesian_product`) lands here
//! incrementally; this first slice ports the two pure helpers
//! `all_values` (specialize.py) and `cartesian_product`
//! (specialize.py), which carry no bookkeeper back-references
//! and are exercised directly by unit tests.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::bookkeeper::{Bookkeeper, PositionKey};
use super::description::{FunctionDesc, GraphBuilder, GraphCacheKey, SpecializeResult};
use super::model::{AnnotatorError, SomeObjectTrait, SomeValue, unionof};
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue,
    HostObject, Link, SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::tool::algo::unionfind::{SharedUnionFindInfo, UnionFind};

/// RPython `MemoTable.fieldnamecounter` (specialize.py) — a process-
/// wide counter feeding `getuniquefieldname`.
static MEMO_FIELDNAME_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RPython `all_values(s)` (specialize.py).
///
/// ```python
/// def all_values(s):
///     if s.is_constant():
///         return [s.const]
///     elif isinstance(s, SomePBC):
///         values = []
///         assert not s.can_be_None, "memo call: cannot mix None and PBCs"
///         for desc in s.descriptions:
///             if desc.pyobj is None:
///                 raise annmodel.AnnotatorError(...)
///             values.append(desc.pyobj)
///         return values
///     elif isinstance(s, SomeImpossibleValue):
///         return []
///     elif isinstance(s, SomeBool):
///         return [False, True]
///     else:
///         raise annmodel.AnnotatorError(...)
/// ```
///
/// Returns the exhaustive list of host values matching annotation `s`,
/// in `ConstValue` form — the same carrier `Bookkeeper.immutablevalue`
/// consumes. A frozen-PBC member contributes its `desc.pyobj`
/// (wrapped as [`ConstValue::HostObject`]); a non-constant `SomeBool`
/// contributes both `False` and `True`. The `assert not s.can_be_None`
/// precondition is surfaced as an [`AnnotatorError`] rather than a
/// panic so the memo caller can report it.
pub fn all_values(s: &SomeValue) -> Result<Vec<ConstValue>, AnnotatorError> {
    // upstream: `if s.is_constant(): return [s.const]`.
    if s.is_constant() {
        let c = s
            .const_()
            .ok_or_else(|| AnnotatorError::new("all_values: is_constant() but const_box absent"))?;
        return Ok(vec![c.clone()]);
    }
    match s {
        // upstream: `elif isinstance(s, SomePBC): ...`.
        SomeValue::PBC(pbc) => {
            // upstream: `assert not s.can_be_None, "memo call: cannot
            // mix None and PBCs"`.
            if pbc.can_be_none {
                return Err(AnnotatorError::new("memo call: cannot mix None and PBCs"));
            }
            let mut values = Vec::with_capacity(pbc.descriptions.len());
            // upstream: `for desc in s.descriptions:`.
            for desc in pbc.descriptions.values() {
                // upstream: `if desc.pyobj is None: raise AnnotatorError(...)`.
                let pyobj = desc.pyobj().ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "memo call with a class or PBC that has no \
                         corresponding Python object ({desc:?})"
                    ))
                })?;
                // upstream: `values.append(desc.pyobj)`.
                values.push(ConstValue::HostObject(pyobj));
            }
            Ok(values)
        }
        // upstream: `elif isinstance(s, SomeImpossibleValue): return []`.
        SomeValue::Impossible => Ok(Vec::new()),
        // upstream: `elif isinstance(s, SomeBool): return [False, True]`.
        // A *constant* bool was already handled by the is_constant()
        // arm above, so this only fires for the unknown-bool case.
        SomeValue::Bool(_) => Ok(vec![ConstValue::Bool(false), ConstValue::Bool(true)]),
        // upstream: `else: raise AnnotatorError("memo call: argument
        // must be a class or a frozen PBC, got %r")`.
        _ => Err(AnnotatorError::new(format!(
            "memo call: argument must be a class or a frozen PBC, got {s:?}"
        ))),
    }
}

/// RPython `cartesian_product(lstlst)` (specialize.py).
///
/// ```python
/// def cartesian_product(lstlst):
///     if not lstlst:
///         yield ()
///         return
///     for tuple_tail in cartesian_product(lstlst[1:]):
///         for value in lstlst[0]:
///             yield (value,) + tuple_tail
/// ```
///
/// Upstream is a generator; the Rust port materialises the same
/// sequence eagerly. The emission order is preserved exactly: the
/// first list varies fastest within each tail, so `memo`'s
/// `possiblevalues.next()` picks the same `firstvalues` tuple as
/// upstream.
pub fn cartesian_product<T: Clone>(lstlst: &[Vec<T>]) -> Vec<Vec<T>> {
    // upstream: `if not lstlst: yield (); return`.
    if lstlst.is_empty() {
        return vec![Vec::new()];
    }
    // upstream: `for tuple_tail in cartesian_product(lstlst[1:]):`.
    let tails = cartesian_product(&lstlst[1..]);
    let mut out = Vec::with_capacity(tails.len() * lstlst[0].len());
    for tail in &tails {
        // upstream: `for value in lstlst[0]: yield (value,) + tuple_tail`.
        for value in &lstlst[0] {
            let mut tuple = Vec::with_capacity(tail.len() + 1);
            tuple.push(value.clone());
            tuple.extend(tail.iter().cloned());
            out.push(tuple);
        }
    }
    out
}

/// RPython `flatten_star_args(funcdesc, args_s)` (specialize.py).
///
/// The Rust implementation lives on [`FunctionDesc`] because it needs direct
/// access to the descriptor's signature, defaults, graph builder, and cache-key
/// machinery. This top-level wrapper preserves the upstream module function
/// name for call sites that are ported line-by-line.
#[allow(dead_code)] // RPython module-level port surface; implementation lives on FunctionDesc.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn flatten_star_args<'a>(
    funcdesc: &'a FunctionDesc,
    args_s: &[Option<SomeValue>],
) -> Result<
    (
        Vec<Option<SomeValue>>,
        GraphCacheKey,
        Option<GraphBuilder<'a>>,
    ),
    AnnotatorError,
> {
    funcdesc.flatten_star_args(args_s)
}

/// RPython `default_specialize(funcdesc, args_s)` (specialize.py).
pub fn default_specialize(
    funcdesc: &FunctionDesc,
    args_s: &mut Vec<Option<SomeValue>>,
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.default_specialize(args_s)
}

/// RPython `getuniquenondirectgraph(desc)` (specialize.py).
#[allow(dead_code)] // RPython module-level port surface; implementation lives on FunctionDesc.
pub fn getuniquenondirectgraph(desc: &FunctionDesc) -> Result<Rc<PyGraph>, AnnotatorError> {
    desc.getuniquenondirectgraph()
}

/// RPython `maybe_star_args(funcdesc, key, args_s)` (specialize.py).
#[allow(dead_code)] // RPython module-level port surface; implementation lives on FunctionDesc.
pub fn maybe_star_args(
    funcdesc: &FunctionDesc,
    key: GraphCacheKey,
    args_s: &[Option<SomeValue>],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.maybe_star_args(key, args_s)
}

/// RPython `specialize_argvalue(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_argvalue(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[String],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.specialize_argvalue(args_s, argindices)
}

/// RPython `specialize_arg_or_var(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_arg_or_var(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[String],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.specialize_arg_or_var(args_s, argindices)
}

/// RPython `specialize_argtype(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_argtype(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[String],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.specialize_argtype(args_s, argindices)
}

/// RPython `specialize_arglistitemtype(funcdesc, args_s, i)`
/// (specialize.py).
pub fn specialize_arglistitemtype(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[String],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.specialize_arglistitemtype(args_s, argindices)
}

/// RPython `specialize_call_location(funcdesc, args_s, op)`
/// (specialize.py).
pub fn specialize_call_location(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    op: Option<PositionKey>,
) -> Result<Rc<PyGraph>, AnnotatorError> {
    funcdesc.specialize_call_location(args_s, op)
}

/// RPython `class MemoTable(object)` (specialize.py).
///
/// One table accumulates `{args_tuple: result}` for a family of memo
/// calls; families merge via the `UnionFind` in
/// `Bookkeeper.all_specializations`. Upstream holds the live `funcdesc`
/// to reach `.bookkeeper` / `.name` / `.defaults` /
/// `desc.create_new_attribute`. The Rust port keeps that subset directly
/// — `funcdesc_name`, `funcdesc_defaults`, and a `Weak<Bookkeeper>` (weak
/// so the table does not pin the bookkeeper into a reference cycle).
/// `finish`'s dispatch-graph synthesis reaches every PBC desc through
/// `bookkeeper.getdesc(host)` (the host objects come from the table keys),
/// so no live `funcdesc` reference is needed — see [`MemoTable::finish`].
pub struct MemoTable {
    /// `self.funcdesc.name` — used by `getuniquefieldname` (with `finish`)
    /// and error messages.
    pub funcdesc_name: String,
    /// `self.funcdesc.defaults` — `finish` copies these onto the graph.
    pub funcdesc_defaults: Vec<Constant>,
    /// Back-reference to the owning bookkeeper for `register_finish`.
    pub bookkeeper: Weak<Bookkeeper>,
    /// RPython `self.table = {args: value}` (specialize.py).
    pub table: HashMap<Vec<ConstValue>, ConstValue>,
    /// RPython `self.graph = None` (specialize.py). Stays `None`
    /// until `finish` synthesises the dispatch graph (deferred).
    pub graph: Option<Rc<PyGraph>>,
    /// RPython `self.do_not_process = False` (specialize.py).
    pub do_not_process: bool,
}

impl MemoTable {
    /// RPython `MemoTable.__init__(funcdesc, args, value)`
    /// (specialize.py).
    pub fn new(
        funcdesc_name: String,
        funcdesc_defaults: Vec<Constant>,
        bookkeeper: Weak<Bookkeeper>,
        args: Vec<ConstValue>,
        value: ConstValue,
    ) -> Self {
        // upstream: `self.table = {args: value}`.
        let mut table = HashMap::new();
        table.insert(args, value);
        MemoTable {
            funcdesc_name,
            funcdesc_defaults,
            bookkeeper,
            table,
            graph: None,
            do_not_process: false,
        }
    }

    /// RPython `MemoTable.register_finish(self)` (specialize.py):
    /// `bookkeeper.pending_specializations.append(self.finish)`.
    ///
    /// Takes the `Rc` so the scheduled closure can hold the table; the
    /// closure runs `finish` when the bookkeeper drains pending
    /// specializations. `finish` synthesises the dispatch graph (see
    /// [`MemoTable::finish`]); any `Err` it yields — the "already
    /// finished" invariant or a dropped bookkeeper/annotator — propagates
    /// out of the drain to the annotator's `complete()`, matching
    /// upstream's exception flow.
    pub fn register_finish(this: &Rc<RefCell<MemoTable>>) {
        let Some(bookkeeper) = this.borrow().bookkeeper.upgrade() else {
            return;
        };
        let this = this.clone();
        bookkeeper
            .pending_specializations
            .borrow_mut()
            // upstream: `bookkeeper.pending_specializations.append(self.finish)`.
            .push(Box::new(move || MemoTable::finish(&this)));
    }

    /// RPython `MemoTable.getuniquefieldname(self)` (specialize.py):
    /// `'$memofield_%s_%d' % (self.funcdesc.name, MemoTable.fieldnamecounter)`
    /// with a process-wide post-increment of the counter.
    fn getuniquefieldname(name: &str) -> String {
        let counter = MEMO_FIELDNAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("$memofield_{name}_{counter}")
    }

    /// RPython `MemoTable.finish(self)` (specialize.py).
    ///
    /// Synthesises the dispatch graph that, at run time, maps each
    /// argument tuple to its precomputed memo result, then schedules it
    /// for annotation. Upstream builds a tree of nested helper functions
    /// via `make_helper` (Python source + `exec` + `buildflowgraph`) —
    /// the only way RPython can produce a graph, since its sole graph
    /// source is Python bytecode. pyre synthesises flowspace graphs
    /// directly (the Position-2 adaptation), so the helper-function tree
    /// collapses into an equivalent **block tree inside one
    /// `FunctionGraph`** with identical dispatch semantics: leaf →
    /// constant return; single-value → ignore arg, fall through; bool →
    /// `exitswitch` on the arg; PBC set with all-constant results →
    /// `getattr($memofield, ...)` after storing each constant via
    /// `desc.create_new_attribute`. The `store = nextfns` branch
    /// (specialize.py — a PBC arg followed by further varying
    /// args whose results are not all constant) stores subhelper
    /// *callables* in the memo fields: each subhelper is emitted as a
    /// standalone prebuilt [`FunctionGraph`] (see
    /// [`MemoSynth::build_standalone_subhelper`]) and the dispatch calls
    /// the `getattr` result with the remaining arguments.
    pub fn finish(this: &Rc<RefCell<MemoTable>>) -> Result<(), AnnotatorError> {
        // Read the table snapshot, then drop the borrow before mutating
        // `graph` / scheduling, so the recursion and `addpendinggraph`
        // never nest a borrow on this cell.
        let (table, name, defaults, bk_weak, do_not_process, has_graph) = {
            let m = this.borrow();
            (
                m.table.clone(),
                m.funcdesc_name.clone(),
                m.funcdesc_defaults.clone(),
                m.bookkeeper.clone(),
                m.do_not_process,
                m.graph.is_some(),
            )
        };
        // upstream: `if self.do_not_process: return`.
        if do_not_process {
            return Ok(());
        }
        // upstream: `assert self.graph is None, "MemoTable already finished"`.
        if has_graph {
            return Err(AnnotatorError::new("MemoTable already finished"));
        }
        // upstream: `example_args, example_value = self.table.iteritems().next()`.
        let (example_args, example_value) = table
            .iter()
            .next()
            .map(|(k, v)| (k.clone(), v.clone()))
            .ok_or_else(|| AnnotatorError::new("MemoTable.finish: empty table"))?;
        // upstream: `nbargs = len(example_args)`.
        let nbargs = example_args.len();
        // upstream: `sets = [set() for i in range(nbargs)]` then
        // `for args in self.table: for i: sets[i].add(args[i])`.
        let mut sets: Vec<Vec<ConstValue>> = vec![Vec::new(); nbargs];
        for args in table.keys() {
            for (i, slot) in sets.iter_mut().enumerate() {
                if !slot.contains(&args[i]) {
                    slot.push(args[i].clone());
                }
            }
        }

        // upstream: `bookkeeper = self.funcdesc.bookkeeper; annotator =
        // bookkeeper.annotator`.
        let bookkeeper = bk_weak
            .upgrade()
            .ok_or_else(|| AnnotatorError::new("MemoTable.finish: bookkeeper dropped"))?;
        let annotator = bookkeeper
            .try_annotator()
            .ok_or_else(|| AnnotatorError::new("MemoTable.finish: annotator not attached"))?;

        // upstream: `argnames = ['a%d' % i for i in range(nbargs)]`.
        let argnames: Vec<String> = (0..nbargs).map(|i| format!("a{i}")).collect();

        // Build the dispatch as a block tree (see method doc). The
        // returnblock comes from `FunctionGraph::new`, so create the graph
        // with a placeholder startblock first, then overwrite it with the
        // synthesised root.
        let full_name = format!("memo_{name}_0");
        let mut func = GraphFunc::new(
            &full_name,
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.defaults = defaults.clone();
        let placeholder_inputs: Vec<Hlvalue> = argnames
            .iter()
            .map(|n| Hlvalue::Variable(Variable::named(n)))
            .collect();
        let mut fg = FunctionGraph::new(&full_name, Block::shared(placeholder_inputs));
        fg.func = Some(func.clone());
        let returnblock = fg.returnblock.clone();

        let synth = MemoSynth {
            bookkeeper: &bookkeeper,
            name: &name,
            table: &table,
            example_value: &example_value,
            sets: &sets,
            nbargs,
            argnames: &argnames,
            returnblock: &returnblock,
        };
        // upstream: `entrypoint = make_subhelper(args_so_far = ())`.
        let root = synth.build_subhelper(&[])?;
        fg.startblock = root;

        // upstream: `self.graph = annotator.translator.buildflowgraph(
        // entrypoint); self.graph.defaults = self.funcdesc.defaults`.
        let signature = Signature::new(argnames.clone(), None, None);
        let pygraph = Rc::new(PyGraph {
            graph: fg.into_ref(),
            func,
            signature: RefCell::new(signature),
            defaults: RefCell::new(Some(defaults)),
            access_directly: Cell::new(false),
        });
        this.borrow_mut().graph = Some(pygraph.clone());
        // `buildflowgraph` stores every graph it builds in the translator's
        // graph list (`self.graphs.append(graph)`, translator.py). The
        // direct block-tree synthesis bypasses `buildflowgraph`, so register
        // the synthesised graph here to keep the translator's graph list
        // complete.
        annotator
            .translator
            .graphs
            .borrow_mut()
            .push(pygraph.graph.clone());

        // upstream: `args_s = []; for arg_types in sets: values_s =
        // [bookkeeper.immutablevalue(x) for x in arg_types]; args_s.append(
        // unionof(*values_s))`.
        let mut args_s: Vec<Option<SomeValue>> = Vec::with_capacity(nbargs);
        for arg_types in &sets {
            let mut values_s = Vec::with_capacity(arg_types.len());
            for x in arg_types {
                values_s.push(bookkeeper.immutablevalue(x)?);
            }
            let u = unionof(values_s.iter())
                .map_err(|e| AnnotatorError::new(format!("memo finish unionof: {e}")))?;
            args_s.push(Some(u));
        }
        // upstream: `annotator.addpendinggraph(self.graph, args_s)`.
        annotator.addpendinggraph(&pygraph.graph, &args_s);
        Ok(())
    }
}

/// Bundles the read-only inputs threaded through `make_subhelper`'s
/// recursion (specialize.py) so the block-tree synthesis can
/// stay a set of `&self` methods.
struct MemoSynth<'a> {
    bookkeeper: &'a Rc<Bookkeeper>,
    name: &'a str,
    table: &'a HashMap<Vec<ConstValue>, ConstValue>,
    example_value: &'a ConstValue,
    sets: &'a [Vec<ConstValue>],
    nbargs: usize,
    argnames: &'a [String],
    returnblock: &'a BlockRef,
}

impl MemoSynth<'_> {
    /// The `fn.constant_result` of the subhelper for `args_so_far`
    /// (specialize.py, `make_constant_subhelper`): a subtree
    /// folds to a constant only through a chain of leaves and
    /// single-valued positions; a bool or PBC-set position makes
    /// `make_helper` (no `constant_result`), so it returns `None`.
    fn const_result(&self, args_so_far: &[ConstValue]) -> Option<ConstValue> {
        let firstarg = args_so_far.len();
        // upstream leaf: `result = self.table.get(args_so_far, example_value)`.
        if firstarg == self.nbargs {
            return Some(
                self.table
                    .get(args_so_far)
                    .cloned()
                    .unwrap_or_else(|| self.example_value.clone()),
            );
        }
        let values = &self.sets[firstarg];
        if values.len() == 1 {
            // single value: subhelper inherits the child's constant_result.
            let mut next = args_so_far.to_vec();
            next.push(values[0].clone());
            self.const_result(&next)
        } else {
            None
        }
    }

    /// RPython `make_subhelper(args_so_far)` (specialize.py),
    /// emitting one dispatch block instead of a helper function.
    fn build_subhelper(&self, args_so_far: &[ConstValue]) -> Result<BlockRef, AnnotatorError> {
        let firstarg = args_so_far.len();
        // The block consumes positions `firstarg..nbargs` (upstream
        // `argnames[firstarg:]`); a fresh Variable per position.
        let inputs: Vec<Hlvalue> = (firstarg..self.nbargs)
            .map(|i| Hlvalue::Variable(Variable::named(&self.argnames[i])))
            .collect();
        let block = Block::shared(inputs.clone());

        // upstream: `if firstarg == nbargs:` — no argument left, return
        // the known result (or example_value).
        if firstarg == self.nbargs {
            let result = self
                .table
                .get(args_so_far)
                .cloned()
                .unwrap_or_else(|| self.example_value.clone());
            let link = Link::new(
                vec![Hlvalue::Constant(Constant::new(result))],
                Some(self.returnblock.clone()),
                None,
            )
            .into_ref();
            block.closeblock(vec![link]);
            return Ok(block);
        }

        // upstream: `nextargvalues = list(sets[len(args_so_far)]); if
        // nextargvalues == [True, False]: nextargvalues = [False, True]`.
        let values = normalize_bool_order(&self.sets[firstarg]);
        // upstream: `restargs = ', '.join(argnames[firstarg+1:])` — here the
        // tail input vars forwarded to a subhelper block.
        let restargs: Vec<Hlvalue> = inputs[1..].to_vec();

        // upstream: `if len(nextargvalues) == 1:`.
        if values.len() == 1 {
            if let Some(result) = self.const_result(args_so_far) {
                // upstream: `if constants: result = constants[0];
                // return make_constant_subhelper(...)`.
                let link = Link::new(
                    vec![Hlvalue::Constant(Constant::new(result))],
                    Some(self.returnblock.clone()),
                    None,
                )
                .into_ref();
                block.closeblock(vec![link]);
            } else {
                // upstream: `else: stmt = 'return subhelper(%s)' % restargs`
                // — ignore the first argument and forward to the subhelper.
                let mut next = args_so_far.to_vec();
                next.push(values[0].clone());
                let sub = self.build_subhelper(&next)?;
                let link = Link::new(restargs, Some(sub), None).into_ref();
                block.closeblock(vec![link]);
            }
            return Ok(block);
        }

        // upstream: `elif nextargvalues == [False, True]:` — bool arg.
        if is_false_true(&values) {
            let switch_var = inputs[0].clone();
            let mut exits = Vec::with_capacity(2);
            // upstream emits the True branch then the False branch; the
            // exit order is immaterial because each carries its exitcase.
            for case in [false, true] {
                let mut next = args_so_far.to_vec();
                next.push(ConstValue::Bool(case));
                let exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(case))));
                if let Some(result) = self.const_result(&next) {
                    // upstream: `case = nextfns[...].constant_result;
                    // stmt.append('    return case')`.
                    exits.push(
                        Link::new(
                            vec![Hlvalue::Constant(Constant::new(result))],
                            Some(self.returnblock.clone()),
                            exitcase,
                        )
                        .into_ref(),
                    );
                } else {
                    // upstream: `stmt.append('    return case(%s)' % restargs)`.
                    let sub = self.build_subhelper(&next)?;
                    exits.push(Link::new(restargs.clone(), Some(sub), exitcase).into_ref());
                }
            }
            block.borrow_mut().exitswitch = Some(switch_var);
            block.closeblock(exits);
            return Ok(block);
        }

        // upstream: `else:` — the arg is a set of PBCs.
        // upstream: `descs = [bookkeeper.getdesc(pbc) for pbc in
        // nextargvalues]`. `create_new_attribute` is defined on both
        // FrozenDesc and ClassDesc, so a memo argument set of frozen
        // PBCs *or* classes is accepted (specialize.py).
        let mut descs = Vec::with_capacity(values.len());
        for v in &values {
            let host = match v {
                ConstValue::HostObject(h) => h.clone(),
                other => {
                    return Err(AnnotatorError::new(format!(
                        "memo finish: PBC set member is not a host object: {other:?}"
                    )));
                }
            };
            descs.push(self.bookkeeper.getdesc(&host)?);
        }
        // upstream: `constants = [fn.constant_result for fn in nextfns]`
        // (None if any branch lacks one).
        let mut branch_consts = Vec::with_capacity(values.len());
        for v in &values {
            let mut next = args_so_far.to_vec();
            next.push(v.clone());
            branch_consts.push(self.const_result(&next));
        }
        // upstream: `if constants: store = constants else: store = nextfns`.
        let all_constant = branch_consts.iter().all(Option::is_some);
        // upstream: `fieldname = self.getuniquefieldname()`.
        let fieldname = MemoTable::getuniquefieldname(self.name);
        // upstream: `for desc, value_to_store in zip(descs, store):
        // desc.create_new_attribute(fieldname, value_to_store)`. When
        // every subhelper folds to a constant (`store = constants`) the
        // memo field holds the precomputed answer. Otherwise (`store =
        // nextfns`) the field holds the subhelper *callable* — emitted as
        // a standalone prebuilt graph — and the dispatch calls the
        // `getattr` result with the remaining arguments.
        for (i, v) in values.iter().enumerate() {
            let value_to_store = if all_constant {
                branch_consts[i]
                    .clone()
                    .expect("branch_consts all Some under all_constant")
            } else {
                let mut next = args_so_far.to_vec();
                next.push(v.clone());
                ConstValue::HostObject(self.build_standalone_subhelper(&next)?)
            };
            descs[i].create_new_attribute(&fieldname, value_to_store)?;
        }
        // upstream: `stmt = 'return getattr(%s, %r)' % (argnames[firstarg],
        // fieldname)`.
        let attr_var = Hlvalue::Variable(Variable::new());
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getattr",
            vec![
                inputs[0].clone(),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(&fieldname))),
            ],
            attr_var.clone(),
        ));
        let result = if all_constant {
            // `store = constants`: the getattr result is the answer.
            attr_var
        } else {
            // `store = nextfns`: `stmt += '(%s)' % restargs` — call the
            // getattr result with the remaining arguments.
            let call_result = Hlvalue::Variable(Variable::new());
            let mut call_args = Vec::with_capacity(1 + restargs.len());
            call_args.push(attr_var);
            call_args.extend(restargs.iter().cloned());
            block.borrow_mut().operations.push(SpaceOperation::new(
                "simple_call",
                call_args,
                call_result.clone(),
            ));
            call_result
        };
        let link = Link::new(vec![result], Some(self.returnblock.clone()), None).into_ref();
        block.closeblock(vec![link]);
        Ok(block)
    }

    /// Build the subhelper for `args_so_far` as a standalone callable
    /// graph and register it so a later `getattr`+call resolves to it.
    ///
    /// RPython's `make_subhelper` always returns a *function*; the
    /// block-tree port inlines those functions wherever the dispatch
    /// folds into a block, but the `store = nextfns` branch
    /// (specialize.py) stores the subhelper *callables* in the
    /// PBC memo fields and calls the `getattr` result. A shared block
    /// tree has no callable to store, so each such subhelper is emitted
    /// here as its own [`FunctionGraph`] wrapped in a [`GraphFunc`],
    /// registered as a prebuilt graph (translator.py __init__) so the
    /// funcdesc's `buildgraph` returns it instead of flowing absent
    /// bytecode, and appended to the translator graph list. The returned
    /// [`HostObject`] is the callable PBC stored in the memo field.
    fn build_standalone_subhelper(
        &self,
        args_so_far: &[ConstValue],
    ) -> Result<HostObject, AnnotatorError> {
        let firstarg = args_so_far.len();
        // upstream `make_helper` builds `def f(argnames[firstarg:])`.
        let sub_argnames: Vec<String> = self.argnames[firstarg..self.nbargs].to_vec();
        // upstream: `func_with_new_name(f, 'memo_%s_%d' % (name, firstarg))`.
        let full_name = format!("memo_{}_{}", self.name, firstarg);

        // A fresh returnblock for this standalone graph — the parent's
        // returnblock belongs to a different graph.
        let placeholder_inputs: Vec<Hlvalue> = sub_argnames
            .iter()
            .map(|n| Hlvalue::Variable(Variable::named(n)))
            .collect();
        let mut fg = FunctionGraph::new(&full_name, Block::shared(placeholder_inputs));
        let returnblock = fg.returnblock.clone();
        let sub_synth = MemoSynth {
            bookkeeper: self.bookkeeper,
            name: self.name,
            table: self.table,
            example_value: self.example_value,
            sets: self.sets,
            nbargs: self.nbargs,
            argnames: self.argnames,
            returnblock: &returnblock,
        };
        let root = sub_synth.build_subhelper(args_so_far)?;
        fg.startblock = root;

        // A `GraphFunc` plus a synthetic `HostCode` so `newfuncdesc`
        // derives the subhelper signature (`argnames[firstarg:]`) when
        // the call site is annotated (bookkeeper.py).
        let mut func = GraphFunc::new(
            &full_name,
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.code = Some(Box::new(synthetic_subhelper_code(
            &full_name,
            &sub_argnames,
        )));
        let signature = Signature::new(sub_argnames.clone(), None, None);
        let pygraph = Rc::new(PyGraph {
            graph: fg.into_ref(),
            func: func.clone(),
            signature: RefCell::new(signature),
            defaults: RefCell::new(Some(Vec::new())),
            access_directly: Cell::new(false),
        });
        let host = HostObject::new_user_function(func);

        let annotator = self
            .bookkeeper
            .try_annotator()
            .ok_or_else(|| AnnotatorError::new("memo finish: annotator not attached"))?;
        // translator.py: `buildflowgraph` returns a prebuilt graph
        // as-is; translator.py would otherwise append it to the graph
        // list, so register it there here too.
        annotator
            .translator
            ._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph.clone());
        annotator
            .translator
            .graphs
            .borrow_mut()
            .push(pygraph.graph.clone());
        Ok(host)
    }
}

/// A minimal `HostCode` whose `co_argcount` / `co_varnames` carry the
/// subhelper parameter list so `cpython_code_signature` derives the
/// matching `Signature`. The bytecode body is empty — the graph is
/// supplied prebuilt, so `buildflowgraph` never flows it.
fn synthetic_subhelper_code(name: &str, argnames: &[String]) -> HostCode {
    HostCode::new(
        argnames.len() as u32,
        argnames.len() as u32,
        0,
        0,
        rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
        Vec::new(),
        Vec::new(),
        argnames.to_vec(),
        "<memo>".to_string(),
        name.to_string(),
        0,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new().into_boxed_slice(),
    )
}

/// RPython `if nextargvalues == [True, False]: nextargvalues = [False,
/// True]` (specialize.py:171-172) — normalise a two-valued bool set to
/// `[False, True]`; every other set is returned unchanged.
fn normalize_bool_order(values: &[ConstValue]) -> Vec<ConstValue> {
    if values.len() == 2
        && matches!(values[0], ConstValue::Bool(true))
        && matches!(values[1], ConstValue::Bool(false))
    {
        return vec![ConstValue::Bool(false), ConstValue::Bool(true)];
    }
    values.to_vec()
}

/// RPython `nextargvalues == [False, True]` (specialize.py) — the
/// bool-arg discriminator after [`normalize_bool_order`].
fn is_false_true(values: &[ConstValue]) -> bool {
    values.len() == 2
        && matches!(values[0], ConstValue::Bool(false))
        && matches!(values[1], ConstValue::Bool(true))
}

impl SharedUnionFindInfo for MemoTable {
    /// RPython `MemoTable.absorb(self, other)` (specialize.py):
    /// `self.table.update(other.table); assert self.graph is None;
    /// other.do_not_process = True`.
    fn absorb_shared(this: &Rc<RefCell<Self>>, other: Rc<RefCell<Self>>) {
        {
            let other_ref = other.borrow();
            let mut me = this.borrow_mut();
            // upstream: `self.table.update(other.table)`.
            for (k, v) in &other_ref.table {
                me.table.insert(k.clone(), v.clone());
            }
            // upstream: `assert self.graph is None, "too late for MemoTable merge!"`.
            assert!(me.graph.is_none(), "too late for MemoTable merge!");
        }
        // upstream: `other.do_not_process = True`.
        other.borrow_mut().do_not_process = true;
    }
}

/// One `Bookkeeper.all_specializations[funcdesc]` entry: the
/// `UnionFind(compute_one_result)` family of argument tuples
/// (specialize.py) plus a per-family latch for the host-call error.
///
/// The UnionFind info factory has an infallible `Fn(&K) -> V` signature,
/// so the `compute_one_result` host call (`func(*args)`) cannot return a
/// `Result` from inside it. Rather than the previous pre-flight (which
/// called `func(*args)` once to validate and once more in the factory),
/// the factory latches the first host-call error into `host_err` and
/// [`memo`] surfaces it after `find`/`union` — a single call per argument
/// tuple. The latch lives with the family (not the `memo` call) so a
/// later reflow that triggers a fresh `find` miss latches into the same
/// cell the next `memo` call reads.
pub struct MemoFamily {
    pub uf: UnionFind<Vec<ConstValue>, Rc<RefCell<MemoTable>>>,
    pub host_err: Rc<RefCell<Option<AnnotatorError>>>,
}

/// `specialize.py memo` KeyError branch:
/// `bookkeeper.all_specializations[funcdesc] = UnionFind(compute_one_result)`,
/// then `find(args)`.
///
/// `funcdesc_key` is the same key kind [`memo`] uses
/// (`funcdesc as *const FunctionDesc as usize`). Does not call
/// [`MemoTable::register_finish`]: [`memo`] does that so `finish` can
/// synthesise the pycall dispatch graph. Callers that only need the
/// host result (rtyper-time `specialize:memo` helpers) read
/// [`MemoTable::table`].
pub fn memotable_for_args(
    bookkeeper: &Bookkeeper,
    funcdesc_key: usize,
    funcdesc_name: &str,
    args: Vec<ConstValue>,
    compute_one_result: impl Fn(&[ConstValue]) -> Result<ConstValue, String> + 'static,
) -> Result<Rc<RefCell<MemoTable>>, AnnotatorError> {
    let memotable = {
        let mut specs = bookkeeper.all_specializations.borrow_mut();
        let family = specs.entry(funcdesc_key).or_insert_with(|| {
            let name = funcdesc_name.to_string();
            let host_err: Rc<RefCell<Option<AnnotatorError>>> = Rc::new(RefCell::new(None));
            let factory_err = host_err.clone();
            let uf = UnionFind::new(
                move |args: &Vec<ConstValue>| match compute_one_result(args) {
                    Ok(value) => Rc::new(RefCell::new(MemoTable::new(
                        name.clone(),
                        Vec::new(),
                        Weak::new(),
                        args.clone(),
                        value,
                    ))),
                    Err(e) => {
                        if factory_err.borrow().is_none() {
                            *factory_err.borrow_mut() = Some(AnnotatorError::new(format!(
                                "memo call: compute_one_result for {name} failed on {args:?}: {e}"
                            )));
                        }
                        let memotable = Rc::new(RefCell::new(MemoTable::new(
                            name.clone(),
                            Vec::new(),
                            Weak::new(),
                            args.clone(),
                            ConstValue::None,
                        )));
                        memotable.borrow_mut().do_not_process = true;
                        memotable
                    }
                },
            );
            MemoFamily { uf, host_err }
        });
        let rep = family.uf.find(args).1;
        if let Some(e) = family.host_err.borrow_mut().take() {
            return Err(e);
        }
        family
            .uf
            .get(&rep)
            .expect("memotable_for_args: representative MemoTable present after find")
            .clone()
    };
    Ok(memotable)
}

/// RPython `memo(funcdesc, args_s)` (specialize.py).
///
/// Calls the memo function now for every combination of possible
/// argument values, accumulates the results into the per-funcdesc
/// `UnionFind(compute_one_result)` family stored in
/// `Bookkeeper.all_specializations`, and returns either the family's
/// dispatch graph (once `finish` synthesises it) or the union of the
/// per-tuple results. The host call (`func(*args)`) runs through the
/// [`crate::flowspace::model::HostCall`] hook on the function's
/// `GraphFunc`.
///
/// upstream's `compute_one_result` runs `func(*args)` inside the
/// `UnionFind` info factory, whose `Fn(&K) -> V` signature cannot carry a
/// `Result`. The factory therefore latches the first host-call error into
/// the family's [`MemoFamily::host_err`] cell (and yields a
/// `do_not_process` placeholder table); `memo` surfaces that error after
/// `find`/`union`. Each argument tuple is thus evaluated exactly once —
/// no pre-flight double call.
pub fn memo(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
) -> Result<SpecializeResult, AnnotatorError> {
    // upstream: `possiblevalues = cartesian_product([all_values(s_arg)
    //            for s_arg in args_s])`.
    let mut per_arg = Vec::with_capacity(args_s.len());
    for (i, slot) in args_s.iter().enumerate() {
        let s = slot
            .as_ref()
            .ok_or_else(|| AnnotatorError::new(format!("memo: argument {i} is unannotated")))?;
        per_arg.push(all_values(s)?);
    }
    let possiblevalues = cartesian_product(&per_arg);

    // upstream: `bookkeeper = funcdesc.bookkeeper`.
    let bookkeeper: Rc<Bookkeeper> = funcdesc.base.bookkeeper.clone();

    // upstream: `func = funcdesc.pyobj; if func is None: raise ...`.
    let func_ho = funcdesc.base.pyobj.as_ref().ok_or_else(|| {
        AnnotatorError::new(format!(
            "memo call: no Python function object to call ({})",
            funcdesc.name
        ))
    })?;
    let graph_func = func_ho.user_function().ok_or_else(|| {
        AnnotatorError::new(format!(
            "memo call: pyobj of {} is not a user function",
            funcdesc.name
        ))
    })?;
    // upstream keys `all_specializations` by the funcdesc object itself
    // (specialize.py:285); use the funcdesc's pointer identity, stable
    // for the lifetime of the registry entry.
    let key = funcdesc as *const FunctionDesc as usize;
    let host_call = graph_func.host_call.clone().ok_or_else(|| {
        AnnotatorError::new(format!(
            "memo call: function {} has no host_call hook registered",
            funcdesc.name
        ))
    })?;

    let name = funcdesc.name.clone();
    let defaults = funcdesc.defaults.clone();
    let bk_weak = Rc::downgrade(&bookkeeper);

    // upstream: `firstvalues = possiblevalues.next()` (StopIteration if
    // any argument's value set is empty).
    let firstvalues = possiblevalues.first().cloned().ok_or_else(|| {
        AnnotatorError::new(format!(
            "memo call: {} has no possible argument combinations",
            funcdesc.name
        ))
    })?;

    // Resolve the merged MemoTable. Scope the all_specializations borrow
    // so the per-table `register_finish` (which borrows the *separate*
    // pending_specializations cell) and the later immutablevalue calls
    // run without nesting on this RefCell.
    let memotable: Rc<RefCell<MemoTable>> = {
        let mut specs = bookkeeper.all_specializations.borrow_mut();
        // upstream: `memotables = bookkeeper.all_specializations[funcdesc]`
        //           with the KeyError branch building a fresh
        //           `UnionFind(compute_one_result)`.
        let family = specs.entry(key).or_insert_with(|| {
            let host_call = host_call.clone();
            let name = name.clone();
            let defaults = defaults.clone();
            let bk_weak = bk_weak.clone();
            let host_err: Rc<RefCell<Option<AnnotatorError>>> = Rc::new(RefCell::new(None));
            let factory_err = host_err.clone();
            let uf = UnionFind::new(move |args: &Vec<ConstValue>| {
                // upstream `compute_one_result`: `value = func(*args);
                // memotable = MemoTable(funcdesc, args, value);
                // memotable.register_finish(); return memotable`.
                match (host_call.0)(args) {
                    Ok(value) => {
                        let memotable = Rc::new(RefCell::new(MemoTable::new(
                            name.clone(),
                            defaults.clone(),
                            bk_weak.clone(),
                            args.clone(),
                            value,
                        )));
                        MemoTable::register_finish(&memotable);
                        memotable
                    }
                    Err(e) => {
                        // The factory cannot return a Result; latch the
                        // first error and yield a `do_not_process`
                        // placeholder (never registered for finish).
                        // `memo` reads `host_err` after find/union.
                        if factory_err.borrow().is_none() {
                            *factory_err.borrow_mut() = Some(AnnotatorError::new(format!(
                                "memo call: host call for {name} failed on {args:?}: {e}"
                            )));
                        }
                        let memotable = Rc::new(RefCell::new(MemoTable::new(
                            name.clone(),
                            defaults.clone(),
                            bk_weak.clone(),
                            args.clone(),
                            ConstValue::None,
                        )));
                        memotable.borrow_mut().do_not_process = true;
                        memotable
                    }
                }
            });
            MemoFamily { uf, host_err }
        });

        // upstream: `_, _, memotable = memotables.find(firstvalues)`.
        let mut rep = family.uf.find(firstvalues.clone()).1;
        // upstream: `for values in possiblevalues: _, _, memotable =
        //            memotables.union(firstvalues, values)`.
        for values in possiblevalues.iter().skip(1) {
            rep = family.uf.union(firstvalues.clone(), values.clone()).1;
        }
        // Surface a host-call error the factory latched (single call per
        // argument tuple — no pre-flight).
        if let Some(e) = family.host_err.borrow_mut().take() {
            return Err(e);
        }
        family
            .uf
            .get(&rep)
            .expect("memo: representative MemoTable present after find/union")
            .clone()
    };

    let m = memotable.borrow();
    // upstream: `if memotable.graph is not None: return memotable.graph`.
    // `finish`'s synthesis is deferred so `graph` stays `None` and the
    // union path runs; this arm returns the graph for when `finish` lands,
    // and `MemoDesc::pycall` projects it to its return-var annotation
    // exactly as upstream `pycall` does for a FunctionGraph result.
    if let Some(graph) = m.graph.clone() {
        return Ok(SpecializeResult::Graph(graph));
    }
    // upstream: `else: return unionof(*[bookkeeper.immutablevalue(v)
    //            for v in memotable.table.values()])`.
    let mut results_s = Vec::with_capacity(m.table.len());
    for v in m.table.values() {
        results_s.push(bookkeeper.immutablevalue(v)?);
    }
    let s =
        unionof(results_s.iter()).map_err(|e| AnnotatorError::new(format!("memo unionof: {e}")))?;
    Ok(SpecializeResult::Annotation(Box::new(s)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeBool, SomeInteger, SomeValue};
    use crate::flowspace::model::{ConstValue, Constant};

    #[test]
    fn cartesian_product_empty_yields_one_empty_tuple() {
        // upstream: `if not lstlst: yield ()`.
        let out = cartesian_product::<i32>(&[]);
        assert_eq!(out, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn cartesian_product_single_list() {
        let out = cartesian_product(&[vec![1, 2, 3]]);
        assert_eq!(out, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn cartesian_product_orders_first_list_fastest() {
        // Mirrors the generator's emission order: tails come from the
        // tail product, the first list varies fastest within each.
        let out = cartesian_product(&[vec![1, 2], vec![3, 4]]);
        assert_eq!(out, vec![vec![1, 3], vec![2, 3], vec![1, 4], vec![2, 4]]);
    }

    #[test]
    fn all_values_constant_returns_single_value() {
        // upstream: `if s.is_constant(): return [s.const]`.
        let mut si = SomeInteger::default();
        si.base.const_box = Some(Constant::new(ConstValue::Int(7)));
        let s = SomeValue::Integer(si);
        let values = all_values(&s).unwrap();
        assert_eq!(values, vec![ConstValue::Int(7)]);
    }

    #[test]
    fn all_values_unknown_bool_returns_false_true() {
        // upstream: `elif isinstance(s, SomeBool): return [False, True]`.
        let s = SomeValue::Bool(SomeBool::new());
        let values = all_values(&s).unwrap();
        assert_eq!(
            values,
            vec![ConstValue::Bool(false), ConstValue::Bool(true)]
        );
    }

    #[test]
    fn all_values_impossible_returns_empty() {
        // upstream: `elif isinstance(s, SomeImpossibleValue): return []`.
        let values = all_values(&SomeValue::Impossible).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn all_values_non_pbc_non_bool_errors() {
        // upstream: `else: raise AnnotatorError("memo call: argument
        // must be a class or a frozen PBC, got %r")`.
        let s = SomeValue::Integer(SomeInteger::default());
        let err = all_values(&s).unwrap_err();
        assert!(err.to_string().contains("must be a class or a frozen PBC"));
    }

    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::Hlvalue;

    fn ann_bk() -> (
        Rc<RPythonAnnotator>,
        Rc<crate::annotator::bookkeeper::Bookkeeper>,
    ) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        (ann, bk)
    }

    fn as_int(w: &Hlvalue) -> i64 {
        match w {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(i) => *i,
                other => panic!("expected Int constant, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    fn as_bool(w: &Hlvalue) -> bool {
        match w {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Bool(b) => *b,
                other => panic!("expected Bool constant, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    /// `finish` on a one-bool-arg table synthesises a single dispatch
    /// block: `exitswitch` on the arg, with a False exit and a True exit,
    /// each returning the precomputed constant (specialize.py).
    #[test]
    fn finish_bool_arg_builds_exitswitch_dispatch() {
        let (_ann, bk) = ann_bk();
        // table: False -> 20, True -> 10.
        let table = Rc::new(RefCell::new(MemoTable::new(
            "f".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![ConstValue::Bool(false)],
            ConstValue::Int(20),
        )));
        table
            .borrow_mut()
            .table
            .insert(vec![ConstValue::Bool(true)], ConstValue::Int(10));

        MemoTable::finish(&table).expect("finish");

        let m = table.borrow();
        let pygraph = m.graph.as_ref().expect("graph synthesised");
        let graph = pygraph.graph.borrow();
        assert_eq!(graph.name, "memo_f_0");
        let startblock = graph.startblock.borrow();
        // exitswitch is the single bool arg.
        assert!(
            matches!(startblock.exitswitch, Some(Hlvalue::Variable(_))),
            "expected exitswitch on a Variable, got {:?}",
            startblock.exitswitch
        );
        assert_eq!(startblock.exits.len(), 2);
        for exit in &startblock.exits {
            let link = exit.borrow();
            // every exit targets the returnblock with one constant arg.
            assert!(Rc::ptr_eq(
                link.target.as_ref().expect("exit target"),
                &graph.returnblock
            ));
            let case = as_bool(link.exitcase.as_ref().expect("bool exitcase"));
            let ret = as_int(link.args[0].as_ref().expect("return constant"));
            // False -> 20, True -> 10.
            assert_eq!(ret, if case { 10 } else { 20 });
        }
    }

    /// A one-valued argument collapses to a constant return with no
    /// `exitswitch` (specialize.py:182-184, the `constants` branch of the
    /// single-value case).
    #[test]
    fn finish_single_value_arg_collapses_to_constant() {
        let (_ann, bk) = ann_bk();
        let table = Rc::new(RefCell::new(MemoTable::new(
            "g".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![ConstValue::Bool(false)],
            ConstValue::Int(5),
        )));

        MemoTable::finish(&table).expect("finish");

        let m = table.borrow();
        let pygraph = m.graph.as_ref().expect("graph synthesised");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.borrow();
        assert!(startblock.exitswitch.is_none());
        assert_eq!(startblock.exits.len(), 1);
        let link = startblock.exits[0].borrow();
        assert!(Rc::ptr_eq(
            link.target.as_ref().expect("exit target"),
            &graph.returnblock
        ));
        assert_eq!(as_int(link.args[0].as_ref().expect("return constant")), 5);
    }

    /// `finish` is idempotent-guarded: a second call once `graph` is set
    /// reports the "already finished" invariant (specialize.py).
    #[test]
    fn finish_twice_errors() {
        let (_ann, bk) = ann_bk();
        let table = Rc::new(RefCell::new(MemoTable::new(
            "h".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![ConstValue::Bool(false)],
            ConstValue::Int(1),
        )));
        MemoTable::finish(&table).expect("first finish");
        let err = MemoTable::finish(&table).expect_err("second finish errors");
        assert!(err.to_string().contains("already finished"));
    }

    /// `finish` on a `[PBC-set, bool]` table takes the `store = nextfns`
    /// branch (specialize.py): after fixing the PBC argument the
    /// remaining bool dispatch is not constant, so each PBC member's memo
    /// field holds a standalone subhelper *callable* (registered as a
    /// prebuilt graph), and the top dispatch reads the field with
    /// `getattr` then calls it with the remaining argument.
    #[test]
    fn finish_pbc_set_nonconstant_builds_nextfns_dispatch() {
        let (ann, bk) = ann_bk();
        let class_a = HostObject::new_class("A", Vec::new());
        let class_b = HostObject::new_class("B", Vec::new());
        // table: (A, False)->1, (A, True)->2, (B, False)->3, (B, True)->4.
        let table = Rc::new(RefCell::new(MemoTable::new(
            "k".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![
                ConstValue::HostObject(class_a.clone()),
                ConstValue::Bool(false),
            ],
            ConstValue::Int(1),
        )));
        {
            let mut m = table.borrow_mut();
            m.table.insert(
                vec![
                    ConstValue::HostObject(class_a.clone()),
                    ConstValue::Bool(true),
                ],
                ConstValue::Int(2),
            );
            m.table.insert(
                vec![
                    ConstValue::HostObject(class_b.clone()),
                    ConstValue::Bool(false),
                ],
                ConstValue::Int(3),
            );
            m.table.insert(
                vec![
                    ConstValue::HostObject(class_b.clone()),
                    ConstValue::Bool(true),
                ],
                ConstValue::Int(4),
            );
        }

        let prebuilt_before = ann.translator._prebuilt_graphs.borrow().len();
        MemoTable::finish(&table).expect("finish nextfns");

        // One standalone subhelper per PBC member was registered as a
        // prebuilt graph.
        assert_eq!(
            ann.translator._prebuilt_graphs.borrow().len() - prebuilt_before,
            2
        );

        let m = table.borrow();
        let pygraph = m.graph.as_ref().expect("graph synthesised");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.borrow();
        // The PBC dispatch is via a memo field (getattr + call), not an
        // exitswitch.
        assert!(startblock.exitswitch.is_none());
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert!(opnames.contains(&"getattr"), "ops: {opnames:?}");
        assert!(opnames.contains(&"simple_call"), "ops: {opnames:?}");

        // Each subhelper dispatches on its remaining bool argument.
        for sub in ann.translator._prebuilt_graphs.borrow().values() {
            let g = sub.graph.borrow();
            let sb = g.startblock.borrow();
            assert!(
                matches!(sb.exitswitch, Some(Hlvalue::Variable(_))),
                "subhelper should dispatch on its bool arg, got {:?}",
                sb.exitswitch
            );
        }
    }

    #[test]
    fn memotable_for_args_reuses_all_specializations_entry() {
        let bk = Bookkeeper::new();
        let key = 0xF4F4_F4F4;
        let args = vec![ConstValue::Int(1)];
        let t1 = memotable_for_args(&bk, key, "helper", args.clone(), |_| Ok(ConstValue::Int(7)))
            .expect("first");
        let t2 = memotable_for_args(&bk, key, "helper", args, |_| {
            panic!("compute_one_result must not rerun on a hit")
        })
        .expect("reuse");
        assert!(Rc::ptr_eq(&t1, &t2));
        assert_eq!(bk.all_specializations.borrow().len(), 1);
        assert_eq!(
            t1.borrow().table.get(&vec![ConstValue::Int(1)]),
            Some(&ConstValue::Int(7))
        );
    }
}
