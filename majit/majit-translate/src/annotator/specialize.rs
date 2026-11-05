//! Port of `rpython/annotator/specialize.py` specialization helpers.
//!
//! Module-level functions from specialize.py (`flatten_star_args`,
//! `default_specialize`, `memo`, `all_values`, `cartesian_product`, …)
//! live here. `FunctionDesc.specialize` dispatches into them.

use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};

use indexmap::IndexMap;
use indexmap::IndexSet;
use indexmap::map::Entry;

use super::bookkeeper::{Bookkeeper, PositionKey};
use super::description::{
    DescKey, FunctionDesc, GraphBuilder, GraphCacheKey, SourceGraph, SpecializeResult,
};
use super::model::{AnnotatorError, SomeInstance, SomeObjectTrait, SomeValue, not_const, unionof};
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, HostObject, Link,
    SpaceOperation, Variable, checkgraph,
};
use crate::flowspace::pygraph::PyGraph;
use crate::tool::algo::unionfind::{SharedUnionFindInfo, UnionFind};
use crate::translator::simplify;

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
/// Like upstream's generator, tuples are produced lazily. The first
/// list varies fastest within each tail, so `memo`'s
/// `possiblevalues.next()` picks the same `firstvalues` tuple as
/// upstream.
pub fn cartesian_product<'a, T: Clone + 'a>(
    lstlst: &'a [Vec<T>],
) -> Box<dyn Iterator<Item = Vec<T>> + 'a> {
    // upstream: `if not lstlst: yield (); return`.
    if lstlst.is_empty() {
        return Box::new(std::iter::once(Vec::new()));
    }
    // upstream: `for tuple_tail in cartesian_product(lstlst[1:]):`.
    Box::new(cartesian_product(&lstlst[1..]).flat_map(move |tail| {
        // upstream: `for value in lstlst[0]: yield (value,) + tuple_tail`.
        lstlst[0].iter().map(move |value| {
            let mut tuple = Vec::with_capacity(tail.len() + 1);
            tuple.push(value.clone());
            tuple.extend(tail.iter().cloned());
            tuple
        })
    }))
}

/// RPython `flatten_star_args(funcdesc, args_s)` (specialize.py).
///
/// Flattens the trailing `SomeTuple` for a `*arg` signature into
/// individual argument annotations, returning the new args_s and
/// an optional cache-key suffix `('star', N)` where `N` is the
/// flattened tuple length. For signatures without a `*arg` this
/// is the identity: `(args_s, None)`.
///
/// The star-args path uses an upstream-style builder closure to
/// rewrite the fresh graph so it no longer takes a `*arg` tuple.
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
    // upstream: `argnames, vararg, kwarg = funcdesc.signature`.
    let vararg = funcdesc.signature.varargname.clone();
    let kwarg = funcdesc.signature.kwargname.clone();
    // upstream: `assert not kwarg`.
    if kwarg.is_some() {
        return Err(AnnotatorError::new(format!(
            "{}: functions with ** arguments are not supported",
            funcdesc.name
        )));
    }
    if vararg.is_none() {
        // upstream: `return args_s, None, None`.
        return Ok((args_s.to_vec(), GraphCacheKey::None, None));
    }
    // upstream: `assert len(args_s) == len(argnames) + 1`.
    let argnames_len = funcdesc.signature.argnames.len();
    if args_s.len() != argnames_len + 1 {
        return Err(AnnotatorError::new(format!(
            "{}.flatten_star_args: expected {} args, got {}",
            funcdesc.name,
            argnames_len + 1,
            args_s.len()
        )));
    }
    // upstream: `s_tuple = args_s[-1]; assert isinstance(s_tuple, SomeTuple)`.
    // `specialize.py` reads `args_s[-1]` directly; an unbound
    // (None) cell would raise AttributeError on `isinstance`.
    // Pyre's `.expect()` on the trailing *arg slot mirrors that
    // fail-loud surface.
    let s_tuple = args_s[args_s.len() - 1].as_ref().ok_or_else(|| {
        AnnotatorError::new(format!(
            "{}.flatten_star_args: trailing *arg slot is unannotated",
            funcdesc.name
        ))
    })?;
    let tuple = match s_tuple {
        SomeValue::Tuple(t) => t,
        other => {
            return Err(AnnotatorError::new(format!(
                "{}.flatten_star_args: *arg is {:?}, expected SomeTuple",
                funcdesc.name, other
            )));
        }
    };
    // upstream: `s_len = s_tuple.len(); assert s_len.is_constant();
    //             nb_extra_args = s_len.const`.
    let nb_extra_args = tuple.items.len();
    // upstream: `flattened_s = list(args_s[:-1]); flattened_s.extend(s_tuple.items)`.
    let mut flattened_s: Vec<Option<SomeValue>> = args_s[..args_s.len() - 1].to_vec();
    flattened_s.extend(tuple.items.iter().cloned().map(Some));
    // upstream: `key = ('star', nb_extra_args)`.
    let key_suffix = GraphCacheKey::Tuple(vec![
        GraphCacheKey::String("star".to_string()),
        GraphCacheKey::Int(nb_extra_args as i64),
    ]);
    let builder_name = funcdesc.name.clone();
    let builder: Option<GraphBuilder<'a>> = Some(Box::new(
        move |translator: &crate::translator::translator::TranslationContext, func: HostObject| {
            let pygraph = translator.buildflowgraph(func, false).map_err(|err| {
                AnnotatorError::new(format!("FunctionDesc.buildgraph({}): {err}", builder_name))
            })?;
            apply_star_args_builder(pygraph.as_ref(), nb_extra_args)?;
            Ok(pygraph)
        },
    ));
    Ok((flattened_s, key_suffix, builder))
}

/// RPython `default_specialize(funcdesc, args_s)` (specialize.py).
pub fn default_specialize(
    funcdesc: &FunctionDesc,
    args_s: &mut Vec<Option<SomeValue>>,
) -> Result<Rc<PyGraph>, AnnotatorError> {
    // upstream `default_specialize(funcdesc, args_s)`
    // (specialize.py).  RPython iterates `args_s` and only
    // touches entries that are `SomeInstance`; unbound (None)
    // cells naturally fall through the `isinstance(s_obj,
    // SomeInstance)` check.  Pyre mirrors that by skipping `None`
    // slots in the access_directly scan.
    let (flattened_cells, mut key, builder) = flatten_star_args(funcdesc, args_s)?;
    let mut flattened_cells = flattened_cells;
    let inputcells: &mut [Option<SomeValue>] = if !matches!(key, GraphCacheKey::None) {
        flattened_cells.as_mut_slice()
    } else {
        args_s.as_mut_slice()
    };

    let jit_look_inside = funcdesc
        .base
        .pyobj
        .as_ref()
        .and_then(|pyobj| pyobj.user_function())
        .and_then(|func| func._jit_look_inside_)
        .unwrap_or(true);
    let mut access_directly = false;
    for slot in inputcells.iter_mut() {
        let Some(s_obj) = slot else {
            continue;
        };
        if let SomeValue::Instance(inst) = s_obj {
            if inst.flags.contains_key("access_directly") {
                if jit_look_inside {
                    access_directly = true;
                    key = GraphCacheKey::Tuple(vec![GraphCacheKey::AccessDirect, key.clone()]);
                    break;
                } else {
                    let mut new_flags = inst.flags.clone();
                    new_flags.remove("access_directly");
                    *slot = Some(SomeValue::Instance(SomeInstance::new(
                        inst.classdef.clone(),
                        inst.can_be_none,
                        new_flags,
                    )));
                }
            }
        }
    }
    let graph = funcdesc.cachedgraph(key, None, builder)?;
    if access_directly {
        graph.access_directly.set(true);
    }
    Ok(graph)
}

/// RPython `getuniquenondirectgraph(desc)` (specialize.py).
///
/// Returns the only cached graph whose specialization key is not
/// `(AccessDirect, key)`.
pub fn getuniquenondirectgraph(desc: &FunctionDesc) -> Result<Rc<PyGraph>, AnnotatorError> {
    // `FunctionDesc.getgraphs` reads `_cache` alone.
    let mut result = Vec::new();
    for (key, graph) in desc.cache.borrow().iter() {
        if !key.is_access_direct() {
            result.push(graph.clone());
        }
    }
    if result.len() != 1 {
        return Err(AnnotatorError::new(format!(
            "getuniquenondirectgraph({}): expected one non-direct graph, got {}",
            desc.name,
            result.len()
        )));
    }
    Ok(result.pop().expect("length checked"))
}

/// RPython `maybe_star_args(funcdesc, key, args_s)` (specialize.py).
pub fn maybe_star_args(
    funcdesc: &FunctionDesc,
    key: GraphCacheKey,
    args_s: &[Option<SomeValue>],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let (_flattened_s, star_key, builder) = flatten_star_args(funcdesc, args_s)?;
    let combined_key = funcdesc.append_cache_key(key, star_key)?;
    funcdesc.cachedgraph(combined_key, None, builder)
}

/// RPython `specialize_argvalue(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_argvalue(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[usize],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let mut key_parts = Vec::with_capacity(argindices.len());
    for &idx in argindices {
        let s_slot = args_s.get(idx).ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:arg index {idx} out of range for {}",
                funcdesc.name
            ))
        })?;
        // upstream `specialize_argvalue(funcdesc, args_s, *parms)`
        // (specialize.py) reaches `args_s[i].is_constant()`
        // / `args_s[i].const`; an unbound cell raises AttributeError
        // there.  Mirror with fail-loud at access.
        let s = s_slot.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!("specialize:arg({idx}): argument is unannotated"))
        })?;
        // `specialize_argvalue`: `if s.is_constant(): key.append(s.const)`
        // then the single-desc SomePBC arm (a constant function/class/frozen
        // PBC must key by `s.const`, not the description).
        if s.is_constant() {
            let value = s.const_().expect("constant annotation missing const_box");
            key_parts.push(GraphCacheKey::Const(value.clone()));
        } else if let SomeValue::PBC(pbc) = s
            && pbc.descriptions.len() == 1
        {
            // for test_specialize_arg_bound_method
            let desc = pbc.any_description().expect("single-desc PBC");
            key_parts.push(GraphCacheKey::Desc {
                key: desc.desc_key(),
                name: FunctionDesc::nameof_desc(desc),
            });
        } else {
            return Err(AnnotatorError::new(format!(
                "specialize:arg({idx}): argument not constant: {s:?}"
            )));
        }
    }
    maybe_star_args(funcdesc, GraphCacheKey::Tuple(key_parts), args_s)
}

/// RPython `specialize_arg_or_var(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_arg_or_var(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[usize],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    for &idx in argindices {
        let s_slot = args_s.get(idx).ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:arg_or_var index {idx} out of range for {}",
                funcdesc.name
            ))
        })?;
        let s = s_slot.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:arg_or_var({idx}): argument is unannotated"
            ))
        })?;
        if !s.is_constant() {
            return maybe_star_args(funcdesc, GraphCacheKey::None, args_s);
        }
    }
    specialize_argvalue(funcdesc, args_s, argindices)
}

/// RPython `specialize_argtype(funcdesc, args_s, *argindices)`
/// (specialize.py).
pub fn specialize_argtype(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[usize],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let mut key_parts = Vec::with_capacity(argindices.len());
    for &idx in argindices {
        let s_slot = args_s.get(idx).ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:argtype index {idx} out of range for {}",
                funcdesc.name
            ))
        })?;
        let s = s_slot.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:argtype({idx}): argument is unannotated"
            ))
        })?;
        // `specialize.py` `specialize_argtype`: `args_s[i].knowntype`.
        // `FunctionDesc.cachedgraph` formats a `ClassDesc` through
        // `.name` while cache identity stays the classdesc object.
        key_parts.push(FunctionDesc::argtype_knowntype_key(s));
    }
    maybe_star_args(funcdesc, GraphCacheKey::Tuple(key_parts), args_s)
}

/// Bodyless foreign stub: no source template (pending lift or built
/// graph) and no `HostCode`. `FunctionDesc.cachedgraph` with
/// `builder is None` copies `source_template()` (`translator.buildflowgraph`
/// analogue); a bodyless `specialize:argtype` external must take the
/// builder stub path. An empty `_cache` is not the test — the second
/// knowntype must still take this path after the first graph is cached.
pub(crate) fn is_bodyless_argtype_stub(funcdesc: &FunctionDesc) -> bool {
    if funcdesc.source_graph.borrow().is_some() {
        return false;
    }
    let has_host_code = funcdesc
        .base
        .pyobj
        .as_ref()
        .and_then(|pyobj| pyobj.user_function())
        .and_then(|func| func.code.as_ref())
        .is_some();
    !has_host_code
}

/// Specialize-miss path for a bodyless `specialize:argtype` external.
///
/// `specialize_argtype` itself stays line-by-line with
/// `specialize.py`. A stub has no body to re-annotate, so the miss
/// constructs a graph whose result shell is `not_const(args_s[i])`
/// (`EntryNonConstant.compute_result_annotation`, `rlib/nonconst.py`;
/// `ExtRegistryEntry.compute_result_annotation`, `extregistry.py`).
pub fn specialize_argtype_stub_result(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    argindices: &[usize],
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let mut key_parts = Vec::with_capacity(argindices.len());
    let mut result_shell: Option<SomeValue> = None;
    for &idx in argindices {
        let s_slot = args_s.get(idx).ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:argtype stub index {idx} out of range for {}",
                funcdesc.name
            ))
        })?;
        let s = s_slot.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "specialize:argtype stub({idx}): argument is unannotated"
            ))
        })?;
        key_parts.push(FunctionDesc::argtype_knowntype_key(s));
        if result_shell.is_none() {
            let mut shell = not_const(s);
            if let SomeValue::Instance(inst) = &mut shell {
                // Nullable `*mut T` (`raw_ptr_token_annotation`).
                inst.can_be_none = true;
            }
            result_shell = Some(shell);
        }
    }
    let result_shell = result_shell.ok_or_else(|| {
        AnnotatorError::new(format!(
            "specialize:argtype stub for {} requires at least one index",
            funcdesc.name
        ))
    })?;
    let name = funcdesc.name.clone();
    let signature = funcdesc.signature.clone();
    let builder: GraphBuilder<'_> = Box::new(move |_translator, _func| {
        Ok(
            crate::translator::rtyper::cutover::build_stub_pygraph_for_somevalue(
                name,
                signature,
                result_shell,
            ),
        )
    });
    funcdesc.cachedgraph(GraphCacheKey::Tuple(key_parts), None, Some(builder))
}

/// RPython `specialize_arglistitemtype(funcdesc, args_s, i)`
/// (specialize.py).
pub fn specialize_arglistitemtype(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    i: usize,
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let idx = i;
    // `specialize.py specialize_arglistitemtype` reads
    // `args_s[i].knowntype` directly — if the slot is None,
    // Python raises `AttributeError: 'NoneType' object has no
    // attribute 'knowntype'`.  Mirror the fail-loud rather than
    // folding None into a `GraphCacheKey::None` bucket that would
    // silently collapse cache identity across unrelated unbound
    // callers.
    let slot = args_s.get(idx).ok_or_else(|| {
        AnnotatorError::new(format!(
            "specialize:arglistitemtype index {idx} out of range \
             (inputcells len {})",
            args_s.len()
        ))
    })?;
    let s = slot.as_ref().ok_or_else(|| {
        AnnotatorError::new(format!(
            "specialize:arglistitemtype: args_s[{idx}] is None \
             (specialize_arglistitemtype reads .knowntype directly — \
             caller must bind every slot before specialising)"
        ))
    })?;
    let key = match s {
        SomeValue::List(list) => GraphCacheKey::KnownType(list.listdef.s_value().knowntype()),
        _ => GraphCacheKey::None,
    };
    maybe_star_args(funcdesc, key, args_s)
}

/// RPython `specialize_call_location(funcdesc, args_s, op)`
/// (specialize.py).
pub fn specialize_call_location(
    funcdesc: &FunctionDesc,
    args_s: &[Option<SomeValue>],
    op: Option<PositionKey>,
) -> Result<Rc<PyGraph>, AnnotatorError> {
    let position_key = op.ok_or_else(|| {
        AnnotatorError::new("specialize_call_location: missing call-site position")
    })?;
    let key = GraphCacheKey::Tuple(vec![GraphCacheKey::Position(position_key)]);
    maybe_star_args(funcdesc, key, args_s)
}

/// Mutation step inside the upstream `builder(translator, func)`
/// closure from `specialize.py`.
fn apply_star_args_builder(pygraph: &PyGraph, nb_extra_args: usize) -> Result<(), AnnotatorError> {
    use crate::flowspace::model::{Block, Hlvalue, Link, Variable};
    use crate::flowspace::operation::OpKind;

    // upstream: `argnames, vararg, kwarg = graph.signature; assert vararg`.
    let old_sig = pygraph.signature.borrow().clone();
    if old_sig.varargname.is_none() {
        return Err(AnnotatorError::new(format!(
            "apply_star_args_builder({}): graph signature missing *arg",
            pygraph.graph.borrow().name
        )));
    }
    if old_sig.kwargname.is_some() {
        return Err(AnnotatorError::new(format!(
            "apply_star_args_builder({}): graph signature unexpectedly has **arg",
            pygraph.graph.borrow().name
        )));
    }
    let mut graph = pygraph.graph.borrow_mut();
    // upstream: `argscopy = [Variable(v) for v in graph.getargs()]`.
    // `Variable(v)` is `Variable.rename` of `v._name` (the prefix).
    let old_args = graph.getargs();
    let mut argscopy: Vec<Variable> = Vec::with_capacity(old_args.len());
    for hl in &old_args {
        match hl {
            Hlvalue::Variable(v) => {
                let mut copy = Variable::new();
                copy.rename_from(v);
                argscopy.push(copy);
            }
            _ => {
                return Err(AnnotatorError::new(format!(
                    "apply_star_args_builder({}): startblock arg is not a Variable",
                    graph.name
                )));
            }
        }
    }
    // upstream: `starargs = [Variable('stararg%d'%i) for i in range(nb_extra_args)]`.
    let starargs: Vec<Variable> = (0..nb_extra_args)
        .map(|i| Variable::named(format!("stararg{}", i)))
        .collect();
    // upstream: `newstartblock = Block(argscopy[:-1] + starargs)`.
    let mut new_inputargs: Vec<Hlvalue> = argscopy[..argscopy.len() - 1]
        .iter()
        .map(|v| Hlvalue::Variable(v.clone()))
        .collect();
    new_inputargs.extend(starargs.iter().map(|v| Hlvalue::Variable(v.clone())));
    let newstartblock = Block::shared(new_inputargs);
    // upstream: `newtup = op.newtuple(*starargs); newtup.result = argscopy[-1]`.
    //
    // `HLOperation` is the flow-time handler object; the persisted
    // form is `SpaceOperation(opname, args, result)`. The annotator
    // consumes `operations: Vec<SpaceOperation>`, so emit the
    // SpaceOperation form directly.
    let star_args_hl: Vec<Hlvalue> = starargs
        .iter()
        .map(|v| Hlvalue::Variable(v.clone()))
        .collect();
    let newtup_result = Hlvalue::Variable(argscopy[argscopy.len() - 1].clone());
    let newtup_sp = crate::flowspace::model::SpaceOperation::new(
        OpKind::NewTuple.opname(),
        star_args_hl,
        newtup_result,
    );
    newstartblock.borrow_mut().operations.push(newtup_sp);
    // upstream: `newstartblock.closeblock(Link(argscopy, graph.startblock))`.
    let link_args: Vec<Hlvalue> = argscopy
        .iter()
        .map(|v| Hlvalue::Variable(v.clone()))
        .collect();
    let link = Link::new(link_args, Some(graph.startblock.clone()), None).into_ref();
    newstartblock.closeblock(vec![link]);
    // upstream: `graph.startblock = newstartblock`.
    graph.startblock = newstartblock;
    drop(graph);

    let mut argnames = old_sig.argnames;
    argnames.extend((0..nb_extra_args).map(|i| format!(".star{}", i)));
    *pygraph.signature.borrow_mut() = Signature::new(argnames, None, None);
    if nb_extra_args > 0 {
        *pygraph.defaults.borrow_mut() = None;
    }
    checkgraph(&pygraph.graph.borrow());
    Ok(())
}

/// A memo argument value under Python dict-key equality. `memo`'s
/// containers (`MemoTable.table`, `finish`'s `sets`, the
/// `all_specializations[funcdesc]` `UnionFind`) and `nextargvalues == [False,
/// True]` compare argument values the Python way: `False == 0 == 0.0`,
/// `True == 1 == 1.0`, with equal hashes, and tuples element-wise.
/// `ConstValue` equality keeps those variants apart.
#[derive(Clone, Debug)]
pub struct MemoKey(pub ConstValue);

/// A numeric memo argument value as the exact number Python compares: `UInt` only above `i128::MAX`, `Float` only for non-integral or out-of-range floats.
#[derive(PartialEq, Eq, Hash)]
enum PyNumber {
    Int(i128),
    UInt(u128),
    Float(u64),
}

/// The exact number a numeric `ConstValue` equals under Python `==`, `None` for non-numbers.
fn py_number(value: &ConstValue) -> Option<PyNumber> {
    match value {
        ConstValue::Bool(b) => Some(PyNumber::Int(*b as i128)),
        ConstValue::Int(v) => Some(PyNumber::Int(*v as i128)),
        ConstValue::UInt(v) => Some(PyNumber::Int(*v as i128)),
        ConstValue::Int128(v) => Some(PyNumber::Int(*v)),
        ConstValue::UInt128(v) => {
            if *v <= i128::MAX as u128 {
                Some(PyNumber::Int(*v as i128))
            } else {
                Some(PyNumber::UInt(*v))
            }
        }
        ConstValue::RInt(c, v, _) => match crate::rlib::rarithmetic::rint_value(*c, *v) {
            crate::rlib::rarithmetic::RIntValue::Signed(x) => Some(PyNumber::Int(x)),
            crate::rlib::rarithmetic::RIntValue::Unsigned(x) => {
                if x <= i128::MAX as u128 {
                    Some(PyNumber::Int(x as i128))
                } else {
                    Some(PyNumber::UInt(x))
                }
            }
        },
        ConstValue::Float(bits) => {
            let f = f64::from_bits(*bits);
            // `float == int` compares exactly: an integral float maps to the
            // integer it equals. `2**127` and `2**128` are exact as `f64`, so the
            // half-open bounds keep each cast exact (no saturation).
            const TWO_127: f64 = (1u128 << 127) as f64;
            const TWO_128: f64 = 2.0 * TWO_127;
            if f.fract() == 0.0 && f >= -TWO_127 && f < TWO_127 {
                Some(PyNumber::Int(f as i128))
            } else if f.fract() == 0.0 && f >= TWO_127 && f < TWO_128 {
                Some(PyNumber::UInt(f as u128))
            } else {
                Some(PyNumber::Float(*bits))
            }
        }
        _ => None,
    }
}

impl PartialEq for MemoKey {
    fn eq(&self, other: &Self) -> bool {
        match (py_number(&self.0), py_number(&other.0)) {
            (Some(a), Some(b)) => a == b,
            _ => match (&self.0, &other.0) {
                (ConstValue::Tuple(a), ConstValue::Tuple(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| MemoKey(x.clone()) == MemoKey(y.clone()))
                }
                _ => self.0 == other.0,
            },
        }
    }
}

impl Eq for MemoKey {}

impl Hash for MemoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(n) = py_number(&self.0) {
            0u8.hash(state);
            n.hash(state);
        } else if let ConstValue::Tuple(items) = &self.0 {
            1u8.hash(state);
            items.len().hash(state);
            for item in items {
                MemoKey(item.clone()).hash(state);
            }
        } else {
            2u8.hash(state);
            self.0.hash(state);
        }
    }
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
    /// RPython `self.table = {args: value}` (specialize.py). Insertion-
    /// ordered so `finish`'s `self.table.iteritems().next()` is the first
    /// inserted entry, matching a PyPy dict. Keys use [`MemoKey`]
    /// equality.
    pub table: IndexMap<Vec<MemoKey>, ConstValue>,
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
        args: Vec<MemoKey>,
        value: ConstValue,
    ) -> Self {
        // upstream: `self.table = {args: value}`.
        let mut table = IndexMap::new();
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
        // upstream: `bookkeeper = self.funcdesc.bookkeeper`.
        let bookkeeper = this
            .borrow()
            .bookkeeper
            .upgrade()
            .expect("MemoTable.register_finish: bookkeeper dropped");
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
    /// Builds the entry helper (`make_subhelper(args_so_far=())`) the way
    /// `translator.buildflowgraph` builds from source: write the body,
    /// `simplify_graph`, append to `translator.graphs`. Child helpers are
    /// installed as the funcdesc's pending source graph and built when
    /// annotation first observes the call (`FunctionDesc.cachedgraph` →
    /// `buildgraph` → `buildflowgraph`).
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
        let mut sets: Vec<IndexSet<MemoKey>> = vec![IndexSet::new(); nbargs];
        for args in table.keys() {
            for (i, slot) in sets.iter_mut().enumerate() {
                slot.insert(args[i].clone());
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

        let synth = MemoSynth {
            bookkeeper: &bookkeeper,
            name: &name,
            table: &table,
            example_value: &example_value,
            sets: &sets,
            nbargs,
            argnames: &argnames,
        };
        // upstream: entrypoint = make_subhelper(args_so_far = ())
        let entrypoint = synth.make_subhelper(&[])?;
        // upstream: self.graph = annotator.translator.buildflowgraph(entrypoint)
        // — the source path builds the graph, simplifies it and appends it to
        // `translator.graphs`; the helpers it calls are built when annotation
        // reaches them.
        let pygraph = (entrypoint.graph)()?;
        annotator
            .translator
            .graphs
            .borrow_mut()
            .push(pygraph.graph.clone());
        // upstream: self.graph.defaults = self.funcdesc.defaults
        *pygraph.defaults.borrow_mut() = Some(defaults);
        this.borrow_mut().graph = Some(pygraph.clone());

        // upstream: `args_s = []; for arg_types in sets: values_s =
        // [bookkeeper.immutablevalue(x) for x in arg_types]; args_s.append(
        // unionof(*values_s))`.
        let mut args_s: Vec<Option<SomeValue>> = Vec::with_capacity(nbargs);
        for arg_types in &sets {
            let mut values_s = Vec::with_capacity(arg_types.len());
            for x in arg_types {
                values_s.push(bookkeeper.immutablevalue(&x.0)?);
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

/// The pending build of a helper's graph: upstream `build_flow(f)` of the
/// helper source followed by `simplify_graph`.
type HelperGraph = Box<dyn FnOnce() -> Result<Rc<PyGraph>, AnnotatorError>>;

/// A memo helper: `make_helper`'s function `f`, the pending build of its
/// graph, and `make_constant_subhelper`'s `f.constant_result` (`None` when
/// the attribute is absent).
struct MemoHelper {
    f: HostObject,
    graph: HelperGraph,
    constant_result: Option<ConstValue>,
}

/// Bundles the read-only inputs threaded through `make_subhelper`'s
/// recursion (specialize.py).
struct MemoSynth<'a> {
    bookkeeper: &'a Rc<Bookkeeper>,
    name: &'a str,
    table: &'a IndexMap<Vec<MemoKey>, ConstValue>,
    example_value: &'a ConstValue,
    sets: &'a [IndexSet<MemoKey>],
    nbargs: usize,
    argnames: &'a [String],
}

impl MemoSynth<'_> {
    /// RPython `make_helper(firstarg, stmt, miniglobals)` (specialize.py).
    ///
    /// The helper body is written into a fresh graph (startblock +
    /// returnblock) standing for the source upstream `exec`s, and
    /// [`synthetic_subhelper_code`] carries its signature. The returned
    /// closure builds that graph (then `simplify_graph`) when the helper
    /// is first observed.
    fn make_helper(
        &self,
        firstarg: usize,
        fill: impl FnOnce(&FunctionGraph, &[Hlvalue]) + 'static,
    ) -> MemoHelper {
        // upstream: `def f(argnames[firstarg:])`.
        let sub_argnames: Vec<String> = self.argnames[firstarg..self.nbargs].to_vec();
        // upstream: `func_with_new_name(f, 'memo_%s_%d' % (name, firstarg))`.
        let full_name = format!("memo_{}_{}", self.name, firstarg);
        let mut func = GraphFunc::new(
            &full_name,
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.code = Some(Box::new(synthetic_subhelper_code(
            &full_name,
            &sub_argnames,
        )));
        let host = HostObject::new_user_function(func.clone());
        MemoHelper {
            f: host,
            graph: Box::new(move || {
                let inputs: Vec<Hlvalue> = sub_argnames
                    .iter()
                    .map(|n| Hlvalue::Variable(Variable::named(n)))
                    .collect();
                let mut fg = FunctionGraph::new(&full_name, Block::shared(inputs.clone()));
                // upstream `PyGraph.__init__`: `self.func = func`
                fg.func = Some(func.clone());
                fill(&fg, &inputs);
                // upstream: buildflowgraph — simplify.simplify_graph(graph)
                simplify::simplify_graph(&fg, None);
                Ok(Rc::new(PyGraph {
                    graph: fg.into_ref(),
                    func,
                    signature: RefCell::new(Signature::new(sub_argnames, None, None)),
                    defaults: RefCell::new(Some(Vec::new())),
                    access_directly: Cell::new(false),
                }))
            }),
            constant_result: None,
        }
    }

    /// RPython `make_constant_subhelper` via
    /// `make_helper(firstarg, 'return result', {'result': result})`.
    fn make_constant_subhelper(&self, firstarg: usize, result: ConstValue) -> MemoHelper {
        let mut helper = self.make_helper(firstarg, {
            let result = result.clone();
            move |graph, _inputs| {
                graph.startblock.closeblock(vec![
                    Link::new(
                        vec![Hlvalue::Constant(Constant::new(result))],
                        Some(graph.returnblock.clone()),
                        None,
                    )
                    .into_ref(),
                ]);
            }
        });
        // upstream: f.constant_result = result
        helper.constant_result = Some(result);
        helper
    }

    /// Turn a [`MemoHelper`] into the function a caller references. Annotation
    /// builds its graph when it reaches the call (upstream `FunctionDesc.cachedgraph`
    /// → `buildgraph` → `buildflowgraph`), so the build is installed as the
    /// funcdesc's pending source graph.
    fn callable(&self, helper: MemoHelper) -> Result<HostObject, AnnotatorError> {
        let f = helper.f;
        let graph = helper.graph;
        let desc = self.bookkeeper.getdesc(&f)?;
        let funcdesc = desc
            .as_function()
            .ok_or_else(|| AnnotatorError::new("memo helper is not a function"))?;
        funcdesc
            .borrow()
            .set_source_graph(SourceGraph::new(move || graph().map_err(|e| e.to_string())));
        Ok(f)
    }

    /// RPython `make_subhelper(args_so_far)` (specialize.py).
    fn make_subhelper(&self, args_so_far: &[MemoKey]) -> Result<MemoHelper, AnnotatorError> {
        let firstarg = args_so_far.len();
        // upstream: `if firstarg == nbargs:` — no argument left, return
        // the known result (or a dummy value if none corresponds exactly).
        if firstarg == self.nbargs {
            // upstream: `result = self.table.get(args_so_far, example_value)`.
            let result = self
                .table
                .get(args_so_far)
                .cloned()
                .unwrap_or_else(|| self.example_value.clone());
            // upstream: `return make_constant_subhelper(firstarg, result)`.
            return Ok(self.make_constant_subhelper(firstarg, result));
        }

        // upstream: `nextargvalues = list(sets[len(args_so_far)])`;
        // `if nextargvalues == [True, False]: nextargvalues = [False, True]`.
        let nextargvalues =
            normalize_bool_order(&self.sets[firstarg].iter().cloned().collect::<Vec<_>>());
        // upstream: `nextfns = [make_subhelper(args_so_far + (arg,))
        //            for arg in nextargvalues]`.
        let mut nextfns = Vec::with_capacity(nextargvalues.len());
        for arg in &nextargvalues {
            let mut next = args_so_far.to_vec();
            next.push(arg.clone());
            nextfns.push(self.make_subhelper(&next)?);
        }
        // upstream: `constants = [fn.constant_result for fn in nextfns]`
        // except `AttributeError` → `constants = None`.
        let constants = {
            let mut cs = Vec::with_capacity(nextfns.len());
            let mut all = true;
            for fn_ in &nextfns {
                match fn_.constant_result.as_ref() {
                    Some(c) => cs.push(c.clone()),
                    None => {
                        all = false;
                        break;
                    }
                }
            }
            if all { Some(cs) } else { None }
        };
        // upstream: `restargs = ', '.join(argnames[firstarg+1:])` — in
        // the helper graph these are its own startblock inputs `[1..]`.

        // upstream: `if len(nextargvalues) == 1:`.
        if nextargvalues.len() == 1 {
            if let Some(constants) = constants {
                // upstream: `result = constants[0];
                // return make_constant_subhelper(firstarg, result)`.
                return Ok(self.make_constant_subhelper(
                    firstarg,
                    constants
                        .into_iter()
                        .next()
                        .expect("len(nextargvalues) == 1"),
                ));
            }
            // upstream: `stmt = 'return subhelper(%s)' % restargs`;
            // `return make_helper(firstarg, stmt, {'subhelper': nextfns[0]})`.
            let subhelper =
                self.callable(nextfns.into_iter().next().expect("len(nextargvalues) == 1"))?;
            return Ok(self.make_helper(firstarg, move |graph, inputs| {
                let v = Hlvalue::Variable(Variable::new());
                let mut args = vec![Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                    subhelper,
                )))];
                args.extend(inputs[1..].iter().cloned());
                graph
                    .startblock
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new("simple_call", args, v.clone()));
                graph.startblock.closeblock(vec![
                    Link::new(vec![v], Some(graph.returnblock.clone()), None).into_ref(),
                ]);
            }));
        }

        // upstream: `elif nextargvalues == [False, True]:`.
        if is_false_true(&nextargvalues) {
            // upstream: False first then True; `nextfns[0]` is the False
            // helper, `nextfns[1]` the True helper.
            let mut nextfns = nextfns.into_iter();
            let mut cases: Vec<(bool, Result<ConstValue, HostObject>)> = Vec::with_capacity(2);
            for case in [false, true] {
                let helper = nextfns.next().expect("bool nextfns is [False, True]");
                let arm = match helper.constant_result.clone() {
                    Some(result) => Ok(result),
                    None => Err(self.callable(helper)?),
                };
                cases.push((case, arm));
            }
            let call_argnames: Vec<String> = self.argnames[firstarg + 1..self.nbargs].to_vec();
            return Ok(self.make_helper(firstarg, move |graph, inputs| {
                // upstream: `if aN:` flows through `POP_JUMP_IF_FALSE` → `op.bool(aN)`,
                // and the block switches on its result.
                let w_bool = Hlvalue::Variable(Variable::new());
                {
                    let mut startblock = graph.startblock.borrow_mut();
                    startblock.operations.push(SpaceOperation::new(
                        "bool",
                        vec![inputs[0].clone()],
                        w_bool.clone(),
                    ));
                    startblock.exitswitch = Some(w_bool);
                }
                let mut exits = Vec::with_capacity(2);
                for (case, arm) in cases {
                    let exitcase = Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(case))));
                    match arm {
                        Ok(result) => {
                            // upstream: `return case` — link the constant
                            // straight to the returnblock.
                            exits.push(
                                Link::new(
                                    vec![Hlvalue::Constant(Constant::new(result))],
                                    Some(graph.returnblock.clone()),
                                    exitcase,
                                )
                                .into_ref(),
                            );
                        }
                        Err(host) => {
                            // upstream: `return case0/case1(%s) % restargs`.
                            let restargs = inputs[1..].to_vec();
                            let call_inputs: Vec<Hlvalue> = call_argnames
                                .iter()
                                .map(|n| Hlvalue::Variable(Variable::named(n)))
                                .collect();
                            let call_block = Block::shared(call_inputs.clone());
                            let v = Hlvalue::Variable(Variable::new());
                            let mut call_args = vec![Hlvalue::Constant(Constant::new(
                                ConstValue::HostObject(host),
                            ))];
                            call_args.extend(call_inputs.iter().cloned());
                            call_block.borrow_mut().operations.push(SpaceOperation::new(
                                "simple_call",
                                call_args,
                                v.clone(),
                            ));
                            call_block.closeblock(vec![
                                Link::new(vec![v], Some(graph.returnblock.clone()), None)
                                    .into_ref(),
                            ]);
                            exits.push(Link::new(restargs, Some(call_block), exitcase).into_ref());
                        }
                    }
                }
                graph.startblock.closeblock(exits);
            }));
        }

        // upstream: `else:` — the arg is a set of PBCs.
        // upstream: `descs = [bookkeeper.getdesc(pbc) for pbc in nextargvalues]`.
        let mut descs = Vec::with_capacity(nextargvalues.len());
        for v in &nextargvalues {
            let host = match &v.0 {
                ConstValue::HostObject(h) => h.clone(),
                other => {
                    return Err(AnnotatorError::new(format!(
                        "memo finish: PBC set member is not a host object: {other:?}"
                    )));
                }
            };
            descs.push(self.bookkeeper.getdesc(&host)?);
        }
        // upstream: `fieldname = self.getuniquefieldname()`.
        let fieldname = MemoTable::getuniquefieldname(self.name);
        // upstream: `store = constants` or, without them, `nextfns`.
        let mut store = Vec::with_capacity(nextfns.len());
        if let Some(constants) = &constants {
            store.extend(constants.iter().cloned());
        } else {
            for helper in nextfns {
                store.push(ConstValue::HostObject(self.callable(helper)?));
            }
        }
        // upstream: `for desc, value_to_store in zip(descs, store):
        // desc.create_new_attribute(fieldname, value_to_store)`.
        for (desc, value_to_store) in descs.iter().zip(store) {
            desc.create_new_attribute(&fieldname, value_to_store)?;
        }
        // upstream: `stmt = 'return getattr(%s, %r)' % (...)` and, when
        // there are no constants, `stmt += '(%s)' % restargs`.
        let has_constants = constants.is_some();
        Ok(self.make_helper(firstarg, move |graph, inputs| {
            let attr = Hlvalue::Variable(Variable::new());
            graph
                .startblock
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getattr",
                    vec![
                        inputs[0].clone(),
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&fieldname))),
                    ],
                    attr.clone(),
                ));
            let result = if has_constants {
                attr
            } else {
                let call_result = Hlvalue::Variable(Variable::new());
                let mut call_args = Vec::with_capacity(1 + inputs.len().saturating_sub(1));
                call_args.push(attr);
                call_args.extend(inputs[1..].iter().cloned());
                graph
                    .startblock
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "simple_call",
                        call_args,
                        call_result.clone(),
                    ));
                call_result
            };
            graph.startblock.closeblock(vec![
                Link::new(vec![result], Some(graph.returnblock.clone()), None).into_ref(),
            ]);
        }))
    }
}

/// A minimal `HostCode` whose `co_argcount` / `co_varnames` carry the
/// subhelper parameter list so `cpython_code_signature` derives the
/// matching `Signature`. The bytecode body is empty — the graph is
/// supplied by the funcdesc's pending source graph, so `buildflowgraph`
/// never flows it.
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
/// True]` (`MemoTable.finish.make_subhelper`) — normalise a two-valued bool set to
/// `[False, True]`; every other set is returned unchanged.
fn normalize_bool_order(values: &[MemoKey]) -> Vec<MemoKey> {
    if values
        == [
            MemoKey(ConstValue::Bool(true)),
            MemoKey(ConstValue::Bool(false)),
        ]
    {
        return vec![
            MemoKey(ConstValue::Bool(false)),
            MemoKey(ConstValue::Bool(true)),
        ];
    }
    values.to_vec()
}

/// RPython `nextargvalues == [False, True]` (`MemoTable.finish.make_subhelper`) —
/// the bool-arg discriminator after [`normalize_bool_order`].
fn is_false_true(values: &[MemoKey]) -> bool {
    values
        == [
            MemoKey(ConstValue::Bool(false)),
            MemoKey(ConstValue::Bool(true)),
        ]
}

impl SharedUnionFindInfo for MemoTable {
    /// RPython `MemoTable.absorb(self, other)` (specialize.py):
    /// `self.table.update(other.table); assert self.graph is None;
    /// del other.graph; other.do_not_process = True`.
    fn absorb_shared(this: &Rc<RefCell<Self>>, other: &Rc<RefCell<Self>>) {
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
        let mut other_mut = other.borrow_mut();
        // upstream: `del other.graph   # just in case`.
        other_mut.graph = None;
        // upstream: `other.do_not_process = True`.
        other_mut.do_not_process = true;
    }
}

/// One `Bookkeeper.all_specializations[funcdesc]` entry: the
/// `UnionFind(compute_one_result)` family of argument tuples
/// (specialize.py). A failing `compute_one_result` (`func(*args)`)
/// propagates out of `try_find`/`try_union` and leaves the tuple
/// untracked, so the next `memo` call runs it again.
pub type MemoFamily = UnionFind<Vec<MemoKey>, Rc<RefCell<MemoTable>>, AnnotatorError>;

/// Build the `UnionFind(compute_one_result)` family both [`memo`] and
/// [`memotable_for_args`] store in `Bookkeeper.all_specializations`.
///
/// `register_finish` matches `MemoTable.register_finish` after a
/// successful `compute_one_result` (`memo`); rtyper-time table lookups
/// skip it. `fail_kind` is the clause in the factory error (`host call`
/// vs `compute_one_result`) so each caller keeps its existing message.
fn new_memo_family(
    name: String,
    defaults: Vec<Constant>,
    bookkeeper: Weak<Bookkeeper>,
    register_finish: bool,
    fail_kind: &'static str,
    compute_one_result: impl Fn(&[ConstValue]) -> Result<ConstValue, String> + 'static,
) -> MemoFamily {
    UnionFind::new_fallible(move |args: &Vec<MemoKey>| {
        let plain: Vec<ConstValue> = args.iter().map(|k| k.0.clone()).collect();
        match compute_one_result(&plain) {
            Ok(value) => {
                let memotable = Rc::new(RefCell::new(MemoTable::new(
                    name.clone(),
                    defaults.clone(),
                    bookkeeper.clone(),
                    args.clone(),
                    value,
                )));
                if register_finish {
                    MemoTable::register_finish(&memotable);
                }
                Ok(memotable)
            }
            Err(e) => Err(AnnotatorError::new(format!(
                "memo call: {fail_kind} for {name} failed on {plain:?}: {e}"
            ))),
        }
    })
}

/// `specialize.py memo` KeyError branch:
/// `bookkeeper.all_specializations[funcdesc] = UnionFind(compute_one_result)`,
/// then `find(args)`.
///
/// `funcdesc_key` is the same key kind [`memo`] uses (`Desc.identity`).
/// Does not call [`MemoTable::register_finish`]: [`memo`] does that so
/// `finish` can synthesise the pycall dispatch graph. Callers that only
/// need the host result (rtyper-time `specialize:memo` helpers) read
/// [`MemoTable::table`].
pub fn memotable_for_args(
    bookkeeper: &Bookkeeper,
    funcdesc_key: DescKey,
    funcdesc_name: &str,
    args: Vec<ConstValue>,
    compute_one_result: impl Fn(&[ConstValue]) -> Result<ConstValue, String> + 'static,
) -> Result<Rc<RefCell<MemoTable>>, AnnotatorError> {
    let memotable = {
        let family = bookkeeper
            .all_specializations
            .borrow_mut()
            .entry(funcdesc_key)
            .or_insert_with(|| {
                Rc::new(RefCell::new(new_memo_family(
                    funcdesc_name.to_string(),
                    Vec::new(),
                    Weak::new(),
                    false,
                    "compute_one_result",
                    compute_one_result,
                )))
            })
            .clone();
        let mut family = family.borrow_mut();
        let args: Vec<MemoKey> = args.into_iter().map(MemoKey).collect();
        let rep = family.try_find(args)?.1;
        family
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
/// The factory returns the host-call error, which `try_find`/`try_union`
/// propagate before the tuple is recorded.
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
    let mut possiblevalues = cartesian_product(&per_arg);

    // upstream: `bookkeeper = funcdesc.bookkeeper`.
    let bookkeeper: Rc<Bookkeeper> = funcdesc.base.bookkeeper.clone();

    // upstream keys `all_specializations` by the funcdesc object;
    // `Desc.identity` is that object's key in every bookkeeper table.
    let key = funcdesc.base.identity;
    let name = funcdesc.name.clone();
    let defaults = funcdesc.defaults.clone();
    let bk_weak = Rc::downgrade(&bookkeeper);

    // upstream: the family is fetched from `all_specializations`, and its
    // `find`/`union` (which run `func(*args)`) do not hold the registry.
    let family = {
        let mut specs = bookkeeper.all_specializations.borrow_mut();
        // upstream: `memotables = bookkeeper.all_specializations[funcdesc]`
        //           with the KeyError branch building a fresh
        //           `UnionFind(compute_one_result)`.
        match specs.entry(key) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
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
                let host_call = graph_func.host_call.clone().ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "memo call: function {} has no host_call hook registered",
                        funcdesc.name
                    ))
                })?;
                // upstream `compute_one_result`: `value = func(*args);
                // memotable = MemoTable(funcdesc, args, value);
                // memotable.register_finish(); return memotable`.
                v.insert(Rc::new(RefCell::new(new_memo_family(
                    name.clone(),
                    defaults.clone(),
                    bk_weak.clone(),
                    true,
                    "host call",
                    move |args| (host_call.0)(args),
                ))))
                .clone()
            }
        }
    };
    let memotable = {
        let mut family = family.borrow_mut();

        // upstream: `firstvalues = possiblevalues.next()` (StopIteration if
        // any argument's value set is empty).
        let firstvalues: Vec<MemoKey> = possiblevalues
            .next()
            .ok_or_else(|| {
                AnnotatorError::new(format!(
                    "memo call: {} has no possible argument combinations",
                    funcdesc.name
                ))
            })?
            .into_iter()
            .map(MemoKey)
            .collect();

        // upstream: `_, _, memotable = memotables.find(firstvalues)`.
        let mut rep = family.try_find(firstvalues.clone())?.1;
        // upstream: `for values in possiblevalues: _, _, memotable =
        //            memotables.union(firstvalues, values)`.
        for values in possiblevalues {
            let values: Vec<MemoKey> = values.into_iter().map(MemoKey).collect();
            rep = family.try_union(firstvalues.clone(), values)?.1;
        }
        family
            .get(&rep)
            .expect("memo: representative MemoTable present after find/union")
            .clone()
    };

    let m = memotable.borrow();
    // upstream: `if memotable.graph is not None: return memotable.graph`.
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
        let out = cartesian_product::<i32>(&[]).collect::<Vec<_>>();
        assert_eq!(out, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn cartesian_product_single_list() {
        let out = cartesian_product(&[vec![1, 2, 3]]).collect::<Vec<_>>();
        assert_eq!(out, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn cartesian_product_orders_first_list_fastest() {
        // Mirrors the generator's emission order: tails come from the
        // tail product, the first list varies fastest within each.
        let out = cartesian_product(&[vec![1, 2], vec![3, 4]]).collect::<Vec<_>>();
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
    use crate::annotator::classdesc::ClassDictEntry;
    use crate::flowspace::model::{BlockRef, Hlvalue, LinkRef};

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

    fn is_call_op(op: &SpaceOperation) -> bool {
        matches!(
            op.opname.as_str(),
            "simple_call" | "direct_call" | "indirect_call"
        )
    }

    fn follow_exit_return_int(exit: &LinkRef, returnblock: &BlockRef) -> i64 {
        let mut incoming: Vec<Option<Hlvalue>> = exit.borrow().args.clone();
        let mut link = exit.clone();
        loop {
            let target = link.borrow().target.clone().expect("exit target");
            if Rc::ptr_eq(&target, returnblock) {
                return as_int(incoming[0].as_ref().expect("return constant"));
            }
            let tb = target.borrow();
            let next = if tb.exits.len() == 1 {
                tb.exits[0].clone()
            } else {
                tb.exits
                    .iter()
                    .find(|e| e.borrow().exitcase.is_none())
                    .cloned()
                    .unwrap_or_else(|| {
                        panic!(
                            "expected a unique successor; ops={:?} switch={:?} nexits={}",
                            tb.operations
                                .iter()
                                .map(|o| o.opname.as_str())
                                .collect::<Vec<_>>(),
                            tb.exitswitch,
                            tb.exits.len()
                        )
                    })
            };
            let mut mapping = std::collections::HashMap::new();
            for (slot, val) in tb.inputargs.iter().zip(incoming.iter()) {
                if let Hlvalue::Variable(v) = slot
                    && let Some(val) = val
                {
                    mapping.insert(v.clone(), val.clone());
                }
            }
            let next_args = next.borrow().args.clone();
            drop(tb);
            incoming = next_args
                .into_iter()
                .map(|a| match a {
                    Some(Hlvalue::Variable(ref v)) => mapping.get(v).cloned().or(a),
                    other => other,
                })
                .collect();
            link = next;
        }
    }

    fn bool_exits(startblock_exits: &[LinkRef]) -> Vec<LinkRef> {
        startblock_exits
            .iter()
            .filter(|e| {
                matches!(
                    e.borrow().exitcase,
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::Bool(_),
                        ..
                    }))
                )
            })
            .cloned()
            .collect()
    }

    fn memo_field_functions(
        bk: &Rc<crate::annotator::bookkeeper::Bookkeeper>,
        class: &HostObject,
    ) -> Vec<HostObject> {
        let desc = bk.getdesc(class).expect("getdesc");
        let class_desc = desc.as_class().expect("class desc");
        let mut out = Vec::new();
        for (name, entry) in &class_desc.borrow().classdict {
            if !name.starts_with("$memofield_") {
                continue;
            }
            if let ClassDictEntry::Constant(c) = entry
                && let ConstValue::HostObject(h) = &c.value
            {
                out.push(h.clone());
            }
        }
        out
    }

    fn host_in_graphs(ann: &RPythonAnnotator, h: &HostObject) -> bool {
        let Some(gf) = h.user_function() else {
            return false;
        };
        ann.translator
            .graphs
            .borrow()
            .iter()
            .any(|g| g.borrow().func.as_ref().is_some_and(|f| f.id == gf.id))
    }

    fn cached_helper_graph(
        bk: &Rc<crate::annotator::bookkeeper::Bookkeeper>,
        h: &HostObject,
    ) -> Rc<PyGraph> {
        bk.getdesc(h)
            .expect("getdesc")
            .as_function()
            .expect("function")
            .borrow()
            .cachedgraph(GraphCacheKey::None, None, None)
            .expect("cachedgraph")
    }

    /// `finish` on a one-bool-arg table synthesises a single dispatch
    /// block: `exitswitch` on the arg, with a False exit and a True exit,
    /// each returning the precomputed constant (`MemoTable.finish`).
    /// `0`/`1` int arguments take the same dispatch.
    #[test]
    fn finish_bool_arg_builds_exitswitch_dispatch() {
        for (first, inserted) in [
            // bools: table False -> 20, True -> 10.
            ((ConstValue::Bool(false), 20), (ConstValue::Bool(true), 10)),
            // ints compare equal to bools upstream; `{1, 0}` also exercises
            // `normalize_bool_order`.
            ((ConstValue::Int(1), 10), (ConstValue::Int(0), 20)),
        ] {
            let (ann, bk) = ann_bk();
            let table = Rc::new(RefCell::new(MemoTable::new(
                "f".to_string(),
                Vec::new(),
                Rc::downgrade(&bk),
                vec![MemoKey(first.0)],
                ConstValue::Int(first.1),
            )));
            table
                .borrow_mut()
                .table
                .insert(vec![MemoKey(inserted.0)], ConstValue::Int(inserted.1));

            let graphs_before = ann.translator.graphs.borrow().len();
            MemoTable::finish(&table).expect("finish");
            assert_eq!(ann.translator.graphs.borrow().len() - graphs_before, 1);
            assert!(ann.translator._prebuilt_graphs.borrow().is_empty());

            let m = table.borrow();
            assert_eq!(
                m.table.get(&vec![MemoKey(ConstValue::Bool(false))]),
                Some(&ConstValue::Int(20))
            );
            let pygraph = m.graph.as_ref().expect("graph synthesised");
            assert!(Rc::ptr_eq(
                &ann.translator.graphs.borrow()[graphs_before],
                &pygraph.graph,
            ));
            let graph = pygraph.graph.borrow();
            assert_eq!(graph.name, "memo_f_0");
            assert!(graph.func.is_some());
            let startblock = graph.startblock.borrow();
            assert!(
                matches!(startblock.exitswitch, Some(Hlvalue::Variable(_))),
                "expected exitswitch on a Variable, got {:?}",
                startblock.exitswitch
            );
            assert_eq!(startblock.operations.len(), 1);
            let bool_op = &startblock.operations[0];
            assert_eq!(bool_op.opname, "bool");
            assert_eq!(bool_op.args.len(), 1);
            assert_eq!(bool_op.args[0], startblock.inputargs[0]);
            assert_eq!(Some(&bool_op.result), startblock.exitswitch.as_ref());
            let exits = bool_exits(&startblock.exits);
            assert_eq!(exits.len(), 2, "expected False/True exits");
            for exit in &exits {
                let case = as_bool(exit.borrow().exitcase.as_ref().expect("bool exitcase"));
                let ret = follow_exit_return_int(exit, &graph.returnblock);
                // False -> 20, True -> 10.
                assert_eq!(ret, if case { 10 } else { 20 });
            }
        }
    }

    /// A one-valued argument collapses to a constant return with no
    /// `exitswitch` (`make_subhelper` constants branch of the
    /// single-value case).
    #[test]
    fn finish_single_value_arg_collapses_to_constant() {
        let (_ann, bk) = ann_bk();
        let table = Rc::new(RefCell::new(MemoTable::new(
            "g".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![MemoKey(ConstValue::Bool(false))],
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

    /// Single-value arg whose child is not constant: the entry calls
    /// that subhelper (`make_subhelper` step 6, specialize.py).
    #[test]
    fn finish_single_value_arg_calls_its_subhelper() {
        let (ann, bk) = ann_bk();
        let table = Rc::new(RefCell::new(MemoTable::new(
            "s".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![
                MemoKey(ConstValue::Int(7)),
                MemoKey(ConstValue::Bool(false)),
            ],
            ConstValue::Int(1),
        )));
        table.borrow_mut().table.insert(
            vec![MemoKey(ConstValue::Int(7)), MemoKey(ConstValue::Bool(true))],
            ConstValue::Int(2),
        );

        let graphs_before = ann.translator.graphs.borrow().len();
        MemoTable::finish(&table).expect("finish");
        assert_eq!(ann.translator.graphs.borrow().len() - graphs_before, 1);

        let m = table.borrow();
        let pygraph = m.graph.as_ref().expect("graph synthesised");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        let call = startblock
            .operations
            .iter()
            .find(|op| op.opname == "simple_call")
            .unwrap_or_else(|| panic!("expected a simple_call, ops={opnames:?}"));
        let h = match &call.args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => h.clone(),
                other => panic!("expected HostObject constant, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        };
        drop(startblock);
        drop(graph);

        assert!(
            !host_in_graphs(&ann, &h),
            "subhelper HostObject is not yet in translator.graphs"
        );
        let sub = cached_helper_graph(&bk, &h);
        assert_eq!(sub.graph.borrow().name, "memo_s_1");
        assert!(sub.graph.borrow().func.is_some());
        assert_eq!(ann.translator.graphs.borrow().len() - graphs_before, 2);
        let g = sub.graph.borrow();
        let sb = g.startblock.borrow();
        assert!(
            matches!(sb.exitswitch, Some(Hlvalue::Variable(_))),
            "subhelper should dispatch on its bool arg, got {:?}",
            sb.exitswitch
        );
        let exits = bool_exits(&sb.exits);
        assert_eq!(exits.len(), 2);
        for exit in &exits {
            let case = as_bool(exit.borrow().exitcase.as_ref().expect("bool exitcase"));
            let ret = follow_exit_return_int(exit, &g.returnblock);
            assert_eq!(ret, if case { 2 } else { 1 });
        }
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
            vec![MemoKey(ConstValue::Bool(false))],
            ConstValue::Int(1),
        )));
        MemoTable::finish(&table).expect("first finish");
        let err = MemoTable::finish(&table).expect_err("second finish errors");
        assert!(err.to_string().contains("already finished"));
    }

    /// `finish` on a `[PBC-set, bool]` table takes the `store = nextfns`
    /// branch (specialize.py): after fixing the PBC argument the
    /// remaining bool dispatch is not constant, so each PBC member's memo
    /// field holds a standalone subhelper *callable*, and the top dispatch
    /// reads the field with `getattr` then calls it with the remaining
    /// argument.
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
                MemoKey(ConstValue::HostObject(class_a.clone())),
                MemoKey(ConstValue::Bool(false)),
            ],
            ConstValue::Int(1),
        )));
        {
            let mut m = table.borrow_mut();
            m.table.insert(
                vec![
                    MemoKey(ConstValue::HostObject(class_a.clone())),
                    MemoKey(ConstValue::Bool(true)),
                ],
                ConstValue::Int(2),
            );
            m.table.insert(
                vec![
                    MemoKey(ConstValue::HostObject(class_b.clone())),
                    MemoKey(ConstValue::Bool(false)),
                ],
                ConstValue::Int(3),
            );
            m.table.insert(
                vec![
                    MemoKey(ConstValue::HostObject(class_b.clone())),
                    MemoKey(ConstValue::Bool(true)),
                ],
                ConstValue::Int(4),
            );
        }

        let graphs_before = ann.translator.graphs.borrow().len();
        MemoTable::finish(&table).expect("finish nextfns");
        assert_eq!(ann.translator.graphs.borrow().len() - graphs_before, 1);

        let m = table.borrow();
        let pygraph = m.graph.as_ref().expect("graph synthesised");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert!(opnames.contains(&"getattr"), "ops: {opnames:?}");
        assert!(
            startblock.operations.iter().any(is_call_op),
            "expected getattr then a call, ops={opnames:?}"
        );
        drop(startblock);
        drop(graph);

        let mut stored = memo_field_functions(&bk, &class_a);
        stored.extend(memo_field_functions(&bk, &class_b));
        assert!(!stored.is_empty(), "memo fields should hold subhelpers");
        for h in &stored {
            assert!(
                !host_in_graphs(&ann, h),
                "subhelper HostObject is not yet in translator.graphs"
            );
        }
        let sub = cached_helper_graph(&bk, &stored[0]);
        let g = sub.graph.borrow();
        let sb = g.startblock.borrow();
        assert!(
            matches!(sb.exitswitch, Some(Hlvalue::Variable(_))),
            "subhelper should dispatch on its bool arg, got {:?}",
            sb.exitswitch
        );
        assert_eq!(bool_exits(&sb.exits).len(), 2);
    }

    /// `finish` takes `example_value` from `self.table.iteritems().next()`
    /// (`MemoTable.finish`): the first inserted entry. Missing cartesian
    /// combinations fall back to that value (`table.get(args, example_value)`).
    #[test]
    fn finish_example_value_is_first_inserted_entry() {
        let (_ann, bk) = ann_bk();
        let class_a = HostObject::new_class("A", Vec::new());
        let class_b = HostObject::new_class("B", Vec::new());
        // First insert: (A, False)->1. Incomplete product: (B, True)->2
        // only. Missing (A, True) and (B, False) must return 1.
        let table = Rc::new(RefCell::new(MemoTable::new(
            "k".to_string(),
            Vec::new(),
            Rc::downgrade(&bk),
            vec![
                MemoKey(ConstValue::HostObject(class_a.clone())),
                MemoKey(ConstValue::Bool(false)),
            ],
            ConstValue::Int(1),
        )));
        table.borrow_mut().table.insert(
            vec![
                MemoKey(ConstValue::HostObject(class_b.clone())),
                MemoKey(ConstValue::Bool(true)),
            ],
            ConstValue::Int(2),
        );

        MemoTable::finish(&table).expect("finish");

        let mut helpers = memo_field_functions(&bk, &class_a);
        helpers.extend(memo_field_functions(&bk, &class_b));
        let mut seen_true = Vec::new();
        let mut seen_false = Vec::new();
        for h in helpers {
            let pygraph = cached_helper_graph(&bk, &h);
            let g = pygraph.graph.borrow();
            let sb = g.startblock.borrow();
            let exits = bool_exits(&sb.exits);
            assert_eq!(exits.len(), 2);
            for exit in &exits {
                let case = as_bool(exit.borrow().exitcase.as_ref().expect("bool exitcase"));
                let ret = follow_exit_return_int(exit, &g.returnblock);
                if case {
                    seen_true.push(ret);
                } else {
                    seen_false.push(ret);
                }
            }
        }
        seen_true.sort();
        seen_false.sort();
        // (A, True) -> 1 (example_value); (B, True) -> 2
        assert_eq!(seen_true, vec![1, 2]);
        // (A, False) -> 1; (B, False) -> 1 (example_value)
        assert_eq!(seen_false, vec![1, 1]);
    }

    #[test]
    fn memotable_for_args_reuses_all_specializations_entry() {
        let bk = Bookkeeper::new();
        let key = crate::annotator::description::alloc_desc_key();
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
            t1.borrow().table.get(&vec![MemoKey(ConstValue::Int(1))]),
            Some(&ConstValue::Int(7))
        );
        // `MemoKey` follows Python `==` exactly at the 128-bit edges.
        let two_127 = 2f64.powi(127);
        assert_ne!(
            MemoKey(ConstValue::Float(two_127.to_bits())),
            MemoKey(ConstValue::Int128(i128::MAX))
        );
        assert_eq!(
            MemoKey(ConstValue::Float(two_127.to_bits())),
            MemoKey(ConstValue::UInt128(1u128 << 127))
        );
        assert_eq!(
            MemoKey(ConstValue::Float((3.0 * 2f64.powi(126)).to_bits())),
            MemoKey(ConstValue::UInt128(3u128 << 126))
        );
        assert_eq!(
            MemoKey(ConstValue::Float((-two_127).to_bits())),
            MemoKey(ConstValue::Int128(i128::MIN))
        );
        assert_eq!(
            MemoKey(ConstValue::Float(1.0f64.to_bits())),
            MemoKey(ConstValue::Bool(true))
        );
    }
}
