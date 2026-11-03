//! RPython `rpython/rtyper/callparse.py` (164 LOC).
//!
//! `callparse` walks a `HighLevelOp` (`simple_call` / `call_args`) and
//! emits the low-level argument list expected by the callee
//! [`crate::flowspace::pygraph::PyGraph`]. It is the bridge that
//! [`super::rpbc::FunctionReprBase::call`] depends on (rpbc.py).
//!
//! The port follows upstream class-by-class:
//!
//! * `class ArgumentsForRtype(ArgumentsForTranslation)` (callparse.py)
//!   — overrides `newtuple` / `unpackiterable` so `_match_signature`
//!   returns [`Holder`] instances instead of [`crate::annotator::model::SomeTuple`].
//! * `class Holder` (callparse.py) — base class with an `_emit`
//!   cache; cached emit is a per-Holder side table.
//! * `class VarHolder(Holder)` (callparse.py) — wraps a positional
//!   argument index with its `SomeValue` annotation.
//! * `class ConstHolder(Holder)` (callparse.py) — wraps a default
//!   value or stararg constant.
//! * `class NewTupleHolder(Holder)` (callparse.py) — synthetic
//!   tuple built from peers (e.g. stararg unpack rewrap).
//! * `class ItemHolder(Holder)` (callparse.py) — projects the
//!   `index`-th item of a tuple-typed parent holder.
//!
//! ## TODO: `ArgumentsForRtype` storage duplication
//!
//! Upstream `class ArgumentsForRtype(ArgumentsForTranslation):` inherits
//! `_match_signature` and `arguments_w`/`keywords`/`w_stararg`
//! attribute storage from `ArgumentsForTranslation`. Python's duck
//! typing lets the same `_match_signature` body operate on either
//! `SomeValue` or `Holder` items because `newtuple` / `unpackiterable`
//! are method overrides.
//!
//! Rust requires a single concrete type per struct field, so the
//! `ArgumentsForTranslation` port in [`crate::annotator::argument`]
//! hard-codes `Vec<SomeValue>`. This file mirrors the field shape over
//! [`Vec<Holder>`] and re-implements `_match_signature` against the
//! `Holder`-typed storage. Both implementations stay
//! synchronised with upstream `argument.py:_match_signature`
//! (`annotator/argument.py`).
//!
//! Convergence path: when the [`crate::annotator::argument::ArgumentsForTranslation`]
//! is generalised to a trait-parameterised storage, the duplicated
//! `_match_signature` body in this file collapses into a single shared
//! implementation. Tracking item: cascade R1 follow-up after R1.m.
//!
//! ## TODO: `Holder::Item` parent sharing
//!
//! Upstream `VarHolder.items()` (callparse.py) builds
//! `ItemHolder(self, i)` so every sibling `ItemHolder` shares the
//! `self` parent (Python reference semantics). Pyre's [`Holder::item`]
//! wraps a `Box<Holder>` whose construction at [`Holder::items`]
//! (line 423) clones the parent — sibling `ItemHolder`s therefore each
//! own a private parent copy. The per-Holder `_emit` cache below still
//! avoids re-emitting under the same Holder, but cross-sibling parent
//! sharing is lost relative to upstream. Switching `Box<Holder>` →
//! `Rc<Holder>` would restore parity; deferred until a consumer
//! actually needs cross-sibling cache hits.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::argument::ArgErr;
use crate::annotator::model::{SomeValue, SomeValueTag};
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{Repr, inputconst, inputconst_from_lltype};
use crate::translator::rtyper::rtuple::TupleRepr;
use crate::translator::rtyper::rtyper::{HighLevelOp, RPythonTyper};

// callparse.py — class ArgumentsForRtype(ArgumentsForTranslation):

/// RPython `class ArgumentsForRtype(ArgumentsForTranslation)`
/// (callparse.py).
///
/// Storage mirrors `ArgumentsForTranslation` over [`Holder`] (see the
/// TODO note in the module doc).
#[derive(Clone, Debug)]
pub struct ArgumentsForRtype {
    /// RPython `CallSpec.arguments_w`.
    pub arguments_w: Vec<Holder>,
    /// RPython `CallSpec.keywords`.
    pub keywords: HashMap<String, Holder>,
    /// RPython `CallSpec.w_stararg`.
    pub w_stararg: Option<Holder>,
}

impl ArgumentsForRtype {
    /// RPython `CallSpec.__init__(args_w, keywords=None, w_stararg=None)`
    /// (flowspace/argument.py).
    pub fn new(arguments_w: Vec<Holder>) -> Self {
        ArgumentsForRtype {
            arguments_w,
            keywords: HashMap::new(),
            w_stararg: None,
        }
    }

    /// RPython `CallSpec.__init__` full form (flowspace/argument.py).
    pub fn with_keywords_and_stararg(
        arguments_w: Vec<Holder>,
        keywords: HashMap<String, Holder>,
        w_stararg: Option<Holder>,
    ) -> Self {
        ArgumentsForRtype {
            arguments_w,
            keywords,
            w_stararg,
        }
    }

    /// RPython `CallSpec.fromshape(cls, (shape_cnt, shape_keys,
    /// shape_star), data_w)` (flowspace/argument.py).
    pub fn fromshape(shape: &CallShape, data_w: Vec<Holder>) -> Self {
        let shape_cnt = shape.shape_cnt;
        let end_keys = shape_cnt + shape.shape_keys.len();
        let mut p = end_keys;
        let args_w: Vec<Holder> = data_w[..shape_cnt].to_vec();
        let keyword_slice = &data_w[shape_cnt..end_keys];
        let mut keywords: HashMap<String, Holder> = HashMap::new();
        for (name, value) in shape.shape_keys.iter().zip(keyword_slice.iter()) {
            keywords.insert(name.clone(), value.clone());
        }
        let w_stararg = if shape.shape_star {
            let v = data_w[p].clone();
            p += 1;
            Some(v)
        } else {
            None
        };
        let _ = p;
        ArgumentsForRtype {
            arguments_w: args_w,
            keywords,
            w_stararg,
        }
    }

    /// RPython `ArgumentsForRtype.newtuple(self, items)`
    /// (callparse.py): `return NewTupleHolder(items)`.
    pub fn newtuple(items: Vec<Holder>) -> Holder {
        Holder::new_tuple(items)
    }

    /// RPython `ArgumentsForRtype.unpackiterable(self, it)`
    /// (callparse.py):
    /// ```python
    /// assert it.is_tuple()
    /// items = it.items()
    /// return list(items)
    /// ```
    pub fn unpackiterable(it: &Holder) -> Vec<Holder> {
        assert!(
            it.is_tuple(),
            "ArgumentsForRtype.unpackiterable: holder is not tuple-typed"
        );
        it.items()
    }

    /// RPython `ArgumentsForTranslation.positional_args` property
    /// (annotator/argument.py):
    /// ```python
    /// if self.w_stararg is not None:
    ///     args_w = self.unpackiterable(self.w_stararg)
    ///     return self.arguments_w + args_w
    /// else:
    ///     return self.arguments_w
    /// ```
    pub fn positional_args(&self) -> Vec<Holder> {
        match &self.w_stararg {
            Some(star) => {
                let mut out = self.arguments_w.clone();
                out.extend(Self::unpackiterable(star));
                out
            }
            None => self.arguments_w.clone(),
        }
    }

    /// RPython `_match_signature(scope_w, signature, defaults_w=None)`
    /// (annotator/argument.py) — Holder-typed storage variant.
    /// Body mirrors the [`crate::annotator::argument::ArgumentsForTranslation::match_signature_into`]
    /// port; only the item type differs.
    fn match_signature_into(
        &self,
        scope_w: &mut [Option<Holder>],
        signature: &Signature,
        defaults_w: Option<&[Holder]>,
    ) -> Result<(), ArgErr> {
        let co_argcount = signature.num_argnames();

        let args_w = self.positional_args();
        let num_args = args_w.len();
        let num_kwds = self.keywords.len();

        // upstream argument.py:57-60 — `take = min(num_args,
        // co_argcount)`; `scope_w[:take] = args_w[:take]`.
        let take = num_args.min(co_argcount);
        for i in 0..take {
            scope_w[i] = Some(args_w[i].clone());
        }
        let input_argcount = take;

        // upstream argument.py:63-70 — *vararg collection.
        if signature.has_vararg() {
            let starargs_w: Vec<Holder> = if num_args > co_argcount {
                args_w[co_argcount..].to_vec()
            } else {
                Vec::new()
            };
            scope_w[co_argcount] = Some(Self::newtuple(starargs_w));
        } else if num_args > co_argcount {
            return Err(ArgErr::Count {
                signature: signature.clone(),
                num_defaults: defaults_w.map_or(0, |d| d.len()),
                missing_args: 0,
                num_args,
                num_kwds,
            });
        }

        // upstream argument.py — `assert not signature.has_kwarg()`.
        assert!(
            !signature.has_kwarg(),
            "signature.has_kwarg() not supported"
        );

        // upstream argument.py:74-101 — keyword-argument matching.
        if num_kwds > 0 {
            let mut mapping = Vec::<String>::new();
            let mut num_remainingkwds = self.keywords.len();
            for name in self.keywords.keys() {
                let j = signature.find_argname(name);
                if j < input_argcount as i32 {
                    if j >= 0 {
                        return Err(ArgErr::MultipleValues {
                            argname: name.clone(),
                        });
                    }
                } else {
                    mapping.push(name.clone());
                    num_remainingkwds -= 1;
                }
            }

            if num_remainingkwds > 0 {
                if co_argcount == 0 {
                    return Err(ArgErr::Count {
                        signature: signature.clone(),
                        num_defaults: defaults_w.map_or(0, |d| d.len()),
                        missing_args: 0,
                        num_args,
                        num_kwds,
                    });
                }
                // upstream `ArgErrUnknownKwds.__init__`
                // (argument.py:237-245): when exactly one unmatched
                // keyword remains, scan keywords for the first not in
                // `kwds_mapping`.
                let kwd_name = if num_remainingkwds == 1 {
                    self.keywords
                        .keys()
                        .find(|k| !mapping.contains(*k))
                        .cloned()
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                return Err(ArgErr::UnknownKwds {
                    num_kwds: num_remainingkwds,
                    kwd_name,
                });
            }
            drop(mapping);
        }

        // upstream argument.py:103-120 — defaults-fill.
        let mut missing = 0usize;
        if input_argcount < co_argcount {
            let def_first = co_argcount - defaults_w.map_or(0, |d| d.len());
            for (i, slot) in scope_w
                .iter_mut()
                .enumerate()
                .take(co_argcount)
                .skip(input_argcount)
            {
                let name = &signature.argnames[i];
                if let Some(v) = self.keywords.get(name) {
                    *slot = Some(v.clone());
                    continue;
                }
                let defnum = i as isize - def_first as isize;
                if defnum >= 0 {
                    if let Some(defaults) = defaults_w {
                        *slot = Some(defaults[defnum as usize].clone());
                    } else {
                        missing += 1;
                    }
                } else {
                    missing += 1;
                }
            }
            if missing > 0 {
                return Err(ArgErr::Count {
                    signature: signature.clone(),
                    num_defaults: defaults_w.map_or(0, |d| d.len()),
                    missing_args: missing,
                    num_args,
                    num_kwds,
                });
            }
        }
        Ok(())
    }

    /// RPython `match_signature(self, signature, defaults_w)`
    /// (annotator/argument.py).
    pub fn match_signature(
        &self,
        signature: &Signature,
        defaults_w: Option<&[Holder]>,
    ) -> Result<Vec<Holder>, ArgErr> {
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<Option<Holder>> = vec![None; scopelen];
        self.match_signature_into(&mut scope_w, signature, defaults_w)?;
        // upstream returns the scope_w list with possible None slots.
        // Pyre's caller [`callparse`] already asserts the count matches
        // `rinputs`, so a None slot here is a typer bug — surface a
        // Holder::Const(Impossible-equivalent) is not safe; instead,
        // require all slots filled and panic if not (matches upstream
        // behaviour where None reaches `vlist.append(h.emit(...))` and
        // crashes with AttributeError).
        Ok(scope_w
            .into_iter()
            .map(|opt| opt.expect("match_signature: scope_w slot left None"))
            .collect())
    }
}

// callparse.py — Holder hierarchy.

/// RPython `class Holder(object)` + four direct subclasses
/// (callparse.py). Rust collapses the inheritance into a single
/// `enum` because Holder leaves are dispatched via concrete-class
/// branches (no virtual override beyond the four listed variants).
///
/// Each variant carries a per-instance `_cache` that mirrors upstream
/// `Holder.emit`'s `cache = self._cache = {}` lazy-init pattern
/// (callparse.py). Key is the data-pointer identity of the
/// requested target [`Repr`] (Python's `dict[repr]` defaults to
/// `id(repr)` because `Repr` doesn't override `__hash__`/`__eq__`),
/// matching pyre's singleton-Arc Repr layout where the same logical
/// repr is the same allocation.
#[derive(Clone, Debug)]
pub enum Holder {
    /// RPython `class VarHolder(Holder)` (callparse.py).
    Var {
        /// RPython `VarHolder.num` — index into `hop.args_v`.
        num: usize,
        /// RPython `VarHolder.s_obj` — annotation copy from
        /// `hop.args_s[num]`.
        s_obj: Box<SomeValue>,
        cache: RefCell<HashMap<usize, Hlvalue>>,
    },
    /// RPython `class ConstHolder(Holder)` (callparse.py).
    Const {
        /// RPython `ConstHolder.value` — the untyped Python value;
        /// pyre carries it as a [`ConstValue`].
        value: ConstValue,
        cache: RefCell<HashMap<usize, Hlvalue>>,
    },
    /// RPython `class NewTupleHolder(Holder)` (callparse.py).
    NewTuple {
        /// RPython `NewTupleHolder.holders` — items of the synthetic
        /// tuple.
        holders: Vec<Holder>,
        cache: RefCell<HashMap<usize, Hlvalue>>,
    },
    /// RPython `class ItemHolder(Holder)` (callparse.py).
    Item {
        /// RPython `ItemHolder.holder` — parent tuple holder.
        holder: Box<Holder>,
        /// RPython `ItemHolder.index`.
        index: usize,
        cache: RefCell<HashMap<usize, Hlvalue>>,
    },
}

impl Holder {
    /// RPython `VarHolder(num, s_obj)` (callparse.py).
    pub fn var(num: usize, s_obj: SomeValue) -> Self {
        Holder::Var {
            num,
            s_obj: Box::new(s_obj),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `ConstHolder(value)` (callparse.py).
    pub fn const_(value: ConstValue) -> Self {
        Holder::Const {
            value,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `ItemHolder(holder, index)` (callparse.py).
    pub fn item(holder: Holder, index: usize) -> Self {
        Holder::Item {
            holder: Box::new(holder),
            index,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `NewTupleHolder.__new__(cls, holders)` (callparse.py).
    ///
    /// Upstream short-circuits when every input is an `ItemHolder` over
    /// the same parent and the count matches the parent's `items()`
    /// length — in that case it returns the parent holder unchanged
    /// (avoiding a needless wrap-and-unwrap).
    ///
    /// TODO: collapse optimisation deferred. The
    /// upstream Python check `h.holder == holders[0].holder` defaults
    /// to object-identity (`is`); porting it requires either
    /// `PartialEq` on `Holder` (which transitively requires
    /// `PartialEq` on `ConstValue`, currently `Clone + Debug` only) or
    /// a hashable identity wrapper threaded through every `Holder::Item`
    /// constructor.  Both are larger than the scope of a single
    /// collapse-port slice.  The non-collapsed path always wraps into
    /// [`Holder::NewTuple`]; functional equivalence preserved, only IR
    /// size differs.
    pub fn new_tuple(holders: Vec<Holder>) -> Self {
        Holder::NewTuple {
            holders,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Per-instance `_cache` accessor (callparse.py).
    fn cache(&self) -> &RefCell<HashMap<usize, Hlvalue>> {
        match self {
            Holder::Var { cache, .. }
            | Holder::Const { cache, .. }
            | Holder::NewTuple { cache, .. }
            | Holder::Item { cache, .. } => cache,
        }
    }

    /// RPython `Holder.is_tuple()` (callparse.py → `False`,
    /// overridden by `VarHolder` / `ConstHolder` / `NewTupleHolder`).
    pub fn is_tuple(&self) -> bool {
        match self {
            Holder::Var { s_obj, .. } => s_obj.tag() == SomeValueTag::Tuple,
            Holder::Const { value, .. } => matches!(value, ConstValue::Tuple(_)),
            Holder::NewTuple { .. } => true,
            Holder::Item { .. } => false,
        }
    }

    /// RPython `Holder.items()` — overridden by tuple-typed subclasses
    /// (`VarHolder.items` callparse.py, `ConstHolder.items`
    /// callparse.py, `NewTupleHolder.items` callparse.py).
    pub fn items(&self) -> Vec<Holder> {
        match self {
            Holder::Var { s_obj, .. } => {
                // upstream callparse.py — `n = len(self.s_obj.items)`,
                // `return tuple([ItemHolder(self, i) for i in range(n)])`.
                let n = match s_obj.as_ref() {
                    SomeValue::Tuple(st) => st.items.len(),
                    other => panic!("Holder::Var::items called on non-tuple s_obj: {other:?}"),
                };
                (0..n).map(|i| Holder::item(self.clone(), i)).collect()
            }
            Holder::Const { value, .. } => {
                // upstream callparse.py — `return self.value`
                // (because the value is a Python tuple already).
                let items = match value {
                    ConstValue::Tuple(items) => items.clone(),
                    other => panic!("Holder::Const::items called on non-tuple value: {other:?}"),
                };
                items.into_iter().map(Holder::const_).collect()
            }
            Holder::NewTuple { holders, .. } => holders.clone(),
            other => panic!("Holder::items called on non-tuple holder: {other:?}"),
        }
    }

    /// RPython `Holder.emit(self, repr, hop)` (callparse.py).
    ///
    /// ```python
    /// def emit(self, repr, hop):
    ///     try:
    ///         cache = self._cache
    ///     except AttributeError:
    ///         cache = self._cache = {}
    ///     try:
    ///         return cache[repr]
    ///     except KeyError:
    ///         v = self._emit(repr, hop)
    ///         cache[repr] = v
    ///         return v
    /// ```
    ///
    /// Pyre keys the cache by `Arc::as_ptr` of the requested repr.
    /// Singleton-Arc Repr layout means the same logical repr is the
    /// same allocation, so pointer identity matches Python's default
    /// `id(repr)` hash.
    pub fn emit(&self, repr: &Arc<dyn Repr>, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        let key = Arc::as_ptr(repr) as *const () as usize;
        if let Some(cached) = self.cache().borrow().get(&key) {
            return Ok(cached.clone());
        }
        let v = self._emit(repr, hop)?;
        self.cache().borrow_mut().insert(key, v.clone());
        Ok(v)
    }

    /// RPython `_emit(self, repr, hop)` per-subclass body
    /// (callparse.py, :123-124, :145-152, :160-164).
    fn _emit(&self, repr: &Arc<dyn Repr>, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        match self {
            // upstream callparse.py — `VarHolder._emit`:
            // `return hop.inputarg(repr, arg=self.num)`.
            Holder::Var { num, .. } => hop.inputarg(repr.as_ref(), *num),
            // upstream callparse.py — `ConstHolder._emit`:
            // `return hop.inputconst(repr, self.value)`.
            Holder::Const { value, .. } => inputconst(repr.as_ref(), value).map(Hlvalue::Constant),
            // upstream callparse.py — `NewTupleHolder._emit`:
            // verifies repr is a TupleRepr, recursively emits each
            // sub-holder under the matching `items_r` repr, then
            // builds the tuple via `repr.newtuple(hop.llops, repr,
            // tupleitems_v)`.
            Holder::NewTuple { holders, .. } => {
                let r_tup = (repr.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<TupleRepr>()
                    .ok_or_else(|| {
                        TyperError::message("NewTupleHolder._emit: target repr is not a TupleRepr")
                    })?;
                let mut tupleitems_v: Vec<Hlvalue> = Vec::with_capacity(holders.len());
                for h in holders.iter() {
                    let r_item = r_tup.items_r[tupleitems_v.len()].clone();
                    let v = h.emit(&r_item, hop)?;
                    tupleitems_v.push(v);
                }
                let mut llops = hop.llops.borrow_mut();
                TupleRepr::newtuple(&mut llops, r_tup, tupleitems_v)
            }
            // upstream callparse.py — `ItemHolder._emit`:
            // `r_tup, v_tuple = self.holder.access(hop)`; then
            // `v = r_tup.getitem_internal(hop, v_tuple, index)`; then
            // `return hop.llops.convertvar(v, r_tup.items_r[index],
            //                              repr)`.
            Holder::Item { holder, index, .. } => {
                let (r_tup_arc, v_tuple) = holder.access(hop)?;
                let r_tup = (r_tup_arc.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<TupleRepr>()
                    .ok_or_else(|| {
                        TyperError::message(
                            "ItemHolder._emit: parent holder repr is not a TupleRepr",
                        )
                    })?;
                let mut llops = hop.llops.borrow_mut();
                let v_var = r_tup.getitem_internal(&mut llops, v_tuple, *index)?;
                let r_item = r_tup.items_r[*index].clone();
                llops.convertvar(Hlvalue::Variable(v_var), r_item.as_ref(), repr.as_ref())
            }
        }
    }

    /// RPython `VarHolder.access(self, hop)` (callparse.py):
    /// ```python
    /// repr = hop.args_r[self.num]
    /// return repr, self.emit(repr, hop)
    /// ```
    ///
    /// Only `VarHolder` defines `access` upstream; `ItemHolder._emit`
    /// is the sole caller and it is only invoked when the parent
    /// chain bottoms out at a `VarHolder` (a `NewTupleHolder` would
    /// have been collapsed by the `__new__` short-circuit). The
    /// non-Var arms therefore surface a TyperError.
    pub fn access(&self, hop: &HighLevelOp) -> Result<(Arc<dyn Repr>, Hlvalue), TyperError> {
        match self {
            Holder::Var { num, .. } => {
                let repr = hop.args_r.borrow()[*num].clone().ok_or_else(|| {
                    TyperError::message("VarHolder.access: hop.args_r[num] is None")
                })?;
                let v = self.emit(&repr, hop)?;
                Ok((repr, v))
            }
            other => Err(TyperError::message(format!(
                "Holder.access called on non-Var holder: {other:?}"
            ))),
        }
    }
}

// callparse.py — getrinputs / getrresult / getsig.

/// RPython `getrinputs(rtyper, graph)` (callparse.py):
/// ```python
/// return [rtyper.bindingrepr(v) for v in graph.getargs()]
/// ```
pub fn getrinputs(
    rtyper: &RPythonTyper,
    graph: &Rc<PyGraph>,
) -> Result<Vec<Arc<dyn Repr>>, TyperError> {
    let args = graph.graph.borrow().getargs();
    args.iter().map(|v| rtyper.bindingrepr(v)).collect()
}

/// Result of [`getrresult`].
///
/// Upstream returns either an `Repr` instance or `lltype.Void` directly
/// (callparse.py). Pyre keeps the two cases distinct so callers
/// can branch without inventing a synthetic VoidRepr clone.
pub enum RResult {
    /// RPython `rtyper.bindingrepr(graph.getreturnvar())` — the
    /// concrete return repr.
    Repr(Arc<dyn Repr>),
    /// RPython `return lltype.Void` — the return variable carries no
    /// annotation (graph never returns a typed value).
    Void,
}

impl RResult {
    /// Lowleveltype the `direct_call` op's `resulttype` slot expects.
    /// Mirrors upstream's mixed `Repr | lltype.Void` use site at
    /// `rpbc.py:214`.
    pub fn lowleveltype(&self) -> LowLevelType {
        match self {
            RResult::Repr(r) => r.lowleveltype().clone(),
            RResult::Void => LowLevelType::Void,
        }
    }
}

/// RPython `getrresult(rtyper, graph)` (callparse.py):
/// ```python
/// if graph.getreturnvar().annotation is not None:
///     return rtyper.bindingrepr(graph.getreturnvar())
/// else:
///     return lltype.Void
/// ```
pub fn getrresult(rtyper: &RPythonTyper, graph: &Rc<PyGraph>) -> Result<RResult, TyperError> {
    let return_var = graph.graph.borrow().getreturnvar();
    let has_annotation = matches!(
        &return_var,
        Hlvalue::Variable(v) if v.annotation.borrow().is_some(),
    );
    if has_annotation {
        Ok(RResult::Repr(rtyper.bindingrepr(&return_var)?))
    } else {
        Ok(RResult::Void)
    }
}

/// RPython `getsig(rtyper, graph)` (callparse.py):
/// ```python
/// return (graph.signature, graph.defaults,
///         getrinputs(rtyper, graph), getrresult(rtyper, graph))
/// ```
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub fn getsig(
    rtyper: &RPythonTyper,
    graph: &Rc<PyGraph>,
) -> Result<(Signature, Vec<ConstValue>, Vec<Arc<dyn Repr>>, RResult), TyperError> {
    let signature = graph.signature.borrow().clone();
    let defaults: Vec<ConstValue> = graph
        .defaults
        .borrow()
        .as_ref()
        .map(|defs| defs.iter().map(|c| c.value.clone()).collect())
        .unwrap_or_default();
    Ok((
        signature,
        defaults,
        getrinputs(rtyper, graph)?,
        getrresult(rtyper, graph)?,
    ))
}

// callparse.py — `def callparse(rtyper, graph, hop, r_self=None)`.

/// RPython `callparse(rtyper, graph, hop, r_self=None)`
/// (callparse.py):
///
/// ```python
/// def callparse(rtyper, graph, hop, r_self=None):
///     rinputs = getrinputs(rtyper, graph)
///     def args_h(start):
///         return [VarHolder(i, hop.args_s[i])
///                         for i in range(start, hop.nb_args)]
///     if r_self is None:
///         start = 1
///     else:
///         start = 0
///         rinputs[0] = r_self
///     opname = hop.spaceop.opname
///     if opname == "simple_call":
///         arguments =  ArgumentsForRtype(args_h(start))
///     elif opname == "call_args":
///         arguments = ArgumentsForRtype.fromshape(
///                 hop.args_s[start].const, # shape
///                 args_h(start+1))
///     # parse the arguments according to the function we are calling
///     signature = graph.signature
///     defs_h = []
///     if graph.defaults:
///         for x in graph.defaults:
///             defs_h.append(ConstHolder(x))
///     try:
///         holders = arguments.match_signature(signature, defs_h)
///     except ArgErr as e:
///         raise TyperError("signature mismatch: %s: %s" % (
///             graph.name, e.getmsg()))
///
///     assert len(holders) == len(rinputs), "argument parsing mismatch"
///     vlist = []
///     for h,r in zip(holders, rinputs):
///         v = h.emit(r, hop)
///         vlist.append(v)
///     return vlist
/// ```
pub fn callparse(
    rtyper: &RPythonTyper,
    graph: &Rc<PyGraph>,
    hop: &HighLevelOp,
    r_self: Option<Arc<dyn Repr>>,
) -> Result<Vec<Hlvalue>, TyperError> {
    let mut rinputs = getrinputs(rtyper, graph)?;
    let start = if let Some(r_self) = r_self {
        if rinputs.is_empty() {
            return Err(TyperError::message(
                "callparse: r_self provided but graph has 0 inputargs",
            ));
        }
        rinputs[0] = r_self;
        0usize
    } else {
        1usize
    };
    let opname = hop.spaceop.opname.as_str();
    let arguments = match opname {
        "simple_call" => {
            let args_h: Vec<Holder> = (start..hop.nb_args())
                .map(|i| Holder::var(i, hop.args_s.borrow()[i].clone()))
                .collect();
            ArgumentsForRtype::new(args_h)
        }
        "call_args" => {
            let shape_value = hop
                .args_s
                .borrow()
                .get(start)
                .and_then(|sv| sv.const_().cloned())
                .ok_or_else(|| {
                    TyperError::message("callparse(call_args): args_s[start] missing const shape")
                })?;
            let shape = call_shape_from_const(&shape_value)?;
            let args_h: Vec<Holder> = (start + 1..hop.nb_args())
                .map(|i| Holder::var(i, hop.args_s.borrow()[i].clone()))
                .collect();
            ArgumentsForRtype::fromshape(&shape, args_h)
        }
        other => {
            return Err(TyperError::message(format!(
                "callparse: unsupported opname {other:?}"
            )));
        }
    };

    let signature = graph.signature.borrow().clone();
    let defs_h: Vec<Holder> = graph
        .defaults
        .borrow()
        .as_ref()
        .map(|defs| {
            defs.iter()
                .map(|c| Holder::const_(c.value.clone()))
                .collect()
        })
        .unwrap_or_default();
    let holders = arguments
        .match_signature(&signature, Some(&defs_h))
        .map_err(|e| {
            TyperError::message(format!(
                "signature mismatch: {}: {}",
                graph.graph.borrow().name,
                e.getmsg()
            ))
        })?;

    if holders.len() != rinputs.len() {
        return Err(TyperError::message(format!(
            "argument parsing mismatch: {} holders vs {} rinputs",
            holders.len(),
            rinputs.len()
        )));
    }

    let mut vlist: Vec<Hlvalue> = Vec::with_capacity(holders.len());
    for (h, r) in holders.into_iter().zip(rinputs.iter()) {
        let v = h.emit(r, hop)?;
        vlist.push(v);
    }
    Ok(vlist)
}

// Local helper — decode the `(shape_cnt, shape_keys, shape_star)` tuple
// constant that `call_args` ops carry as their first argument. Mirrors
// `bookkeeper::call_shape_from_const` (which is annotator-level only).

/// Decode a `(shape_cnt, shape_keys, shape_star)` Constant into the
/// corresponding [`CallShape`]. Mirrors the shape encoding produced by
/// `flowcontext::build_call_shape_constant` and consumed by
/// `bookkeeper::build_args_for_op`.
fn call_shape_from_const(cv: &ConstValue) -> Result<CallShape, TyperError> {
    let items = match cv {
        ConstValue::Tuple(items) => items,
        _ => {
            return Err(TyperError::message(
                "call_shape_from_const: expected ConstValue::Tuple",
            ));
        }
    };
    if items.len() != 3 {
        return Err(TyperError::message(
            "call_shape_from_const: tuple must have 3 elements",
        ));
    }
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => {
            return Err(TyperError::message(
                "call_shape_from_const: shape_cnt is not Int",
            ));
        }
    };
    let shape_keys: Vec<String> = match &items[1] {
        ConstValue::Tuple(keys) => keys
            .iter()
            .map(|k| match k.as_text() {
                Some(s) => Ok(s.to_string()),
                None => Err(TyperError::message(format!(
                    "call_shape_from_const: shape_keys element is not Str: {k:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(TyperError::message(
                "call_shape_from_const: shape_keys is not Tuple",
            ));
        }
    };
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => {
            return Err(TyperError::message(
                "call_shape_from_const: shape_star is not Bool",
            ));
        }
    };
    Ok(CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}

// Suppress unused warning for the lltype helper used by future
// resulttype-typed inputconst paths (e.g. callparse callers that pass
// `Void` directly rather than a Repr).
fn _force_use_inputconst_from_lltype() {
    let _ = inputconst_from_lltype;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeBool, SomeInteger, SomeTuple};
    use crate::flowspace::argument::Signature;

    fn s_int() -> SomeValue {
        SomeValue::Integer(SomeInteger::default())
    }

    fn s_bool() -> SomeValue {
        SomeValue::Bool(SomeBool::default())
    }

    fn s_tuple_of(items: Vec<SomeValue>) -> SomeValue {
        SomeValue::Tuple(SomeTuple::new(items))
    }

    fn h_var(num: usize, s_obj: SomeValue) -> Holder {
        Holder::var(num, s_obj)
    }

    fn h_const_int(n: i64) -> Holder {
        Holder::const_(ConstValue::Int(n))
    }

    fn sig(argnames: &[&str]) -> Signature {
        Signature::new(argnames.iter().map(|s| s.to_string()).collect(), None, None)
    }

    fn sig_with_vararg(argnames: &[&str], varargname: &str) -> Signature {
        Signature::new(
            argnames.iter().map(|s| s.to_string()).collect(),
            Some(varargname.to_string()),
            None,
        )
    }

    // Holder structural

    #[test]
    fn var_holder_is_tuple_returns_true_for_some_tuple_annotation() {
        let h = h_var(0, s_tuple_of(vec![s_int(), s_int()]));
        assert!(h.is_tuple());
    }

    #[test]
    fn var_holder_is_tuple_returns_false_for_some_int_annotation() {
        let h = h_var(0, s_int());
        assert!(!h.is_tuple());
    }

    #[test]
    fn const_holder_is_tuple_recognises_constvalue_tuple() {
        let h = Holder::const_(ConstValue::Tuple(vec![
            ConstValue::Int(1),
            ConstValue::Int(2),
        ]));
        assert!(h.is_tuple());
    }

    #[test]
    fn new_tuple_holder_is_always_tuple() {
        let h = Holder::new_tuple(vec![h_const_int(1), h_const_int(2)]);
        assert!(h.is_tuple());
    }

    #[test]
    fn item_holder_is_not_tuple_by_default() {
        let parent = h_var(0, s_tuple_of(vec![s_int(), s_int()]));
        let item = Holder::item(parent, 0);
        assert!(!item.is_tuple());
    }

    #[test]
    fn var_holder_items_returns_itemholders_for_each_tuple_slot() {
        let h = h_var(7, s_tuple_of(vec![s_int(), s_bool(), s_int()]));
        let items = h.items();
        assert_eq!(items.len(), 3);
        for (i, item) in items.iter().enumerate() {
            match item {
                Holder::Item { holder, index, .. } => {
                    assert_eq!(*index, i);
                    match holder.as_ref() {
                        Holder::Var { num, .. } => assert_eq!(*num, 7),
                        other => panic!("expected Var parent, got {other:?}"),
                    }
                }
                other => panic!("expected ItemHolder, got {other:?}"),
            }
        }
    }

    #[test]
    fn const_holder_items_returns_per_element_const_holders() {
        let h = Holder::const_(ConstValue::Tuple(vec![
            ConstValue::Int(1),
            ConstValue::Int(2),
        ]));
        let items = h.items();
        assert_eq!(items.len(), 2);
        match &items[0] {
            Holder::Const { value, .. } => assert_eq!(*value, ConstValue::Int(1)),
            other => panic!("expected Const, got {other:?}"),
        }
    }

    // ArgumentsForRtype.match_signature

    #[test]
    fn match_signature_simple_positional_pass_through() {
        let args = ArgumentsForRtype::new(vec![h_var(0, s_int()), h_var(1, s_int())]);
        let signature = sig(&["x", "y"]);
        let scope = args
            .match_signature(&signature, None)
            .expect("simple positional should match");
        assert_eq!(scope.len(), 2);
    }

    #[test]
    fn match_signature_too_few_positional_no_defaults_errors() {
        let args = ArgumentsForRtype::new(vec![h_var(0, s_int())]);
        let signature = sig(&["x", "y"]);
        let err = args
            .match_signature(&signature, None)
            .expect_err("missing arg should fail");
        assert!(matches!(
            err,
            ArgErr::Count {
                missing_args: 1,
                ..
            }
        ));
    }

    #[test]
    fn match_signature_uses_default_for_missing_positional() {
        let args = ArgumentsForRtype::new(vec![h_var(0, s_int())]);
        let defaults = vec![h_const_int(42)];
        let signature = sig(&["x", "y"]);
        let scope = args
            .match_signature(&signature, Some(&defaults))
            .expect("default should fill missing arg");
        assert_eq!(scope.len(), 2);
        match &scope[1] {
            Holder::Const { value, .. } => assert_eq!(*value, ConstValue::Int(42)),
            other => panic!("expected Const fill, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_too_many_positional_no_vararg_errors() {
        let args = ArgumentsForRtype::new(vec![
            h_var(0, s_int()),
            h_var(1, s_int()),
            h_var(2, s_int()),
        ]);
        let signature = sig(&["x", "y"]);
        let err = args
            .match_signature(&signature, None)
            .expect_err("extra positional should fail");
        assert!(matches!(
            err,
            ArgErr::Count {
                missing_args: 0,
                ..
            }
        ));
    }

    #[test]
    fn match_signature_collects_extras_into_vararg_tuple() {
        let args = ArgumentsForRtype::new(vec![
            h_var(0, s_int()),
            h_var(1, s_int()),
            h_var(2, s_int()),
        ]);
        let signature = sig_with_vararg(&["x"], "rest");
        let scope = args
            .match_signature(&signature, None)
            .expect("vararg should accept extras");
        assert_eq!(scope.len(), 2);
        match &scope[1] {
            Holder::NewTuple { holders, .. } => assert_eq!(holders.len(), 2),
            other => panic!("expected NewTupleHolder, got {other:?}"),
        }
    }

    #[test]
    fn match_signature_keyword_arg_fills_named_slot() {
        let mut keywords = HashMap::new();
        keywords.insert("y".to_string(), h_var(1, s_int()));
        let args =
            ArgumentsForRtype::with_keywords_and_stararg(vec![h_var(0, s_int())], keywords, None);
        let signature = sig(&["x", "y"]);
        let scope = args
            .match_signature(&signature, None)
            .expect("keyword should match named slot");
        assert_eq!(scope.len(), 2);
    }

    #[test]
    fn match_signature_unknown_keyword_errors() {
        let mut keywords = HashMap::new();
        keywords.insert("z".to_string(), h_var(1, s_int()));
        let args =
            ArgumentsForRtype::with_keywords_and_stararg(vec![h_var(0, s_int())], keywords, None);
        let signature = sig(&["x", "y"]);
        let err = args
            .match_signature(&signature, None)
            .expect_err("unknown keyword should fail");
        assert!(matches!(
            err,
            ArgErr::UnknownKwds { num_kwds: 1, kwd_name } if kwd_name == "z"
        ));
    }

    #[test]
    fn match_signature_keyword_collision_errors() {
        let mut keywords = HashMap::new();
        keywords.insert("x".to_string(), h_var(1, s_int()));
        let args =
            ArgumentsForRtype::with_keywords_and_stararg(vec![h_var(0, s_int())], keywords, None);
        let signature = sig(&["x", "y"]);
        let err = args
            .match_signature(&signature, None)
            .expect_err("keyword colliding with positional should fail");
        assert!(matches!(err, ArgErr::MultipleValues { argname } if argname == "x"));
    }

    #[test]
    fn match_signature_stararg_unpacks_into_positional() {
        // 1 positional + stararg holding tuple-of-2 → 3 positional total.
        let stararg = Holder::const_(ConstValue::Tuple(vec![
            ConstValue::Int(2),
            ConstValue::Int(3),
        ]));
        let args = ArgumentsForRtype::with_keywords_and_stararg(
            vec![h_var(0, s_int())],
            HashMap::new(),
            Some(stararg),
        );
        let signature = sig(&["x", "y", "z"]);
        let scope = args
            .match_signature(&signature, None)
            .expect("stararg should unpack into positional slots");
        assert_eq!(scope.len(), 3);
        match &scope[1] {
            Holder::Const { value, .. } => assert_eq!(*value, ConstValue::Int(2)),
            other => panic!("expected Const at slot 1, got {other:?}"),
        }
        match &scope[2] {
            Holder::Const { value, .. } => assert_eq!(*value, ConstValue::Int(3)),
            other => panic!("expected Const at slot 2, got {other:?}"),
        }
    }

    // Shape decoder

    #[test]
    fn call_shape_from_const_decodes_simple_positional_only() {
        let c = ConstValue::Tuple(vec![
            ConstValue::Int(2),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(false),
        ]);
        let shape = call_shape_from_const(&c).expect("shape decode");
        assert_eq!(shape.shape_cnt, 2);
        assert!(shape.shape_keys.is_empty());
        assert!(!shape.shape_star);
    }

    #[test]
    fn call_shape_from_const_decodes_star_present() {
        let c = ConstValue::Tuple(vec![
            ConstValue::Int(1),
            ConstValue::Tuple(vec![ConstValue::byte_str("kw")]),
            ConstValue::Bool(true),
        ]);
        let shape = call_shape_from_const(&c).expect("shape decode");
        assert_eq!(shape.shape_cnt, 1);
        assert_eq!(shape.shape_keys, vec!["kw".to_string()]);
        assert!(shape.shape_star);
    }
}
