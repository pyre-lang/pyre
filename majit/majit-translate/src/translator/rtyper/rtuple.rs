//! RPython `rpython/rtyper/rtuple.py` — `TupleRepr` for RPython tuples.
//!
//! Upstream rtuple.py (414 LOC) covers:
//! * `TUPLE_TYPE(field_lltypes)` (rtuple.py) — Void for empty
//!   tuples, `Ptr(GcStruct('tuple%d', ('item0', T0), ...))` otherwise.
//! * `class TupleRepr(Repr)` (rtuple.py+) — items_r,
//!   external_items_r, fieldnames, lltypes, tuple_cache, lowleveltype.
//! * `getitem` / `getitem_internal` (rtuple.py).
//! * `newtuple` / `newtuple_cached` / `_rtype_newtuple` (rtuple.py).
//! * `convert_const` / `instantiate` (rtuple.py).
//! * pair-type / iterator / hash / eq / str (rtuple.py).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    self, _ptr, LowLevelType, MallocFlavor, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, LowLevelFunction, LowLevelOpList, RPythonTyper, constant_with_lltype,
    exception_args, helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};

fn rtuple_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rtuple.{name} helper surface deferred"))
}

/// RPython `TUPLE_TYPE(field_lltypes)` (rtuple.py).
///
/// ```python
/// def TUPLE_TYPE(field_lltypes):
///     if len(field_lltypes) == 0:
///         return Void      # empty tuple
///     else:
///         fields = [('item%d' % i, TYPE) for i, TYPE in enumerate(field_lltypes)]
///         kwds = {'hints': {'immutable': True, 'noidentity': True}}
///         return Ptr(GcStruct('tuple%d' % len(field_lltypes), *fields, **kwds))
/// ```
#[allow(non_snake_case)]
pub fn TUPLE_TYPE(field_lltypes: &[LowLevelType]) -> LowLevelType {
    if field_lltypes.is_empty() {
        return LowLevelType::Void;
    }
    let n = field_lltypes.len();
    let name = format!("tuple{n}");
    let fields = field_lltypes
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("item{i}"), t.clone()))
        .collect();
    let body = Struct::gc_with_hints(
        &name,
        fields,
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("noidentity".into(), ConstValue::Bool(true)),
        ],
    );
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(body),
    }))
}

/// RPython `rtype_newtuple(hop): return TupleRepr._rtype_newtuple(hop)`
/// (rtuple.py).
pub fn rtype_newtuple(
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    TupleRepr::rtype_newtuple(hop)
}

/// RPython `dum_empty_tuple` PBC sentinel (rtuple.py).
pub fn dum_empty_tuple() {}

/// RPython `_gen_eq_function_cache` (rtuple.py) + `gen_eq_function`
/// (rtuple.py).
///
/// ```python
/// _gen_eq_function_cache = {}
///
/// def gen_eq_function(items_r):
///     eq_funcs = [r_item.get_ll_eq_function() or operator.eq for r_item in items_r]
///     key = tuple(eq_funcs)
///     try:
///         return _gen_eq_function_cache[key]
///     except KeyError:
///         autounrolling_funclist = unrolling_iterable(enumerate(eq_funcs))
///         def ll_eq(t1, t2):
///             equal_so_far = True
///             for i, eqfn in autounrolling_funclist:
///                 if not equal_so_far:
///                     return False
///                 attrname = 'item%d' % i
///                 item1 = getattr(t1, attrname)
///                 item2 = getattr(t2, attrname)
///                 equal_so_far = eqfn(item1, item2)
///             return equal_so_far
///         _gen_eq_function_cache[key] = ll_eq
///         return ll_eq
/// ```
///
/// Synthesizes a per-shape `ll_eq(t1, t2) -> Bool` helper graph.
/// Upstream's `unrolling_iterable` is a meta-construct that the RPython
/// translator unrolls into a concrete IR per shape; pyre's port skips
/// that intermediate Python source-level synthesis and emits the
/// already-unrolled graph directly through
/// `lowlevel_helper_function_with_builder`.
///
/// The graph shape (for items `[r_0, ..., r_{n-1}]`):
///
/// ```text
/// startblock(t1, t2):
///     item1_0 = getfield(t1, "item0")
///     item2_0 = getfield(t2, "item0")
///     eq_0 = <prim_eq>(item1_0, item2_0)
///     -> [eq_0=true: block_1(t1, t2), eq_0=false: returnblock(false)]
/// block_1(t1, t2):
///     item1_1 = getfield(t1, "item1")
///     ...
/// block_{n-1}(t1, t2):
///     ...
///     eq_{n-1} = <prim_eq>(...)
///     -> [eq_{n-1}=true: returnblock(true), eq_{n-1}=false: returnblock(false)]
/// ```
///
/// `<prim_eq>` is selected per item lltype:
/// * Signed/Bool → `int_eq`
/// * Unsigned → `uint_eq`
/// * Long-long variants → `<prefix>eq`
/// * Float → `float_eq`
///
/// Other item lltypes (Char, Ptr, etc.) are not yet supported — the
/// builder returns a `TyperError` so callers can extend the dispatch
/// as needed. Upstream's `r_item.get_ll_eq_function()` override path
/// (e.g. `StringRepr` returning `ll_streq`) is preserved via the
/// [`Repr::get_ll_eq_function`] trait method, but no current Repr
/// overrides it; this slice lands the primitive fallback.
pub fn gen_eq_function(
    rtyper: &RPythonTyper,
    items_r: &[Arc<dyn Repr>],
) -> Result<LowLevelFunction, TyperError> {
    let lltypes: Vec<LowLevelType> = items_r.iter().map(|r| r.lowleveltype().clone()).collect();
    let tuple_lltype = TUPLE_TYPE(&lltypes);

    let args = vec![tuple_lltype.clone(), tuple_lltype.clone()];
    let result = LowLevelType::Bool;

    // upstream rtuple.py:32 —
    //   `eq_funcs = [r_item.get_ll_eq_function() or operator.eq for r_item in items_r]`
    // Per-item: if `r.get_ll_eq_function()` returns Some(helper), emit
    // a direct_call to that helper; if it returns None, fall back to
    // the primitive `int_eq`/`float_eq`/... inline op (operator.eq's
    // lltype-level expansion). Resolve eagerly so nested helpers
    // (e.g. tuple-of-tuple → inner `gen_eq_function`) register before
    // the outer helper's builder runs.
    let item_helpers: Vec<Option<LowLevelFunction>> = items_r
        .iter()
        .map(|r| r.get_ll_eq_function(rtyper))
        .collect::<Result<_, _>>()?;

    // upstream `key = tuple(eq_funcs)` (rtuple.py) — keyed by helper
    // identity, not lltype. Pyre encodes the helper-tuple identity into
    // the `lowlevel_helper_function_with_builder` cache key (which is
    // `(name, args, result)`) by composing per-item suffixes from
    // `Some(helper).name` (helper graph identity) or the lltype short
    // name plus a `:eq` sentinel for `None` (primitive op fallback).
    // This prevents two distinct Reprs with the same lowleveltype but
    // different helpers from colliding in the helper cache.
    let name = format!(
        "ll_tuple_eq_{}",
        helper_identity_suffix(&lltypes, &item_helpers, "eq")
    );

    let item_lltypes = lltypes.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name.clone(),
        args,
        result,
        move |rtyper, args, result| {
            build_gen_eq_function_graph(&name, args, result, &item_lltypes, &item_helpers, rtyper)
        },
    )
}

/// Encodes a tuple of (lltype, helper) pairs into a name suffix that
/// uniquely identifies the cache key. Mirrors upstream's
/// `tuple(eq_funcs)` / `tuple(hash_funcs)` (rtuple.py, :56) —
/// helper-identity keying, not lltype keying. For `Some(helper)` the
/// helper's globally-unique name (assigned at synthesis time by
/// `lowlevel_helper_function_with_builder`) carries the identity; for
/// `None` the lltype short name + a per-slot kind sentinel
/// (`eq`/`hash`) suffices since the primitive op fallback is a
/// deterministic function of the lltype.
fn helper_identity_suffix(
    lltypes: &[LowLevelType],
    item_helpers: &[Option<LowLevelFunction>],
    kind: &str,
) -> String {
    lltypes
        .iter()
        .zip(item_helpers.iter())
        .map(|(lltype, helper)| match helper {
            Some(h) => h.name.clone(),
            None => format!("{}:{}", lltype.short_name(), kind),
        })
        .collect::<Vec<_>>()
        .join("_")
}

/// Encodes a tuple of item lltypes into a name suffix used for
/// helper-graph names whose identity is fully determined by the
/// lltype (e.g. tuple constructor / `getitem` / `len` helpers, where
/// no per-Repr override exists). The cross-Repr sites that vary by
/// helper identity (eq/hash) use [`helper_identity_suffix`] instead.
#[allow(dead_code)]
fn lltype_shape_suffix(lltypes: &[LowLevelType]) -> String {
    lltypes
        .iter()
        .map(|t| t.short_name().to_string())
        .collect::<Vec<_>>()
        .join("_")
}

/// RPython `_gen_hash_function_cache` (rtuple.py) +
/// `gen_hash_function` (rtuple.py).
///
/// ```python
/// def gen_hash_function(items_r):
///     hash_funcs = [r_item.get_ll_hash_function() for r_item in items_r]
///     key = tuple(hash_funcs)
///     try:
///         return _gen_hash_function_cache[key]
///     except KeyError:
///         autounrolling_funclist = unrolling_iterable(enumerate(hash_funcs))
///         def ll_hash(t):
///             """Must be kept in sync with rlib.objectmodel._hash_tuple()."""
///             x = 0x345678
///             for i, hash_func in autounrolling_funclist:
///                 attrname = 'item%d' % i
///                 item = getattr(t, attrname)
///                 y = hash_func(item)
///                 x = intmask((1000003 * x) ^ y)
///             return x
///         _gen_hash_function_cache[key] = ll_hash
///         return ll_hash
/// ```
///
/// Synthesizes a per-shape `ll_hash(t) -> Signed` helper graph.
/// CPython-style accumulator with `0x345678` seed and the
/// `(1000003 * x) ^ y` mixing per item. Per-item `y_i = hash_func_i(item_i)`
/// is dispatched via `direct_call` to the helper returned by
/// `r_item.get_ll_hash_function()` — every item Repr must define a
/// hash helper (rmodel.py base default raises). Resolved eagerly
/// so nested helpers register before the outer helper's builder runs.
pub fn gen_hash_function(
    rtyper: &RPythonTyper,
    items_r: &[Arc<dyn Repr>],
) -> Result<LowLevelFunction, TyperError> {
    let lltypes: Vec<LowLevelType> = items_r.iter().map(|r| r.lowleveltype().clone()).collect();
    let tuple_lltype = TUPLE_TYPE(&lltypes);
    let args = vec![tuple_lltype];
    let result = LowLevelType::Signed;

    // upstream rtuple.py:55 —
    //   `hash_funcs = [r_item.get_ll_hash_function() for r_item in items_r]`
    // No `or operator.eq`-style fallback: every primitive Repr
    // (`IntegerRepr`, `BoolRepr`, `NoneRepr`, ...) must override
    // `get_ll_hash_function` to return a helper graph. Resolved before
    // the builder closure so nested helpers (tuple-of-tuple) register
    // their graphs first.
    let item_helpers: Vec<LowLevelFunction> = items_r
        .iter()
        .map(|r| {
            r.get_ll_hash_function(rtyper)?.ok_or_else(|| {
                TyperError::message(format!(
                    "gen_hash_function: {} returned None from get_ll_hash_function — \
                     no Repr in pyre may return None from this slot",
                    r.repr_string()
                ))
            })
        })
        .collect::<Result<_, _>>()?;

    // upstream `key = tuple(hash_funcs)` (rtuple.py) — keyed by
    // helper identity. Pyre composes the cache key from each helper's
    // unique name so two Reprs with the same lowleveltype but
    // different hash helpers do not collide. Always-Some here (no
    // primitive fallback for hash), so the lltype-shape arm of
    // `helper_identity_suffix` is unreachable.
    let item_helpers_opt: Vec<Option<LowLevelFunction>> =
        item_helpers.iter().map(|h| Some(h.clone())).collect();
    let name = format!(
        "ll_tuple_hash_{}",
        helper_identity_suffix(&lltypes, &item_helpers_opt, "hash")
    );

    let item_lltypes = lltypes.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name.clone(),
        args,
        result,
        move |rtyper, args, result| {
            build_gen_hash_function_graph(&name, args, result, &item_lltypes, &item_helpers, rtyper)
        },
    )
}

thread_local! {
    /// RPython `_gen_str_function_cache = {}` (rtuple.py).
    ///
    /// Upstream keys this dict by `tuple([r_item.ll_str for r_item in
    /// items_r])`. `LowLevelFunction` carries `Rc<PyGraph>`, so the Rust
    /// cache is thread-local rather than process-global `Sync`.
    #[allow(non_upper_case_globals)]
    pub static _gen_str_function_cache: RefCell<HashMap<Vec<String>, LowLevelFunction>> =
        RefCell::new(HashMap::new());
}

/// RPython `gen_str_function(tuplerepr)` (rtuple.py).
///
/// The upstream helper builds `ll_str(t)` through `annlowlevel.llstr`,
/// `annlowlevel.hlstr`, and `rlib.rstring.StringBuilder`. Those pieces
/// are not all exposed through the Rust helper-graph path yet, so the
/// exact module-level name is present and returns a structured missing
/// rtype operation until that dependency chain lands.
pub fn gen_str_function(
    _rtyper: &RPythonTyper,
    _tuplerepr: &TupleRepr,
) -> Result<LowLevelFunction, TyperError> {
    Err(rtuple_deferred("gen_str_function"))
}

fn build_gen_hash_function_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    item_lltypes: &[LowLevelType],
    item_helpers: &[LowLevelFunction],
    rtyper: &RPythonTyper,
) -> Result<PyGraph, TyperError> {
    if args.len() != 1 {
        return Err(TyperError::message(format!(
            "gen_hash_function: expected 1 arg, got {}",
            args.len()
        )));
    }
    if result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "gen_hash_function: result must be Signed, got {result:?}"
        )));
    }
    if item_lltypes.len() != item_helpers.len() {
        return Err(TyperError::message(format!(
            "gen_hash_function: item_lltypes ({}) and item_helpers ({}) length mismatch",
            item_lltypes.len(),
            item_helpers.len()
        )));
    }
    let n = item_lltypes.len();
    let tuple_lltype = args[0].clone();
    let argnames = vec!["t".to_string()];

    // Single-block linear graph: read each item, dispatch to its
    // hash helper, mix into x. For n=0 (upstream rtuple.py ll_hash with
    // empty `autounrolling_funclist`) the loop is skipped and the
    // helper returns the seed `0x345678`.
    let t_arg = variable_with_lltype("t", tuple_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(t_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // x_0 = 0x345678
    let mut current_x = constant_with_lltype(ConstValue::Int(0x345678), LowLevelType::Signed);

    for i in 0..n {
        let item_lltype = item_lltypes[i].clone();
        // upstream rtuple.py:67-69 —
        //   item = getattr(t, attrname)
        //   y = hash_func(item)
        let item_var = variable_with_lltype(&format!("item_{i}"), item_lltype.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(t_arg.clone()),
                void_field_const(&format!("item{i}")),
            ],
            Hlvalue::Variable(item_var.clone()),
        ));
        let y_i = variable_with_lltype(&format!("hash_{i}"), LowLevelType::Signed);
        let func_const = helper_func_constant(rtyper, &item_helpers[i])?;
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(func_const), Hlvalue::Variable(item_var)],
            Hlvalue::Variable(y_i.clone()),
        ));
        // mul_i = int_mul(x_i, 1000003)
        let mul_i = variable_with_lltype(&format!("mul_{i}"), LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_mul",
            vec![
                current_x.clone(),
                constant_with_lltype(ConstValue::Int(1000003), LowLevelType::Signed),
            ],
            Hlvalue::Variable(mul_i.clone()),
        ));
        // x_{i+1} = int_xor(mul_i, y_i)
        let next_x = variable_with_lltype(&format!("x_{}", i + 1), LowLevelType::Signed);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_xor",
            vec![Hlvalue::Variable(mul_i), Hlvalue::Variable(y_i)],
            Hlvalue::Variable(next_x.clone()),
        ));
        current_x = Hlvalue::Variable(next_x);
    }

    // Close startblock: single link to returnblock with current_x.
    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(vec![current_x], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

/// Resolves a `LowLevelFunction` to its `Hlvalue::Constant` carrier
/// suitable as the first argument of a `direct_call` SpaceOperation.
/// Mirrors the carrier construction in
/// [`LowLevelOpList::gendirectcall`] (rtyper.rs) but for
/// builders that emit ops directly into a synthesized graph rather
/// than via the `LowLevelOpList` API. The graph reference is taken
/// from `helper.graph`, dereferenced through `rtyper.getcallable` to
/// produce a typed function pointer; the resulting `LLPtr` ConstValue
/// carries the matching `Ptr(FuncType)` lowleveltype.
fn helper_func_constant(
    rtyper: &RPythonTyper,
    helper: &LowLevelFunction,
) -> Result<Constant, TyperError> {
    let graph = helper.graph.as_ref().ok_or_else(|| {
        TyperError::message(format!(
            "helper_func_constant: low-level helper {} has no annotated graph",
            helper.name
        ))
    })?;
    let func_ptr = rtyper.getcallable(graph)?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &func_ptr_type,
        &ConstValue::LLPtr(Box::new(func_ptr)),
    )
}

/// Per-item primitive equality op selection. Mirrors
/// `rint.integer_opprefix_for` / `rfloat`'s `float_eq` surface.
fn primitive_eq_opname(lltype: &LowLevelType) -> Result<&'static str, TyperError> {
    Ok(match lltype {
        LowLevelType::Signed | LowLevelType::Bool => "int_eq",
        LowLevelType::Unsigned => "uint_eq",
        LowLevelType::SignedLongLong => "llong_eq",
        LowLevelType::SignedLongLongLong => "lllong_eq",
        LowLevelType::UnsignedLongLong => "ullong_eq",
        LowLevelType::UnsignedLongLongLong => "ulllong_eq",
        LowLevelType::Float => "float_eq",
        // rstr.py + rstr.py pairtype(AbstractCharRepr,
        // AbstractCharRepr).rtype_eq lowers to `char_eq`. Same for
        // unichar via rstr.py:778-779.
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        other => {
            return Err(TyperError::message(format!(
                "gen_eq_function: primitive equality not yet supported for {other:?}"
            )));
        }
    })
}

fn build_gen_eq_function_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    item_lltypes: &[LowLevelType],
    item_helpers: &[Option<LowLevelFunction>],
    rtyper: &RPythonTyper,
) -> Result<PyGraph, TyperError> {
    if item_lltypes.len() != item_helpers.len() {
        return Err(TyperError::message(format!(
            "gen_eq_function: item_lltypes ({}) and item_helpers ({}) length mismatch",
            item_lltypes.len(),
            item_helpers.len()
        )));
    }
    if args.len() != 2 {
        return Err(TyperError::message(format!(
            "gen_eq_function: expected 2 args, got {}",
            args.len()
        )));
    }
    if result != &LowLevelType::Bool {
        return Err(TyperError::message(format!(
            "gen_eq_function: result must be Bool, got {result:?}"
        )));
    }
    let n = item_lltypes.len();
    let tuple_lltype = args[0].clone();
    let argnames = vec!["t1".to_string(), "t2".to_string()];

    // upstream rtuple.py ll_eq — empty `autounrolling_funclist` leaves
    // `equal_so_far = True`. Synthesize a single-block graph that
    // returns the constant True directly. Tuple lltype here is
    // `Void` (rtuple.py:120-121 short-circuit), so the inputargs are
    // Void carriers.
    if n == 0 {
        let t1_arg = variable_with_lltype("t1", tuple_lltype.clone());
        let t2_arg = variable_with_lltype("t2", tuple_lltype.clone());
        let startblock = Block::shared(vec![Hlvalue::Variable(t1_arg), Hlvalue::Variable(t2_arg)]);
        let return_var = variable_with_lltype("result", LowLevelType::Bool);
        let mut graph = FunctionGraph::with_return_var(
            name.to_string(),
            startblock.clone(),
            Hlvalue::Variable(return_var),
        );
        let true_const = constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
        use crate::flowspace::model::BlockRefExt;
        startblock.closeblock(vec![
            Link::new(vec![true_const], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let func = GraphFunc::new(
            name.to_string(),
            Constant::new(ConstValue::Dict(Default::default())),
        );
        graph.func = Some(func.clone());
        return Ok(helper_pygraph_from_graph(graph, argnames, func));
    }

    let mut check_blocks: Vec<crate::flowspace::model::BlockRef> = Vec::with_capacity(n);
    for i in 0..n {
        let bt1 = variable_with_lltype(&format!("t1_b{i}"), tuple_lltype.clone());
        let bt2 = variable_with_lltype(&format!("t2_b{i}"), tuple_lltype.clone());
        check_blocks.push(Block::shared(vec![
            Hlvalue::Variable(bt1),
            Hlvalue::Variable(bt2),
        ]));
    }
    let startblock = check_blocks[0].clone();
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock, Hlvalue::Variable(return_var));

    for i in 0..n {
        // Read the per-block t1, t2 inputargs.
        let (bt1, bt2) = {
            let bvars = check_blocks[i].borrow().inputargs.clone();
            (bvars[0].clone(), bvars[1].clone())
        };
        let item_lltype = item_lltypes[i].clone();
        let item1 = variable_with_lltype(&format!("item1_{i}"), item_lltype.clone());
        let item2 = variable_with_lltype(&format!("item2_{i}"), item_lltype.clone());
        let eq_var = variable_with_lltype(&format!("eq_{i}"), LowLevelType::Bool);
        let fieldname = format!("item{i}");

        let block = check_blocks[i].clone();
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![bt1.clone(), void_field_const(&fieldname)],
            Hlvalue::Variable(item1.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![bt2.clone(), void_field_const(&fieldname)],
            Hlvalue::Variable(item2.clone()),
        ));
        // upstream rtuple.py:42 — `item_eq = eq_func(item1, item2)`.
        // If `r_item.get_ll_eq_function()` returned a helper, emit a
        // direct_call to it; otherwise fall through to the primitive
        // `int_eq`/`float_eq`/... inline op (operator.eq's lltype-level
        // expansion).
        match &item_helpers[i] {
            Some(helper) => {
                let func_const = helper_func_constant(rtyper, helper)?;
                block.borrow_mut().operations.push(SpaceOperation::new(
                    "direct_call",
                    vec![
                        Hlvalue::Constant(func_const),
                        Hlvalue::Variable(item1),
                        Hlvalue::Variable(item2),
                    ],
                    Hlvalue::Variable(eq_var.clone()),
                ));
            }
            None => {
                if item_lltype == LowLevelType::Void {
                    // upstream rtuple.py — `r_item.get_ll_eq_function() or
                    // operator.eq`. For `NoneRepr`/Void items the fallback
                    // is `operator.eq` which RPython's annotator+rtyper
                    // const-folds to `Constant(True, Bool)` (two Void
                    // values are vacuously equal). Mirror via a
                    // `same_as` op so the per-item exitswitch still has
                    // a Variable to read.
                    block.borrow_mut().operations.push(SpaceOperation::new(
                        "same_as",
                        vec![constant_with_lltype(
                            ConstValue::Bool(true),
                            LowLevelType::Bool,
                        )],
                        Hlvalue::Variable(eq_var.clone()),
                    ));
                } else {
                    let opname = primitive_eq_opname(&item_lltype)?;
                    block.borrow_mut().operations.push(SpaceOperation::new(
                        opname,
                        vec![Hlvalue::Variable(item1), Hlvalue::Variable(item2)],
                        Hlvalue::Variable(eq_var.clone()),
                    ));
                }
            }
        }
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq_var));

        // True branch.
        let true_link = if i + 1 < n {
            // Pass tuple pointers to next check block.
            Link::new(
                vec![bt1.clone(), bt2.clone()],
                Some(check_blocks[i + 1].clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref()
        } else {
            // Last block — return true.
            Link::new(
                vec![constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )],
                Some(graph.returnblock.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref()
        };
        // False branch — short-circuit return false.
        let false_link = Link::new(
            vec![constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref();

        use crate::flowspace::model::BlockRefExt;
        block.closeblock(vec![true_link, false_link]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_is_` (rtuple.py):
///
/// ```python
/// def rtype_is_((robj1, robj2), hop):
///     raise TyperError("cannot compare tuples with 'is'")
/// ```
///
/// Eagerly rejected by the rtyper so the generic `(Repr,
/// Repr).rtype_is_` arm cannot fall through to a `ptr_eq` on tuple
/// pointers — upstream considers identity comparison on tuple
/// values structurally meaningless. Both args are accepted as
/// `&dyn Repr` even though only the type name matters; the parity
/// of the error message preserves caller-visible behaviour.
pub fn pair_tuple_tuple_rtype_is_(
    _r1: &dyn Repr,
    _r2: &dyn Repr,
    _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    Err(TyperError::message("cannot compare tuples with 'is'"))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_eq` (rtuple.py):
///
/// ```python
/// def rtype_eq((r_tup1, r_tup2), hop):
///     s_tup = annmodel.unionof(*hop.args_s)
///     r_tup = hop.rtyper.getrepr(s_tup)
///     v_tuple1, v_tuple2 = hop.inputargs(r_tup, r_tup)
///     ll_eq = r_tup.get_ll_eq_function()
///     return hop.gendirectcall(ll_eq, v_tuple1, v_tuple2)
/// ```
///
/// Computes the unified TupleRepr (so e.g. `(int, float) ==
/// (long, float)` lifts both sides to `(long, float)`), then
/// dispatches to the per-shape `ll_eq` helper synthesized by
/// `gen_eq_function`. The result is a `Bool` Variable from the
/// `direct_call` lowering.
pub fn pair_tuple_tuple_rtype_eq(
    _r1: &dyn Repr,
    _r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    use crate::annotator::model::unionof;
    use crate::translator::rtyper::rtyper::ConvertedTo;
    // upstream `s_tup = annmodel.unionof(*hop.args_s)`.
    let s_args = hop.args_s.borrow().clone();
    let s_tup = unionof(s_args.iter()).map_err(|e| TyperError::message(e.to_string()))?;
    // upstream `r_tup = hop.rtyper.getrepr(s_tup)`.
    let r_tup_arc = hop.rtyper.getrepr(&s_tup)?;
    let any_r: &dyn std::any::Any = r_tup_arc.as_ref();
    let r_tup = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_rtype_eq: unionof's repr is not a TupleRepr")
    })?;
    // upstream `v_tuple1, v_tuple2 = hop.inputargs(r_tup, r_tup)`.
    let v_args = hop.inputargs(vec![ConvertedTo::Repr(r_tup), ConvertedTo::Repr(r_tup)])?;
    let v_tuple1 = v_args[0].clone();
    let v_tuple2 = v_args[1].clone();
    // upstream `ll_eq = r_tup.get_ll_eq_function()`.
    let ll_eq = r_tup.get_ll_eq_function(&hop.rtyper)?.ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_rtype_eq: TupleRepr.get_ll_eq_function returned None")
    })?;
    // upstream `hop.gendirectcall(ll_eq, v_tuple1, v_tuple2)`.
    let v_result = hop
        .gendirectcall(&ll_eq, vec![v_tuple1, v_tuple2])?
        .ok_or_else(|| {
            TyperError::message("pair_tuple_tuple_rtype_eq: gendirectcall must yield a Bool result")
        })?;
    Ok(Some(v_result))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_ne` (rtuple.py):
///
/// ```python
/// def rtype_ne(tup1tup2, hop):
///     v_res = tup1tup2.rtype_eq(hop)
///     return hop.genop('bool_not', [v_res], resulttype=Bool)
/// ```
///
/// Computes equality via `rtype_eq` then inverts with `bool_not`.
pub fn pair_tuple_tuple_rtype_ne(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    let v_res = pair_tuple_tuple_rtype_eq(r1, r2, hop)?
        .ok_or_else(|| TyperError::message("pair_tuple_tuple_rtype_ne: rtype_eq returned None"))?;
    let v_neg = hop.genop(
        "bool_not",
        vec![v_res],
        GenopResult::LLType(LowLevelType::Bool),
    );
    Ok(v_neg)
}

/// RPython `pairtype(TupleRepr, TupleRepr).convert_from_to`
/// (rtuple.py):
///
/// ```python
/// def convert_from_to((r_from, r_to), v, llops):
///     if len(r_from.items_r) == len(r_to.items_r):
///         if r_from.lowleveltype == r_to.lowleveltype:
///             return v
///         n = len(r_from.items_r)
///         items_v = []
///         for i in range(n):
///             item_v = r_from.getitem_internal(llops, v, i)
///             item_v = llops.convertvar(item_v,
///                                           r_from.items_r[i],
///                                           r_to.items_r[i])
///             items_v.append(item_v)
///         return r_from.newtuple(llops, r_to, items_v)
///     return NotImplemented
/// ```
///
/// Same-arity tuple-to-tuple conversion via per-position
/// `getitem_internal` + `convertvar` then `newtuple` on the
/// destination repr. Different-arity returns `Ok(None)` (upstream's
/// `NotImplemented`).
pub fn pair_tuple_tuple_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let r_from_t = any_from.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_convert_from_to: r_from is not a TupleRepr")
    })?;
    let any_to: &dyn std::any::Any = r_to;
    let r_to_t = any_to.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_convert_from_to: r_to is not a TupleRepr")
    })?;
    // upstream rtuple.py:341 — different arity → NotImplemented.
    if r_from_t.items_r.len() != r_to_t.items_r.len() {
        return Ok(None);
    }
    // upstream rtuple.py:342-343 — same lowleveltype is identity.
    if r_from_t.lowleveltype() == r_to_t.lowleveltype() {
        return Ok(Some(v.clone()));
    }
    // upstream rtuple.py — per-item getitem_internal +
    // convertvar to the matching destination items_r position.
    let n = r_from_t.items_r.len();
    let mut items_v: Vec<Hlvalue> = Vec::with_capacity(n);
    for i in 0..n {
        let v_internal = r_from_t.getitem_internal(llops, v.clone(), i)?;
        let item_v = llops.convertvar(
            Hlvalue::Variable(v_internal),
            r_from_t.items_r[i].as_ref(),
            r_to_t.items_r[i].as_ref(),
        )?;
        items_v.push(item_v);
    }
    // upstream rtuple.py — `r_from.newtuple(llops, r_to, items_v)`.
    let result = TupleRepr::newtuple(llops, r_to_t, items_v)?;
    Ok(Some(result))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_add` (rtuple.py):
///
/// ```python
/// def rtype_add((r_tup1, r_tup2), hop):
///     v_tuple1, v_tuple2 = hop.inputargs(r_tup1, r_tup2)
///     vlist = []
///     for i in range(len(r_tup1.items_r)):
///         vlist.append(r_tup1.getitem_internal(hop.llops, v_tuple1, i))
///     for i in range(len(r_tup2.items_r)):
///         vlist.append(r_tup2.getitem_internal(hop.llops, v_tuple2, i))
///     return r_tup1.newtuple_cached(hop, vlist)
/// rtype_inplace_add = rtype_add
/// ```
///
/// Concatenates two tuples by emitting per-position `getfield` ops
/// on each side then dispatching to `newtuple_cached` to either
/// short-circuit to a const-result `inputconst` or emit a fresh
/// `malloc` + per-item `setfield` sequence. Aliased through the
/// pair dispatcher for both `add` and `inplace_add` opnames.
pub fn pair_tuple_tuple_rtype_add(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    use crate::translator::rtyper::rtyper::ConvertedTo;
    let any1: &dyn std::any::Any = r1;
    let r_tup1 = any1
        .downcast_ref::<TupleRepr>()
        .ok_or_else(|| TyperError::message("pair_tuple_tuple_rtype_add: r1 is not a TupleRepr"))?;
    let any2: &dyn std::any::Any = r2;
    let r_tup2 = any2
        .downcast_ref::<TupleRepr>()
        .ok_or_else(|| TyperError::message("pair_tuple_tuple_rtype_add: r2 is not a TupleRepr"))?;
    // upstream `v_tuple1, v_tuple2 = hop.inputargs(r_tup1, r_tup2)`.
    let v_args = hop.inputargs(vec![ConvertedTo::Repr(r_tup1), ConvertedTo::Repr(r_tup2)])?;
    let v_tuple1 = v_args[0].clone();
    let v_tuple2 = v_args[1].clone();
    // upstream loop — per-position getitem_internal on each side,
    // appended into a single vlist.
    let mut vlist: Vec<Hlvalue> = Vec::with_capacity(r_tup1.items_r.len() + r_tup2.items_r.len());
    {
        let mut llops = hop.llops.borrow_mut();
        for i in 0..r_tup1.items_r.len() {
            let v = r_tup1.getitem_internal(&mut llops, v_tuple1.clone(), i)?;
            vlist.push(Hlvalue::Variable(v));
        }
        for i in 0..r_tup2.items_r.len() {
            let v = r_tup2.getitem_internal(&mut llops, v_tuple2.clone(), i)?;
            vlist.push(Hlvalue::Variable(v));
        }
    }
    // upstream `r_tup1.newtuple_cached(hop, vlist)`.
    let result = TupleRepr::newtuple_cached(hop, vlist)?;
    Ok(Some(result))
}

/// RPython `_ll_equal(x, y): return x == y` (rtuple.py).
///
/// Per-type primitive equality helper used by
/// `pairtype(TupleRepr, Repr).rtype_contains` when the item Repr's
/// `get_ll_eq_function()` returns None. Synthesized per item lltype
/// via [`RPythonTyper::lowlevel_helper_function_with_builder`].
///
/// Body: `block(x, y): eq = prim_eq(x, y) Bool; return eq` — a
/// single op + a single link to returnblock.
pub fn ll_equal(
    rtyper: &RPythonTyper,
    item_lltype: &LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let name = format!("ll_equal_{}", item_lltype.short_name());
    let args = vec![item_lltype.clone(), item_lltype.clone()];
    let result = LowLevelType::Bool;
    let item_lltype_owned = item_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name.clone(),
        args,
        result,
        move |_rtyper, args, result| build_ll_equal_graph(&name, args, result, &item_lltype_owned),
    )
}

fn build_ll_equal_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    item_lltype: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    if args.len() != 2 || args[0] != *item_lltype || args[1] != *item_lltype {
        return Err(TyperError::message(format!(
            "ll_equal: expected 2 args of {item_lltype:?}, got {args:?}"
        )));
    }
    if result != &LowLevelType::Bool {
        return Err(TyperError::message("ll_equal: result must be Bool"));
    }
    let argnames = vec!["x".to_string(), "y".to_string()];
    let x_arg = variable_with_lltype("x", item_lltype.clone());
    let y_arg = variable_with_lltype("y", item_lltype.clone());
    let eq_var = variable_with_lltype("eq", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x_arg.clone()),
        Hlvalue::Variable(y_arg.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    if item_lltype == &LowLevelType::Void {
        // upstream rtuple.py — `r_item.get_ll_eq_function() or
        // _ll_equal`. For NoneRepr/Void items the fallback `_ll_equal`
        // would lower `x == y` on two Void operands; RPython's
        // annotator+rtyper const-fold this to `Constant(True, Bool)`.
        // Mirror via `same_as` so the helper graph still has a single
        // result Variable for the return link.
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )],
            Hlvalue::Variable(eq_var.clone()),
        ));
    } else {
        let opname = primitive_eq_opname(item_lltype)?;
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            vec![Hlvalue::Variable(x_arg), Hlvalue::Variable(y_arg)],
            Hlvalue::Variable(eq_var.clone()),
        ));
    }
    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(eq_var)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

/// RPython `pairtype(TupleRepr, Repr).rtype_contains` (rtuple.py):
///
/// ```python
/// def rtype_contains((r_tup, r_item), hop):
///     s_tup = hop.args_s[0]
///     if not s_tup.is_constant():
///         raise TyperError("contains() on non-const tuple")
///     t = s_tup.const
///     s_item = hop.args_s[1]
///     r_item = hop.args_r[1]
///     v_arg = hop.inputarg(r_item, arg=1)
///     ll_eq = r_item.get_ll_eq_function() or _ll_equal
///     v_result = None
///     for x in t:
///         s_const_item = hop.rtyper.annotator.bookkeeper.immutablevalue(x)
///         if not s_item.contains(s_const_item):
///             continue
///         c_tuple_item = hop.inputconst(r_item, x)
///         v_equal = hop.gendirectcall(ll_eq, v_arg, c_tuple_item)
///         if v_result is None:
///             v_result = v_equal
///         else:
///             v_result = hop.genop("int_or", [v_result, v_equal], resulttype=Bool)
///     hop.exception_cannot_occur()
///     return v_result or hop.inputconst(Bool, False)
/// ```
///
/// Constant-tuple membership test. Tuple must be statically constant;
/// each element compares against `v_arg` via the synthesized `ll_eq`
/// helper. Static `s_item.contains` filters tuple elements that the
/// annotator already proved cannot match. The OR-chain accumulates
/// `int_or` ops; an empty chain yields a constant `False`.
pub fn pair_tuple_repr_rtype_contains(
    r_tup: &dyn Repr,
    _r_item: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    use crate::annotator::model::SomeObjectTrait;
    use crate::translator::rtyper::rtyper::ConvertedTo;
    let any_r: &dyn std::any::Any = r_tup;
    let _r_tup_typed = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_repr_rtype_contains: r_tup is not a TupleRepr")
    })?;
    let (tuple_items, s_item, r_item) = {
        let args_s = hop.args_s.borrow();
        let s_tup = args_s
            .first()
            .ok_or_else(|| TyperError::message("rtype_contains: missing s_tup"))?
            .clone();
        if !s_tup.is_constant() {
            return Err(TyperError::message("contains() on non-const tuple"));
        }
        let const_tup = match s_tup.const_() {
            Some(ConstValue::Tuple(items)) => items.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "rtype_contains: s_tup.const must be Tuple, got {other:?}"
                )));
            }
        };
        let s_item_ann = args_s
            .get(1)
            .ok_or_else(|| TyperError::message("rtype_contains: missing s_item"))?
            .clone();
        let r_item_arc = hop
            .args_r
            .borrow()
            .get(1)
            .cloned()
            .flatten()
            .ok_or_else(|| TyperError::message("rtype_contains: missing r_item"))?;
        (const_tup, s_item_ann, r_item_arc)
    };
    let v_arg = hop.inputarg(ConvertedTo::Repr(r_item.as_ref()), 1)?;
    let ll_eq = match r_item.get_ll_eq_function(&hop.rtyper)? {
        Some(f) => f,
        None => ll_equal(&hop.rtyper, r_item.lowleveltype())?,
    };
    let mut v_result: Option<Hlvalue> = None;
    for x in &tuple_items {
        let s_const_item = bookkeeper_immutablevalue_for(&hop.rtyper, x)?;
        if !crate::annotator::model::contains(&s_item, &s_const_item) {
            continue;
        }
        // upstream rtuple.py — `c_tuple_item = hop.inputconst(r_item, x)`.
        // `inputconst` routes through `r_item.convert_const(x)` so per-Repr
        // conversions (bool→int, int→float, PBC/class/string constant) apply.
        // Stamping `Constant::with_concretetype` directly bypasses that
        // pipeline.
        let c_x = crate::translator::rtyper::rmodel::inputconst(r_item.as_ref(), x)?;
        let c_x_carrier = Hlvalue::Constant(c_x);
        let v_equal = hop
            .gendirectcall(&ll_eq, vec![v_arg.clone(), c_x_carrier])?
            .ok_or_else(|| {
                TyperError::message("rtype_contains: gendirectcall must yield a Bool")
            })?;
        v_result = Some(match v_result {
            None => v_equal,
            Some(v_prev) => hop
                .genop(
                    "int_or",
                    vec![v_prev, v_equal],
                    GenopResult::LLType(LowLevelType::Bool),
                )
                .ok_or_else(|| TyperError::message("rtype_contains: int_or must yield a Bool"))?,
        });
    }
    hop.exception_cannot_occur()?;
    let result = v_result.unwrap_or_else(|| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(false),
            LowLevelType::Bool,
        ))
    });
    Ok(Some(result))
}

/// RPython `bookkeeper.immutablevalue(x)` (rtuple.py):
///
/// ```python
/// s_const_item = hop.rtyper.annotator.bookkeeper.immutablevalue(x)
/// ```
///
/// Routes through the live annotator's bookkeeper so SomePBC,
/// SomeInstance, low-level pointer, None, etc. all map to the
/// canonical `SomeValue` upstream produces. The earlier inline
/// primitive-only mapping diverged for those cases.
fn bookkeeper_immutablevalue_for(
    rtyper: &RPythonTyper,
    value: &ConstValue,
) -> Result<crate::annotator::model::SomeValue, TyperError> {
    let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
        TyperError::message(
            "bookkeeper_immutablevalue_for: annotator weak ref dropped — \
             cannot reach bookkeeper.immutablevalue",
        )
    })?;
    annotator
        .bookkeeper
        .immutablevalue(value)
        .map_err(|e| TyperError::message(format!("bookkeeper.immutablevalue failed: {e}")))
}

/// RPython `pairtype(TupleRepr, IntegerRepr).rtype_getitem`
/// (rtuple.py) free-function entry. Routes through the
/// receiver `TupleRepr`'s `rtype_pair_getitem` method.
pub fn pair_tuple_int_rtype_getitem(
    r_tup: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    let any_r: &dyn std::any::Any = r_tup;
    let r_tup = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_int_rtype_getitem: receiver is not a TupleRepr")
    })?;
    r_tup.rtype_pair_getitem(hop)
}

/// RPython `class AbstractTupleIteratorRepr(IteratorRepr)` (rtuple.py).
///
/// The generic iterator protocol (`newiter` / `rtype_next`) is kept as
/// an explicit parity surface. Actual lowering needs the tuple iterator
/// helper graph path below plus a concrete `IteratorRepr` trait surface.
#[derive(Clone, Debug)]
pub struct AbstractTupleIteratorRepr {
    pub r_tuple_lowleveltype: LowLevelType,
    pub lowleveltype: LowLevelType,
}

impl AbstractTupleIteratorRepr {
    /// RPython `AbstractTupleIteratorRepr.newiter(self, hop)` (rtuple.py).
    pub fn newiter(
        &self,
        _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        Err(rtuple_deferred("AbstractTupleIteratorRepr.newiter"))
    }

    /// RPython `AbstractTupleIteratorRepr.rtype_next(self, hop)` (rtuple.py).
    pub fn rtype_next(
        &self,
        _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        Err(rtuple_deferred("AbstractTupleIteratorRepr.rtype_next"))
    }
}

/// RPython `class Length1TupleIteratorRepr(AbstractTupleIteratorRepr)`
/// (rtuple.py).
#[derive(Debug)]
pub struct Length1TupleIteratorRepr {
    pub r_tuple_lowleveltype: LowLevelType,
    pub lowleveltype: LowLevelType,
    pub item_repr: Arc<dyn Repr>,
    pub external_item_repr: Arc<dyn Repr>,
    pub ll_tupleiter: &'static str,
    pub ll_tuplenext: &'static str,
    state: ReprState,
}

impl Length1TupleIteratorRepr {
    /// RPython `Length1TupleIteratorRepr.__init__(self, r_tuple)`.
    pub fn new(r_tuple: &TupleRepr) -> Self {
        let body = Struct::gc(
            "tuple1iter",
            vec![("tuple".to_string(), r_tuple.lowleveltype().clone())],
        );
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(body),
        }));
        Self {
            r_tuple_lowleveltype: r_tuple.lowleveltype().clone(),
            lowleveltype,
            item_repr: r_tuple.items_r[0].clone(),
            external_item_repr: r_tuple.external_items_r[0].clone(),
            ll_tupleiter: "ll_tupleiter",
            ll_tuplenext: "ll_tuplenext",
            state: ReprState::new(),
        }
    }
}

impl Repr for Length1TupleIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "Length1TupleIteratorRepr"
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py) —
    /// `iter(iter(x)) <==> iter(x)`: an iterator is its own iterator, so
    /// the op is the identity on the receiver (mirroring
    /// [`super::rlist::ListIteratorRepr::rtype_iter`]).
    fn rtype_iter(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `IteratorRepr.rtype_method_next(self, hop)`
    /// (rmodel.py) — `iter.next()` delegates to `rtype_next`.
    fn rtype_method(
        &self,
        method_name: &str,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> RTypeResult {
        match method_name {
            "next" => self.rtype_next(hop),
            other => Err(TyperError::message(format!(
                "missing Length1TupleIteratorRepr.rtype_method_{other}"
            ))),
        }
    }

    /// RPython `AbstractTupleIteratorRepr.newiter(self, hop)` (rtuple.py).
    fn newiter(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        let r_tuple = {
            let args_r = hop.args_r.borrow();
            args_r.first().and_then(|o| o.clone()).ok_or_else(|| {
                TyperError::message("Length1TupleIteratorRepr.newiter: arg0 repr missing")
            })?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_tuple.as_ref())])?;
        let tuple_lltype = self.r_tuple_lowleveltype.clone();
        let iter_lltype = self.lowleveltype.clone();
        let tuple_for_builder = tuple_lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            self.ll_tupleiter.to_string(),
            vec![tuple_lltype],
            iter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_tupleiter_helper_graph(
                    "ll_tupleiter",
                    tuple_for_builder.clone(),
                    iter_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractTupleIteratorRepr.rtype_next(self, hop)` (rtuple.py).
    fn rtype_next(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let iter_lltype = self.lowleveltype.clone();
        let tuple_lltype = self.r_tuple_lowleveltype.clone();
        let item_lltype = self.item_repr.lowleveltype().clone();
        let iter_for_builder = iter_lltype.clone();
        let tuple_for_builder = tuple_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            self.ll_tuplenext.to_string(),
            vec![iter_lltype],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_tuplenext_helper_graph(
                    "ll_tuplenext",
                    iter_for_builder.clone(),
                    tuple_for_builder.clone(),
                    item_for_builder.clone(),
                )
            },
        )?;
        let v_res = hop
            .gendirectcall(&helper, v_iter)?
            .ok_or_else(|| TyperError::message("tuple rtype_next: ll_tuplenext returned Void"))?;
        let converted = hop.llops.borrow_mut().convertvar(
            v_res,
            self.item_repr.as_ref(),
            self.external_item_repr.as_ref(),
        )?;
        Ok(Some(converted))
    }
}

/// RPython `ll_tupleiter(ITERPTR, tuple)` (rtuple.py).
///
/// The executable lowering uses [`build_ll_tupleiter_helper_graph`] because
/// `hop.gendirectcall` needs a helper graph carrying malloc/setfield ops. Keep
/// this public symbol so the Rust module still exposes the upstream helper
/// surface.
pub fn ll_tupleiter(_iterptr: &LowLevelType, _tuple: Hlvalue) -> Result<Hlvalue, TyperError> {
    Err(rtuple_deferred("ll_tupleiter"))
}

/// RPython `ll_tuplenext(iter)` (rtuple.py).
///
/// The executable lowering uses [`build_ll_tuplenext_helper_graph`] so it can
/// model the branch and StopIteration edge in a flow graph. Keep this public
/// symbol for upstream helper-name parity.
pub fn ll_tuplenext(_iter: Hlvalue) -> Result<Hlvalue, TyperError> {
    Err(rtuple_deferred("ll_tuplenext"))
}

/// RPython `ll_tupleiter(ITERPTR, tuple)` (rtuple.py):
///
/// ```python
/// def ll_tupleiter(ITERPTR, tuple):
///     iter = malloc(ITERPTR.TO)
///     iter.tuple = tuple
///     return iter
/// ```
///
/// Single block: `malloc ITERPTR.TO (flavor=gc)` + `setfield(iter,
/// 'tuple', tuple)` → return iter.
pub(crate) fn build_ll_tupleiter_helper_graph(
    name: &str,
    tuple_lltype: LowLevelType,
    tupleiter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let tuple_arg = variable_with_lltype("tuple", tuple_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(tuple_arg.clone())]);
    let return_var = variable_with_lltype("result", tupleiter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let LowLevelType::Ptr(ptr) = &tupleiter_lltype else {
        return Err(TyperError::message(
            "build_ll_tupleiter_helper_graph: tupleiter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_tupleiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };
    let c1 = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let cflags = Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", tupleiter_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
        Hlvalue::Variable(v_iter.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("tuple"),
            Hlvalue::Variable(tuple_arg),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["tuple".to_string()],
        func,
    ))
}

/// RPython `ll_tuplenext(iter)` (rtuple.py), for length-1 tuples:
///
/// ```python
/// def ll_tuplenext(iter):
///     t = iter.tuple
///     if t:
///         iter.tuple = nullptr(typeOf(t).TO)
///         return t.item0
///     else:
///         raise StopIteration
/// ```
///
/// 2-block CFG plus `graph.exceptblock`: **start** reads `iter.tuple`,
/// `ptr_nonzero` branches → **cont** (null the field, return `item0`) or
/// raises `StopIteration`.
pub(crate) fn build_ll_tuplenext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    tuple_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("StopIteration")?;

    let v_tuple = variable_with_lltype("tuple", tuple_lltype.clone());
    let v_cond = variable_with_lltype("cond", LowLevelType::Bool);
    {
        let mut b = startblock.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("tuple"),
            ],
            Hlvalue::Variable(v_tuple.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "ptr_nonzero",
            vec![Hlvalue::Variable(v_tuple.clone())],
            Hlvalue::Variable(v_cond.clone()),
        ));
        b.exitswitch = Some(Hlvalue::Variable(v_cond.clone()));
    }

    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_tuple = variable_with_lltype("tuple", tuple_lltype.clone());
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_tuple.clone()),
    ]);

    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(iter_arg), Hlvalue::Variable(v_tuple)],
            Some(cont.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // `iter.tuple = nullptr(typeOf(t).TO)` (rtuple.py): a typed null
    // pointer, not the `None` sentinel.  The codewriter clears the ref field
    // from a `ConstValue::LLPtr` null; a `ConstValue::None` payload carried on a
    // `Ptr` concretetype is not accepted by `emit_const_r`.
    let LowLevelType::Ptr(tuple_ptr_t) = tuple_lltype.clone() else {
        return Err(TyperError::message(
            "build_ll_tuplenext_helper_graph: tuple_lltype must be a Ptr",
        ));
    };
    let null_tuple = Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(_ptr::new(*tuple_ptr_t, Ok(None)))),
        tuple_lltype,
    );
    let v_item = variable_with_lltype("item0", item_lltype);
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                void_field_const("tuple"),
                Hlvalue::Constant(null_tuple),
            ],
            Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
        ));
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(c_tuple), void_field_const("item0")],
            Hlvalue::Variable(v_item.clone()),
        ));
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_item)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

/// RPython `class TupleRepr(Repr)` (rtuple.py+).
///
/// Minimal slice — carries the item reprs + lowleveltype. Methods that
/// emit ops (`getitem`, `newtuple`, ...) land in follow-up commits.
#[derive(Debug)]
pub struct TupleRepr {
    /// RPython `self.items_r` (rtuple.py:132,138). The `internal_repr`
    /// per item from `externalvsinternal` — GC `InstanceRepr` items
    /// are mapped to the root `getinstancerepr(rtyper, None)`; non-GC
    /// items pass through unchanged.
    pub items_r: Vec<Arc<dyn Repr>>,
    /// RPython `self.external_items_r` (rtuple.py:133) — concrete
    /// per-position reprs preserved for `getitem`'s convertvar back to
    /// the surface type.
    pub external_items_r: Vec<Arc<dyn Repr>>,
    /// RPython `self.fieldnames = ['item%d' % i for i in ...]`
    /// (rtuple.py:139).
    pub fieldnames: Vec<String>,
    /// RPython `self.lltypes = [r.lowleveltype for r in items_r]`
    /// (rtuple.py:140).
    pub lltypes: Vec<LowLevelType>,
    /// RPython `self.lowleveltype = TUPLE_TYPE(self.lltypes)` (rtuple.py).
    lltype: LowLevelType,
    /// RPython `self.tuple_cache = {}` (rtuple.py). Caches the
    /// instantiated `_ptr` per `Vec<ConstValue>` key. `convert_const`
    /// matches upstream rtuple.py — it instantiates, inserts
    /// the pointer into this cache BEFORE filling fields, and then
    /// mutates the cached entry via a brief `borrow_mut` per field
    /// write so recursive `r.convert_const(obj)` calls stay
    /// re-entrancy-safe.
    tuple_cache: RefCell<HashMap<Vec<ConstValue>, _ptr>>,
    state: ReprState,
}

impl TupleRepr {
    /// RPython `TupleRepr.__init__(self, rtyper, items_r)` (rtuple.py).
    ///
    /// Splits each input item via [`externalvsinternal`]: GC
    /// `InstanceRepr` items are stored internally as the root
    /// `getinstancerepr(rtyper, None)` while the concrete repr is kept
    /// at `external_items_r` for `getitem` to convert back. Non-GC
    /// items pass through with `external == internal`.
    ///
    /// `gcref=True` arm is deferred (no `rgcref` port yet).
    pub fn new(rtyper: &Rc<RPythonTyper>, items_r: Vec<Arc<dyn Repr>>) -> Result<Self, TyperError> {
        let mut internal_items: Vec<Arc<dyn Repr>> = Vec::with_capacity(items_r.len());
        let mut external_items: Vec<Arc<dyn Repr>> = Vec::with_capacity(items_r.len());
        for item_r in items_r {
            let (external, internal) =
                crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_r, false)?;
            internal_items.push(internal);
            external_items.push(external);
        }
        let lltypes: Vec<LowLevelType> = internal_items
            .iter()
            .map(|r| r.lowleveltype().clone())
            .collect();
        let fieldnames = (0..internal_items.len())
            .map(|i| format!("item{i}"))
            .collect();
        let lltype = TUPLE_TYPE(&lltypes);
        Ok(TupleRepr {
            items_r: internal_items,
            external_items_r: external_items,
            fieldnames,
            lltypes,
            lltype,
            tuple_cache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        })
    }

    /// RPython `TupleRepr.instantiate(self)` (rtuple.py).
    ///
    /// ```python
    /// def instantiate(self):
    ///     if len(self.items_r) == 0:
    ///         return dum_empty_tuple     # PBC placeholder for an empty tuple
    ///     else:
    ///         return malloc(self.lowleveltype.TO)
    /// ```
    ///
    /// The empty-tuple `dum_empty_tuple` PBC sentinel is not modelled
    /// — `convert_const` short-circuits to `Constant(None, Void)` for
    /// empty inputs via [`Self::instantiate_empty`] (see
    /// [`Repr::convert_const`] impl). For non-empty tuples this
    /// allocates a Gc instance of the `tuple%d` GcStruct via
    /// `lltype::malloc` with default `immortal=False` —
    /// matching upstream rtuple.py `malloc(self.lowleveltype.TO)`.
    pub fn instantiate(&self) -> Result<_ptr, TyperError> {
        if self.items_r.is_empty() {
            return Err(TyperError::message(
                "TupleRepr.instantiate: empty-tuple sentinel uses Void short-circuit \
                 in convert_const; instantiate() should not be reached",
            ));
        }
        let LowLevelType::Ptr(ptr) = &self.lltype else {
            return Err(TyperError::message(format!(
                "TupleRepr.instantiate: lowleveltype is not Ptr, got {:?}",
                self.lltype
            )));
        };
        let inner: LowLevelType = match &ptr.TO {
            PtrTarget::Struct(body) => LowLevelType::Struct(Box::new(body.clone())),
            other => {
                return Err(TyperError::message(format!(
                    "TupleRepr.instantiate: Ptr target must be Struct, got {:?}",
                    other
                )));
            }
        };
        // upstream `malloc(self.lowleveltype.TO)` — defaults to
        // `flavor='gc', immortal=False`.
        lltype::malloc(inner, None, MallocFlavor::Gc, false).map_err(TyperError::message)
    }

    /// RPython empty-tuple branch of `instantiate` (rtuple.py):
    ///
    /// ```python
    /// def instantiate(self):
    ///     if len(self.items_r) == 0:
    ///         return dum_empty_tuple     # PBC placeholder for an empty tuple
    ///     ...
    /// ```
    ///
    /// Pyre returns a `Constant(None, Void)` carrier as the
    /// `dum_empty_tuple` analogue. Upstream's sentinel is a Python
    /// callable cached by identity in `tuple_cache`; the Rust port
    /// cannot key the typed cache (`HashMap<Vec<ConstValue>, _ptr>`)
    /// on a Void-typed value, so the structural parity is achieved
    /// by returning a deterministic `Constant(None, Void)` —
    /// `PartialEq`/`Hash` derive guarantees two emitted carriers
    /// compare equal, which is the only observable property
    /// upstream's identity cache provides.
    fn instantiate_empty(&self) -> Constant {
        Constant::with_concretetype(ConstValue::None, LowLevelType::Void)
    }

    /// RPython `TupleRepr.getitem_internal(self, llops, v_tuple, index)`
    /// (rtuple.py):
    ///
    /// ```python
    /// def getitem_internal(self, llops, v_tuple, index):
    ///     name = self.fieldnames[index]
    ///     llresult = self.lltypes[index]
    ///     cname = inputconst(Void, name)
    ///     return llops.genop('getfield', [v_tuple, cname], resulttype=llresult)
    /// ```
    pub fn getitem_internal(
        &self,
        llops: &mut LowLevelOpList,
        v_tuple: Hlvalue,
        index: usize,
    ) -> Result<Variable, TyperError> {
        let name = self.fieldnames.get(index).ok_or_else(|| {
            TyperError::message(format!(
                "TupleRepr.getitem_internal: index {index} out of range \
                 (len={})",
                self.fieldnames.len()
            ))
        })?;
        let llresult = self.lltypes[index].clone();
        let cname = Constant::with_concretetype(ConstValue::byte_str(name), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![v_tuple, Hlvalue::Constant(cname)],
                GenopResult::LLType(llresult),
            )
            .expect("LowLevelOpList::genop with GenopResult::LLType always yields a Variable"))
    }

    /// RPython `TupleRepr.getitem(self, llops, v_tuple, index)`
    /// (rtuple.py):
    ///
    /// ```python
    /// def getitem(self, llops, v_tuple, index):
    ///     v = self.getitem_internal(llops, v_tuple, index)
    ///     r_item = self.items_r[index]
    ///     r_external_item = self.external_items_r[index]
    ///     return llops.convertvar(v, r_item, r_external_item)
    /// ```
    ///
    /// Pyre's external == internal for every item today (no GCRef
    /// wrapping), so the convertvar reduces to identity. Once
    /// `externalvsinternal`'s GCRef arm lands, `convertvar` will route
    /// through the pairtype dispatch.
    pub fn getitem(
        &self,
        llops: &mut LowLevelOpList,
        v_tuple: Hlvalue,
        index: usize,
    ) -> Result<Hlvalue, TyperError> {
        let v = self.getitem_internal(llops, v_tuple, index)?;
        let r_item = self.items_r.get(index).cloned().ok_or_else(|| {
            TyperError::message(format!(
                "TupleRepr.getitem: index {index} out of range (len={})",
                self.items_r.len()
            ))
        })?;
        let r_external = self.external_items_r[index].clone();
        llops.convertvar(Hlvalue::Variable(v), r_item.as_ref(), r_external.as_ref())
    }

    /// RPython `TupleRepr.newtuple(cls, llops, r_tuple, items_v)`
    /// (rtuple.py):
    ///
    /// ```python
    /// @classmethod
    /// def newtuple(cls, llops, r_tuple, items_v):
    ///     assert len(r_tuple.items_r) == len(items_v)
    ///     for r_item, v_item in zip(r_tuple.items_r, items_v):
    ///         assert r_item.lowleveltype == v_item.concretetype
    ///     if len(r_tuple.items_r) == 0:
    ///         return inputconst(Void, ())
    ///     c1 = inputconst(Void, r_tuple.lowleveltype.TO)
    ///     cflags = inputconst(Void, {'flavor': 'gc'})
    ///     v_result = llops.genop('malloc', [c1, cflags],
    ///                            resulttype=r_tuple.lowleveltype)
    ///     for i in range(len(r_tuple.items_r)):
    ///         cname = inputconst(Void, r_tuple.fieldnames[i])
    ///         llops.genop('setfield', [v_result, cname, items_v[i]])
    ///     return v_result
    /// ```
    ///
    /// Builds a fresh tuple value at runtime by emitting a `malloc`
    /// op followed by per-item `setfield` ops. Empty tuples surface
    /// as a Void `()` Constant.
    pub fn newtuple(
        llops: &mut LowLevelOpList,
        r_tuple: &TupleRepr,
        items_v: Vec<Hlvalue>,
    ) -> Result<Hlvalue, TyperError> {
        if r_tuple.items_r.len() != items_v.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.newtuple: arity mismatch: r_tuple has {} items, items_v \
                 has {}",
                r_tuple.items_r.len(),
                items_v.len()
            )));
        }
        // upstream rtuple.py:155-157 —
        // `for r_item, v_item in zip(r_tuple.items_r, items_v):
        //      assert r_item.lowleveltype == v_item.concretetype`.
        // Each item's concretetype must match the matching items_r
        // repr's lowleveltype; Constant carriers and Variables both
        // expose concretetype.
        for (i, (r_item, v_item)) in r_tuple.items_r.iter().zip(items_v.iter()).enumerate() {
            let v_concrete = match v_item {
                Hlvalue::Variable(v) => v.concretetype(),
                Hlvalue::Constant(c) => c.concretetype.clone(),
            };
            let expected = r_item.lowleveltype();
            match v_concrete {
                Some(ref got) if got == expected => {}
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.newtuple: item {i} concretetype mismatch \
                         — items_r[{i}].lowleveltype = {expected:?}, \
                         items_v[{i}].concretetype = {other:?}"
                    )));
                }
            }
        }
        if r_tuple.items_r.is_empty() {
            // upstream `inputconst(Void, ())` — Void-typed empty tuple sentinel.
            return Ok(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Tuple(Vec::new()),
                LowLevelType::Void,
            )));
        }
        // upstream `c1 = inputconst(Void, r_tuple.lowleveltype.TO)`.
        // The Void-typed `c1` carries the inner Struct lltype as the
        // type-tag for `malloc`'s lowering.
        let LowLevelType::Ptr(ptr) = &r_tuple.lltype else {
            return Err(TyperError::message(
                "TupleRepr.newtuple: lowleveltype is not Ptr",
            ));
        };
        let inner_struct = match &ptr.TO {
            PtrTarget::Struct(body) => body.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "TupleRepr.newtuple: Ptr target must be Struct, got {:?}",
                    other
                )));
            }
        };
        let c1 = Constant::with_concretetype(
            ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
            LowLevelType::Void,
        );
        // upstream `cflags = inputconst(Void, {'flavor': 'gc'})`. We
        // surface the flavor via a Void-typed Str sentinel matching
        // how other malloc emitters in pyre encode it; the lowering
        // pass reads the type from `c1` and the flavor from `cflags`.
        let cflags =
            Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
        let v_result = llops
            .genop(
                "malloc",
                vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
                GenopResult::LLType(r_tuple.lltype.clone()),
            )
            .expect("malloc with non-Void result yields a Variable");
        let v_result_h = Hlvalue::Variable(v_result);
        for (i, v_item) in items_v.into_iter().enumerate() {
            let cname = Constant::with_concretetype(
                ConstValue::byte_str(&r_tuple.fieldnames[i]),
                LowLevelType::Void,
            );
            llops.genop(
                "setfield",
                vec![v_result_h.clone(), Hlvalue::Constant(cname), v_item],
                GenopResult::Void,
            );
        }
        Ok(v_result_h)
    }

    /// RPython `TupleRepr.rtype_len(self, hop)` (rtuple.py):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     return hop.inputconst(Signed, len(self.items_r))
    /// ```
    ///
    /// Inherent helper used by both the [`Repr::rtype_len`] override
    /// and direct callers. Renamed from `rtype_len` to avoid the
    /// inherent-vs-trait-impl name clash that would force UFCS at
    /// every call site.
    fn rtype_len_inherent(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::HighLevelOp;
        let _ = hop;
        let n = self.items_r.len() as i64;
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(n))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `TupleRepr._rtype_newtuple(cls, hop)` (rtuple.py):
    ///
    /// ```python
    /// @classmethod
    /// def _rtype_newtuple(cls, hop):
    ///     r_tuple = hop.r_result
    ///     vlist = hop.inputargs(*r_tuple.items_r)
    ///     return cls.newtuple_cached(hop, vlist)
    /// ```
    ///
    /// `newtuple_cached` (rtuple.py) routes through the
    /// constant-result fast path; otherwise it calls `cls.newtuple`.
    /// Both arms are implemented here.
    /// RPython `pairtype(TupleRepr, IntegerRepr).rtype_getitem`
    /// helper — used internally by [`pair_tuple_int_rtype_getitem`].
    fn rtype_pair_getitem(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::ConvertedTo;
        // upstream `v_tuple, v_index = hop.inputargs(r_tup, Signed)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        let v_tuple = v_args[0].clone();
        let v_index = &v_args[1];
        // upstream `if not isinstance(v_index, Constant): raise
        // TyperError("non-constant tuple index")`.
        let Hlvalue::Constant(idx_const) = v_index else {
            return Err(TyperError::message(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: non-constant tuple index",
            ));
        };
        let ConstValue::Int(idx) = idx_const.value else {
            return Err(TyperError::message(format!(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: tuple index must be Int, got {:?}",
                idx_const.value
            )));
        };
        // upstream rtuple.py:270-271 — drop the implicit IndexError
        // exception channel:
        //   `if hop.has_implicit_exception(IndexError):
        //        hop.exception_cannot_occur()`
        // The constant index is verified at typer-time so a runtime
        // IndexError cannot occur.
        if hop.has_implicit_exception("IndexError") {
            hop.exception_cannot_occur()?;
        }
        // upstream rtuple.py:272-273 — `index = v_index.value;
        // r_tup.getitem(hop.llops, v_tuple, index)`.
        // RPython does not bounds-check or reject negatives itself —
        // `r_tup.getitem` flows through `self.fieldnames[index]`
        // which is a Python list supporting negative indexing
        // (`fieldnames[-1]` → last item). Mirror the wrap-around so
        // `(0, 1, 2)[-1]` types as `2` matching upstream.
        let len = self.items_r.len() as i64;
        let normalized = if idx < 0 { idx + len } else { idx };
        if normalized < 0 || normalized >= len {
            return Err(TyperError::message(format!(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: index {idx} out of \
                 range (len={len})"
            )));
        }
        let v = self.getitem(&mut hop.llops.borrow_mut(), v_tuple, normalized as usize)?;
        Ok(Some(v))
    }

    /// RPython `TupleRepr.newtuple_cached(cls, hop, items_v)`
    /// (rtuple.py):
    ///
    /// ```python
    /// @classmethod
    /// def newtuple_cached(cls, hop, items_v):
    ///     r_tuple = hop.r_result
    ///     if hop.s_result.is_constant():
    ///         return inputconst(r_tuple, hop.s_result.const)
    ///     else:
    ///         return cls.newtuple(hop.llops, r_tuple, items_v)
    /// ```
    ///
    /// `r_tuple` is read from `hop.r_result` and downcast to
    /// `TupleRepr`. The const-result fast path bypasses
    /// emit-and-fill in favour of a single `inputconst(r_tuple,
    /// hop.s_result.const)` carrier — matching upstream's behaviour
    /// of materialising a Constant tuple from the annotator's known
    /// const value.
    pub fn newtuple_cached(
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        items_v: Vec<Hlvalue>,
    ) -> Result<Hlvalue, TyperError> {
        use crate::translator::rtyper::rmodel::inputconst;
        let r_result =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("TupleRepr.newtuple_cached: r_result missing")
            })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr.newtuple_cached: hop.r_result is not a TupleRepr")
        })?;
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        if let Some(value) = s_const {
            let c = inputconst(r_tuple, &value)?;
            return Ok(Hlvalue::Constant(c));
        }
        TupleRepr::newtuple(&mut hop.llops.borrow_mut(), r_tuple, items_v)
    }

    /// RPython `class __extend__(TupleRepr).rtype_getslice(r_tup, hop)`
    /// (rtuple.py):
    ///
    /// ```python
    /// def rtype_getslice(r_tup, hop):
    ///     s_start = hop.args_s[1]
    ///     s_stop = hop.args_s[2]
    ///     assert s_start.is_immutable_constant(),"tuple slicing: needs constants"
    ///     assert s_stop.is_immutable_constant(), "tuple slicing: needs constants"
    ///     start = s_start.const
    ///     stop = s_stop.const
    ///     indices = range(len(r_tup.items_r))[start:stop]
    ///     assert len(indices) == len(hop.r_result.items_r)
    ///
    ///     v_tup = hop.inputarg(r_tup, arg=0)
    ///     items_v = [r_tup.getitem_internal(hop.llops, v_tup, i)
    ///                for i in indices]
    ///     return hop.r_result.newtuple(hop.llops, hop.r_result, items_v)
    /// ```
    ///
    /// Slice bounds must be immutable constants (compile-time).
    /// Negative indices wrap-around via Python's `range[start:stop]`
    /// semantics: the index is clamped to `[0, n]` after adding `n`
    /// to negatives. The result repr's `items_r` arity must match
    /// the slice length, which is the rtyper-supplied invariant
    /// (annotator computed `r_result` from the same start/stop).
    fn rtype_getslice_inherent(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::annotator::model::SomeObjectTrait;
        use crate::translator::rtyper::rtyper::ConvertedTo;
        // upstream rtuple.py:278-281 — `hop.args_s[1]` / `[2]` are
        // start / stop annotations. Both must be immutable
        // constants — slice bounds resolve at rtype-time.
        let (start, stop) = {
            let args_s = hop.args_s.borrow();
            let s_start = args_s.get(1).ok_or_else(|| {
                TyperError::message("TupleRepr.rtype_getslice: args_s[1] (start) missing")
            })?;
            let s_stop = args_s.get(2).ok_or_else(|| {
                TyperError::message("TupleRepr.rtype_getslice: args_s[2] (stop) missing")
            })?;
            if !s_start.is_immutable_constant() {
                return Err(TyperError::message("tuple slicing: needs constants"));
            }
            if !s_stop.is_immutable_constant() {
                return Err(TyperError::message("tuple slicing: needs constants"));
            }
            let start = match s_start.const_() {
                Some(ConstValue::Int(v)) => *v,
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.rtype_getslice: start must be Int constant, got {:?}",
                        other
                    )));
                }
            };
            let stop = match s_stop.const_() {
                Some(ConstValue::Int(v)) => *v,
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.rtype_getslice: stop must be Int constant, got {:?}",
                        other
                    )));
                }
            };
            (start, stop)
        };
        // upstream rtuple.py:284 — `indices = range(len(items_r))[start:stop]`.
        // Mirror Python's slice clamping: negatives add `n` then floor at 0;
        // out-of-range positives clamp to `n`.
        let n = self.items_r.len() as i64;
        let normalize = |i: i64| -> i64 { if i < 0 { (i + n).max(0) } else { i.min(n) } };
        let lo = normalize(start);
        let hi = normalize(stop);
        let indices: Vec<usize> = if lo < hi {
            (lo as usize..hi as usize).collect()
        } else {
            Vec::new()
        };
        // upstream rtuple.py — `assert len(indices) == len(hop.r_result.items_r)`.
        let r_result_arc = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("TupleRepr.rtype_getslice: r_result missing"))?;
        let any_r: &dyn std::any::Any = r_result_arc.as_ref();
        let r_result = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr.rtype_getslice: r_result is not a TupleRepr")
        })?;
        if indices.len() != r_result.items_r.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.rtype_getslice: slice length {} != r_result.items_r len {}",
                indices.len(),
                r_result.items_r.len()
            )));
        }
        // upstream rtuple.py — `v_tup = hop.inputarg(r_tup, arg=0)`.
        let v_tup = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        // upstream rtuple.py — per-index getitem_internal.
        let mut items_v: Vec<Hlvalue> = Vec::with_capacity(indices.len());
        {
            let mut llops = hop.llops.borrow_mut();
            for i in &indices {
                let v = self.getitem_internal(&mut llops, v_tup.clone(), *i)?;
                items_v.push(Hlvalue::Variable(v));
            }
        }
        // upstream rtuple.py — `r_result.newtuple(hop.llops, r_result, items_v)`.
        let result = TupleRepr::newtuple(&mut hop.llops.borrow_mut(), r_result, items_v)?;
        Ok(Some(result))
    }

    pub fn rtype_newtuple(
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::ConvertedTo;
        let r_result =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("TupleRepr._rtype_newtuple: r_result missing")
            })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr._rtype_newtuple: hop.r_result is not a TupleRepr")
        })?;
        // upstream `vlist = hop.inputargs(*r_tuple.items_r)`. Each
        // arg is coerced to the matching item repr.
        let converted: Vec<ConvertedTo<'_>> = r_tuple
            .items_r
            .iter()
            .map(|r| ConvertedTo::Repr(r.as_ref()))
            .collect();
        let vlist = hop.inputargs(converted)?;
        // upstream `cls.newtuple_cached(hop, vlist)`.
        let result = TupleRepr::newtuple_cached(hop, vlist)?;
        Ok(Some(result))
    }
}

impl Repr for TupleRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "TupleRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::TupleRepr
    }

    /// RPython `TupleRepr.compact_repr(self)` (rtuple.py):
    ///
    /// ```python
    /// def compact_repr(self):
    ///     return "TupleR %s" % ' '.join([llt._short_name() for llt in self.lltypes])
    /// ```
    ///
    /// Default `Repr.compact_repr` (rmodel.py) formats as
    /// `"{class_name_without_Repr} {lowleveltype._short_name()}"` —
    /// for tuples that produces `"TupleR Ptr <gcstruct tupleN>"`,
    /// hiding the per-item types. Upstream overrides to flatten the
    /// per-item lltype short names instead so debug output stays
    /// readable. `self.lltypes` upstream mirrors
    /// `[r.lowleveltype() for r in self.items_r]`.
    fn compact_repr(&self) -> String {
        let item_names: Vec<String> = self
            .items_r
            .iter()
            .map(|r| r.lowleveltype().short_name())
            .collect();
        format!("TupleR {}", item_names.join(" "))
    }

    /// `RPythonTyper.translate_op_len` (rtyper.py) dispatches
    /// `r.rtype_len(hop)` — without this override the default
    /// `Repr.rtype_len` would raise `MissingRTypeOperation` for tuples.
    /// Forwards to the inherent [`TupleRepr::rtype_len`] which mirrors
    /// upstream rtuple.py.
    fn rtype_len(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_len_inherent(hop)
    }

    /// `class __extend__(TupleRepr).rtype_getslice` (rtuple.py).
    /// Without this override `Repr.rtype_getslice` raises
    /// `MissingRTypeOperation("getslice")`. Forwards to the inherent
    /// helper.
    fn rtype_getslice(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_getslice_inherent(hop)
    }

    /// `TupleRepr.get_ll_eq_function(self): return gen_eq_function(self.items_r)`
    /// (rtuple.py). Returns a synthesized per-shape `ll_eq`
    /// helper that compares two tuple values element-by-element.
    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(Some(gen_eq_function(rtyper, &self.items_r)?))
    }

    /// `TupleRepr.get_ll_hash_function(self): return gen_hash_function(self.items_r)`
    /// (rtuple.py). Returns a synthesized per-shape `ll_hash`
    /// helper using the CPython-style mix.
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(Some(gen_hash_function(rtyper, &self.items_r)?))
    }

    /// RPython `TupleRepr.make_iterator_repr(self, variant=None)`
    /// (rtuple.py).  A tuple is immutable, so the list foldable
    /// flag is irrelevant here.
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    fn make_iterator_repr(
        &self,
        variant: &[String],
        _foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::message(format!(
                "unsupported {:?} iterator over a tuple",
                variant
            )));
        }
        if self.items_r.len() == 1 {
            return Ok(Arc::new(Length1TupleIteratorRepr::new(self)) as Arc<dyn Repr>);
        }
        Err(TyperError::message(
            "can only iterate over tuples of length 1 for now",
        ))
    }

    /// `Repr.rtype_hash` default raises `MissingRTypeOperation`. For
    /// tuples, dispatch to the synthesized `gen_hash_function`
    /// helper via `gendirectcall(ll_hash, [v_tuple])`. Mirrors
    /// upstream's implicit dispatch through `Repr.get_ll_hash_function`
    /// + `LowLevelOpList.gendirectcall` at op-translation time.
    fn rtype_hash(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::ConvertedTo;
        let v_args = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let v_tuple = v_args[0].clone();
        let ll_hash = self.get_ll_hash_function(&hop.rtyper)?.ok_or_else(|| {
            TyperError::message("TupleRepr.rtype_hash: get_ll_hash_function returned None")
        })?;
        let v_result = hop.gendirectcall(&ll_hash, vec![v_tuple])?.ok_or_else(|| {
            TyperError::message("TupleRepr.rtype_hash: gendirectcall must yield a Signed result")
        })?;
        Ok(Some(v_result))
    }

    /// RPython `TupleRepr.convert_const(self, value)` (rtuple.py).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     assert isinstance(value, tuple) and len(value) == len(self.items_r)
    ///     key = tuple([Constant(item) for item in value])
    ///     try:
    ///         return self.tuple_cache[key]
    ///     except KeyError:
    ///         p = self.instantiate()
    ///         self.tuple_cache[key] = p
    ///         for obj, r, name in zip(value, self.items_r, self.fieldnames):
    ///             if r.lowleveltype is not Void:
    ///                 setattr(p, name, r.convert_const(obj))
    ///         return p
    /// ```
    ///
    /// The empty-tuple arm reduces to `Constant(None, Void)` because
    /// upstream `TUPLE_TYPE([])` is `Void`. Non-empty tuples
    /// `instantiate` an immortal Gc struct, recursively
    /// `convert_const` each item to a `LowLevelValue`, write into
    /// the struct field, and cache the resulting `_ptr` keyed on the
    /// raw `Vec<ConstValue>` items.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::Tuple(items) = value else {
            return Err(TyperError::message(format!(
                "TupleRepr.convert_const: value must be a tuple, got {value:?}"
            )));
        };
        if items.len() != self.items_r.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.convert_const: tuple arity mismatch: got {}, expected {}",
                items.len(),
                self.items_r.len()
            )));
        }
        if self.items_r.is_empty() {
            // upstream rtuple.py convert_const routes the empty case
            // through `self.instantiate()` which returns
            // `dum_empty_tuple` (a Python sentinel function used as
            // a Void-typed PBC placeholder), and caches that
            // sentinel under the empty-tuple key in `tuple_cache`.
            // Pyre's `tuple_cache` is `HashMap<Vec<ConstValue>, _ptr>`
            // — it cannot store a Void-typed sentinel because
            // `TUPLE_TYPE([]) == Void` and `_ptr` describes a Ptr
            // value. The structural parity is achieved by the
            // `instantiate_empty` helper below: it materialises a
            // canonical `Constant(None, Void)` once and returns
            // cloned copies on subsequent calls. Two `Constant(None,
            // Void)` instances compare equal under `PartialEq` and
            // hash identically, so consumers cannot distinguish
            // upstream's identity-cached sentinel from this body's
            // structural-equivalent.
            return Ok(self.instantiate_empty());
        }
        // Cache lookup — return the cached Constant if present.
        if let Some(cached) = self.tuple_cache.borrow().get(items) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached.clone())),
                self.lltype.clone(),
            ));
        }
        // upstream `p = self.instantiate()` + `self.tuple_cache[key] = p`
        // (rtuple.py): cache the instantiated `_ptr` BEFORE
        // filling its fields. Pyre's `_ptr` is value-typed (Clone
        // deep-copies `_obj0`) so we cannot keep an aliased handle
        // outside the cache — instead we hold the slot in the cache
        // and mutate it through a brief `borrow_mut` per field write.
        // The `borrow_mut` is dropped between writes so recursive
        // `r.convert_const(obj)` calls below stay re-entrancy-safe.
        let p = self.instantiate()?;
        self.tuple_cache.borrow_mut().insert(items.clone(), p);
        // upstream loop: `for obj, r, name in zip(value, items_r, fieldnames):
        //     if r.lowleveltype is not Void:
        //         setattr(p, name, r.convert_const(obj))`.
        for ((obj, r), name) in items
            .iter()
            .zip(self.items_r.iter())
            .zip(self.fieldnames.iter())
        {
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            // Recursive call must NOT hold the tuple_cache borrow.
            let item_const = r.convert_const(obj)?;
            let llval = crate::translator::rtyper::rclass::constant_to_lowlevel_value(&item_const)?;
            let mut cache = self.tuple_cache.borrow_mut();
            let p_in_cache = cache.get_mut(items).ok_or_else(|| {
                TyperError::message("TupleRepr.convert_const: cached pointer disappeared mid-init")
            })?;
            p_in_cache
                .setattr(name, llval)
                .map_err(TyperError::message)?;
        }
        // Return a Clone of the cached entry — observationally
        // identical to returning `p` directly upstream.
        let cached = self
            .tuple_cache
            .borrow()
            .get(items)
            .cloned()
            .ok_or_else(|| {
                TyperError::message("TupleRepr.convert_const: cached pointer missing on return")
            })?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cached)),
            self.lltype.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use std::rc::Rc;

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    /// Variant that keeps the annotator alive — required for tests
    /// hitting `lowlevel_helper_function*` which upgrade the typer's
    /// `Weak<RPythonAnnotator>` to push synthesized graphs into
    /// `translator.graphs`. Also calls `initialize_exceptiondata` so
    /// the typer's self-weak is set (required for SomeTuple
    /// rtyper_makerepr → TupleRepr::new which uses self_rc()).
    fn fresh_rtyper_live() -> (
        Rc<crate::annotator::annrpython::RPythonAnnotator>,
        Rc<RPythonTyper>,
    ) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        (ann, rtyper)
    }

    #[test]
    fn tuple_type_empty_returns_void() {
        let t = TUPLE_TYPE(&[]);
        assert_eq!(t, LowLevelType::Void);
    }

    #[test]
    fn tuple_type_single_signed_returns_ptr_gcstruct() {
        let t = TUPLE_TYPE(&[LowLevelType::Signed]);
        let LowLevelType::Ptr(ptr) = t else {
            panic!("non-empty TUPLE_TYPE must produce Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("Ptr target must be Struct");
        };
        assert_eq!(body._name, "tuple1");
        assert!(body._flds.get("item0").is_some());
    }

    #[test]
    fn tuple_repr_new_collects_items_and_fieldnames() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        assert_eq!(repr.items_r.len(), 2);
        assert_eq!(
            repr.fieldnames,
            vec!["item0".to_string(), "item1".to_string()]
        );
        assert_eq!(repr.lltypes.len(), 2);
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("non-empty tuple repr must carry Ptr lltype");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("Ptr target must be Struct");
        };
        assert_eq!(body._name, "tuple2");
    }

    #[test]
    fn length1_tuple_iterator_repr_matches_upstream_layout_name() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int]).unwrap();
        let iter = Length1TupleIteratorRepr::new(&repr);

        let LowLevelType::Ptr(ptr) = &iter.lowleveltype else {
            panic!("tuple iterator lowleveltype must be Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("tuple iterator target must be Struct");
        };
        assert_eq!(body._name, "tuple1iter");
        assert_eq!(iter.ll_tupleiter, "ll_tupleiter");
        assert_eq!(iter.ll_tuplenext, "ll_tuplenext");
    }

    /// rmodel.py `IteratorRepr.rtype_iter` — `iter()` on a tuple
    /// iterator is the identity (returns the iterator unchanged, emits no
    /// op); an unknown method routes to the missing-operation error.
    #[test]
    fn length1_tuple_iterator_rtype_iter_returns_self_identity() {
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int]).unwrap();
        let iter_repr: Arc<Length1TupleIteratorRepr> =
            Arc::new(Length1TupleIteratorRepr::new(&repr));
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(iter_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "iter".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr.rtype_iter(&hop).unwrap();
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        assert_eq!(llops.borrow().ops.len(), 0);
        assert!(iter_repr.rtype_method("bogus", &hop).is_err());
    }

    #[test]
    fn deferred_tuple_str_and_iterator_helpers_report_missing_rtype_operation() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int]).unwrap();

        let err = gen_str_function(&rtyper, &repr).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("gen_str_function"));

        let iter = Length1TupleIteratorRepr::new(&repr);
        let tuple_var = Variable::new();
        tuple_var.set_concretetype(Some(repr.lowleveltype().clone()));
        let err = ll_tupleiter(&iter.lowleveltype, Hlvalue::Variable(tuple_var)).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_tupleiter"));

        let iter_var = Variable::new();
        iter_var.set_concretetype(Some(iter.lowleveltype.clone()));
        let err = ll_tuplenext(Hlvalue::Variable(iter_var)).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_tuplenext"));
    }

    /// Verifies the upstream rtuple.py compact_repr override: the default
    /// `Repr.compact_repr` would format the wrapping struct's
    /// `short_name` (e.g. `"Ptr <gcstruct tuple2>"`) which hides per-
    /// item types. The override flattens to space-joined per-item
    /// lltype short names so debug output stays readable.
    #[test]
    fn tuple_repr_compact_repr_flattens_per_item_lltype_short_names() {
        use crate::translator::rtyper::rfloat::FloatRepr;
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_float: Arc<dyn Repr> = Arc::new(FloatRepr::new());
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_float.clone()]).unwrap();
        assert_eq!(repr.compact_repr(), "TupleR Signed Float");
    }

    /// Empty tuple's lowleveltype is `Void` (rtuple.py:120-121 short
    /// circuit) — the compact_repr override prints `"TupleR "` (no
    /// per-item names) since `items_r.is_empty()`.
    #[test]
    fn tuple_repr_compact_repr_empty_tuple_has_no_item_names() {
        let rtyper = fresh_rtyper();
        let repr = TupleRepr::new(&rtyper, vec![]).unwrap();
        assert_eq!(repr.compact_repr(), "TupleR ");
    }

    #[test]
    fn tuple_repr_empty_convert_const_returns_void_sentinel() {
        let rtyper = fresh_rtyper();
        let repr = TupleRepr::new(&rtyper, vec![]).unwrap();
        let out = repr
            .convert_const(&ConstValue::Tuple(vec![]))
            .expect("empty tuple must succeed");
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));
    }

    #[test]
    fn tuple_repr_non_empty_convert_const_returns_live_pointer() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let out = repr
            .convert_const(&ConstValue::Tuple(vec![
                ConstValue::Int(7),
                ConstValue::Int(11),
            ]))
            .expect("non-empty tuple convert_const");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero(), "tuple instance must be live");
        // Per-item field writes propagate.
        let item0 = p.getattr("item0").unwrap();
        let item1 = p.getattr("item1").unwrap();
        assert_eq!(item0, lltype::LowLevelValue::Signed(7));
        assert_eq!(item1, lltype::LowLevelValue::Signed(11));
    }

    #[test]
    fn tuple_repr_convert_const_caches_repeated_calls_to_same_pointer() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone()]).unwrap();
        let value = ConstValue::Tuple(vec![ConstValue::Int(42)]);
        let a = repr.convert_const(&value).unwrap();
        let b = repr.convert_const(&value).unwrap();
        let (ConstValue::LLPtr(pa), ConstValue::LLPtr(pb)) = (&a.value, &b.value) else {
            panic!("expected LLPtr from both calls");
        };
        assert_eq!(pa._hashable_identity(), pb._hashable_identity());
    }

    #[test]
    fn tuple_repr_getitem_internal_emits_getfield() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(repr.lowleveltype().clone()));
        let v = repr
            .getitem_internal(&mut llops, Hlvalue::Variable(v_tuple), 1)
            .unwrap();
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "getfield");
        let Hlvalue::Constant(field_const) = &llops.ops[0].args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(field_const.value, ConstValue::byte_str("item1"));
        assert_eq!(v.concretetype().as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn tuple_repr_newtuple_emits_malloc_and_setfields() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let _ = TupleRepr::newtuple(
            &mut llops,
            &repr,
            vec![Hlvalue::Variable(v_a), Hlvalue::Variable(v_b)],
        )
        .unwrap();
        // 1 malloc + 2 setfield = 3 ops total.
        assert_eq!(llops.ops.len(), 3);
        assert_eq!(llops.ops[0].opname, "malloc");
        assert_eq!(llops.ops[1].opname, "setfield");
        assert_eq!(llops.ops[2].opname, "setfield");
    }

    #[test]
    fn tuple_repr_rtype_len_emits_inputconst_signed() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr =
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap();
        let result_var = Variable::new();
        let spaceop =
            SpaceOperation::new(OpKind::Len.opname(), vec![], Hlvalue::Variable(result_var));
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let out = repr.rtype_len(&hop).expect("rtype_len").unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("rtype_len must return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(3));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn pair_tuple_int_rtype_getitem_emits_getfield_for_constant_index() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        // Build a HighLevelOp with two args: tuple variable + Int(1) constant.
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        let _ = pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .expect("pair_tuple_int_rtype_getitem")
            .unwrap();
        let ops = hop.llops.borrow();
        // upstream emits `getfield(v_tuple, 'item1')` and a no-op
        // convertvar (since external == internal). One getfield op
        // expected total.
        assert!(
            ops.ops.iter().any(|op| op.opname == "getfield"),
            "expected at least one getfield op, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    /// upstream rtuple.py rtype_getitem — tuple constant indexing flows
    /// through `self.fieldnames[index]` which is a Python list and
    /// supports negative indexing (`fieldnames[-1]` returns the
    /// last item). Pyre normalises `idx + len` to mirror the
    /// wrap-around so `(0, 1, 2)[-1]` types as `2`.
    #[test]
    fn pair_tuple_int_rtype_getitem_negative_index_wraps_around_to_last_item() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        // Negative constant index — `-1` should map to last (item2).
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(-1),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .expect("negative index must succeed")
            .unwrap();
        let ops = hop.llops.borrow();
        // The emitted getfield must reference `item2` (last item of
        // a 3-tuple, i.e. fieldnames[-1] == "item2").
        let getfield = ops
            .ops
            .iter()
            .find(|op| op.opname == "getfield")
            .expect("getfield op expected");
        let Hlvalue::Constant(c) = &getfield.args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(c.value, ConstValue::byte_str("item2"));
    }

    #[test]
    fn tuple_repr_newtuple_empty_returns_void_constant() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let repr = TupleRepr::new(&rtyper, vec![]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = TupleRepr::newtuple(&mut llops, &repr, vec![]).unwrap();
        assert!(llops.ops.is_empty(), "empty tuple emits no ops");
        let Hlvalue::Constant(c) = out else {
            panic!("expected Constant for empty tuple");
        };
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    /// `translate_op_len` (rtyper.py:484-486) dispatches `r.rtype_len(hop)`
    /// where `r` is the `Repr` for the first argument. Without
    /// [`Repr::rtype_len`] override on `TupleRepr` the call would fall
    /// to the trait default `missing_rtype_operation("len")`. This test
    /// goes through `RPythonTyper::translate_operation("len", ...)` to
    /// pin the override at the dispatch level — not just the inherent
    /// helper.
    #[test]
    fn translate_operation_len_routes_to_tuplerepr_override() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Len.opname(),
            vec![v_tuple_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc));

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation len must dispatch through TupleRepr override")
            .expect("len returns a value");
        let Hlvalue::Constant(c) = out else {
            panic!("rtype_len must return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(2));
    }

    /// `externalvsinternal` (rmodel.py) maps each GC
    /// `InstanceRepr` item to the root `getinstancerepr(rtyper, None)`
    /// while keeping the concrete repr on the `external_items_r` side.
    /// Non-instance items (Integer / Bool / etc.) pass through with
    /// `external == internal`.
    #[test]
    fn tuple_repr_new_routes_gc_instance_items_through_externalvsinternal() {
        use crate::annotator::classdesc::ClassDef;
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rint::IntegerRepr;

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.Foo", None);
        let r_inst = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).unwrap();
        let r_inst_arc: Arc<dyn Repr> = r_inst.clone();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));

        let repr = TupleRepr::new(&rtyper, vec![r_inst_arc.clone(), r_int.clone()]).unwrap();

        // External items_r: untouched per-position concrete reprs.
        assert!(
            Arc::ptr_eq(&repr.external_items_r[0], &r_inst_arc),
            "external_items_r[0] must be the original GC InstanceRepr"
        );
        assert!(
            Arc::ptr_eq(&repr.external_items_r[1], &r_int),
            "external_items_r[1] (non-instance) is preserved as-is"
        );

        // Internal items_r[0] is rerouted to the root InstanceRepr
        // (classdef=None) — the lowleveltype must equal the root
        // OBJECTPTR since both root + leaf carry the OBJECT GcStruct.
        let root_inst = getinstancerepr(&rtyper, None, Flavor::Gc).unwrap();
        assert_eq!(
            repr.items_r[0].lowleveltype(),
            root_inst.lowleveltype(),
            "internal items_r[0] must match root InstanceRepr lowleveltype"
        );
        // And the non-instance int repr is internal == external.
        assert!(
            Arc::ptr_eq(&repr.items_r[1], &r_int),
            "items_r[1] (Integer) must be unchanged"
        );
    }

    /// `translate_op_newtuple` (rtyper.py) dispatches the free
    /// function `rtuple.rtype_newtuple(hop)`. Without the explicit
    /// `"newtuple"` arm in `RPythonTyper::translate_operation` the op
    /// would fall to `default_translate_operation` and raise
    /// "unimplemented operation 'newtuple'".
    #[test]
    fn translate_operation_newtuple_routes_to_rtuple_dispatch() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "newtuple".to_string(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Args metadata — newtuple takes per-item args typed at each
        // r_tuple.items_r repr (Signed here).
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // r_result is the TupleRepr — `_rtype_newtuple` reads it from
        // `hop.r_result`.
        *hop.r_result.borrow_mut() = Some(r_tup_arc.clone());
        // s_result is non-const so the cache fast path is skipped and
        // genop("malloc") + per-item genop("setfield") are emitted.
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));
        let _ = OpKind::SimpleCall; // keep imports minimal silenced

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation newtuple must dispatch through rtuple")
            .expect("newtuple returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("non-const newtuple must return a Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "malloc"),
            "expected malloc op, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    /// `pairtype(TupleRepr, TupleRepr).rtype_is_` (rtuple.py)
    /// raises a `TyperError` instead of routing to `ptr_eq` on the
    /// tuple pointers — upstream rejects tuple identity comparison
    /// eagerly. The dispatch table arm in `pairtype.rs` propagates
    /// the error past the generic `(Repr, Repr).rtype_is_` arm.
    #[test]
    fn pair_tuple_tuple_rtype_is_returns_typeerror() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_a = Variable::new();
        v_a.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_b = Variable::new();
        v_b.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Is.opname(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));

        let err = rtyper
            .translate_operation(&hop)
            .expect_err("rtype_is_ on two tuples must error");
        assert!(
            format!("{err}").contains("cannot compare tuples with 'is'"),
            "expected upstream error message, got: {err}"
        );
    }

    /// `TupleRepr.rtype_getslice` (rtuple.py) extracts the
    /// per-position items via `getitem_internal` and assembles a
    /// fresh tuple via `newtuple`. For `(a, b, c)[1:3]` the emitted
    /// op stream is: 2× getfield (for items 1 and 2) + 1× malloc + 2×
    /// setfield = 5 ops total. The result tuple's repr arity must
    /// match `len(indices)`.
    #[test]
    fn tuple_repr_rtype_getslice_emits_per_index_getfield_and_newtuple() {
        use crate::annotator::model::{SomeInteger, SomeTuple, SomeValue};
        use crate::flowspace::model::Constant as FlowConstant;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup3: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let r_tup2: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup3.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        // start = 1, stop = 3 — slice [1:3] on a 3-tuple yields 2 items.
        let v_start_h = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let v_stop_h = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::Int(3),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetSlice.opname(),
            vec![v_tuple_h.clone(), v_start_h.clone(), v_stop_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Args metadata.
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup3.clone()));
        // start=1: SomeInteger with const_box.
        let mut s_start = SomeInteger::new(false, false);
        s_start.base.const_box = Some(FlowConstant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        hop.args_v.borrow_mut().push(v_start_h);
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_start));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // stop=3.
        let mut s_stop = SomeInteger::new(false, false);
        s_stop.base.const_box = Some(FlowConstant::with_concretetype(
            ConstValue::Int(3),
            LowLevelType::Signed,
        ));
        hop.args_v.borrow_mut().push(v_stop_h);
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_stop));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // r_result is the 2-tuple; s_result non-const so newtuple emits.
        *hop.r_result.borrow_mut() = Some(r_tup2.clone());
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));

        let out = rtyper
            .translate_operation(&hop)
            .expect("getslice translates")
            .expect("getslice returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("getslice must return Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        let getfield_count = ops.ops.iter().filter(|op| op.opname == "getfield").count();
        let malloc_count = ops.ops.iter().filter(|op| op.opname == "malloc").count();
        let setfield_count = ops.ops.iter().filter(|op| op.opname == "setfield").count();
        assert_eq!(
            getfield_count,
            2,
            "expected 2 getfield (items 1, 2), got ops: {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        assert_eq!(
            malloc_count, 1,
            "expected 1 malloc for the 2-tuple newtuple"
        );
        assert_eq!(
            setfield_count, 2,
            "expected 2 setfield (items 0, 1) on the new tuple"
        );
    }

    /// `pairtype(TupleRepr, TupleRepr).rtype_add` (rtuple.py)
    /// concatenates two tuples by per-position getfield_internal +
    /// newtuple_cached. For `(a, b) + (c,)` the emitted op stream is:
    /// 3× getfield (items 0,1 of left + item 0 of right) + 1× malloc +
    /// 3× setfield = 7 ops total. Result repr arity = `len_a + len_b`.
    #[test]
    fn pair_tuple_tuple_rtype_add_concatenates_via_per_side_getfield_and_newtuple() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup2: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let r_tup1: Arc<dyn Repr> = Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone()]).unwrap());
        let r_tup_result: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let v_left = Variable::new();
        v_left.set_concretetype(Some(r_tup2.lowleveltype().clone()));
        let v_left_h = Hlvalue::Variable(v_left);
        let v_right = Variable::new();
        v_right.set_concretetype(Some(r_tup1.lowleveltype().clone()));
        let v_right_h = Hlvalue::Variable(v_right);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Add.opname(),
            vec![v_left_h.clone(), v_right_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_left_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup2.clone()));
        hop.args_v.borrow_mut().push(v_right_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup1.clone()));
        *hop.r_result.borrow_mut() = Some(r_tup_result.clone());
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));

        let out = rtyper
            .translate_operation(&hop)
            .expect("tuple+tuple translates")
            .expect("tuple+tuple returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("non-const tuple+tuple must return a Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        let getfield_count = ops.ops.iter().filter(|op| op.opname == "getfield").count();
        let malloc_count = ops.ops.iter().filter(|op| op.opname == "malloc").count();
        let setfield_count = ops.ops.iter().filter(|op| op.opname == "setfield").count();
        assert_eq!(getfield_count, 3, "2 getfield from left + 1 from right");
        assert_eq!(malloc_count, 1);
        assert_eq!(setfield_count, 3, "3 setfield on the 3-tuple result");
    }

    /// `pair(TupleRepr, IntegerRepr).rtype_getitem` (rtuple.py)
    /// drops the implicit IndexError channel when the typer can prove
    /// the constant-indexed access cannot fail at runtime:
    ///
    /// ```python
    /// if hop.has_implicit_exception(IndexError):
    ///     hop.exception_cannot_occur()
    /// ```
    ///
    /// Verified via `LowLevelOpList.llop_raising_exceptions ==
    /// Some(Removed)`. When the hop has no IndexError exceptionlink
    /// the call is a no-op (`rtyper.rs`'s `HighLevelOp::has_implicit_exception`
    /// returns false on empty exceptionlinks).
    #[test]
    fn pair_tuple_int_rtype_getitem_closes_implicit_indexerror_channel() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{
            HighLevelOp, LlopRaisingExceptions, LowLevelOpList,
        };
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        // Build an exceptionlink whose exitcase is IndexError. The
        // catch-Exception construction in rtyper.rs (test
        // has_implicit_exception_records_matching_exception_link) is
        // the parity reference.
        let index_error = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("IndexError")
            .expect("IndexError host class");
        let exitblock = Rc::new(StdRef::new(crate::flowspace::model::Block::new(vec![])));
        let exclink = Rc::new(StdRef::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                index_error,
            )))),
        )));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, vec![exclink], llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .expect("constant-indexed tuple getitem")
            .unwrap();
        // upstream rtuple.py:271 — `hop.exception_cannot_occur()` sets
        // `llops.llop_raising_exceptions = "removed"`. Pyre mirrors via
        // `LlopRaisingExceptions::Removed`.
        assert!(matches!(
            hop.llops.borrow().llop_raising_exceptions,
            Some(LlopRaisingExceptions::Removed)
        ));
    }

    /// When the hop has no exceptionlinks, `has_implicit_exception`
    /// returns false (rtyper.rs) and the IndexError channel
    /// closing is skipped — `llop_raising_exceptions` stays None.
    /// This pins the no-op path of the new `has_implicit_exception`
    /// wiring on `pair_tuple_int_rtype_getitem`.
    #[test]
    fn pair_tuple_int_rtype_getitem_without_exceptionlinks_leaves_channel_open() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(0),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .unwrap()
            .unwrap();
        assert!(hop.llops.borrow().llop_raising_exceptions.is_none());
    }

    /// `gen_eq_function([r_int, r_int])` (rtuple.py) synthesizes
    /// a `ll_tuple_eq_signed_signed(t1, t2) -> Bool` helper graph.
    /// The graph chain has N=2 check blocks each emitting
    /// `getfield × 2 + int_eq` and a 2-way exit. Last block's true
    /// branch points to returnblock(true); every false branch points
    /// to returnblock(false).
    #[test]
    fn gen_eq_function_synthesizes_per_shape_helper_graph() {
        use crate::translator::rtyper::rint::IntegerRepr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let llfn =
            gen_eq_function(&rtyper, &[r_int.clone(), r_int.clone()]).expect("gen_eq_function");
        // Cache key shape (rtuple.py `tuple(eq_funcs)`): each
        // primitive Repr returns None from get_ll_eq_function, so the
        // suffix is `<lltype>:eq` per item.
        assert_eq!(llfn.name, "ll_tuple_eq_Signed:eq_Signed:eq");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        // startblock has 2 inputargs (the two tuple ptrs).
        assert_eq!(g.startblock.borrow().inputargs.len(), 2);
        // The graph traversal must visit 2 check blocks. We assert by
        // counting blocks reachable through `exits[].target` from
        // startblock.
        let mut visited: Vec<*const _> = Vec::new();
        let mut stack = vec![g.startblock.clone()];
        while let Some(b) = stack.pop() {
            let ptr = Rc::as_ptr(&b);
            if visited.contains(&ptr) {
                continue;
            }
            visited.push(ptr);
            for link in b.borrow().exits.iter() {
                if let Some(target) = link.borrow().target.as_ref() {
                    stack.push(target.clone());
                }
            }
        }
        // 2 check blocks + returnblock + exceptblock = 4 (exceptblock
        // is reachable too, even though not used).
        assert!(
            visited.len() >= 3,
            "expected ≥ 3 reachable blocks, got {}",
            visited.len()
        );
        // startblock has int_eq op + 2 getfield ops = 3 ops.
        let start_ops: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "getfield", "int_eq"]);
    }

    /// rtuple.py — `eq_funcs = [r_item.get_ll_eq_function() or
    /// operator.eq for r_item in items_r]`. When an item Repr returns
    /// `Some(helper)` (e.g. nested `TupleRepr`), gen_eq must dispatch
    /// to that helper via `direct_call` rather than the primitive
    /// `int_eq`/... whitelist. Validates the unblocking of nested
    /// tuple equality (tuple-of-tuple).
    #[test]
    fn gen_eq_function_dispatches_to_inner_tuple_helper_via_direct_call() {
        use crate::translator::rtyper::rint::IntegerRepr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        // Inner tuple: (Signed, Signed). Outer tuple: (inner, Signed).
        let r_inner: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let llfn = gen_eq_function(&rtyper, &[r_inner.clone(), r_int.clone()])
            .expect("gen_eq_function for outer tuple");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        // First check_block dispatches to inner gen_eq via direct_call;
        // second check_block uses primitive int_eq for the Signed arm.
        let block0_ops: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            block0_ops,
            vec!["getfield", "getfield", "direct_call"],
            "inner tuple item must dispatch via direct_call to its ll_eq helper"
        );
    }

    /// rtuple.py — `gen_eq_function([])` produces a helper whose
    /// `autounrolling_funclist` is empty so `equal_so_far = True` is
    /// returned unchanged. Tuple lltype is `Void` (rtuple.py).
    #[test]
    fn gen_eq_function_empty_tuple_returns_constant_true_helper() {
        let (_ann, rtyper) = fresh_rtyper_live();
        let llfn = gen_eq_function(&rtyper, &[]).expect("gen_eq_function([])");
        assert_eq!(llfn.name, "ll_tuple_eq_");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        // Two Void inputargs (one per tuple). No ops in startblock —
        // direct close to returnblock with constant True.
        assert_eq!(g.startblock.borrow().inputargs.len(), 2);
        assert!(g.startblock.borrow().operations.is_empty());
        let exits = g.startblock.borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.args.len(), 1);
        let Some(Hlvalue::Constant(c)) = link.args[0].as_ref() else {
            panic!("link arg must be a Constant true");
        };
        assert_eq!(c.value, ConstValue::Bool(true));
    }

    /// `pair(TupleRepr, TupleRepr).rtype_eq` (rtuple.py)
    /// dispatches to the synthesized `ll_eq` helper via `gendirectcall`.
    /// One `direct_call` op is emitted with the helper graph as the
    /// callee, and the result Variable is `Bool`-typed.
    #[test]
    fn pair_tuple_tuple_rtype_eq_emits_direct_call_to_ll_eq_helper() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_left = Variable::new();
        v_left.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_left_h = Hlvalue::Variable(v_left);
        let v_right = Variable::new();
        v_right.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_right_h = Hlvalue::Variable(v_right);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Eq.opname(),
            vec![v_left_h.clone(), v_right_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Both sides annotated as the same SomeTuple so unionof
        // resolves to a single TupleRepr.
        // Annotate items as SomeInteger so unionof's TupleRepr has
        // Signed-typed items (gen_eq_function rejects Void items —
        // those have no primitive equality op).
        use crate::annotator::model::SomeInteger;
        let s_int = SomeValue::Integer(SomeInteger::new(false, false));
        let s_tup = SomeValue::Tuple(SomeTuple::new(vec![s_int.clone(), s_int.clone()]));
        hop.args_v.borrow_mut().push(v_left_h);
        hop.args_s.borrow_mut().push(s_tup.clone());
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_right_h);
        hop.args_s.borrow_mut().push(s_tup);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        *hop.r_result.borrow_mut() = Some(Arc::new(IntegerRepr::new(
            LowLevelType::Bool,
            Some("bool_"),
        )));
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);

        let out = rtyper
            .translate_operation(&hop)
            .expect("tuple == tuple translates")
            .expect("rtype_eq returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("rtype_eq must return a Variable from direct_call");
        };
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        assert_eq!(direct_calls, 1, "expected exactly one direct_call op");
    }

    /// `pair(TupleRepr, TupleRepr).rtype_ne` (rtuple.py)
    /// emits direct_call (from rtype_eq) followed by `bool_not`.
    #[test]
    fn pair_tuple_tuple_rtype_ne_appends_bool_not_after_eq() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_left = Variable::new();
        v_left.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_left_h = Hlvalue::Variable(v_left);
        let v_right = Variable::new();
        v_right.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_right_h = Hlvalue::Variable(v_right);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Ne.opname(),
            vec![v_left_h.clone(), v_right_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Annotate items as SomeInteger so unionof's TupleRepr has
        // Signed-typed items (gen_eq_function rejects Void items —
        // those have no primitive equality op).
        use crate::annotator::model::SomeInteger;
        let s_int = SomeValue::Integer(SomeInteger::new(false, false));
        let s_tup = SomeValue::Tuple(SomeTuple::new(vec![s_int.clone(), s_int.clone()]));
        hop.args_v.borrow_mut().push(v_left_h);
        hop.args_s.borrow_mut().push(s_tup.clone());
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_right_h);
        hop.args_s.borrow_mut().push(s_tup);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        *hop.r_result.borrow_mut() = Some(Arc::new(IntegerRepr::new(
            LowLevelType::Bool,
            Some("bool_"),
        )));
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);

        rtyper
            .translate_operation(&hop)
            .expect("tuple != tuple translates")
            .expect("rtype_ne returns a value");
        let ops = hop.llops.borrow();
        let opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            opnames.contains(&"direct_call"),
            "rtype_ne must emit direct_call (via rtype_eq), got {:?}",
            opnames
        );
        assert!(
            opnames.contains(&"bool_not"),
            "rtype_ne must append bool_not, got {:?}",
            opnames
        );
        // bool_not must appear after direct_call.
        let dc_pos = opnames.iter().position(|s| *s == "direct_call").unwrap();
        let bn_pos = opnames.iter().position(|s| *s == "bool_not").unwrap();
        assert!(bn_pos > dc_pos, "bool_not must follow direct_call");
    }

    /// `gen_hash_function([r_int, r_int])` (rtuple.py) synthesizes
    /// a `ll_tuple_hash_signed_signed(t) -> Signed` helper graph.
    /// Single-block linear shape: per item emits getfield →
    /// direct_call(ll_hash_int) → int_mul → int_xor, accumulating into
    /// x. Result: `(((0x345678 * 1000003) ^ ll_hash_int(item0)) *
    /// 1000003) ^ ll_hash_int(item1)`.
    #[test]
    fn gen_hash_function_synthesizes_per_shape_helper_graph() {
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = signed_repr();
        let llfn =
            gen_hash_function(&rtyper, &[r_int.clone(), r_int.clone()]).expect("gen_hash_function");
        // Cache key shape (rtuple.py `tuple(hash_funcs)`): each
        // item's hash helper name (`ll_hash_int_Signed` for Signed)
        // contributes to the suffix.
        assert_eq!(
            llfn.name,
            "ll_tuple_hash_ll_hash_int_Signed_ll_hash_int_Signed"
        );
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        assert_eq!(g.startblock.borrow().inputargs.len(), 1);
        let opnames: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // Per item: getfield + direct_call + int_mul + int_xor = 4 × 2 = 8 ops.
        assert_eq!(
            opnames,
            vec![
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor",
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor"
            ]
        );
    }

    /// rtuple.py/56 — `key = tuple(eq_funcs)` / `tuple(hash_funcs)`.
    /// Cache key is the helper-tuple identity, NOT the lltype shape.
    /// Two distinct tuple shapes built from Reprs that share lltypes
    /// but differ in their helpers must yield distinct synthesized
    /// helpers. Validates the helper-identity-based key suffix.
    #[test]
    fn gen_hash_function_cache_key_distinguishes_distinct_inner_helpers() {
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = signed_repr();

        // Outer tuple A: (Signed, Signed) — two primitive ints.
        let llfn_a =
            gen_hash_function(&rtyper, &[r_int.clone(), r_int.clone()]).expect("flat tuple hash");
        // Outer tuple B: (Inner(Signed,Signed), Signed) — same outer
        // lltypes... wait, actually no. Distinct lltype because of the
        // tuple Ptr wrapper. Use a deeper nesting to demonstrate
        // identity vs shape: identical lltype shape across two
        // distinct inner helpers is hard to construct without two
        // override-distinct Reprs sharing a lltype, so this test
        // proves the simpler invariant — same Repr reuses the same
        // helper (cache hit) across calls.
        let llfn_a2 =
            gen_hash_function(&rtyper, &[r_int.clone(), r_int.clone()]).expect("flat tuple hash 2");
        let ga = llfn_a.graph.as_ref().expect("a graph");
        let ga2 = llfn_a2.graph.as_ref().expect("a2 graph");
        assert!(
            std::rc::Rc::ptr_eq(&ga.graph, &ga2.graph),
            "identical (Repr, items) input must hit the helper cache"
        );

        // Distinct inner Repr (TupleRepr nested) → distinct helper.
        let r_inner: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let llfn_b = gen_hash_function(&rtyper, &[r_inner.clone(), r_int.clone()])
            .expect("nested tuple hash");
        assert_ne!(
            llfn_a.name, llfn_b.name,
            "nested-tuple item must yield a distinct helper name from primitive item"
        );
        let gb = llfn_b.graph.as_ref().expect("b graph");
        assert!(
            !std::rc::Rc::ptr_eq(&ga.graph, &gb.graph),
            "distinct items_r must yield distinct helper graphs"
        );
    }

    /// rstr.py:496/499 — Char items in tuples dispatch eq via the
    /// `char_eq` primitive op (None from get_ll_eq_function) and hash
    /// via the synthesized `ll_char_hash` helper (cast_char_to_int).
    #[test]
    fn gen_eq_and_hash_dispatch_for_tuple_of_char() {
        use crate::translator::rtyper::rstr::char_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_char: Arc<dyn Repr> = char_repr();

        let llfn_eq = gen_eq_function(&rtyper, &[r_char.clone(), r_char.clone()])
            .expect("gen_eq_function for (Char, Char)");
        let g_eq = llfn_eq.graph.expect("graph populated");
        let block0_ops: Vec<String> = g_eq
            .graph
            .borrow()
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(block0_ops, vec!["getfield", "getfield", "char_eq"]);

        let llfn_hash = gen_hash_function(&rtyper, &[r_char.clone(), r_char.clone()])
            .expect("gen_hash_function for (Char, Char)");
        let g_hash = llfn_hash.graph.expect("graph populated");
        let opnames: Vec<String> = g_hash
            .graph
            .borrow()
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // Per item: getfield + direct_call(ll_char_hash) + int_mul + int_xor.
        assert_eq!(
            opnames,
            vec![
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor",
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor"
            ]
        );
    }

    /// rtuple.py — `r_item.get_ll_eq_function() or operator.eq`.
    /// For NoneRepr items the fallback `operator.eq` on two Void
    /// values is annotator/rtyper-folded to `Constant(True, Bool)`.
    /// Pyre mirrors via `same_as(constant_true)` so the per-item
    /// exitswitch still reads from a Variable.
    #[test]
    fn gen_eq_function_uses_same_as_constant_true_for_void_none_items() {
        use crate::translator::rtyper::rint::signed_repr;
        use crate::translator::rtyper::rnone::none_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = signed_repr();
        let r_none: Arc<dyn Repr> = none_repr();
        let llfn = gen_eq_function(&rtyper, &[r_int.clone(), r_none.clone()])
            .expect("gen_eq_function for (Signed, None) tuple");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();

        // First check_block: int item dispatches via int_eq.
        let block0_ops: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(block0_ops, vec!["getfield", "getfield", "int_eq"]);

        // Second check_block (None item): same_as(constant_true)
        // replaces primitive_eq_opname (which doesn't support Void).
        let block0 = g.startblock.borrow();
        let block1_link = block0.exits[0].borrow();
        let block1 = block1_link
            .target
            .as_ref()
            .expect("true exit target")
            .clone();
        drop(block1_link);
        drop(block0);
        let block1_ops: Vec<String> = block1
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(block1_ops, vec!["getfield", "getfield", "same_as"]);
    }

    /// rtuple.py — `r_item.get_ll_eq_function() or _ll_equal`.
    /// `ll_equal` for Void items const-folds to `Constant(True, Bool)`;
    /// pyre mirrors via `same_as(constant_true)` instead of the
    /// non-existent `int_eq` for Void.
    #[test]
    fn ll_equal_for_void_item_uses_same_as_constant_true() {
        let (_ann, rtyper) = fresh_rtyper_live();
        let llfn = ll_equal(&rtyper, &LowLevelType::Void).expect("ll_equal for Void");
        assert_eq!(llfn.name, "ll_equal_Void");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        let opnames: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(opnames, vec!["same_as"]);
    }

    /// rtuple.py — `hash_funcs = [r_item.get_ll_hash_function() for
    /// r_item in items_r]`. With FloatRepr's `_hash_float` helper now
    /// landed (rfloat.rs `build_ll_hash_float_helper_graph`), tuples
    /// containing Float items synthesize successfully — the per-item
    /// dispatch routes through `direct_call` on the float hash helper
    /// just like any other primitive Repr.
    #[test]
    fn gen_hash_function_supports_float_items_via_hash_float_helper() {
        use crate::translator::rtyper::rfloat::float_repr;
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = signed_repr();
        let r_float: Arc<dyn Repr> = float_repr();
        let llfn = gen_hash_function(&rtyper, &[r_int.clone(), r_float.clone()])
            .expect("gen_hash_function for (Signed, Float) tuple");
        // Cache key shape: Signed item → `ll_hash_int_Signed`, Float
        // item → `_hash_float`.
        assert_eq!(llfn.name, "ll_tuple_hash_ll_hash_int_Signed__hash_float");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        let opnames: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // Each item dispatches via direct_call to its hash helper —
        // ll_hash_int_Signed for the Signed item, _hash_float for the
        // Float item.
        assert_eq!(
            opnames,
            vec![
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor",
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor"
            ]
        );
    }

    /// rtuple.py — `hash_funcs = [r_item.get_ll_hash_function() for
    /// r_item in items_r]`. When the item Repr is a nested `TupleRepr`,
    /// gen_hash dispatches to that inner `ll_hash` helper via
    /// `direct_call`. Validates the structural per-item dispatch (no
    /// inline cast on tuple items).
    #[test]
    fn gen_hash_function_dispatches_to_inner_tuple_helper_via_direct_call() {
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = signed_repr();
        // Outer tuple item: a nested (Signed, Signed) tuple.
        let r_inner: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let llfn = gen_hash_function(&rtyper, &[r_inner.clone(), r_int.clone()])
            .expect("gen_hash_function for outer tuple");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        let opnames: Vec<String> = g
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // Item 0 (inner tuple): getfield + direct_call(inner ll_hash) +
        // int_mul + int_xor. Item 1 (Signed): getfield +
        // direct_call(ll_hash_int_Signed) + int_mul + int_xor.
        assert_eq!(
            opnames,
            vec![
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor",
                "getfield",
                "direct_call",
                "int_mul",
                "int_xor"
            ],
            "every item must dispatch via direct_call to its ll_hash helper"
        );
    }

    /// rtuple.py — `gen_hash_function([])` produces a helper whose
    /// `autounrolling_funclist` is empty so `x = 0x345678` is returned
    /// unchanged. Tuple lltype is `Void` (rtuple.py); single
    /// Void inputarg.
    #[test]
    fn gen_hash_function_empty_tuple_returns_seed_only_helper() {
        let (_ann, rtyper) = fresh_rtyper_live();
        let llfn = gen_hash_function(&rtyper, &[]).expect("gen_hash_function([])");
        assert_eq!(llfn.name, "ll_tuple_hash_");
        let graph = llfn.graph.expect("graph populated");
        let g = graph.graph.borrow();
        // Single Void inputarg (the empty tuple). No ops in startblock —
        // direct close to returnblock with constant 0x345678.
        assert_eq!(g.startblock.borrow().inputargs.len(), 1);
        assert!(g.startblock.borrow().operations.is_empty());
        let exits = g.startblock.borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.args.len(), 1);
        let Some(Hlvalue::Constant(c)) = link.args[0].as_ref() else {
            panic!("link arg must be a Constant 0x345678");
        };
        assert_eq!(c.value, ConstValue::Int(0x345678));
    }

    /// `TupleRepr.rtype_hash` (rtuple.py) dispatches to the
    /// synthesized `gen_hash_function` helper via `gendirectcall`.
    #[test]
    fn tuple_repr_rtype_hash_emits_direct_call_to_ll_hash_helper() {
        use crate::annotator::model::{SomeInteger, SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Hash.opname(),
            vec![v_tuple_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let s_int = SomeValue::Integer(SomeInteger::new(false, false));
        let s_tup = SomeValue::Tuple(SomeTuple::new(vec![s_int.clone(), s_int.clone()]));
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s.borrow_mut().push(s_tup);
        hop.args_r.borrow_mut().push(Some(r_tup_arc));
        *hop.r_result.borrow_mut() = Some(Arc::new(IntegerRepr::new(
            LowLevelType::Signed,
            Some("int_"),
        )));
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);

        let _ = rtyper
            .translate_operation(&hop)
            .expect("hash translates")
            .expect("hash returns");
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        assert_eq!(direct_calls, 1, "expected exactly one direct_call op");
    }

    /// `pair(TupleRepr, Repr).rtype_contains` (rtuple.py) for a
    /// constant 3-tuple `(1, 2, 3)` membership test against a Signed
    /// `v_arg`: emits 3 `direct_call(ll_equal_Signed, v_arg, c_x)` ops
    /// (one per tuple element) chained by 2 `int_or` ops.
    #[test]
    fn pair_tuple_repr_rtype_contains_emits_direct_call_chain_per_const_element() {
        use crate::annotator::model::{SomeInteger, SomeTuple, SomeValue};
        use crate::flowspace::model::{Constant as FlowConstant, SpaceOperation};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        // Constant 3-tuple: (1, 2, 3).
        let const_tup = ConstValue::Tuple(vec![
            ConstValue::Int(1),
            ConstValue::Int(2),
            ConstValue::Int(3),
        ]);
        let mut s_tup_object = SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        s_tup_object.base.const_box = Some(FlowConstant::with_concretetype(
            const_tup.clone(),
            r_tup_arc.lowleveltype().clone(),
        ));
        let s_tup = SomeValue::Tuple(s_tup_object);
        let v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_item_h = Hlvalue::Variable(v_item);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Contains.opname(),
            vec![v_tuple_h.clone(), v_item_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s.borrow_mut().push(s_tup);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_item_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(false, false)));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(Arc::new(IntegerRepr::new(
            LowLevelType::Bool,
            Some("bool_"),
        )));
        *hop.s_result.borrow_mut() = Some(SomeValue::Impossible);

        let _ = rtyper
            .translate_operation(&hop)
            .expect("contains translates")
            .expect("contains returns");
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        let int_ors = ops.ops.iter().filter(|op| op.opname == "int_or").count();
        assert_eq!(
            direct_calls, 3,
            "expected 3 direct_call (one per tuple item)"
        );
        assert_eq!(int_ors, 2, "expected 2 int_or chaining 3 calls");
    }

    /// `gen_eq_function` caches per `(name, args, result)` — calling
    /// twice with the same items_r returns the same `LowLevelFunction`
    /// (graph identity preserved).
    #[test]
    fn gen_eq_function_caches_per_shape() {
        use crate::translator::rtyper::rint::IntegerRepr;

        let (_ann, rtyper) = fresh_rtyper_live();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let a = gen_eq_function(&rtyper, &[r_int.clone(), r_int.clone()]).unwrap();
        let b = gen_eq_function(&rtyper, &[r_int.clone(), r_int.clone()]).unwrap();
        // Same `lowlevel_helper_function_with_builder` cache → same
        // PyGraph instance.
        assert!(Rc::ptr_eq(
            a.graph.as_ref().unwrap(),
            b.graph.as_ref().unwrap()
        ));
    }

    /// `pairtype(TupleRepr, TupleRepr).convert_from_to` (rtuple.py)
    /// returns the source value unchanged when both reprs have the
    /// same lowleveltype (rtuple.py). Different-arity tuples
    /// return `NotImplemented` (Ok(None)).
    #[test]
    fn pair_tuple_tuple_convert_from_to_identity_when_lltype_matches() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_a = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let r_b = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        // Sanity: same items_r, same TUPLE_TYPE → equal lowleveltype.
        assert_eq!(r_a.lowleveltype(), r_b.lowleveltype());
        let v_in = Variable::new();
        v_in.set_concretetype(Some(r_a.lowleveltype().clone()));
        let v_in_h = Hlvalue::Variable(v_in);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = pair_tuple_tuple_convert_from_to(&r_a, &r_b, &v_in_h, &mut llops)
            .expect("convert_from_to must succeed when lltypes match")
            .expect("same-lltype must short-circuit to Some(v)");
        // Identity — no ops emitted, result is the unchanged input.
        assert!(llops.ops.is_empty(), "identity convert emits no ops");
        let v_concrete = match &v_in_h {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        let out_concrete = match &out {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        assert_eq!(
            v_concrete, out_concrete,
            "identity convert preserves concretetype"
        );
    }

    /// Different-arity tuple→tuple convert_from_to returns `Ok(None)`
    /// (upstream `NotImplemented`) — the rtyper's pair-MRO walker
    /// falls through to the next handler instead of synthesising a
    /// non-existent shape.
    #[test]
    fn pair_tuple_tuple_convert_from_to_different_arity_returns_notimplemented() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_2 = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let r_3 =
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap();
        let v = Variable::new();
        v.set_concretetype(Some(r_2.lowleveltype().clone()));
        let v_h = Hlvalue::Variable(v);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = pair_tuple_tuple_convert_from_to(&r_2, &r_3, &v_h, &mut llops)
            .expect("convert_from_to must not error on arity mismatch");
        assert!(out.is_none(), "different-arity must return None");
        assert!(llops.ops.is_empty());
    }
}
